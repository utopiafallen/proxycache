// matcher.go — global request matcher + per-backend processing queues.
//
// Pipeline:
//
//	HTTP handler (ChatHandler)         -> Submit()            [arrival]
//	RequestDispatcher.run (1 goroutine) scans each incoming request
//	    (per-backend tokenize + disk/pending-slot cache scan), picks a target
//	    backend, and enqueues it on that backend's queue. A request whose
//	    chosen backend has a full queue is parked in the global overflow queue
//	    (FIFO) and re-matched whenever capacity frees up.
//	backendWorker.run (1 goroutine per backend) processes queued requests
//	    sequentially: acquire a slot on this backend, restore the cache if it
//	    is available here (including files that arrived via P2P transfer),
//	    dispatch to llama.cpp, save the cache, release the slot.
//
// Routing rule: a request with a cache hit goes to the hit backend while its
// queue depth stays below CacheHitQueueLimit; otherwise it is distributed
// round-robin across the other matching backends (starting after the hit
// backend), which triggers an async P2P cache transfer to the chosen backend.

package proxycache

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// --- Request plumbing ---

// ProcRequest carries one chat request through the pipeline.
type ProcRequest struct {
	w             http.ResponseWriter
	r             *http.Request
	ctx           context.Context // derived from r.Context(); cancelled on client disconnect or abort
	cancel        context.CancelFunc
	requestJSON   map[string]any
	messages      []map[string]any
	stream        bool
	clientModel   string
	requestID     string
	promptPreview string
	reqClass      RequestClass
	t0            float64
	ip            string

	done     chan struct{}
	doneOnce sync.Once
	route    *RouteDecision // set by the matcher before enqueue
}

func (p *ProcRequest) finish() {
	p.doneOnce.Do(func() {
		if p.cancel != nil {
			p.cancel()
		}
		close(p.done)
	})
}

// abort cancels the request context so any in-flight processing stops at the
// next cancellation check. Safe to call multiple times.
func (p *ProcRequest) abort() {
	if p.cancel != nil {
		p.cancel()
	}
}

// RouteDecision is the matcher's scan + routing result for one request.
type RouteDecision struct {
	canonicalName      string
	promptTokens       int
	firstTokenIDs      []int // prompt tokens from the sizing backend
	hitType            *CacheHitType
	restoreBackend     string // backend holding the best cache ("" = none)
	restoreKey         string // disk cache key ("" for pending-slot hits)
	diskRestoreKey     string // best disk cache candidate found, independent of pending-slot hits
	diskRestoreBackend string // backend holding diskRestoreKey ("" = none)
	candidates         []CandidateBackend
	selected           CandidateBackend
	backendTokenIDs    map[string][]int
	backendBlocks      map[string][]string
	bestRatio          float64
	scanDiagnostics    []map[string]any
	backendCacheRatios map[string]float64
	p2pTriggered       bool
}

// HitInfo identifies the best cache hit found during a scan (nil = no hit).
type HitInfo struct {
	Backend   string
	Canonical string // model name of the hit
}

// --- Per-backend worker ---

// backendWorker is one backend's queue + sequential processing goroutine.
type backendWorker struct {
	beID  string
	queue chan *ProcRequest
	// queued counts items currently sitting in `queue` (guarded by the
	// dispatcher's mu). Incremented when enqueued, decremented when popped;
	// at all times queued == len(queue).
	queued int
	// current is the request being processed right now (nil between
	// requests), guarded by the dispatcher's mu.
	current *ProcRequest
}

// --- Dispatcher ---

type RequestDispatcher struct {
	mu       sync.Mutex
	ctx      context.Context
	cancel   context.CancelFunc
	started  bool
	stopped  bool
	incoming chan *ProcRequest
	activity chan struct{}
	workers  map[string]*backendWorker
	overflow []*ProcRequest // FIFO of requests whose chosen backend was full
	rrIndex  map[string]int // client model -> next round-robin position
	wg       sync.WaitGroup
	active   sync.WaitGroup // workers inside processWorkerRequest
}

var theDispatcher = &RequestDispatcher{}

// GetDispatcher returns the global request dispatcher singleton.
func GetDispatcher() *RequestDispatcher { return theDispatcher }

// Start launches the matcher goroutine (idempotent).
func (d *RequestDispatcher) Start() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.started {
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	d.ctx = ctx
	d.cancel = cancel
	d.incoming = make(chan *ProcRequest, 128)
	d.activity = make(chan struct{}, 64)
	d.workers = map[string]*backendWorker{}
	d.rrIndex = map[string]int{}
	d.started = true
	d.wg.Add(1)
	go d.run()
	logInfo("matcher", "Request matcher started (queue_max=%d, hit_limit=%d)", BackendQueueMax, CacheHitQueueLimit)
}

// Stop cancels the matcher and workers, discards parked requests, and waits
// up to timeout for in-flight work to drain.
func (d *RequestDispatcher) Stop(timeout time.Duration) {
	d.mu.Lock()
	if !d.started || d.stopped {
		d.mu.Unlock()
		return
	}
	d.stopped = true
	d.cancel()
	overflow := d.overflow
	d.overflow = nil
	d.mu.Unlock()
	for _, req := range overflow {
		d.discard(req, "shutdown")
	}
	done := make(chan struct{})
	go func() { d.wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(timeout):
		logWarn("matcher", "Timed out waiting for workers to drain after shutdown")
	}
}

// AbortAll cancels every parked, queued, and in-flight request, then waits up
// to timeout for in-flight processing to wind down. Queued items are dropped
// by their workers at the next dequeue (the cancellation check). The matcher
// loop itself keeps running; this is meant for tests that swap out the global
// backend manager while worker goroutines may still reference it.
func (d *RequestDispatcher) AbortAll(timeout time.Duration) {
	d.mu.Lock()
	if !d.started {
		d.mu.Unlock()
		return
	}
	var toAbort []*ProcRequest
	toAbort = append(toAbort, d.overflow...)
	d.overflow = nil
	for _, w := range d.workers {
		if w.current != nil {
			toAbort = append(toAbort, w.current)
		}
	}
	d.mu.Unlock()
	for _, req := range toAbort {
		req.abort()
	}
	done := make(chan struct{})
	go func() { d.active.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(timeout):
		logWarn("matcher", "Timed out waiting for in-flight requests to abort")
	}
}

// Submit hands a request to the matcher. Returns false when the dispatcher
// is stopped (shutting down). Blocks while the matcher is busy scanning.
func (d *RequestDispatcher) Submit(req *ProcRequest) bool {
	d.mu.Lock()
	if !d.started || d.stopped {
		d.mu.Unlock()
		return false
	}
	d.mu.Unlock()
	d.incoming <- req
	return true
}

// notify wakes the matcher so it re-examines the overflow queue (e.g. after
// liveness discovery changes the backend set).
func (d *RequestDispatcher) notify() {
	d.mu.Lock()
	ok := d.started && !d.stopped
	d.mu.Unlock()
	if !ok {
		return
	}
	select {
	case d.activity <- struct{}{}:
	default:
	}
}

// QueueDepth returns the number of queued (not yet processed) requests for a backend.
func (d *RequestDispatcher) QueueDepth(beID string) int {
	d.mu.Lock()
	defer d.mu.Unlock()
	if w, ok := d.workers[beID]; ok {
		return w.queued
	}
	return 0
}

// OverflowLen returns the number of requests parked in the global queue.
func (d *RequestDispatcher) OverflowLen() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.overflow)
}

// QueuesSnapshot returns per-backend queue state (queued count, in-flight
// flag), the per-backend queue cap, and the global overflow length.
func (d *RequestDispatcher) QueuesSnapshot() map[string]any {
	d.mu.Lock()
	defer d.mu.Unlock()
	out := map[string]any{}
	for beID, w := range d.workers {
		inFlight := 0
		if w.current != nil {
			inFlight = 1
		}
		out[beID] = map[string]any{"queued": w.queued, "in_flight": inFlight}
	}
	out["queue_max"] = BackendQueueMax
	out["overflow"] = len(d.overflow)
	return out
}

// --- Matcher loop ---

func (d *RequestDispatcher) run() {
	defer d.wg.Done()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	var burst []*ProcRequest
	for {
		select {
		case <-d.ctx.Done():
			return
		case req, ok := <-d.incoming:
			if !ok {
				return
			}
			burst = append(burst, req)
		case <-d.activity:
		case <-ticker.C:
		}
		drain:
		for {
			select {
			case req := <-d.incoming:
				burst = append(burst, req)
			default:
				break drain
			}
		}
		for _, req := range burst {
			d.handleArrival(req)
		}
		burst = burst[:0]
		d.drainOverflow()
	}
}

func (d *RequestDispatcher) handleArrival(req *ProcRequest) {
	if err := req.ctx.Err(); err != nil {
		d.discard(req, "client_disconnected")
		return
	}
	decision, ok := d.match(req)
	if !ok {
		return // early error already written and finished
	}
	req.route = decision
	if decision.selected.BackendID == "" {
		d.parkOverflow(req)
		return
	}
	if d.tryEnqueue(req, decision.selected.BackendID) {
		logInfo("matcher", "Routed request %s to backend '%s' (%s)", req.requestID, decision.selected.BackendID, routingLogExtra(decision))
	} else {
		d.parkOverflow(req)
	}
}

func (d *RequestDispatcher) drainOverflow() {
	for {
		d.mu.Lock()
		if len(d.overflow) == 0 {
			d.mu.Unlock()
			return
		}
		req := d.overflow[0]
		d.overflow = d.overflow[1:]
		d.mu.Unlock()

		if err := req.ctx.Err(); err != nil {
			d.discard(req, "client_disconnected")
			continue
		}
		decision, ok := d.match(req)
		if !ok {
			continue
		}
		req.route = decision
		if decision.selected.BackendID == "" || !d.tryEnqueue(req, decision.selected.BackendID) {
			// Head of line cannot be routed yet — put it back and stop the
			// pass so FIFO order is preserved.
			d.mu.Lock()
			d.overflow = append([]*ProcRequest{req}, d.overflow...)
			d.mu.Unlock()
			return
		}
		logInfo("matcher", "Routed overflow request %s to backend '%s' (%s)", req.requestID, decision.selected.BackendID, routingLogExtra(decision))
	}
}

func (d *RequestDispatcher) parkOverflow(req *ProcRequest) {
	d.mu.Lock()
	d.overflow = append(d.overflow, req)
	n := len(d.overflow)
	d.mu.Unlock()
	logInfo("matcher", "No backend queue has room; parked request %s in global overflow queue (%d waiting)", req.requestID, n)
}

// Requeue hands a request back to the tail of the global overflow queue for
// re-matching (used when the serving backend stops serving the model).
func (d *RequestDispatcher) Requeue(req *ProcRequest) {
	d.mu.Lock()
	d.overflow = append(d.overflow, req)
	n := len(d.overflow)
	d.mu.Unlock()
	logWarn("matcher", "Requeued request %s to global overflow queue (%d waiting)", req.requestID, n)
	d.notify()
}

func (d *RequestDispatcher) discard(req *ProcRequest, reason string) {
	Metrics.Record(map[string]any{
		"request_id":    req.requestID,
		"model":         req.clientModel,
		"t0":            req.t0,
		"latency_ms":    latencyMS(req.t0),
		"stream":        req.stream,
		"status":        "cancelled",
		"cancel_reason": reason,
	})
	logWarn("matcher", "Discarded request %s: %s", req.requestID, reason)
	req.finish()
}

// failEarly writes an error response for a request that never entered a queue.
func (d *RequestDispatcher) failEarly(req *ProcRequest, status int, msg string) {
	recordEarlyError(req.requestID, req.clientModel, req.t0)
	WriteJSON(req.w, status, map[string]any{"error": msg})
	req.finish()
}

func routingLogExtra(decision *RouteDecision) string {
	if decision.p2pTriggered {
		return fmt.Sprintf("p2p key=%s", key16(decision.restoreKey))
	}
	return "direct"
}

// --- Matching (scan + selection) ---

// match scans the request (model resolution, per-backend tokenize, disk and
// pending-slot cache scan) and selects a target backend. On early failures it
// writes the error response itself and returns ok=false.
func (d *RequestDispatcher) match(req *ProcRequest) (*RouteDecision, bool) {
	bm := backendManager
	options := bm.GetDiscoveredModels(req.clientModel)
	if len(options) == 0 {
		_ = bm.DiscoverModels()
		options = bm.GetDiscoveredModels(req.clientModel)
	}
	if len(options) == 0 {
		d.failEarly(req, http.StatusBadRequest, fmt.Sprintf("model '%s' not found", req.clientModel))
		return nil, false
	}
	firstOpt := options[0]
	if len(firstOpt.Backends) == 0 {
		d.failEarly(req, http.StatusBadGateway, "backend unavailable")
		return nil, false
	}

	scanCtx, cancel := context.WithTimeout(req.ctx, time.Duration(MatchScanTimeout*float64(time.Second)))
	defer cancel()

	// Size the prompt on the first live backend of the first option.
	firstBeID := ""
	for _, beID := range firstOpt.Backends {
		if bm.GetBackendState(beID) {
			firstBeID = beID
			break
		}
	}
	if firstBeID == "" {
		d.failEarly(req, http.StatusServiceUnavailable, "backend unreachable")
		return nil, false
	}
	firstClient := bm.GetClient(firstBeID)
	templated, err := firstClient.ApplyChatTemplate(scanCtx, req.messages)
	if err != nil {
		d.handleTokenizeError(req, err)
		return nil, false
	}
	firstTokenIDs, err := firstClient.Tokenize(scanCtx, templated, true)
	if err != nil {
		d.handleTokenizeError(req, err)
		return nil, false
	}
	promptTokens := len(firstTokenIDs)

	minCtx := options[0].NCtx
	for _, opt := range options[1:] {
		if opt.NCtx < minCtx {
			minCtx = opt.NCtx
		}
	}
	if promptTokens >= minCtx {
		d.failEarly(req, http.StatusBadRequest, fmt.Sprintf("prompt too long (tokens=%d, n_ctx=%d)", promptTokens, minCtx))
		return nil, false
	}

	decision := &RouteDecision{
		canonicalName:      firstOpt.Name,
		promptTokens:       promptTokens,
		firstTokenIDs:      firstTokenIDs,
		backendTokenIDs:    map[string][]int{},
		backendBlocks:      map[string][]string{},
		backendCacheRatios: map[string]float64{},
	}

	scanDiagnostics := []map[string]any{}
	var hit *HitInfo
	bestDiskKey := ""
	bestDiskBackend := ""
	bestDiskRatio := 0.0

	for _, opt := range options {
		for _, beID := range opt.Backends {
			optClient := bm.GetClient(beID)
			diagEntry := map[string]any{
				"model":   opt.Name,
				"backend": beID,
			}
			if !bm.GetBackendState(beID) {
				diagEntry["status"] = "down"
				scanDiagnostics = append(scanDiagnostics, diagEntry)
				continue
			}
			var optTokenIDs []int
			optTemplated, tErr := optClient.ApplyChatTemplate(scanCtx, req.messages)
			if tErr == nil {
				optTokenIDs, tErr = optClient.Tokenize(scanCtx, optTemplated, true)
			}
			if tErr != nil {
				if errors.Is(tErr, ErrConn) {
					logWarn("matcher", "Backend %s error, skipping for model '%s' from client %s", beID, opt.Name, req.ip)
					diagEntry["status"] = "unreachable"
					scanDiagnostics = append(scanDiagnostics, diagEntry)
					continue
				}
				logError("matcher", "Backend %s scan error for model '%s' from client %s: %v", beID, opt.Name, req.ip, tErr)
				recordEarlyError(req.requestID, req.clientModel, req.t0)
				WriteJSON(req.w, http.StatusInternalServerError, map[string]any{"error": tErr.Error()})
				req.finish()
				return nil, false
			}
			optBlocks := BlockHashesFromTokens(optTokenIDs, WordsPerBlock)
			decision.backendTokenIDs[beID] = optTokenIDs
			decision.backendBlocks[beID] = optBlocks
			diagEntry["n_blocks"] = len(optBlocks)
			diagEntry["n_tokens"] = len(optTokenIDs)

			if bm.CacheEnabled(beID) {
				if candKey, candRatio, found := kvMeta.FindBestRestoreCandidate(optBlocks, WordsPerBlock, LCPTh, opt.Name, beID); found {
					diagEntry["cache_file_key"] = key16(candKey)
					diagEntry["cache_file_ratio"] = round4(candRatio)
					if bestDiskKey == "" || candRatio > bestDiskRatio {
						bestDiskKey = candKey
						bestDiskBackend = beID
						bestDiskRatio = candRatio
					}
					if hit == nil || candRatio > decision.bestRatio {
						decision.bestRatio = candRatio
						decision.restoreKey = candKey
						decision.restoreBackend = beID
						decision.canonicalName = opt.Name
						dt := CacheHitDiskRestore
						decision.hitType = &dt
						hit = &HitInfo{Backend: beID, Canonical: opt.Name}
						logInfo("matcher", "Cache hit: key '%s' (model '%s', backend '%s', ratio %.3f)",
							key16(candKey), opt.Name, beID, candRatio)
					}
				} else {
					diagEntry["cache_file_ratio"] = nil
				}

				beSm := slotManager.Get(beID)
				pendingRatios := []map[string]any{}
				for slotID, kvBlocks := range beSm.GetKVStates() {
					if len(optBlocks) < len(kvBlocks)-1 {
						pendingRatios = append(pendingRatios, map[string]any{
							"slot": slotID, "lcp_blocks": 0, "slot_blocks": len(kvBlocks), "ratio": 0.0,
						})
						continue
					}
					lcp := 0
					var ratio float64
					if len(optBlocks) > 0 {
						lcp = LCPBlocks(optBlocks, kvBlocks)
						ratio = float64(lcp) / float64(len(optBlocks))
					}
					pendingRatios = append(pendingRatios, map[string]any{
						"slot": slotID, "lcp_blocks": lcp, "slot_blocks": len(kvBlocks), "ratio": round4(ratio),
					})
					if ratio >= LCPTh && (hit == nil || ratio >= decision.bestRatio) {
						decision.bestRatio = ratio
						decision.restoreKey = ""
						decision.restoreBackend = beID
						decision.canonicalName = opt.Name
						dt := CacheHitSkip
						decision.hitType = &dt
						hit = &HitInfo{Backend: beID, Canonical: opt.Name}
						logInfo("matcher", "Pending slot cache hit: model '%s', backend '%s', slot %d, ratio %.3f",
							opt.Name, beID, slotID, ratio)
					}
				}
				diagEntry["pending_slots"] = pendingRatios
			} else {
				diagEntry["cache_file_ratio"] = nil
				diagEntry["pending_slots"] = []map[string]any{}
			}
			scanDiagnostics = append(scanDiagnostics, diagEntry)

			beBest := 0.0
			if v, okV := diagEntry["cache_file_ratio"].(float64); okV {
				beBest = v
			}
			if psList, okV := diagEntry["pending_slots"].([]map[string]any); okV {
				for _, ps := range psList {
					if v, okV := ps["ratio"].(float64); okV && v > beBest {
						beBest = v
					}
				}
			}
			if beBest > 0 {
				decision.backendCacheRatios[beID] = beBest
			}
		}
	}
	decision.scanDiagnostics = scanDiagnostics
	decision.diskRestoreKey = bestDiskKey
	decision.diskRestoreBackend = bestDiskBackend

	// Candidates: flattened (backend, model) pairs, deduped, live backends only.
	cands := []CandidateBackend{}
	seen := map[string]bool{}
	for _, opt := range options {
		for _, beID := range opt.Backends {
			if !bm.GetBackendState(beID) {
				continue
			}
			pairKey := beID + "|" + opt.Name
			if seen[pairKey] {
				continue
			}
			seen[pairKey] = true
			cands = append(cands, CandidateBackend{BackendID: beID, ModelName: opt.Name})
		}
	}
	decision.candidates = cands

	pendingOf := func(beID string) int {
		d.mu.Lock()
		defer d.mu.Unlock()
		if w, ok := d.workers[beID]; ok {
			return w.queued
		}
		return 0
	}
	startIdx := 0
	d.mu.Lock()
	startIdx = d.rrIndex[req.clientModel]
	d.mu.Unlock()

	idx, reason := SelectBackend(cands, hit, pendingOf, startIdx)
	if idx < 0 {
		logInfo("matcher", "No backend queue has room for request %s (%s)", req.requestID, reason)
		return decision, true
	}

	d.mu.Lock()
	d.rrIndex[req.clientModel] = (idx + 1) % len(cands)
	d.mu.Unlock()
	decision.selected = cands[idx]

	// Cache hit that cannot be served where the cache lives: migrate the
	// best disk cache to the selected backend so it is ready when the request
	// is processed (async; bounded wait in the worker). This covers both a
	// disk-restore hit and a pending-slot hit (whose RAM state can't move, but
	// whose on-disk key can).
	if decision.diskRestoreKey != "" && cands[idx].BackendID != decision.diskRestoreBackend {
		if RequestCacheTransfer(decision.diskRestoreBackend, cands[idx].BackendID, decision.diskRestoreKey) {
			decision.p2pTriggered = true
		}
	}
	return decision, true
}

func (d *RequestDispatcher) handleTokenizeError(req *ProcRequest, err error) {
	var statusErr *HTTPStatusError
	if errors.As(err, &statusErr) {
		detail := statusErr.Body
		if detail == "" {
			detail = statusErr.Error()
		}
		recordEarlyError(req.requestID, req.clientModel, req.t0)
		WriteJSON(req.w, statusErr.StatusCode, map[string]any{"error": detail})
		req.finish()
		return
	}
	if errors.Is(err, ErrConn) {
		recordEarlyError(req.requestID, req.clientModel, req.t0)
		WriteJSON(req.w, http.StatusServiceUnavailable, map[string]any{"error": "backend unreachable"})
		req.finish()
		return
	}
	recordEarlyError(req.requestID, req.clientModel, req.t0)
	WriteJSON(req.w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
	req.finish()
}

// --- Backend selection (pure decision, exported for tests) ---

// SelectBackend picks the index of a candidate backend for a request given
// current per-backend queue depths. It returns (-1, reason) when no candidate
// has room.
//
//  1. If there is a cache hit and the hit backend's queue depth is below
//     CacheHitQueueLimit, the hit backend wins (preferring the candidate pair
//     whose model matches the hit).
//  2. Otherwise round-robin across matching backends starting from the first
//     non-hit candidate at/after startIdx; the saturated hit backend is only
//     considered in a second pass when every other candidate is full.
func SelectBackend(cands []CandidateBackend, hit *HitInfo, pendingOf func(backendID string) int, startIdx int) (int, string) {
	n := len(cands)
	if n == 0 {
		return -1, "no_candidates"
	}
	hasHit := hit != nil && hit.Backend != ""

	if hasHit && pendingOf(hit.Backend) < CacheHitQueueLimit {
		firstOnHitBackend := -1
		for i, c := range cands {
			if c.BackendID != hit.Backend {
				continue
			}
			if c.ModelName == hit.Canonical {
				return i, "cache_hit"
			}
			if firstOnHitBackend == -1 {
				firstOnHitBackend = i
			}
		}
		if firstOnHitBackend != -1 {
			return firstOnHitBackend, "cache_hit"
		}
	}

	start := startIdx % n
	if hasHit {
		for k := 0; k < n; k++ {
			i := (start + k) % n
			if cands[i].BackendID != hit.Backend {
				start = i
				break
			}
		}
	}
	// First pass: every candidate except the saturated hit backend.
	for k := 0; k < n; k++ {
		i := (start + k) % n
		if hasHit && cands[i].BackendID == hit.Backend {
			continue
		}
		if pendingOf(cands[i].BackendID) < BackendQueueMax {
			return i, "fallback_round_robin"
		}
	}
	// Second pass: all others full — take anyone with room (possibly the hit
	// backend, still bounded by the queue cap).
	for k := 0; k < n; k++ {
		i := (start + k) % n
		if pendingOf(cands[i].BackendID) < BackendQueueMax {
			return i, "fallback_round_robin"
		}
	}
	return -1, "all_queues_full"
}

// --- Worker queue management ---

func (d *RequestDispatcher) getWorkerLocked(beID string) *backendWorker {
	w, ok := d.workers[beID]
	if !ok {
		w = &backendWorker{beID: beID, queue: make(chan *ProcRequest, BackendQueueMax)}
		d.workers[beID] = w
		d.wg.Add(1)
		go w.run(d)
	}
	return w
}

// tryEnqueue reserves a queue slot for the backend and enqueues the request.
// Returns false when the queue is full. The reservation and the send are
// consistent: queued counts channel contents exactly, so a successful
// reservation guarantees the send cannot block.
func (d *RequestDispatcher) tryEnqueue(req *ProcRequest, beID string) bool {
	d.mu.Lock()
	w := d.getWorkerLocked(beID)
	if w.queued >= BackendQueueMax {
		d.mu.Unlock()
		return false
	}
	w.queued++
	d.mu.Unlock()
	w.queue <- req
	return true
}

func (w *backendWorker) run(d *RequestDispatcher) {
	defer d.wg.Done()
	for {
		select {
		case <-d.ctx.Done():
			for {
				select {
				case req := <-w.queue:
					d.mu.Lock()
					w.queued--
					d.mu.Unlock()
					d.discard(req, "shutdown")
				default:
					return
				}
			}
		case req := <-w.queue:
			d.mu.Lock()
			w.queued--
			d.mu.Unlock()
			if err := req.ctx.Err(); err != nil {
				d.discard(req, "client_disconnected")
				d.notify()
				continue
			}
			d.mu.Lock()
			w.current = req
			d.mu.Unlock()
			d.active.Add(1)
			processWorkerRequest(d, w, req)
			d.active.Done()
			d.mu.Lock()
			w.current = nil
			d.mu.Unlock()
			d.notify()
		}
	}
}
