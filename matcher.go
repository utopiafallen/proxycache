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
//	The dispatcher's pump (same goroutine) dispatches each backend's queue
//	    head to a fresh worker goroutine while TryAcquire succeeds — the slot
//	    pool is the concurrency limiter, so a backend with N free slots runs
//	    up to N requests in flight. Each worker restores the cache if it is
//	    available on its backend (including files that arrived via P2P
//	    transfer), dispatches to llama.cpp, saves the cache, and releases the
//	    slot; completion re-wakes the pump for the next queued request. The
//	    first request for a model whose pool does not exist yet gets a worker
//	    that performs lazy discovery (refresh + ServesModel gate).
//
// Routing rule: a request whose conversation_id matches a backend's cache or
// warm slot goes to that backend (shortest queue, ties broken by longest LCP)
// while its queue depth stays below CacheHitQueueLimit — a conversation match
// always beats a plain LCP match. Otherwise a request with a cache hit goes to
// the hit backend while its queue depth stays below CacheHitQueueLimit;
// otherwise it is distributed round-robin across the other matching backends
// (starting after the hit backend), which triggers an async P2P cache transfer
// to the chosen backend.
//
// Immediate idle migration: when the chosen hit/conversation backend is
// slot-busy (in-flight work, no free slot) and a fully idle live
// cache-enabled backend serves the model, a request with a transferable disk
// cache is sent to that idle backend at arrival instead of queueing on the hit
// backend — its disk cache is P2P-transferred over, so it restores there
// rather than waiting QueueMigrationAfter for the stale-queue monitor.

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
	w              http.ResponseWriter
	r              *http.Request
	ctx            context.Context // derived from r.Context(); cancelled on client disconnect or abort
	cancel         context.CancelFunc
	requestJSON    map[string]any
	messages       []map[string]any
	stream         bool
	clientModel    string
	conversationID string
	requestID      string
	promptPreview  string
	reqClass       RequestClass
	t0             float64
	ip             string

	done     chan struct{}
	doneOnce sync.Once
	route    *RouteDecision // set by the matcher before enqueue

	// enqueuedAt is when the request was placed on its current backend queue
	// (used for queue-migration aging); migratedFrom is the source backend
	// when the dispatcher moved the request to a different (idle) one, "" if
	// never migrated. A request migrates at most once.
	enqueuedAt   time.Time
	migratedFrom string
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
	convEntries        map[string]*ConvCandidate // per-backend best conversation_id entry (nil when the request carries none)
	selectionReason    string                    // SelectBackend's reason ("conversation_hit", "cache_hit", ...)
}

// HitInfo identifies the best cache hit found during a scan (nil = no hit).
type HitInfo struct {
	Backend   string
	Canonical string // model name of the hit
}

// --- Per-backend queue ---

// backendWorker is one backend's FIFO request queue. There is no long-lived
// goroutine per backend: the dispatcher's pump spawns a worker goroutine for
// each dispatched request, bounded by the slot pool (one in flight per free
// slot). All fields are guarded by the dispatcher's mu.
type backendWorker struct {
	beID string
	// queue holds requests waiting for a free slot, FIFO.
	queue []*ProcRequest
	// inFlight holds requests whose worker goroutine has been spawned and
	// not yet finished (each holds — or, in the lazy-discovery case, is
	// working toward — exactly one slot).
	inFlight []*ProcRequest
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

// Stop cancels the matcher, discards parked and queued requests, and waits up
// to timeout for in-flight work to drain.
func (d *RequestDispatcher) Stop(timeout time.Duration) {
	d.mu.Lock()
	if !d.started || d.stopped {
		d.mu.Unlock()
		return
	}
	d.stopped = true
	d.cancel()
	toDiscard := d.takeParkedLocked()
	d.mu.Unlock()
	for _, req := range toDiscard {
		d.discard(req, "shutdown")
	}
	done := make(chan struct{})
	go func() { d.active.Wait(); d.wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(timeout):
		logWarn("matcher", "Timed out waiting for workers to drain after shutdown")
	}
}

// takeParkedLocked returns and clears every non-in-flight request: the global
// overflow queue plus everything sitting in per-backend queues. Caller holds d.mu.
func (d *RequestDispatcher) takeParkedLocked() []*ProcRequest {
	out := d.overflow
	d.overflow = nil
	for _, w := range d.workers {
		out = append(out, w.queue...)
		w.queue = nil
	}
	return out
}

// AbortAll cancels every parked, queued, and in-flight request, then waits up
// to timeout for in-flight processing to wind down. The matcher loop itself
// keeps running; this is meant for tests that swap out the global backend
// manager while worker goroutines may still reference it.
func (d *RequestDispatcher) AbortAll(timeout time.Duration) {
	d.mu.Lock()
	if !d.started {
		d.mu.Unlock()
		return
	}
	toDiscard := d.takeParkedLocked()
	var toAbort []*ProcRequest
	for _, w := range d.workers {
		toAbort = append(toAbort, w.inFlight...)
		w.inFlight = nil
	}
	d.mu.Unlock()
	for _, req := range toDiscard {
		d.discard(req, "shutdown")
	}
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

// QueueDepth returns the number of queued (not yet dispatched) requests for a backend.
func (d *RequestDispatcher) QueueDepth(beID string) int {
	d.mu.Lock()
	defer d.mu.Unlock()
	if w, ok := d.workers[beID]; ok {
		return len(w.queue)
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
// count — up to the number of free slots), the per-backend queue cap, and the
// global overflow length.
func (d *RequestDispatcher) QueuesSnapshot() map[string]any {
	d.mu.Lock()
	defer d.mu.Unlock()
	out := map[string]any{}
	for beID, w := range d.workers {
		out[beID] = map[string]any{"queued": len(w.queue), "in_flight": len(w.inFlight)}
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
		d.migrateStale()
		d.pumpAll()
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

// migrateStale moves queued requests that have waited longer than
// QueueMigrationAfter to a fully idle backend (empty queue, no in-flight work)
// that serves the same model, so they start immediately instead of waiting out
// the source's backlog. Runs on the dispatcher goroutine after every overflow
// drain and on each tick. Requests with a transferable disk cache are normally
// migrated at arrival instead (immediateMigrationTarget); this monitor covers
// the rest — idle capacity that appeared only after arrival, and requests
// without a disk hit. Guard rails:
//   - only requests older than QueueMigrationAfter are eligible; a request
//     whose hit is not a transferable disk key (pending-slot-only, or no cache
//     at all) needs twice that age — for it, migrating trades a cheap future
//     restore for an immediate full recompute on the target.
//   - each request migrates at most once (migratedFrom stays set).
//   - at most one request per idle backend per pass (the target stops being
//     idle the moment it takes work).
//
// When the request's best disk key does not already live on the target, a P2P
// transfer is triggered so the target worker can restore from it (bounded wait;
// recompute fallback if the transfer loses the race). A request whose arrival
// scan found no disk hit at all gets its target's disk re-scanned after the
// move (rescanMigratedTarget): a backend that was down at arrival may have
// come back up with a warm cache the arrival scan never saw.
func (d *RequestDispatcher) migrateStale() {
	d.mu.Lock()
	now := time.Now()
	base := time.Duration(QueueMigrationAfter * float64(time.Second))

	var idle []string
	// Idle targets come from the backend registry, not d.workers — a backend
	// that has never been enqueued to has no worker entry yet.
	bm := backendManager
	bm.Mu.RLock()
	beIDs := make([]string, len(bm.KeyOrder))
	copy(beIDs, bm.KeyOrder)
	states := make(map[string]bool, len(bm.BackendState))
	for k, v := range bm.BackendState {
		states[k] = v
	}
	bm.Mu.RUnlock()
	for _, beID := range beIDs {
		if !states[beID] {
			continue
		}
		if w := d.workers[beID]; w != nil && (len(w.queue) > 0 || len(w.inFlight) > 0) {
			continue
		}
		idle = append(idle, beID)
	}

	moved := 0
	type xfer struct{ src, dst, key string }
	var transfers []xfer
	type rescan struct {
		req    *ProcRequest
		target string
	}
	var rescans []rescan
	for srcID, sw := range d.workers {
		for i := 0; i < len(sw.queue); i++ {
			req := sw.queue[i]
			if req.migratedFrom != "" || req.route == nil {
				continue
			}
			need := base
			if req.route.diskRestoreKey == "" {
				need = 2 * base
			}
			if now.Sub(req.enqueuedAt) < need {
				continue
			}
			target := ""
			for _, tb := range idle {
				if tb != srcID && backendManager.ServesModel(req.route.selected.ModelName, tb) {
					target = tb
					break
				}
			}
			if target == "" {
				continue
			}
			age := now.Sub(req.enqueuedAt)
			sw.queue = append(sw.queue[:i], sw.queue[i+1:]...)
			i--
			tw := d.getWorkerLocked(target)
			tw.queue = append(tw.queue, req)
			req.enqueuedAt = now
			req.migratedFrom = srcID
			moved++
			for k, tb := range idle {
				if tb == target {
					idle = append(idle[:k], idle[k+1:]...)
					break
				}
			}
			if req.route.diskRestoreKey != "" && req.route.diskRestoreBackend != target {
				transfers = append(transfers, xfer{req.route.diskRestoreBackend, target, req.route.diskRestoreKey})
			} else if req.route.diskRestoreKey == "" {
				// No disk hit at arrival — the target may have come up (with a
				// warm cache) since. Re-check its disk after the lock drops.
				rescans = append(rescans, rescan{req, target})
			}
			migKey := "none"
			if k := req.route.diskRestoreKey; k != "" {
				migKey = key16(k)
			}
			logInfo("matcher", "Migrated stale request %s from backend '%s' to idle backend '%s' (queued %.0fs, key=%s)",
				req.requestID, srcID, target, age.Seconds(), migKey)
		}
	}
	d.mu.Unlock()
	// Re-scan the disk of targets for cold migrated requests, then fire P2P
	// transfers — both outside the lock (tokenize HTTP + disk I/O + file stats).
	// The pump runs after migrateStale returns, so route updates here are always
	// visible to the moved request's worker.
	for _, rs := range rescans {
		d.rescanMigratedTarget(rs.req, rs.target)
	}
	for _, xf := range transfers {
		RequestCacheTransfer(xf.src, xf.dst, xf.key)
	}
	if moved > 0 {
		logInfo("matcher", "Queue migration: moved %d stale request(s) to idle backends", moved)
	}
}

// immediateMigrationTarget returns a fully idle live cache-enabled backend
// that serves the selected model, when the selected backend is slot-busy
// (in-flight work, no free slot for the model) and would therefore make the
// request wait in its queue. It is the selection-time counterpart of
// migrateStale's idle scan: instead of queueing on the hit backend and waiting
// QueueMigrationAfter for the monitor to move the request, the request is
// routed to the idle backend at arrival and its disk cache is P2P-transferred
// there. Returns "" when the selected backend can dispatch the request right
// away or no eligible idle backend exists. Runs on the dispatcher goroutine.
func (d *RequestDispatcher) immediateMigrationTarget(dec *RouteDecision) string {
	sel := dec.selected.BackendID
	model := dec.selected.ModelName
	d.mu.Lock()
	defer d.mu.Unlock()
	busy := false
	if w := d.workers[sel]; w != nil && len(w.inFlight) > 0 {
		// The pump dispatches a queued request only when it can acquire a
		// slot: GetPool lists ALL of the model's slot IDs (free ones tracked
		// separately in inUse), so the pool "existing" is not the same as a
		// slot being available. Busy = pool exists but every slot is held.
		beSm := slotManager.Get(sel)
		busy = beSm.GetPool(model) != nil && !beSm.HasFreeSlot(model)
	}
	if !busy {
		return ""
	}
	// Idle targets come from the backend registry, not d.workers — a backend
	// that has never been enqueued to has no worker entry yet.
	bm := backendManager
	bm.Mu.RLock()
	beIDs := make([]string, len(bm.KeyOrder))
	copy(beIDs, bm.KeyOrder)
	states := make(map[string]bool, len(bm.BackendState))
	for k, v := range bm.BackendState {
		states[k] = v
	}
	bm.Mu.RUnlock()
	for _, beID := range beIDs {
		if beID == sel || !states[beID] {
			continue
		}
		if w := d.workers[beID]; w != nil && (len(w.queue) > 0 || len(w.inFlight) > 0) {
			continue
		}
		// The transferred disk cache must be able to land on the target.
		if !bm.CacheEnabled(beID) || !bm.ServesModel(model, beID) {
			continue
		}
		return beID
	}
	return ""
}

// rescanMigratedTarget re-checks the migration target's disk cache for a
// request whose arrival scan found no disk hit. The arrival decision is stale
// in one direction: a backend that was down when the request arrived can come
// back up (with a warm cache) before the request migrates to it, and the
// arrival scan never saw its disk. When a candidate is found the route
// decision is updated as if the scan had hit there, so the target worker
// restores from the local file directly (no transfer needed) and recompute
// detection / save heuristics treat the request as a normal cache hit.
// Runs after the dispatcher lock drops (tokenize HTTP + disk I/O) but before
// the pump dispatches the moved request, so the worker always sees the update.
func (d *RequestDispatcher) rescanMigratedTarget(req *ProcRequest, target string) {
	dec := req.route
	modelName := dec.selected.ModelName
	if !backendManager.CacheEnabled(target) {
		return
	}
	blocks := dec.backendBlocks[target]
	ctx, cancel := context.WithTimeout(req.ctx, time.Duration(MatchScanTimeout*float64(time.Second)))
	defer cancel()
	if len(blocks) == 0 {
		// The target was down at arrival, so it was never tokenized. Do it
		// now — for the scan below, and so the worker's save key uses this
		// backend's tokenizer instead of falling back to the sizing backend's.
		client := backendManager.GetClient(target)
		templated, err := client.ApplyChatTemplate(ctx, req.messages)
		if err != nil {
			logWarn("matcher", "Migration rescan: tokenize on target '%s' failed for request %s: %v", target, req.requestID, err)
			return
		}
		tokenIDs, err := client.Tokenize(ctx, templated, true)
		if err != nil {
			logWarn("matcher", "Migration rescan: tokenize on target '%s' failed for request %s: %v", target, req.requestID, err)
			return
		}
		blocks = BlockHashesFromTokens(tokenIDs, WordsPerBlock)
		dec.backendTokenIDs[target] = tokenIDs
		dec.backendBlocks[target] = blocks
	}
	key, ratio, found := kvMeta.FindBestRestoreCandidate(blocks, WordsPerBlock, LCPTh, modelName, target)
	if !found {
		return
	}
	dec.diskRestoreKey = key
	dec.diskRestoreBackend = target
	dec.restoreKey = key
	dec.restoreBackend = target
	dec.bestRatio = ratio
	dec.canonicalName = modelName
	dt := CacheHitDiskRestore
	dec.hitType = &dt
	if dec.backendCacheRatios[target] < ratio {
		dec.backendCacheRatios[target] = ratio
	}
	logInfo("matcher", "Migration rescan: target '%s' holds a local disk cache for request %s (key=%s, ratio %.3f)",
		target, req.requestID, key16(key), ratio)
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
		key := decision.restoreKey
		if key == "" {
			key = decision.diskRestoreKey
		}
		return fmt.Sprintf("p2p key=%s", key16(key))
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

	// Size the prompt on the first live cache-enabled backend of the first
	// option (exact tokenization). When no such backend is live, size
	// approximately from the raw messages — cache-less backends (e.g. ninfer)
	// expose no /apply-template or /tokenize endpoint to call.
	firstBeID := ""
	for _, beID := range firstOpt.Backends {
		if bm.GetBackendState(beID) && bm.CacheEnabled(beID) {
			firstBeID = beID
			break
		}
	}
	approximateSizing := false
	if firstBeID == "" {
		for _, beID := range firstOpt.Backends {
			if bm.GetBackendState(beID) {
				firstBeID = beID
				approximateSizing = true
				break
			}
		}
	}
	if firstBeID == "" {
		d.failEarly(req, http.StatusServiceUnavailable, "backend unreachable")
		return nil, false
	}

	var promptTokens int
	var firstTokenIDs []int
	if approximateSizing {
		promptTokens = len(MessageBlocks(req.messages, WordsPerBlock)) * WordsPerBlock
		logInfo("matcher", "Approximate prompt sizing: ~%d tokens for model '%s' (no live cache-enabled backend)", promptTokens, req.clientModel)
	} else {
		firstClient := bm.GetClient(firstBeID)
		templated, err := firstClient.ApplyChatTemplate(scanCtx, req.messages)
		if err != nil {
			d.handleTokenizeError(req, err)
			return nil, false
		}
		firstTokenIDs, err = firstClient.Tokenize(scanCtx, templated, true)
		if err != nil {
			d.handleTokenizeError(req, err)
			return nil, false
		}
		promptTokens = len(firstTokenIDs)
	}

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

	// Conversation tracking: the request's top-level conversation_id is
	// matched against conversation-tagged disk entries and warm slots.
	var convEntries map[string]*ConvCandidate
	var convDiskKey, convDiskBackend string
	var convDiskRatio float64
	if req.conversationID != "" {
		convEntries = map[string]*ConvCandidate{}
		decision.convEntries = convEntries
	}

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
			var optBlocks []string
			var optTemplated string
			var tErr error
			if bm.CacheEnabled(beID) {
				optTemplated, tErr = optClient.ApplyChatTemplate(scanCtx, req.messages)
				if tErr == nil {
					optTokenIDs, tErr = optClient.Tokenize(scanCtx, optTemplated, true)
				}
				if tErr == nil {
					optBlocks = BlockHashesFromTokens(optTokenIDs, WordsPerBlock)
				}
			} else {
				// Cache-less backend (e.g. ninfer): no tokenize endpoint.
				// Approximate the KV state from the raw messages so pending-hit
				// routing still works (affinity only — not a real token stream).
				optBlocks = MessageBlocks(req.messages, WordsPerBlock)
				diagEntry["approximate"] = true
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
			decision.backendTokenIDs[beID] = optTokenIDs
			decision.backendBlocks[beID] = optBlocks
			diagEntry["n_blocks"] = len(optBlocks)
			if len(optTokenIDs) > 0 {
				diagEntry["n_tokens"] = len(optTokenIDs)
			} else {
				diagEntry["n_tokens"] = nil
			}

			if bm.CacheEnabled(beID) {
				mc := kvMeta.FindBestCandidates(optBlocks, WordsPerBlock, LCPTh, opt.Name, beID, req.conversationID)
				if mc.BestFound {
					diagEntry["cache_file_key"] = key16(mc.BestKey)
					diagEntry["cache_file_ratio"] = round4(mc.BestRatio)
					if bestDiskKey == "" || mc.BestRatio > bestDiskRatio {
						bestDiskKey = mc.BestKey
						bestDiskBackend = beID
						bestDiskRatio = mc.BestRatio
					}
					if hit == nil || mc.BestRatio > decision.bestRatio {
						decision.bestRatio = mc.BestRatio
						decision.restoreKey = mc.BestKey
						decision.restoreBackend = beID
						decision.canonicalName = opt.Name
						dt := CacheHitDiskRestore
						decision.hitType = &dt
						hit = &HitInfo{Backend: beID, Canonical: opt.Name}
						logInfo("matcher", "Cache hit: key '%s' (model '%s', backend '%s', ratio %.3f)",
							key16(mc.BestKey), opt.Name, beID, mc.BestRatio)
					}
				} else {
					diagEntry["cache_file_ratio"] = nil
				}
				if convEntries != nil {
					if mc.ConvFound {
						diagEntry["conv_key"] = key16(mc.ConvKey)
						diagEntry["conv_ratio"] = round4(mc.ConvRatio)
						if ce := convEntries[beID]; ce == nil || mc.ConvRatio > ce.Ratio {
							convEntries[beID] = &ConvCandidate{Ratio: mc.ConvRatio, Key: mc.ConvKey, Model: opt.Name}
						}
						if mc.ConvRatio > convDiskRatio {
							convDiskRatio = mc.ConvRatio
							convDiskKey = mc.ConvKey
							convDiskBackend = beID
						}
					} else {
						diagEntry["conv_key"] = nil
						diagEntry["conv_ratio"] = nil
					}
				}
			} else {
				diagEntry["cache_file_ratio"] = nil
				if convEntries != nil {
					diagEntry["conv_key"] = nil
					diagEntry["conv_ratio"] = nil
				}
			}

			// Pending-slot scan runs for cache-less backends too, against their
			// approximate message blocks.
			beSm := slotManager.Get(beID)
			slotConvs := beSm.GetSlotConversations()
			pendingRatios := []map[string]any{}
			for slotID, kvBlocks := range beSm.GetKVStates() {
				if len(optBlocks) < len(kvBlocks)-1 {
					pendingRatios = append(pendingRatios, map[string]any{
						"slot": slotID, "lcp_blocks": 0, "slot_blocks": len(kvBlocks), "ratio": 0.0,
						"conversation": slotConvs[slotID],
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
					"conversation": slotConvs[slotID],
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
				// Conversation-tagged pending hit: a warm slot whose last
				// conversation is this request's conversation — the strongest
				// conversation match (nothing to restore).
				if convEntries != nil && ratio >= LCPTh && slotConvs[slotID] == req.conversationID {
					if ce := convEntries[beID]; ce == nil || ratio > ce.Ratio {
						convEntries[beID] = &ConvCandidate{Ratio: ratio, Model: opt.Name}
					}
				}
			}
			diagEntry["pending_slots"] = pendingRatios
			scanDiagnostics = append(scanDiagnostics, diagEntry)

			beBest := 0.0
			if v, okV := diagEntry["cache_file_ratio"].(float64); okV {
				beBest = v
			}
			if v, okV := diagEntry["conv_ratio"].(float64); okV && v > beBest {
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
	// Conversation override: when the request carries a conversation_id and a
	// conversation entry exists on some live backend, the conversation's best
	// disk entry is the restore/transfer key — the conversation's state moves
	// with the request even when another conversation's entry has a higher LCP
	// ratio.
	if convEntries != nil && convDiskKey != "" {
		decision.diskRestoreKey = convDiskKey
		decision.diskRestoreBackend = convDiskBackend
	}

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
			return len(w.queue)
		}
		return 0
	}
	startIdx := 0
	d.mu.Lock()
	startIdx = d.rrIndex[req.clientModel]
	d.mu.Unlock()

	idx, reason := SelectBackend(cands, hit, decision.convEntries, pendingOf, startIdx)
	if idx < 0 {
		logInfo("matcher", "No backend queue has room for request %s (%s)", req.requestID, reason)
		return decision, true
	}

	d.mu.Lock()
	d.rrIndex[req.clientModel] = (idx + 1) % len(cands)
	d.mu.Unlock()
	decision.selected = cands[idx]
	decision.selectionReason = reason
	// Immediate idle migration: a request routed to the backend that holds
	// its cache (or conversation) would sit in that backend's queue while
	// in-flight work drains. If the selected backend is slot-busy right now
	// and a fully idle live cache-enabled backend serves the model, send the
	// request there instead — the P2P trigger below ships the disk cache
	// over and the target worker restores after its bounded transfer wait —
	// rather than letting it wait QueueMigrationAfter for migrateStale. Only
	// requests with a transferable disk cache qualify: pending-slot-only or
	// cold requests would trade a cheap future restore for an immediate full
	// recompute on the target, so they keep the timer.
	if (reason == "cache_hit" || reason == "conversation_hit") && decision.diskRestoreKey != "" {
		if target := d.immediateMigrationTarget(decision); target != "" {
			newIdx := -1
			for i, c := range cands {
				if c.BackendID != target {
					continue
				}
				if c.ModelName == decision.selected.ModelName {
					newIdx = i
					break
				}
				if newIdx == -1 {
					newIdx = i
				}
			}
			if newIdx != -1 {
				src := cands[idx].BackendID
				idx = newIdx
				decision.selected = cands[idx]
				decision.selectionReason = reason + "_idle_migration"
				req.migratedFrom = src
				logInfo("matcher", "Immediate migration: request %s sent to idle backend '%s' instead of slot-busy hit backend '%s' (key=%s)",
					req.requestID, cands[idx].BackendID, src, key16(decision.diskRestoreKey))
			}
		}
	}
	if reason == "conversation_hit" {
		if ce := decision.convEntries[cands[idx].BackendID]; ce != nil {
			decision.bestRatio = ce.Ratio
			decision.canonicalName = ce.Model
			decision.restoreBackend = cands[idx].BackendID
			dt := CacheHitDiskRestore
			if ce.Key == "" {
				dt = CacheHitSkip // warm slot from the same conversation
			}
			decision.restoreKey = ce.Key
			decision.hitType = &dt
			if decision.backendCacheRatios[cands[idx].BackendID] < ce.Ratio {
				decision.backendCacheRatios[cands[idx].BackendID] = ce.Ratio
			}
			convKey := "none"
			if ce.Key != "" {
				convKey = key16(ce.Key)
			}
			logInfo("matcher", "Conversation hit: conversation '%s' on backend '%s' (request %s): key=%s, ratio %.3f, queue depth %d",
				req.conversationID, cands[idx].BackendID, req.requestID, convKey, ce.Ratio, pendingOf(cands[idx].BackendID))
		}
	}

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

// ConvCandidate is one backend's strongest match for the request's
// conversation_id: the best of that backend's conversation-tagged disk entry
// and its warm pending slots. Key is the disk entry's cache key; it is ""
// when the strongest source is a pending slot (nothing to restore from disk
// on that backend).
type ConvCandidate struct {
	Ratio float64
	Key   string
	Model string // canonical model name of the best source
}

// SelectBackend picks the index of a candidate backend for a request given
// current per-backend queue depths. It returns (-1, reason) when no candidate
// has room.
//
//  1. Conversation pass: when the request carries a conversation_id and some
//     candidate backends hold a match for it (conv), a conversation match
//     always beats plain LCP matches regardless of ratio. Among conversation
//     backends whose queue depth is below CacheHitQueueLimit the shortest
//     queue wins, ties broken by longest LCP.
//  2. If there is a cache hit and the hit backend's queue depth is below
//     CacheHitQueueLimit, the hit backend wins (preferring the candidate pair
//     whose model matches the hit).
//  3. Otherwise pick the candidate on the least-loaded backend (lowest queue
//     depth) so idle backends absorb new work first, breaking exact depth ties
//     in round-robin order from the first non-hit candidate at/after startIdx;
//     the saturated hit backend is only considered in a second pass when every
//     other candidate's queue is at the cap.
func SelectBackend(cands []CandidateBackend, hit *HitInfo, conv map[string]*ConvCandidate, pendingOf func(backendID string) int, startIdx int) (int, string) {
	n := len(cands)
	if n == 0 {
		return -1, "no_candidates"
	}
	hasHit := hit != nil && hit.Backend != ""

	if len(conv) > 0 {
		start := startIdx % n
		winBe := ""
		winDepth := 0
		winRatio := 0.0
		for k := 0; k < n; k++ {
			i := (start + k) % n
			cc, ok := conv[cands[i].BackendID]
			if !ok {
				continue
			}
			depth := pendingOf(cands[i].BackendID)
			if depth >= CacheHitQueueLimit {
				continue
			}
			if winBe == "" || depth < winDepth || (depth == winDepth && cc.Ratio > winRatio) {
				winBe = cands[i].BackendID
				winDepth = depth
				winRatio = cc.Ratio
			}
		}
		if winBe != "" {
			cc := conv[winBe]
			firstOnWin := -1
			for i, c := range cands {
				if c.BackendID != winBe {
					continue
				}
				if c.ModelName == cc.Model {
					return i, "conversation_hit"
				}
				if firstOnWin == -1 {
					firstOnWin = i
				}
			}
			return firstOnWin, "conversation_hit"
		}
	}

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
	// Fallback distribution: prefer the candidate on the least-loaded backend
	// (lowest queue depth) so idle backends absorb new work first. Exact depth
	// ties are broken in round-robin order from `start` — the candidate reached
	// first at the minimal depth wins — keeping the rotation fair across equal
	// backends. pick(allowHit) skips the saturated hit backend unless allowHit,
	// so the first pass excludes it and the second takes it only as a last
	// resort.
	pick := func(allowHit bool) int {
		best := -1
		bestDepth := 0
		for k := 0; k < n; k++ {
			i := (start + k) % n
			if !allowHit && cands[i].BackendID == hit.Backend {
				continue
			}
			depth := pendingOf(cands[i].BackendID)
			if depth >= BackendQueueMax {
				continue
			}
			if best == -1 || depth < bestDepth {
				best = i
				bestDepth = depth
			}
		}
		return best
	}

	if i := pick(!hasHit); i != -1 {
		return i, "fallback_round_robin"
	}
	// Only a saturated hit backend makes the second pass distinct: allow it once
	// every other candidate is at the cap. (With no hit, pick(true) above already
	// covered every candidate.)
	if hasHit {
		if i := pick(true); i != -1 {
			return i, "fallback_round_robin"
		}
	}
	return -1, "all_queues_full"
}

// --- Backend queue management + pump ---

func (d *RequestDispatcher) getWorkerLocked(beID string) *backendWorker {
	w, ok := d.workers[beID]
	if !ok {
		w = &backendWorker{beID: beID}
		d.workers[beID] = w
	}
	return w
}

// tryEnqueue reserves a queue slot for the backend and enqueues the request.
// Returns false when the queue is full. The reservation and the append happen
// under the same lock, so a successful reservation is the send itself.
func (d *RequestDispatcher) tryEnqueue(req *ProcRequest, beID string) bool {
	d.mu.Lock()
	w := d.getWorkerLocked(beID)
	if len(w.queue) >= BackendQueueMax {
		d.mu.Unlock()
		return false
	}
	req.enqueuedAt = time.Now()
	w.queue = append(w.queue, req)
	d.mu.Unlock()
	return true
}

// pumpAll dispatches queued requests on every backend as far as free slots
// allow. Runs on the dispatcher goroutine; called after each batch of
// arrivals, after overflow drain, and on every activity/tick wake so that a
// freeing slot promptly starts the next queued request.
func (d *RequestDispatcher) pumpAll() {
	d.mu.Lock()
	beIDs := make([]string, 0, len(d.workers))
	for beID := range d.workers {
		beIDs = append(beIDs, beID)
	}
	d.mu.Unlock()
	for _, beID := range beIDs {
		d.pumpBackend(beID)
	}
}

// pumpBackend keeps dispatching the head of one backend's queue while a free
// slot is available (FIFO head-only). For each dispatch it atomically
// TryAcquires a slot and hands it to a fresh worker goroutine, so the number
// of in-flight workers never exceeds the number of free slots. When the head
// model's pool does not exist yet and nothing is in flight, it instead spawns
// one worker that performs lazy discovery (refresh + ServesModel gate).
func (d *RequestDispatcher) pumpBackend(beID string) {
	for {
		var req *ProcRequest
		slotID := -1
		d.mu.Lock()
		if w, ok := d.workers[beID]; ok && len(w.queue) > 0 {
			head := w.queue[0]
			modelName := head.route.selected.ModelName
			beSm := slotManager.Get(beID)
			pool := beSm.GetPool(modelName)
			if pool != nil && len(pool) > 0 {
				slotID = beSm.TryAcquire(modelName)
			} else if len(w.inFlight) == 0 {
				slotID = -2 // bootstrap: no pool yet, spawn a discovering worker
			}
			if slotID != -1 {
				req = head
				w.queue = w.queue[1:]
				w.inFlight = append(w.inFlight, req)
			}
		}
		d.mu.Unlock()
		if req == nil {
			return
		}
		if err := req.ctx.Err(); err != nil {
			// The client disconnected while the request was queued. If the pump
			// pre-acquired a real slot for it (slotID >= 0), release that slot
			// back to the pool. Discarding without doing so leaks the slot — it
			// stays inUse forever — and once every slot on a backend has leaked,
			// TryAcquire always returns -1 and the backend can never dispatch
			// again (its queue fills to the cap with zero in-flight work).
			if slotID >= 0 {
				slotManager.Get(beID).Release(slotID)
			}
			d.dropInFlight(beID, req)
			d.discard(req, "client_disconnected")
			continue
		}
		d.active.Add(1)
		go d.runWorker(beID, req, slotID)
	}
}

// runWorker is one dispatched request's goroutine: process it (slotID >= 0
// means the pump pre-acquired the slot; slotID == -2 means acquire it here via
// lazy discovery), then release the in-flight mark and re-wake the pump.
func (d *RequestDispatcher) runWorker(beID string, req *ProcRequest, slotID int) {
	defer d.active.Done()
	defer d.workerFinished(beID, req)
	processWorkerRequest(d, beID, req, slotID)
}

// workerFinished clears the in-flight mark and wakes the dispatcher so a freed
// slot can start the next queued request.
func (d *RequestDispatcher) workerFinished(beID string, req *ProcRequest) {
	d.dropInFlight(beID, req)
	d.notify()
}

func (d *RequestDispatcher) dropInFlight(beID string, req *ProcRequest) {
	d.mu.Lock()
	w := d.workers[beID]
	for i, r := range w.inFlight {
		if r == req {
			w.inFlight = append(w.inFlight[:i], w.inFlight[i+1:]...)
			break
		}
	}
	d.mu.Unlock()
}
