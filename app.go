package proxycache

import (
	"context"
	cr "crypto/rand"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

//go:embed dashboard.html
var dashboardHTML []byte

const (
	streamQueueSize         = 16
	recomputeThresholdRatio = 0.7
)

type CandidateBackend struct {
	BackendID string
	ModelName string
}

func newRequestID() string {
	b := make([]byte, 16)
	if _, err := cr.Read(b); err != nil {
		return fmt.Sprintf("req-%d", time.Now().UnixNano())
	}
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

func clientIP(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

func pyBool(v any) bool {
	switch x := v.(type) {
	case bool:
		return x
	case string:
		return x != ""
	case float64:
		return x != 0
	case nil:
		return false
	case []any:
		return len(x) > 0
	case map[string]any:
		return len(x) > 0
	}
	return true
}

func rawMessages(requestJSON map[string]any) []map[string]any {
	raw, ok := requestJSON["messages"].([]any)
	if !ok {
		return []map[string]any{}
	}
	out := make([]map[string]any, 0, len(raw))
	for _, item := range raw {
		if m, ok := item.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

func intFromAny(v any) int {
	switch x := v.(type) {
	case int:
		return x
	case int64:
		return int(x)
	case float64:
		return int(x)
	case json.Number:
		n, _ := x.Int64()
		return int(n)
	case bool:
		if x {
			return 1
		}
		return 0
	}
	return 0
}

func strFromAny(v any) string {
	s, _ := v.(string)
	return s
}

func key16(key string) string {
	if len(key) > 16 {
		return key[:16]
	}
	return key
}

func key16OrNil(key string) any {
	if key == "" {
		return nil
	}
	return key16(key)
}

func strOrNone(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func queryTruthy(s string) bool {
	switch strings.ToLower(s) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

func queryInt(r *http.Request, key string, def int) int {
	s := r.URL.Query().Get(key)
	if s == "" {
		return def
	}
	var v int
	if _, err := fmt.Sscanf(s, "%d", &v); err == nil {
		return v
	}
	return def
}

func containsString(slice []string, s string) bool {
	for _, item := range slice {
		if item == s {
			return true
		}
	}
	return false
}

func hasRoutingDiagnostics(v any) bool {
	switch x := v.(type) {
	case nil:
		return false
	case map[string]any:
		return len(x) > 0
	default:
		return true
	}
}

func latencyMS(t0 float64) float64 {
	return (NowFloat() - t0) * 1000
}

func ctxSleep(ctx context.Context, seconds float64) error {
	if seconds <= 0 {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			return nil
		}
	}
	t := time.NewTimer(time.Duration(seconds * float64(time.Second)))
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func isRecompute(cachedTokens, llmPromptTokens, cacheNTokens int) bool {
	if llmPromptTokens == 0 {
		return false
	}
	expected := cacheNTokens
	if llmPromptTokens < expected {
		expected = llmPromptTokens
	}
	if expected == 0 {
		return false
	}
	return float64(cachedTokens) < float64(expected)*recomputeThresholdRatio
}

func metaNTokens(meta map[string]any, fallback int) int {
	if meta == nil {
		return fallback
	}
	if v, exists := meta["n_tokens"]; exists {
		return intFromAny(v)
	}
	return fallback
}

func recordEarlyError(requestID, model string, t0 float64) {
	Metrics.Record(map[string]any{
		"request_id": requestID,
		"model":      model,
		"latency_ms": latencyMS(t0),
		"status":     "backend_error",
	})
}

func recordChatError(requestID, model, backend string, slotID int, t0 float64, routingReason string) {
	Metrics.Record(map[string]any{
		"request_id":     requestID,
		"model":          model,
		"backend":        backend,
		"slot_id":        slotID,
		"latency_ms":     latencyMS(t0),
		"routing_reason": routingReason,
		"status":         "backend_error",
	})
}

func restorePrevKV(beSm *BackendSlotManager, slotID int, prevKV []string, modelName, backendID string) {
	if prevKV != nil {
		beSm.SetKVState(slotID, prevKV)
		logInfo("app", "Restored slot %d KV state for model '%s' on backend '%s'", slotID, modelName, backendID)
	}
}

func ModelsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		WriteJSON(w, http.StatusMethodNotAllowed, map[string]any{"error": "method not allowed"})
		return
	}
	models := backendManager.SnapshotModels()
	minCtx := DefaultNCtx
	for i, m := range models {
		n := m.NCtx
		if n <= 0 {
			n = DefaultNCtx
		}
		if i == 0 || n < minCtx {
			minCtx = n
		}
	}
	data := []map[string]any{}
	for _, m := range models {
		n := m.NCtx
		if n <= 0 {
			n = DefaultNCtx
		}
		ownedBy := "backend"
		if m.Synthetic {
			ownedBy = "synthetic"
		}
		data = append(data, map[string]any{
			"id":       m.Name,
			"object":   "model",
			"owned_by": ownedBy,
			"n_ctx":    n,
		})
	}
	data = append(data, map[string]any{
		"id":       "any",
		"object":   "model",
		"owned_by": "proxycache",
		"n_ctx":    minCtx,
	})
	WriteJSON(w, http.StatusOK, map[string]any{"data": data})
}

// ChatHandler parses the request, records arrival metrics, hands it to the
// global matcher, and blocks until the response is complete (or the client
// disconnects, which cancels the request context the worker watches).
func ChatHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		WriteJSON(w, http.StatusMethodNotAllowed, map[string]any{"error": "method not allowed"})
		return
	}
	t0 := NowFloat()
	ip := clientIP(r)
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		WriteJSON(w, http.StatusInternalServerError, map[string]any{"error": "failed to read request body"})
		return
	}
	var parsed any
	if err := json.Unmarshal(raw, &parsed); err != nil {
		WriteJSON(w, http.StatusInternalServerError, map[string]any{"error": "invalid JSON body"})
		return
	}
	requestJSON, ok := parsed.(map[string]any)
	if !ok {
		WriteJSON(w, http.StatusInternalServerError, map[string]any{"error": "invalid JSON body"})
		return
	}

	messages := rawMessages(requestJSON)
	stream := pyBool(requestJSON["stream"])
	clientModel := strFromAny(requestJSON["model"])
	if clientModel == "" {
		clientModel = ModelID
	}
	requestID := newRequestID()
	promptPreview := ExtractPromptPreview(requestJSON)
	reqClass := ClassifyRequest(messages, requestJSON)

	Metrics.Record(map[string]any{
		"request_id":            requestID,
		"request_json":          requestJSON,
		"model":                 clientModel,
		"stream":                stream,
		"status":                "incomplete",
		"prompt_preview":        promptPreview,
		"summarization_score":   reqClass.Score,
		"summarization_signals": reqClass.Signals,
	})

	d := GetDispatcher()
	d.Start()
	reqCtx, reqCancel := context.WithCancel(r.Context())
	req := &ProcRequest{
		w:             w,
		r:             r,
		ctx:           reqCtx,
		cancel:        reqCancel,
		requestJSON:   requestJSON,
		messages:      messages,
		stream:        stream,
		clientModel:   clientModel,
		requestID:     requestID,
		promptPreview: promptPreview,
		reqClass:      reqClass,
		t0:            t0,
		ip:            ip,
		done:          make(chan struct{}),
	}
	if !d.Submit(req) {
		recordEarlyError(requestID, clientModel, t0)
		WriteJSON(w, http.StatusServiceUnavailable, map[string]any{"error": "shutting down"})
		return
	}
	select {
	case <-req.done:
	case <-r.Context().Done():
	}
}

func metricsDashboardHandler(w http.ResponseWriter, r *http.Request) {
	summary := Metrics.GetSummary()
	summary["backends"] = getBackendHealth()
	summary["slots"] = getSlotStatus()
	summary["cache"] = getCacheStats()
	summary["backend_model_performance"] = backendManager.GetAllLatencyEMA()
	summary["queues"] = GetDispatcher().QueuesSnapshot()
	WriteJSON(w, http.StatusOK, summary)
}

func metricsHealthHandler(w http.ResponseWriter, r *http.Request) {
	WriteJSON(w, http.StatusOK, getBackendHealth())
}

func metricsSlotsHandler(w http.ResponseWriter, r *http.Request) {
	WriteJSON(w, http.StatusOK, getSlotStatus())
}

func metricsCacheHandler(w http.ResponseWriter, r *http.Request) {
	WriteJSON(w, http.StatusOK, getCacheStats())
}

func metricsDiagnosticsHandler(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	requestID := q.Get("request_id")
	timeline := queryTruthy(q.Get("timeline"))
	livenessDiag := queryTruthy(q.Get("liveness_diag"))
	liveness := queryTruthy(q.Get("liveness"))
	switch {
	case timeline:
		WriteJSON(w, http.StatusOK, map[string]any{"timeline": Metrics.GetTimeline(200)})
	case livenessDiag:
		WriteJSON(w, http.StatusOK, map[string]any{"liveness_diagnostics": Metrics.GetEvents("liveness_diag", 50)})
	case liveness:
		WriteJSON(w, http.StatusOK, map[string]any{"liveness_events": Metrics.GetEvents("liveness_change", 50)})
	case requestID != "":
		req := Metrics.GetRequestByID(requestID)
		if req == nil {
			WriteJSON(w, http.StatusNotFound, map[string]any{"error": "Request not found"})
			return
		}
		WriteJSON(w, http.StatusOK, map[string]any{
			"request_id":          requestID,
			"routing_diagnostics": req["routing_diagnostics"],
		})
	default:
		reqs := Metrics.GetRequests(100, 0)
		diags := []map[string]any{}
		for _, rec := range reqs {
			if hasRoutingDiagnostics(rec["routing_diagnostics"]) {
				diags = append(diags, map[string]any{
					"request_id":          rec["request_id"],
					"routing_diagnostics": rec["routing_diagnostics"],
				})
			}
		}
		WriteJSON(w, http.StatusOK, map[string]any{"diagnostics": diags})
	}
}

func metricsRequestsHandler(w http.ResponseWriter, r *http.Request) {
	limit := queryInt(r, "limit", 100)
	offset := queryInt(r, "offset", 0)
	WriteJSON(w, http.StatusOK, map[string]any{
		"requests": Metrics.GetRequests(limit, offset),
		"total":    Metrics.GetTotalCount(),
	})
}

func metricsRequestByIDHandler(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/metrics/request/")
	if id == "" {
		WriteJSON(w, http.StatusNotFound, map[string]any{"error": "Request not found"})
		return
	}
	req := Metrics.GetRequestByID(id)
	if req == nil {
		WriteJSON(w, http.StatusNotFound, map[string]any{"error": "Request not found"})
		return
	}
	WriteJSON(w, http.StatusOK, req)
}

func metricsPerformanceHandler(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	WriteJSON(w, http.StatusOK, Metrics.GetPerformance(q.Get("model"), q.Get("backend"), ""))
}

func DashboardHandler(w http.ResponseWriter, r *http.Request) {
	if !DashboardEnabled {
		WriteJSON(w, http.StatusNotFound, map[string]any{"error": "Dashboard disabled"})
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(dashboardHTML)
}

func getBackendHealth() map[string]any {
	result := map[string]any{}
	models := backendManager.SnapshotModels()
	for _, key := range backendManager.Keys() {
		info := backendManager.GetBackendInfo(key)
		if info == nil {
			continue
		}
		modelsMap := map[string]any{}
		for _, model := range models {
			if model.Synthetic || !containsString(model.Backends, key) {
				continue
			}
			inUse := 0
			if slotManager.HasBackend(key) {
				inUse = slotManager.Get(key).CountInUse(model.Name)
			}
			modelsMap[model.Name] = map[string]any{
				"n_ctx":           model.NCtx,
				"total_slots":     model.TotalSlots,
				"in_use":          inUse,
				"last_discovered": model.LastDiscovered,
				"last_refresh":    backendManager.GetRefreshTS(model.Name, key),
			}
		}
		result[key] = map[string]any{
			"url":         info.URL,
			"up":          backendManager.GetBackendState(key),
			"cache_dir":   info.CacheDir,
			"has_agent":   info.HasAgent,
			"queue_depth": GetDispatcher().QueueDepth(key),
			"models":      modelsMap,
		}
	}
	return result
}

func getSlotStatus() map[string]any {
	result := map[string]any{}
	for _, backendID := range slotManager.Backends() {
		result[backendID] = slotManager.Get(backendID).SlotStatus()
	}
	return result
}

func getCacheStats() map[string]any {
	result := map[string]any{}
	for _, key := range backendManager.Keys() {
		if !slotManager.HasBackend(key) {
			continue
		}
		beSm := slotManager.Get(key)
		totalBytes := beSm.GetTotalBytes()
		maxGB := backendManager.GetCacheMaxSizeGB(key)
		maxBytes := maxGB * 1024 * 1024 * 1024
		utilization := 0.0
		if maxBytes > 0 {
			utilization = float64(totalBytes) / maxBytes * 100
		}
		oldest, newest, ok := beSm.RingOldestNewest()
		var oldestVal, newestVal any
		if ok {
			oldestVal = oldest
			newestVal = newest
		}
		result[key] = map[string]any{
			"ring_size":       beSm.GetRingSize(),
			"total_bytes":     totalBytes,
			"total_gb":        roundTo(float64(totalBytes)/(1024*1024*1024), 2),
			"max_gb":          maxGB,
			"utilization_pct": roundTo(utilization, 1),
			"cache_dir":       backendManager.GetCacheDir(key),
			"oldest_entry":    oldestVal,
			"newest_entry":    newestVal,
		}
	}
	return result
}

type StreamState struct {
	Resp               *http.Response
	w                  http.ResponseWriter
	flusher            http.Flusher
	r                  *http.Request
	cancel             context.CancelFunc
	modelName          string
	backendID          string
	slotID             int
	key                string
	keyShort           string
	nTokens            int
	blocks             []string
	beSm               *BackendSlotManager
	bestRatio          float64
	hitType            *CacheHitType
	restoreKey         string
	restoreBackend     string
	t0                 float64
	requestJSON        map[string]any
	requestID          string
	routingReason      string
	backendCacheRatios map[string]float64
	summarizationScore float64
	messages           []map[string]any
	promptPreview      string
	clientIP           string

	Chunks          chan []byte
	Done            chan struct{}
	doneOnce        sync.Once
	bodyOnce        sync.Once
	cleanupOnce     sync.Once
	cancelled       bool
	StreamComplete  bool
	ssePromptTokens int
	sseCachedTokens int
	lineBuf         string
}

func (ss *StreamState) finish() {
	ss.doneOnce.Do(func() { close(ss.Done) })
}

func (ss *StreamState) closeBody() {
	if ss.Resp == nil || ss.Resp.Body == nil {
		return
	}
	ss.bodyOnce.Do(func() {
		done := make(chan struct{})
		go func() {
			ss.Resp.Body.Close()
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			logWarn("app", "Response aclose() timed out for model '%s' on backend '%s' slot %d (key %s)", ss.modelName, ss.backendID, ss.slotID, ss.keyShort)
		}
	})
}

func (ss *StreamState) ReadLoop() {
	buf := make([]byte, 32*1024)
	chunksReceived := 0
	totalBytes := 0
	for {
		n, readErr := ss.Resp.Body.Read(buf)
		if n > 0 {
			totalBytes += n
			chunk := make([]byte, n)
			copy(chunk, buf[:n])
			sseDone := false
			ss.lineBuf += string(chunk)
			for strings.Contains(ss.lineBuf, "\n") {
				idx := strings.Index(ss.lineBuf, "\n")
				line := ss.lineBuf[:idx]
				ss.lineBuf = ss.lineBuf[idx+1:]
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "data: ") {
					data := strings.TrimSpace(line[6:])
					if data == "[DONE]" {
						sseDone = true
						break
					}
					var event map[string]any
					if json.Unmarshal([]byte(data), &event) == nil {
						if usage, ok := event["usage"].(map[string]any); ok {
							pt := intFromAny(usage["prompt_tokens"])
							details, _ := usage["prompt_tokens_details"].(map[string]any)
							if details == nil {
								details = map[string]any{}
							}
							ct := intFromAny(details["cached_tokens"])
							if pt != 0 && ss.ssePromptTokens == 0 {
								ss.ssePromptTokens = pt
							}
							if ct != 0 && ss.sseCachedTokens == 0 {
								ss.sseCachedTokens = ct
								logInfo("app", "Parsed SSE usage: model '%s' slot %d, cached_tokens=%d, prompt_tokens=%d", ss.modelName, ss.slotID, ct, pt)
							}
						}
					}
				}
			}
			if sseDone {
				logInfo("app", "SSE [DONE] received for model '%s' on backend '%s' slot %d (key %s)", ss.modelName, ss.backendID, ss.slotID, ss.keyShort)
				ss.StreamComplete = true
				// Forward the [DONE] chunk itself: clients treat a stream that
				// ends without data: [DONE] as truncated even when all content
				// arrived.
				select {
				case ss.Chunks <- chunk:
					chunksReceived++
				default:
					logWarn("app", "Stream queue full while flushing SSE [DONE] for model '%s' on backend '%s' slot %d (key %s)", ss.modelName, ss.backendID, ss.slotID, ss.keyShort)
				}
				ss.finish()
				return
			}
			select {
			case ss.Chunks <- chunk:
				chunksReceived++
			default:
				logWarn("app", "Stream queue full for model '%s' on backend '%s' slot %d (key %s)", ss.modelName, ss.backendID, ss.slotID, ss.keyShort)
				ss.cancelled = true
				ss.finish()
				return
			}
		}
		if readErr != nil {
			if readErr == io.EOF {
				ss.StreamComplete = true
				logInfo("app", "Stream complete from client %s for model '%s' on backend '%s' slot %d (key %s): %d chunks, %d bytes", ss.clientIP, ss.modelName, ss.backendID, ss.slotID, ss.keyShort, chunksReceived, totalBytes)
			} else {
				logWarn("app", "Backend disconnected for model '%s' on backend '%s' slot %d (key %s): incomplete body (%d chunks, %d bytes), error=%v", ss.modelName, ss.backendID, ss.slotID, ss.keyShort, chunksReceived, totalBytes, readErr)
			}
			ss.finish()
			return
		}
	}
}

func (ss *StreamState) run() {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case chunk := <-ss.Chunks:
			if _, werr := ss.w.Write(chunk); werr != nil {
				logWarn("app", "Stream write error for model '%s' on backend '%s' slot %d (key %s): %v", ss.modelName, ss.backendID, ss.slotID, ss.keyShort, werr)
				ss.cancelled = true
				ss.closeBody()
				ss.cleanup()
				return
			}
			if ss.flusher != nil {
				ss.flusher.Flush()
			}
		case <-ss.Done:
			if !ss.cancelled {
			drain:
				for {
					select {
					case chunk := <-ss.Chunks:
						if _, werr := ss.w.Write(chunk); werr != nil {
							ss.cancelled = true
							break drain
						}
						if ss.flusher != nil {
							ss.flusher.Flush()
						}
					default:
						break drain
					}
				}
			}
			ss.cleanup()
			return
		case <-ss.r.Context().Done():
			ss.cancelled = true
			ss.closeBody()
			ss.cleanup()
			return
		case <-ticker.C:
			if ss.r.Context().Err() != nil {
				logWarn("app", "Client disconnected (heartbeat check) for model '%s' on backend '%s' slot %d (key %s)", ss.modelName, ss.backendID, ss.slotID, ss.keyShort)
				ss.cancelled = true
				ss.closeBody()
				ss.cleanup()
				return
			}
		}
	}
}

func (ss *StreamState) save() (bool, int) {
	recomputeHappened := false
	if ss.hitType != nil && ss.restoreBackend != "" {
		cachedTokens := ss.sseCachedTokens
		llmPromptTokens := ss.ssePromptTokens
		if llmPromptTokens == 0 {
			llmPromptTokens = ss.nTokens
		}
		logInfo("app", "Recompute check for model '%s' slot %d: cached_tokens=%d, request_prompt_tokens=%d (llm=%d), ratio=%.3f", ss.modelName, ss.slotID, cachedTokens, ss.nTokens, llmPromptTokens, ss.bestRatio)
		var cacheMeta map[string]any
		if ss.restoreKey != "" {
			cacheMeta = kvMeta.ReadMeta(ss.restoreKey, ss.restoreBackend)
		}
		cacheNTokens := metaNTokens(cacheMeta, llmPromptTokens)
		if isRecompute(cachedTokens, llmPromptTokens, cacheNTokens) {
			recomputeHappened = true
			logWarn("app", "Recompute detected for model '%s' on backend '%s' slot %d (key %s): cached_tokens=%d llm_prompt_tokens=%d cache_n_tokens=%d, KV cache restore was partial/useless", ss.modelName, ss.backendID, ss.slotID, ss.keyShort, cachedTokens, llmPromptTokens, cacheNTokens)
			if ss.restoreKey != "" {
				kvMeta.IncrementRecomputePenalty(ss.restoreKey, ss.restoreBackend)
			}
		}
	}

	servingBeRatio := ss.backendCacheRatios[ss.backendID]
	if ShouldSkipSaveHeuristic(ss.nTokens, backendManager.GetBackendNCtx(ss.modelName, ss.backendID), ss.messages, ss.requestJSON) {
		logInfo("app", "Skipping cache save for model '%s' on backend '%s' slot %d (key %s): heuristic skip", ss.modelName, ss.backendID, ss.slotID, ss.keyShort)
		ss.beSm.MarkSaveSkipped(ss.slotID, &SaveSkipEntry{
			Key:          ss.key,
			Model:        ss.modelName,
			Blocks:       ss.blocks,
			NTokens:      ss.nTokens,
			HitType:      ss.hitType,
			ServingRatio: servingBeRatio,
			Recompute:    recomputeHappened,
			Legacy:       false,
		})
		return false, 0
	}
	if !ShouldSaveCache(servingBeRatio, recomputeHappened) {
		logInfo("app", "Skipping cache save for model '%s' on backend '%s' slot %d (key %s): restore ratio %.3f >= threshold (no recompute, cache was useful)", ss.modelName, ss.backendID, ss.slotID, ss.keyShort, servingBeRatio)
		ss.beSm.MarkSaveSkipped(ss.slotID, &SaveSkipEntry{
			Key:          ss.key,
			Model:        ss.modelName,
			Blocks:       ss.blocks,
			NTokens:      ss.nTokens,
			HitType:      ss.hitType,
			ServingRatio: servingBeRatio,
			Recompute:    recomputeHappened,
			Legacy:       false,
		})
		return false, 0
	}
	ok, cacheSize := ss.beSm.SaveAfter(ss.modelName, ss.slotID, ss.key, ss.blocks, ss.nTokens)
	logInfo("app", "SAVE: model_name='%s', key='%s', backend='%s', model_id_in_meta='%s'", ss.modelName, key16(ss.key), ss.backendID, ss.modelName)
	return ok, cacheSize
}

func (ss *StreamState) cleanup() {
	ss.cleanupOnce.Do(func() {
		if ss.cancel != nil {
			ss.cancel()
		}
		logInfo("app", "Starting cleanup for model '%s' on backend '%s' slot %d (key %s): cancelled=%v, stream_complete=%v", ss.modelName, ss.backendID, ss.slotID, ss.keyShort, ss.cancelled, ss.StreamComplete)
		ss.closeBody()
		select {
		case <-ss.Done:
		case <-time.After(5 * time.Second):
			logWarn("app", "Reader task did not finish after cancel for model '%s' on backend '%s' slot %d (key %s)", ss.modelName, ss.backendID, ss.slotID, ss.keyShort)
		}
		ok := false
		cacheSize := 0
		if ss.StreamComplete {
			logInfo("app", "Saving cache for model '%s' on backend '%s' slot %d (key %s)", ss.modelName, ss.backendID, ss.slotID, ss.keyShort)
			ok, cacheSize = ss.save()
			logInfo("app", "Cache save completed for model '%s' on backend '%s' slot %d (key %s): %v, %d bytes", ss.modelName, ss.backendID, ss.slotID, ss.keyShort, ok, cacheSize)
		} else {
			logInfo("app", "Skipping cache save for model '%s' on backend '%s' slot %d (key %s): stream incomplete", ss.modelName, ss.backendID, ss.slotID, ss.keyShort)
			if ss.cancelled {
				logInfo("app", "Request cancelled — invalidating KV cache for model '%s' on backend '%s' slot %d (key %s)", ss.modelName, ss.backendID, ss.slotID, ss.keyShort)
			} else {
				logInfo("app", "Backend disconnected — invalidating KV cache for model '%s' on backend '%s' slot %d (key %s)", ss.modelName, ss.backendID, ss.slotID, ss.keyShort)
			}
			ss.beSm.Invalidate(ss.slotID)
		}
		ss.beSm.Release(ss.slotID)
		logInfo("app", "Released slot %d for model '%s' on backend '%s' (key %s)", ss.slotID, ss.modelName, ss.backendID, ss.keyShort)

		if ss.t0 > 0 {
			recomputeHappened := false
			if ss.hitType != nil {
				cachedTokens := ss.sseCachedTokens
				llmPromptTokens := ss.ssePromptTokens
				if llmPromptTokens == 0 {
					llmPromptTokens = ss.nTokens
				}
				var cacheMeta map[string]any
				if ss.restoreKey != "" {
					cacheMeta = kvMeta.ReadMeta(ss.restoreKey, ss.restoreBackend)
				}
				cacheNTokens := metaNTokens(cacheMeta, llmPromptTokens)
				if isRecompute(cachedTokens, llmPromptTokens, cacheNTokens) {
					recomputeHappened = true
				}
			}
			status := "backend_error"
			if ss.StreamComplete {
				status = "complete"
			} else if ss.cancelled {
				status = "cancelled"
			}
			latency := latencyMS(ss.t0)
			var restored *bool
			if ss.hitType != nil {
				if *ss.hitType == CacheHitDiskRestore && ss.restoreKey != "" {
					v := true
					restored = &v
				} else if *ss.hitType == CacheHitSkip {
					restored = nil
				} else {
					v := false
					restored = &v
				}
			} else {
				v := false
				restored = &v
			}
			nTokens := ss.nTokens
			if ss.ssePromptTokens != 0 {
				nTokens = ss.ssePromptTokens
			}
			Metrics.Record(map[string]any{
				"request_id":       ss.requestID,
				"t0":               ss.t0,
				"request_json":     ss.requestJSON,
				"model":            ss.modelName,
				"backend":          ss.backendID,
				"slot_id":          ss.slotID,
				"cache_hit":        ss.hitType != nil,
				"restored":         restored,
				"recompute":        recomputeHappened,
				"saved":            ok,
				"latency_ms":       latency,
				"n_tokens":         nTokens,
				"cached_tokens":    ss.sseCachedTokens,
				"stream":           true,
				"cache_size_bytes": cacheSize,
				"prompt_preview":   ss.promptPreview,
				"routing_reason":   ss.routingReason,
				"status":           status,
			})
			backendManager.UpdateBackendLatency(ss.backendID, latency)
			reqType := "conversation"
			if ss.summarizationScore >= 0.4 {
				reqType = "summarization"
			}
			backendManager.UpdateBackendModelLatency(ss.backendID, ss.modelName, latency, reqType)
		}
		logInfo("app", "Stream reader finished for model '%s' on backend '%s' slot %d (key %s): saved=%v", ss.modelName, ss.backendID, ss.slotID, ss.keyShort, ok)
	})
}

func serveStream(
	w http.ResponseWriter,
	r *http.Request,
	requestJSON, body map[string]any,
	beID string,
	slotID int,
	modelName string,
	key string,
	blocks []string,
	promptTokens int,
	bestRatio float64,
	hitType *CacheHitType,
	restoreKey string,
	restoreBackend string,
	backendCacheRatios map[string]float64,
	summarizationScore float64,
	requestID string,
	routingReason string,
	promptPreview string,
	oldKV []string,
	t0 float64,
	messages []map[string]any,
) {
	client := backendManager.GetClient(beID)
	beSm := slotManager.Get(beID)
	streamCtx, streamCancel := context.WithCancel(context.Background())
	defer streamCancel()

	resp, err := client.ChatCompletionsStream(streamCtx, body, slotID)
	if err != nil {
		if errors.Is(err, ErrTimeout) {
			logError("app", "Chat timeout for client %s, model '%s' on backend '%s' slot %d (key %s): %v", clientIP(r), modelName, beID, slotID, key16(key), err)
			recordChatError(requestID, modelName, beID, slotID, t0, routingReason)
			beSm.Release(slotID)
			WriteJSON(w, http.StatusGatewayTimeout, map[string]any{"error": err.Error()})
			return
		}
		if errors.Is(err, ErrConn) {
			logError("app", "Backend connection error for client %s, model '%s' on backend '%s' slot %d (key %s): %v", clientIP(r), modelName, beID, slotID, key16(key), err)
			restorePrevKV(beSm, slotID, oldKV, modelName, beID)
			beSm.Release(slotID)
			recordChatError(requestID, modelName, beID, slotID, t0, routingReason)
			WriteJSON(w, http.StatusServiceUnavailable, map[string]any{"error": "backend connection failed"})
			return
		}
		logError("app", "Chat error for client %s, model '%s' on backend '%s' slot %d (key %s): %v", clientIP(r), modelName, beID, slotID, key16(key), err)
		restorePrevKV(beSm, slotID, oldKV, modelName, beID)
		beSm.Release(slotID)
		recordChatError(requestID, modelName, beID, slotID, t0, routingReason)
		WriteJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		restorePrevKV(beSm, slotID, oldKV, modelName, beID)
		recordChatError(requestID, modelName, beID, slotID, t0, routingReason)
		beSm.Release(slotID)
		WriteJSON(w, resp.StatusCode, map[string]any{"error": string(errBody)})
		return
	}

	logInfo("app", "Stream started for model '%s' on backend '%s' slot %d (key %s): status %d", modelName, beID, slotID, key16(key), resp.StatusCode)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	var flusher http.Flusher
	if f, ok := w.(http.Flusher); ok {
		flusher = f
	}

	ss := &StreamState{
		Resp:               resp,
		w:                  w,
		flusher:            flusher,
		r:                  r,
		cancel:             streamCancel,
		modelName:          modelName,
		backendID:          beID,
		slotID:             slotID,
		key:                key,
		keyShort:           key16(key),
		nTokens:            promptTokens,
		blocks:             blocks,
		beSm:               beSm,
		bestRatio:          bestRatio,
		hitType:            hitType,
		restoreKey:         restoreKey,
		restoreBackend:     restoreBackend,
		t0:                 t0,
		requestJSON:        requestJSON,
		requestID:          requestID,
		routingReason:      routingReason,
		backendCacheRatios: backendCacheRatios,
		summarizationScore: summarizationScore,
		messages:           messages,
		promptPreview:      promptPreview,
		clientIP:           clientIP(r),
		Chunks:             make(chan []byte, streamQueueSize),
		Done:               make(chan struct{}),
	}
	go ss.ReadLoop()
	ss.run()
}
