package proxycache

import (
	"context"
	cr "crypto/rand"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"sort"
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

type candidateBackend struct {
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

func clampF(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
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

func handleTokenizeError(w http.ResponseWriter, requestID, model string, t0 float64, err error) {
	var statusErr *HTTPStatusError
	if errors.As(err, &statusErr) {
		detail := statusErr.Body
		if detail == "" {
			detail = statusErr.Error()
		}
		recordEarlyError(requestID, model, t0)
		WriteJSON(w, statusErr.StatusCode, map[string]any{"error": detail})
		return
	}
	if errors.Is(err, ErrConn) {
		recordEarlyError(requestID, model, t0)
		WriteJSON(w, http.StatusServiceUnavailable, map[string]any{"error": "backend unreachable"})
		return
	}
	recordEarlyError(requestID, model, t0)
	WriteJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
}

func acquireSlotForRequest(
	ctx context.Context,
	candidateBackends []candidateBackend,
	rebuild func() []candidateBackend,
	restoreBackend string,
	canonicalName string,
	hitType *CacheHitType,
	restoreKey string,
	backendBlocks map[string][]string,
	promptTokens int,
) (GSlot, *bool, map[string]any, []string, error) {
	slotManager.RefreshSlotCounts()
	skipRestoreDiag := map[string]any{}
	var cacheBlocks []string
	if hitType != nil && *hitType == CacheHitDiskRestore && restoreBackend != "" {
		cacheBlocks = backendBlocks[restoreBackend]
	}

	doRestoreCall := func(beSm *BackendSlotManager, slotID int, key string, blocks []string, prevBlocks []string) *bool {
		if skipEntry := beSm.FlushSaveSkipped(slotID); skipEntry != nil {
			if skipEntry.Legacy {
				beSm.SaveAfter(canonicalName, slotID, skipEntry.Key, skipEntry.Blocks, skipEntry.NTokens)
				logInfo("app", "Saved skipped cache (legacy) for model '%s' on backend '%s' slot %d before restore", canonicalName, beSm.BackendID, slotID)
			} else {
				nCtxFlush := backendManager.GetBackendNCtx(canonicalName, beSm.BackendID)
				if !ShouldSkipSaveHeuristic(skipEntry.NTokens, nCtxFlush, nil, nil) && ShouldSaveCache(skipEntry.ServingRatio, skipEntry.Recompute) {
					beSm.SaveAfter(canonicalName, slotID, skipEntry.Key, skipEntry.Blocks, skipEntry.NTokens)
					logInfo("app", "Saved skipped cache for model '%s' on backend '%s' slot %d before restore", canonicalName, beSm.BackendID, slotID)
				}
			}
		}

		var restored *bool
		if len(blocks) > 0 {
			logWarn("app", "[diag] _do_restore_call: slot %d, blocks=%d, prev_blocks=%d, key=%v", slotID, len(blocks), len(prevBlocks), key16OrNil(key))
			if beSm.ShouldSkipRestore(slotID, blocks, prevBlocks, true) {
				logInfo("app", "Skipping restore for model '%s' on backend '%s' slot %d: slot cache already matches", canonicalName, beSm.BackendID, slotID)
				skipRestoreDiag = map[string]any{
					"skipped":       true,
					"backend":       beSm.BackendID,
					"slot_id":       slotID,
					"old_kv_blocks": len(prevBlocks),
					"req_blocks":    len(blocks),
					"restore_key":   key16OrNil(key),
				}
				v := false
				restored = &v
			} else if key != "" {
				beSm.Restore(slotID, key, canonicalName, true)
				v := true
				restored = &v
			} else {
				logInfo("app", "No restore key for model '%s' on backend '%s' slot %d", canonicalName, beSm.BackendID, slotID)
				v := false
				restored = &v
			}
		} else if key != "" {
			beSm.Restore(slotID, key, canonicalName, true)
			v := true
			restored = &v
		} else {
			restored = nil
		}
		return restored
	}

	tryCacheBackend := func() (GSlot, *bool, []string, bool) {
		if restoreBackend == "" || promptTokens >= backendManager.GetBackendNCtx(canonicalName, restoreBackend) {
			return GSlot{}, nil, nil, false
		}
		beSm := slotManager.Get(restoreBackend)
		slotID := beSm.TryAcquire(canonicalName)
		if slotID == -1 {
			return GSlot{}, nil, nil, false
		}
		prevKV := beSm.GetKVState(slotID)
		logWarn("app", "[diag] _try_cache_backend: slot %d, old_kv blocks=%d, cache_blocks=%d", slotID, len(prevKV), len(cacheBlocks))
		acquireBlocks := backendBlocks[restoreBackend]
		if len(acquireBlocks) == 0 {
			acquireBlocks = []string{}
		}
		beSm.SetKVState(slotID, acquireBlocks)
		restored := doRestoreCall(beSm, slotID, restoreKey, cacheBlocks, prevKV)
		backendManager.TouchBackend(restoreBackend)
		return GSlot{ModelName: canonicalName, BackendID: restoreBackend, SlotID: slotID}, restored, prevKV, true
	}

	if restoreBackend != "" && hitType != nil && promptTokens < backendManager.GetBackendNCtx(canonicalName, restoreBackend) {
		if g, restored, prevKV, ok := tryCacheBackend(); ok {
			return g, restored, skipRestoreDiag, prevKV, nil
		}
		pending := slotManager.GetCacheWaitPending(restoreBackend)
		if pending < CacheHitWaitMaxPending {
			ema := slotManager.Get(restoreBackend).GetSlotDurationEMA()
			waitTimeout := clampF(ema, CacheHitWaitEMAMinT, CacheHitWaitEMAMaxT)
			logInfo("app", "Cache backend '%s' busy for model '%s', polling up to %.1fs", restoreBackend, canonicalName, waitTimeout)
			decPending := func() {
				cur := slotManager.GetCacheWaitPending(restoreBackend)
				slotManager.SetCacheWaitPending(restoreBackend, cur-1)
			}
			slotManager.SetCacheWaitPending(restoreBackend, pending+1)
			elapsed := 0.0
			for elapsed < waitTimeout {
				if err := ctxSleep(ctx, math.Min(5.0, waitTimeout-elapsed)); err != nil {
					decPending()
					return GSlot{}, nil, skipRestoreDiag, nil, err
				}
				elapsed += 5.0
				if g, restored, prevKV, ok := tryCacheBackend(); ok {
					decPending()
					return g, restored, skipRestoreDiag, prevKV, nil
				}
			}
			decPending()
		}
	}

	// Retry forever until a slot frees (aborts only on context cancellation)
	attempt := 0
	for {
		if restoreBackend != "" && hitType != nil && promptTokens < backendManager.GetBackendNCtx(canonicalName, restoreBackend) {
			if g, restored, prevKV, ok := tryCacheBackend(); ok {
				return g, restored, skipRestoreDiag, prevKV, nil
			}
		}
		for _, cb := range candidateBackends {
			if cb.ModelName == "" {
				continue
			}
			if promptTokens >= backendManager.GetBackendNCtx(cb.ModelName, cb.BackendID) {
				continue
			}
			beSm := slotManager.Get(cb.BackendID)
			slotID := beSm.TryAcquire(cb.ModelName)
			if slotID == -1 {
				continue
			}
			fbBlocks := backendBlocks[cb.BackendID]
			prevKV := beSm.GetKVState(slotID)
			logWarn("app", "[diag] fallback: slot %d, old_kv=%d, fb_blocks=%d", slotID, len(prevKV), len(fbBlocks))
			if len(fbBlocks) == 0 {
				fbBlocks = []string{}
			}
			beSm.SetKVState(slotID, fbBlocks)
			restored := doRestoreCall(beSm, slotID, "", fbBlocks, prevKV)
			backendManager.TouchBackend(cb.BackendID)
			return GSlot{ModelName: cb.ModelName, BackendID: cb.BackendID, SlotID: slotID}, restored, skipRestoreDiag, prevKV, nil
		}
		attempt++
		backoff := float64(attempt) * SlotAcquireRetryBaseSeconds
		logInfo("app", "No slots available across all backends, retrying in %ds (attempt %d)", int(backoff), attempt)
		if err := ctxSleep(ctx, backoff); err != nil {
			return GSlot{}, nil, skipRestoreDiag, nil, err
		}
		// Re-derive candidates so backends/models that came online during
		// the wait (liveness loop discovery) are considered, and backends
		// that went offline drop out.
		candidateBackends = rebuild()
	}
}

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

	options := backendManager.GetDiscoveredModels(clientModel)
	if len(options) == 0 {
		_ = backendManager.DiscoverModels()
		options = backendManager.GetDiscoveredModels(clientModel)
	}
	if len(options) == 0 {
		recordEarlyError(requestID, clientModel, t0)
		WriteJSON(w, http.StatusBadRequest, map[string]any{"error": fmt.Sprintf("model '%s' not found", clientModel)})
		return
	}

	firstOpt := options[0]
	if len(firstOpt.Backends) == 0 {
		recordEarlyError(requestID, clientModel, t0)
		WriteJSON(w, http.StatusBadGateway, map[string]any{"error": "backend unavailable"})
		return
	}
	firstBeID := firstOpt.Backends[0]
	firstClient := backendManager.GetClient(firstBeID)
	templated, err := firstClient.ApplyChatTemplate(r.Context(), messages)
	if err != nil {
		handleTokenizeError(w, requestID, clientModel, t0, err)
		return
	}
	firstTokenIDs, err := firstClient.Tokenize(r.Context(), templated, true)
	if err != nil {
		handleTokenizeError(w, requestID, clientModel, t0, err)
		return
	}

	promptTokens := len(firstTokenIDs)
	minCtx := options[0].NCtx
	for _, opt := range options[1:] {
		if opt.NCtx < minCtx {
			minCtx = opt.NCtx
		}
	}
	if promptTokens >= minCtx {
		recordEarlyError(requestID, clientModel, t0)
		WriteJSON(w, http.StatusBadRequest, map[string]any{
			"error": fmt.Sprintf("prompt too long (tokens=%d, n_ctx=%d)", promptTokens, minCtx),
		})
		return
	}
	canonicalName := firstOpt.Name

	restoreKey := ""
	restoreBackend := ""
	bestRatio := 0.0
	var hitType *CacheHitType

	scanDiagnostics := []map[string]any{}
	backendCacheRatios := map[string]float64{}
	backendTokenIDs := map[string][]int{}
	backendBlocks := map[string][]string{}

	for _, opt := range options {
		for _, beID := range opt.Backends {
			optClient := backendManager.GetClient(beID)
			var optTokenIDs []int
			optTemplated, tErr := optClient.ApplyChatTemplate(r.Context(), messages)
			if tErr == nil {
				optTokenIDs, tErr = optClient.Tokenize(r.Context(), optTemplated, true)
			}
			if tErr != nil {
				if errors.Is(tErr, ErrConn) {
					logWarn("app", "Backend %s error, skipping for model '%s' from client %s", beID, opt.Name, ip)
					scanDiagnostics = append(scanDiagnostics, map[string]any{
						"model":   opt.Name,
						"backend": beID,
						"status":  "unreachable",
					})
					continue
				}
				logError("app", "Backend %s scan error for model '%s' from client %s: %v", beID, opt.Name, ip, tErr)
				recordEarlyError(requestID, clientModel, t0)
				WriteJSON(w, http.StatusInternalServerError, map[string]any{"error": tErr.Error()})
				return
			}
			optBlocks := BlockHashesFromTokens(optTokenIDs, WordsPerBlock)
			backendTokenIDs[beID] = optTokenIDs
			backendBlocks[beID] = optBlocks
			diagEntry := map[string]any{
				"model":    opt.Name,
				"backend":  beID,
				"n_blocks": len(optBlocks),
				"n_tokens": len(optTokenIDs),
			}

			if backendManager.CacheEnabled(beID) {
				if candKey, candRatio, found := kvMeta.FindBestRestoreCandidate(optBlocks, WordsPerBlock, LCPTh, opt.Name, beID); found {
					diagEntry["cache_file_key"] = key16(candKey)
					diagEntry["cache_file_ratio"] = round4(candRatio)
					if candRatio > bestRatio {
						bestRatio = candRatio
						restoreKey = candKey
						restoreBackend = beID
						canonicalName = opt.Name
						dt := CacheHitDiskRestore
						hitType = &dt
						logInfo("app", "Cache hit: key '%s' (model '%s', backend '%s', ratio %.3f) — replacing previous best",
							key16(restoreKey), canonicalName, restoreBackend, bestRatio)
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
					if ratio >= LCPTh && ratio >= bestRatio {
						bestRatio = ratio
						restoreKey = ""
						restoreBackend = beID
						canonicalName = opt.Name
						dt := CacheHitSkip
						hitType = &dt
						logInfo("app", "Pending slot cache hit: model '%s', backend '%s', slot %d, ratio %.3f",
							canonicalName, restoreBackend, slotID, ratio)
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
				backendCacheRatios[beID] = beBest
			}
		}
	}

	logInfo("app", "Chat request from %s: model '%s', %d tokens, restore key=%v on backend %s",
		ip, clientModel, promptTokens, key16OrNil(restoreKey), restoreBackend)

	reqType := "conversation"
	if reqClass.Score >= 0.4 {
		reqType = "summarization"
	}

	buildCandidateBackends := func(opts []*DiscoveredModel) []candidateBackend {
		cbs := []candidateBackend{}
		for _, opt := range opts {
			for _, beID := range opt.Backends {
				if restoreBackend != "" && beID == restoreBackend {
					continue
				}
				cbs = append(cbs, candidateBackend{BackendID: beID, ModelName: opt.Name})
			}
		}
		ringSizeOf := func(beID string) int {
			if slotManager.HasBackend(beID) {
				return slotManager.Get(beID).GetRingSize()
			}
			return 0
		}
		sort.SliceStable(cbs, func(i, j int) bool {
			a, b := cbs[i], cbs[j]
			ar, br := backendCacheRatios[a.BackendID], backendCacheRatios[b.BackendID]
			if ar != br {
				return ar < br
			}
			au, bu := backendManager.GetBackendLastUsed(a.BackendID), backendManager.GetBackendLastUsed(b.BackendID)
			if au != bu {
				return au < bu
			}
			ari, bri := ringSizeOf(a.BackendID), ringSizeOf(b.BackendID)
			if ari != bri {
				return ari < bri
			}
			return backendManager.GetBackendModelLatencyEMA(a.BackendID, a.ModelName, reqType) <
				backendManager.GetBackendModelLatencyEMA(b.BackendID, b.ModelName, reqType)
		})
		return cbs
	}
	// Re-resolves models from the backend manager on each retry so the
	// fallback set reflects backends that came online (or offline) while
	// this request waits for a slot.
	rebuildCandidates := func() []candidateBackend {
		return buildCandidateBackends(backendManager.GetDiscoveredModels(clientModel))
	}
	candidateBackends := buildCandidateBackends(options)

	g, restored, skipRestoreDiag, prevKV, acqErr := acquireSlotForRequest(
		r.Context(), candidateBackends, rebuildCandidates, restoreBackend, canonicalName, hitType, restoreKey, backendBlocks, promptTokens,
	)
	if acqErr != nil {
		logError("app", "Could not acquire slot from client %s for model '%s': %v", ip, clientModel, acqErr)
		recordEarlyError(requestID, clientModel, t0)
		WriteJSON(w, http.StatusServiceUnavailable, map[string]any{"error": "all slots busy, please retry later"})
		return
	}

	modelName := g.ModelName
	beID := g.BackendID
	slotID := g.SlotID
	client := backendManager.GetClient(beID)
	beSm := slotManager.Get(beID)

	servingTokenIDs := backendTokenIDs[beID]
	if len(servingTokenIDs) == 0 {
		servingTokenIDs = firstTokenIDs
	}
	key := MetaKey(modelName, servingTokenIDs)
	blocks := BlockHashesFromTokens(servingTokenIDs, WordsPerBlock)

	var routingReason string
	switch {
	case hitType != nil && *hitType == CacheHitSkip && beID == restoreBackend:
		routingReason = "pending_slot_hit"
	case hitType != nil && *hitType == CacheHitDiskRestore && beID == restoreBackend:
		routingReason = "cache_hit"
	case hitType == nil:
		routingReason = "no_cache_entry"
	default:
		routingReason = "cache_backend_unavailable"
	}

	if restoreKey == "" && restoreBackend == "" {
		logInfo("app", "No cache hit: using key '%s' for model '%s' (client model '%s')", key16(key), modelName, clientModel)
	}
	logInfo("app", "Slot acquired: model '%s' on backend '%s' slot %d, restored=%v, save_key='%s', canonical_name='%s'",
		modelName, beID, slotID, restored, key16(key), canonicalName)

	candidateIDs := make([]string, 0, len(candidateBackends))
	for _, cb := range candidateBackends {
		candidateIDs = append(candidateIDs, cb.BackendID)
	}
	Metrics.Record(map[string]any{
		"request_id":     requestID,
		"model":          modelName,
		"backend":        beID,
		"slot_id":        slotID,
		"routing_reason": routingReason,
		"cache_hit":      hitType != nil,
		"restored":       restored,
		"status":         "incomplete",
		"routing_diagnostics": map[string]any{
			"best_ratio":           round4(bestRatio),
			"restore_key":          key16OrNil(restoreKey),
			"restore_backend":      strOrNone(restoreBackend),
			"restore_info_backend": strOrNone(restoreBackend),
			"candidate_backends":   candidateIDs,
			"scan":                 scanDiagnostics,
			"skip_restore":         skipRestoreDiag,
		},
	})

	body := make(map[string]any, len(requestJSON)+3)
	for k, v := range requestJSON {
		body[k] = v
	}
	body["model"] = canonicalName
	opts := map[string]any{}
	if existingOpts, okOpts := requestJSON["options"].(map[string]any); okOpts {
		for k, v := range existingOpts {
			opts[k] = v
		}
	}
	opts["slot_id"] = slotID
	opts["id_slot"] = slotID
	opts["n_keep"] = -1
	opts["cache_prompt"] = true
	body["options"] = opts
	body["n_keep"] = -1
	body["cache_prompt"] = true

	logInfo("app", "Dispatching request from client %s: model '%s' on backend '%s' slot %d, restore=%v, restored=%v",
		ip, modelName, beID, slotID, key16OrNil(restoreKey), restored)

	if stream {
		serveStream(w, r, requestJSON, body, beID, slotID, modelName, key, blocks, promptTokens,
			bestRatio, hitType, restoreKey, restoreBackend, backendCacheRatios, reqClass.Score,
			requestID, routingReason, promptPreview, prevKV, t0, messages)
		return
	}

	released := false
	defer func() {
		if !released {
			beSm.Release(slotID)
		}
	}()

	out, err := client.ChatCompletions(r.Context(), body, slotID)
	if err != nil {
		if errors.Is(err, ErrTimeout) {
			logError("app", "Chat timeout for client %s, model '%s' on backend '%s' slot %d (key %s): %v",
				ip, modelName, beID, slotID, key16(key), err)
			recordChatError(requestID, modelName, beID, slotID, t0, routingReason)
			WriteJSON(w, http.StatusGatewayTimeout, map[string]any{"error": err.Error()})
			return
		}
		if errors.Is(err, ErrConn) {
			logError("app", "Backend connection error for client %s, model '%s' on backend '%s' slot %d (key %s): %v",
				ip, modelName, beID, slotID, key16(key), err)
			restorePrevKV(beSm, slotID, prevKV, modelName, beID)
			beSm.Release(slotID)
			released = true
			recordChatError(requestID, modelName, beID, slotID, t0, routingReason)
			WriteJSON(w, http.StatusServiceUnavailable, map[string]any{"error": "backend connection failed"})
			return
		}
		logError("app", "Chat error for client %s, model '%s' on backend '%s' slot %d (key %s): %v",
			ip, modelName, beID, slotID, key16(key), err)
		restorePrevKV(beSm, slotID, prevKV, modelName, beID)
		recordChatError(requestID, modelName, beID, slotID, t0, routingReason)
		WriteJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}
	if out == nil {
		restorePrevKV(beSm, slotID, prevKV, modelName, beID)
		recordChatError(requestID, modelName, beID, slotID, t0, routingReason)
		WriteJSON(w, http.StatusBadGateway, map[string]any{"error": "provider non-JSON body"})
		return
	}

	saveOK := false
	cacheSize := 0
	recomputeHappened := false
	llmPromptTokens := 0
	cachedTokens := 0
	if hitType != nil {
		usage, _ := out["usage"].(map[string]any)
		if usage == nil {
			usage = map[string]any{}
		}
		ptDetails, _ := usage["prompt_tokens_details"].(map[string]any)
		if ptDetails == nil {
			ptDetails = map[string]any{}
		}
		cachedTokens = intFromAny(ptDetails["cached_tokens"])
		llmPromptTokens = intFromAny(usage["prompt_tokens"])
		logInfo("app", "Recompute check for model '%s' slot %d: cached_tokens=%d, request_prompt_tokens=%d (llm=%d), ratio=%.3f",
			modelName, slotID, cachedTokens, promptTokens, llmPromptTokens, bestRatio)
		var cacheMeta map[string]any
		if restoreKey != "" {
			cacheMeta = kvMeta.ReadMeta(restoreKey, restoreBackend)
		}
		cacheNTokens := metaNTokens(cacheMeta, llmPromptTokens)
		if isRecompute(cachedTokens, llmPromptTokens, cacheNTokens) {
			recomputeHappened = true
			logWarn("app", "Recompute detected for model '%s' on backend '%s' slot %d (key %s): cached_tokens=%d llm_prompt_tokens=%d cache_n_tokens=%d, KV cache restore was partial/useless",
				modelName, beID, slotID, key16(key), cachedTokens, llmPromptTokens, cacheNTokens)
			if restoreKey != "" && restoreBackend != "" {
				kvMeta.IncrementRecomputePenalty(restoreKey, restoreBackend)
			}
		}
	}

	servingBeRatio := backendCacheRatios[beID]
	skipEntry := &SaveSkipEntry{
		Key:          key,
		Blocks:       blocks,
		NTokens:      promptTokens,
		HitType:      hitType,
		ServingRatio: servingBeRatio,
		Recompute:    recomputeHappened,
		Legacy:       false,
	}
	if ShouldSkipSaveHeuristic(promptTokens, backendManager.GetBackendNCtx(modelName, beID), messages, requestJSON) {
		beSm.MarkSaveSkipped(slotID, skipEntry)
	} else if ShouldSaveCache(servingBeRatio, recomputeHappened) {
		saveOK, cacheSize = beSm.SaveAfter(modelName, slotID, key, blocks, promptTokens)
	} else {
		beSm.MarkSaveSkipped(slotID, skipEntry)
	}

	latency := latencyMS(t0)
	cacheSizeBytes := 0
	if saveOK {
		cacheSizeBytes = cacheSize
	}
	Metrics.Record(map[string]any{
		"request_id":       requestID,
		"t0":               t0,
		"request_json":     requestJSON,
		"model":            modelName,
		"backend":          beID,
		"slot_id":          slotID,
		"cache_hit":        hitType != nil,
		"restored":         restored,
		"recompute":        recomputeHappened,
		"saved":            saveOK,
		"latency_ms":       latency,
		"n_tokens":         llmPromptTokens,
		"cached_tokens":    cachedTokens,
		"stream":           false,
		"cache_size_bytes": cacheSizeBytes,
		"prompt_preview":   promptPreview,
		"routing_reason":   routingReason,
		"status":           "complete",
	})
	backendManager.UpdateBackendLatency(beID, latency)
	backendManager.UpdateBackendModelLatency(beID, modelName, latency, reqType)
	WriteJSON(w, http.StatusOK, out)
}

func metricsDashboardHandler(w http.ResponseWriter, r *http.Request) {
	summary := Metrics.GetSummary()
	summary["backends"] = getBackendHealth()
	summary["slots"] = getSlotStatus()
	summary["cache"] = getCacheStats()
	summary["backend_model_performance"] = backendManager.GetAllLatencyEMA()
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
			"url":       info.URL,
			"up":        backendManager.GetBackendState(key),
			"cache_dir": info.CacheDir,
			"has_agent": info.HasAgent,
			"models":    modelsMap,
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
