// processor.go — per-backend request processing: cache restore (including
// files that arrived via P2P transfer), dispatch to llama.cpp, and
// post-response cache save.
//
// Runs on a worker goroutine spawned by the dispatcher's pump, one in flight
// per free slot on the backend, unlike the old design where every HTTP handler
// drove its own slot acquisition across all backends.

package proxycache

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"
)

// acquireSlotOnBackend acquires a free slot for the model on this backend,
// retrying with growing backoff until one frees or the client disconnects.
// If the model is no longer loaded on this backend (empty pool and not listed
// in the registry), the request is handed back to the global overflow queue
// for re-matching against the current backend set.
// Returns (slotID, prevKV, requeued); slotID == -1 when the request was
// discarded or requeued (requeued=true only in the latter case).
func acquireSlotOnBackend(d *RequestDispatcher, req *ProcRequest, beSm *BackendSlotManager, modelName string) (int, []string, bool) {
	attempt := 0
	for {
		slotID := beSm.TryAcquire(modelName)
		if slotID != -1 {
			prevKV := beSm.GetKVState(slotID)
			return slotID, prevKV, false
		}
		if err := req.ctx.Err(); err != nil {
			d.discard(req, "client_disconnected")
			return -1, nil, false
		}
		pool := beSm.GetPool(modelName)
		if pool == nil || len(pool) == 0 {
			// Model not loaded here: refresh the registry once. If this
			// backend no longer serves the model, let the matcher re-route.
			slotManager.RefreshSlotCounts()
			if err := req.ctx.Err(); err != nil {
				d.discard(req, "client_disconnected")
				return -1, nil, false
			}
			if !backendManager.ServesModel(modelName, beSm.BackendID) {
				logInfo("processor", "Backend '%s' no longer serves model '%s'; requeueing request %s for re-matching",
					beSm.BackendID, modelName, req.requestID)
				d.Requeue(req)
				return -1, nil, true
			}
			// The refresh may have created the pool: try again right away
			// instead of falling into the backoff.
			if slotID := beSm.TryAcquire(modelName); slotID != -1 {
				prevKV := beSm.GetKVState(slotID)
				return slotID, prevKV, false
			}
		}
		attempt++
		backoff := float64(attempt) * SlotAcquireRetryBaseSeconds
		logInfo("processor", "No free slot for model '%s' on backend '%s', retrying in %.1fs (attempt %d)",
			modelName, beSm.BackendID, backoff, attempt)
		if err := ctxSleep(req.ctx, backoff); err != nil {
			d.discard(req, "client_disconnected")
			return -1, nil, false
		}
	}
}

// resolveRestoreKey decides whether (and from which key) this backend can
// restore a cache for the request. It uses the best disk cache candidate:
// directly when it lives here, or after a bounded wait for the P2P transfer
// of that key when the request was routed elsewhere.
//
// canSkip reports whether the slot actually acquired still holds KV state
// matching the request (the matcher's pending-slot hit is only a snapshot —
// the warm slot's state may have changed, or a different free slot may have
// been picked). When it is false and the primary hit was a pending-slot one,
// the local disk cache is used as a fallback instead of recomputing.
func resolveRestoreKey(req *ProcRequest, dec *RouteDecision, beID string, canSkip bool) string {
	if dec.diskRestoreKey == "" {
		return ""
	}
	if beID == dec.diskRestoreBackend {
		if !canSkip && dec.hitType != nil && *dec.hitType == CacheHitSkip && dec.restoreBackend == beID {
			logInfo("processor", "Pending-slot hit stale for request %s on backend '%s'; falling back to disk cache",
				req.requestID, beID)
		} else if canSkip {
			return "" // slot is warm: no restore needed
		}
		return dec.diskRestoreKey
	}
	if TransferInFlight(dec.diskRestoreKey, beID) {
		waitForTransferComplete(req.ctx, dec.diskRestoreKey, beID,
			transferWaitFor(int64(backendManager.CacheGetSize(dec.diskRestoreBackend, dec.diskRestoreKey))))
	}
	if backendManager.CacheExists(beID, dec.diskRestoreKey) {
		logInfo("processor", "P2P cache ready on fallback backend '%s' for key %s (request %s)",
			beID, key16(dec.diskRestoreKey), req.requestID)
		return dec.diskRestoreKey
	}
	return ""
}

// transferWaitFor computes the worker's bounded wait for an in-flight P2P
// transfer: at least CacheTransferWait, scaled up by file size assuming
// CacheTransferAssumedMBps so multi-GB entries can actually land before the
// request falls back to recompute. The wait loop exits as soon as the
// transfer finishes, so the scale-up never delays a fast transfer.
func transferWaitFor(sizeBytes int64) time.Duration {
	wait := time.Duration(CacheTransferWait * float64(time.Second))
	if sizeBytes > 0 && CacheTransferAssumedMBps > 0 {
		est := time.Duration(float64(sizeBytes) / (1024*1024*CacheTransferAssumedMBps) * float64(time.Second))
		if est > wait {
			wait = est
		}
	}
	return wait
}

// waitForTransferComplete blocks until the transfer of key to beID finishes
// or wait elapses (or the client disconnects).
func waitForTransferComplete(ctx context.Context, key, beID string, wait time.Duration) {
	deadline := time.Now().Add(wait)
	logInfo("processor", "Waiting up to %.1fs for P2P transfer of key %s to backend '%s'",
		wait.Seconds(), key16(key), beID)
	for TransferInFlight(key, beID) {
		if time.Now().After(deadline) || ctx.Err() != nil {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// flushSkippedSave persists a previously-skipped save when this request is
// about to clobber the slot's content — but only when the skip's original
// rationale no longer holds. A ratio-based skip was safe because an on-disk
// ancestor already covered the content well enough, so saving would have
// grown the ring (eviction risk) for a tail that is cheap to recompute. If
// that coverage has since degraded (e.g. the ancestor was evicted), a
// re-evaluation would say "save" — and clobbering without saving would make
// the loss permanent, so flush then. Explicit content-class skips
// (near-max-context / summarization heuristics) are respected either way, as
// is the case where a copy of the key already exists on this backend.
func flushSkippedSave(beSm *BackendSlotManager, slotID int, blocks []string, canSkip bool, modelName, beID string) {
	entry := beSm.FlushSaveSkipped(slotID)
	if entry == nil {
		return
	}
	save := false
	switch {
	case entry.Legacy:
		save = true
	case !canSkip && !backendManager.CacheExists(beID, entry.Key):
		nCtxFlush := backendManager.GetBackendNCtx(modelName, beID)
		if !ShouldSkipSaveHeuristic(entry.NTokens, nCtxFlush, nil, nil) {
			// Re-evaluate against CURRENT disk coverage: flush only when the
			// content is no longer covered well enough on disk, i.e. when
			// ShouldSaveCache would now return true.
			_, covRatio, covered := kvMeta.FindBestRestoreCandidate(entry.Blocks, WordsPerBlock, LCPTh, entry.Model, beID)
			save = !covered || covRatio <= CacheSaveRatioThreshold
		}
	}
	if !save {
		return
	}
	beSm.SaveAfter(modelName, slotID, entry.Key, entry.Blocks, entry.NTokens)
	// SaveAfter re-tracks the saved (older) blocks; restore this request's.
	beSm.SetKVState(slotID, blocks)
	logInfo("processor", "Flushed skipped cache %s for model '%s' on backend '%s' slot %d before clobber (disk coverage degraded)",
		key16(entry.Key), entry.Model, beID, slotID)
}

// doWorkerRestore performs the pre-dispatch restore for an acquired slot:
// apply the skip-restore heuristic and issue the restore call. Returns
// (restored, skipRestoreDiag). Skipped-save flushing happens earlier, in
// processWorkerRequest, where the clobber decision (canSkip) is known.
func doWorkerRestore(beSm *BackendSlotManager, slotID int, restoreKey string, blocks, prevBlocks []string, modelName, backendID string) (*bool, map[string]any) {
	skipRestoreDiag := map[string]any{}

	var restored *bool
	if len(blocks) > 0 {
		if beSm.ShouldSkipRestore(slotID, blocks, prevBlocks, true) {
			logInfo("processor", "Skipping restore for model '%s' on backend '%s' slot %d: slot cache already matches", modelName, backendID, slotID)
			skipRestoreDiag = map[string]any{
				"skipped":       true,
				"backend":       backendID,
				"slot_id":       slotID,
				"old_kv_blocks": len(prevBlocks),
				"req_blocks":    len(blocks),
				"restore_key":   key16OrNil(restoreKey),
			}
			v := false
			restored = &v
		} else if restoreKey != "" {
			beSm.Restore(slotID, restoreKey, modelName, true)
			v := true
			restored = &v
		} else {
			logInfo("processor", "No restore key for model '%s' on backend '%s' slot %d", modelName, backendID, slotID)
			v := false
			restored = &v
		}
	} else if restoreKey != "" {
		beSm.Restore(slotID, restoreKey, modelName, true)
		v := true
		restored = &v
	}
	return restored, skipRestoreDiag
}

// buildChatBody pins the slot into the request body (root, options dict, and
// query params via withSlotID downstream) and forces prompt caching.
func buildChatBody(requestJSON map[string]any, modelName string, slotID int) map[string]any {
	body := make(map[string]any, len(requestJSON)+3)
	for k, v := range requestJSON {
		body[k] = v
	}
	body["model"] = modelName
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
	return body
}

// processWorkerRequest handles one queued request end to end on its assigned
// backend: restore, dispatch (stream or non-stream), save, release. The
// dispatcher's pump pre-acquires a slot and passes it in (slotID >= 0); when
// slotID is negative this worker performs slot acquisition itself — the lazy
// discovery path for a model whose pool does not exist on this backend yet
// (refresh + ServesModel requeue gate + backoff retry). Always finishes the
// request (closes req.done) before returning.
func processWorkerRequest(d *RequestDispatcher, beID string, req *ProcRequest, slotID int) {
	// A requeued request is still alive on the global overflow queue; only
	// finish it when this worker owns its terminal state.
	requeued := false
	defer func() {
		if !requeued {
			req.finish()
		}
	}()
	dec := req.route
	modelName := dec.selected.ModelName

	beSm := slotManager.Get(beID)
	client := backendManager.GetClient(beID)

	if dec.promptTokens >= backendManager.GetBackendNCtx(modelName, beID) {
		if slotID >= 0 {
			beSm.Release(slotID) // pump pre-acquired it; the check below never did
		}
		recordChatError(req.requestID, modelName, beID, -1, req.t0, "prompt_too_long")
		WriteJSON(req.w, http.StatusBadRequest, map[string]any{
			"error": fmt.Sprintf("prompt too long for backend (tokens=%d)", dec.promptTokens),
		})
		return
	}

	var prevKV []string
	if slotID < 0 {
		var wasRequeued bool
		slotID, prevKV, wasRequeued = acquireSlotOnBackend(d, req, beSm, modelName)
		requeued = wasRequeued
		if slotID == -1 {
			return // discarded or requeued by acquireSlotOnBackend
		}
	} else {
		prevKV = beSm.GetKVState(slotID)
	}

	blocks := dec.backendBlocks[beID]
	if len(blocks) == 0 {
		blocks = []string{}
	}
	// Re-validate the matcher's pending-slot hit against the slot we actually
	// got: its KV state may have changed since the scan, or a different free
	// slot was picked. prevKV is the state captured before SetKVState.
	canSkip := beSm.ShouldSkipRestore(slotID, blocks, prevKV, true)
	beSm.SetKVState(slotID, blocks)

	// If this slot holds a previously-skipped save for content we are about
	// to clobber, persist it now — before the restore/prefill below destroys
	// the physical KV cache.
	flushSkippedSave(beSm, slotID, blocks, canSkip, modelName, beID)

	restoreKey := resolveRestoreKey(req, dec, beID, canSkip)
	restored, skipRestoreDiag := doWorkerRestore(beSm, slotID, restoreKey, blocks, prevKV, modelName, beID)
	backendManager.TouchBackend(beID)

	// For queue-migrated requests with a disk cache hit: did usable cache end
	// up on the target? True when the target restored from the (transferred,
	// or locally found at migration) disk key or skipped restore on an
	// already-warm slot; false when the P2P transfer lost the bounded wait
	// and the request recomputed. nil = not applicable (not migrated, or no
	// disk hit by processing time).
	cacheMigrated := any(nil)
	if req.migratedFrom != "" && dec.diskRestoreKey != "" {
		cacheMigrated = restoreKey != "" || skipRestoreDiag["skipped"] == true
	}

	servingTokenIDs := dec.backendTokenIDs[beID]
	if len(servingTokenIDs) == 0 {
		servingTokenIDs = dec.firstTokenIDs
	}
	key := MetaKey(modelName, servingTokenIDs)
	reqBlocks := BlockHashesFromTokens(servingTokenIDs, WordsPerBlock)

	// Effective restore coordinates for recompute detection / save decisions:
	// the scanned key on the hit backend, the transferred key when a P2P
	// restore actually happened on a fallback, or nothing.
	effRestoreKey := ""
	effRestoreBackend := ""
	if dec.hitType != nil {
		if beID == dec.restoreBackend {
			effRestoreKey = dec.restoreKey
			effRestoreBackend = dec.restoreBackend
		} else if restoreKey != "" {
			effRestoreKey = restoreKey
			effRestoreBackend = beID
		}
	}

	var routingReason string
	switch {
	case dec.hitType != nil && *dec.hitType == CacheHitSkip && beID == dec.restoreBackend:
		routingReason = "pending_slot_hit"
	case dec.hitType != nil && *dec.hitType == CacheHitDiskRestore && beID == dec.restoreBackend:
		routingReason = "cache_hit"
	case dec.hitType == nil:
		routingReason = "no_cache_entry"
	default:
		routingReason = "cache_backend_unavailable"
	}

	logInfo("processor", "Slot acquired: model '%s' on backend '%s' slot %d, restored=%v, save_key='%s', request %s",
		modelName, beID, slotID, restored, key16(key), req.requestID)

	candidateIDs := make([]string, 0, len(dec.candidates))
	for _, cb := range dec.candidates {
		if cb.BackendID == dec.restoreBackend {
			continue
		}
		candidateIDs = append(candidateIDs, cb.BackendID)
	}
	Metrics.Record(map[string]any{
		"request_id":     req.requestID,
		"model":          modelName,
		"backend":        beID,
		"slot_id":        slotID,
		"routing_reason": routingReason,
		"cache_hit":      dec.hitType != nil,
		"restored":       restored,
		"status":         "incomplete",
		"routing_diagnostics": map[string]any{
			"best_ratio":           round4(dec.bestRatio),
			"restore_key":          key16OrNil(dec.restoreKey),
			"restore_backend":      strOrNone(dec.restoreBackend),
			"restore_info_backend": strOrNone(dec.restoreBackend),
			"candidate_backends":   candidateIDs,
			"scan":                 dec.scanDiagnostics,
			"skip_restore":         skipRestoreDiag,
			"p2p_transfer":         dec.p2pTriggered,
			"migrated_from":        strOrNone(req.migratedFrom),
			"cache_migrated":       cacheMigrated,
		},
	})

	body := buildChatBody(req.requestJSON, modelName, slotID)

	if req.stream {
		serveStream(req.w, req.r, req.requestJSON, body, beID, slotID, modelName, key, reqBlocks,
			dec.promptTokens, dec.bestRatio, dec.hitType, effRestoreKey, effRestoreBackend,
			dec.backendCacheRatios, req.reqClass.Score, req.requestID, routingReason,
			req.promptPreview, prevKV, req.t0, req.messages)
		return
	}

	processNonStream(req, dec, client, beSm, body, slotID, modelName, key, reqBlocks, restored, routingReason, effRestoreKey, effRestoreBackend)
}

// processNonStream runs the non-streaming path: chat call, error handling,
// recompute detection, save heuristics, completion metrics, response write.
func processNonStream(
	req *ProcRequest,
	dec *RouteDecision,
	client *LlamaClient,
	beSm *BackendSlotManager,
	body map[string]any,
	slotID int,
	modelName string,
	key string,
	blocks []string,
	restored *bool,
	routingReason string,
	effRestoreKey string,
	effRestoreBackend string,
) {
	beID := beSm.BackendID
	prevKV := beSm.GetKVState(slotID)
	released := false
	defer func() {
		if !released {
			beSm.Release(slotID)
		}
	}()

	out, err := client.ChatCompletions(req.ctx, body, slotID)
	if err != nil {
		if errors.Is(err, ErrTimeout) {
			logError("processor", "Chat timeout for client %s, model '%s' on backend '%s' slot %d (key %s): %v",
				req.ip, modelName, beID, slotID, key16(key), err)
			recordChatError(req.requestID, modelName, beID, slotID, req.t0, routingReason)
			WriteJSON(req.w, http.StatusGatewayTimeout, map[string]any{"error": err.Error()})
			return
		}
		if errors.Is(err, ErrConn) {
			logError("processor", "Backend connection error for client %s, model '%s' on backend '%s' slot %d (key %s): %v",
				req.ip, modelName, beID, slotID, key16(key), err)
			restorePrevKV(beSm, slotID, prevKV, modelName, beID)
			beSm.Release(slotID)
			released = true
			recordChatError(req.requestID, modelName, beID, slotID, req.t0, routingReason)
			WriteJSON(req.w, http.StatusServiceUnavailable, map[string]any{"error": "backend connection failed"})
			return
		}
		logError("processor", "Chat error for client %s, model '%s' on backend '%s' slot %d (key %s): %v",
			req.ip, modelName, beID, slotID, key16(key), err)
		restorePrevKV(beSm, slotID, prevKV, modelName, beID)
		recordChatError(req.requestID, modelName, beID, slotID, req.t0, routingReason)
		WriteJSON(req.w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}
	if out == nil {
		restorePrevKV(beSm, slotID, prevKV, modelName, beID)
		recordChatError(req.requestID, modelName, beID, slotID, req.t0, routingReason)
		WriteJSON(req.w, http.StatusBadGateway, map[string]any{"error": "provider non-JSON body"})
		return
	}

	saveOK := false
	cacheSize := 0
	recomputeHappened := false
	llmPromptTokens := 0
	cachedTokens := 0
	if dec.hitType != nil {
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
		logInfo("processor", "Recompute check for model '%s' slot %d: cached_tokens=%d, request_prompt_tokens=%d (llm=%d), ratio=%.3f",
			modelName, slotID, cachedTokens, dec.promptTokens, llmPromptTokens, dec.bestRatio)
		var cacheMeta map[string]any
		if effRestoreKey != "" && effRestoreBackend != "" {
			cacheMeta = kvMeta.ReadMeta(effRestoreKey, effRestoreBackend)
		}
		cacheNTokens := metaNTokens(cacheMeta, llmPromptTokens)
		if isRecompute(cachedTokens, llmPromptTokens, cacheNTokens) {
			recomputeHappened = true
			logWarn("processor", "Recompute detected for model '%s' on backend '%s' slot %d (key %s): cached_tokens=%d llm_prompt_tokens=%d cache_n_tokens=%d, KV cache restore was partial/useless",
				modelName, beID, slotID, key16(key), cachedTokens, llmPromptTokens, cacheNTokens)
			if effRestoreKey != "" && effRestoreBackend != "" {
				kvMeta.IncrementRecomputePenalty(effRestoreKey, effRestoreBackend)
			}
		}
	}

	servingBeRatio := dec.backendCacheRatios[beID]
	skipEntry := &SaveSkipEntry{
		Key:          key,
		Model:        modelName,
		Blocks:       blocks,
		NTokens:      dec.promptTokens,
		HitType:      dec.hitType,
		ServingRatio: servingBeRatio,
		Recompute:    recomputeHappened,
		Legacy:       false,
	}
	if ShouldSkipSaveHeuristic(dec.promptTokens, backendManager.GetBackendNCtx(modelName, beID), req.messages, req.requestJSON) {
		beSm.MarkSaveSkipped(slotID, skipEntry)
	} else if ShouldSaveCache(servingBeRatio, recomputeHappened) {
		saveOK, cacheSize = beSm.SaveAfter(modelName, slotID, key, blocks, dec.promptTokens)
	} else {
		beSm.MarkSaveSkipped(slotID, skipEntry)
	}

	latency := latencyMS(req.t0)
	cacheSizeBytes := 0
	if saveOK {
		cacheSizeBytes = cacheSize
	}
	Metrics.Record(map[string]any{
		"request_id":       req.requestID,
		"t0":               req.t0,
		"request_json":     req.requestJSON,
		"model":            modelName,
		"backend":          beID,
		"slot_id":          slotID,
		"cache_hit":        dec.hitType != nil,
		"restored":         restored,
		"recompute":        recomputeHappened,
		"saved":            saveOK,
		"latency_ms":       latency,
		"n_tokens":         llmPromptTokens,
		"cached_tokens":    cachedTokens,
		"stream":           false,
		"cache_size_bytes": cacheSizeBytes,
		"prompt_preview":   req.promptPreview,
		"routing_reason":   routingReason,
		"status":           "complete",
	})
	backendManager.UpdateBackendLatency(beID, latency)
	reqType := "conversation"
	if req.reqClass.Score >= 0.4 {
		reqType = "summarization"
	}
	backendManager.UpdateBackendModelLatency(beID, modelName, latency, reqType)
	WriteJSON(req.w, http.StatusOK, out)
}
