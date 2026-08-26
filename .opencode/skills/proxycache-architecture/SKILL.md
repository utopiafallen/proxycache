---
name: proxycache-architecture
description: Proxycache KV cache slot management, cache hit scanning, slot acquisition, restore/save logic, skip-restore, pending slot hits, backend error handling, liveness loop, and HTTP client lifecycle. Use when working with slots, cache hits, KV cache state, restore, save, routing, request flow, health checks, or backend discovery.
---

# Proxycache Architecture

Go-native codebase at the repo root. The original Python implementation lives in `archive/python/` as a behavior reference only — prefer the Go code as canonical.

## Cache Key Computation

**Critical:** Cache keys are computed from **prompt token IDs only**, NOT prompt+response.

```go
// hashing.go
func MetaKey(canonicalName string, tokenIDs []int) string // sha256(canonical_name + '\n' + token_ids)
```

`tokenIDs` comes from `firstTokenIDs` or `optTokenIDs` (computed in the chat handler before the cache hit scan), which are the result of tokenizing the chat template — the prompt only.

This means the cache key is **known at slot acquisition time**, before the response is generated.

## Slot KV State Tracking (`slotKVState`)

`BackendSlotManager.slotKVState map[int][]string` tracks KV cache block hashes per slot (keyed by `slotID`, per-backend instance).

- Set at slot acquisition time, making the slot's KV state visible to subsequent requests' cache hit scans while the slot is in-flight
- Updated after a successful restore or save to reflect the slot's current state
- On backend error (400+), restored to `prevKV` (the state captured before acquisition's `SetKVState()`) since the request was never processed
- Cleared by `Invalidate()` on cancellation/failure in `streamState.cleanup()` before `Release()`

## Cache Hit Scan Flow (in `chatHandler`, `app.go`)

1. **Tokenize on each backend**: Each backend applies its chat template and tokenizes the messages
2. **Compute block hashes**: `BlockHashesFromTokensDefault(optTokenIDs)` (or explicit `WORDS_PER_BLOCK`)
3. **Disk scan**: `kvMeta.FindBestRestoreCandidate()` scans disk meta files for best cache hit
4. **Pending slot scan**: Iterate `slotKVState` for the same model+backend, compute LCP ratio against request blocks. If a pending slot has a better ratio than the disk hit, use it instead. **Clears `restoreKey`** — the slot already has the KV content in its cache, so no restore is needed.
5. **Select best match**: Use the candidate with the highest LCP ratio

The cache hit scan lives in `chatHandler` in `app.go`, between model resolution and slot acquisition. The pending slot scan runs inside the same per-backend loop as the disk scan, so it uses the correct `blocks` for each backend's tokenizer.

**Pending slot hit semantics**: When a pending slot is found, the slot's KV cache already contains the relevant content (from the previous request that filled it). Setting `restoreKey` would cause a failed restore attempt against a non-existent cache file. The code explicitly sets `restoreKey = ""` to skip the restore — the request proceeds directly with the slot's existing KV cache.

## Slot Acquisition Flow (`acquireSlotForRequest`)

**Phase 0 — Wait for cache backend:**
- Uses EMA-derived timeout per backend (slot duration EMA clamped to `CACHE_HIT_WAIT_EMA_MIN_TIMEOUT`/`MAX_TIMEOUT`)
- Polls every 5s for a free slot
- Max `CACHE_HIT_WAIT_MAX_PENDING_REQS` concurrent waiters per backend

**Phase 1 — Try cache backend directly:**
- Attempts to acquire the cache backend slot without waiting
- If successful, restores cache and returns

**Phase 2 — Retry loop:**
- Iterates all candidate backends (fallback only, excludes cache backend), sorted by composite score: `(cache_ratio, ring_size, latency_ema, last_used)` to minimize cache churn and latency
- Retries forever until a slot frees (backoff grows 5s, 10s, 15s, ...); aborts only on client disconnect / context cancellation
- Picks first available slot

**After slot acquisition succeeds (all phases):**
- `slotKVState[slotID]` is set to the request's blocks (before any restore decision)
- The slot's previous KV state (`prevKV`) is captured before `SetKVState()` and returned as the 4th return value
- The nested `doRestoreCall` closure checks skip-restore using the captured previous state
- On backend error, `prevKV` is used to restore `slotKVState` so tracking stays accurate

## `ShouldSkipRestore` Constraints

Only applies to **single-slot backends**:
```go
for _, pool := range b.slotPools {
    if len(pool) > 1 {
        return false
    }
}
```

With multiple slots, skipping a restore is unsafe because llama.cpp may evict the chosen slot's cache under memory pressure.

The function compares the slot's **previous** KV state (captured at acquisition, before being overwritten with the request blocks) against the request blocks. If the LCP ratio is >= `KV_CACHE_SKIP_THRESHOLD` (0.9), the restore is skipped — but only if the block-count difference is within `KV_CACHE_SKIP_MAX_BLOCK_DIFF_PCT` (0.1), guarding against near-identical-length but divergent caches.

## Save Timing

When a request completes and starts saving, subsequent requests that arrive during the save window find the cache hit via the pending slot scan (the meta file hasn't been written yet). Once the save completes and the slot is released, the disk scan picks up the new meta file.

**Flow:**
1. Stream completes → `streamState.cleanup()` calls `save()`
2. `save()` calls `SaveAfter()` which writes KV cache to disk and writes meta file
3. Slot is released via `Release()`
4. **During steps 1-2:** Pending slot scan finds the in-flight slot's KV state and uses it as a cache hit candidate
5. **After step 3:** Disk scan picks up the new meta file

## Backend Discovery & Liveness Loop (`livenessLoop`)

Pings backends every 5s, triggers model discovery on state change.

**Health check flow (per backend):**
1. `client.HealthCheck(ctx)` — hits `/health` with a 2s timeout using the current HTTP client
2. If it fails: `client.Recreate()` (the pool may be poisoned), retry once with the fresh client
3. Record per-backend health result: `is_up`, timing, error name, whether recreated/retried

**State change triggers:**
- Backend up↔down transition → sets `changed=true`, recreates the client
- Up backend has no models in registry (`up_no_models`) → also triggers discovery, **gated per backend** to at most once per `MISSING_MODELS_RETRY_INTERVAL` (30s)
- When `changed`: runs `discoverModelsCtx()` and `refreshSlotCountsCtx()` **concurrently** (two goroutines, shared 10s `context.WithTimeout`)

**Model discovery (`DiscoverModels`):**
- Iterates backends sequentially, skipping those marked down
- For each: calls `LlamaClient.DiscoverModels()` which tries router `/models` then fallback `/v1/models`
- Stores per-backend timing in `lastDiscoverTiming`
- Result stored in `discoveredModels`, used by slot manager and routing

**Slot refresh (`RefreshSlotCounts`):**
- Iterates discovered models → queries each backend's slots via `GetSlotsInfo()`
- Updates `refreshState[backend][model]`
- Used by fallback sorting and EMA latency tracking

## Liveness Diagnostics

On every noteworthy liveness iteration (state change, health errors, retries, or discover/refresh errors), a `liveness_diag` event is recorded **when the `livenessDiagDue()` gate allows** (see metrics-architecture skill for the rate-limiting rationale):
- Per-backend health results: `is_up`, `old_state`, `state_changed`, timing in ms, error name, `recreated`, `retry_succeeded`
- Current backend states dict
- Whether discovery ran (`changed`) and count of discovered models
- Per-backend discovery timing from `lastDiscoverTiming`
- Discovery/refresh timing and error names (or `nil` if none)
- Total loop duration in ms

Query via `GET /metrics/diagnostics?liveness_diag=true`.

## Gotchas

- **Tokenization is backend-specific**: Each backend applies its own chat template and tokenizer. Different backends may produce different token IDs for the same messages.
- **Cache key = prompt tokens only**: The key is computed from `optTokenIDs` (prompt), not the full request+response. This is known at acquisition time.
- **`slotKVState` is keyed by `slotID`** (per `BackendSlotManager` instance, not globally). Access via `slotManager.Get(backendID).GetKVState(slotID)`.
- **`Invalidate()` clears `slotKVState`**: Called on cancellation/failure in `streamState.cleanup()` before `Release()`.
- **Ring buffer eviction is per-backend**: Uses `cache_max_size_gb` per backend (default 25 GB). Evicts age-first, then LRU.
- **Pending slot scan uses per-backend blocks**: The scan runs inside the per-backend loop in `chatHandler`, so `blocks` always matches the current backend's tokenizer output.
- **Pending slot hit clears `restoreKey`**: The cache hit scan iterates backends in a loop — a disk hit on backend A sets `restoreKey`, then a pending slot hit on backend B must clear it. Leaving the stale key causes a failed restore attempt against a cache file that doesn't exist on backend B.
- **Backend error KV state restore**: When the backend errors (400+ status, connection error, non-JSON body) before processing the request, the slot's actual KV cache is untouched. `acquireSlotForRequest` captures `prevKV = beSm.GetKVState(slotID)` before `SetKVState()`, returns it as the 4th element. Error handlers in `chatHandler` restore `prevKV` via `beSm.SetKVState(slotID, prevKV)`. Timeout errors (504) do NOT restore — the request may have been partially processed.
- **`prevPassed` parameter in `ShouldSkipRestore`**: Go uses a `prevPassed bool` argument (no sentinel). `prevPassed=false` falls back to the tracked `slotKVState` (test-compat); `prevPassed=true` with `prevBlocks=nil` means a fresh slot and always returns false — can't skip restore on a slot with no tracked state.
- **Backend keys are sanitized everywhere in Go**: `SanitizeBackendDir()` (colons → dashes, e.g. `10.0.0.1:8000` → `10.0.0.1-8000`) is used for in-memory maps, discovered-model backends, AND filesystem paths. The archived Python version kept raw colon keys in memory and sanitized only for disk — when cross-referencing archive code or its tests, note this difference.
- **Package-level config vars are read once at init**: `config.go` vars (`WordsPerBlock`, `LCPTh`, ...) are initialized from env vars during package init. Changing them in tests requires assigning the package var directly (e.g., `WordsPerBlock = 3`); re-reading env vars has no effect after startup.
- **Liveness discovery/refresh runs under a shared 10s context**: `discoverModelsCtx` and `refreshSlotCountsCtx` run as two goroutines under one `context.WithTimeout(10s)`. Errors are captured in closure vars (`discErr`/`slotsErr`) and inspected individually — don't rely on the outer select. On timeout the loop recreates ALL backend clients (a stalled request can poison the whole pool) and records `discover_error`/`slots_error` as `"timeout"`.
- **Health check retry**: Every failed health check gets a client recreation + retry before flipping state. This prevents a single transient failure from marking an up backend as down, which would trigger discovery (potentially timing out) and start an oscillation cycle.
- **Don't cancel the request context while streaming the body**: cancelling an in-flight response read leaves the pooled connection broken and subsequent requests on it can hang. `streamState` never cancels for timeouts; the only disconnect paths are backend data/error/close or client disconnect (heartbeat).

## Key Functions

| Function | Location | Role |
|----------|----------|------|
| `MetaKey()` | `hashing.go` | Compute cache key from canonical name + token IDs |
| `BlockHashesFromTokens()` / `BlockHashesFromTokensDefault()` | `hashing.go` | Convert token IDs to block hashes for LCP matching |
| `LCPBlocks()` | `hashing.go` | Compute longest common prefix between two block lists |
| `SanitizeBackendDir()` | `hashing.go` | Colons → dashes for backend keys (used everywhere) |
| `FindBestRestoreCandidate()` | `kvmeta.go` | Scan disk meta files for best cache hit |
| `ScanAllMeta()` | `kvmeta.go` | Load all meta files from disk for a backend |
| `acquireSlotForRequest()` | `app.go` | Slot acquisition with Phase 0/1/2 logic, captures `prevKV`, sets `slotKVState`, returns `(GSlot, restored, skipRestoreDiag, prevKV, error)` |
| `doRestoreCall` (closure) | `app.go` (in `acquireSlotForRequest`) | Execute restore: flush skipped save, skip-restore check, disk restore, update `slotKVState` |
| `ShouldSkipRestore()` | `slotmanager.go` | Skip restore if LCP ratio >= threshold (and block diff within pct), single-slot backend, and valid prev state |
| `Invalidate()` | `slotmanager.go` | Clear `slotKVState` for a slot |
| `GetKVState()` / `SetKVState()` | `slotmanager.go` | Get/set slot KV block tracking |
| `SaveAfter()` | `slotmanager.go` | Save KV cache to disk, write meta file, update ring buffer, update `slotKVState` |
| `evictIfNeeded()` | `slotmanager.go` | Age-first then LRU eviction when per-backend bytes exceed limit |
| `streamState.save()` | `app.go` | Detect recompute, apply save heuristics, call `SaveAfter()`, return ok/cacheSize |
| `streamState.cleanup()` | `app.go` | Stream lifecycle: save, invalidate, release slot, record metrics |
| `DiscoverModels()` | `backendmanager.go` | Discover models across all up backends sequentially, store in `discoveredModels` |
| `RefreshSlotCounts()` | `backendmanager.go` | Query slots for each discovered model+backend pair |
| `livenessLoop()` | `backendmanager.go` | 5s health check loop, concurrent discovery/refresh on state change, rate-limited liveness events |
| `livenessDiagDue()` | `backendmanager.go` | Pure gate: should a `liveness_diag` event record this tick |
| `LlamaClient.DiscoverModels()` | `llamaclient.go` | Try router `/models`, fallback `/v1/models` |
