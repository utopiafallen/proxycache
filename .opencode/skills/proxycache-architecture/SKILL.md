# Skill: proxycache-architecture

# Proxycache Architecture

Go-native codebase at the repo root. The original Python implementation lives in `archive/python/` as a behavior reference only — prefer the Go code as canonical.

## Request Pipeline (matcher + per-backend workers)

The old design had every HTTP handler drive its own multi-phase slot acquisition across all backends. That is gone. Now:

```
ChatHandler (app.go)          thin: parse, build ProcRequest, Submit(), block on req.done
    |
RequestDispatcher (matcher.go)  global singleton
    |-- 1 matcher goroutine: scans each arrival (per-backend tokenize + disk/pending-slot
    |   cache scan), runs SelectBackend(), enqueues onto the chosen backend's queue
    |-- per-backend worker goroutines: process that backend's queue sequentially
    |   (one in-flight request per backend): acquire slot ON THIS BACKEND ONLY, restore,
    |   dispatch to llama.cpp, save, release
    |-- global FIFO overflow queue: requests whose chosen backend is full; re-matched on
        capacity free or liveness change (notify())
```

**Why this shape:** one in-flight request per backend keeps slot/KV state deterministic (no cross-backend races), and the single matcher serializes the expensive scan+decide step so routing decisions see consistent queue depths.

### Routing rule (`SelectBackend`, exported pure function — unit-tested directly)

1. Cache hit + hit backend's **queue depth** (queued items, not in-flight count) < `CacheHitQueueLimit` (default 2) → route to the hit backend (prefers the candidate whose model name matches the hit's canonical name).
2. Otherwise round-robin across the other matching backends, starting from the first non-hit candidate at/after the model's RR index; a P2P transfer of the best **disk** cache is triggered to the chosen backend. The saturated hit backend is only taken in a second pass when every other candidate is full.

Per-backend queues cap at `BackendQueueMax` (default 5); beyond that, requests park in the global overflow queue (head-of-line ordering preserved — a non-routable head stops the drain pass).

### Worker processing (`processor.go`)

- `acquireSlotOnBackend`: retries forever on its own backend (growing backoff `SlotAcquireRetryBaseSeconds` × attempt). Empty/missing pool → `RefreshSlotCounts()` then immediate retry (pool may just have been created); if the backend no longer serves the model (`ServesModel()` false) → **requeue** to the global overflow for re-matching instead of failing. No 503 on slot exhaustion — abort only on client disconnect / ctx cancellation.
- `resolveRestoreKey`: restores from the best *disk* candidate — directly when local; otherwise bounded-wait (`CacheTransferWait`, default 5s) for an in-flight P2P transfer, then `CacheExists()`. A pending-slot hit on the same backend suppresses disk restore (slot already warm).
- `doWorkerRestore`: flush previously-skipped save, apply `ShouldSkipRestore` heuristic, issue restore.
- Then dispatch (stream via `serveStream`, non-stream via `processNonStream`), recompute detection, save heuristics, post-response save.

**Request contexts:** each request runs on a context derived from the client's `r.Context()`; `ProcRequest.finish()` cancels it and closes `done`. A **requeued** request must NOT be finished by the worker that requeued it — its terminal state belongs to whoever completes it later; finishing early makes the overflow drain discard it as disconnected.

## P2P Cache Transfer (`transfer.go`)

When a cache-hit request is routed away from the backend holding the cache, `RequestCacheTransfer(src, dst, key)` migrates the file asynchronously so a restore is possible when the worker runs.

- Deduped per (key, dst) via an in-flight map; skipped when dst already holds the file.
- Four backend-type combos: agent→agent push (`/cache/transfer`), local→local direct copy, local→agent upload (`/cache/receive`), agent→local pull (`/cache/file`).
- Receiving side makes space within its `cache_max_size_gb` budget (oldest-mtime first; `.ckpt*` sidecars removed with their key).
- On success: meta rewritten for the target + `AddTransferredEntry()` registers it in the target's ring so disk scans/eviction treat it like a native entry.

## Cache Key Computation

**Critical:** Cache keys are computed from **prompt token IDs only**, NOT prompt+response.

```go
// hashing.go
func MetaKey(canonicalName string, tokenIDs []int) string // sha256(canonical_name + '\n' + token_ids)
```

`tokenIDs` come from per-backend tokenization during the matcher scan (the chat-templated prompt only), so the key is known before any slot work.

## Slot KV State Tracking (`slotKVState`)

`BackendSlotManager.slotKVState map[int][]string` tracks KV cache block hashes per slot (keyed by `slotID`, per-backend instance).

- Set at slot acquisition time, making the slot's KV state visible to subsequent requests' cache hit scans while the slot is in-flight
- Updated after a successful restore or save to reflect the slot's current state
- On backend error (400+), restored to `prevKV` (the state captured before acquisition's `SetKVState()`) since the request was never processed
- Cleared by `Invalidate()` on cancellation/failure in `StreamState.cleanup()` before `Release()`

## Cache Hit Scan Flow (in `match`, `matcher.go`)

1. **Model resolution**: `GetDiscoveredModels(clientModel)` (exact → substring → `"any"`); zero options → retry `DiscoverModels()` once.
2. **Size the prompt** on the first live backend of the first option (template + tokenize).
3. **Per-backend, per-option scan**: template + tokenize on that backend's tokenizer; compute block hashes;
   - **Disk scan**: `FindBestRestoreCandidate()` for best meta-file hit (also recorded as `diskRestoreKey`/`diskRestoreBackend`, independent of the primary hit)
   - **Pending slot scan**: LCP ratio of request blocks vs each in-flight slot's tracked KV state. A pending-slot hit at ratio >= `LCPTh` can become the *primary* hit (`CacheHitSkip`, clears `restoreKey`) — but the disk candidate is still tracked separately for P2P purposes.
4. **Select** via `SelectBackend(candidates, hit, pendingOf, rrIndex)`.

**Pending slot hit semantics**: the slot's KV cache already contains the relevant content from a previous request. Setting a restore key would cause a failed restore against a non-existent file, so the primary hit carries no key and `resolveRestoreKey` suppresses disk restore when the worker lands on that same backend. When the request is routed to a *different* backend, the best **disk** key is transferred there (RAM KV state can't move, files can).

## Save Timing

When a request completes and starts saving, subsequent requests that arrive during the save window find the cache via the pending slot scan (the meta file hasn't been written yet). Once the save completes and the slot is released, the disk scan picks up the new meta file.

**Flow:**
1. Stream completes → `StreamState.cleanup()` calls `save()`
2. `save()` calls `SaveAfter()` which writes KV cache to disk and writes meta file
3. Slot is released via `Release()`
4. **During steps 1-2:** Pending slot scan finds the in-flight slot's KV state and uses it as a cache hit candidate
5. **After step 3:** Disk scan picks up the new meta file

`SaveAfter()` updates the ring entry **in place** when the key is already registered (e.g. immediately after a P2P transfer of the same key) instead of appending a duplicate.

### Clobber-flush of skipped saves

The ratio-based save skip is a ring-budget decision: while an on-disk ancestor covers the content above `CACHE_SAVE_RATIO_THRESHOLD`, persisting the slot's longer version would grow the ring (eviction risk) for a tail that's cheap to recompute. The unsaved version lives only in the slot — and any unrelated request that takes that slot clobbers it. `flushSkippedSave()` (processor.go) runs when a request acquires a slot that holds a `MarkSaveSkipped` entry: if it is about to clobber different content (`canSkip` false, no copy of the key on this backend), it re-evaluates coverage **at current disk state** and persists the skipped cache only when `ShouldSaveCache` would now return true (coverage degraded below threshold, e.g. the ancestor was evicted). While coverage still holds, the clobber is allowed to lose the tail — that's exactly what the skip priced in. Explicit content-class skips (near-max-context / summarization) are never overridden.

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

## Backend Discovery & Liveness Loop (`livenessLoop`)

Pings backends every 5s, triggers model discovery on state change.

**Health check flow (per backend):**
1. `client.HealthCheck(ctx)` — hits `/health` with a 2s timeout using the current HTTP client
2. If it fails: `client.Recreate()` (the pool may be poisoned), retry once with the fresh client
3. Record per-backend health result: `is_up`, timing, error name, whether recreated/retried

**State change triggers:**
- Backend up↔down transition → sets `changed=true`, recreates the client
- Up backend has no models in registry (`up_no_models`) → also triggers discovery, **gated per backend** to at most once per `MISSING_MODELS_RETRY_INTERVAL` (30s)
- When `changed`: runs `discoverModelsCtx()` and `refreshSlotCountsCtx()` **concurrently** (two goroutines, shared 10s `context.WithTimeout`), then wakes the dispatcher (`GetDispatcher().notify()`) so overflow-parked requests re-match against the fresh backend set

**Model discovery (`DiscoverModels`):**
- Iterates backends sequentially, skipping those marked down
- For each: calls `LlamaClient.DiscoverModels()` which tries router `/models` then fallback `/v1/models`
- Stores per-backend timing in `lastDiscoverTiming`
- Result stored in `discoveredModels`, used by the matcher and slot manager

**Slot pool creation:** pools are (re)built from `RefreshSlotCounts()` — driven by liveness state changes, or lazily by a worker whose backend pool for the model is missing/empty (first request after discovery).

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
- **Cache key = prompt tokens only**: The key is computed from `optTokenIDs` (prompt), not the full request+response. This is known at match time.
- **Queue depth, not in-flight count, drives the hit limit**: `SelectBackend` sees only queued items; the single in-flight request per backend doesn't count against `CacheHitQueueLimit`. That's why a busy hit backend still accepts 2 queued followers.
- **`slotKVState` is keyed by `slotID`** (per `BackendSlotManager` instance, not globally). Access via `GetSlotManager().Get(backendID).GetKVState(slotID)`.
- **`Invalidate()` clears `slotKVState`**: Called on cancellation/failure in `StreamState.cleanup()` before `Release()`.
- **Ring buffer eviction is per-backend**: Uses `cache_max_size_gb` per backend (default 25 GB). Evicts age-first, then LRU. `MakeSpaceFor(need)` reuses the same scoring to free room for an incoming P2P transfer.
- **Pending slot scan uses per-backend blocks**: The scan runs inside the per-backend loop in `match`, so `blocks` always matches the current backend's tokenizer output.
- **Pending slot hit clears the primary `restoreKey` but not the disk candidate**: a disk hit on backend A followed by a pending-slot hit must clear the primary key (a stale key causes a failed restore), while the best disk key is retained separately so a fallback routing can still P2P-transfer it.
- **Requeued requests must not be finished early**: `processWorkerRequest` tracks the requeued flag; finishing (which cancels the derived ctx) before the overflow drain re-routes the request makes the drain discard it as disconnected.
- **Backend error KV state restore**: When the backend errors (400+ status, connection error, non-JSON body) before processing the request, the slot's actual KV cache is untouched. The worker captures `prevKV = beSm.GetKVState(slotID)` before `SetKVState()`; error handlers restore it via `beSm.SetKVState(slotID, prevKV)`. Timeout errors (504) do NOT restore — the request may have been partially processed.
- **`prevPassed` parameter in `ShouldSkipRestore`**: Go uses a `prevPassed bool` argument (no sentinel). `prevPassed=false` falls back to the tracked `slotKVState` (test-compat); `prevPassed=true` with `prevBlocks=nil` means a fresh slot and always returns false — can't skip restore on a slot with no tracked state.
- **Backend keys are sanitized everywhere in Go**: `SanitizeBackendDir()` (colons → dashes, e.g. `10.0.0.1:8000` → `10.0.0.1-8000`) is used for in-memory maps, discovered-model backends, AND filesystem paths. The archived Python version kept raw colon keys in memory and sanitized only for disk — when cross-referencing archive code or its tests, note this difference.
- **Package-level config vars are read once at init**: `config.go` vars (`WordsPerBlock`, `LCPTh`, ... pipeline limits) are initialized from env vars during package init. Changing them in tests requires assigning the package var directly; re-reading env vars has no effect after startup.
- **Liveness discovery/refresh runs under a shared 10s context**: `discoverModelsCtx` and `refreshSlotCountsCtx` run as two goroutines under one `context.WithTimeout(10s)`. Errors are captured in closure vars (`discErr`/`slotsErr`) and inspected individually — don't rely on the outer select. On timeout the loop recreates ALL backend clients (a stalled request can poison the whole pool) and records `discover_error`/`slots_error` as `"timeout"`.
- **Health check retry**: Every failed health check gets a client recreation + retry before flipping state. This prevents a single transient failure from marking an up backend as down, which would trigger discovery (potentially timing out) and start an oscillation cycle.
- **Don't cancel the request context while streaming the body**: cancelling an in-flight response read leaves the pooled connection broken and subsequent requests on it can hang. `StreamState` never cancels for timeouts; the only disconnect paths are backend data/error/close or client disconnect (heartbeat).
- **Test isolation**: tests swap the global backend manager per test (`withTestBackend`); its cleanup calls `GetDispatcher().AbortAll()` to cancel in-flight/queued work and wait for workers to wind down *before* the manager is swapped back, so no orphaned worker goroutine touches a stale manager (that panics in `GetClient`).

## Key Functions

| Function | Location | Role |
|----------|----------|------|
| `MetaKey()` | `hashing.go` | Compute cache key from canonical name + token IDs |
| `BlockHashesFromTokens()` / `BlockHashesFromTokensDefault()` | `hashing.go` | Convert token IDs to block hashes for LCP matching |
| `LCPBlocks()` | `hashing.go` | Compute longest common prefix between two block lists |
| `SanitizeBackendDir()` | `hashing.go` | Colons → dashes for backend keys (used everywhere) |
| `FindBestRestoreCandidate()` | `kvmeta.go` | Scan disk meta files for best cache hit |
| `ScanAllMeta()` | `kvmeta.go` | Load all meta files from disk for a backend |
| `RequestDispatcher.match()` | `matcher.go` | Scan (tokenize + disk/pending-slot) + route one request; sets `RouteDecision` |
| `SelectBackend()` | `matcher.go` | Exported pure routing decision (hit-under-limit vs RR fallback); unit-testable |
| `acquireSlotOnBackend()` | `processor.go` | Worker-side slot acquisition on its own backend: backoff retry, refresh, requeue on model departure |
| `resolveRestoreKey()` | `processor.go` | Pick the effective restore key (local disk or P2P-landed), suppress when slot is warm |
| `doWorkerRestore()` | `processor.go` | Flush skipped save, skip-restore check, issue restore |
| `RequestCacheTransfer()` | `transfer.go` | Async, deduped P2P transfer across the four backend-type combos |
| `ShouldSkipRestore()` | `slotmanager.go` | Skip restore if LCP ratio >= threshold (and block diff within pct), single-slot backend, and valid prev state |
| `AddTransferredEntry()` | `slotmanager.go` | Register a P2P-received file in the ring (dedupe by key) |
| `MakeSpaceFor()` | `slotmanager.go` | Proactively evict until an incoming transfer fits the budget |
| `Invalidate()` | `slotmanager.go` | Clear `slotKVState` for a slot |
| `GetKVState()` / `SetKVState()` | `slotmanager.go` | Get/set slot KV block tracking |
| `SaveAfter()` | `slotmanager.go` | Save KV cache to disk, write meta file, update ring buffer (in-place on known key), update `slotKVState` |
| `EvictIfNeeded()` | `slotmanager.go` | Age-first then LRU eviction when per-backend bytes exceed limit |
| `StreamState.save()` | `app.go` | Detect recompute, apply save heuristics, call `SaveAfter()`, return ok/cacheSize |
| `StreamState.cleanup()` | `app.go` | Stream lifecycle: save, invalidate, release slot, record metrics |
| `DiscoverModels()` | `backendmanager.go` | Discover models across all up backends sequentially, store in `discoveredModels` |
| `ServesModel()` | `backendmanager.go` | Whether a backend currently lists the model (requeue gate) |
| `RefreshSlotCounts()` | `backendmanager.go` | Query slots for each discovered model+backend pair |
| `livenessLoop()` | `backendmanager.go` | 5s health check loop, concurrent discovery/refresh on state change, dispatcher notify, rate-limited liveness events |
| `livenessDiagDue()` | `backendmanager.go` | Pure gate: should a `liveness_diag` event record this tick |
| `LlamaClient.DiscoverModels()` | `llamaclient.go` | Try router `/models`, fallback `/v1/models` |
