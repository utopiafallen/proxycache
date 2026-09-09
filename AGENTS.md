# proxycache

OpenAI-compatible proxy for `llama.cpp` KV cache slot management with disk save/restore. **Go-native** — the original Python implementation is preserved under `archive/python/` as a behavior reference only.

## Environment

**WSL2** — builds target Windows using the Windows Go toolchain: `/mnt/c/Program Files/Go/bin/go.exe` (the build scripts locate it automatically; plain `go build -o proxycache.exe ./cmd/proxycache` works too). Do NOT use a Linux Go toolchain — the binaries must run on Windows.

The archived Python app (only if you're working on `archive/python/`) uses `uv` from Windows. Do NOT install Python packages via pip/apt inside WSL2.

## Commands

```bash
./build-proxycache.sh                        # builds proxycache.exe at repo root
./build-cache-agent.sh                       # builds cache-agent.exe at repo root
go build -o proxycache.exe ./cmd/proxycache  # same as build-proxycache.sh
go vet ./...                                 # no linter configured; vet is the check
go test -count=1 ./...                       # full test suite (stdlib `testing`, no framework)
                                             # agent/P2P transfer tests spawn the REAL cache-agent.exe from the repo
                                             # root — run ./build-cache-agent.sh first, or those tests skip
./proxycache.exe                             # run (all config via env vars, see config.go)
```

**No linter, no typechecker, no test framework** — stdlib `testing` only.

## Architecture

| File | Role |
|------|------|
| `entry.go` | Library entry point (`Main()`): slot init from disk, meta reconciliation, embedded cache-agent start (for `agent_serve_port` backends), liveness start, request-dispatcher start/stop, HTTP mux, graceful shutdown |
| `cmd/proxycache/main.go` | Thin `package main` wrapper that calls `proxycache.Main()`; the only `package main` in the root module |
| `tests/` | External test package (`package tests`) importing `proxycache`; all test files live here |
| `app.go` | HTTP routes, thin chat handler (submit to dispatcher + wait), streaming pipeline (`serveStream` → `StreamState`) |
| `matcher.go` | `RequestDispatcher` (global singleton): one matcher goroutine scans each incoming request (tokenize + cache scan) and routes it; per-backend FIFO queues pumped by the same goroutine (spawns one worker per free slot); queue-migration monitor; global FIFO overflow queue; exported pure `SelectBackend()` decision function |
| `processor.go` | Worker-side processing: cache restore (`resolveRestoreKey`, P2P-aware, with pending-hit revalidation), dispatch to llama.cpp (stream/non-stream), skipped-save clobber flush, recompute detection, save heuristics, post-response save. Slot acquisition lives in the dispatcher pump; only the lazy-discovery worker acquires itself |
| `transfer.go` | P2P cache file transfer orchestration (`RequestCacheTransfer`, in-flight dedupe) across the four backend-type combos (agent+agent push, local+local copy, local→agent upload, agent→local pull) |
| `backendmanager.go` | `BackendManager`: backend registry, LlamaClient/CacheAgentClient instances, model-to-backend mapping, liveness checker |
| `config.go` | All config via env vars (no .env file), save heuristics, request classification |
| `hashing.go` | Hashing primitives: block hashes from tokens, LCP matching, cache key generation, backend key sanitization |
| `kvmeta.go` | All meta I/O: read/write/delete `.meta.json` (atomic via temp+rename), scan, reconcile, find restore candidates |
| `llamaclient.go` | HTTP client to llama.cpp; slot save/restore, router-mode slot discovery |
| `slotmanager.go` | Per-model slot pools (`BackendSlotManager`), ring buffer eviction, KV cache skip logic, transferred-entry registration + make-space for P2P |
| `cacheagent.go` | `CacheAgentClient` (remote cache deletion + P2P transfer/upload/make-space/fetch/sidecar listing) + `AgentServer`, the embedded cache-agent server — started for local backends that set `agent_serve_port` so this proxy can serve as the cache-agent for its own local llama instances. Keep it in sync with the standalone binary in `cache-agent/` |
| `metrics.go` | In-memory metrics collector: single ring buffer (retention 200) holds request records and diagnostic events; three-phase recording (arrival → routing → completion); `GetRequests()` filters out events; `GetTimeline()` returns unified view |
| `logger.go` | Levelled logging (`LOG_LEVEL`) |
| `dashboard.html` | Metrics dashboard served at `/dashboard` (toggle via `DASHBOARD_ENABLED`), embedded into the binary |
| `cache-agent/` | Separate Go module: standalone cache-agent binary — the full cache API (delete, size/batch lookups, make-space) plus the four P2P transfer endpoints. Runs in production on remote hosts; tests spawn this exact binary |
| `kv_meta/` | Per-cache `.meta.json` files (gitignored) |
| `archive/python/` | Original Python implementation — reference for intended behavior, not maintained |

## Gotchas

- **llama.cpp prerequisite**: MUST start with `--slot-save-path <dir>`. Cache save/restore fails silently without it.
- **Config**: all env vars only — `config.go` has defaults. No `.env` file support.
- **Binary ignores CLI args**: `entry.go` binds `0.0.0.0:$PORT` from the environment; `--host`/`--port` flags on the command line are silently ignored. Set the `PORT` env var instead.
- **Cache key**: `sha256(canonical_name + '\n' + token_ids)` (`MetaKey()` in `hashing.go`) — based on token IDs, not raw text.
- **Backend keys**: sanitized `host-port` strings (only colons replaced with dashes, e.g. `"10.0.0.1:8000"` → `"10.0.0.1-8000"`), NOT raw `host:port`. `SanitizeBackendDir()` in `hashing.go`. Used as directory names under `META_DIR/`.
- **BACKEND_MODE**: env var, default `"llama-cpp"`. Set to `"llama-swap"` to route slot save/restore through `/upstream/{model}/slots/{id}` instead of `/slots/{id}`.
- **Slot pinning** is duplicated 3 ways in every request body (`withSlotID()` in `llamaclient.go`): root (`slot_id`, `id_slot`, `_slot_id`), `options` dict, and query params.
- **Save happens after response** completes (both stream and non-stream), never before (`StreamState.save()`).
- **Streaming**: `serveStream` runs a `StreamState` whose `ReadLoop` pumps body reads; a heartbeat goroutine checks client disconnection every 0.5s. `cleanup()` saves the slot only if the stream completed normally (not cancelled mid-stream), then releases it.
- **No timeout on the stream body read**: CANNOT add a read deadline to `ReadLoop` — if it fires, the connection is closed prematurely while the backend is still processing (e.g., slow prompt prefill on large context). The only safe disconnection paths are: backend sends data/error, backend closes the connection, or the upstream client disconnects (heartbeat).
- **Don't cancel the request context while the body is streaming**: cancelling an in-flight response read leaves the pooled connection in a broken state and subsequent requests on it can hang. Timeouts on cleanup operations (closing the body after the stream completes, waiting on an already-cancelled task) are safe since no in-flight read exists.
- **Request pipeline**: `ChatHandler` (app.go) only parses, builds a `ProcRequest`, and submits it to the global `RequestDispatcher` (matcher.go), then blocks on `req.done`. One matcher goroutine scans each request (per-backend tokenize + disk/pending-slot cache scan) and enqueues it on the chosen backend's queue; the dispatcher's **pump** (same goroutine, runs after every arrival/overflow-drain/tick) dispatches each queue head to a fresh worker goroutine while `TryAcquire` succeeds — the slot pool is the concurrency limiter, so a backend with N free slots runs up to N requests in flight. On single-slot backends this degenerates to one at a time (current behavior). A first request for a model whose slot pool doesn't exist yet gets a worker that performs lazy discovery (refresh + `ServesModel` gate) instead of a pre-acquired slot.
- **Routing rule** (`SelectBackend()`, exported pure function): a cache-hit request goes to the hit backend while its *queue depth* (not in-flight count) stays below `CACHE_HIT_QUEUE_LIMIT` (default 2); otherwise it falls back to round-robin across the other matching backends (start position advanced past the first non-hit candidate at/after the model's RR index), and a P2P transfer of the best disk cache is triggered to the chosen backend. The saturated hit backend is only taken in a second pass when every other candidate is full.
- **Queue caps + overflow**: per-backend queues cap at `BACKEND_QUEUE_MAX` (default 5). A request whose chosen backend is full is parked in a global FIFO overflow queue and re-matched on every dispatcher wake (capacity frees, liveness change, or the 1s tick); head-of-line ordering is preserved (a non-routable head stops the drain pass).
- **Queue migration** (`migrateStale()`, matcher.go): the dispatcher's 1s monitor moves a queued request to an **idle** backend (empty queue, no in-flight work, live, serving the model) once it has waited `QUEUE_MIGRATION_AFTER` (default 120s). Idle targets come from the backend registry, not just backends with a queue. Guards: requests whose hit is not a transferable disk key (pending-slot-only or cold) need **2×** that age — migrating them trades a cheap future restore for an immediate full recompute; each request migrates **at most once**; at most one request per idle backend per pass. On migration the best disk key's P2P transfer to the target is triggered if it doesn't already live there (the target worker's `CACHE_TRANSFER_WAIT` bounded wait picks it up; recompute fallback if the transfer loses). Idle-only targeting means a migrated request starts immediately — no drain-speed estimation, no flapping.
- **Slot acquire retry**: only the lazy-discovery worker acquires itself (pump-pre-acquired workers never do); it retries forever on its *own* backend until a slot frees (growing backoff `SLOT_ACQUIRE_RETRY_BASE_SECONDS` × attempt, default base 5s). If the model is no longer served by that backend (empty pool + not in the registry after a refresh), the request is requeued to the global overflow queue for re-matching (`ServesModel()` gate) instead of failing. No 503 on slot exhaustion — a request only aborts on client disconnect / context cancellation.
- **Backends are proxy-exclusive**: the slot pool only tracks slots *we* assign. A backend also driven directly (outside the proxy) can have its slots double-booked, more so now that a backend runs multiple in-flight requests. Don't mix direct and proxied traffic on one backend.
- **P2P cache transfer** (transfer.go): async, deduped per (key, target), skipped when the target already holds the file. Four paths: agent→agent push (`/cache/transfer`), local→local direct copy, local→agent upload (`/cache/receive`), agent→local pull (`/cache/file`). Checkpoint sidecar files (`<key>.ckpt`, `<key>.ckpt.N`) are part of a complete entry when present: all four paths transfer them with the main file and any sidecar failure fails the transfer (make-space budgets include their bytes). Transfers **stream** end to end with fixed 16 MiB copy buffers (no full-file RAM buffering — multi-GB safe); one HTTP stream per file, one sender (distinct transfers run concurrently on their own goroutines) — a single TCP stream saturates LAN links, so parallelism is not worth the protocol complexity; upload/push senders set explicit `Content-Length` (a bare `*os.File` HTTP body would go chunked and hide the size). The receiving side makes space within its `cache_max_size_gb` budget (oldest-mtime first, sidecars `.ckpt*` removed with their key) — uploads (`/cache/receive`) self-enforce it from their content length; agent pushes and local paths ask the target explicitly. Both cache-agent implementations log per-file bytes + elapsed + MiB/s/GiB/s on receive/push/download; the proxy logs total bytes/sidecars/elapsed/rate on completion and records a `cache_transfer` metrics event (bytes, sidecars, ms). On success the meta is rewritten for the target and `AddTransferredEntry()` registers it in the target's ring. The worker on the target bounded-waits for an in-flight transfer before checking `CacheExists()`: at least `CACHE_TRANSFER_WAIT` (default 5s), scaled up by file size at `CACHE_TRANSFER_ASSUMED_MBPS` (default 100) so multi-GB entries get a realistic window — the wait still ends the instant the transfer completes.
- **Effective restore key**: the worker restores from the best *disk* candidate — directly when it lives on its backend, or after a P2P transfer lands. A pending-slot hit on the same backend suppresses the disk restore (the slot is already warm; `ShouldSkipRestore` no-ops it anyway).
- **Request contexts**: each request runs on a context derived from the client's `r.Context()`; `ProcRequest.finish()` cancels it. Requeued requests must NOT be finished by the worker that requeued them (their terminal state belongs to whoever completes them later) — cancelling early makes the overflow drain discard them as disconnected.
- **Slot timeout**: `SLOT_TIMEOUT` (default 30s) wraps `/slots/{id}?action=save|restore`. Separate from `REQUEST_TIMEOUT` (600s).
- **KV cache skip**: `ShouldSkipRestore()` in `slotmanager.go` checks the slot's tracked KV state before restoring. If the LCP ratio >= `KV_CACHE_SKIP_THRESHOLD` (default 0.9) — and the block-count difference is within `KV_CACHE_SKIP_MAX_BLOCK_DIFF_PCT` (default 0.1) — restore is skipped; llama.cpp appends to the existing cache. Only safe on single-slot backends.
- **Cache save skip**: `ShouldSaveCache()` in `config.go` skips save when the restore candidate ratio > `CACHE_SAVE_RATIO_THRESHOLD` (default 0.8) and no recompute happened — the intent is ring-budget protection: don't grow the cache (and risk evicting other entries) when disk already covers the content well enough; the unsaved tail is cheap to recompute. `ShouldSkipSaveHeuristic()` additionally skips near-max-context and classified-summarization requests. Skipped saves are tracked per slot (`MarkSaveSkipped`); if a later request is about to **clobber** that slot's content, `flushSkippedSave()` (processor.go) persists it first — but only when re-evaluating against *current* disk coverage would pass the same check (coverage degraded, e.g. ancestor evicted). While disk still covers it, the clobber is allowed to lose the tail (that's what the skip priced in).
- **Recompute detection**: `isRecompute()` in `app.go` with `recomputeThresholdRatio = 0.7` (the archived Python used 0.92 — Go is canonical). If `cached_tokens < expected * 0.7`, the restore was partial/useless. Increments `recompute_penalty` on the meta file (`IncrementRecomputePenalty()` in `kvmeta.go`), which degrades its candidate score.
- **Ring buffer eviction**: `EvictIfNeeded()` in `slotmanager.go` evicts expired entries (age-first) then LRU when total bytes exceed `cache_max_size_gb` (per-backend, default 25 GB). Only triggers on saves. `MakeSpaceFor(need)` reuses the same scoring to proactively free room before an incoming P2P transfer. `SaveAfter()` updates the existing ring entry in place when the key is already registered (e.g. right after a transfer of the same key) instead of appending a duplicate.
- **Slot refresh**: no cooldown throttle — the liveness loop re-queries slot counts on any state change, and a worker lazily refreshes when its backend's pool for the model is missing/empty (first request after discovery). On-demand discovery via `GET /slots` (non-router) or `GET /models` + child `/slots` (router mode). Falls back to 1 slot if discovery fails.
- **Meta reconciliation**: on startup, orphaned/corrupted `.meta.json` files are deleted via `GetKVMeta().Reconcile()` in `entry.go`.
- **Backend config validation**: `NewBackendManager()` in `backendmanager.go` — each backend MUST specify exactly one of `cache_dir` (local filesystem) or `agent_port` (remote cache-agent). Mutually exclusive. Violations panic at startup. A local (`cache_dir`) backend may additionally set `agent_serve_port` (never with `agent_port`, never equal to `PORT`, unique across backends): it makes the proxy serve an embedded cache-agent for that cache dir, so other proxies can use this backend as a normal `agent_port` backend and P2P-transfer caches to/from it (`StartEmbeddedAgentServers()` in `entry.go`).
- **BACKENDS default**: when empty, defaults to `[{"url":"http://127.0.0.1:8000","cache_dir":"/tmp/llama-cache"}]`.
- **Liveness checker**: `livenessLoop()` in `backendmanager.go` pings backends every 5s, triggers model discovery and slot refresh on state change, and wakes the dispatcher's overflow drain (`GetDispatcher().notify()`) so parked requests re-match against the new backend set. Liveness *events* are rate-limited per backend — `liveness_diag` records at most once per `LIVENESS_DIAG_RECORD_INTERVAL` (default 60s) per backend unless state changed (gate: `livenessDiagDue()`), and missing-models discovery re-triggers at most once per `MISSING_MODELS_RETRY_INTERVAL` (default 30s) per backend. Rationale: liveness events share the metrics ring buffer with request records, and unthrottled events (a busy backend flapping its health check) evict all request history.
- **Model resolution**: `GetDiscoveredModels()` — exact match → substring match (case-insensitive) → `"any"` matches all discovered models.
- **Archive note**: `archive/python/test_smoke.py` contains tests referencing backend keys with colons (e.g. `"10.0.0.1:8000"`) instead of the sanitized form. Go tests use sanitized keys — verify against `SanitizeBackendDir()` output.
- `.gitignore` covers `kv_meta/`, `venv/`, `__pycache__/`, `run-proxycache.ps1`, `uv.lock`, `*.exe~`, `cache-agent.exe`, and `proxycache.exe`.
- **Builds**: `./build-proxycache.sh` (repo root) builds the root Go module with Windows Go from WSL and outputs `proxycache.exe` to the repo root. `./build-cache-agent.sh` does the same for the cache-agent module.

## Commits

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

**Types**: `feat`, `fix`, `refactor`, `perf`, `chore`, `docs`, `test`

**Requirements**:
- The subject line must be detailed enough to understand **what changed** and **why** without reading the diff.
- A body is **required** for every commit — explain the reasoning, trade-offs, or context.
- Scope should be the file or component affected (e.g., `slotmanager`, `hashing`, `metrics`).

**Examples**:
```
fix(slotmanager): skip restore when KV cache LCP ratio exceeds threshold

Restoring to a slot with nearly-identical KV cache is wasteful. Added
LCP comparison against the slot's tracked KV state before restore,
skipping when ratio >= KV_CACHE_SKIP_THRESHOLD (0.9). Only safe on
single-slot backends where the slot is dedicated to one request at a time.
```

```
refactor(hashing): extract backend key sanitization to dedicated function

Colon-to-dash replacement was inline in multiple places. Centralized
into SanitizeBackendDir() to ensure consistency across meta I/O,
cache agent client, and test fixtures.
```

## Writing Skills

When creating or updating skills under `.opencode/skills/`, follow these guidelines:

- **Never use line numbers** — they change as code is modified. Reference functions by name and file location instead (e.g., `FindBestRestoreCandidate()` in `kvmeta.go`, not "line 284").
- **Focus on architecture and concepts** — document *why* things work, not *where* they are. Implementation details change; reasoning endures.
- **Use tables for function registries** — a table of key functions with their location and role is more durable than scattered references.
- **Call out gotchas explicitly** — unusual constraints, edge cases, and "only works with" conditions are the most valuable parts of a skill.
- **Mark planned vs current behavior** — when a skill documents work-in-progress, clearly separate what exists now from what's planned. When there's planned behavior, review the skill each time it is loaded to see if planned behavior has been implemented and rewrite the skill as needed.
