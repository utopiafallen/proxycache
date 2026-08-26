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
./proxycache.exe                             # run (all config via env vars, see config.go)
```

**No linter, no typechecker, no test framework** — stdlib `testing` only.

## Architecture

| File | Role |
|------|------|
| `entry.go` | Library entry point (`Main()`): slot init from disk, meta reconciliation, liveness start, HTTP mux, graceful shutdown |
| `cmd/proxycache/main.go` | Thin `package main` wrapper that calls `proxycache.Main()`; the only `package main` in the root module |
| `tests/` | External test package (`package tests`) importing `proxycache`; all test files live here |
| `app.go` | HTTP routes, request handling, streaming pipeline (`serveStream` → `StreamState`) |
| `backendmanager.go` | `BackendManager`: backend registry, LlamaClient/CacheAgentClient instances, model-to-backend mapping, liveness checker |
| `config.go` | All config via env vars (no .env file), save heuristics, request classification |
| `hashing.go` | Hashing primitives: block hashes from tokens, LCP matching, cache key generation, backend key sanitization |
| `kvmeta.go` | All meta I/O: read/write/delete `.meta.json` (atomic via temp+rename), scan, reconcile, find restore candidates |
| `llamaclient.go` | HTTP client to llama.cpp; slot save/restore, router-mode slot discovery |
| `slotmanager.go` | Per-model slot pools (`BackendSlotManager`), ring buffer eviction, KV cache skip logic, cache hit wait queue |
| `cacheagent.go` | `CacheAgentClient` (remote cache file deletion) + embeddable `AgentServer` (not wired into the proxy; the standalone binary in `cache-agent/` is what runs in production) |
| `metrics.go` | In-memory metrics collector: single ring buffer (retention 200) holds request records and diagnostic events; three-phase recording (arrival → routing → completion); `GetRequests()` filters out events; `GetTimeline()` returns unified view |
| `logger.go` | Levelled logging (`LOG_LEVEL`) |
| `dashboard.html` | Metrics dashboard served at `/dashboard` (toggle via `DASHBOARD_ENABLED`), embedded into the binary |
| `cache-agent/` | Separate Go module: standalone cache-agent binary (lightweight HTTP server for remote cache deletion) |
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
- **Slot acquire retry**: retries forever until a slot frees (growing backoff 5s, 10s, ...). No 503 on slot exhaustion — a request only aborts on client disconnect / context cancellation.
- **Slot timeout**: `SLOT_TIMEOUT` (default 30s) wraps `/slots/{id}?action=save|restore`. Separate from `REQUEST_TIMEOUT` (600s).
- **Fallback sorting**: cache-miss requests sort `candidateBackends` (in `chatHandler`) by composite score: `(cache_ratio, ring_size, latency_ema, last_used)` — prefer backends where new cache is most unique (low ratio), with room (few entries), fastest (low EMA latency), and least recently used. Cache ratio from the routing scan; ring size from `GetRingSize()`; latency EMA from `GetBackendLatencyEMA()` (updated after each request completion); last_used from `GetBackendLastUsed()`.
- **Cache hit wait queue**: when a cache-hit request's backend has no free slots, Phase 0 polls every 5s up to an EMA-derived timeout (`CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT`, default 30s). On slot release, the EMA is updated with actual occupancy duration. Clamped between `CACHE_HIT_WAIT_EMA_MIN_TIMEOUT` (10s) and `CACHE_HIT_WAIT_EMA_MAX_TIMEOUT` (300s). Max concurrent waiters per backend: `CACHE_HIT_WAIT_MAX_PENDING_REQS` (3). Falls through to normal retry loop on timeout.
- **KV cache skip**: `ShouldSkipRestore()` in `slotmanager.go` checks the slot's tracked KV state before restoring. If the LCP ratio >= `KV_CACHE_SKIP_THRESHOLD` (default 0.9) — and the block-count difference is within `KV_CACHE_SKIP_MAX_BLOCK_DIFF_PCT` (default 0.1) — restore is skipped; llama.cpp appends to the existing cache. Only safe on single-slot backends.
- **Cache save skip**: `ShouldSaveCache()` in `config.go` skips save when the restore candidate ratio > `CACHE_SAVE_RATIO_THRESHOLD` (default 0.8) and no recompute happened — avoids saving redundant cache entries. `ShouldSkipSaveHeuristic()` additionally skips near-max-context and classified-summarization requests.
- **Recompute detection**: `isRecompute()` in `app.go` with `recomputeThresholdRatio = 0.7` (the archived Python used 0.92 — Go is canonical). If `cached_tokens < expected * 0.7`, the restore was partial/useless. Increments `recompute_penalty` on the meta file (`IncrementRecomputePenalty()` in `kvmeta.go`), which degrades its candidate score.
- **Ring buffer eviction**: `EvictIfNeeded()` in `slotmanager.go` evicts expired entries (age-first) then LRU when total bytes exceed `cache_max_size_gb` (per-backend, default 25 GB). Only triggers on saves.
- **Slot refresh**: no cooldown throttle — every request triggers a refresh. On-demand discovery via `GET /slots` (non-router) or `GET /models` + child `/slots` (router mode). Falls back to 1 slot if discovery fails.
- **Meta reconciliation**: on startup, orphaned/corrupted `.meta.json` files are deleted via `GetKVMeta().Reconcile()` in `entry.go`.
- **Backend config validation**: `initBackends()` in `backendmanager.go` — each backend MUST specify exactly one of `cache_dir` (local filesystem) or `agent_port` (remote cache-agent). Mutually exclusive. Violations panic at startup.
- **BACKENDS default**: when empty, defaults to `[{"url":"http://127.0.0.1:8000","cache_dir":"/tmp/llama-cache"}]`.
- **Liveness checker**: `livenessLoop()` in `backendmanager.go` pings backends every 5s, triggers model discovery and slot refresh on state change. Liveness *events* are rate-limited per backend — `liveness_diag` records at most once per `LIVENESS_DIAG_RECORD_INTERVAL` (default 60s) per backend unless state changed (gate: `livenessDiagDue()`), and missing-models discovery re-triggers at most once per `MISSING_MODELS_RETRY_INTERVAL` (default 30s) per backend. Rationale: liveness events share the metrics ring buffer with request records, and unthrottled events (a busy backend flapping its health check) evict all request history.
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
