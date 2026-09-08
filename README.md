# proxycache

OpenAI-compatible proxy for `llama.cpp` that manages KV cache slots with disk save/restore, automatic model discovery, and cache-aware multi-backend routing.

Written in Go. The original Python implementation is preserved under `archive/python/` as a behavior reference only.

## Architecture

```
Client → proxycache (:8081) → llama.cpp (:8000)
                          ↘ llama-swap (:9292) → llama-server
```

proxycache sits between clients and one or more `llama.cpp` backends. It intercepts chat completion requests, looks up cached KV state on disk, routes requests to the optimal backend, and persists KV state after responses complete.

### Components

- **Model discovery** — automatically discovers models served by each backend via `GET /models` (router mode) or `GET /v1/models` (non-router). A liveness checker pings backends every 5s and triggers discovery on state changes.

- **Name resolution** — resolves client model names (e.g. "qwen3.6-32b") to canonical names discovered from backends. Exact match first, then case-insensitive substring match. The special name "any" matches all discovered models. In addition, chunk-level prefix aliases are auto-generated from pairwise model names (e.g. `unsloth/Qwen3.6-27B` matches both `unsloth/Qwen3.6-27B-MTP-GGUF:Q6_K` and `unsloth/Qwen3.6-27B-GGUF:Q5_K_S`). Prefixes shorter than `provider/model` (fewer than 1 `/`) are filtered out. Using a more generic name (e.g. "qwen3.6") matches multiple canonical models and distributes requests across all backends that serve them.

- **Request dispatcher** — a single goroutine scans each incoming request (per-backend tokenization plus disk and in-flight-slot cache scan across all matching backends), routes it, and enqueues it on the chosen backend's FIFO queue. The same goroutine's *pump* then dispatches each queue head to a fresh worker while a slot is free — the slot pool is the concurrency limiter, so a backend with N free slots runs up to N requests in flight (single-slot backends degrade to one at a time). A request whose chosen backend's queue is full (default 5) parks in a global FIFO overflow queue and is re-matched as capacity frees.

- **Cache-aware routing** — a request with a cache hit goes to the backend holding the cache while its queue depth stays below `CACHE_HIT_QUEUE_LIMIT` (default 2); otherwise it is distributed round-robin across the other matching backends, which triggers an async P2P transfer of the cache file so the chosen backend has it when the request runs. The proxy also scans in-flight (pending) slots for matches, letting subsequent requests reuse a slot's KV state before the cache is persisted to disk. Routing diagnostics capture the full per-backend scan trace for post-hoc analysis.

- **Queue migration** — a monitor (1s cadence) moves a request that has waited in a backend queue past `QUEUE_MIGRATION_AFTER` (default 120s) to a fully idle backend that serves the same model, so it starts immediately; the best disk cache is P2P-transferred to the target when needed. Each request migrates at most once, and requests without a transferable disk key (pending-slot-only or cold) need twice the age before they move — for them, migration trades a cheap future restore for an immediate full recompute.

- **P2P cache transfer** — cache files move between backends over four paths (agent→agent push, local↔local direct copy, local→agent upload, agent→local pull). Transfers are async and deduped per (key, target), skipped when the target already holds the file; the receiving side makes space within its own `cache_max_size_gb` budget. This is what makes "route to a backend that doesn't have the cache yet" safe: the serving worker bounded-waits (`CACHE_TRANSFER_WAIT`, default 5s) for an in-flight transfer before falling back to recompute.

- **Slot management** — per-model, per-backend slot pools with lazy discovery (first request for an unseen model triggers a refresh; if the backend no longer serves the model, the request is requeued for re-matching rather than failing). Free slots are preferred; when all are busy the request simply waits in the queue. Slots whose tracked KV cache already matches the incoming prompt skip restore entirely (the proxy re-validates this against the actually-acquired slot, falling back to a disk restore if the warm state went stale).

- **Cache lifecycle** — KV state is saved to disk after a response completes, but only when the new state is worth persisting: skipped for cancelled streams, skipped when the restore ratio >= `CACHE_SAVE_RATIO_THRESHOLD` with no recompute (disk already covers the content well enough; the unsaved tail is cheap to recompute and saving would risk evicting other ring entries), skipped when request tokens exceed `CACHE_SAVE_CTX_THRESHOLD` of the backend's max context (impending compaction), and skipped for requests classified as summarization. Skipped saves are tracked per slot and flushed to disk if a later request is about to clobber that slot's content and disk coverage has degraded in the meantime. Recompute is detected by comparing llama.cpp's `cached_tokens` against request length, covering both disk cache restores and pending slot hits; a useless restore increments the meta file's recompute penalty, degrading its future candidate score. A per-backend ring buffer evicts expired entries (age-first) then LRU when cache exceeds the configured size. Orphaned/corrupted metadata is reconciled on startup.

### Request flow

1. Client sends `POST /v1/chat/completions` with a model name (e.g. "qwen3.6-32b")
2. The dispatcher resolves the model name to canonical name(s) via discovered models and tokenizes the prompt on each matching backend
3. Scans disk cache files and in-flight slots on every matching backend for the best match
4. Routes the request: cache hit → the hit backend (while its queue is below the limit); otherwise round-robin across matching backends, triggering a P2P transfer of the best disk cache to the chosen one. A full queue parks the request in the global overflow queue instead.
5. The pump dispatches the request when a slot on that backend is free (one worker per free slot) and restores KV state from disk if available — including files that just arrived via P2P transfer
6. Forwards the request to llama.cpp with the canonical model name and pinned slot
7. Saves KV state to disk if the response completed normally and the new cache is worth persisting (see Cache lifecycle); skipped saves are remembered per slot and flushed if a later request is about to clobber that slot's content

The proxy supports both streaming (SSE) and non-streaming responses.

## Configuration

All config via environment variables (defaults in `config.go`). No `.env` file support. The binary ignores command-line flags — set the `PORT` env var to change the listen port.

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8081` | Proxy listen port |
| `BACKENDS` | `[]` | JSON array of backend configs (see below). Empty defaults to `[{"url":"http://127.0.0.1:8000","cache_dir":"/tmp/llama-cache"}]`. Each backend must specify exactly one of `cache_dir` or `agent_port`. |
| `BACKEND_MODE` | `llama-cpp` | `llama-cpp` or `llama-swap` (changes `/slots` URL paths) |
| `META_DIR` | `./kv_meta` | Local metadata directory (organized by backend subdirectories) |
| `WORDS_PER_BLOCK` | `100` | Words per block for LCP matching |
| `LCP_TH` | `0.2` | LCP similarity threshold for cache match (0–1) |
| `KV_CACHE_SKIP_THRESHOLD` | `0.9` | Skip restore if slot KV cache matches >= this ratio (and the block-count difference is within `KV_CACHE_SKIP_MAX_BLOCK_DIFF_PCT`) |
| `KV_CACHE_SKIP_MAX_BLOCK_DIFF_PCT` | `0.1` | Max relative block-count difference for a skip-restore to apply |
| `CACHE_SAVE_RATIO_THRESHOLD` | `0.8` | Skip cache save if restore ratio >= this and no recompute (ring-budget protection: don't grow the cache when disk already covers the content) |
| `CACHE_SAVE_CTX_THRESHOLD` | `0.7` | Skip cache save if request tokens >= this fraction of backend's max context (avoids saving cache that will be invalidated by compaction) |
| `SLOT_TIMEOUT` | `30` | Timeout for slot save/restore operations (seconds) |
| `SLOT_ACQUIRE_RETRY_BASE_SECONDS` | `5` | Base backoff for the lazy-discovery worker's slot acquire retries (grows with attempt count) |
| `REQUEST_TIMEOUT` | `600` | HTTP timeout to backend (seconds) |
| `CLIENT_RECREATE_INTERVAL` | `50` | Recreate the HTTP client after this many requests per backend to avoid stale connections |
| `MODEL_ID` | `llama.cpp` | Default model ID when client omits it |
| `DEFAULT_N_CTX` | `16384` | Fallback context length when backend doesn't report `n_ctx` |
| `BACKEND_QUEUE_MAX` | `5` | Max queued (not yet dispatched) requests per backend; overflow goes to the global FIFO queue |
| `CACHE_HIT_QUEUE_LIMIT` | `2` | Max queued requests routed to a cache-hit backend before falling back to round-robin + P2P transfer |
| `CACHE_TRANSFER_TIMEOUT` | `300` | Timeout for a single P2P transfer HTTP exchange (seconds) |
| `CACHE_TRANSFER_WAIT` | `5` | How long a serving worker waits for an in-flight P2P transfer of its cache key before proceeding without a restore (seconds) |
| `MATCH_SCAN_TIMEOUT` | `15` | Timeout for the matcher's tokenize/disk-scan phase per request (seconds) |
| `QUEUE_MIGRATION_AFTER` | `120` | How long a request may sit in a backend queue before migrating to an idle backend serving its model (seconds; 2x for requests without a transferable disk key) |
| `LIVENESS_DIAG_RECORD_INTERVAL` | `60` | Min seconds between `liveness_diag` metrics events per backend unless state changed |
| `MISSING_MODELS_RETRY_INTERVAL` | `30` | Min seconds between re-triggering model discovery for backends with missing models |
| `CACHE_HIT_WAIT_EMA_*` | see `config.go` | Legacy env vars (`INITIAL/MIN/MAX_TIMEOUT`, `ALPHA`) — slot-occupancy EMA bookkeeping is kept but no longer drives routing |
| `LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARN`, `ERROR`) |
| `METRICS_RETENTION` | `200` | Single ring buffer size for requests + diagnostic events |
| `DASHBOARD_ENABLED` | `true` | Enable the monitoring dashboard (`false`, `no`, `0` to disable) |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Main chat endpoint (proxied to backend) |
| `GET` | `/v1/models` | Returns discovered models with `n_ctx`, auto-generated chunk-level prefix aliases, plus `"any"` option |
| `GET` | `/metrics/dashboard` | Full dashboard data (backends, slots, cache, performance, requests, summarization stats) |
| `GET` | `/metrics/health` | Backend health: up/down, model info, slot counts |
| `GET` | `/metrics/slots` | Per-slot state: in_use, last_used, KV block count |
| `GET` | `/metrics/cache` | Per-backend cache utilization: ring size, bytes, utilization % |
| `GET` | `/metrics/requests` | Recent requests with full JSON (`?limit=N&offset=M`) |
| `GET` | `/metrics/performance` | Cache hit/mispredict/save rates, latency percentiles (`?model=X&backend=Y`) |
| `GET` | `/metrics/diagnostics` | Routing diagnostics (`?request_id=UUID`), liveness events (`?liveness=true`), unified timeline (`?timeline=true`) |
| `GET` | `/dashboard` | Monitoring dashboard HTML page |

## Monitoring

proxycache includes an in-memory metrics collector and a single-page HTML dashboard (zero external dependencies) for real-time monitoring.

### Metrics

The metrics collector uses a single ring buffer (default 200 entries, configurable via `METRICS_RETENTION`) that holds both request records and diagnostic events. Non-request events (backend liveness changes) are distinguished by an `event` field and filtered out of request-oriented queries.

Each request record includes:

- **Cache performance**: hit rate, mispredict rate (cache hit attempted but restore was partial/useless), utility rate, save rate, restore success rate
- **Latency**: avg, p50, p95, p99 percentiles
- **Per-model and per-backend breakdowns**
- **Routing diagnostics**: per-backend cache scan results (cache file ratio, pending slot ratios, unreachable status), best match ratio, selected backend, candidate list, whether a P2P transfer was triggered, `migrated_from` when the queue-migration monitor moved the request to an idle backend before it ran, and `cache_migrated` (true/false/null) indicating whether a migrated request's disk cache hit followed it to the target backend (restored or warm slot) or was lost to a recompute

Metrics are recorded in three phases: arrival (status=`incomplete`), routing decision (backend, slot, routing reason), and completion (latency, cache hit/miss, save status). Streaming requests record metrics in `StreamState.cleanup()` after the full response lifecycle.

Diagnostic events are recorded when backends change liveness state (up/down), capturing `state_changes` and `discovered_models` snapshots. The unified timeline (`GET /metrics/diagnostics?timeline=true`) preserves the chronological sequence of requests and events for post-mortem analysis.

### Dashboard

The dashboard at `/dashboard` provides a real-time view of:

- **Backend Health** — up/down status, discovered models, slot counts, last discovery time, plus per-backend request queues (in-flight ▶ cells + queued segments up to the cap) and the global overflow count
- **Cache Performance** — hit rate, mispredict rate, utility rate, save rate, restore success rate, latency percentiles
- **Cache Utilization** — per-backend cache ring size, total bytes, utilization percentage, cache directory
- **Slot Status** — per-model slot grid showing in-use (green), free (gray), and restoring (blue) states
- **Recent Requests** — last N requests with expandable full JSON payloads, prompt previews, and filters by type (hit/miss/recompute) and backend

The dashboard auto-refreshes (configurable 2/5/10/30s), supports dark mode, and has collapsible panels. It is served as a single HTML file with no external dependencies.

To disable the dashboard, set `DASHBOARD_ENABLED=false`.

## Multi-backend

Each backend is identified by a sanitized `host-port` key derived from its URL (colons replaced with dashes). Cache files are stored per-backend in `META_DIR/{backend_key}/` subdirectories, keyed by `sha256(canonical_name + '\n' + token_ids)`. Each backend manages its own slot pool and cache ring buffer independently.

### Router mode

When `llama.cpp` runs in router mode (`--models-preset`), proxycache auto-detects it: if `GET /slots` returns HTTP 400, it queries `GET /models` to find loaded child-process models, then queries `/slots` on each child's port. No explicit router-mode flag is needed.

### llama-swap

Set `BACKEND_MODE=llama-swap` and ensure `llama-swap` model configs include `--slot-save-path`:

```yaml
models:
  "my-model":
    cmd: "llama-server -m model.gguf --slot-save-path /path/to/kv-cache ..."
```

## Cache Management

Each backend can be configured with either `cache_dir` (local filesystem) or `agent_port` (remote cache-agent). These options are mutually exclusive. Each backend can also optionally specify `cache_max_size_gb` to control its individual cache ring buffer size (default: `25`).

These settings double as the transport for inter-backend P2P cache transfers: when routing sends a request to a backend that lacks its best cache, the proxy moves the file there using the agents' transfer/upload/fetch endpoints (or direct file copy between local backends).

### Local cache management

For backends on the same host, set `cache_dir` to the path matching llama.cpp's `--slot-save-path`:

```bash
BACKENDS='[{"url":"http://10.0.0.1:8000","cache_dir":"/var/kvcache","cache_max_size_gb":50}]'
```

### Cache Agent

For remote backends, use a lightweight Go HTTP server alongside each `llama.cpp` instance to manage cache files.

#### Building

```bash
./build-cache-agent.sh
```

Requires Go 1.21+. Produces a `cache-agent.exe` binary in the project root.

#### Running

```bash
./cache-agent.exe -cache-dir /var/kvcache -port 8082
```

#### Configuration

Add `agent_port` to the backend config in `BACKENDS`:

```bash
BACKENDS='[{"url":"http://10.0.0.1:8000","agent_port":8082}]'
```

When `agent_port` is set, cache operations use the agent's HTTP endpoints. When `cache_dir` is set, cache operations use the local filesystem directly. A backend without either has no cache management.

### Mixed configuration

You can mix both styles across backends:

```bash
BACKENDS='[
  {"url":"http://10.0.0.1:8000","cache_dir":"/var/kvcache/b1","cache_max_size_gb":50},
  {"url":"http://10.0.0.2:8000","agent_port":8082,"cache_max_size_gb":10},
  {"url":"http://10.0.0.3:8000"}
]'
```

The first backend uses local filesystem, the second uses the cache agent, and the third has no cache management.

## Quick Start

### 1. Start `llama.cpp`

```bash
llama-server -m ./model.gguf -np 4 --slot-save-path /var/kvcache --host 0.0.0.0 --port 8000 --swa-full
```

**Note:** For the most effective cache management, run llama.cpp with a single slot (`-np 1`) or with unified KV cache disabled (`-no-kvu`). Unified KV cache can cause slot-level cache restores to fail across requests due to fragmentation or contention inside the unified KV cache. Refer to your llama.cpp version's documentation for the appropriate flags.

### 2. Build and run the proxy

```bash
./build-proxycache.sh     # or: go build -o proxycache.exe ./cmd/proxycache
./proxycache.exe
```

The build script locates the Windows Go toolchain automatically (on WSL2 it uses `/mnt/c/Program Files/Go/bin/go.exe`) and emits `proxycache.exe` at the repo root. The binary binds `0.0.0.0:$PORT` — all configuration comes from environment variables; command-line flags are ignored.

Point clients at the proxy's `/v1/chat/completions` endpoint.
