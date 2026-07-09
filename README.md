# proxycache

OpenAI-compatible proxy for `llama.cpp` that manages KV cache slots with disk save/restore, automatic model discovery, and cache-aware multi-backend routing.

## Architecture

```
Client → proxycache (:8081) → llama.cpp (:8000)
                          ↘ llama-swap (:9292) → llama-server
```

proxycache sits between clients and one or more `llama.cpp` backends. It intercepts chat completion requests, looks up cached KV state on disk, routes requests to the optimal backend, and persists KV state after responses complete.

### Components

- **Model discovery** — automatically discovers models served by each backend via `GET /models` (router mode) or `GET /v1/models` (non-router). A liveness checker pings backends every 5s and triggers discovery on state changes.

- **Name resolution** — resolves client model names (e.g. "qwen3.6-32b") to canonical names discovered from backends. Exact match first, then case-insensitive substring match. The special name "any" matches all discovered models. Using a more generic name (e.g. "qwen3.6") matches multiple canonical models and distributes requests across all backends that serve them.

- **Cache-first routing** — when multiple backends serve the same model, requests are routed to the backend that holds the matching cache file. The proxy also scans in-flight (pending) slots for matches during the save window, allowing subsequent requests to reuse slots before the cache is persisted. If the preferred backend's slots are busy, the proxy falls back to other backends. Routing diagnostics capture the full per-backend scan trace for post-hoc analysis.

- **Slot management** — per-model, per-backend slot pools with lazy discovery. Free slots are preferred; when none are available, the least-recently-used slot is reclaimed. For cache-hit requests whose backend is busy, the proxy waits on a per-backend semaphore (up to an EMA-derived timeout) before falling back. Slots with existing KV cache that already matches the incoming prompt skip restore entirely.

- **Cache lifecycle** — KV state is saved to disk after a response completes, but only when the new state is worth persisting: skipped for cancelled streams, and skipped when the serving backend's cache ratio >= threshold with no recompute. Recompute is detected by comparing llama.cpp's `cached_tokens` against request length, covering both disk cache restores and pending slot hits. A per-backend ring buffer evicts expired entries (age-first) then LRU when cache exceeds the configured size. Orphaned/corrupted metadata is reconciled on startup.

### Request flow

1. Client sends `POST /v1/chat/completions` with a model name (e.g. "qwen3.6-32b")
2. Proxy resolves the model name to a canonical name via discovered models
3. Scans cache files across all backends for matching prefixes
4. Builds an ordered backend list — cache backend first, then fallback backends
5. If the cache backend is busy, waits briefly for a slot to free up (Phase 0 wait queue)
6. Acquires a slot (with lock retry across backends) and restores KV cache if available
7. Forwards the request to llama.cpp with the canonical model name and pinned slot
8. Saves KV state to disk if the response completed normally and the new cache is worth persisting (skipped for cancelled streams or when existing cache was already a good match)

The proxy supports both streaming (SSE) and non-streaming responses.

## Configuration

All config via environment variables (defaults in `config.py`). No `.env` file support.

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8081` | Proxy listen port |
| `BACKENDS` | `[]` | JSON array of backend configs (see below). Empty defaults to `[{"url":"http://127.0.0.1:8000","cache_dir":"/tmp/llama-cache"}]`. Each backend must specify exactly one of `cache_dir` or `agent_port`. |
| `BACKEND_MODE` | `llama-cpp` | `llama-cpp` or `llama-swap` (changes `/slots` URL paths) |
| `META_DIR` | `./kv_meta` | Local metadata directory (organized by backend subdirectories) |
| `CACHE_MAX_AGE_HOURS` | `168` | Delete cache files older than this (0=disabled) |
| `WORDS_PER_BLOCK` | `100` | Words per block for LCP matching |
| `LCP_TH` | `0.2` | LCP similarity threshold for cache match (0–1) |
| `KV_CACHE_SKIP_THRESHOLD` | `0.9` | Skip restore if slot KV cache matches >= this ratio |
| `CACHE_SAVE_RATIO_THRESHOLD` | `0.8` | Skip cache save if restore ratio >= this (avoids overwriting good cache) |
| `SLOT_TIMEOUT` | `30` | Timeout for slot save/restore operations (seconds) |
| `REQUEST_TIMEOUT` | `600` | HTTP timeout to backend (seconds) |
| `MODEL_ID` | `llama.cpp` | Default model ID when client omits it |
| `CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT` | `30` | Initial EMA timeout for cache-hit wait queue (seconds) |
| `CACHE_HIT_WAIT_EMA_MIN_TIMEOUT` | `10` | Minimum wait queue timeout (seconds) |
| `CACHE_HIT_WAIT_EMA_MAX_TIMEOUT` | `300` | Maximum wait queue timeout (seconds) |
| `CACHE_HIT_WAIT_EMA_ALPHA` | `0.2` | EMA smoothing factor (0–1) |
| `CACHE_HIT_WAIT_MAX_PENDING_REQS` | `3` | Max concurrent waiters per backend |
| `DEFAULT_N_CTX` | `16384` | Fallback context length when backend doesn't report `n_ctx` |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `METRICS_RETENTION` | `200` | Single ring buffer size for requests + diagnostic events |
| `DASHBOARD_ENABLED` | `true` | Enable the monitoring dashboard (`false`, `no`, `0` to disable) |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Main chat endpoint (proxied to backend) |
| `GET` | `/v1/models` | Returns discovered models with `n_ctx`, plus `"any"` option |
| `GET` | `/metrics/summary` | Full metrics summary (backends, slots, cache, performance, requests) |
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
- **Routing diagnostics**: per-backend cache scan results (cache file ratio, pending slot ratios, unreachable status), best match ratio, selected backend, and candidate fallback list

Metrics are recorded in three phases: arrival (status=`incomplete`), routing decision (backend, slot, routing reason), and completion (latency, cache hit/miss, save status). Streaming requests record metrics in `StreamReader._cleanup()` after the full response lifecycle.

Diagnostic events are recorded when backends change liveness state (up/down), capturing `state_changes` and `discovered_models` snapshots. The unified timeline (`GET /metrics/diagnostics?timeline=true`) preserves the chronological sequence of requests and events for post-mortem analysis.

### Dashboard

The dashboard at `/dashboard` provides a real-time view of:

- **Backend Health** — up/down status, discovered models, slot counts, last discovery time
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

### 2. Run the proxy

```bash
uv sync
uv run python proxycache.py
# or: uvicorn app:app --host 0.0.0.0 --port 8081
```

Point clients at the proxy's `/v1/chat/completions` endpoint.
