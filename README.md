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

- **Name resolution** — resolves client model names (e.g. "qwen3.6-32b") to canonical names discovered from backends. Tries exact match first, then case-insensitive substring match, then "any" → all models. Using a more generic name (e.g. "qwen3.6") matches multiple canonical models and distributes requests across all backends that serve them.

- **Cache-first routing** — when multiple backends serve the same model, requests are routed to the backend that holds the matching cache file. If that backend's slots are busy, the proxy falls back to other backends with the same model.

- **Slot management** — per-model, per-backend slot pools with lazy discovery (300s cooldown on success, 30s on failure). Free slots are preferred; when none are available, the least-recently-used slot is reclaimed. Slots with existing KV cache that already matches the incoming prompt skip restore entirely.

- **Cache lifecycle** — KV state is saved to disk after each response completes. A per-backend ring buffer evicts expired entries (age-first) then LRU when cache exceeds the configured size. Orphaned/corrupted metadata is reconciled on startup.

- **Small request optimization** — requests under `BIG_THRESHOLD_WORDS` (default 500 words) skip cache I/O entirely, avoiding the overhead of hashing, scanning meta files, and disk reads/writes.

### Request flow

1. Client sends `POST /v1/chat/completions` with a model name (e.g. "qwen3.6-32b")
2. Proxy resolves the model name to a canonical name via discovered models
3. For large requests, scans cache files across all backends for matching prefixes
4. Builds an ordered backend list — cache backend first, then fallback backends
5. Acquires a slot (with lock retry across backends) and restores KV cache if available
6. Forwards the request to llama.cpp with the canonical model name and pinned slot
7. Saves KV state to disk after the response completes (streaming or non-streaming)

The proxy supports both streaming (SSE) and non-streaming responses. For streaming, a background reader task races socket reads against client disconnects; save only happens if the stream completes normally.

## Configuration

All config via environment variables (defaults in `config.py`). No `.env` file support.

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8081` | Proxy listen port |
| `BACKENDS` | `[]` | JSON array `[{"url":"..."}]` of backend URLs. Empty defaults to `http://127.0.0.1:8000`. |
| `BACKEND_MODE` | `llama-cpp` | `llama-cpp` or `llama-swap` (changes `/slots` URL paths) |
| `META_DIR` | `./kv_meta` | Local metadata directory (organized by backend subdirectories) |
| `CACHE_DIR` | — | `llama.cpp` `--slot-save-path` directory |
| `CACHE_MAX_SIZE_GB` | `25` | Max total cache size per backend in GB |
| `CACHE_MAX_AGE_HOURS` | `168` | Delete cache files older than this (0=disabled) |
| `BIG_THRESHOLD_WORDS` | `500` | Min words to trigger cache restore/save |
| `WORDS_PER_BLOCK` | `100` | Words per block for LCP matching |
| `LCP_TH` | `0.2` | LCP similarity threshold for cache match (0–1) |
| `KV_CACHE_SKIP_THRESHOLD` | `0.9` | Skip restore if slot KV cache matches >= this ratio |
| `SLOT_TIMEOUT` | `30` | Timeout for slot save/restore operations (seconds) |
| `REQUEST_TIMEOUT` | `600` | HTTP timeout to backend (seconds) |
| `MODEL_ID` | `llama.cpp` | Default model ID when client omits it |
| `REFRESH_COOLDOWN_SECONDS` | `300` | Cooldown between slot refreshes per (model, backend) |
| `DEFAULT_N_CTX` | `16384` | Fallback context length when backend doesn't report `n_ctx` |
| `LOG_LEVEL` | `INFO` | Python logging level |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Main chat endpoint (proxied to backend) |
| `GET` | `/v1/models` | Returns discovered models with `n_ctx`, plus `"any"` option |

Slot counts are discovered on-demand with a 300s cooldown per (model, backend) pair. Falls back to 1 slot if discovery fails.

## Multi-backend

Configure multiple backends via `BACKENDS`:

```bash
BACKENDS='[{"url":"http://10.0.0.1:8000"},{"url":"http://10.0.0.2:8000"}]'
```

Each backend is identified by a stable `host:port` key derived from its URL. Cache files are stored per-backend in `META_DIR/{backend_key}/` subdirectories, keyed by `sha256(canonical_name + '\n' + prefix)`. Each backend manages its own slot pool and cache ring buffer independently.

### Router mode

When `llama.cpp` runs in router mode (`--models-preset`), proxycache auto-detects it: if `GET /slots` returns HTTP 400, it queries `GET /models` to find loaded child-process models, then queries `/slots` on each child's port. No explicit router-mode flag is needed.

### llama-swap

Set `BACKEND_MODE=llama-swap` and ensure `llama-swap` model configs include `--slot-save-path`:

```yaml
models:
  "my-model":
    cmd: "llama-server -m model.gguf --slot-save-path /path/to/kv-cache ..."
```

## Cache Agent

When backends are on remote hosts, proxycache uses a lightweight Go HTTP server alongside each `llama.cpp` instance to delete cache files for eviction.

### Building

```bash
./build-cache-agent.sh
```

Requires Go 1.21+. Produces a `cache-agent.exe` binary in the project root.

### Running

```bash
./cache-agent.exe -cache-dir /var/kvcache -port 8082
```

### Configuration

Add `agent_port` to the backend config in `BACKENDS`:

```bash
BACKENDS='[{"url":"http://10.0.0.1:8000","agent_port":8082}]'
```

When `agent_port` is set, eviction uses the agent's `POST /cache/delete?key=<basename>` endpoint. When unset, proxycache falls back to local filesystem deletion.

## Quick Start

### 1. Start `llama.cpp`

```bash
llama-server -m ./model.gguf -np 4 --slot-save-path /var/kvcache --host 0.0.0.0 --port 8000 --swa-full
```

### 2. Run the proxy

```bash
uv sync
uv run python proxycache.py
# or: uvicorn app:app --host 0.0.0.0 --port 8081
```

Point clients at the proxy's `/v1/chat/completions` endpoint.

## Deploying

### Systemd service

`~/.config/systemd/user/proxycache.service`:

```ini
[Unit]
Description=ProxyCache for `llama.cpp` KV Cache Management
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/proxycache
Environment="META_DIR=/path/to/proxycache-meta"
Environment="PORT=5000"
Environment="BACKENDS=[{\"url\":\"http://127.0.0.1:8000\"}]"
ExecStart=/path/to/proxycache/venv/bin/python proxycache.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload && systemctl --user enable --now proxycache
```
