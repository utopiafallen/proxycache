# proxycache

OpenAI-compatible proxy for `llama.cpp` that manages KV cache slots with disk save/restore and automatic cache cleanup. Compatible with `llama-swap` for multi-model routing.

## Why it's needed

`llama.cpp` provides the primitives — per-slot KV cache, `/slots/{id}?action=save|restore` API, and in-memory LRU eviction — but leaves cache lifecycle management to the operator. proxycache automates this:

- **Disk persistence** — saves KV state to disk so caches survive restarts and idle periods.
- **Cross-session matching** — scans `.meta.json` files on disk to find the best previously cached prompt, with a tunable `LCP_TH` threshold (vs. llama.cpp's hardcoded 50%).
- **Smart slot assignment** — picks an unused slot first, then falls back to LRU, protecting cached contexts from accidental overwrites.
- **Automatic cleanup** — ring buffer evicts expired entries (age-first) then LRU when total cache exceeds `CACHE_MAX_SIZE_GB`. Orphaned/corrupted metadata is reconciled on startup.
- **KV cache skip** — if a slot's current KV cache already matches the incoming prompt (LCP ratio >= `KV_CACHE_SKIP_THRESHOLD`, default 0.9), the restore is skipped entirely and llama.cpp appends to the existing cache.

Small requests (`< BIG_THRESHOLD_WORDS`, default 500 words) skip cache I/O entirely — the overhead of hashing, scanning meta files, and disk reads/writes exceeds any prefill savings.

## How it works

```
Client → proxycache (:8081) → llama.cpp (:8000)
```

1. **Request arrives** at `POST /v1/chat/completions`. The proxy strips message roles, concatenates content with `\n\n`, and hashes it into word-blocks (SHA256 per block, default 100 words/block).

2. **Cache lookup** (big requests only): `find_best_restore_candidate()` scans all `.meta.json` files matching the model, computes LCP ratio between request blocks and each cached entry, and picks the best match above `LCP_TH`.

3. **Slot acquisition**: `SlotManager` discovers available slots on-demand (lazy, with 300s cooldown per model/backend pair), then picks a free slot or the least-recently-used one. If the slot's tracked KV cache already matches the request (>= `KV_CACHE_SKIP_THRESHOLD`), no disk restore happens.

4. **Dispatch**: the proxy forwards the request to llama.cpp with `cache_prompt=true`, `n_keep=-1`, and the slot pinned via three fields (`slot_id`, `id_slot`, `_slot_id` in root, `options`, and query params).

5. **Response**:
   - **Streaming**: a background `reader` task reads raw SSE bytes → `asyncio.Queue` → `StreamingResponse`. The reader's `finally` always calls `save_after` + `write_meta` + `release`.
   - **Non-streaming**: the proxy waits for the full response, saves the slot to disk, writes the meta file, then releases the slot.

6. **Save** happens *after* the response completes, never before. Small requests skip save entirely.

## Quick Start

### 1. Start `llama.cpp`

```bash
llama-server -m ./model.gguf -np 4 --slot-save-path /var/kvcache --host 0.0.0.0 --port 8000 --swa-full
```

### 2. Run the proxy

```bash
pip install -r requirements.txt
python proxycache.py
# or: uvicorn app:app --host 0.0.0.0 --port 8081
```

Point clients at the proxy's `/v1/chat/completions` endpoint.

## Configuration

All config via environment variables (defaults in `config.py`). No `.env` file support.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_URL` | `http://127.0.0.1:8000` | Backend URL (used when `BACKENDS` is empty) |
| `PORT` | `8081` | Proxy listen port |
| `BACKENDS` | `[]` | JSON array `[{"url":"..."}]` — multi-backend support |
| `BACKEND_MODE` | `llama-cpp` | `llama-cpp` or `llama-swap` (changes `/slots` URL paths) |
| `META_DIR` | `./kv_meta` | Local metadata directory |
| `BIG_THRESHOLD_WORDS` | `500` | Min words to trigger cache restore/save |
| `WORDS_PER_BLOCK` | `100` | Words per block for LCP matching |
| `LCP_TH` | `0.6` | LCP similarity threshold for cache match (0–1) |
| `KV_CACHE_SKIP_THRESHOLD` | `0.9` | Skip restore if slot KV cache matches >= this ratio |
| `REQUEST_TIMEOUT` | `600` | HTTP timeout to backend (seconds) |
| `MODEL_ID` | `llama.cpp` | Default model ID |
| `CACHE_DIR` | — | `llama.cpp` `--slot-save-path` dir (required for cleanup) |
| `CACHE_MAX_AGE_HOURS` | `168` | Delete cache files older than this (0=disabled) |
| `CACHE_MAX_SIZE_GB` | `25` | Max total cache size in GB |
| `LOG_LEVEL` | `INFO` | Python logging level |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Main chat endpoint (proxied to backend) |
| `GET` | `/v1/models` | Proxied to first backend |

Slot counts are discovered on-demand via `GET /slots` (non-router mode) or `GET /models` + child `/slots` (router mode), with a 300s cooldown per (model, backend) pair. Falls back to 1 slot if discovery fails.

## Router mode

When `llama.cpp` runs in router mode (`--models-preset`), proxycache auto-detects it: if `GET /slots` returns HTTP 400, it falls back to `GET /models` to find loaded child-process models, then queries `/slots` on each child's port. Slots are tagged with `_router_model` and filtered by model name.

No explicit router-mode flag is needed — detection is automatic.

## llama-swap setup

```
Client → proxycache (:8081) → llama-swap (:9292) → llama-server (:PORT)
```

Set `BACKEND_MODE=llama-swap` and ensure `llama-swap` model configs include `--slot-save-path`:

```yaml
models:
  "my-model":
    cmd: "llama-server -m model.gguf --slot-save-path /path/to/kv-cache ..."
```

## Systemd service

`~/.config/systemd/user/proxycache.service`:

```ini
[Unit]
Description=ProxyCache for `llama.cpp` KV Cache Management
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/proxycache
Environment="LLAMA_URL=http://127.0.0.1:9292"
Environment="META_DIR=/path/to/proxycache-meta"
Environment="PORT=5000"
ExecStart=/path/to/proxycache/venv/bin/python proxycache.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload && systemctl --user enable --now proxycache
```

## Architecture

| File | Role |
|------|------|
| `proxycache.py` | 13-line uvicorn entry point — **not** where main logic lives |
| `app.py` | FastAPI app, routes, streaming pipeline, request handling |
| `config.py` | All config via env vars (no .env file) |
| `hashing.py` | Text → word-block hashing, LCP matching, meta I/O |
| `llama_client.py` | httpx client to llama.cpp; slot save/restore, router mode slot discovery |
| `slot_manager.py` | Per-model slot pools, ring buffer eviction, KV cache skip logic |
| `kv_meta/` | Per-cache `.meta.json` files (gitignored) |
