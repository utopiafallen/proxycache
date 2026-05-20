<img width="1000"  alt="image_" src="https://github.com/user-attachments/assets/0d966dde-f1d8-432f-bad0-aa79a5ccf396" />

# proxycache

OpenAI-compatible proxy for `llama.cpp` that manages KV cache slots with disk save/restore and automatic cache cleanup. Compatible with `llama-swap` for multi-model routing.

## Why it's needed

`llama.cpp` provides all the primitives: per-slot KV cache, `/slots/{id}?action=save|restore` API, and LRU eviction. proxycache automates their management so you don't have to — it persists caches to disk, matches prompts against historical caches, assigns slots intelligently, and cleans up old files, all without modifying `llama.cpp`.

## How it works

`llama.cpp` provides in-memory KV cache with per-slot prefix matching (50% threshold) and LRU eviction. proxycache sits between clients and `llama.cpp` and adds:

- **Disk persistence** — calls `llama.cpp`'s `/slots/{id}?action=save|restore` to persist KV state to disk so caches survive restarts and idle periods.
- **Configurable prefix matching** — `llama.cpp`'s 50% similarity threshold is hardcoded. proxycache uses tunable `LCP_TH` and scans `.meta.json` files on disk, so it can match against any previously cached prompt — not just ones currently held in active slots.
- **Smart slot assignment** — picks an unused slot first, then falls back to least-recently-used, protecting cached contexts from accidental overwrites.
- **Automatic cleanup** — background task deletes old/expired cache files and reconciles orphaned metadata. `llama.cpp` has no cache eviction.
- **Multi-backend routing** — supports multiple backends and `llama-swap` mode for model routing. `llama.cpp` is a single server.

Small requests (< `BIG_THRESHOLD_WORDS`) skip cache I/O entirely — the overhead of hashing, scanning meta files, and disk reads/writes exceeds any prefill savings on short prompts.

## Quick Start

### 1. Start `llama.cpp`

```bash
llama-server -m ./model.gguf -np 4 --slot-save-path /var/kvcache --host 0.0.0.0 --port 8080 --swa-full
```

### 2. Run the proxy

```bash
pip install -r requirements.txt
python proxycache.py
# or: uvicorn app:app --host 0.0.0.0 --port 8081
```

Point clients at the proxy's `/v1/chat/completions` endpoint.

## Configuration

All config via environment variables (defaults in `config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_URL` | `http://127.0.0.1:8000` | Backend URL |
| `N_SLOTS` | `2` | Slots per backend |
| `PORT` | `8081` | Proxy listen port |
| `BACKENDS` | — | JSON array `[{"url":"...", "n_slots":N}]` — overrides `LLAMA_URL`+`N_SLOTS` |
| `BACKEND_MODE` | `llama-cpp` | `llama-cpp` or `llama-swap` (changes `/slots` URL paths) |
| `META_DIR` | `./kv_meta` | Local metadata directory |
| `BIG_THRESHOLD_WORDS` | `500` | Min words to trigger cache restore/save |
| `WORDS_PER_BLOCK` | `100` | Words per block for LCP matching |
| `LCP_TH` | `0.6` | LCP similarity threshold (0–1) |
| `REQUEST_TIMEOUT` | `600` | HTTP timeout (seconds) |
| `MODEL_ID` | `llama.cpp` | Default model ID |
| `CACHE_DIR` | — | `llama.cpp` `--slot-save-path` dir (required for cleanup) |
| `CACHE_MAX_AGE_HOURS` | `168` | Delete files older than this (0=disabled) |
| `CACHE_MAX_SIZE_GB` | `25` | Max total cache size |
| `CACHE_CLEANUP_INTERVAL_MINUTES` | `30` | Cleanup check interval |

## llama-swap setup

```
Client → proxycache (:8081) → llama-swap (:9292) → llama-server (:PORT)
```

Ensure `llama-swap` model configs include `--slot-save-path`:

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
Environment="N_SLOTS=1"
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
