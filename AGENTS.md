# proxycache

OpenAI-compatible proxy for llama.cpp that manages KV cache slots with disk save/restore and automatic cache cleanup. Compatible with `llama-swap` for multi-model routing.

## Commands

```bash
# Install deps
pip install -r requirements.txt

# Run (default env vars from config.py)
python proxycache.py

# Or via uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8081

# Local Windows dev (sets env vars, uses `uv run`)
.\run-proxycache.ps1          # gitignored — local config only
```

**No tests, no linter, no typechecker.**

## Architecture

| File | Role |
|------|------|
| `proxycache.py` | 13-line uvicorn wrapper — **not** where main logic lives |
| `app.py` | FastAPI app, routes (`/v1/chat/completions`, `/v1/models`), streaming pipeline |
| `config.py` | All config via env vars (no .env file) |
| `hashing.py` | Text → word-block hashing, LCP matching, meta I/O, cache cleanup |
| `llama_client.py` | httpx AsyncClient to llama.cpp; slot save/restore via `/slots/{id}` |
| `slot_manager.py` | Slot pool: free → oldest (LRU). `acquire_for_request` does optional restore |
| `kv_meta/` | Per-cache `.meta.json` files (prefix blocks, model_id, timestamps) |

## Config (env vars only)

`config.py` has all defaults. Full table in `README.md`. Key relationships:

- `BACKENDS` (JSON array) **overrides** `LLAMA_URL` + `N_SLOTS` when set
- `BACKEND_MODE` = `"llama-cpp"` (default) or `"llama-swap"` — changes `/slots` URL paths
- `CACHE_DIR` must point to llama.cpp's `--slot-save-path` for cleanup to work

## Gotchas

- **llama.cpp prerequisite**: MUST be started with `--slot-save-path <dir>` or cache save/restore will fail silently.
- **Cache key**: `sha256(model_id + "\n" + raw_prefix)` where `raw_prefix` strips roles and concatenates message content with `\n\n`.
- **LCP matching**: text split into N-word blocks (default 100), each SHA256-hashed; longest-common-prefix of block hash sequences determines restore candidates.
- **Small requests** (`< BIG_THRESHOLD_WORDS`, default 500) skip cache save/restore entirely — routed to free/oldest slot with no disk I/O.
- **Slot pinning** is duplicated 3 ways in every request body: root (`slot_id`, `id_slot`, `_slot_id`), `options` dict, and query params.
- **Save happens after response** completes (both stream and non-stream), never before.
- **Slot acquire timeout**: 300s hardcoded (`app.py:43`). Returns 503 if all slots busy.
- **Streaming**: a background `reader` task reads raw SSE bytes → `asyncio.Queue` → `StreamingResponse`. The `reader`'s `finally` block always calls `save_after` + `write_meta` + `release`.
- **Meta reconciliation**: on startup and during cleanup, orphaned/corrupted `.meta.json` files are deleted.
- **Fork of** `airnsk/proxycache` with llama-swap compatibility and auto cleanup.
- `.gitignore` covers `kv_meta/`, `venv/`, `__pycache__/`, and `run-proxycache.ps1` (local dev script, not tracked).
