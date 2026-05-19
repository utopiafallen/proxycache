# proxycache

OpenAI-compatible proxy for llama.cpp that manages KV cache slots with smart slot selection, disk save/restore, and automatic cache cleanup.

## Commands

```powershell
# Run with default env vars
python proxycache.py

# Run via uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8081

# Local dev (sets env vars first)
.\run-proxycache.ps1
```

No tests, no linter, no typechecker configured.

## Architecture

| File | Role |
|------|------|
| `proxycache.py` | uvicorn entry point |
| `app.py` | FastAPI app, routes (`/v1/chat/completions`, `/v1/models`), streaming via background reader + `asyncio.Queue` |
| `config.py` | **All** config via env vars (no .env file) |
| `hashing.py` | Text → word-block hashing, LCP matching, meta file I/O, periodic cache cleanup |
| `llama_client.py` | httpx AsyncClient to llama.cpp; slot save/restore via `/slots/{id}?action=save\|restore` with `{"filename":..., "model":...}` JSON body |
| `slot_manager.py` | Slot pool: free → oldest (by LRU). `acquire_for_request` does optional restore before returning |
| `kv_meta/` | Per-cache `.meta.json` files (prefix blocks, model_id, timestamp) |

## Key env vars

`LLAMA_URL` / `N_SLOTS` / `PORT` / `BACKENDS` (JSON) / `META_DIR` / `BIG_THRESHOLD_WORDS` / `WORDS_PER_BLOCK` / `LCP_TH` / `CACHE_DIR` / `CACHE_MAX_AGE_HOURS` / `CACHE_MAX_SIZE_GB` / `CACHE_CLEANUP_INTERVAL_MINUTES` / `REQUEST_TIMEOUT` / `MODEL_ID` / `LOG_LEVEL` / `BACKEND_MODE`

See `config.py` for defaults.

## Conventions & quirks

- **Backend Requirement**: `llama.cpp` MUST be started with `--slot-save-path <dir>` for cache save/restore to function.
- **Cache key**: `sha256(model_id + "\n" + raw_prefix)` where `raw_prefix` concatenates message content (no roles) with double newlines
- **LCP blocks**: text split into N-word blocks (default 100), each SHA256-hashed; matching is longest-common-prefix of block hash sequences
- **Small requests** (below `BIG_THRESHOLD_WORDS`, default 500) skip cache save/restore entirely, routed to any free/oldest slot
- **Slot pinning** duplicated 3 ways in request body: root (`slot_id`, `id_slot`, `_slot_id`), `options` dict, and query params
- **Save on response**: cache is saved after the response completes (both stream and non-stream), never before
- **Slot acquire timeout**: 300s hardcoded in `app.py`, returns 503 if all slots busy
- **Cache cleanup**: periodic background task (default hourly) deletes by age (default 7d) then by total size (default 50GB) oldest-first
- **Streaming**: background `reader` task reads raw bytes from llama.cpp → `asyncio.Queue` → `StreamingResponse` generator; reader always handles save+release in its `finally` block
- **This is a fork** of `airnsk/proxycache` with llama-swap compatibility and auto cleanup; `/slots` calls pass `model` in JSON body
- `.gitignore` covers `kv_meta/`, `venv/`, `__pycache__/`

## Dependencies

FastAPI, uvicorn, httpx — listed in `requirements.txt`, Python 3.10+.
