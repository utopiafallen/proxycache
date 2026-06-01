# proxycache

OpenAI-compatible proxy for llama.cpp that manages KV cache slots with disk save/restore and automatic cache cleanup. Compatible with `llama-swap` for multi-model routing.

## Environment

**Running inside WSL2** — use `uv` from Windows for package management. Do NOT install Python packages via pip/apt inside WSL2 as it pollutes the system Python. Use `uv pip` or `uv run` from the Windows side or WSL2 with `uv`'s venv support.

## Commands

```bash
uv sync                                        # install deps (uses pyproject.toml)
python proxycache.py                           # run (default env vars from config.py)
uvicorn app:app --host 0.0.0.0 --port 8081     # or via uvicorn directly
python test_smoke.py                           # smoke tests (no framework, uses unittest.mock)
```

**No linter, no typechecker, no test framework.**

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

## Gotchas

- **llama.cpp prerequisite**: MUST be started with `--slot-save-path <dir>`. Cache save/restore fails silently without it.
- **Config**: all env vars only — `config.py` has defaults. No `.env` file support.
- **Cache key**: `sha256(model_id + "\n" + raw_prefix)` where `raw_prefix` strips message roles and concatenates content with `\n\n`.
- **Slot pinning** is duplicated 3 ways in every request body: root (`slot_id`, `id_slot`, `_slot_id`), `options` dict, and query params.
- **Save happens after response** completes (both stream and non-stream), never before.
- **Streaming**: a background `reader` task races socket reads against a disconnect event → `asyncio.Queue`. A heartbeat task checks `is_disconnected()` every 0.5s. `stream()`'s `finally` calls `_cleanup()` which saves the slot (only if `_stream_complete` is True — i.e. the stream finished normally, not if cancelled mid-stream), releases it, and puts a sentinel in the queue.
- **Slot acquire timeout**: 60s hardcoded (`ACQUIRE_TIMEOUT` in app.py). Returns 503 if all slots busy.
- **Slot timeout**: `SLOT_TIMEOUT` (default 30s) wraps `/slots/{id}?action=save|restore` calls. Separate from `REQUEST_TIMEOUT` (600s) so slot operations fail fast on dead backends instead of blocking for 10min.
- **Adaptive cooldown**: after a failed `refresh_slots()`, slot discovery retries every 30s instead of 300s, so requests recover faster when the backend comes back up.
- **Small requests** (`< BIG_THRESHOLD_WORDS`, default 500 words) skip cache save/restore entirely — routed to free/oldest slot with no disk I/O.
- **KV cache skip**: `acquire_for_request` checks `_slot_kv_state` before restoring. If the slot's tracked KV cache blocks have LCP ratio >= `KV_CACHE_SKIP_THRESHOLD` (default 0.9), the restore is skipped — llama.cpp appends to existing cache. State updates after every save (from `blocks` param) and after every restore (from meta file).
- **Ring buffer eviction**: `SlotManager` evicts expired entries (age-first) then LRU when `_total_bytes > CACHE_MAX_SIZE_GB`. Eviction only triggers on saves; stale entries accumulate if no saves happen.
- **Slot refresh cooldown**: 300s per (model, backend) pair on success, 30s on failure. No startup discovery or periodic refresh — slots discovered on-demand via `GET /slots` (non-router) or `GET /models` + child `/slots` (router mode). Falls back to 1 slot if discovery fails.
- **Meta reconciliation**: on startup, orphaned/corrupted `.meta.json` files are deleted via `reconcile_meta()`.
- `.gitignore` covers `kv_meta/`, `venv/`, `__pycache__/`, `run-proxycache.ps1`, and `uv.lock`.
