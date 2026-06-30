# proxycache

OpenAI-compatible proxy for `llama.cpp` KV cache slot management with disk save/restore.

## Environment

**WSL2** — use `uv` from Windows for package management. Do NOT install Python packages via pip/apt inside WSL2.

## Commands

```bash
uv sync                                        # install deps (pyproject.toml)
python proxycache.py                           # run (env vars from config.py)
uvicorn app:app --host 0.0.0.0 --port 8081     # or via uvicorn directly
python test_smoke.py                           # smoke tests (no framework, uses unittest.mock)
```

**No linter, no typechecker, no test framework.**

## Architecture

| File | Role |
|------|------|
| `proxycache.py` | 13-line uvicorn entry point — **not** where main logic lives |
| `app.py` | FastAPI app, routes, streaming pipeline, request handling |
| `backend_manager.py` | Singleton: backend registry, LlamaClient/CacheAgentClient instances, model-to-backend mapping, refresh cooldowns |
| `config.py` | All config via env vars (no .env file) |
| `hashing.py` | Text → word-block hashing, LCP matching, meta I/O, reconciliation |
| `llama_client.py` | httpx client to llama.cpp; slot save/restore, router mode slot discovery |
| `slot_manager.py` | Per-model slot pools, ring buffer eviction, KV cache skip logic, cache hit wait queue |
| `cache_agent_client.py` | HTTP client for remote cache file deletion |
| `cache-agent/` | Go cache agent (lightweight HTTP server for remote cache deletion) |
| `kv_meta/` | Per-cache `.meta.json` files (gitignored) |

## Gotchas

- **llama.cpp prerequisite**: MUST start with `--slot-save-path <dir>`. Cache save/restore fails silently without it.
- **Config**: all env vars only — `config.py` has defaults. No `.env` file support.
- **Cache key**: `sha256(canonical_name + '\n' + ','.join(token_ids))` — based on token IDs, not raw text.
- **Backend keys**: sanitized `host-port` strings (colons replaced with dashes, e.g. `"10-0-0-1-8000"`), NOT raw `host:port`. Used as directory names under `META_DIR/`.
- **Slot pinning** is duplicated 3 ways in every request body: root (`slot_id`, `id_slot`, `_slot_id`), `options` dict, and query params.
- **Save happens after response** completes (both stream and non-stream), never before.
- **Streaming**: background `reader` task races socket reads against disconnect event → `asyncio.Queue`. Heartbeat checks `is_disconnected()` every 0.5s. `stream()`'s `finally` calls `_cleanup()` which saves the slot only if `_stream_complete` is True (stream finished normally, not cancelled mid-stream), then releases it.
- **Slot acquire timeout**: 60s hardcoded (`ACQUIRE_TIMEOUT` in app.py). Returns 503 if all slots busy.
- **Slot timeout**: `SLOT_TIMEOUT` (default 30s) wraps `/slots/{id}?action=save|restore`. Separate from `REQUEST_TIMEOUT` (600s).
- **Cache hit wait queue**: when a cache-hit request's backend has no free slots, Phase 0 waits up to an EMA-derived timeout (`CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT`, default 30s) on a per-backend semaphore. On slot release, the EMA is updated with actual occupancy duration and one waiter is woken. Clamped between `CACHE_HIT_WAIT_EMA_MIN_TIMEOUT` (10s) and `CACHE_HIT_WAIT_EMA_MAX_TIMEOUT` (300s). Max concurrent waiters per backend: `CACHE_HIT_WAIT_MAX_PENDING_REQS` (3). Falls through to normal retry loop on timeout.
- **KV cache skip**: `acquire_for_request` checks `_slot_kv_state` before restoring. If slot's tracked KV cache blocks have LCP ratio >= `KV_CACHE_SKIP_THRESHOLD` (default 0.9), restore is skipped — llama.cpp appends to existing cache. Only safe on single-slot backends.
- **Ring buffer eviction**: `SlotManager` evicts expired entries (age-first) then LRU when `_total_bytes > backend.cache_max_size_gb * 1024**3`. Per-backend, defaults to 25 GB. Only triggers on saves.
- **Slot refresh cooldown**: 300s per (model, backend) pair on success, 30s on failure. On-demand discovery via `GET /slots` (non-router) or `GET /models` + child `/slots` (router mode). Falls back to 1 slot if discovery fails.
- **Meta reconciliation**: on startup, orphaned/corrupted `.meta.json` files are deleted via `reconcile_meta()`.
- **Backend config validation**: each backend MUST specify exactly one of `cache_dir` (local filesystem) or `agent_port` (remote cache-agent). Mutually exclusive. Missing either raises `ValueError` at startup.
- **BACKENDS default**: when empty, defaults to `[{"url":"http://127.0.0.1:8000","cache_dir":"/tmp/llama-cache"}]`.
- `.gitignore` covers `kv_meta/`, `venv/`, `__pycache__/`, `run-proxycache.ps1`, `uv.lock`, and `cache-agent.exe`.

## Writing Skills

When creating or updating skills under `.opencode/skills/`, follow these guidelines:

- **Never use line numbers** — they change as code is modified. Reference functions by name and file location instead (e.g., `find_best_restore_candidate()` in `kv_meta_manager.py` not "line 284").
- **Focus on architecture and concepts** — document *why* things work, not *where* they are. Implementation details change; reasoning endures.
- **Use tables for function registries** — a table of key functions with their location and role is more durable than scattered references.
- **Call out gotchas explicitly** — unusual constraints, edge cases, and "only works with" conditions are the most valuable parts of a skill.
- **Mark planned vs current behavior** — when a skill documents work-in-progress, clearly separate what exists now from what's planned. When there's planned behavior, review the skill each time it is loaded to see if planned behavior has been implemented and rewrite the skill as needed.
