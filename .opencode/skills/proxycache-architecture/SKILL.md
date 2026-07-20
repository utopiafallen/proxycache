---
name: proxycache-architecture
description: Proxycache KV cache slot management, cache hit scanning, slot acquisition, restore/save logic, skip-restore, pending slot hits, and backend error handling. Use when working with slots, cache hits, KV cache state, restore, save, routing, or any proxycache request flow.
---

# Proxycache Architecture

## Cache Key Computation

**Critical:** Cache keys are computed from **prompt token IDs only**, NOT prompt+response.

```python
# hashing.py
def meta_key(canonical_name: str, token_ids: List[int]) -> str:
    """sha256(canonical_name + '\n' + ','.join(token_ids))"""
```

`token_ids` comes from `first_token_ids` or `opt_token_ids` (computed in the chat handler before the cache hit scan), which are the result of tokenizing the chat template — the prompt only.

This means the cache key is **known at slot acquisition time**, before the response is generated.

## Slot KV State Tracking (`_slot_kv_state`)

`BackendSlotManager._slot_kv_state: Dict[int, List[str]]` tracks KV cache block hashes per slot (keyed by `slot_id`, per-backend).

- Set at slot acquisition time, making the slot's KV state visible to subsequent requests' cache hit scans while the slot is in-flight
- Updated after a successful restore or save to reflect the slot's current state
- On backend error (400+), restored to `prev_kv` (the state captured before acquisition's `set_kv_state()`) since the request was never processed
- Cleared by `invalidate()` on cancellation/failure in `StreamReader._cleanup()` before `release()`

## Cache Hit Scan Flow (in `app.py` chat handler)

1. **Tokenize on each backend**: Each backend applies its chat template and tokenizes the messages
2. **Compute block hashes**: `hs.block_hashes_from_tokens(opt_token_ids, WORDS_PER_BLOCK)`
3. **Disk scan**: `kv_meta.find_best_restore_candidate()` scans disk meta files for best cache hit
4. **Pending slot scan**: Iterate `_slot_kv_state` for the same model+backend, compute LCP ratio against request blocks. If a pending slot has a better ratio than the disk hit, use it instead. **Clears `restore_key`** — the slot already has the KV content in its cache, so no restore is needed.
5. **Select best match**: Use the candidate with the highest LCP ratio

The cache hit scan lives in the chat handler in `app.py`, between model resolution and slot acquisition. The pending slot scan runs inside the same per-backend loop as the disk scan, so it uses the correct `blocks` for each backend's tokenizer.

**Pending slot hit semantics**: When a pending slot is found, the slot's KV cache already contains the relevant content (from the previous request that filled it). Setting `restore_key` would cause a failed restore attempt against a non-existent cache file. The code explicitly sets `restore_key = None` to skip the restore — the request proceeds directly with the slot's existing KV cache.

## Slot Acquisition Flow (`acquire_for_request`)

**Phase 0 — Wait for cache backend:**
- Uses EMA-derived timeout per backend
- Polls every 5s for a free slot
- Max `CACHE_HIT_WAIT_MAX_PENDING_REQS` concurrent waiters per backend

**Phase 1 — Try cache backend directly:**
- Attempts to acquire the cache backend slot without waiting
- If successful, restores cache and returns

**Phase 2 — Retry loop:**
- Iterates all candidate backends (fallback only, excludes cache backend), sorted by composite score: `(cache_ratio, ring_size, latency_ema, last_used)` to minimize cache churn and latency
- Sleeps 5s between attempts (up to 11 attempts)
- Picks first available slot

**After slot acquisition succeeds (all phases):**
- `_slot_kv_state[slot_id]` is set to the request's blocks (before any restore decision)
- The slot's previous KV state (`old_kv`) is captured before `set_kv_state()` and returned as 4th element
- `_do_restore_call()` checks skip-restore using the captured previous state
- On backend error, `old_kv` is used to restore `_slot_kv_state` so tracking stays accurate

## `_should_skip_restore` Constraints

Only applies to **single-slot backends**:
```python
pool = self._slot_pools.get((g[0], g[1]))
if pool is not None and len(pool) > 1:
    return False
```

With multiple slots, skipping a restore is unsafe because llama.cpp may evict the chosen slot's cache under memory pressure.

The function compares the slot's **previous** KV state (captured at acquisition, before being overwritten with the request blocks) against the request blocks. If the LCP ratio is >= `KV_CACHE_SKIP_THRESHOLD` (0.9), the restore is skipped.

## Save Timing

When a request completes and starts saving, subsequent requests that arrive during the save window find the cache hit via the pending slot scan (the meta file hasn't been written yet). Once the save completes and the slot is released, the disk scan picks up the new meta file.

**Flow:**
1. Stream completes → `StreamReader._cleanup()` calls `_save()`
2. `_save()` calls `save_after()` which writes KV cache to disk and writes meta file
3. Slot is released via `sm.release()`
4. **During steps 1-2:** Pending slot scan finds the in-flight slot's KV state and uses it as a cache hit candidate
5. **After step 3:** Disk scan picks up the new meta file

## Gotchas

- **Tokenization is backend-specific**: Each backend applies its own chat template and tokenizer. Different backends may produce different token IDs for the same messages.
- **Cache key = prompt tokens only**: The key is computed from `opt_token_ids` (prompt), not the full request+response. This is known at acquisition time.
- **`_slot_kv_state` is keyed by `slot_id`** (per `BackendSlotManager` instance, not globally). Access via `sm.get(backend_id)._slot_kv_state[slot_id]`.
- **`invalidate_slot()` clears `_slot_kv_state`**: Called on cancellation/failure in `app.py:_cleanup()` before `release()`.
- **Ring buffer eviction is per-backend**: Uses `CACHE_MAX_SIZE_GB` per backend (default 25 GB). Evicts age-first, then LRU.
- **Pending slot scan uses per-backend blocks**: The scan runs inside the per-backend loop in `app.py`, so `blocks` always matches the current backend's tokenizer output.
- **Pending slot hit clears `restore_key`**: The cache hit scan iterates backends in a loop — a disk hit on backend A sets `restore_key`, then a pending slot hit on backend B must clear it. Leaving the stale key causes a failed restore attempt against a cache file that doesn't exist on backend B.
- **Backend error KV state restore**: When the backend errors (400+ status, connection error, non-JSON body) before processing the request, the slot's actual KV cache is untouched. `_acquire_slot_for_request` captures `old_kv = be_sm.get_kv_state(slot_id)` before `set_kv_state()`, returns it as 4th element. Error handlers in `chat()` restore `prev_kv` via `be_sm.set_kv_state(slot_id, prev_kv)`. Timeout errors (504) do NOT restore — the request may have been partially processed.
- **`_NO_PREV` sentinel in `should_skip_restore`**: `prev_blocks` parameter defaults to sentinel `_NO_PREV` (not `None`) to distinguish "not passed" (fallback to `_slot_kv_state` for backward compat with tests) from "explicitly `None`" (fresh slot, always return False — can't skip restore on a slot with no tracked state).
- **Backend key sanitization**: Filesystem paths use `sanitize_backend_dir()` (colons → dashes, e.g. `10.0.0.1:8000` → `10.0.0.1-8000`). But in-memory `_backends` dict keys, `DiscoveredModel.backends`, and all app-level code use raw colon keys. Tests that manually construct `BackendManager` must use raw colon keys for `_backends` dict.
- **Module-level config imports**: `app.py`, `hashing.py`, `slot_manager.py` all import config values (like `WORDS_PER_BLOCK`, `LCP_TH`) at module load time. Changing `config.WORDS_PER_BLOCK` in tests doesn't affect already-imported copies — must patch the importing module directly (e.g. `app_mod.WORDS_PER_BLOCK = 3`).

## Key Functions

| Function | Location | Role |
|----------|----------|------|
| `meta_key()` | `hashing.py` | Compute cache key from canonical name + token IDs |
| `block_hashes_from_tokens()` | `hashing.py` | Convert token IDs to block hashes for LCP matching |
| `lcp_blocks()` | `hashing.py` | Compute longest common prefix between two block lists |
| `find_best_restore_candidate()` | `kv_meta_manager.py` | Scan disk meta files for best cache hit |
| `scan_all_meta()` | `kv_meta_manager.py` | Load all meta files from disk for a backend |
| `_acquire_slot_for_request()` | `app.py` | Slot acquisition with Phase 0/1/2 logic, captures `old_kv`, sets `_slot_kv_state`, returns `(gslot, restored, skip_restore_diag, prev_kv)` |
| `_do_restore_call()` | `app.py` (nested) | Execute restore: flush skipped save, skip-restore check, disk restore, update `_slot_kv_state` |
| `should_skip_restore()` | `slot_manager.py` | Skip restore if LCP ratio >= threshold, single-slot backend, and valid prev state |
| `invalidate()` | `slot_manager.py` | Clear `_slot_kv_state` for a slot |
| `get_kv_state()` / `set_kv_state()` | `slot_manager.py` | Get/set slot KV block tracking |
| `save_after()` | `slot_manager.py` | Save KV cache to disk, write meta file, update ring buffer, update `_slot_kv_state` |
| `StreamReader._save()` | `app.py` | Detect recompute, call `save_after()`, return ok/cache_size |
| `StreamReader._cleanup()` | `app.py` | Stream lifecycle: save, invalidate, release slot, record metrics |
