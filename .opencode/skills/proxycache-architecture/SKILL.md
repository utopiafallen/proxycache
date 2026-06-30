---
name: proxycache-architecture
description: Proxycache KV cache slot management, cache hit scanning, and pending cache optimization. Use when modifying slot_manager.py, app.py cache hit logic, or kv_meta_manager.py.
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

`SlotManager._slot_kv_state: Dict[GSlot, List[str]]` tracks KV cache block hashes per slot.

**Current behavior (as of pending cache hit plan):**
- Set in `save_after()` after the KV cache is saved
- Used by `_should_skip_restore()` to skip unnecessary restores when the slot's KV cache already matches the request

**Planned behavior (pending cache hit):**
- Set immediately at slot acquisition in `acquire_for_request()` (after `_try_acquire()` succeeds)
- Makes the KV state visible to subsequent requests' cache hit scans during the save window
- Cleared by `invalidate_slot()` on cancellation/failure in `StreamReader._cleanup()` before `release()`

## Cache Hit Scan Flow

1. **Tokenize on each backend**: Each backend applies its chat template and tokenizes the messages
2. **Compute block hashes**: `hs.block_hashes_from_tokens(opt_token_ids, WORDS_PER_BLOCK)`
3. **Find best restore candidate**: `kv_meta.find_best_restore_candidate()` scans disk meta files
4. **(Planned)** Check pending slots: Iterate `_slot_kv_state` for the model+backend, compute LCP ratio against request blocks
5. **Select best match**: Use the candidate with the highest LCP ratio that exceeds `LCP_TH` threshold

The cache hit scan lives in the chat handler in `app.py`, between model resolution and slot acquisition.

## Slot Acquisition Flow (`acquire_for_request`)

**Phase 0 — Wait for cache backend:**
- Uses EMA-derived timeout per backend
- Waits for semaphore release (triggered by slot release)
- Max `CACHE_HIT_WAIT_MAX_PENDING_REQS` concurrent waiters per backend

**Phase 1 — Try cache backend directly:**
- Attempts to acquire the cache backend slot without waiting
- If successful, restores cache and returns

**Phase 2 — Retry loop:**
- Iterates all candidate backends (fallback only, excludes cache backend)
- Sleeps 5s between attempts (up to 11 attempts)
- Picks first available slot

**Pending slot handling (planned):**
- If best match is on a pending (busy) slot, Phase 0/Phase 1 tries that backend
- Finds it busy, falls through to retry loop
- Eventually the slot becomes available and gets acquired
- `_should_skip_restore` returns True (KV already matches) → skip restore

## `_should_skip_restore` Constraints

Only applies to **single-slot backends**:
```python
pool = self._slot_pools.get((g[0], g[1]))
if pool is not None and len(pool) > 1:
    return False
```

With multiple slots, skipping a restore is unsafe because llama.cpp may evict the chosen slot's cache under memory pressure.

## Save Timing and the Pending Cache Window

**Current problem:** When a request completes and starts saving, subsequent requests that arrive during the save window miss the cache hit because `find_best_restore_candidate()` only scans disk meta files. The meta file hasn't been written yet.

**Flow:**
1. Stream completes → `StreamReader._cleanup()` calls `_save()`
2. `_save()` calls `save_after()` which writes KV cache to disk and writes meta file
3. Slot is released via `sm.release()`

**Solution (pending cache hit):** By updating `_slot_kv_state` at slot acquisition time, the KV state is visible to the cache hit scan immediately, covering the full window from acquisition → save completion.

## Key Functions

| Function | Location | Role |
|----------|----------|------|
| `meta_key()` | `hashing.py` | Compute cache key from canonical name + token IDs |
| `block_hashes_from_tokens()` | `hashing.py` | Convert token IDs to block hashes for LCP matching |
| `lcp_blocks()` | `hashing.py` | Compute longest common prefix between two block lists |
| `find_best_restore_candidate()` | `kv_meta_manager.py` | Scan disk meta files for best cache hit |
| `scan_all_meta()` | `kv_meta_manager.py` | Load all meta files from disk for a backend |
| `acquire_for_request()` | `slot_manager.py` | Slot acquisition with Phase 0/1/2 logic |
| `_should_skip_restore()` | `slot_manager.py` | Skip restore if slot KV cache already matches |
| `invalidate_slot()` | `slot_manager.py` | Clear `_slot_kv_state` for a slot |
| `save_after()` | `slot_manager.py` | Save KV cache to disk, write meta file, update ring buffer |
| `StreamReader._save()` | `app.py` | Detect recompute, call `save_after()`, return ok/cache_size |
| `StreamReader._cleanup()` | `app.py` | Stream lifecycle: save, release slot, record metrics |

## Gotchas

- **Tokenization is backend-specific**: Each backend applies its own chat template and tokenizer. Different backends may produce different token IDs for the same messages (different quantization variants can have different tokenizers).
- **Cache key = prompt tokens only**: The key is computed from `opt_token_ids` (prompt), not the full request+response. This is known at acquisition time.
- **`_slot_kv_state` is keyed by `(model, backend, slot_id)`**: Use the full GSlot tuple, not just the backend.
- **`invalidate_slot()` clears `_slot_kv_state`**: Called on cancellation/failure in `app.py:_cleanup()` before `release()`.
- **Ring buffer eviction is per-backend**: Uses `CACHE_MAX_SIZE_GB` per backend (default 25 GB). Evicts age-first, then LRU.
