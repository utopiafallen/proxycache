# slot_manager.py

# -*- coding: utf-8 -*-

"""
SlotManager: per-model slot pools with lazy discovery.

- Slot pools keyed by model name, not backend index.
- refresh_slots() called inside acquire_for_request() on every request (no cooldown).
- Router mode: discovers slot counts via GET /models + child /slots.
- Non-router mode: uses GET /slots as before.
- Ring buffer: tracks cache size in memory, evicts expired entries first, then LRU.
- Slots tracked by in-use flag (not lock) — acquisition checks flag non-blocking,
  falls back to next slot; if no slots available, sleeps 5s and retries up to 6 times.
"""

import os
import time
import asyncio
import logging
from collections import deque
from typing import List, Tuple, Dict, Optional

import httpx

from config import META_DIR, CACHE_MAX_AGE_HOURS, \
    KV_CACHE_SKIP_THRESHOLD, LCP_TH, WORDS_PER_BLOCK, SLOT_TIMEOUT, DEFAULT_N_CTX, \
    should_save_cache, CACHE_HIT_WAIT_EMA_MIN_TIMEOUT, CACHE_HIT_WAIT_MAX_PENDING_REQS, \
    CACHE_HIT_WAIT_EMA_ALPHA, CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT, CACHE_HIT_WAIT_EMA_MAX_TIMEOUT
import hashing as hs
from backend_manager import backend_manager
from kv_meta_manager import kv_meta

log = logging.getLogger(__name__)

# (canonical_model_name, backend_id, slot_id)
GSlot = Tuple[str, str, int]
# (canonical_model_name, backend_id)
ModelBackend = Tuple[str, str]
# (canonical_model_name, backend_id) -> set of slot_ids
SlotPools = Dict[ModelBackend, set[int]]
# (canonical_model_name, backend_id, slot_id) -> last_used timestamp
LastUsedMap = Dict[GSlot, float]


class SlotManager:
    def __init__(self):
        self._slot_pools: SlotPools = {}
        self._last_used: LastUsedMap = {}
        self._in_use: Dict[GSlot, bool] = {}

        # Per-backend cache ring buffer: backend_id -> deque of (key, size_bytes, last_used_time)
        self._cache_ring: Dict[str, deque] = {}
        # Per-backend total cache bytes: backend_id -> int
        self._total_bytes: Dict[str, int] = {}
        self._max_age_seconds: float = CACHE_MAX_AGE_HOURS * 3600

        # Per-slot KV cache block state — tracks hash blocks currently in each slot
        self._slot_kv_state: Dict[GSlot, List[str]] = {}
        # Per-slot save skip tracking — (key, blocks, n_tokens) if last save was intentionally skipped
        self._slot_save_skipped: Dict[GSlot, tuple] = {}

        # Cache hit wait queue: per-backend pending waiter count
        self._cache_wait_pending: Dict[str, int] = {}

        # Per-backend save lock to protect ring buffer + eviction from concurrent saves
        self._save_locks: Dict[str, asyncio.Lock] = {}

        # EMA of slot occupancy duration per backend: backend_id -> float
        self._slot_duration_ema: Dict[str, float] = {}

        # Slot acquisition timestamp: (model, backend, slot) -> float
        self._slot_acquired_at: Dict[GSlot, float] = {}

        # Per-backend last-used timestamp: backend_id -> float
        self._backend_last_used: Dict[str, float] = {}

        # Per-backend average request latency (EMA): backend_id -> float in ms
        self._backend_latency_ema: Dict[str, float] = {}

        log.info("Cache entry expiry set to %d hours", CACHE_MAX_AGE_HOURS)

    async def _evict_entry(self, key: str, backend_id: str, log_msg: str, log_extra: str):
        """Delete a cache file and its meta file in one step.

        Args:
            key: Cache file basename
            backend_id: Backend key (str) or None for unknown/local
            log_msg: Log message template prefix (e.g. "ring_evict_expired")
            log_extra: Extra info for log message
        """
        if backend_id is not None:
            ok = await backend_manager.cache_delete(backend_id, key)
            if ok:
                log.info("%s: %s %s", log_msg, key[:16], log_extra)
            else:
                log.warning("%s_agent_fail: %s", log_msg, key[:16])
        else:
            log.warning("%s: no backend_id, skipping cache file deletion", log_msg)
        kv_meta.delete_meta_file(key)

    async def init_from_disk(self):
        """Populate ring buffer from existing cache files on disk.
        Also performs a cleanup pass to remove expired entries.
        """
        if not os.path.isdir(META_DIR):
            return

        total_loaded = 0
        total_bytes_loaded = 0
        expired_evicted = 0
        expired_bytes = 0
        lru_evicted = 0
        lru_bytes = 0

        # Scan meta files in backend subdirectories to discover cached keys per backend
        for backend_dir in os.listdir(META_DIR):
            backend_path = os.path.join(META_DIR, backend_dir)
            if not os.path.isdir(backend_path):
                continue
            if backend_dir not in backend_manager.keys():
                continue
            ring = self._cache_ring.setdefault(backend_dir, deque())
            for key in kv_meta.list_keys(backend_dir):
                if any(entry[0] == key for entry in ring):
                    continue
                try:
                    cache_size = await kv_meta.get_cache_size(key, backend_dir)
                    if not cache_size:
                        continue
                    last_used = kv_meta.get_last_used_time(key, backend_dir)
                    ring.append((key, cache_size, last_used))
                    self._total_bytes[backend_dir] = self._total_bytes.get(backend_dir, 0) + cache_size
                    total_loaded += 1
                    total_bytes_loaded += cache_size
                except OSError:
                    continue

        # Cleanup pass: evict expired + over-size entries
        now = time.time()
        for backend_id, ring in self._cache_ring.items():
            max_bytes = backend_manager.get_cache_max_size_gb(backend_id) * 1024**3
            while ring:
                entry = ring[0]
                if now - entry[2] > self._max_age_seconds:
                    evict_key, evict_size, _ = entry
                    ring.popleft()
                    self._total_bytes[backend_id] -= evict_size
                    expired_evicted += 1
                    expired_bytes += evict_size
                    log.info(
                        "Startup cleanup: evicting expired entry '%s' for backend '%s' (%d bytes)",
                        evict_key, backend_id, evict_size,
                    )
                    await self._evict_entry(evict_key, backend_id, "startup_evict",
                                            "(%d bytes)" % evict_size)
                else:
                    break

            # Evict LRU entries if over target size
            while self._total_bytes.get(backend_id, 0) > max_bytes and ring:
                lru_idx = len(ring) - 1
                lru_ts = ring[-1][2]
                for i in range(len(ring) - 1):
                    if ring[i][2] < lru_ts:
                        lru_ts = ring[i][2]
                        lru_idx = i
                evict_key, evict_size, _ = ring[lru_idx]
                ring.remove(ring[lru_idx])
                self._total_bytes[backend_id] -= evict_size
                lru_evicted += 1
                lru_bytes += evict_size
                log.info(
                    "Startup cleanup: evicting LRU entry '%s' for backend '%s' (%d bytes, total now=%.1f GB)",
                    evict_key, backend_id, evict_size, self._total_bytes.get(backend_id, 0) / 1024**3,
                )
                await self._evict_entry(evict_key, backend_id, "startup_evict",
                                        "(%d bytes)" % evict_size)

        per_backend = [(bid, self._total_bytes.get(bid, 0), len(self._cache_ring.get(bid, [])))
                       for bid in self._cache_ring]
        log.info(
            "Loaded %d cache files from disk (%.1f GB), evicted %d expired (%.1f GB), "
            "%d LRU (%.1f GB), per-backend: %s",
            total_loaded, total_bytes_loaded / 1024**3,
            expired_evicted, expired_bytes / 1024**3,
            lru_evicted, lru_bytes / 1024**3,
            "; ".join(f"{bid}: {sz / 1024**3:.1f} GB ({cnt} files)" for bid, sz, cnt in per_backend),
        )

    def _is_free(self, model_name: str, backend_id: str, slot_id: int) -> bool:
        return not self._in_use.get((model_name, backend_id, slot_id), False)

    def _get_free_or_oldest_from_pool(
        self, model_name: str, backend_id: str
    ) -> Tuple[Optional[int], bool]:
        """Pick free slot or oldest (LRU) from a single backend's pool for a model.

        Returns (slot_id, in_use_flag) on success, (None, False) if no pool exists.
        """
        key = (model_name, backend_id)
        pool = self._slot_pools.get(key)
        if not pool:
            return None, False

        free = [s for s in pool if self._is_free(model_name, backend_id, s)]
        if free:
            return free[0], self._in_use.get((model_name, backend_id, free[0]), False)

        oldest = min(pool, key=lambda s: self._last_used.get((model_name, backend_id, s), 0.0))
        return oldest, self._in_use.get((model_name, backend_id, oldest), False)

    def _ensure_pool(self, model_name: str, backend_id: str, n_slots: int):
        """Create or update a slot pool for (model_name, backend_key)."""
        key = (model_name, backend_id)
        if key not in self._slot_pools:
            new_pool = set(range(n_slots))
            for s in new_pool:
                self._last_used[(model_name, backend_id, s)] = 0.0
                self._in_use[(model_name, backend_id, s)] = False
            self._slot_pools[key] = new_pool
            log.info(
                "Created slot pool for model '%s' on backend '%s' with %d slots",
                model_name, backend_id, n_slots,
            )
        else:
            old_pool = self._slot_pools[key]
            old_count = len(old_pool)
            new_pool = set(range(n_slots))
            # Add new slots
            for s in new_pool - old_pool:
                self._last_used[(model_name, backend_id, s)] = 0.0
                self._in_use[(model_name, backend_id, s)] = False
            # Remove old slots (only free ones)
            for s in old_pool - new_pool:
                if self._is_free(model_name, backend_id, s):
                    g = (model_name, backend_id, s)
                    self._slot_pools[key].discard(s)
                    self._last_used.pop(g, None)
                    self._in_use.pop(g, None)
                    self._slot_kv_state.pop(g, None)
                    self._slot_save_skipped.pop(g, None)
            self._slot_pools[key] = new_pool
            log.info(
                "Updated slot pool for model '%s' on backend '%s': %d -> %d slots",
                model_name, backend_id, old_count, n_slots,
            )

    def _should_skip_restore(self, g: GSlot, req_blocks: List[str], prev_blocks: Optional[List[str]] = None) -> bool:
        """Check if the slot's current KV cache already matches the request well enough.

        Only applies when the backend has a single slot. With multiple slots, skipping a
        restore is unsafe — llama.cpp may evict the chosen slot's cache under memory pressure
        from serving concurrent requests. A single-slot backend is safe because the proxy
        knows exactly what traffic goes to the backend and the slot's state is predictable.

        If prev_blocks is provided (the slot's KV state before the current acquisition),
        it is used for the comparison. When prev_blocks is None, the slot has never been
        saved — its contents are unknown, so a restore must be attempted.

        Returns True if the slot's tracked KV cache blocks overlap >= KV_CACHE_SKIP_THRESHOLD
        with the request blocks, meaning no restore is required.
        """
        if not prev_blocks:
            return False

        pool = self._slot_pools.get((g[0], g[1]))
        if pool is not None and len(pool) > 1:
            return False

        lcp = hs.lcp_blocks(req_blocks, prev_blocks)
        denom = max(1, min(len(req_blocks), len(prev_blocks)))
        ratio = lcp / denom
        log.warning(
            "Checking skip restore for model '%s' on backend '%s' slot %d: ratio=%.3f",
            g[0], g[1], g[2], ratio,
        )
        return ratio >= KV_CACHE_SKIP_THRESHOLD

    def _try_acquire(self, model_name: str, backend_id: str, slot_id: int) -> bool:
        """Try to acquire a slot by setting its in-use flag. Returns True if acquired."""
        g = (model_name, backend_id, slot_id)
        if self._in_use.get(g, False):
            return False
        now = time.time()
        self._in_use[g] = True
        self._last_used[g] = now
        self._backend_last_used[backend_id] = now
        return True

    def get_backend_last_used(self, backend_id: str) -> float:
        """Return the last time any slot on the backend was acquired. 0.0 if never used."""
        return self._backend_last_used.get(backend_id, 0.0)

    def update_backend_latency(self, backend_id: str, latency_ms: float) -> None:
        """Update the EMA of average request latency for a backend (in ms)."""
        alpha = CACHE_HIT_WAIT_EMA_ALPHA
        old = self._backend_latency_ema.get(backend_id, latency_ms)
        self._backend_latency_ema[backend_id] = alpha * latency_ms + (1 - alpha) * old

    def get_backend_latency_ema(self, backend_id: str) -> float:
        """Return the EMA of average request latency for a backend. 0.0 if no data."""
        return self._backend_latency_ema.get(backend_id, 0.0)

    async def acquire_for_request(
        self,
        candidate_backends: list[tuple[str, str]],
        restore_info: Optional[tuple[str, str, str]] = None,
        backend_blocks: Optional[Dict[str, List[str]]] = None,
        prompt_tokens: int = 0,
    ) -> Tuple[GSlot, Optional[bool]]:
        """Acquire a slot, checking all candidate backends before sleeping.

        Uses in-use flag (not lock) — checks flag non-blocking across all backends.
        If all slots across all backends are unavailable, sleeps 5s and retries up to 6 times.

        candidate_backends: list of fallback (backend_id, canonical_name) pairs
            (does NOT include the cache backend).
        restore_info: optional (restore_key, cache_backend, canonical_name) tuple.
            If provided, the cache backend is checked first. But if no slot is available
            there, fallback candidates are tried — restore_key is only used if the
            cache backend was acquired (cache files are not shared between backends).
        backend_blocks: dict mapping backend_id to block hashes from that backend's
            tokenization. Each backend gets its own blocks for skip-restore checks.
        """
        # Refresh slot counts for all discovered models
        try:
            slot_counts = await backend_manager.refresh_slot_counts()
        except Exception as e:
            log.warning("Failed to refresh slot counts: %s — proceeding with existing pool state", e)
            slot_counts = {}
        for backend_key, model_slots in slot_counts.items():
            for canonical_name, n_slots in model_slots.items():
                self._ensure_pool(canonical_name, backend_key, n_slots)

        # Phase 0: try cache backend first; if busy, poll every 5s up to EMA timeout
        if restore_info:
            restore_key, cache_backend, canonical_name = restore_info
            min_ctx = backend_manager.get_model_n_ctx(canonical_name)
            if prompt_tokens < min_ctx:
                be_blocks = backend_blocks.get(cache_backend) if backend_blocks else None
                result = await self._try_acquire_and_restore(canonical_name, cache_backend, restore_key, be_blocks)
                if result:
                    return result

                # No free slot — poll every 5s up to EMA timeout (limited concurrency)
                pending = self._cache_wait_pending.get(cache_backend, 0)
                if pending < CACHE_HIT_WAIT_MAX_PENDING_REQS:
                    ema = self._slot_duration_ema.get(cache_backend, CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT)
                    wait_timeout = max(min(ema, CACHE_HIT_WAIT_EMA_MAX_TIMEOUT), CACHE_HIT_WAIT_EMA_MIN_TIMEOUT)
                    log.info(
                        "Cache backend '%s' busy for model '%s', polling up to %.1fs",
                        cache_backend, canonical_name, wait_timeout,
                    )
                    try:
                        self._cache_wait_pending[cache_backend] = pending + 1
                        elapsed = 0.0
                        while elapsed < wait_timeout:
                            await asyncio.sleep(min(5.0, wait_timeout - elapsed))
                            elapsed += 5.0
                            be_blocks = backend_blocks.get(cache_backend) if backend_blocks else None
                            result = await self._try_acquire_and_restore(canonical_name, cache_backend, restore_key, be_blocks)
                            if result:
                                return result
                    finally:
                        self._cache_wait_pending[cache_backend] -= 1

        # Retry loop: check all backends, sleep 5s if none available
        RETRY_COUNT=11
        for attempt in range(RETRY_COUNT):
            # Phase 1: check cache backend first (if provided)
            if restore_info:
                restore_key, cache_backend, canonical_name = restore_info
                min_ctx = backend_manager.get_model_n_ctx(canonical_name)
                if prompt_tokens < min_ctx:
                    be_blocks = backend_blocks.get(cache_backend) if backend_blocks else None
                    result = await self._try_acquire_and_restore(canonical_name, cache_backend, restore_key, be_blocks)
                    if result:
                        return result

            # Phase 2: check all fallback candidate backends
            for backend_id, canonical_name in candidate_backends:
                if not canonical_name:
                    continue
                min_ctx = backend_manager.get_model_n_ctx(canonical_name)
                if prompt_tokens >= min_ctx:
                    continue
                be_blocks = backend_blocks.get(backend_id) if backend_blocks else None
                result = await self._try_acquire_and_restore(canonical_name, backend_id, None, be_blocks)
                if result:
                    return result

            # No slot available — exponential backoff before retrying (last iteration skips sleep)
            if attempt < RETRY_COUNT - 1:
                backoff = (attempt + 1) * 5
                log.info("No slots available across all backends, retrying in %ds (attempt %d/%d)", backoff, attempt + 1, RETRY_COUNT)
                await asyncio.sleep(backoff)

        raise RuntimeError(f"No slots available for candidate_backends={len(candidate_backends)}")

    async def _try_acquire_and_restore(
        self,
        model_name: str,
        backend_id: str,
        restore_key: Optional[str],
        blocks: Optional[List[str]],
    ) -> Optional[Tuple[GSlot, Optional[bool]]]:
        """Try to acquire a slot on a backend and restore cache.

        Returns (g, restored) on success, None if acquisition failed.
        """
        slot_id, _ = self._get_free_or_oldest_from_pool(model_name, backend_id)
        if slot_id is None:
            return None
        if not self._try_acquire(model_name, backend_id, slot_id):
            return None
        g = (model_name, backend_id, slot_id)
        self._slot_acquired_at[g] = time.time()
        prev_kv = self._slot_kv_state.get(g)
        self._slot_kv_state[g] = blocks or []
        return await self._restore_and_return(
            model_name, backend_id, slot_id,
            restore_key, blocks, prev_kv,
        )

    async def _restore_and_return(
        self,
        model_name: str,
        backend_id: str,
        slot_id: int,
        effective_restore_key: Optional[str],
        blocks: Optional[List[str]],
        prev_kv: Optional[List[str]] = None,
    ) -> Tuple[GSlot, Optional[bool]]:
        """Restore KV cache and return (g, restored).

        prev_kv: the slot's KV state before the current acquisition (captured at
            acquisition time before _slot_kv_state[g] was overwritten with request blocks).
        """
        g = (model_name, backend_id, slot_id)
        restored: Optional[bool] = None

        # If the slot's last save was skipped, re-evaluate and do a full save now before
        # overwriting the slot with new cache
        if g in self._slot_save_skipped:
            skip_entry = self._slot_save_skipped[g]
            del self._slot_save_skipped[g]
            if len(skip_entry) >= 6:
                save_key, save_blocks, save_n_tokens, skip_restored, skip_ratio, skip_recompute = skip_entry
                if not should_save_cache(skip_ratio, skip_recompute):
                    log.info(
                        "Flushed skipped cache for model '%s' on backend '%s' slot %d: "
                        "still not worth saving (ratio %.3f, restored=%s, recompute=%s)",
                        model_name, backend_id, slot_id, skip_ratio, skip_restored, skip_recompute,
                    )
                    ok, size = 0, 0
                else:
                    ok, size = await self.save_after(
                        model_name, backend_id, slot_id, save_key, save_blocks, save_n_tokens,
                    )
                    log.info(
                        "Saved skipped cache for model '%s' on backend '%s' slot %d (%d bytes) before restore",
                        model_name, backend_id, slot_id, size,
                    )
            else:
                # Legacy format: (key, blocks, n_tokens) — always save for backward compat
                save_key, save_blocks, save_n_tokens = skip_entry[:3]
                ok, size = await self.save_after(
                    model_name, backend_id, slot_id, save_key, save_blocks, save_n_tokens,
                )
                log.info(
                    "Saved skipped cache for model '%s' on backend '%s' slot %d (%d bytes) before restore (legacy format)",
                    model_name, backend_id, slot_id, size,
                )

        if blocks:
            if self._should_skip_restore(g, blocks, prev_kv):
                log.info(
                    "Skipping restore for model '%s' on backend '%s' slot %d: slot cache already matches",
                    model_name, backend_id, slot_id,
                )
                restored = False
            elif effective_restore_key:
                restored = await self._do_restore(
                    slot_id, effective_restore_key, model_name, backend_id,
                    touch_ring=True,
                )
                if restored:
                    blocks = kv_meta.get_blocks(effective_restore_key, backend_id)
                    if blocks is not None:
                        self._slot_kv_state[g] = blocks
            else:
                cand = kv_meta.find_best_restore_candidate(
                    blocks, WORDS_PER_BLOCK, LCP_TH, model_name, backend_id,
                )
                if cand:
                    cand_key, cand_ratio = cand
                    restored = await self._do_restore(
                        slot_id, cand_key, model_name, backend_id,
                        touch_ring=False,
                        log_prefix="Dynamic",
                        log_extra=f", ratio {cand_ratio:.3f}",
                    )
                    if restored:
                        blocks = kv_meta.get_blocks(cand_key, backend_id)
                        if blocks is not None:
                            self._slot_kv_state[g] = blocks
                else:
                    log.info(
                        "No dynamic restore candidate for model '%s' on backend '%s' slot %d",
                        model_name, backend_id, slot_id,
                    )
                    restored = False

        elif effective_restore_key:
            restored = await self._do_restore(
                slot_id, effective_restore_key, model_name, backend_id,
                touch_ring=True,
            )

        return g, restored

    async def _do_restore(
        self,
        slot_id: int,
        key: str,
        model_name: str,
        backend_id: str,
        touch_ring: bool = False,
        log_prefix: str = "Restore",
        log_extra: str = "",
    ) -> bool:
        """Perform a slot restore with standardized error handling.

        Args:
            slot_id: Target slot.
            key: Cache file basename.
            model_name: Model name for logging.
            backend_id: Backend key for client lookup.
            touch_ring: Update the ring buffer last_used timestamp on success.
            update_kv: Update _slot_kv_state from the restored meta on success.
            log_prefix: Prefix for the success log message (e.g. "Restore", "Dynamic").
            log_extra: Extra text appended to the success log message.
        """
        client = backend_manager.get_client(backend_id)
        try:
            restored = await client.restore_slot(slot_id, key, model_name)
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout) as e:
            log.warning(
                "%s failed for model '%s' on backend '%s' slot %d (key %s): %s",
                log_prefix, model_name, backend_id, slot_id, key[:16], e,
            )
            restored = False
        except Exception as e:
            log.warning(
                "Unexpected error %s cache for model '%s' on backend '%s' slot %d (key %s): %s",
                log_prefix.lower(), model_name, backend_id, slot_id, key[:16], e,
            )
            restored = False
        log.info(
            "%s restored cache for model '%s' on backend '%s' slot %d (key %s): ok=%s%s",
            log_prefix, model_name, backend_id, slot_id, key[:16], restored, log_extra,
        )
        if restored:
            if touch_ring:
                self._touch_ring(key, backend_id)
        return restored

    async def save_after(
        self,
        model_name: str,
        backend_id: str,
        slot_id: int,
        key: str,
        blocks: Optional[List[str]] = None,
        n_tokens: int = 0,
    ) -> Tuple[bool, int]:
        client = backend_manager.get_client(backend_id)
        ok, size = await client.save_slot(slot_id, key, model_name)

        # Always track KV state for this slot — even if save failed, the slot
        # was used for this request and its cache now corresponds to these blocks
        if blocks:
            g = (model_name, backend_id, slot_id)
            self._slot_kv_state[g] = blocks
            log.warning(
                "Updated KV cache state for model '%s' on backend '%s' slot %d: %d blocks",
                model_name, backend_id, slot_id, len(blocks),
            )

        if ok and size > 0:
            lock = self._save_locks.setdefault(backend_id, asyncio.Lock())
            async with lock:
                # Write meta file — we never want cache files without corresponding meta
                try:
                    kv_meta.write_meta(
                        key, n_tokens, blocks, WORDS_PER_BLOCK, model_name, backend_id, size,
                    )
                except Exception as e:
                    log.warning("Failed to write meta file for key %s: %s", key[:16], e)
                ring = self._cache_ring.setdefault(backend_id, deque())
                ring.append((key, size, time.time()))
                self._total_bytes[backend_id] = self._total_bytes.get(backend_id, 0) + size

                # Ring buffer eviction: age-first, then LRU (per backend)
                max_bytes = backend_manager.get_cache_max_size_gb(backend_id) * 1024**3
                now = time.time()
                total = self._total_bytes.get(backend_id, 0)
                log.info(
                    "Cache ring check for backend '%s': total=%d bytes, max=%d bytes, ring_size=%d",
                    backend_id, total, max_bytes, len(ring),
                )
                while self._total_bytes.get(backend_id, 0) > max_bytes and ring:
                    # First pass: evict expired entries
                    evicted_expired = False
                    for entry in ring:
                        if now - entry[2] > self._max_age_seconds:
                            evict_key, evict_size, _ = entry
                            ring.remove(entry)
                            self._total_bytes[backend_id] -= evict_size
                            age_hours = (now - entry[2]) / 3600
                            await self._evict_entry(evict_key, backend_id, "ring_evict_expired",
                                                    "(%d bytes, age=%.1fh)" % (evict_size, age_hours))
                            evicted_expired = True
                            break
                    if evicted_expired:
                        continue

                    # Second pass: evict LRU entry (no expired entries left)
                    lru_idx = 0
                    lru_ts = ring[0][2]
                    for i in range(1, len(ring)):
                        if ring[i][2] < lru_ts:
                            lru_ts = ring[i][2]
                            lru_idx = i
                    evict_key, evict_size, _ = ring[lru_idx]
                    ring.remove(ring[lru_idx])
                    self._total_bytes[backend_id] -= evict_size
                    log.info(
                        "Ring buffer eviction: evicted LRU entry '%s' for backend '%s' (%d bytes, remaining=%d)",
                        evict_key, backend_id, evict_size, self._total_bytes.get(backend_id, 0),
                    )
                    await self._evict_entry(evict_key, backend_id, "ring_evict_lru",
                                            "(%d bytes, last_used=%.0f)" % (evict_size, lru_ts))

        return ok, size

    def invalidate_slot(self, model_name: str, backend_id: str, slot_id: int):
        """Clear KV cache tracking for a slot whose cache is in an unknown state."""
        g = (model_name, backend_id, slot_id)
        if g in self._slot_kv_state:
            del self._slot_kv_state[g]
            log.info(
                "Invalidated KV cache tracking for model '%s' on backend '%s' slot %d",
                model_name, backend_id, slot_id,
            )

    def release(self, model_name: str, backend_id: str, slot_id: int):
        g = (model_name, backend_id, slot_id)
        self._in_use[g] = False

        # Update EMA of slot occupancy duration
        if g in self._slot_acquired_at:
            duration = time.time() - self._slot_acquired_at[g]
            del self._slot_acquired_at[g]
            old = self._slot_duration_ema.get(backend_id, CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT)
            self._slot_duration_ema[backend_id] = CACHE_HIT_WAIT_EMA_ALPHA * duration + (1 - CACHE_HIT_WAIT_EMA_ALPHA) * old

        # (waiters poll every 5s, no explicit signaling needed)

    def _touch_ring(self, key: str, backend_id: str):
        """Update the last_used timestamp for a cache entry in the ring buffer."""
        ring = self._cache_ring.get(backend_id)
        if not ring:
            return
        for i in range(len(ring)):
            if ring[i][0] == key:
                ring[i] = (key, ring[i][1], time.time())
                break
