# slot_manager.py

# -*- coding: utf-8 -*-

"""
SlotManager: per-model slot pools with lazy discovery and refresh cooldown.

- Slot pools keyed by model name, not backend index.
- refresh_slots() called inside acquire_for_request() with per-(model, backend) cooldown.
- Router mode: discovers slot counts via GET /models + child /slots.
- Non-router mode: uses GET /slots as before.
- Ring buffer: tracks cache size in memory, evicts expired entries first, then LRU.
"""

import os
import json
import time
import asyncio
import logging
from collections import deque
from typing import List, Tuple, Dict, Optional

from config import META_DIR, CACHE_DIR, CACHE_MAX_AGE_HOURS, CACHE_MAX_SIZE_GB, \
    KV_CACHE_SKIP_THRESHOLD, LCP_TH, WORDS_PER_BLOCK, SLOT_TIMEOUT, DEFAULT_N_CTX
import hashing as hs
from backend_manager import backend_manager

log = logging.getLogger(__name__)

# (canonical_model_name, backend_id, slot_id)
GSlot = Tuple[str, str, int]
# (canonical_model_name, backend_id)
ModelBackend = Tuple[str, str]
# (canonical_model_name, backend_id) -> set of slot_ids
SlotPools = Dict[ModelBackend, set[int]]
# (canonical_model_name, backend_id, slot_id) -> last_used timestamp
LastUsedMap = Dict[GSlot, float]
# (canonical_model_name, backend_id, slot_id) -> asyncio.Lock
LockMap = Dict[GSlot, asyncio.Lock]


class SlotManager:
    def __init__(self):
        self._slot_pools: SlotPools = {}
        self._last_used: LastUsedMap = {}
        self._locks: LockMap = {}

        # Per-backend cache ring buffer: backend_id -> deque of (key, size_bytes, last_used_time)
        self._cache_ring: Dict[str, deque] = {}
        # Per-backend total cache bytes: backend_id -> int
        self._total_bytes: Dict[str, int] = {}
        self._max_age_seconds: float = CACHE_MAX_AGE_HOURS * 3600

        # Per-slot KV cache block state — tracks hash blocks currently in each slot
        self._slot_kv_state: Dict[GSlot, List[str]] = {}

        log.info("Cache entry expiry set to %d hours", CACHE_MAX_AGE_HOURS)

    async def _evict_cache_file(self, key: str, backend_id: str, log_msg: str, log_extra: str):
        """Delete a cache file via agent or local fallback.

        Args:
            key: Cache file basename
            backend_id: Backend key (str) or None for unknown/local
            log_msg: Log message template prefix (e.g. "ring_evict_expired")
            log_extra: Extra info for log message
        """
        if backend_id is not None:
            agent = backend_manager.get_agent(backend_id)
            if agent:
                ok = await agent.delete(key)
                if ok:
                    log.info("%s: %s %s", log_msg, key[:16], log_extra)
                else:
                    log.warning("%s_agent_fail: %s", log_msg, key[:16])
                return
        cache_path = CACHE_DIR + "/" + key if CACHE_DIR else None
        if cache_path and os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                log.info("%s: %s %s", log_msg, key[:16], log_extra)
            except OSError:
                pass

    def _evict_meta_file(self, key: str):
        """Delete the meta file for a cache entry."""
        hs.delete_meta_file(key)

    def init_from_disk(self, cache_dir: str):
        """Populate ring buffer from existing cache files on disk."""
        if not cache_dir or not os.path.isdir(cache_dir):
            return
        # Scan meta files in backend subdirectories to discover cached keys per backend
        if os.path.isdir(META_DIR):
            for backend_dir in os.listdir(META_DIR):
                backend_path = os.path.join(META_DIR, backend_dir)
                if not os.path.isdir(backend_path):
                    continue
                ring = self._cache_ring.setdefault(backend_dir, deque())
                total = self._total_bytes.setdefault(backend_dir, 0)
                for meta_file in os.listdir(backend_path):
                    if not meta_file.endswith(hs.META_SUFFIX):
                        continue
                    key = meta_file.removesuffix(hs.META_SUFFIX)
                    # Skip if already in ring buffer for this backend
                    if any(entry[0] == key for entry in ring):
                        continue
                    try:
                        cache_path = os.path.join(cache_dir, key)
                        if os.path.exists(cache_path):
                            size = os.stat(cache_path).st_size
                            last_used = hs._get_last_used_time(key, META_DIR, cache_dir, backend_dir)
                            ring.append((key, size, last_used))
                            self._total_bytes[backend_dir] = total + size
                    except OSError:
                        continue
        total_files = sum(len(r) for r in self._cache_ring.values())
        total_bytes = sum(self._total_bytes.values())
        log.info("Loaded %d cache files from disk, total %.1f GB", total_files, total_bytes / 1024**3)

    def _is_free(self, model_name: str, backend_id: str, slot_id: int) -> bool:
        return self._last_used.get((model_name, backend_id, slot_id), 0.0) == 0.0

    def _get_free_or_oldest_from_pool(
        self, model_name: str, backend_id: str
    ) -> Tuple[Optional[int], Optional[asyncio.Lock]]:
        """Pick free slot or oldest (LRU) from a single backend's pool for a model.

        Returns (slot_id, lock) on success, (None, None) if no pool exists.
        """
        key = (model_name, backend_id)
        pool = self._slot_pools.get(key)
        if not pool:
            return None, None

        free = [s for s in pool if self._is_free(model_name, backend_id, s)]
        if free:
            return free[0], self._locks[(model_name, backend_id, free[0])]

        oldest = min(pool, key=lambda s: self._last_used.get((model_name, backend_id, s), 0.0))
        return oldest, self._locks[(model_name, backend_id, oldest)]

    def _ensure_pool(self, model_name: str, backend_id: str, n_slots: int):
        """Create or update a slot pool for (model_name, backend_key)."""
        key = (model_name, backend_id)
        if key not in self._slot_pools:
            new_pool = set(range(n_slots))
            for s in new_pool:
                self._last_used[(model_name, backend_id, s)] = 0.0
                self._locks[(model_name, backend_id, s)] = asyncio.Lock()
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
                self._locks[(model_name, backend_id, s)] = asyncio.Lock()
            # Remove old slots (only free ones)
            for s in old_pool - new_pool:
                if self._is_free(model_name, backend_id, s):
                    self._slot_pools[key].discard(s)
                    self._last_used.pop((model_name, backend_id, s), None)
                    self._locks.pop((model_name, backend_id, s), None)
            self._slot_pools[key] = new_pool
            log.info(
                "Updated slot pool for model '%s' on backend '%s': %d -> %d slots",
                model_name, backend_id, old_count, n_slots,
            )

    def _should_skip_restore(self, g: GSlot, req_blocks: List[str]) -> bool:
        """Check if the slot's current KV cache already matches the request well enough.

        Only applies when the backend has a single slot. With multiple slots, skipping a
        restore is unsafe — llama.cpp may evict the chosen slot's cache under memory pressure
        from serving concurrent requests. A single-slot backend is safe because the proxy
        knows exactly what traffic goes to the backend and the slot's state is predictable.

        Returns True if the slot's tracked KV cache blocks overlap >= KV_CACHE_SKIP_THRESHOLD
        with the request blocks, meaning no restore is needed.
        """
        kv_blocks = self._slot_kv_state.get(g)
        if not kv_blocks:
            return False

        pool = self._slot_pools.get((g[0], g[1]))
        if pool is not None and len(pool) > 1:
            return False

        lcp = hs.lcp_blocks(req_blocks, kv_blocks)
        denom = max(1, min(len(req_blocks), len(kv_blocks)))
        ratio = lcp / denom
        log.debug(
            "Checking skip restore for model '%s' on backend '%s' slot %d: ratio=%.3f",
            g[0], g[1], g[2], ratio,
        )
        return ratio >= KV_CACHE_SKIP_THRESHOLD

    async def acquire_for_request(
        self,
        candidate_backends: list[tuple[str, str]],
        restore_info: Optional[tuple[str, str, str]] = None,
        blocks: Optional[List[str]] = None,
        prompt_tokens: int = 0,
    ) -> Tuple[GSlot, asyncio.Lock, Optional[bool]]:
        """Acquire a slot, trying the cache backend first, then fallback candidates.

        candidate_backends: list of fallback (backend_id, canonical_name) pairs
            (does NOT include the cache backend).
        restore_info: optional (restore_key, cache_backend, canonical_name) tuple.
            If provided, the cache backend is tried first. If its lock times out,
            fallback candidates are tried — but restore_key is only used if the
            cache backend was acquired (cache files are not shared between backends).
        """
        # Refresh slot counts for all discovered models (with cooldown)
        slot_counts = await backend_manager.refresh_slot_counts()
        for backend_key, model_slots in slot_counts.items():
            for canonical_name, n_slots in model_slots.items():
                self._ensure_pool(canonical_name, backend_key, n_slots)

        # Cap lock acquire timeout at 30s — don't let requests hang forever
        lock_timeout = min(30.0, SLOT_TIMEOUT)

        # Try cache backend first (if provided)
        if restore_info:
            restore_key, cache_backend, canonical_name = restore_info
            min_ctx = backend_manager.get_model_n_ctx(canonical_name)
            slot_id, lock = self._get_free_or_oldest_from_pool(canonical_name, cache_backend);
            if prompt_tokens < min_ctx and slot_id is not None:
                try:
                    await asyncio.wait_for(lock.acquire(), timeout=lock_timeout)
                except asyncio.TimeoutError:
                    log.warning(
                        "Lock timed out after %ds for model '%s' on backend '%s' slot %d, trying fallback backends",
                        canonical_name, cache_backend, slot_id, lock_timeout,
                    )
                else:
                    self._last_used[(canonical_name, cache_backend, slot_id)] = time.time()
                    effective_restore_key = restore_key
                    return await self._restore_and_return(
                        canonical_name, cache_backend, slot_id, lock,
                        effective_restore_key, blocks,
                    )

        # Fallback: try candidate backends
        for backend_id, canonical_name in candidate_backends:
            if not canonical_name:
                continue
            min_ctx = backend_manager.get_model_n_ctx(canonical_name)
            if prompt_tokens >= min_ctx:
                continue
            slot_id, lock = self._get_free_or_oldest_from_pool(canonical_name, backend_id)
            if slot_id is None:
                continue
            try:
                await asyncio.wait_for(lock.acquire(), timeout=lock_timeout)
            except asyncio.TimeoutError:
                log.warning(
                    "Lock timed out after %ds for model '%s' on backend '%s' slot %d, trying next backend",
                    canonical_name, backend_id, slot_id, lock_timeout,
                )
                continue
            self._last_used[(canonical_name, backend_id, slot_id)] = time.time()
            effective_restore_key: Optional[str] = None
            return await self._restore_and_return(
                canonical_name, backend_id, slot_id, lock,
                effective_restore_key, blocks,
            )

        raise RuntimeError(f"No slots available for candidate_backends={len(candidate_backends)}")

    async def _restore_and_return(
        self,
        model_name: str,
        backend_id: str,
        slot_id: int,
        lock: asyncio.Lock,
        effective_restore_key: Optional[str],
        blocks: Optional[List[str]],
    ) -> Tuple[GSlot, asyncio.Lock, Optional[bool]]:
        """Restore KV cache and return (g, lock, restored)."""
        g = (model_name, backend_id, slot_id)
        restored: Optional[bool] = None

        if blocks:
            if self._should_skip_restore(g, blocks):
                log.info(
                    "Skipping restore for model '%s' on backend '%s' slot %d: slot cache already matches",
                    model_name, backend_id, slot_id,
                )
                restored = False
            elif effective_restore_key:
                client = backend_manager.get_client(backend_id)
                restored = await client.restore_slot(slot_id, effective_restore_key, model_name)
                log.info(
                    "Restored cache for model '%s' on backend '%s' slot %d: ok=%s",
                    model_name, backend_id, slot_id, restored,
                )
                if restored:
                    self._touch_ring(effective_restore_key, backend_id)
                    blocks = hs.get_meta_blocks(effective_restore_key, backend_id)
                    if blocks is not None:
                        self._slot_kv_state[g] = blocks
            else:
                client = backend_manager.get_client(backend_id)
                cand = hs.find_best_restore_candidate(
                    blocks, WORDS_PER_BLOCK, LCP_TH, model_name, backend_id,
                )
                if cand:
                    cand_key, cand_ratio = cand
                    restored = await client.restore_slot(slot_id, cand_key, model_name)
                    log.info(
                        "Dynamically restored cache for model '%s' on backend '%s' slot %d (key %s, ratio %.3f): ok=%s",
                        model_name, backend_id, slot_id, cand_key[:16], cand_ratio, restored,
                    )
                    if restored:
                        blocks = hs.get_meta_blocks(cand_key, backend_id)
                        if blocks is not None:
                            self._slot_kv_state[g] = blocks
                else:
                    log.info(
                        "No dynamic restore candidate for model '%s' on backend '%s' slot %d",
                        model_name, backend_id, slot_id,
                    )
                    restored = False

        elif effective_restore_key:
            client = backend_manager.get_client(backend_id)
            restored = await client.restore_slot(slot_id, effective_restore_key, model_name)
            log.info(
                "Restored cache for model '%s' on backend '%s' slot %d: ok=%s",
                model_name, backend_id, slot_id, restored,
            )
            if restored:
                self._touch_ring(effective_restore_key, backend_id)

        return g, lock, restored

    async def save_after(
        self,
        model_name: str,
        backend_id: str,
        slot_id: int,
        key: str,
        blocks: Optional[List[str]] = None,
    ) -> Tuple[bool, int]:
        client = backend_manager.get_client(backend_id)
        ok, size = await client.save_slot(slot_id, key, model_name)

        if ok and size > 0:
            ring = self._cache_ring.setdefault(backend_id, deque())
            ring.append((key, size, time.time()))
            self._total_bytes[backend_id] = self._total_bytes.get(backend_id, 0) + size

            # Update tracked KV state for this slot
            if blocks:
                    g = (model_name, backend_id, slot_id)
                    self._slot_kv_state[g] = blocks
                    log.debug(
                        "Updated KV cache state for model '%s' on backend '%s' slot %d: %d blocks",
                        model_name, backend_id, slot_id, len(blocks),
                    )

            # Ring buffer eviction: age-first, then LRU (per backend)
            max_bytes = CACHE_MAX_SIZE_GB * 1024**3
            now = time.time()
            while self._total_bytes.get(backend_id, 0) > max_bytes and ring:
                # First pass: evict expired entries
                evicted_expired = False
                for entry in ring:
                    if now - entry[2] > self._max_age_seconds:
                        evict_key, evict_size, _ = entry
                        ring.remove(entry)
                        self._total_bytes[backend_id] -= evict_size
                        age_hours = (now - entry[2]) / 3600
                        await self._evict_cache_file(evict_key, backend_id, "ring_evict_expired",
                                                     "(%d bytes, age=%.1fh)" % (evict_size, age_hours))
                        self._evict_meta_file(evict_key)
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
                await self._evict_cache_file(evict_key, backend_id, "ring_evict_lru",
                                             "(%d bytes, last_used=%.0f)" % (evict_size, lru_ts))
                self._evict_meta_file(evict_key)

        return ok, size

    def release(self, model_name: str, backend_id: str, slot_id: int):
        g = (model_name, backend_id, slot_id)
        lock = self._locks.get(g)
        if lock and lock.locked():
            lock.release()
            self._last_used[g] = 0.0

    def _touch_ring(self, key: str, backend_id: str):
        """Update the last_used timestamp for a cache entry in the ring buffer."""
        ring = self._cache_ring.get(backend_id)
        if not ring:
            return
        for i in range(len(ring)):
            if ring[i][0] == key:
                ring[i] = (key, ring[i][1], time.time())
                break
