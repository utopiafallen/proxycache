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
    KV_CACHE_SKIP_THRESHOLD, LCP_TH, WORDS_PER_BLOCK, SLOT_TIMEOUT
import hashing as hs
from backend_manager import backend_manager

log = logging.getLogger(__name__)

GSlot = Tuple[str, str, int]  # (model_name, backend_key, slot_id)


class SlotManager:
    def __init__(self):
        self._slot_pools: Dict[str, Dict[str, set]] = {}
        self._last_used: Dict[Tuple[str, str, int], float] = {}
        self._locks: Dict[Tuple[str, str, int], asyncio.Lock] = {}

        # Ring buffer for cache size tracking
        # (key, size_bytes, last_used_time, backend_key)
        self._cache_ring: deque = deque()
        self._total_bytes: int = 0
        self._max_age_seconds: float = CACHE_MAX_AGE_HOURS * 3600

        # Per-slot KV cache block state — tracks hash blocks currently in each slot
        self._slot_kv_state: Dict[GSlot, List[str]] = {}

        log.info("slot_manager max_age_hours=%d", CACHE_MAX_AGE_HOURS)

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
        meta_path = os.path.join(META_DIR, f"{key}{hs.META_SUFFIX}")
        if os.path.exists(meta_path):
            try:
                os.remove(meta_path)
            except OSError:
                pass

    def init_from_disk(self, cache_dir: str):
        """Populate ring buffer from existing cache files on disk."""
        if not cache_dir or not os.path.isdir(cache_dir):
            return
        for f in os.listdir(cache_dir):
            filepath = os.path.join(cache_dir, f)
            if os.path.isfile(filepath):
                try:
                    size = os.stat(filepath).st_size
                    last_used = hs._get_last_used_time(f, META_DIR, cache_dir)
                    self._cache_ring.append((f, size, last_used, None))  # None = unknown backend
                    self._total_bytes += size
                except OSError:
                    continue
        log.info("init_from_disk: %d cache files, %.1f GB total",
                  len(self._cache_ring), self._total_bytes / 1024**3)

    def _is_free(self, model_name: str, backend_id: str, slot_id: int) -> bool:
        return self._last_used.get((model_name, backend_id, slot_id), 0.0) == 0.0

    def _get_free_or_oldest_from_pool(
        self, model_name: str, backend_id: str
    ) -> Tuple[int, asyncio.Lock]:
        """Pick free slot or oldest (LRU) from a single backend's pool for a model."""
        pool = self._slot_pools.get(model_name, {}).get(backend_id)
        if not pool:
            raise RuntimeError(f"No pool for model={model_name} be={backend_id}")

        free = [s for s in pool if self._is_free(model_name, backend_id, s)]
        if free:
            return free[0], self._locks[(model_name, backend_id, free[0])]

        oldest = min(pool, key=lambda s: self._last_used.get((model_name, backend_id, s), 0.0))
        return oldest, self._locks[(model_name, backend_id, oldest)]

    def _select_from_pool(self, model_name: str) -> Tuple[str, str, int, asyncio.Lock]:
        """Pick the best backend + slot for a model (free or oldest LRU)."""
        best: Optional[Tuple[str, str, int, asyncio.Lock]] = None
        best_ts = -1.0

        backend_ids = backend_manager.get_backends_for_model(model_name)
        for backend_id in backend_ids:
            try:
                slot_id, lock = self._get_free_or_oldest_from_pool(model_name, backend_id)
            except RuntimeError:
                continue

            ts = self._last_used.get((model_name, backend_id, slot_id), 0.0)

            # Prefer free slots, then prefer oldest (lowest ts)
            is_free = ts == 0.0
            if is_free and best is not None:
                best_is_free = best[3] is not None and self._last_used.get((best[0], best[1], best[2]), 0.0) == 0.0
                if not best_is_free:
                    best = (model_name, backend_id, slot_id, lock)
                    continue
                else:
                    continue  # both free, keep first

            if not is_free and best is not None:
                best_is_free = self._last_used.get((best[0], best[1], best[2]), 0.0) == 0.0
                if best_is_free:
                    continue  # keep the free one

            if best is None or ts < best_ts:
                best = (model_name, backend_id, slot_id, lock)
                best_ts = ts

        if best is None:
            raise RuntimeError(f"No slots available for model={model_name}")

        return best

    def _ensure_pool(self, model_name: str, backend_id: str, n_slots: int):
        """Create or update a slot pool for (model_name, backend_key)."""
        if model_name not in self._slot_pools:
            self._slot_pools[model_name] = {}
        if backend_id not in self._slot_pools[model_name]:
            old_pool = set()
            new_pool = set(range(n_slots))
            # Add new slots
            for s in range(n_slots):
                if s not in old_pool:
                    g = (model_name, backend_id, s)
                    self._last_used[g] = 0.0
                    self._locks[g] = asyncio.Lock()
            self._slot_pools[model_name][backend_id] = new_pool
            log.info(
                "ensure_pool model=%s be=%s slots=%d",
                model_name, backend_id, n_slots,
            )
        else:
            old_count = len(self._slot_pools[model_name][backend_id])
            old_pool = self._slot_pools[model_name][backend_id]
            new_pool = set(range(n_slots))
            # Add new slots
            for s in new_pool - old_pool:
                g = (model_name, backend_id, s)
                self._last_used[g] = 0.0
                self._locks[g] = asyncio.Lock()
            # Remove old slots (only free ones)
            for s in old_pool - new_pool:
                if self._is_free(model_name, backend_id, s):
                    self._slot_pools[model_name][backend_id].discard(s)
                    self._last_used.pop((model_name, backend_id, s), None)
                    self._locks.pop((model_name, backend_id, s), None)
            self._slot_pools[model_name][backend_id] = new_pool
            log.info(
                "update_pool model=%s be=%s slots %d->%d",
                model_name, backend_id, old_count, n_slots,
            )

    async def refresh_slots(self, model_name: str):
        """Refresh slot counts for a model across all backends.

        Uses backend_manager.refresh_models() which handles both router and non-router modes.
        Falls back to 1 slot if discovery fails (model not loaded yet).
        """
        slot_counts = await backend_manager.refresh_models(model_name)

        # Update pools for backends that returned slot info
        for backend_key, n_slots in slot_counts.items():
            self._ensure_pool(model_name, backend_key, n_slots)

        # Fallback: if model has no registered backends, use first backend with 1 slot
        if not backend_manager.get_backends_for_model(model_name):
            first_key = backend_manager.first_key()
            self._ensure_pool(model_name, first_key, 1)

    def _should_skip_restore(self, g: GSlot, req_blocks: List[str]) -> bool:
        """Check if the slot's current KV cache already matches the request well enough.

        Returns True if the slot's tracked KV cache blocks overlap >= KV_CACHE_SKIP_THRESHOLD
        with the request blocks, meaning no restore is needed.
        """
        kv_blocks = self._slot_kv_state.get(g)
        if not kv_blocks:
            return False

        lcp = hs.lcp_blocks(req_blocks, kv_blocks)
        denom = max(1, min(len(req_blocks), len(kv_blocks)))
        ratio = lcp / denom
        log.debug(
            "_should_skip_restore model=%s be=%s slot=%d ratio=%.3f",
            g[0], g[1], g[2], ratio,
        )
        return ratio >= KV_CACHE_SKIP_THRESHOLD

    async def acquire_for_request(
        self,
        model_name: str,
        restore_key: Optional[str] = None,
        blocks: Optional[List[str]] = None,
    ) -> Tuple[GSlot, asyncio.Lock, Optional[bool]]:
        # Refresh before selecting (with cooldown)
        await self.refresh_slots(model_name)

        # Select best backend + slot
        model_name_out, backend_id, slot_id, lock = self._select_from_pool(model_name)
        # Cap lock acquire timeout at 30s — don't let requests hang forever
        lock_timeout = min(30.0, SLOT_TIMEOUT)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=lock_timeout)
        except asyncio.TimeoutError:
            log.warning(
                "acquire_lock_timeout model=%s be=%s slot=%d after %ds",
                model_name, backend_id, slot_id, lock_timeout,
            )
            raise
        self._last_used[(model_name_out, backend_id, slot_id)] = time.time()

        g = (model_name_out, backend_id, slot_id)

        # Restore if needed
        restored: Optional[bool] = None
        try:
            if blocks:
                # Skip restore if slot's KV cache already matches well
                if self._should_skip_restore(g, blocks):
                    log.info(
                        "skip_restore_slot_cached model=%s be=%s slot=%d",
                        model_name, backend_id, slot_id,
                    )
                    restored = False
                elif restore_key:
                    # Restore from pre-computed candidate
                    client = backend_manager.get_client(backend_id)
                    restored = await client.restore_slot(slot_id, restore_key, model_name)
                    log.info(
                        "restore_before_chat model=%s be=%s slot=%d ok=%s",
                        model_name, backend_id, slot_id, restored,
                    )
                    if restored:
                        self._touch_ring(restore_key)
                        # Update tracked KV state with restored candidate blocks
                        try:
                            meta_path = os.path.join(META_DIR, f"{restore_key}{hs.META_SUFFIX}")
                            if os.path.exists(meta_path):
                                with open(meta_path, "r", encoding="utf-8") as mf:
                                    meta = json.load(mf)
                                self._slot_kv_state[g] = meta.get("blocks", [])
                        except Exception as e:
                            log.warning("restore_meta_load_fail key=%s: %s", restore_key[:16], e)
                else:
                    # No pre-computed candidate and KV cache doesn't match —
                    # find best meta file to restore from
                    client = backend_manager.get_client(backend_id)
                    cand = hs.find_best_restore_candidate(
                        blocks, WORDS_PER_BLOCK, LCP_TH, model_name,
                    )
                    if cand:
                        cand_key, cand_ratio = cand
                        restored = await client.restore_slot(slot_id, cand_key, model_name)
                        log.info(
                            "restore_dynamic model=%s be=%s slot=%d key=%s ratio=%.3f ok=%s",
                            model_name, backend_id, slot_id, cand_key[:16], cand_ratio, restored,
                        )
                        if restored:
                            try:
                                meta_path = os.path.join(META_DIR, f"{cand_key}{hs.META_SUFFIX}")
                                if os.path.exists(meta_path):
                                    with open(meta_path, "r", encoding="utf-8") as mf:
                                        meta = json.load(mf)
                                    self._slot_kv_state[g] = meta.get("blocks", [])
                            except Exception as e:
                                log.warning("restore_meta_load_fail key=%s: %s", cand_key[:16], e)
                    else:
                        log.info(
                            "restore_dynamic_none model=%s be=%s slot=%d",
                            model_name, backend_id, slot_id,
                        )
                        restored = False

            elif restore_key:
                # Legacy path: no blocks passed but restore_key provided
                client = backend_manager.get_client(backend_id)
                restored = await client.restore_slot(slot_id, restore_key, model_name)
                log.info(
                    "restore_before_chat model=%s be=%s slot=%d ok=%s",
                    model_name, backend_id, slot_id, restored,
                )
                if restored:
                    self._touch_ring(restore_key)

            return g, lock, restored
        except Exception:
            # Clean up leaked KV state on failure so the slot doesn't get
            # incorrectly skipped on its next use
            self._slot_kv_state.pop(g, None)
            # Release lock so the slot isn't held forever after a failure
            if lock.locked():
                lock.release()
            raise

    async def save_after(
        self,
        model_name: str,
        backend_id: str,
        slot_id: int,
        key: str,
        model_id: Optional[str] = None,
        blocks: Optional[List[str]] = None,
    ) -> Tuple[bool, int]:
        client = backend_manager.get_client(backend_id)
        ok, size = await client.save_slot(slot_id, key, model_id)

        if ok and size > 0:
            self._cache_ring.append((key, size, time.time(), backend_id))
            self._total_bytes += size

            # Update tracked KV state for this slot
            if blocks:
                g = (model_name, backend_id, slot_id)
                self._slot_kv_state[g] = blocks
                log.debug(
                    "update_slot_kv_state model=%s be=%s slot=%d n_blocks=%d",
                    model_name, backend_id, slot_id, len(blocks),
                )

            # Ring buffer eviction: age-first, then LRU
            max_bytes = CACHE_MAX_SIZE_GB * 1024**3
            now = time.time()
            while self._total_bytes > max_bytes and self._cache_ring:
                # First pass: evict expired entries
                evicted_expired = False
                for entry in self._cache_ring:
                    if now - entry[2] > self._max_age_seconds:
                        evict_key, evict_size, _, entry_be_id = entry
                        self._cache_ring.remove(entry)
                        self._total_bytes -= evict_size
                        age_hours = (now - entry[2]) / 3600
                        await self._evict_cache_file(evict_key, entry_be_id, "ring_evict_expired",
                                                     "(%d bytes, age=%.1fh)" % (evict_size, age_hours))
                        self._evict_meta_file(evict_key)
                        evicted_expired = True
                        break
                if evicted_expired:
                    continue

                # Second pass: evict LRU entry (no expired entries left)
                lru_idx = 0
                lru_ts = self._cache_ring[0][2]
                for i in range(1, len(self._cache_ring)):
                    if self._cache_ring[i][2] < lru_ts:
                        lru_ts = self._cache_ring[i][2]
                        lru_idx = i
                evict_key, evict_size, _, entry_be_id = self._cache_ring[lru_idx]
                self._cache_ring.remove(self._cache_ring[lru_idx])
                self._total_bytes -= evict_size
                await self._evict_cache_file(evict_key, entry_be_id, "ring_evict_lru",
                                             "(%d bytes, last_used=%.0f)" % (evict_size, lru_ts))
                self._evict_meta_file(evict_key)

        return ok, size

    def release(self, model_name: str, backend_id: str, slot_id: int):
        g = (model_name, backend_id, slot_id)
        lock = self._locks.get(g)
        if lock and lock.locked():
            lock.release()
            self._last_used[g] = 0.0

    def _touch_ring(self, key: str):
        """Update the last_used timestamp for a cache entry in the ring buffer."""
        for i in range(len(self._cache_ring)):
            if self._cache_ring[i][0] == key:
                self._cache_ring[i] = (key, self._cache_ring[i][1], time.time(), self._cache_ring[i][3])
                break
