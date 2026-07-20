# slot_manager.py

# -*- coding: utf-8 -*-

"""
Slot management split into per-backend managers and a thin global coordinator.

BackendSlotManager: manages slot state for a single backend — pools, KV tracking,
    save/restore, ring buffer eviction. Knows nothing about other backends.

SlotManager: thin global coordinator — creates/holds per-backend managers,
    aggregates cross-backend metrics (EMA latency, slot duration), and provides
    init_from_disk / refresh_slot_counts that delegate to each backend.

Routing scan and retry orchestration live in app.py, not here.
"""

import os
import time
import asyncio
import logging
from collections import deque
from typing import List, Tuple, Dict, Optional

import httpx

from config import META_DIR, CACHE_MAX_AGE_HOURS, \
    KV_CACHE_SKIP_THRESHOLD, KV_CACHE_SKIP_MAX_BLOCK_DIFF_PCT, WORDS_PER_BLOCK, \
    CACHE_HIT_WAIT_EMA_ALPHA, CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT, \
    CACHE_HIT_WAIT_EMA_MAX_TIMEOUT, CACHE_HIT_WAIT_EMA_MIN_TIMEOUT
import hashing as hs
from backend_manager import backend_manager
from kv_meta_manager import kv_meta

log = logging.getLogger(__name__)

# (canonical_model_name, backend_id, slot_id)
GSlot = Tuple[str, str, int]

# Sentinel for should_skip_restore: distinguishes "not passed" from "passed as None"
_NO_PREV = object()


class BackendSlotManager:
    """Manages slot state for a single backend.

    Owns: slot pools per model, KV cache state per slot, in-use flags,
    cache ring buffer with eviction.
    """

    def __init__(self, backend_id: str):
        self.backend_id = backend_id
        self._slot_pools: Dict[str, set[int]] = {}  # model -> slot ids
        self._in_use: Dict[int, bool] = {}
        self._last_used: Dict[int, float] = {}
        self._slot_kv_state: Dict[int, List[str]] = {}  # slot_id -> blocks
        self._slot_save_skipped: Dict[int, tuple] = {}
        self._slot_acquired_at: Dict[int, float] = {}
        self._slot_duration_ema: float = 0.0

        # Cache ring buffer
        self._cache_ring: deque = deque()
        self._total_bytes: int = 0
        self._max_age_seconds: float = CACHE_MAX_AGE_HOURS * 3600
        self._save_lock: asyncio.Lock = asyncio.Lock()

    # ── Slot pool management ──────────────────────────────────────────

    def ensure_pool(self, model_name: str, n_slots: int):
        """Create or update a slot pool for a model."""
        if model_name not in self._slot_pools:
            new_pool = set(range(n_slots))
            for s in new_pool:
                self._last_used[s] = 0.0
                self._in_use[s] = False
            self._slot_pools[model_name] = new_pool
            log.info(
                "Created slot pool for model '%s' on backend '%s' with %d slots",
                model_name, self.backend_id, n_slots,
            )
        else:
            old_pool = self._slot_pools[model_name]
            old_count = len(old_pool)
            new_pool = set(range(n_slots))
            for s in new_pool - old_pool:
                self._last_used[s] = 0.0
                self._in_use[s] = False
            for s in old_pool - new_pool:
                if not self._in_use.get(s, False):
                    self._last_used.pop(s, None)
                    self._in_use.pop(s, None)
                    self._slot_kv_state.pop(s, None)
                    self._slot_save_skipped.pop(s, None)
            self._slot_pools[model_name] = new_pool
            log.info(
                "Updated slot pool for model '%s' on backend '%s': %d -> %d slots",
                model_name, self.backend_id, old_count, n_slots,
            )

    def get_pool(self, model_name: str) -> Optional[set[int]]:
        """Return the slot pool for a model, or None."""
        return self._slot_pools.get(model_name)

    # ── Slot acquisition / release ────────────────────────────────────

    def try_acquire(self, model_name: str) -> Optional[int]:
        """Pick a free slot (or oldest LRU) and mark it in-use. Returns slot_id or None."""
        pool = self._slot_pools.get(model_name)
        if not pool:
            return None

        free = [s for s in pool if not self._in_use.get(s, False)]
        if free:
            slot_id = free[0]
        else:
            slot_id = min(pool, key=lambda s: self._last_used.get(s, 0.0))

        if self._in_use.get(slot_id, False):
            return None  # all slots in-use and we grabbed one

        now = time.time()
        self._in_use[slot_id] = True
        self._last_used[slot_id] = now
        self._slot_acquired_at[slot_id] = now
        return slot_id

    def release(self, slot_id: int) -> Optional[float]:
        """Release a slot. Returns occupancy duration in seconds, or None."""
        self._in_use[slot_id] = False
        if slot_id in self._slot_acquired_at:
            duration = time.time() - self._slot_acquired_at[slot_id]
            del self._slot_acquired_at[slot_id]
            old = self._slot_duration_ema if self._slot_duration_ema > 0 else CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT
            self._slot_duration_ema = CACHE_HIT_WAIT_EMA_ALPHA * duration + (1 - CACHE_HIT_WAIT_EMA_ALPHA) * old
            self._slot_duration_ema = max(
                min(self._slot_duration_ema, CACHE_HIT_WAIT_EMA_MAX_TIMEOUT),
                CACHE_HIT_WAIT_EMA_MIN_TIMEOUT,
            )
            return duration
        return None

    def invalidate(self, slot_id: int):
        """Clear KV cache tracking for a slot whose cache is in an unknown state."""
        if slot_id in self._slot_kv_state:
            del self._slot_kv_state[slot_id]
            log.info(
                "Invalidated KV cache tracking for backend '%s' slot %d",
                self.backend_id, slot_id,
            )

    # ── KV cache state ────────────────────────────────────────────────

    def get_kv_states(self) -> Dict[int, List[str]]:
        """Return all slot_id -> blocks mappings (for routing scan)."""
        return dict(self._slot_kv_state)

    def get_kv_state(self, slot_id: int) -> Optional[List[str]]:
        return self._slot_kv_state.get(slot_id)

    def set_kv_state(self, slot_id: int, blocks: List[str]):
        self._slot_kv_state[slot_id] = blocks

    def get_slot_duration_ema(self) -> float:
        """Return the EMA slot occupancy duration for this backend."""
        return self._slot_duration_ema if self._slot_duration_ema > 0 else CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT

    def should_skip_restore(self, slot_id: int, req_blocks: List[str],
                              prev_blocks: Optional[List[str]] = _NO_PREV) -> bool:
        """Check if the slot's current KV cache already matches the request.

        Only applies for single-slot backends. Three conditions must all pass:
        1. Slot must not have more blocks than the request (stale leftover state)
        2. Block count difference must be within KV_CACHE_SKIP_MAX_BLOCK_DIFF_PCT
        3. LCP ratio must be >= KV_CACHE_SKIP_THRESHOLD

        prev_blocks: if provided, use this as the slot's previous KV state instead of
        reading from _slot_kv_state. This is needed when _slot_kv_state was already
        updated to the request's own blocks before the skip-restore check runs.
        Pass None explicitly for a fresh slot with no prior state (always returns False).
        """
        if prev_blocks is _NO_PREV:
            # Not explicitly passed — fallback to _slot_kv_state (tests, legacy callers)
            prev_blocks = self._slot_kv_state.get(slot_id)
            log.warning(
                "[diag] should_skip_restore: slot %d, prev_blocks not passed, fell back to _slot_kv_state: %d blocks",
                slot_id, len(prev_blocks) if prev_blocks else 0,
            )
        elif prev_blocks is None:
            # Explicitly passed None — fresh slot, no prior state to skip against
            log.warning(
                "[diag] should_skip_restore: slot %d, prev_blocks explicitly None (fresh slot), not skipping",
                slot_id,
            )
            return False

        if not prev_blocks:
            return False

        # Check if any pool for this backend has more than 1 slot
        for pool in self._slot_pools.values():
            if len(pool) > 1:
                return False

        n_prev = len(prev_blocks)
        n_req = len(req_blocks)

        # Slot has more blocks than request — stale leftover state, can't skip
        if n_prev > n_req:
            log.warning(
                "Cannot skip restore for backend '%s' slot %d: slot has %d blocks, request needs %d (stale state)",
                self.backend_id, slot_id, n_prev, n_req,
            )
            return False

        # Block count difference exceeds threshold
        diff = n_req - n_prev
        max_len = max(n_prev, n_req)
        diff_pct = diff / max_len if max_len > 0 else 0
        if diff_pct > KV_CACHE_SKIP_MAX_BLOCK_DIFF_PCT:
            log.warning(
                "Cannot skip restore for backend '%s' slot %d: block diff %d/%d (%.1f%%) > %.0f%% threshold",
                self.backend_id, slot_id, diff, max_len, diff_pct * 100,
                KV_CACHE_SKIP_MAX_BLOCK_DIFF_PCT * 100,
            )
            return False

        lcp = hs.lcp_blocks(req_blocks, prev_blocks)
        denom = max(1, min(n_req, n_prev))
        ratio = lcp / denom
        log.warning(
            "Checking skip restore for backend '%s' slot %d: ratio=%.3f, blocks %d->%d",
            self.backend_id, slot_id, ratio, n_prev, n_req,
        )
        return ratio >= KV_CACHE_SKIP_THRESHOLD

    # ── Save / skip tracking ──────────────────────────────────────────

    def mark_save_skipped(self, slot_id: int, entry: tuple):
        self._slot_save_skipped[slot_id] = entry

    def flush_save_skipped(self, slot_id: int) -> Optional[tuple]:
        """Remove and return the skipped save entry for a slot, if any."""
        return self._slot_save_skipped.pop(slot_id, None)

    # ── Restore ───────────────────────────────────────────────────────

    async def restore(self, slot_id: int, key: str, model_name: str,
                      touch_ring: bool = False) -> bool:
        """Call llama.cpp to restore KV cache into a slot."""
        client = backend_manager.get_client(self.backend_id)
        try:
            restored = await client.restore_slot(slot_id, key, model_name)
        except (httpx.ConnectError, httpx.RemoteProtocolError,
                httpx.ReadError, httpx.ReadTimeout) as e:
            log.warning(
                "Restore failed for model '%s' on backend '%s' slot %d (key %s): %s",
                model_name, self.backend_id, slot_id, key[:16], e,
            )
            restored = False
        except Exception as e:
            log.warning(
                "Unexpected error restoring cache for model '%s' on backend '%s' "
                "slot %d (key %s): %s",
                model_name, self.backend_id, slot_id, key[:16], e,
            )
            restored = False

        log.info(
            "Restore for model '%s' on backend '%s' slot %d (key %s): ok=%s",
            model_name, self.backend_id, slot_id, key[:16], restored,
        )
        if restored and touch_ring:
            self._touch_ring(key)
        return restored

    # ── Save + ring buffer ────────────────────────────────────────────

    async def save_after(self, model_name: str, slot_id: int,
                         key: str, blocks: Optional[List[str]],
                         n_tokens: int) -> Tuple[bool, int]:
        """Save slot to disk, write meta, update ring buffer, evict if needed."""
        client = backend_manager.get_client(self.backend_id)
        ok, size = await client.save_slot(slot_id, key, model_name)

        if blocks:
            self._slot_kv_state[slot_id] = blocks
            log.warning(
                "Updated KV cache state for model '%s' on backend '%s' slot %d: %d blocks",
                model_name, self.backend_id, slot_id, len(blocks),
            )

        if ok and size > 0:
            async with self._save_lock:
                try:
                    kv_meta.write_meta(
                        key, n_tokens, blocks, WORDS_PER_BLOCK,
                        model_name, self.backend_id, size,
                    )
                except Exception as e:
                    log.warning("Failed to write meta file for key %s: %s", key[:16], e)

                self._cache_ring.append((key, size, time.time()))
                self._total_bytes += size

                await self._evict_if_needed()

        return ok, size

    async def _evict_if_needed(self):
        """Evict expired then LRU entries if over budget."""
        max_bytes = backend_manager.get_cache_max_size_gb(self.backend_id) * 1024**3
        now = time.time()
        log.info(
            "Cache ring check for backend '%s': total=%d bytes, max=%d bytes, ring_size=%d",
            self.backend_id, self._total_bytes, max_bytes, len(self._cache_ring),
        )
        while self._total_bytes > max_bytes and self._cache_ring:
            # First pass: evict expired entries
            evicted = False
            for i, entry in enumerate(self._cache_ring):
                if now - entry[2] > self._max_age_seconds:
                    evict_key, evict_size, _ = entry
                    del self._cache_ring[i]
                    self._total_bytes -= evict_size
                    age_hours = (now - entry[2]) / 3600
                    await self._delete_entry(evict_key, "ring_evict_expired",
                                       "(%d bytes, age=%.1fh)" % (evict_size, age_hours))
                    evicted = True
                    break
            if evicted:
                continue

            # Second pass: evict LRU
            lru_idx = min(range(len(self._cache_ring)),
                          key=lambda i: self._cache_ring[i][2])
            evict_key, evict_size, _ = self._cache_ring[lru_idx]
            del self._cache_ring[lru_idx]
            self._total_bytes -= evict_size
            log.info(
                "Ring buffer eviction: evicted LRU entry '%s' for backend '%s' "
                "(%d bytes, remaining=%d)",
                evict_key, self.backend_id, evict_size, self._total_bytes,
            )
            await self._delete_entry(evict_key, "ring_evict_lru",
                                "(%d bytes)" % evict_size)

    async def _delete_entry(self, key: str, log_msg: str, log_extra: str):
        """Delete cache file and meta."""
        ok = await backend_manager.cache_delete(self.backend_id, key)
        if ok:
            log.info("%s: %s %s", log_msg, key[:16], log_extra)
        else:
            log.warning("%s_agent_fail: %s", log_msg, key[:16])
        kv_meta.delete_meta_file(key)

    async def init_from_disk(self):
        """Populate ring buffer from existing cache files. Evict expired/over-size."""
        backend_dir = hs.sanitize_backend_dir(self.backend_id)
        if not os.path.isdir(META_DIR):
            return

        meta_path = os.path.join(META_DIR, backend_dir)
        if not os.path.isdir(meta_path):
            return

        for key in kv_meta.list_keys(backend_dir):
            try:
                cache_size = await kv_meta.get_cache_size(key, backend_dir)
                if not cache_size:
                    continue
                last_used = kv_meta.get_last_used_time(key, backend_dir)
                self._cache_ring.append((key, cache_size, last_used))
                self._total_bytes += cache_size
            except OSError:
                continue

        # Evict expired entries from front of ring
        now = time.time()
        while self._cache_ring:
            entry = self._cache_ring[0]
            if now - entry[2] > self._max_age_seconds:
                evict_key, evict_size, _ = entry
                self._cache_ring.popleft()
                self._total_bytes -= evict_size
                log.info(
                    "Startup cleanup: evicting expired entry '%s' for backend '%s' (%d bytes)",
                    evict_key, self.backend_id, evict_size,
                )
                await self._delete_entry(evict_key, "startup_evict",
                                         "(%d bytes)" % evict_size)
            else:
                break

        await self._evict_if_needed()

    def _touch_ring(self, key: str):
        """Update last_used timestamp for a cache entry."""
        for i in range(len(self._cache_ring)):
            if self._cache_ring[i][0] == key:
                self._cache_ring[i] = (key, self._cache_ring[i][1], time.time())
                break

    def get_ring_size(self) -> int:
        return len(self._cache_ring)

    def get_total_bytes(self) -> int:
        return self._total_bytes


# ── Global coordinator ─────────────────────────────────────────────────


class SlotManager:
    """Thin global coordinator — holds per-backend BackendSlotManager instances.

    Delegates slot operations to per-backend managers. Owns cache wait queue
    tracking and startup initialization. Per-backend metrics (EMA latency,
    slot duration, last used) live on BackendManager/BackendSlotManager.
    """

    def __init__(self):
        self._backends: Dict[str, BackendSlotManager] = {}
        self._cache_wait_pending: Dict[str, int] = {}
        log.info("Cache entry expiry set to %d hours", CACHE_MAX_AGE_HOURS)

    def get(self, backend_id: str) -> BackendSlotManager:
        """Get or create the BackendSlotManager for a backend."""
        if backend_id not in self._backends:
            self._backends[backend_id] = BackendSlotManager(backend_id)
        return self._backends[backend_id]

    def all_kv_states(self) -> Dict[Tuple[str, int], List[str]]:
        """Return {(backend_id, slot_id): blocks} for all backends (for routing scan)."""
        result = {}
        for be_id, be_sm in self._backends.items():
            for slot_id, blocks in be_sm.get_kv_states().items():
                result[(be_id, slot_id)] = blocks
        return result

    # ── Startup / refresh ─────────────────────────────────────────────

    async def init_from_disk(self):
        """Initialize ring buffers for all configured backends."""
        total_loaded = 0
        total_bytes_loaded = 0
        for be_id in backend_manager.keys():
            be_sm = self.get(be_id)
            await be_sm.init_from_disk()
            total_loaded += be_sm.get_ring_size()
            total_bytes_loaded += be_sm.get_total_bytes()

        per_backend = [(bid, self._backends[bid].get_total_bytes(),
                        self._backends[bid].get_ring_size())
                       for bid in self._backends if bid in self._backends]
        log.info(
            "Loaded %d cache entries from disk (%.1f GB), per-backend: %s",
            total_loaded, total_bytes_loaded / 1024**3,
            "; ".join(f"{bid}: {sz / 1024**3:.1f} GB ({cnt} files)"
                      for bid, sz, cnt in per_backend),
        )

    async def refresh_slot_counts(self):
        """Query backends for slot counts and update all per-backend pools."""
        try:
            slot_counts = await backend_manager.refresh_slot_counts()
        except Exception as e:
            log.warning("Failed to refresh slot counts: %s — proceeding with existing pool state", e)
            slot_counts = {}

        for backend_key, model_slots in slot_counts.items():
            be_sm = self.get(backend_key)
            for canonical_name, n_slots in model_slots.items():
                be_sm.ensure_pool(canonical_name, n_slots)
