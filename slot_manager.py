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
from typing import List, Tuple, Dict, Optional, Set

import httpx

from config import META_DIR, KV_CACHE_SKIP_THRESHOLD, \
    KV_CACHE_SKIP_MAX_BLOCK_DIFF_PCT, WORDS_PER_BLOCK, \
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
        """Score all entries by staleness + redundancy, evict highest-scored until under budget."""
        max_bytes = backend_manager.get_cache_max_size_gb(self.backend_id) * 1024**3
        if self._total_bytes <= max_bytes:
            return

        now = time.time()

        # Load blocks for all entries (cache in dict for O(1) LCP lookups)
        blocks_map: Dict[str, List[str]] = {}
        valid_ring: deque = deque()
        for entry in self._cache_ring:
            key, size, ts = entry
            blocks = kv_meta.get_blocks(key, self.backend_id)
            if blocks:
                blocks_map[key] = blocks
                valid_ring.append(entry)
            else:
                # No meta — orphaned ring entry, evict immediately
                self._total_bytes -= size
                log.info(
                    "Evicting orphaned ring entry '%s' for backend '%s' (%d bytes)",
                    key[:16], self.backend_id, size,
                )
                await self._delete_entry(key, "ring_evict_orphan", "(%d bytes)" % size)
        self._cache_ring = valid_ring

        if not self._cache_ring or self._total_bytes <= max_bytes:
            return

        # Score: staleness is primary, redundancy is secondary modifier.
        # score = stale_seconds * (2 - uniqueness)
        #   uniqueness = 1.0 -> multiplier 1.0 (evict by age alone)
        #   uniqueness = 0.0 -> multiplier 2.0 (redundant entries count as "twice as stale")
        # This ensures unique entries still expire, just at half the priority of redundant ones.
        scored: List[Tuple[int, float]] = []
        ring_list = list(self._cache_ring)
        for i, (key, size, ts) in enumerate(ring_list):
            stale = now - ts
            blocks = blocks_map.get(key, [])
            if not blocks:
                scored.append((i, stale * 2.0))
                continue
            max_lcp_ratio = 0.0
            for j, (other_key, _, _) in enumerate(ring_list):
                if i == j:
                    continue
                other_blocks = blocks_map.get(other_key, [])
                if not other_blocks:
                    continue
                lcp = hs.lcp_blocks(blocks, other_blocks)
                ratio = lcp / max(1, len(blocks))
                if ratio > max_lcp_ratio:
                    max_lcp_ratio = ratio
            uniqueness = 1.0 - max_lcp_ratio
            score = stale * (2.0 - uniqueness)
            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Collect keys to evict (indices become stale after deque deletion)
        keys_to_evict: Set[str] = set()
        projected = self._total_bytes
        for idx, score in scored:
            if projected <= max_bytes:
                break
            entry = ring_list[idx]
            keys_to_evict.add(entry[0])
            projected -= entry[1]

        # Evict collected entries
        evicted_any = False
        remaining = deque()
        for entry in self._cache_ring:
            if entry[0] in keys_to_evict:
                evict_key, evict_size, evict_ts = entry
                self._total_bytes -= evict_size
                stale_hours = (now - evict_ts) / 3600
                score_val = next(s for i, s in scored if ring_list[i][0] == evict_key)
                log.info(
                    "Ring buffer eviction: evicted '%s' for backend '%s' "
                    "(%d bytes, score=%.0f, age=%.1fh, remaining=%d)",
                    evict_key[:16], self.backend_id, evict_size, score_val, stale_hours, self._total_bytes,
                )
                await self._delete_entry(evict_key, "ring_evict_scored",
                                        "(score=%.0f, age=%.1fh)" % (score_val, stale_hours))
                evicted_any = True
            else:
                remaining.append(entry)
        self._cache_ring = remaining

        if evicted_any:
            log.info(
                "Cache ring check for backend '%s': total=%d bytes, max=%d bytes, ring_size=%d",
                self.backend_id, self._total_bytes, max_bytes, len(self._cache_ring),
            )

    async def _delete_entry(self, key: str, log_msg: str, log_extra: str):
        """Delete cache file and meta."""
        ok = await backend_manager.cache_delete(self.backend_id, key)
        if ok:
            log.info("%s: %s %s", log_msg, key[:16], log_extra)
        else:
            log.warning("%s_agent_fail: %s", log_msg, key[:16])
        kv_meta.delete_meta_file(key)

    async def init_from_disk(self):
        """Populate ring buffer from existing cache files. Evict over-size."""
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
