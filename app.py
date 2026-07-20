# app.py

# -*- coding: utf-8 -*-

"""
KV Proxy for llama.cpp with disk save/restore:

- Big requests: LCP matching → restore from disk, then chat strictly in that slot, then save+meta.
- Small requests: free/oldest slot, no restore, no disk save/meta.
- Slot pinning is duplicated in root/options/query (via client).

Additionally:

- Slot acquisition is wrapped in a timeout to avoid hanging forever if a slot is never released.
- Streaming:
    * socket reads from llama.cpp run in a background task (reader),
      racing against a disconnect event;
    * reader pushes chunks into asyncio.Queue;
    * stream()'s finally calls _cleanup() — save (if _stream_complete),
      release, and puts a sentinel in the queue;
    * heartbeat task checks is_disconnected() every 0.5s.
- Slot pools are per-model with lazy discovery.
- Cache eviction via ring buffer in BackendSlotManager (age-first, then LRU).
"""

import asyncio
import json
import os
import time
import uuid
import logging
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple

import httpx
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

from config import (BACKENDS, WORDS_PER_BLOCK,
                    LCP_TH, MODEL_ID, PORT, DEFAULT_N_CTX,
                    should_save_cache,
                    CACHE_HIT_WAIT_EMA_MIN_TIMEOUT, CACHE_HIT_WAIT_EMA_MAX_TIMEOUT,
                    CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT, CACHE_HIT_WAIT_MAX_PENDING_REQS)

import hashing as hs
from slot_manager import SlotManager, BackendSlotManager
from metrics import metrics, extract_prompt_preview

log = logging.getLogger(__name__)

from backend_manager import backend_manager
from kv_meta_manager import kv_meta

ACQUIRE_TIMEOUT = 60.0
STREAM_QUEUE_SIZE = 16
STREAM_QUEUE_TIMEOUT = 5.0
RECOMPUTE_THRESHOLD_PERCENT_REQ_TOKENS = 0.92


class CacheHitType(str, Enum):
    """Explicit routing decision — tells downstream what to do with a slot."""
    DISK_RESTORE = "disk_restore"   # disk cache file found, restore from specific key
    SKIP = "skip"                    # pending slot hit, slot already has matching KV cache


app = FastAPI(title="Simple KV Proxy")


@app.on_event("startup")
async def startup():
    sm = SlotManager()
    await sm.init_from_disk()

    app.state.sm = sm
    app.state.executor = ThreadPoolExecutor(max_workers=2)
    log.info("Starting on port %d with %d backends: %s", PORT, len(BACKENDS), [be["url"] for be in BACKENDS])

    # Reconcile meta files on startup (remove corrupted/orphaned entries)
    backend_keys = list(backend_manager._backends.keys())
    reconciled = await kv_meta.reconcile(backend_keys)
    if reconciled > 0:
        log.info("Cleaned up %d orphaned/corrupted meta files at startup", reconciled)

    # Log startup sanity summary
    meta_count = sum(len(kv_meta.list_keys(k)) for k in backend_keys)
    cache_dirs = [k for k in backend_keys if backend_manager.get_cache_dir(k)]
    cache_files = sum(1 for k in cache_dirs for f in os.listdir(backend_manager.get_cache_dir(k)) if os.path.isfile(os.path.join(backend_manager.get_cache_dir(k), f)))
    log.info("After startup reconcile: %d meta files, %d cache files on disk", meta_count, cache_files)

    # Start liveness checker
    await backend_manager.start_liveness_checker()


@app.on_event("shutdown")
async def shutdown():
    await backend_manager.stop_liveness_checker()
    executor = getattr(app.state, "executor", None)
    await backend_manager.close()
    if executor:
        executor.shutdown(wait=False)


@app.get("/v1/models")
async def models():
    discovered = backend_manager._discovered_models
    models_list = []
    for name, info in discovered.items():
        models_list.append({"id": name, "object": "model", "owned_by": "backend", "n_ctx": info.n_ctx})
    min_ctx = min(m.n_ctx for m in discovered.values()) if discovered else DEFAULT_N_CTX
    models_list.append({"id": "any", "object": "model", "owned_by": "proxycache", "n_ctx": min_ctx})
    return {"data": models_list}


# ── Slot acquisition helpers (routing decision executed here) ──────────


async def _acquire_slot_for_request(
    sm: SlotManager,
    candidate_backends: list[tuple[str, str]],
    restore_backend: Optional[str],
    canonical_name: str,
    hit_type: Optional[CacheHitType],
    restore_key: Optional[str],
    backend_blocks: Dict[str, List[str]],
    prompt_tokens: int,
) -> Tuple[Tuple[str, str, int], Optional[bool]]:
    """Acquire a slot, trying cache backend first then fallbacks.

    Orchestrates: slot refresh → cache backend (with poll) → retry loop with fallbacks.
    """
    # Refresh slot counts
    await sm.refresh_slot_counts()

    min_ctx = backend_manager.get_model_n_ctx(canonical_name)

    # Precompute: blocks to pass to cache backend based on hit type
    # DISK_RESTORE: blocks needed for skip-restore check + disk restore
    # SKIP: no blocks — slot already has matching KV cache
    if hit_type == CacheHitType.DISK_RESTORE and restore_backend:
        cache_blocks = backend_blocks.get(restore_backend)
    else:
        cache_blocks = None

    async def _do_restore_call(be_sm: BackendSlotManager, slot_id: int,
                                key: Optional[str], blocks: Optional[List[str]],
                                prev_blocks: Optional[List[str]] = None) -> Optional[bool]:
        """Execute restore logic for an acquired slot. Returns restored flag.

        prev_blocks: the slot's KV state *before* this request. If provided,
        should_skip_restore checks against this instead of whatever is currently
        in _slot_kv_state (which may have been pre-set to the request's own blocks).
        """
        restored: Optional[bool] = None

        # Flush previously skipped save
        skip_entry = be_sm.flush_save_skipped(slot_id)
        if skip_entry:
            if len(skip_entry) >= 6:
                save_key, save_blocks, save_n_tokens, _, skip_ratio, skip_recompute = skip_entry
                if should_save_cache(skip_ratio, skip_recompute):
                    await be_sm.save_after(canonical_name, slot_id, save_key, save_blocks, save_n_tokens)
                    log.info(
                        "Saved skipped cache for model '%s' on backend '%s' slot %d before restore",
                        canonical_name, be_sm.backend_id, slot_id,
                    )
            else:
                save_key, save_blocks, save_n_tokens = skip_entry[:3]
                await be_sm.save_after(canonical_name, slot_id, save_key, save_blocks, save_n_tokens)
                log.info(
                    "Saved skipped cache (legacy) for model '%s' on backend '%s' slot %d before restore",
                    canonical_name, be_sm.backend_id, slot_id,
                )

        if blocks:
            if be_sm.should_skip_restore(slot_id, blocks, prev_blocks):
                log.info(
                    "Skipping restore for model '%s' on backend '%s' slot %d: slot cache already matches",
                    canonical_name, be_sm.backend_id, slot_id,
                )
                restored = False
            elif key:
                await be_sm.restore(slot_id, key, canonical_name, touch_ring=True)
                r_blocks = kv_meta.get_blocks(key, be_sm.backend_id)
                if r_blocks is not None:
                    be_sm.set_kv_state(slot_id, r_blocks)
                restored = True
            else:
                # No restore key — cache miss on this backend, no dynamic search needed
                # (routing scan already determined there's nothing useful here)
                log.info(
                    "No restore key for model '%s' on backend '%s' slot %d",
                    canonical_name, be_sm.backend_id, slot_id,
                )
                restored = False
        elif key:
            # No blocks but have a key (pending slot hit with key — shouldn't happen, but safe)
            await be_sm.restore(slot_id, key, canonical_name, touch_ring=True)
            restored = True
        else:
            # Pending slot hit: no blocks, no key — slot already has matching KV
            restored = None

        return restored

    async def _try_cache_backend() -> Optional[Tuple[Tuple[str, str, int], Optional[bool]]]:
        """Try to acquire a slot on the cache backend and restore."""
        if not restore_backend or prompt_tokens >= min_ctx:
            return None
        be_sm = sm.get(restore_backend)
        slot_id = be_sm.try_acquire(canonical_name)
        if slot_id is None:
            return None
        # Save old KV state before updating — skip-restore must check against previous state
        old_kv = be_sm.get_kv_state(slot_id)
        # Set KV state: use request blocks for disk restore, preserve existing for pending hit
        if cache_blocks:
            be_sm.set_kv_state(slot_id, cache_blocks)
        # For pending slot hit (cache_blocks is None), keep existing KV state intact
        restored = await _do_restore_call(be_sm, slot_id, restore_key, cache_blocks, old_kv)
        backend_manager.touch_backend(restore_backend)
        return (canonical_name, restore_backend, slot_id), restored

    # Phase 0: try cache backend; if busy, poll up to EMA timeout
    if restore_backend and hit_type and prompt_tokens < min_ctx:
        result = await _try_cache_backend()
        if result:
            return result

        # No free slot — poll every 5s up to EMA timeout
        pending = sm._cache_wait_pending.get(restore_backend, 0)
        if pending < CACHE_HIT_WAIT_MAX_PENDING_REQS:
            ema = sm.get(restore_backend).get_slot_duration_ema()
            wait_timeout = max(min(ema, CACHE_HIT_WAIT_EMA_MAX_TIMEOUT), CACHE_HIT_WAIT_EMA_MIN_TIMEOUT)
            log.info(
                "Cache backend '%s' busy for model '%s', polling up to %.1fs",
                restore_backend, canonical_name, wait_timeout,
            )
            try:
                sm._cache_wait_pending[restore_backend] = pending + 1
                elapsed = 0.0
                while elapsed < wait_timeout:
                    await asyncio.sleep(min(5.0, wait_timeout - elapsed))
                    elapsed += 5.0
                    result = await _try_cache_backend()
                    if result:
                        return result
            finally:
                sm._cache_wait_pending[restore_backend] -= 1

    # Retry loop: cache backend + fallbacks
    RETRY_COUNT = 11
    for attempt in range(RETRY_COUNT):
        # Phase 1: cache backend
        if restore_backend and hit_type and prompt_tokens < min_ctx:
            result = await _try_cache_backend()
            if result:
                return result

        # Phase 2: fallback backends
        for be_id, model_name in candidate_backends:
            if not model_name:
                continue
            be_min_ctx = backend_manager.get_model_n_ctx(model_name)
            if prompt_tokens >= be_min_ctx:
                continue
            be_sm = sm.get(be_id)
            slot_id = be_sm.try_acquire(model_name)
            if slot_id is None:
                continue
            # Cache miss: track blocks for this backend
            fb_blocks = backend_blocks.get(be_id)
            old_fb_kv = be_sm.get_kv_state(slot_id)
            if fb_blocks:
                be_sm.set_kv_state(slot_id, fb_blocks)
            else:
                be_sm.set_kv_state(slot_id, [])
            restored = await _do_restore_call(be_sm, slot_id, None, fb_blocks, old_fb_kv)
            backend_manager.touch_backend(be_id)
            return (model_name, be_id, slot_id), restored

        if attempt < RETRY_COUNT - 1:
            backoff = (attempt + 1) * 5
            log.info("No slots available across all backends, retrying in %ds (attempt %d/%d)",
                     backoff, attempt + 1, RETRY_COUNT)
            await asyncio.sleep(backoff)

    raise RuntimeError(f"No slots available for candidate_backends={len(candidate_backends)}")


# ── Streaming ──────────────────────────────────────────────────────────


class StreamReader:
    def __init__(self, resp: httpx.Response, req: Request,
                 model_name: str, backend_id: str, slot_id: int,
                 key: str, n_tokens: int, blocks: List[str],
                 sm: SlotManager, best_ratio: float = 0.0,
                 hit_type: Optional[CacheHitType] = None,
                 restore_key: Optional[str] = None, restore_backend: Optional[str] = None,
                 t0: float = 0, request_json: Optional[Dict] = None,
                 request_id: Optional[str] = None, routing_reason: str = "cache_miss",
                 backend_cache_ratios: Optional[Dict[str, float]] = None):
        self.resp = resp
        self.req = req
        self.model_name = model_name
        self.backend_id = backend_id
        self.slot_id = slot_id
        self.key = key
        self.n_tokens = n_tokens
        self.blocks = blocks
        self.sm = sm
        self.best_ratio = best_ratio
        self._hit_type = hit_type
        self.restore_key = restore_key
        self.restore_backend = restore_backend
        self._backend_cache_ratios = backend_cache_ratios or {}
        self._t0 = t0
        self._request_json = request_json
        self._request_id = request_id
        self._routing_reason = routing_reason
        self.queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=STREAM_QUEUE_SIZE)
        self._cancelled = False
        self._stream_complete = False
        self._task: asyncio.Task | None = None
        self._disconnect_event: asyncio.Event = asyncio.Event()
        self._cleaning_up = False
        self._raw_response_body: bytearray = bytearray()
        self._sse_line_buffer: str = ""
        self._sse_prompt_tokens: int = 0
        self._sse_cached_tokens: int = 0
        self.key_short = key[:16]

    def _log_response_info(self):
        try:
            status = self.resp.status_code
        except Exception:
            status = "?"
        try:
            headers = dict(self.resp.headers)
        except Exception:
            headers = "?"
        return status, headers

    async def _read_loop(self):
        status, headers = self._log_response_info()
        log.info("Stream started for model '%s' on backend '%s' slot %d (key %s): status %s",
                  self.model_name, self.backend_id, self.slot_id, self.key_short, status)
        log.info("Stream response headers for model '%s' on backend '%s' slot %d (key %s): %s",
                  self.model_name, self.backend_id, self.slot_id, self.key_short, headers)
        iterator = self.resp.aiter_raw()
        chunks_received = 0
        total_bytes = 0
        disconnect_wait = None
        try:
            while True:
                anext_task = asyncio.create_task(iterator.__anext__())
                disconnect_wait = asyncio.ensure_future(self._disconnect_event.wait())
                try:
                    done, pending = await asyncio.wait(
                        [anext_task, disconnect_wait],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                except asyncio.CancelledError:
                    log.warning("Reader loop cancelled for model '%s' on backend '%s' slot %d (key %s)",
                                self.model_name, self.backend_id, self.slot_id, self.key_short)
                    for t in [anext_task, disconnect_wait]:
                        if not t.done():
                            t.cancel()
                    raise
                if self._disconnect_event.is_set():
                    log.warning("Client disconnected while reading from model '%s' on backend '%s' slot %d (key %s)",
                                self.model_name, self.backend_id, self.slot_id, self.key_short)
                    for t in pending:
                        t.cancel()
                    self._disconnect_event.set()
                    break
                if anext_task in done:
                    try:
                        chunk = anext_task.result()
                    except StopAsyncIteration:
                        self._stream_complete = True
                        client_ip = self.req.client.host if self.req.client else "-"
                        status, headers = self._log_response_info()
                        log.info(
                            "Stream complete from client %s for model '%s' on backend '%s' slot %d (key %s): status %s, %d chunks, %d bytes",
                            client_ip, self.model_name, self.backend_id, self.slot_id, self.key_short,
                            status, chunks_received, total_bytes,
                        )
                        self._disconnect_event.set()
                        for t in pending:
                            t.cancel()
                        break
                    except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError,
                            httpx.StreamError, asyncio.IncompleteReadError,
                            ConnectionResetError, BrokenPipeError, OSError) as stream_err:
                        log.warning(
                            "Backend disconnected for model '%s' on backend '%s' slot %d (key %s): incomplete body (%d chunks, %d bytes), error=%s: %s",
                            self.model_name, self.backend_id, self.slot_id, self.key_short,
                            chunks_received, total_bytes, type(stream_err).__name__, stream_err,
                        )
                        self._disconnect_event.set()
                        for t in pending:
                            t.cancel()
                        break
                    if chunk:
                        total_bytes += len(chunk)
                        self._raw_response_body.extend(chunk)
                        sse_done = False
                        try:
                            text = chunk.decode("utf-8", errors="replace")
                            self._sse_line_buffer += text
                            while "\n" in self._sse_line_buffer:
                                line, self._sse_line_buffer = self._sse_line_buffer.split("\n", 1)
                                line = line.strip()
                                if line.startswith("data: "):
                                    data = line[6:].strip()
                                    if data == "[DONE]":
                                        sse_done = True
                                        break
                                    try:
                                        event = json.loads(data)
                                        usage = event.get("usage")
                                        if usage and isinstance(usage, dict):
                                            pt = usage.get("prompt_tokens")
                                            pt_details = usage.get("prompt_tokens_details") or {}
                                            ct = pt_details.get("cached_tokens", 0)
                                            if pt and not self._sse_prompt_tokens:
                                                self._sse_prompt_tokens = pt
                                            if ct and not self._sse_cached_tokens:
                                                self._sse_cached_tokens = ct
                                                log.info(
                                                     "Parsed SSE usage: model '%s' slot %d, cached_tokens=%d, prompt_tokens=%d",
                                                     self.model_name, self.slot_id, ct, pt,
                                                 )
                                    except (json.JSONDecodeError, ValueError):
                                        continue
                        except Exception:
                            pass

                        if sse_done:
                            log.info(
                                "SSE [DONE] received for model '%s' on backend '%s' slot %d (key %s)",
                                self.model_name, self.backend_id, self.slot_id, self.key_short,
                            )
                            self._stream_complete = True
                            self._disconnect_event.set()
                            for t in pending:
                                t.cancel()
                            break
                        try:
                            self.queue.put_nowait(chunk)
                            chunks_received += 1
                        except asyncio.QueueFull:
                            log.warning("Stream queue full for model '%s' on backend '%s' slot %d (key %s)",
                                        self.model_name, self.backend_id, self.slot_id, self.key_short)
                            self._cancelled = True
                            self._disconnect_event.set()
                            for t in pending:
                                t.cancel()
                            break
                        except asyncio.CancelledError:
                            log.warning("Stream reader cancelled while putting to queue for model '%s' on backend '%s' slot %d (key %s)",
                                        self.model_name, self.backend_id, self.slot_id, self.key_short)
                            raise
                else:
                    for t in pending:
                        t.cancel()
        finally:
            if disconnect_wait and not disconnect_wait.done():
                disconnect_wait.cancel()
            if anext_task and not anext_task.done():
                anext_task.cancel()

    async def _heartbeat(self):
        while True:
            await asyncio.sleep(0.5)
            try:
                disconnected = await self.req.is_disconnected()
            except Exception:
                disconnected = False
            if disconnected:
                log.warning("Client disconnected (heartbeat check) for model '%s' on backend '%s' slot %d (key %s)",
                            self.model_name, self.backend_id, self.slot_id, self.key_short)
                self._cancelled = True
                self._disconnect_event.set()
                break

    def _signal_done(self):
        self._cancelled = True
        self.queue.put_nowait(None)

    async def _save(self) -> tuple:
        be_sm = self.sm.get(self.backend_id)
        recompute_happened = False
        if self._hit_type and self.restore_backend:
            cached_tokens = self._sse_cached_tokens
            llm_prompt_tokens = self._sse_prompt_tokens or self.n_tokens
            log.info(
                "Recompute check for model '%s' slot %d: cached_tokens=%d, request_prompt_tokens=%d (llm=%d), ratio=%.3f",
                self.model_name, self.slot_id, cached_tokens, self.n_tokens, llm_prompt_tokens,
                self.best_ratio,
            )
            if cached_tokens < llm_prompt_tokens * RECOMPUTE_THRESHOLD_PERCENT_REQ_TOKENS:
                recompute_happened = True
                log.warning(
                    "Recompute detected for model '%s' on backend '%s' slot %d (key %s): "
                    "cached_tokens=%d llm_prompt_tokens=%d, "
                    "KV cache restore was partial/useless",
                    self.model_name, self.backend_id, self.slot_id, self.key_short,
                    cached_tokens, llm_prompt_tokens,
                )
                if self.restore_key:
                    kv_meta.increment_recompute_penalty(self.restore_key, self.restore_backend)

        serving_be_ratio = self._backend_cache_ratios.get(self.backend_id, 0.0)
        if not should_save_cache(serving_be_ratio, recompute_happened):
            log.info(
                "Skipping cache save for model '%s' on backend '%s' slot %d (key %s): "
                "restore ratio %.3f >= threshold (no recompute, cache was useful)",
                self.model_name, self.backend_id, self.slot_id, self.key_short,
                serving_be_ratio,
            )
            be_sm.mark_save_skipped(self.slot_id,
                                    (self.key, self.blocks, self.n_tokens, self._hit_type, serving_be_ratio, recompute_happened))
            return False, 0
        ok = False
        cache_size = 0
        try:
            ok, cache_size = await be_sm.save_after(
                self.model_name, self.slot_id,
                self.key, self.blocks, self.n_tokens,
            )
            log.info("SAVE: model_name='%s', key='%s', backend='%s', model_id_in_meta='%s'",
                     self.model_name, self.key[:16], self.backend_id, self.model_name)
        except asyncio.CancelledError:
            log.warning("Cache save cancelled for model '%s' on backend '%s' slot %d",
                        self.model_name, self.backend_id, self.slot_id)
        except Exception as e:
            log.warning("Cache save failed for model '%s' on backend '%s' slot %d: %s",
                        self.model_name, self.backend_id, self.slot_id, e)
        return ok, cache_size

    async def _cleanup(self):
        be_sm = self.sm.get(self.backend_id)
        log.info("Starting cleanup for model '%s' on backend '%s' slot %d (key %s): cancelled=%s, stream_complete=%s",
                  self.model_name, self.backend_id, self.slot_id, self.key_short, self._cancelled, self._stream_complete)
        try:
            await asyncio.wait_for(self.resp.aclose(), timeout=5.0)
            log.info("Response closed for model '%s' on backend '%s' slot %d (key %s)",
                      self.model_name, self.backend_id, self.slot_id, self.key_short)
        except asyncio.TimeoutError:
            log.warning("Response aclose() timed out for model '%s' on backend '%s' slot %d (key %s) — half-open connection",
                        self.model_name, self.backend_id, self.slot_id, self.key_short)
        except Exception as e:
            log.info("Error closing response for model '%s' on backend '%s' slot %d (key %s): %s",
                      self.model_name, self.backend_id, self.slot_id, self.key_short, e)
        ok = False
        cache_size = 0
        if self._stream_complete:
            log.info("Saving cache for model '%s' on backend '%s' slot %d (key %s)",
                      self.model_name, self.backend_id, self.slot_id, self.key_short)
            ok, cache_size = await self._save()
            log.info("Cache save completed for model '%s' on backend '%s' slot %d (key %s): %s, %d bytes",
                      self.model_name, self.backend_id, self.slot_id, self.key_short, ok, cache_size)
        else:
            log.info("Skipping cache save for model '%s' on backend '%s' slot %d (key %s): stream incomplete",
                      self.model_name, self.backend_id, self.slot_id, self.key_short)
            if self._cancelled:
                log.info("Request cancelled — invalidating KV cache for model '%s' on backend '%s' slot %d (key %s)",
                         self.model_name, self.backend_id, self.slot_id, self.key_short)
                be_sm.invalidate(self.slot_id)
            else:
                log.info("Backend disconnected — invalidating KV cache for model '%s' on backend '%s' slot %d (key %s)",
                         self.model_name, self.backend_id, self.slot_id, self.key_short)
                be_sm.invalidate(self.slot_id)

        be_sm.release(self.slot_id)
        log.info("Released slot %d for model '%s' on backend '%s' (key %s)", self.slot_id,
                  self.model_name, self.backend_id, self.key_short)

        # Record metrics for streaming requests
        if self._t0 > 0:
            recompute_happened = False
            if self._hit_type:
                cached_tokens = self._sse_cached_tokens
                llm_prompt_tokens = self._sse_prompt_tokens or self.n_tokens
                if cached_tokens < llm_prompt_tokens * RECOMPUTE_THRESHOLD_PERCENT_REQ_TOKENS:
                    recompute_happened = True

            prompt_preview = extract_prompt_preview(self._request_json)
            if self._stream_complete:
                stream_status = "complete"
            elif self._cancelled:
                stream_status = "cancelled"
            else:
                stream_status = "backend_error"
            latency_ms = (time.time() - self._t0) * 1000
            metrics.record({
                "request_id": self._request_id,
                "t0": self._t0,
                "request_json": self._request_json or {},
                "model": self.model_name,
                "backend": self.backend_id,
                "slot_id": self.slot_id,
                 "cache_hit": self._hit_type is not None,
                 "restored": (True if self._hit_type == CacheHitType.DISK_RESTORE and self.restore_key
                              else (None if self._hit_type == CacheHitType.SKIP else False)),
                "recompute": recompute_happened,
                "saved": ok,
                "latency_ms": latency_ms,
                "n_tokens": self._sse_prompt_tokens or self.n_tokens,
                "cached_tokens": self._sse_cached_tokens or 0,
                "stream": True,
                "cache_size_bytes": cache_size,
                "prompt_preview": prompt_preview,
                "routing_reason": self._routing_reason,
                "status": stream_status,
            })
            backend_manager.update_backend_latency(self.backend_id, latency_ms)

        log.info("Stream reader finished for model '%s' on backend '%s' slot %d (key %s): saved=%s",
                  self.model_name, self.backend_id, self.slot_id, self.key_short, ok)

    async def _reader(self):
        try:
            await self._read_loop()
            self.queue.put_nowait(None)
        except asyncio.CancelledError:
            log.warning("Stream reader cancelled for model '%s' on backend '%s' slot %d (key %s)",
                        self.model_name, self.backend_id, self.slot_id, self.key_short)
        except Exception as e:
            log.exception("Stream reader error for model '%s' on backend '%s' slot %d (key %s): %s",
                          self.model_name, self.backend_id, self.slot_id, self.key_short, e)

    async def stream(self):
        self._task = asyncio.create_task(self._reader())
        heartbeat = asyncio.create_task(self._heartbeat())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(), timeout=STREAM_QUEUE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    try:
                        disconnected = await self.req.is_disconnected()
                    except Exception:
                        disconnected = False
                    if disconnected:
                        log.warning("Client disconnected during stream for model '%s' on backend '%s' slot %d (key %s)",
                                    self.model_name, self.backend_id, self.slot_id, self.key_short)
                        self._signal_done()
                        self._task.cancel()
                        break
                    continue
                except asyncio.CancelledError:
                    log.warning("gen_cancelled, cancelling reader task")
                    self._signal_done()
                    self._task.cancel()
                    heartbeat.cancel()
                    break
                if item is None:
                    break
                yield item
        finally:
            if self._task and not self._task.done():
                if self._cleaning_up:
                    await self._task
                else:
                    self._cancelled = True
                    self._task.cancel()
                    try:
                        await asyncio.wait_for(self._task, timeout=5.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        log.warning(
                            "Reader task did not finish after cancel for model '%s' on backend '%s' slot %d (key %s)",
                            self.model_name, self.backend_id, self.slot_id, self.key_short,
                        )
            if not heartbeat.done():
                heartbeat.cancel()
            await asyncio.shield(self._cleanup())


# ── Chat Completions ───────────────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat(req: Request):
    sm: SlotManager = app.state.sm

    t0 = time.time()
    client_ip = req.client.host if req.client else "-"

    raw_body = await req.body()
    request_json = json.loads(raw_body)
    data = request_json

    messages: List[Dict] = data.get("messages") or []
    stream = bool(data.get("stream", False))
    client_model = data.get("model") or MODEL_ID

    request_id = str(uuid.uuid4())
    prompt_preview = extract_prompt_preview(request_json)
    try:
        metrics.record({
            "request_id": request_id,
            "request_json": request_json,
            "model": client_model,
            "stream": stream,
            "status": "incomplete",
            "prompt_preview": prompt_preview,
        })
    except Exception as e:
        log.warning("Failed to record request arrival for request_id=%s: %s", request_id, e)

    try:
        # 1. Resolve model name
        options = backend_manager.get_discovered_models(client_model)
        if not options:
            await backend_manager.discover_models()
            options = backend_manager.get_discovered_models(client_model)
            if not options:
                metrics.record({
                    "request_id": request_id,
                    "model": client_model,
                    "latency_ms": (time.time() - t0) * 1000,
                    "status": "backend_error",
                })
                return JSONResponse({"error": f"model '{client_model}' not found"}, status_code=400)

        # 2. Tokenize + scan for cache hits
        first_opt = options[0]
        first_client = backend_manager.get_client(first_opt.backends[0])
        try:
            templated = await first_client.apply_chat_template(messages)
            first_token_ids = await first_client.tokenize(templated, add_special=True)
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
            log.error("Backend %s error for model '%s' from client %s",
                      first_opt.backends[0], client_model, client_ip)
            metrics.record({
                "request_id": request_id,
                "model": client_model,
                "latency_ms": (time.time() - t0) * 1000,
                "status": "backend_error",
            })
            return JSONResponse({"error": "backend unreachable"}, status_code=503)
        prompt_tokens = len(first_token_ids)
        min_ctx = min(opt.n_ctx for opt in options)
        if prompt_tokens >= min_ctx:
            metrics.record({
                "request_id": request_id,
                "model": client_model,
                "latency_ms": (time.time() - t0) * 1000,
                "status": "backend_error",
            })
            return JSONResponse(
                {"error": f"prompt too long (tokens={prompt_tokens}, n_ctx={min_ctx})"},
                status_code=400,
            )
        canonical_name = first_opt.name

        restore_key = None
        restore_backend = None
        best_ratio = 0.0
        hit_type: Optional[CacheHitType] = None

        scan_diagnostics: List[Dict[str, Any]] = []
        backend_cache_ratios: Dict[str, float] = {}
        backend_token_ids: Dict[str, List[int]] = {}
        backend_blocks: Dict[str, List[str]] = {}

        for opt in options:
            for be_id in opt.backends:
                opt_client = backend_manager.get_client(be_id)
                try:
                    opt_templated = await opt_client.apply_chat_template(messages)
                    opt_token_ids = await opt_client.tokenize(opt_templated, add_special=True)
                except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
                    log.warning("Backend %s error, skipping for model '%s' from client %s",
                                be_id, opt.name, client_ip)
                    scan_diagnostics.append({
                        "model": opt.name, "backend": be_id,
                        "status": "unreachable",
                    })
                    continue
                opt_blocks = hs.block_hashes_from_tokens(opt_token_ids, WORDS_PER_BLOCK)
                backend_token_ids[be_id] = opt_token_ids
                backend_blocks[be_id] = opt_blocks
                diag_entry: Dict[str, Any] = {
                    "model": opt.name, "backend": be_id,
                    "n_blocks": len(opt_blocks), "n_tokens": len(opt_token_ids),
                }
                cand = kv_meta.find_best_restore_candidate(opt_blocks, WORDS_PER_BLOCK, LCP_TH, opt.name, be_id)
                if cand:
                    diag_entry["cache_file_key"] = cand[0][:16]
                    diag_entry["cache_file_ratio"] = round(cand[1], 4)
                    if cand[1] > best_ratio:
                        best_ratio = cand[1]
                        restore_key = cand[0]
                        restore_backend = be_id
                        canonical_name = opt.name
                        hit_type = CacheHitType.DISK_RESTORE
                        log.info("Cache hit: key '%s' (model '%s', backend '%s', ratio %.3f) — replacing previous best",
                                 restore_key[:16], canonical_name, restore_backend, best_ratio)
                else:
                    diag_entry["cache_file_ratio"] = None

                # Check pending slots for cache hits
                be_sm = sm.get(be_id)
                pending_ratios: List[Dict[str, Any]] = []
                for slot_id, kv_blocks in be_sm.get_kv_states().items():
                    lcp = hs.lcp_blocks(opt_blocks, kv_blocks)
                    ratio = lcp / len(opt_blocks)
                    pending_ratios.append({"slot": slot_id, "lcp_blocks": lcp,
                                          "slot_blocks": len(kv_blocks), "ratio": round(ratio, 4)})
                    if ratio >= LCP_TH and ratio > best_ratio:
                        best_ratio = ratio
                        restore_key = None
                        restore_backend = be_id
                        canonical_name = opt.name
                        hit_type = CacheHitType.SKIP
                        log.info("Pending slot cache hit: model '%s', backend '%s', slot %d, ratio %.3f",
                                 canonical_name, restore_backend, slot_id, ratio)
                diag_entry["pending_slots"] = pending_ratios
                scan_diagnostics.append(diag_entry)
                be_best = diag_entry.get("cache_file_ratio") or 0.0
                for ps in pending_ratios:
                    if ps["ratio"] > be_best:
                        be_best = ps["ratio"]
                if be_best > 0:
                    backend_cache_ratios[be_id] = be_best

        log.info(
            "Chat request from %s: model '%s', %d tokens, restore key=%s on backend %s",
            client_ip, client_model, prompt_tokens,
            restore_key[:16] if restore_key else None,
            restore_backend,
        )

        # 5. Build fallback candidate backends
        candidate_backends: list[tuple[str, str]] = []
        for opt in options:
            for be_id in opt.backends:
                if restore_backend and be_id == restore_backend:
                    continue
                candidate_backends.append((be_id, opt.name))

        candidate_backends.sort(
            key=lambda cb: (
                -backend_cache_ratios.get(cb[0], 0.0),
                sm.get(cb[0]).get_ring_size(),
                backend_manager.get_backend_latency_ema(cb[0]),
                backend_manager.get_backend_last_used(cb[0]),
            ),
        )

        # 6. Acquire slot (cache backend first, then fallbacks)
        g, restored = await asyncio.wait_for(
            _acquire_slot_for_request(
                sm, candidate_backends, restore_backend, canonical_name,
                hit_type, restore_key, backend_blocks, prompt_tokens,
            ),
            timeout=ACQUIRE_TIMEOUT,
        )
    except (asyncio.TimeoutError, RuntimeError) as e:
        log.error("Could not acquire slot from client %s for model '%s': %s", client_ip, client_model, e)
        metrics.record({
            "request_id": request_id,
            "model": client_model,
            "latency_ms": (time.time() - t0) * 1000,
            "status": "backend_error",
        })
        return JSONResponse({"error": "all slots busy, please retry later"}, status_code=503)

    model_name, be_id, slot_id = g
    client = backend_manager.get_client(be_id)
    be_sm = sm.get(be_id)

    # Derive key and blocks from serving backend's tokenization
    serving_token_ids = backend_token_ids.get(be_id, first_token_ids)
    key = hs.meta_key(model_name, serving_token_ids)
    blocks = hs.block_hashes_from_tokens(serving_token_ids, WORDS_PER_BLOCK)

    # Determine routing reason for metrics
    if hit_type == CacheHitType.SKIP and be_id == restore_backend:
        routing_reason = "pending_slot_hit"
    elif hit_type == CacheHitType.DISK_RESTORE and be_id == restore_backend:
        routing_reason = "cache_hit"
    elif hit_type is None:
        routing_reason = "no_cache_entry"
    else:
        routing_reason = "cache_backend_unavailable"

    if restore_key is None and restore_backend is None:
        log.info("No cache hit: using key '%s' for model '%s' (client model '%s')", key[:16], model_name, client_model)

    log.info("Slot acquired: model '%s' on backend '%s' slot %d, restored=%s, save_key='%s', canonical_name='%s'",
             model_name, be_id, slot_id, restored, key[:16], canonical_name)

    # Update arrival record with routing decision + diagnostics
    try:
        metrics.record({
            "request_id": request_id,
            "model": model_name,
            "backend": be_id,
            "slot_id": slot_id,
            "routing_reason": routing_reason,
            "cache_hit": hit_type is not None,
            "restored": restored,
            "status": "incomplete",
            "routing_diagnostics": {
                "best_ratio": round(best_ratio, 4),
                "restore_key": restore_key[:16] if restore_key else None,
                "restore_backend": restore_backend,
                "restore_info_backend": restore_backend,
                "candidate_backends": [cb[0] for cb in candidate_backends],
                "scan": scan_diagnostics,
            },
        })
    except Exception as e:
        log.warning("Failed to record routing decision for request_id=%s: %s", request_id, e)

    # Forward canonical name to backend
    body = dict(data)
    body["model"] = canonical_name
    opts = dict(body.get("options") or {})
    opts["slot_id"] = slot_id
    opts["id_slot"] = slot_id
    opts["n_keep"] = -1
    opts["cache_prompt"] = True
    body["options"] = opts
    body["n_keep"] = -1
    body["cache_prompt"] = True

    log.info(
        "Dispatching request from client %s: model '%s' on backend '%s' slot %d, restore=%s, restored=%s",
        client_ip, model_name, be_id, slot_id,
        restore_key[:16] if restore_key else None, restored,
    )

    _reader_created = False
    _slot_released = False
    try:
        if stream:
            resp = await client.chat_completions(
                body, slot_id=slot_id, stream=True,
            )
            if resp.status_code != 200:
                err_txt = await resp.aread()
                await resp.aclose()
                metrics.record({
                    "request_id": request_id,
                    "model": model_name,
                    "backend": be_id,
                    "slot_id": slot_id,
                    "latency_ms": (time.time() - t0) * 1000,
                    "routing_reason": routing_reason,
                    "status": "backend_error",
                })
                return JSONResponse(
                    {"error": err_txt.decode("utf-8", "ignore")},
                    status_code=resp.status_code,
                )

            reader = StreamReader(resp, req, model_name, be_id, slot_id,
                                   key, prompt_tokens, blocks, sm, best_ratio,
                                   hit_type=hit_type,
                                   restore_key=restore_key, restore_backend=restore_backend,
                                   t0=t0, request_json=request_json,
                                   request_id=request_id, routing_reason=routing_reason,
                                   backend_cache_ratios=backend_cache_ratios)
            gen = reader.stream()
            _reader_created = True

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
            return StreamingResponse(
                gen, media_type="text/event-stream", headers=headers,
            )

        else:
            out = await client.chat_completions(
                body, slot_id=slot_id, stream=False,
            )
            if not isinstance(out, dict):
                metrics.record({
                    "request_id": request_id,
                    "model": model_name,
                    "backend": be_id,
                    "slot_id": slot_id,
                    "latency_ms": (time.time() - t0) * 1000,
                    "routing_reason": routing_reason,
                    "status": "backend_error",
                })
                return JSONResponse(
                    {"error": "provider non-JSON body"},
                    status_code=502,
                )

            ok = False
            recompute_happened = False
            llm_prompt_tokens = 0
            cached_tokens = 0
            if hit_type:
                usage = out.get("usage") or {}
                pt_details = usage.get("prompt_tokens_details") or {}
                cached_tokens = pt_details.get("cached_tokens", 0)
                llm_prompt_tokens = usage.get("prompt_tokens", 0)
                log.info(
                    "Recompute check for model '%s' slot %d: cached_tokens=%d, request_prompt_tokens=%d (llm=%d), ratio=%.3f",
                    model_name, slot_id, cached_tokens, prompt_tokens, llm_prompt_tokens,
                    best_ratio,
                )
                if cached_tokens < llm_prompt_tokens * RECOMPUTE_THRESHOLD_PERCENT_REQ_TOKENS:
                    recompute_happened = True
                    log.warning(
                        "Recompute detected for model '%s' on backend '%s' slot %d (key %s): "
                        "cached_tokens=%d llm_prompt_tokens=%d, "
                        "KV cache restore was partial/useless",
                        model_name, be_id, slot_id, key[:16],
                        cached_tokens, llm_prompt_tokens,
                    )
                    if restore_key and restore_backend:
                        kv_meta.increment_recompute_penalty(restore_key, restore_backend)

            save_ok = False
            cache_size = 0
            serving_be_ratio = backend_cache_ratios.get(be_id, 0.0)
            if should_save_cache(serving_be_ratio, recompute_happened):
                try:
                    ok, cache_size = await be_sm.save_after(
                        model_name, slot_id, key, blocks, prompt_tokens,
                    )
                    save_ok = ok
                except asyncio.CancelledError:
                    log.warning("Cache save cancelled for model '%s' on backend '%s' slot %d",
                                model_name, be_id, slot_id)
                except Exception as e:
                    log.warning("Cache save failed for model '%s' on backend '%s' slot %d: %s",
                                model_name, be_id, slot_id, e)
            else:
                be_sm.mark_save_skipped(slot_id,
                                         (key, blocks, prompt_tokens, hit_type, serving_be_ratio, recompute_happened))

            prompt_preview = extract_prompt_preview(request_json)
            latency_ms = (time.time() - t0) * 1000
            metrics.record({
                "request_id": request_id,
                "t0": t0,
                "request_json": request_json,
                "model": model_name,
                "backend": be_id,
                "slot_id": slot_id,
                "cache_hit": hit_type is not None,
                "restored": restored,
                "recompute": recompute_happened,
                "saved": save_ok,
                "latency_ms": latency_ms,
                "n_tokens": llm_prompt_tokens,
                "cached_tokens": cached_tokens,
                "stream": False,
                "cache_size_bytes": cache_size if ok else 0,
                "prompt_preview": prompt_preview,
                "routing_reason": routing_reason,
                "status": "complete",
            })
            backend_manager.update_backend_latency(be_id, latency_ms)

            return JSONResponse(content=out, status_code=200)

    except httpx.TimeoutException as e:
        log.exception("Chat timeout for client %s, model '%s' on backend '%s' slot %d (key %s): %s",
                       client_ip, model_name, be_id, slot_id, key[:16], e)
        metrics.record({
            "request_id": request_id,
            "model": model_name,
            "backend": be_id,
            "slot_id": slot_id,
            "latency_ms": (time.time() - t0) * 1000,
            "routing_reason": routing_reason,
            "status": "backend_error",
        })
        return JSONResponse({"error": str(e)}, status_code=504)
    except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
        log.exception("Backend connection error for client %s, model '%s' on backend '%s' slot %d (key %s): %s",
                      client_ip, model_name, be_id, slot_id, key[:16], e)
        be_sm.invalidate(slot_id)
        be_sm.release(slot_id)
        _slot_released = True
        metrics.record({
            "request_id": request_id,
            "model": model_name,
            "backend": be_id,
            "slot_id": slot_id,
            "latency_ms": (time.time() - t0) * 1000,
            "routing_reason": routing_reason,
            "status": "backend_error",
        })
        return JSONResponse({"error": "backend connection failed"}, status_code=503)
    except Exception as e:
        log.exception("Chat error for client %s, model '%s' on backend '%s' slot %d (key %s): %s",
                       client_ip, model_name, be_id, slot_id, key[:16], e)
        metrics.record({
            "request_id": request_id,
            "model": model_name,
            "backend": be_id,
            "slot_id": slot_id,
            "latency_ms": (time.time() - t0) * 1000,
            "routing_reason": routing_reason,
            "status": "backend_error",
        })
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if not _reader_created and not _slot_released:
            be_sm.release(slot_id)


# ── Metrics & Dashboard Endpoints ────────────────────────────────────

@app.get("/metrics/summary")
async def metrics_summary():
    summary = metrics.get_summary()
    summary["backends"] = _get_backend_health()
    summary["slots"] = _get_slot_status()
    summary["cache"] = _get_cache_stats()
    backend_perf = {}
    for be_id in summary["backends"]:
        bp = metrics.get_performance(backend=be_id)
        if bp.get("total_requests", 0) > 0:
            backend_perf[be_id] = bp
    summary["backend_performance"] = backend_perf
    return summary


@app.get("/metrics/health")
async def metrics_health():
    return _get_backend_health()


@app.get("/metrics/slots")
async def metrics_slots():
    return _get_slot_status()


@app.get("/metrics/cache")
async def metrics_cache():
    return _get_cache_stats()


@app.get("/metrics/diagnostics")
async def metrics_diagnostics(request_id: str = None, liveness: bool = False, timeline: bool = False):
    if timeline:
        return {"timeline": metrics.get_timeline(limit=200)}
    if liveness:
        return {"liveness_events": metrics.get_events(event_type="liveness_change", limit=50)}
    if request_id:
        req = metrics.get_request_by_id(request_id)
        if req is None:
            return JSONResponse({"error": "Request not found"}, status_code=404)
        return {"request_id": request_id, "routing_diagnostics": req.get("routing_diagnostics")}
    requests = metrics.get_requests(limit=100)
    return {"diagnostics": [
        {"request_id": r.get("request_id"), "routing_diagnostics": r.get("routing_diagnostics")}
        for r in requests if r.get("routing_diagnostics")
    ]}


@app.get("/metrics/requests")
async def metrics_requests(limit: int = 100, offset: int = 0):
    requests = metrics.get_requests(limit=limit, offset=offset)
    return {"requests": requests, "total": metrics.get_total_count()}


@app.get("/metrics/request/{request_id}")
async def metrics_request_by_id(request_id: str):
    req = metrics.get_request_by_id(request_id)
    if req is None:
        return JSONResponse({"error": "Request not found"}, status_code=404)
    return req


@app.get("/metrics/performance")
async def metrics_performance(model: str = None, backend: str = None):
    perf = metrics.get_performance(model=model, backend=backend)
    return perf


@app.get("/dashboard")
async def dashboard():
    from config import DASHBOARD_ENABLED
    if not DASHBOARD_ENABLED:
        return JSONResponse({"error": "Dashboard disabled"}, status_code=404)
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r") as f:
            return HTMLResponse(content=f.read())
    return JSONResponse({"error": "dashboard.html not found"}, status_code=404)


# ── Helper Functions for Metrics ─────────────────────────────────────


def _get_backend_health() -> dict:
    sm = app.state.sm if hasattr(app.state, "sm") else None
    result = {}
    for key in backend_manager.keys():
        be = backend_manager._backends.get(key)
        if be is None:
            continue
        info = {
            "url": be.client.base_url,
            "up": backend_manager._backend_state.get(key, False),
            "cache_dir": be.cache_dir,
            "has_agent": be.agent_client is not None,
            "models": {},
        }
        for model_name, model_info in backend_manager._discovered_models.items():
            if key not in model_info.backends:
                continue
            in_use = 0
            total_slots = 0
            if sm and key in sm._backends:
                be_sm = sm.get(key)
                pool = be_sm.get_pool(model_name)
                if pool:
                    total_slots = len(pool)
                    for slot_id in pool:
                        if be_sm._in_use.get(slot_id, False):
                            in_use += 1
            refresh_key = (model_name, key)
            last_ts, _, _ = backend_manager._refresh_state.get(refresh_key, (0.0, True, 0))
            info["models"][model_name] = {
                "n_ctx": model_info.n_ctx,
                "total_slots": model_info.total_slots,
                "in_use": in_use,
                "last_discovered": model_info.last_discovered,
                "last_refresh": last_ts,
            }
        result[key] = info
    return result


def _get_slot_status() -> dict:
    sm = app.state.sm if hasattr(app.state, "sm") else None
    if not sm:
        return {}
    result = {}
    for backend_id, be_sm in sm._backends.items():
        if backend_id not in result:
            result[backend_id] = {"models": {}}
        for model_name, pool in be_sm._slot_pools.items():
            if model_name not in result[backend_id]["models"]:
                result[backend_id]["models"][model_name] = {"slots": {}}
            slots = result[backend_id]["models"][model_name]["slots"]
            for slot_id in pool:
                in_use = be_sm._in_use.get(slot_id, False)
                last_used = be_sm._last_used.get(slot_id, 0)
                kv_blocks = len(be_sm._slot_kv_state.get(slot_id, []))
                last_restore_key = None
                if slot_id in be_sm._slot_save_skipped:
                    skip_entry = be_sm._slot_save_skipped[slot_id]
                    if len(skip_entry) >= 3:
                        last_restore_key = skip_entry[0][:16] if skip_entry[0] else None
                slots[str(slot_id)] = {
                    "in_use": in_use,
                    "last_used": last_used,
                    "kv_blocks": kv_blocks,
                    "last_restore": last_restore_key,
                }
    return result


def _get_cache_stats() -> dict:
    sm = app.state.sm if hasattr(app.state, "sm") else None
    if not sm:
        return {}
    result = {}
    for backend_id in backend_manager.keys():
        be_sm = sm.get(backend_id) if backend_id in sm._backends else None
        if not be_sm:
            continue
        total_bytes = be_sm.get_total_bytes()
        max_gb = backend_manager.get_cache_max_size_gb(backend_id)
        max_bytes = max_gb * 1024**3
        utilization = (total_bytes / max_bytes * 100) if max_bytes > 0 else 0
        file_count = be_sm.get_ring_size()
        oldest = None
        newest = None
        if be_sm._cache_ring:
            oldest = be_sm._cache_ring[0][2]
            newest = be_sm._cache_ring[-1][2]
        cache_dir = backend_manager.get_cache_dir(backend_id)
        result[backend_id] = {
            "ring_size": file_count,
            "total_bytes": total_bytes,
            "total_gb": round(total_bytes / 1024**3, 2),
            "max_gb": max_gb,
            "utilization_pct": round(utilization, 1),
            "cache_dir": cache_dir,
            "oldest_entry": oldest,
            "newest_entry": newest,
        }
    return result
