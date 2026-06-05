# app.py

# -*- coding: utf-8 -*-

"""
KV Proxy for llama.cpp with disk save/restore:

- Big requests: LCP matching → restore from disk, then chat strictly in that slot, then save+meta.
- Small requests: free/oldest slot, no restore, no disk save/meta.
- Slot pinning is duplicated in root/options/query (via client).

Additionally:

- acquire_for_request is wrapped in a timeout to avoid hanging forever if a slot is never released.
- Streaming:
    * socket reads from llama.cpp run in a background task (reader),
      racing against a disconnect event;
    * reader pushes chunks into asyncio.Queue;
    * stream()'s finally calls _cleanup() — save (if _stream_complete),
      release, and puts a sentinel in the queue;
    * heartbeat task checks is_disconnected() every 0.5s.
- Slot pools are per-model with lazy discovery + refresh cooldown.
- Cache eviction via ring buffer in SlotManager (age-first, then LRU).
"""

import asyncio
import json
import os
import time
import logging
from typing import List, Dict, AsyncGenerator, Optional

import httpx
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from config import (BACKENDS, WORDS_PER_BLOCK,
                    LCP_TH, META_DIR, MODEL_ID, PORT,
                    CACHE_DIR, CACHE_MAX_AGE_HOURS, CACHE_MAX_SIZE_GB,
                    SLOT_TIMEOUT, DEFAULT_N_CTX, CACHE_SAVE_RATIO_THRESHOLD)

import hashing as hs
from slot_manager import SlotManager

log = logging.getLogger(__name__)

from backend_manager import backend_manager
from kv_meta_manager import kv_meta

ACQUIRE_TIMEOUT = 60.0
STREAM_QUEUE_SIZE = 16
STREAM_QUEUE_TIMEOUT = 5.0

app = FastAPI(title="Simple KV Proxy")


@app.on_event("startup")
async def startup():
    sm = SlotManager()
    await sm.init_from_disk(CACHE_DIR)

    app.state.sm = sm
    app.state.executor = ThreadPoolExecutor(max_workers=2)
    log.info("Starting on port %d with %d backends: %s", PORT, len(BACKENDS), [be["url"] for be in BACKENDS])

    # Reconcile meta files on startup (remove corrupted/orphaned entries)
    backend_keys = list(backend_manager._backends.keys())
    backend_agents = {k: v.agent_client.base_url if v.agent_client else None for k, v in backend_manager._backends.items()}
    backend_agents = {k: v for k, v in backend_agents.items() if v is not None}
    reconciled = await kv_meta.reconcile(CACHE_DIR, backend_keys, backend_agents)
    if reconciled > 0:
        log.info("Cleaned up %d orphaned/corrupted meta files at startup", reconciled)

    # Log startup sanity summary
    if CACHE_DIR and os.path.isdir(CACHE_DIR):
        cache_files = len(os.listdir(CACHE_DIR))
        meta_count = sum(len(kv_meta.list_keys(k)) for k in backend_keys)
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


class StreamReader:
    def __init__(self, resp: httpx.Response, req: Request,
                 model_name: str, backend_id: str, slot_id: int,
                 key: str, n_tokens: int, blocks: List[str],
                 sm: SlotManager, best_ratio: float = 0.0, restored: bool = False):
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
        self.restored = restored
        self.queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=STREAM_QUEUE_SIZE)
        self._cancelled = False
        self._stream_complete = False
        self._task: asyncio.Task | None = None
        self._disconnect_event: asyncio.Event = asyncio.Event()
        self._cleaning_up = False
        self.key_short = key[:16]

    def _log_response_info(self):
        """Return (status, headers) tuple with safe fallbacks."""
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
                done, pending = await asyncio.wait(
                    [anext_task, disconnect_wait],
                    return_when=asyncio.FIRST_COMPLETED,
                )
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
                    if chunk:
                        total_bytes += len(chunk)
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
        """Periodically check if client disconnected; sets _disconnect_event when True."""
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
        """Send None sentinel to unblock stream consumer."""
        self._cancelled = True
        self.queue.put_nowait(None)

    async def _save(self) -> tuple:
        if self.restored and self.best_ratio >= CACHE_SAVE_RATIO_THRESHOLD:
            log.info(
                "Skipping cache save for model '%s' on backend '%s' slot %d (key %s): restore ratio %.3f >= threshold",
                self.model_name, self.backend_id, self.slot_id, self.key_short,
                self.best_ratio,
            )
            return False
        ok = False
        try:
            ok, cache_size = await self.sm.save_after(
                self.model_name, self.backend_id, self.slot_id,
                self.key, self.blocks,
            )
            log.info("SAVE: model_name='%s', key='%s', backend='%s', model_id_in_meta='%s'",
                     self.model_name, self.key[:16], self.backend_id, self.model_name)
        except asyncio.CancelledError:
            log.warning("Cache save cancelled for model '%s' on backend '%s' slot %d",
                        self.model_name, self.backend_id, self.slot_id)
        except Exception as e:
            log.warning("Cache save failed for model '%s' on backend '%s' slot %d: %s",
                        self.model_name, self.backend_id, self.slot_id, e)
        if ok:
            try:
                kv_meta.write_meta(self.key, self.n_tokens, self.blocks,
                                   WORDS_PER_BLOCK, self.model_name, self.backend_id, cache_size)
            except Exception as e:
                log.warning("Failed to write meta file %s: %s", self.key_short, e)
        return ok

    async def _cleanup(self):
        log.info("Starting cleanup for model '%s' on backend '%s' slot %d (key %s): cancelled=%s, stream_complete=%s",
                  self.model_name, self.backend_id, self.slot_id, self.key_short, self._cancelled, self._stream_complete)
        try:
            await self.resp.aclose()
            log.info("Response closed for model '%s' on backend '%s' slot %d (key %s)",
                      self.model_name, self.backend_id, self.slot_id, self.key_short)
        except Exception as e:
            log.info("Error closing response for model '%s' on backend '%s' slot %d (key %s): %s",
                      self.model_name, self.backend_id, self.slot_id, self.key_short, e)
        ok = False
        if self._stream_complete:
            log.info("Saving cache for model '%s' on backend '%s' slot %d (key %s)",
                      self.model_name, self.backend_id, self.slot_id, self.key_short)
            ok = await self._save()
            log.info("Cache save completed for model '%s' on backend '%s' slot %d (key %s): %s",
                      self.model_name, self.backend_id, self.slot_id, self.key_short, ok)
        else:
            log.info("Skipping cache save for model '%s' on backend '%s' slot %d (key %s): stream incomplete",
                      self.model_name, self.backend_id, self.slot_id, self.key_short)
        self.sm.release(self.model_name, self.backend_id, self.slot_id)
        log.info("Released slot %d for model '%s' on backend '%s' (key %s)", self.slot_id,
                  self.model_name, self.backend_id, self.key_short)
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
            if not heartbeat.done():
                heartbeat.cancel()
            await asyncio.shield(self._cleanup())


@app.post("/v1/chat/completions")
async def chat(req: Request):
    sm: SlotManager = app.state.sm

    t0 = time.time()
    client_ip = req.client.host if req.client else "-"
    data = await req.json()

    messages: List[Dict] = data.get("messages") or []
    stream = bool(data.get("stream", False))
    client_model = data.get("model") or MODEL_ID

    try:
        # 1. Resolve model name
        options = backend_manager.get_discovered_models(client_model)
        if not options:
            await backend_manager.discover_models()
            options = backend_manager.get_discovered_models(client_model)
            if not options:
                return JSONResponse({"error": f"model '{client_model}' not found"}, status_code=400)

        # 2. Tokenize + scan for cache hits
        prefix = hs.raw_prefix(messages)
        first_opt = options[0]
        first_client = backend_manager.get_client(first_opt.backends[0])
        first_token_ids = await first_client.tokenize(prefix)
        prompt_tokens = len(first_token_ids)
        min_ctx = min(opt.n_ctx for opt in options)
        if prompt_tokens >= min_ctx:
            return JSONResponse(
                {"error": f"prompt too long (tokens={prompt_tokens}, n_ctx={min_ctx})"},
                status_code=400,
            )
        canonical_name = first_opt.name

        restore_key = None
        restore_backend = None
        best_ratio = 0.0
        key = None
        blocks = None
        for opt in options:
            for be_id in opt.backends:
                opt_client = backend_manager.get_client(be_id)
                try:
                    opt_token_ids = await opt_client.tokenize(prefix)
                except Exception:
                    continue
                opt_blocks = hs.block_hashes_from_tokens(opt_token_ids, WORDS_PER_BLOCK)
                cand = kv_meta.find_best_restore_candidate(opt_blocks, WORDS_PER_BLOCK, LCP_TH, opt.name, be_id)
                if cand and cand[1] > best_ratio:
                    best_ratio = cand[1]
                    restore_key = cand[0]
                    restore_backend = be_id
                    canonical_name = opt.name
                    key = hs.meta_key(opt.name, opt_token_ids)
                    blocks = opt_blocks
                    log.info("Cache hit: key '%s' (model '%s', backend '%s', ratio %.3f) — replacing previous best",
                             restore_key[:16], canonical_name, restore_backend, best_ratio)

        log.info(
            "Chat request from %s: model '%s', %d tokens, restore key=%s on backend %s",
            client_ip, client_model, prompt_tokens,
            restore_key[:16] if restore_key else None,
            restore_backend,
        )

        

        # 5. Build candidate backends list (fallback ONLY, cache backend excluded)
        candidate_backends: list[tuple[str, str]] = []
        if restore_backend:
            for opt in options:
                for be_id in opt.backends:
                    if be_id != restore_backend:
                        candidate_backends.append((be_id, opt.name))
        else:
            for opt in options:
                for be_id in opt.backends:
                    candidate_backends.append((be_id, opt.name))

        # 6. Acquire slot (cache backend tried first via restore_info, then fallback)
        restore_info: Optional[tuple[str, str, str]] = None
        if restore_key and restore_backend:
            restore_info = (restore_key, restore_backend, canonical_name)
        g, restored = await asyncio.wait_for(
            sm.acquire_for_request(candidate_backends, restore_info, blocks, prompt_tokens),
            timeout=ACQUIRE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.error("Timed out waiting for slot from client %s for model '%s'", client_ip, client_model)
        return JSONResponse({"error": "all slots busy, please retry later"}, status_code=503)

    model_name, be_id, slot_id = g
    client = backend_manager.get_client(be_id)

    # Fallback: if no cache hit found, generate key/blocks from the serving backend's model name
    if key is None:
        key = hs.meta_key(model_name, first_token_ids)
        blocks = hs.block_hashes_from_tokens(first_token_ids, WORDS_PER_BLOCK)
        log.info("No cache hit: using key '%s' for model '%s' (client model '%s')", key[:16], model_name, client_model)

    log.info("Slot acquired: model '%s' on backend '%s' slot %d, restored=%s, key='%s', canonical_name='%s'",
             model_name, be_id, slot_id, restored, key[:16], canonical_name)

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
        client_ip,
        model_name,
        be_id,
        slot_id,
        restore_key[:16] if restore_key else None,
        restored,
    )

    _reader_created = False
    try:
        if stream:
            resp = await client.chat_completions(
                body,
                slot_id=slot_id,
                stream=True,
            )
            if resp.status_code != 200:
                err_txt = await resp.aread()
                await resp.aclose()
                return JSONResponse(
                    {"error": err_txt.decode("utf-8", "ignore")},
                    status_code=resp.status_code,
                )

            reader = StreamReader(resp, req, model_name, be_id, slot_id,
                                  key, prompt_tokens, blocks, sm, best_ratio, restored)
            gen = reader.stream()
            _reader_created = True

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
            return StreamingResponse(
                gen,
                media_type="text/event-stream",
                headers=headers,
            )

        else:
            out = await client.chat_completions(
                body,
                slot_id=slot_id,
                stream=False,
            )
            if not isinstance(out, dict):
                return JSONResponse(
                    {"error": "provider non-JSON body"},
                    status_code=502,
                )

            ok = False
            if restored and best_ratio < CACHE_SAVE_RATIO_THRESHOLD:
                ok, cache_size = await sm.save_after(
                    model_name, be_id, slot_id, key, blocks,
                )
                if ok:
                    kv_meta.write_meta(
                        key,
                        prompt_tokens,
                        blocks,
                        WORDS_PER_BLOCK,
                        canonical_name,
                        be_id,
                        cache_size,
                    )
            else:
                sm._slot_save_skipped[(model_name, be_id, slot_id)] = True

            # log.info(
            #     "json_response\n%s",
            #     json.dumps(out, indent=2, ensure_ascii=False),
            # )
            # log.info(
            #     "json_done client_ip=%s model=%s be=%s slot=%d key=%s saved=%s is_big=%s dur_ms=%d",
            #     client_ip, model_name, be_id, slot_id,
            #     key[:16],
            #     ok,
            #     is_big,
            #     int((time.time() - t0) * 1000),
            # )
            return JSONResponse(content=out, status_code=200)

    except Exception as e:
        log.exception("Chat error for client %s, model '%s' on backend '%s' slot %d (key %s): %s",
                       client_ip, model_name, be_id, slot_id, key[:16], e)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if not _reader_created:
            sm.release(model_name, be_id, slot_id)
