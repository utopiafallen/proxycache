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

from config import (BACKENDS, WORDS_PER_BLOCK, BIG_THRESHOLD_WORDS,
                    LCP_TH, META_DIR, MODEL_ID, PORT,
                    CACHE_DIR, CACHE_MAX_AGE_HOURS, CACHE_MAX_SIZE_GB,
                    SLOT_TIMEOUT)

import hashing as hs
from slot_manager import SlotManager

log = logging.getLogger(__name__)

from backend_manager import backend_manager

ACQUIRE_TIMEOUT = 60.0
STREAM_QUEUE_SIZE = 16
STREAM_QUEUE_TIMEOUT = 5.0

app = FastAPI(title="Simple KV Proxy")


@app.on_event("startup")
async def startup():
    sm = SlotManager()
    sm.init_from_disk(CACHE_DIR)

    app.state.sm = sm
    app.state.executor = ThreadPoolExecutor(max_workers=2)
    log.info("app_start n_backends=%d port=%d backends=%s", len(BACKENDS), PORT,
             [be["url"] for be in BACKENDS])

    # Reconcile meta files on startup (remove corrupted/orphaned entries)
    backend_keys = list(backend_manager._backends.keys())
    backend_agents = {k: v.agent_client.base_url if v.agent_client else None for k, v in backend_manager._backends.items()}
    backend_agents = {k: v for k, v in backend_agents.items() if v is not None}
    reconciled = await hs.reconcile_meta(META_DIR, CACHE_DIR, backend_keys, backend_agents)
    if reconciled > 0:
        log.info("Cleaned up %d orphaned/corrupted meta files at startup", reconciled)

    # Log startup sanity summary
    if CACHE_DIR and os.path.isdir(CACHE_DIR):
        cache_files = len(os.listdir(CACHE_DIR))
        log.info("startup_sanity: %d meta files after reconcile, %d cache files on disk",
                 len(hs.scan_all_meta()), cache_files)


@app.on_event("shutdown")
async def shutdown():
    executor = getattr(app.state, "executor", None)
    await backend_manager.close()
    if executor:
        executor.shutdown(wait=False)


@app.get("/v1/models")
async def models():
    resp = await backend_manager.get_client(backend_manager.first_key()).client.get("/v1/models")
    return resp.json()


class StreamReader:
    def __init__(self, resp: httpx.Response, req: Request,
                 model_name: str, backend_id: str, slot_id: int,
                 key: str, prefix: str, blocks: List[str],
                 sm: SlotManager):
        self.resp = resp
        self.req = req
        self.model_name = model_name
        self.backend_id = backend_id
        self.slot_id = slot_id
        self.key = key
        self.prefix = prefix
        self.blocks = blocks
        self.sm = sm
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
        log.info("stream_reader_start model=%s be=%s slot=%d key=%s status=%s",
                 self.model_name, self.backend_id, self.slot_id, self.key_short,
                 status)
        log.info("stream_reader_headers model=%s be=%s slot=%d key=%s headers=%s",
                 self.model_name, self.backend_id, self.slot_id, self.key_short,
                 headers)
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
                    log.warning("read_loop_client_disconnected model=%s be=%s slot=%d key=%s",
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
                            "stream_complete client_ip=%s model=%s be=%s slot=%d key=%s status=%s headers=%s chunks=%d bytes=%d",
                            client_ip, self.model_name, self.backend_id, self.slot_id, self.key_short,
                            status, headers,
                            chunks_received, total_bytes,
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
                            log.warning("stream_queue_full model=%s be=%s slot=%d key=%s",
                                        self.model_name, self.backend_id, self.slot_id, self.key_short)
                            self._cancelled = True
                            self._disconnect_event.set()
                            for t in pending:
                                t.cancel()
                            break
                        except asyncio.CancelledError:
                            log.warning("stream_reader_cancelled_put model=%s be=%s slot=%d key=%s",
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
                log.warning("heartbeat_client_disconnected model=%s be=%s slot=%d key=%s",
                            self.model_name, self.backend_id, self.slot_id, self.key_short)
                self._cancelled = True
                self._disconnect_event.set()
                break

    def _signal_done(self):
        """Send None sentinel to unblock stream consumer."""
        self._cancelled = True
        self.queue.put_nowait(None)

    async def _save(self) -> tuple:
        ok = False
        try:
            ok, _ = await asyncio.wait_for(
                self.sm.save_after(
                    self.model_name, self.backend_id, self.slot_id,
                    self.key, self.model_name, self.blocks,
                ),
                timeout=SLOT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning("save_after_timeout model=%s be=%s slot=%d",
                        self.model_name, self.backend_id, self.slot_id)
        except asyncio.CancelledError:
            log.warning("save_after_cancelled model=%s be=%s slot=%d",
                        self.model_name, self.backend_id, self.slot_id)
        except Exception as e:
            log.warning("save_after_exception model=%s be=%s slot=%d: %s",
                        self.model_name, self.backend_id, self.slot_id, e)
        if ok:
            try:
                hs.write_meta(self.key, self.prefix, self.blocks,
                              WORDS_PER_BLOCK, self.model_name, self.backend_id)
            except Exception as e:
                log.warning("write_meta_exception key=%s: %s", self.key_short, e)
        return ok

    async def _cleanup(self):
        log.info("cleanup_start model=%s be=%s slot=%d key=%s cancelled=%s stream_complete=%s",
                 self.model_name, self.backend_id, self.slot_id, self.key_short, self._cancelled, self._stream_complete)
        try:
            await self.resp.aclose()
            log.info("cleanup_aclose_done model=%s be=%s slot=%d key=%s",
                     self.model_name, self.backend_id, self.slot_id, self.key_short)
        except Exception as e:
            log.info("cleanup_aclose_error model=%s be=%s slot=%d key=%s error=%s",
                     self.model_name, self.backend_id, self.slot_id, self.key_short, e)
        ok = False
        if self._stream_complete:
            log.info("cleanup_save_start model=%s be=%s slot=%d key=%s",
                     self.model_name, self.backend_id, self.slot_id, self.key_short)
            ok = await self._save()
            log.info("cleanup_save_done model=%s be=%s slot=%d key=%s ok=%s",
                     self.model_name, self.backend_id, self.slot_id, self.key_short, ok)
        else:
            log.info("cleanup_save_skipped model=%s be=%s slot=%d key=%s",
                     self.model_name, self.backend_id, self.slot_id, self.key_short)
        self.sm.release(self.model_name, self.backend_id, self.slot_id)
        log.info("cleanup_release_done model=%s be=%s slot=%d key=%s",
                 self.model_name, self.backend_id, self.slot_id, self.key_short)
        log.info("stream_reader_done model=%s be=%s slot=%d key=%s saved=%s",
                 self.model_name, self.backend_id, self.slot_id, self.key_short, ok)

    async def _reader(self):
        try:
            await self._read_loop()
            self.queue.put_nowait(None)
        except asyncio.CancelledError:
            log.warning("stream_reader_cancelled model=%s be=%s slot=%d key=%s",
                        self.model_name, self.backend_id, self.slot_id, self.key_short)
        except Exception as e:
            log.exception("stream_reader_error model=%s be=%s slot=%d key=%s: %s",
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
                        log.warning("stream_client_disconnected model=%s be=%s slot=%d key=%s",
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
    backend_model_id = client_model

    prefix = hs.raw_prefix(messages)
    full_for_key = backend_model_id + "\n" + prefix
    key = hs.prefix_key_sha256(full_for_key)
    blocks = hs.block_hashes_from_text(prefix, WORDS_PER_BLOCK)
    n_words = len(hs.words_from_text(prefix))
    is_big = n_words > BIG_THRESHOLD_WORDS

    restore_key: Optional[str] = None
    if is_big:
        # Get the first backend for this model (will be refined during slot acquisition)
        candidate_backends = backend_manager.get_backends_for_model(backend_model_id)
        if candidate_backends:
            first_backend = candidate_backends[0]
            cand = hs.find_best_restore_candidate(
                blocks,
                WORDS_PER_BLOCK,
                LCP_TH,
                backend_model_id,
                first_backend,
            )
        else:
            cand = None
        if cand:
            restore_key, ratio = cand
            log.info(
                "restore_candidate basename=%s ratio=%.3f",
                restore_key[:16],
                ratio,
            )
        else:
            log.info("restore_candidate none")
    else:
        log.info(
            "small_request n_words=%d threshold=%d",
            n_words,
            BIG_THRESHOLD_WORDS,
        )

    log.info(
        "chat_request client_ip=%s is_big=%s n_words=%d model=%s",
        client_ip, is_big, n_words, backend_model_id,
    )

    try:
        g, lock, restored = await asyncio.wait_for(
            sm.acquire_for_request(
                backend_model_id,
                restore_key if is_big else None,
                blocks if is_big else None,
            ),
            timeout=ACQUIRE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.error(
            "acquire_timeout client_ip=%s is_big=%s restore_key=%s model=%s",
            client_ip,
            is_big,
            restore_key[:16] if restore_key else None,
            backend_model_id,
        )
        return JSONResponse(
            {"error": "all slots busy, please retry later"},
            status_code=503,
        )

    model_name, be_id, slot_id = g
    client = backend_manager.get_client(be_id)

    log.info("after_acquire client_ip=%s model=%s be=%s slot=%d restored=%s",
             client_ip, model_name, be_id, slot_id, restored)

    body = dict(data)
    body["model"] = client_model
    opts = dict(body.get("options") or {})
    opts["slot_id"] = slot_id
    opts["id_slot"] = slot_id
    opts["n_keep"] = -1
    opts["cache_prompt"] = True
    body["options"] = opts
    body["n_keep"] = -1
    body["cache_prompt"] = True

    log.info(
        "dispatch client_ip=%s model=%s be=%s slot=%d is_big=%s (restore_target=%s restored=%s)",
        client_ip,
        model_name,
        be_id,
        slot_id,
        is_big,
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
                                  key, prefix, blocks, sm)
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
            if is_big:
                ok, _ = await sm.save_after(
                    model_name, be_id, slot_id, key, backend_model_id, blocks,
                )
                if ok:
                    hs.write_meta(
                        key,
                        prefix,
                        blocks,
                        WORDS_PER_BLOCK,
                        backend_model_id,
                        be_id,
                    )

            log.info(
                "json_response\n%s",
                json.dumps(out, indent=2, ensure_ascii=False),
            )
            log.info(
                "json_done client_ip=%s model=%s be=%s slot=%d key=%s saved=%s is_big=%s dur_ms=%d",
                client_ip, model_name, be_id, slot_id,
                key[:16],
                ok,
                is_big,
                int((time.time() - t0) * 1000),
            )
            return JSONResponse(content=out, status_code=200)

    except Exception as e:
        log.exception("chat_error client_ip=%s model=%s be=%s slot=%d key=%s: %s",
                       client_ip, model_name, be_id, slot_id, key[:16], e)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if not _reader_created:
            sm.release(model_name, be_id, slot_id)
