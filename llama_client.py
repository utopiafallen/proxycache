# llama_client.py

# -*- coding: utf-8 -*-

"""
HTTP client for llama.cpp: /v1/chat/completions (stream/non-stream), /slots save/restore.

- stream: build_request+send(stream=True), raw bytes.
- non-stream: strict JSON parsing + fallback if content-type is not JSON.
- /slots: filename in JSON body (avoids 500 parse error on some builds).
- Slot pinning is duplicated in root/options/query.
"""

import asyncio
import httpx
import logging
from typing import Dict, Optional, Tuple
from urllib.parse import quote

from config import REQUEST_TIMEOUT, BACKEND_MODE, SLOT_TIMEOUT, DEFAULT_N_CTX

log = logging.getLogger(__name__)


class LlamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=REQUEST_TIMEOUT,
            limits=limits,
        )
        log.info("Initialized HTTP client for %s (httpx %s)", base_url, httpx.__version__)

    async def close(self):
        await self.client.aclose()

    async def tokenize(self, text: str, add_special: bool = False) -> list[int]:
        resp = await self.client.post(
            "/tokenize",
            json={"content": text, "add_special": add_special},
        )
        resp.raise_for_status()
        return resp.json().get("tokens", [])

    @staticmethod
    def _with_slot_id(body: Dict, slot_id: Optional[int]) -> Tuple[Dict, Dict]:
        if slot_id is None:
            return body, {}

        new_body = dict(body)

        # root
        new_body["_slot_id"] = slot_id
        new_body["slot_id"] = slot_id
        new_body["id_slot"] = slot_id

        # options
        opts = dict(new_body.get("options") or {})
        opts["slot_id"] = slot_id
        opts["id_slot"] = slot_id
        new_body["options"] = opts

        # query
        query = {"slot_id": slot_id, "id_slot": slot_id}
        return new_body, query

    async def chat_completions(
        self,
        body: Dict,
        slot_id: Optional[int] = None,
        stream: bool = False,
    ):
        body2, query = self._with_slot_id(body, slot_id)

        log.info("Chat completions: stream=%s, body_stream=%s, %d messages",
                 stream, body2.get("stream", "MISSING"),
                 len(body2.get("messages") or []))

        if stream:
            req = self.client.build_request(
                "POST",
                "/v1/chat/completions",
                json=body2,
                params=query,
            )
            resp = await self.client.send(req, stream=True)
            return resp

        resp = await self.client.post(
            "/v1/chat/completions",
            json=body2,
            params=query,
        )
        resp.raise_for_status()

        ctype = resp.headers.get("content-type", "")
        if "application/json" not in ctype:
            raw = resp.text or ""
            log.error(
                "Non-JSON response from provider: content_type=%s, body length=%d",
                ctype,
                len(raw),
            )
            return {
                "object": "error",
                "message": "provider returned non-JSON",
                "raw": raw[:2048],
            }

        try:
            return resp.json()
        except Exception as e:
            raw = resp.text or ""
            log.error(
                "Failed to parse JSON response: status=%d, body length=%d, error=%s",
                resp.status_code,
                len(raw),
                e,
            )
            return {
                "object": "error",
                "message": "invalid json from provider",
                "raw": raw[:2048],
            }

    async def save_slot(self, slot_id: int, basename: str, model_name: str = None) -> Tuple[bool, int]:
        # JSON body: {"filename": "..."} — avoids 500 on some builds
        if BACKEND_MODE == "llama-swap" and model_name:
            path = f"/upstream/{quote(model_name, safe='')}/slots/{slot_id}"
        else:
            path = f"/slots/{slot_id}"

        try:
            resp = await self.client.post(
                path,
                params={"action": "save"},
                json={"filename": basename, "model": model_name},
            )
        except Exception as e:
            log.warning(
                "Save slot failed: slot=%d, file=%s, error=%s",
                slot_id, basename[:16], e,
            )
            return False, 0

        if resp.status_code == 500:
            log.warning(
                "Save slot returned 500: slot=%d, file=%s",
                slot_id,
                basename[:16],
            )
            return False, 0

        resp.raise_for_status()
        try:
            data = resp.json()
            n_written = data.get("n_written", 0)
        except Exception:
            n_written = 0
        return True, n_written

    async def restore_slot(self, slot_id: int, basename: str, model_name: str = None) -> bool:
        if BACKEND_MODE == "llama-swap" and model_name:
            path = f"/upstream/{quote(model_name, safe='')}/slots/{slot_id}"
        else:
            path = f"/slots/{slot_id}"

        try:
            resp = await asyncio.wait_for(
                self.client.post(
                    path,
                    params={"action": "restore"},
                    json={"filename": basename, "model": model_name},
                ),
                timeout=SLOT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning(
                "Restore slot timed out after %ds: slot=%d, file=%s",
                SLOT_TIMEOUT, slot_id, basename[:16],
            )
            return False

        if resp.status_code != 200:
            log.warning(
                "Restore slot failed: status=%d, slot=%d, file=%s",
                resp.status_code,
                slot_id,
                basename[:16],
            )
            return False

        return True

    async def _parse_router_models(self) -> list:
        """
        GET /models — returns list of {id, status.value, status.args} for each
        child-process model.  Only models with status == 'loaded' are returned.
        Each entry has {'name': str, 'port': int}.
        """
        try:
            resp = await self.client.get("/models")
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning("Failed to parse router models from %s: %s", self.base_url, e)
            return []

        models_data = data.get("data") or []
        result = []
        for m in models_data:
            status_obj = m.get("status") or {}
            if status_obj.get("value") != "loaded":
                continue
            args = status_obj.get("args") or []
            port = None
            for i, arg in enumerate(args):
                if arg == "--port" and i + 1 < len(args):
                    try:
                        port = int(args[i + 1])
                    except ValueError:
                        pass
                    break
            if port is None:
                log.warning("Router model '%s' has no port in status args", m.get("id", "?"))
                continue
            result.append({"name": m.get("id", ""), "port": port})
        return result

    async def _get_child_slots(self, child_url: str) -> Optional[list]:
        """GET /slots on a child process, returns list or None."""
        try:
            resp = await self.client.get(f"{child_url}/slots")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning("Failed to get slots from child process %s: %s", child_url, e)
            return None

    async def get_slots_info(self, model_name: Optional[str] = None) -> Optional[list]:
        """GET /slots — returns list of slot dicts, or None on error.

        Fallback for router mode: when the main /slots endpoint returns
        HTTP 400 ("model name is missing from the request"), we query
        GET /models to find loaded child-process models, then call
        /slots on the relevant child's own port.

        If model_name is provided, only queries that model's child process.
        If model_name is None, queries all loaded child processes.
        """
        # Fast path: normal (non-router) mode
        try:
            resp = await self.client.get("/slots")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400:
                log.info(
                    "Got 400 from /slots on %s — falling back to /models for router mode",
                    self.base_url,
                )
                return await self._get_slots_via_router_models(model_name)
            raise
        except Exception as e:
            log.warning("Failed to get slots info from %s: %s", self.base_url, e)
            return None

    async def _get_slots_via_router_models(self, model_name: Optional[str] = None) -> Optional[list]:
        """Query slots from loaded child processes via GET /models.

        If model_name is provided, only queries that model's child process.
        Otherwise queries all loaded child processes.
        """
        models = await self._parse_router_models()
        if not models:
            log.warning("No loaded models returned from /models on %s", self.base_url)
            return None

        # Filter to requested model if specified
        if model_name:
            models = [m for m in models if m["name"] == model_name]
            if not models:
                log.warning(
                    "Model '%s' not loaded, available models: %s",
                    model_name, [m["name"] for m in models],
                )
                return None

        all_slots = []
        for m in models:
            child_url = f"http://127.0.0.1:{m['port']}"
            slots = await self._get_child_slots(child_url)
            if slots and isinstance(slots, list):
                for s in slots:
                    s["_router_model"] = m["name"]
                    s["_router_port"] = m["port"]
                all_slots.extend(slots)
                log.debug(
                    "Child process %s:%d returned %d slots",
                    m["name"], m["port"], len(slots),
                )
            else:
                log.warning(
                    "Child process %s:%d returned no slots",
                    m["name"], m["port"],
                )

        if all_slots:
            return all_slots
        log.warning("No slots found from any child process on %s", self.base_url)
        return None

    async def discover_models(self) -> list[tuple[str, int]]:
        """Discover models served by this backend. Returns [(name, n_ctx)]."""
        # Router mode: GET /models
        try:
            resp = await self.client.get("/models")
            data = resp.json()
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                models = []
                for entry in data["data"]:
                    status = entry.get("status") or {}
                    if status.get("value") != "loaded":
                        continue
                    name = entry.get("id", "")
                    n_ctx = DEFAULT_N_CTX
                    args = status.get("args") or []
                    for i, arg in enumerate(args):
                        if arg in ("-ctx", "-c", "--ctx-size") and i + 1 < len(args):
                            try:
                                n_ctx = int(args[i + 1])
                                break
                            except ValueError:
                                pass
                    if n_ctx == DEFAULT_N_CTX:
                        loaded = status.get("loaded_info") or {}
                        n_ctx = loaded.get("n_ctx") or loaded.get("n_ctx_train") or DEFAULT_N_CTX
                    if name:
                        models.append((name, int(n_ctx)))
                if models:
                    return models
        except Exception:
            pass

        # Non-router mode: GET /v1/models
        try:
            resp = await self.client.get("/v1/models")
            data = resp.json()
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                for entry in data["data"]:
                    meta = entry.get("meta")
                    name = entry.get("id", "")
                    n_ctx = DEFAULT_N_CTX
                    if meta and isinstance(meta, dict):
                        n_ctx = meta.get("n_ctx", DEFAULT_N_CTX)
                    if name:
                        return [(name, int(n_ctx))]
        except Exception:
            pass

        return []
