# backend_manager.py

# -*- coding: utf-8 -*-

"""
BackendManager: singleton managing backend registry, clients, agent clients,
and model-to-backend mapping.

Key derivation: strips protocol, keeps host:port.
e.g. "http://10.0.0.1:8000" -> "10.0.0.1:8000"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List

from config import BACKENDS, REFRESH_COOLDOWN_SECONDS
from llama_client import LlamaClient
from cache_agent_client import CacheAgentClient

log = logging.getLogger(__name__)


@dataclass
class BackendInfo:
    client: LlamaClient
    agent_client: CacheAgentClient | None


class BackendManager:
    """Singleton managing backend registry, clients, agent clients, and model mapping.

    Parses BACKENDS config directly in the constructor. No public
    registration API — backends are configured once at startup and never change.

    Key derivation: strips protocol, keeps host:port.
    e.g. "http://10.0.0.1:8000" -> "10.0.0.1:8000"
    """

    def __init__(self, backends_config: list[dict]):
        self._backends: dict[str, BackendInfo] = {}
        self._first_key: str | None = None
        self._model_to_backends: dict[str, list[str]] = {}
        self._refresh_state: dict[tuple[str, str], tuple[float, bool]] = {}

        for be in backends_config:
            url = be["url"].rstrip("/")
            key = url.split("://")[-1]  # "10.0.0.1:8000"
            client = LlamaClient(url)
            agent_client = None
            if "agent_port" in be:
                host = key.rsplit(":", 1)[0]
                agent_client = CacheAgentClient(f"http://{host}:{be['agent_port']}")
            self._backends[key] = BackendInfo(client=client, agent_client=agent_client)
            if self._first_key is None:
                self._first_key = key

        log.info("backend_manager n_backends=%d keys=%s",
                 len(self._backends), list(self._backends.keys()))

    # --- Accessors ---

    def get_client(self, key: str) -> LlamaClient:
        be = self._backends.get(key)
        if be is None:
            raise KeyError(f"Unknown backend key: {key}")
        return be.client

    def get_agent(self, key: str) -> CacheAgentClient | None:
        be = self._backends.get(key)
        if be is None:
            raise KeyError(f"Unknown backend key: {key}")
        return be.agent_client

    def keys(self) -> list[str]:
        return list(self._backends.keys())

    def first_key(self) -> str:
        if self._first_key is None:
            raise RuntimeError("No backends configured")
        return self._first_key

    def n_backends(self) -> int:
        return len(self._backends)

    async def close(self):
        for info in self._backends.values():
            await info.client.close()

    # --- Model registration (internal) ---

    def _register_for_model(self, model_name: str, backend_key: str) -> None:
        """Register a backend as serving a model (internal)."""
        if model_name not in self._model_to_backends:
            self._model_to_backends[model_name] = []
        if backend_key not in self._model_to_backends[model_name]:
            self._model_to_backends[model_name].append(backend_key)

    def get_backends_for_model(self, model_name: str) -> list[str]:
        return self._model_to_backends.get(model_name, [])

    async def refresh_models(self, model_name: str) -> dict[str, int]:
        """Query all backends for model slots.

        Returns {backend_key: n_slots} for backends that returned slot info.
        Registers backends for the model on success (not on failure —
        SlotManager handles the fallback).
        Tracks cooldown per (model, backend): 300s after success, 30s after failure.
        """
        backend_keys = self.keys()
        log.info(
            "refresh_models model=%s n_backends=%d known=%s",
            model_name, len(backend_keys),
            list(self._model_to_backends.get(model_name, [])),
        )

        if not backend_keys:
            log.error(
                "refresh_models_no_backends model=%s — no backends configured",
                model_name,
            )
            raise RuntimeError(f"No backends configured for model={model_name}")

        slot_counts: dict[str, int] = {}
        refreshed_any = False

        for backend_key in backend_keys:
            client = self.get_client(backend_key)

            # Check cooldown — 30s after failure, 300s after success
            refresh_key = (model_name, backend_key)
            now = time.time()
            last_ts, last_success = self._refresh_state.get(refresh_key, (0.0, True))
            cooldown = 30 if not last_success else REFRESH_COOLDOWN_SECONDS
            if now - last_ts < cooldown:
                log.debug("refresh_models_cooldown model=%s be=%s last=%.1f",
                          model_name, backend_key, last_ts)
                continue

            # get_slots_info() handles both router and non-router modes internally
            try:
                slots = await client.get_slots_info(model_name)
            except Exception as e:
                log.warning(
                    "refresh_models_get_slots_info_fail model=%s be=%s err=%s",
                    model_name, backend_key, e,
                )
                slots = None

            if slots and isinstance(slots, list):
                # Filter by model name if router mode (slots have _router_model field)
                if slots and isinstance(slots[0], dict) and "_router_model" in slots[0]:
                    model_slots = [s for s in slots if s.get("_router_model") == model_name]
                    n_slots = len(model_slots)
                    log.info(
                        "refresh_models model=%s be=%s slots=%d (router)",
                        model_name, backend_key, n_slots,
                    )
                else:
                    # Non-router mode: all slots belong to this model
                    n_slots = len(slots)
                    log.info(
                        "refresh_models model=%s be=%s slots=%d (non-router)",
                        model_name, backend_key, n_slots,
                    )
                self._register_for_model(model_name, backend_key)
                slot_counts[backend_key] = n_slots
                self._refresh_state[refresh_key] = (now, True)
                refreshed_any = True
            else:
                # Slots unavailable (model not loaded yet or discovery failed)
                log.warning(
                    "refresh_models_model_not_loaded model=%s be=%s",
                    model_name, backend_key,
                )
                self._refresh_state[refresh_key] = (now, False)
                refreshed_any = True

        if not refreshed_any:
            log.warning(
                "refresh_models_nothing_done model=%s — all backends skipped (cooldown or no client)",
                model_name,
            )

        return slot_counts


# Module-level singleton
backend_manager = BackendManager(BACKENDS)
