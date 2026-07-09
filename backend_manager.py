# backend_manager.py

# -*- coding: utf-8 -*-

"""
BackendManager: singleton managing backend registry, clients, agent clients,
and model-to-backend mapping.

Key derivation: strips protocol, keeps host:port.
e.g. "http://10.0.0.1:8000" -> "10.0.0.1:8000"
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config import BACKENDS, DEFAULT_N_CTX
from llama_client import LlamaClient
from cache_agent_client import CacheAgentClient
from hashing import sanitize_backend_dir
from metrics import metrics

log = logging.getLogger(__name__)


@dataclass
class DiscoveredModel:
    name: str
    n_ctx: int
    backends: list[str]
    total_slots: int
    last_discovered: float


@dataclass
class BackendInfo:
    client: LlamaClient
    agent_client: CacheAgentClient | None
    cache_dir: str | None
    cache_max_size_gb: float = 25.0


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
        self._refresh_state: dict[tuple[str, str], tuple[float, bool, int]] = {}
        self._discovered_models: dict[str, DiscoveredModel] = {}
        self._backend_state: dict[str, bool] = {}
        self._discovery_task: asyncio.Task | None = None

        for be in backends_config:
            url = be["url"].rstrip("/")
            raw_key = url.split("://")[-1]  # "10.0.0.1:8000"
            key = sanitize_backend_dir(raw_key)  # "10-0-0-1-8000"
            client = LlamaClient(url)
            agent_client = None
            cache_dir = be.get("cache_dir")
            if "agent_port" in be and cache_dir:
                raise ValueError(
                    f"Backend {url}: cache_dir and agent_port are mutually exclusive. "
                    "Use cache_dir for local cache management or agent_port for remote cache-agent."
                )
            if "agent_port" not in be and not cache_dir:
                raise ValueError(
                    f"Backend {url}: must specify either cache_dir or agent_port. "
                    "cache_dir for local filesystem cache management, agent_port for remote cache-agent."
                )
            if "agent_port" in be:
                host = raw_key.rsplit(":", 1)[0]
                agent_client = CacheAgentClient(f"http://{host}:{be['agent_port']}")
            cache_max_size_gb = float(be.get("cache_max_size_gb", 25.0))
            self._backends[key] = BackendInfo(client=client, agent_client=agent_client, cache_dir=cache_dir, cache_max_size_gb=cache_max_size_gb)
            if self._first_key is None:
                self._first_key = key

        log.info("Backend manager initialized with %d backends: %s",
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

    def get_cache_dir(self, key: str) -> str | None:
        be = self._backends.get(key)
        if be is None:
            raise KeyError(f"Unknown backend key: {key}")
        return be.cache_dir

    def has_cache_config(self, key: str) -> bool:
        be = self._backends.get(key)
        if be is None:
            raise KeyError(f"Unknown backend key: {key}")
        return be.agent_client is not None or be.cache_dir is not None

    def get_cache_max_size_gb(self, key: str) -> float:
        be = self._backends.get(key)
        if be is None:
            raise KeyError(f"Unknown backend key: {key}")
        return getattr(be, 'cache_max_size_gb', 25.0)

    async def cache_delete(self, backend_id: str, key: str) -> bool:
        """Delete a cache file via agent or local filesystem."""
        be = self._backends.get(backend_id)
        if be is None:
            return False
        if be.agent_client:
            return await be.agent_client.delete(key)
        if be.cache_dir:
            import os
            cache_path = os.path.join(be.cache_dir, key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                return True
        return False

    async def cache_get_size(self, backend_id: str, key: str) -> int:
        """Get cache file size via agent or local filesystem."""
        be = self._backends.get(backend_id)
        if be is None:
            return 0
        if be.agent_client:
            from cache_agent_client import get_file_size
            result = await get_file_size(be.agent_client.base_url, key)
            if result and result.get("exists", False):
                return result.get("size", 0)
            return 0
        if be.cache_dir:
            import os
            cache_path = os.path.join(be.cache_dir, key)
            if os.path.exists(cache_path):
                return os.stat(cache_path).st_size
        return 0

    def cache_get_mtime(self, backend_id: str, key: str) -> float:
        """Get cache file last-modified time via local filesystem.
        Agent does not provide mtime, so this only works with cache_dir."""
        be = self._backends.get(backend_id)
        if be is None:
            return time.time()
        if be.cache_dir:
            import os
            cache_path = os.path.join(be.cache_dir, key)
            if os.path.exists(cache_path):
                return os.path.getmtime(cache_path)
        return time.time()

    async def cache_exists(self, backend_id: str, key: str) -> bool:
        """Check if a cache file exists via agent or local filesystem."""
        be = self._backends.get(backend_id)
        if be is None:
            return False
        if be.agent_client:
            result = await be.agent_client.get_file_size(key)
            if result is not None:
                return result.get("exists", False)
            return False
        if be.cache_dir:
            import os
            cache_path = os.path.join(be.cache_dir, key)
            return os.path.exists(cache_path)
        return False

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

    # --- Model discovery ---

    async def discover_models(self) -> dict[str, DiscoveredModel]:
        """Discover models across all backends. Returns merged registry.
        Always performs fresh discovery. Result stored in _discovered_models.
        """
        all_discovered: dict[str, list[tuple[str, int]]] = {}
        for backend_key in self.keys():
            models = await self.get_client(backend_key).discover_models()
            log.info("discover_models on backend '%s': %s", backend_key, models)
            if not models:
                log.warning("No models discovered on backend '%s'", backend_key)
                continue
            for name, n_ctx in models:
                if name not in all_discovered:
                    all_discovered[name] = []
                all_discovered[name].append((backend_key, n_ctx))

        merged = {}
        for name, entries in all_discovered.items():
            backends = [be for be, _ in entries]
            min_ctx = min(ctx for _, ctx in entries)
            merged[name] = DiscoveredModel(
                name=name, n_ctx=min_ctx, backends=backends,
                total_slots=0,
                last_discovered=time.time(),
            )
        self._discovered_models = merged
        for name, info in merged.items():
            log.info("Discovered model '%s' on backends %s with n_ctx=%d",
                     name, info.backends, info.n_ctx)
        return merged

    def get_model_n_ctx(self, canonical_name: str) -> int:
        if canonical_name in self._discovered_models:
            return self._discovered_models[canonical_name].n_ctx
        return DEFAULT_N_CTX

    def get_discovered_models(self, model_name: str) -> list[DiscoveredModel]:
        """Resolve client model name to list of matching DiscoveredModel objects.
        1. Exact match in _discovered_models
        2. Substring match (case-insensitive)
        3. 'any' -> all discovered models
        """
        if model_name == "any":
            if not self._discovered_models:
                return []
            return list(self._discovered_models.values())

        if model_name in self._discovered_models:
            return [self._discovered_models[model_name]]

        return [info for info in self._discovered_models.values()
                if model_name.lower() in info.name.lower()]

    async def refresh_slot_counts(self) -> dict[str, dict[str, int]]:
        """Query all backends for slot counts. Returns {backend_key: {model_name: n_slots}}.
        No longer registers models -- discover_models() handles that.
        Refreshes every call — no cooldown throttle.
        """
        backend_keys = self.keys()
        log.info(
            "Refreshing slot counts: %d known models, %d backends",
            len(self._discovered_models), len(backend_keys),
        )

        if not backend_keys:
            log.error(
                "No backends configured — cannot refresh slot counts",
            )
            raise RuntimeError("No backends configured")

        slot_counts: dict[str, dict[str, int]] = {}
        refreshed_any = False

        for canonical_name, info in self._discovered_models.items():
            log.info("Model '%s' has backends: %s", canonical_name, info.backends)
            for backend_key in info.backends:
                if backend_key not in self._backends:
                    continue
                client = self.get_client(backend_key)
                refresh_key = (canonical_name, backend_key)
                now = time.time()

                try:
                    slots = await client.get_slots_info(canonical_name)
                except Exception as e:
                    log.warning(
                        "Failed to get slot info for model '%s' on backend '%s': %s",
                        canonical_name, backend_key, e,
                    )
                    slots = None

                if slots and isinstance(slots, list):
                    if slots and isinstance(slots[0], dict) and "_router_model" in slots[0]:
                        model_slots = [s for s in slots if s.get("_router_model") == canonical_name]
                        n_slots = len(model_slots)
                        log.info(
                            "Model '%s' on backend '%s' has %d slots (router mode)",
                            canonical_name, backend_key, n_slots,
                        )
                    else:
                        n_slots = len(slots)
                        log.info(
                            "Model '%s' on backend '%s' has %d slots",
                            canonical_name, backend_key, n_slots,
                        )
                    if backend_key not in slot_counts:
                        slot_counts[backend_key] = {}
                    slot_counts[backend_key][canonical_name] = n_slots
                    self._refresh_state[refresh_key] = (now, True, n_slots)
                    refreshed_any = True
                else:
                    log.warning(
                        "Model '%s' not loaded on backend '%s'",
                        canonical_name, backend_key,
                    )
                    self._refresh_state[refresh_key] = (now, False, 0)
                    refreshed_any = True

        if not refreshed_any:
            log.warning(
                "No backends refreshed: all failed or missing client",
            )

        # Update total_slots on each DiscoveredModel
        for canonical_name, info in self._discovered_models.items():
            total = sum(
                slot_counts.get(be, {}).get(canonical_name, 0)
                for be in info.backends
            )
            info.total_slots = total

        return slot_counts

    # --- Liveness checker ---

    async def start_liveness_checker(self):
        self._discovery_task = asyncio.create_task(self._liveness_loop())

    async def stop_liveness_checker(self):
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass

    async def _liveness_loop(self):
        """Ping backends every 5s, trigger discovery on state change."""
        while True:
            await asyncio.sleep(5.0)
            changed = False
            state_changes: List[Dict[str, Any]] = []
            for backend_key in self.keys():
                client = self.get_client(backend_key)
                is_up = False
                try:
                    await client.client.get("/health", timeout=2.0)
                    is_up = True
                except Exception:
                    pass
                old_state = self._backend_state.get(backend_key, False)
                if is_up != old_state:
                    self._backend_state[backend_key] = is_up
                    changed = True
                    state_changes.append({
                        "backend": backend_key,
                        "old_state": old_state,
                        "new_state": is_up,
                    })
            # Also trigger if an up backend has no models in the registry
            # (discovery previously failed or never ran for that backend)
            up_keys = {k for k, v in self._backend_state.items() if v}
            discovered_backends = {be
                                    for info in self._discovered_models.values()
                                    for be in info.backends}
            missing_models = up_keys - discovered_backends
            if missing_models:
                changed = True
                for be in missing_models:
                    state_changes.append({
                        "backend": be,
                        "old_state": "up_no_models",
                        "new_state": "up",
                    })
            if changed:
                try:
                    await self.discover_models()
                except Exception as e:
                    log.error("discover_models failed during liveness check: %s", e)
                try:
                    await self.refresh_slot_counts()
                except Exception:
                    log.exception("Failed to refresh slot counts after backend state change")

                # Record liveness event for diagnostics
                discovered_models = {name: list(info.backends)
                                      for name, info in self._discovered_models.items()}
                metrics.record({
                    "event": "liveness_change",
                    "state_changes": state_changes,
                    "discovered_models": discovered_models,
                })


# Module-level singleton
backend_manager = BackendManager(BACKENDS)
