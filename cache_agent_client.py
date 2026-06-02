# cache_agent_client.py

"""
HTTP client for cache-agent: POST /cache/delete?key=<basename>

Each backend has its own cache-agent running on the same host as llama.cpp,
typically on a separate port (e.g., 8082 vs 8080).
"""

import logging
from urllib.parse import urljoin

import httpx

from config import SLOT_TIMEOUT

log = logging.getLogger(__name__)


class CacheAgentClient:
    def __init__(self, base_url: str):
        """
        Args:
            base_url: Agent URL, e.g. "http://10.0.0.1:8082"
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=SLOT_TIMEOUT)

    async def delete(self, key: str) -> bool:
        """
        Delete a cache file from the agent's CACHE_DIR.

        Args:
            key: Cache file basename (e.g. "abc123.kv-cache.bin")

        Returns:
            True if deletion succeeded, False otherwise (agent down, file not found, etc.)
        """
        url = urljoin(self.base_url + "/", "/cache/delete")
        try:
            resp = await self._client.post(url, params={"key": key})
        except Exception as e:
            log.warning("cache_agent_error url=%s key=%s: %s",
                        self.base_url, key[:16], e)
            return False

        if resp.status_code != 200:
            log.warning("cache_agent_status=%d key=%s body=%s",
                        resp.status_code, key[:16], resp.text[:200])
            return False

        return True

    async def close(self):
        await self._client.aclose()
