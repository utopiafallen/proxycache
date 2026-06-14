# cache_agent_client.py

"""
HTTP client for cache-agent:
  - POST /cache/delete?key=<basename>
  - GET  /cache/files/<basename>

Each backend has its own cache-agent running on the same host as llama.cpp,
typically on a separate port (e.g., 8082 vs 8080).
"""

import logging
from typing import Optional, Dict, Any, List
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
            log.warning("Cache agent error on %s for key %s: %s",
                        self.base_url, key[:16], e)
            return False

        if resp.status_code != 200:
            log.warning("Cache agent returned status %d for key %s: %s",
                        resp.status_code, key[:16], resp.text[:200])
            return False

        return True

    async def get_file_size(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get file size and existence info for a cache file.

        Args:
            key: Cache file basename (the SHA256 key)

        Returns:
            {"size": int, "exists": True} if file exists,
            {"exists": False} if not found,
            None on connection error.
        """
        url = urljoin(self.base_url + "/", f"/cache/files/{key}")
        try:
            resp = await self._client.get(url)
        except Exception as e:
            log.warning("Cache agent size check error on %s for key %s: %s",
                        self.base_url, key[:16], e)
            return None

        if resp.status_code == 404:
            return {"exists": False}
        if resp.status_code != 200:
            log.warning("Cache agent size check returned status %d for key %s: %s",
                        resp.status_code, key[:16], resp.text[:200])
            return None

        return resp.json()

    async def batch_get_file_sizes(self, keys: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get file sizes for multiple cache files in one API call.

        Args:
            keys: List of cache file basenames (SHA256 keys)

        Returns:
            Dict mapping key -> {"size": int, "exists": bool} or {"exists": False}.
            Empty dict on connection error.
        """
        if not keys:
            return {}
        url = urljoin(self.base_url + "/", "/cache/files/batch")
        try:
            resp = await self._client.post(url, json={"keys": keys})
        except Exception as e:
            log.warning("Cache agent batch size check error on %s (%d keys): %s",
                        self.base_url, len(keys), e)
            return {}

        if resp.status_code != 200:
            log.warning("Cache agent batch returned status %d for %d keys: %s",
                        resp.status_code, len(keys), resp.text[:200])
            return {}

        data = resp.json()
        return data.get("results", {})

    async def close(self):
        await self._client.aclose()


async def get_file_size(agent_url: str, key: str) -> Optional[Dict[str, Any]]:
    """
    Module-level convenience function to get file size from a cache agent.

    Args:
        agent_url: Agent URL, e.g. "http://10.0.0.1:8082"
        key: Cache file basename (the SHA256 key)

    Returns:
        {"size": int, "exists": True} or {"exists": False} or None on error.
    """
    client = CacheAgentClient(agent_url)
    try:
        return await client.get_file_size(key)
    finally:
        await client.close()
