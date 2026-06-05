# kv_meta_manager.py

# -*- coding: utf-8 -*-

"""
KV Meta Manager: single interface for all kv-meta operations.

All code should go through this manager instead of accessing hashing.py functions
directly. This keeps kv-meta internals opaque and makes it easy to change
storage backends or add caching layers.
"""

import os
import json
import glob
import time
import asyncio
import logging
from typing import List, Dict, Optional, Tuple

import hashing as hs
from config import META_DIR, CACHE_DIR
from backend_manager import backend_manager

log = logging.getLogger(__name__)


def meta_file_path(key: str, backend_key: Optional[str] = None) -> str:
    """Full path to a meta file."""
    if backend_key:
        return os.path.join(META_DIR, hs.sanitize_backend_dir(backend_key), f"{key}{hs.META_SUFFIX}")
    return os.path.join(META_DIR, f"{key}{hs.META_SUFFIX}")


def meta_dir(backend_key: str) -> str:
    """Path to a backend's meta directory."""
    return os.path.join(META_DIR, hs.sanitize_backend_dir(backend_key))


def get_meta_blocks(key: str, backend_key: Optional[str] = None) -> Optional[List[str]]:
    """Read a meta file and return its blocks list, or None on failure."""
    path = meta_file_path(key, backend_key)
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("blocks")
    except Exception:
        return None


def delete_meta_file(key: str) -> bool:
    """Delete a meta file by scanning all backend directories. Returns True if deleted."""
    if not os.path.isdir(META_DIR):
        return False
    for entry in os.listdir(META_DIR):
        candidate = os.path.join(META_DIR, entry, f"{key}{hs.META_SUFFIX}")
        if os.path.exists(candidate):
            try:
                os.remove(candidate)
                return True
            except OSError:
                return False
    return False


def scan_all_meta(backend_key: Optional[str] = None) -> List[Dict]:
    if backend_key:
        search_dir = meta_dir(backend_key)
        if not os.path.isdir(search_dir):
            log.debug("Meta directory missing for backend '%s'", backend_key)
            return []
        files = sorted(
            glob.glob(os.path.join(search_dir, "*" + hs.META_SUFFIX)),
            key=os.path.getmtime,
            reverse=True,
        )
    else:
        files = sorted(
            glob.glob(os.path.join(META_DIR, "*" + hs.META_SUFFIX)),
            key=os.path.getmtime,
            reverse=True,
        )
    metas: List[Dict] = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fd:
                meta = json.load(fd)
                metas.append(meta)
        except Exception as e:
            log.warning("Failed to read meta file %s: %s", f, e)
    log.debug("Meta scan found %d entries", len(metas))
    return metas


def find_restore_candidate(
    meta_key_str: str,
    wpb: int,
    th: float,
    req_blocks: List[str],
    backend_key: str,
) -> Optional[Tuple[str, float]]:
    """Read meta file at {META_DIR}/{backend_key}/{meta_key_str}.meta.json,
    compute LCP ratio, return (key, ratio) if >= threshold."""
    path = meta_file_path(meta_key_str, backend_key)
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None

    cand_blocks = meta.get("blocks") or []
    if int(meta.get("wpb") or 0) != wpb:
        return None

    lcp = hs.lcp_blocks(req_blocks, cand_blocks)
    denom = max(1, len(req_blocks))
    ratio = lcp / denom

    if ratio >= th:
        return (meta_key_str, ratio)
    return None


def find_best_restore_candidate(
    req_blocks: List[str],
    wpb: int,
    th: float,
    model_id: str,
    backend_key: str,
) -> Optional[Tuple[str, float]]:
    """
    Find the best restore candidate among meta files for the CURRENT model+backend only.

    Filters by:
    - meta["model_id"] == model_id
    - meta["backend"] == backend_key
    - meta["wpb"] == wpb
    """
    metas = scan_all_meta(backend_key)
    best_key: Optional[str] = None
    best_ratio = 0.0

    for meta in metas:
        if meta.get("model_id") != model_id:
            continue
        if meta.get("backend") != backend_key:
            continue
        if int(meta.get("wpb") or 0) != wpb:
            continue

        cand_blocks = meta.get("blocks") or []
        lcp = hs.lcp_blocks(req_blocks, cand_blocks)
        denom = max(1, min(len(req_blocks), len(cand_blocks)))
        ratio = lcp / denom

        if ratio >= th and ratio > best_ratio:
            best_ratio = ratio
            best_key = meta.get("key")

    return (best_key, best_ratio) if best_key else None


def write_meta(
    key: str,
    n_tokens: int,
    blocks: List[str],
    wpb: int,
    model_id: str,
    backend_key: str,
    cache_size: int = 0,
) -> None:
    """
    Write/overwrite meta file for key, bound to a specific model and backend.
    """
    meta = {
        "key": key,
        "model_id": model_id,
        "backend": backend_key,
        "n_tokens": n_tokens,
        "wpb": wpb,
        "blocks": blocks,
        "cache_size": cache_size,
        "last_written": time.time(),
    }
    d = meta_dir(backend_key)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{key}{hs.META_SUFFIX}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info("Saved cache for key %s (model_id: %s, backend: %s, %d blocks, %d bytes)", key[:16], model_id, backend_key, len(blocks), cache_size)


async def reconcile_meta(meta_base: str, cache_dir: str, backend_keys: Optional[List[str]] = None, backend_agents: Optional[Dict[str, str]] = None) -> int:
    """
    Scans meta files organized by backend and removes corrupted ones or orphans.
    Uses cache agent API for remote file size lookups when available.
    Returns the count of files deleted.
    """
    deleted = 0
    deleted_backends = set()

    if backend_keys:
        for backend_key in backend_keys:
            backend_dir = meta_dir(backend_key)
            if not os.path.isdir(backend_dir):
                continue

            agent_url = None
            if backend_agents and backend_key in backend_agents:
                agent_url = backend_agents[backend_key]

            meta_files = sorted(glob.glob(os.path.join(backend_dir, "*" + hs.META_SUFFIX)))

            for meta_path in meta_files:
                basename = os.path.basename(meta_path)
                cachename = basename.removesuffix(hs.META_SUFFIX)

                # Check for corrupted meta files and load meta data
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except (json.JSONDecodeError, Exception) as e:
                    log.warning("Removed corrupted meta file: %s", basename)
                    try:
                        os.remove(meta_path)
                        deleted += 1
                        deleted_backends.add(backend_key)
                    except OSError:
                        pass
                    continue

                # Check for orphaned meta files (no matching cache on disk)
                cache_exists = False
                meta_cache_size = meta.get("cache_size", 0)

                if meta_cache_size and meta_cache_size > 0:
                    cache_exists = True
                elif agent_url:
                    try:
                        from cache_agent_client import get_file_size
                        result = await get_file_size(agent_url, cachename)
                        if result and result.get("exists", False):
                            cache_exists = True
                    except Exception as e:
                        log.warning("Failed to check cache size via agent on backend '%s' for key %s: %s", backend_key, cachename, e)
                elif cache_dir and os.path.isdir(cache_dir):
                    cache_path = os.path.join(cache_dir, cachename)
                    if os.path.exists(cache_path):
                        cache_exists = True

                if not cache_exists:
                    log.info("Removed orphan meta file (no matching cache): %s", basename)
                    try:
                        os.remove(meta_path)
                        deleted += 1
                        deleted_backends.add(backend_key)
                    except OSError:
                        pass

            # Remove empty backend directories
            if backend_key in deleted_backends:
                try:
                    if not os.listdir(backend_dir):
                        os.rmdir(backend_dir)
                        log.info("Removed empty backend directory: %s", backend_key)
                except OSError:
                    pass
    else:
        # Legacy: scan flat meta_base
        meta_files = sorted(glob.glob(os.path.join(meta_base, "*" + hs.META_SUFFIX)))

        for meta_path in meta_files:
            basename = os.path.basename(meta_path)
            cachename = basename.removesuffix(hs.META_SUFFIX)

            # Check for corrupted meta files
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                log.warning("Removed corrupted meta file: %s", basename)
                try:
                    os.remove(meta_path)
                    deleted += 1
                except OSError:
                    pass
                continue

            # Check for orphaned meta files (no matching cache on disk)
            if cache_dir and os.path.isdir(cache_dir):
                cache_path = os.path.join(cache_dir, cachename)
                if not os.path.exists(cache_path):
                    log.info("Removed orphan meta file (no matching cache): %s", basename)
                    try:
                        os.remove(meta_path)
                        deleted += 1
                    except OSError:
                        pass

    log.info("Finished reconciling meta files with llama cache directory")
    return deleted


def _get_last_used_time(basename: str, meta_base: str, cache_dir: str, backend_key: Optional[str] = None) -> float:
    """
    Determines the last-used timestamp for a cache file.
    Priority: last_read -> last_written -> timestamp -> mtime (filesystem fallback).
    """
    if backend_key:
        meta_path = meta_file_path(basename, backend_key)
    else:
        meta_path = os.path.join(meta_base, f"{basename}{hs.META_SUFFIX}")
    
    # Try to load from meta file
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            # Priority order for last-used timestamp
            for field in ("last_read", "last_written", "timestamp"):
                if field in meta:
                    return meta[field]
        except (json.JSONDecodeError, Exception):
            pass  # Corrupted meta, fall through to mtime
    
    # Fallback to filesystem mtime
    if cache_dir:
        cache_path = os.path.join(cache_dir, basename)
        if os.path.exists(cache_path):
            return os.path.getmtime(cache_path)
    
    return time.time()  # Ultimate fallback


class KVMetaManager:
    """Manages kv-meta files: read, write, list, delete, reconcile."""

    async def get_cache_size(self, key: str, backend_id: str, cache_dir: str = CACHE_DIR) -> int:
        """Return cache size in bytes for a key, or 0 if not found.
        Falls back to cache-agent API, then local filesystem stat.
        """
        meta = self.read_meta(key, backend_id)
        if meta:
            size = meta.get("cache_size", 0) or 0
            if size:
                return size

        # Fallback: query cache-agent for remote size
        try:
            agent = backend_manager.get_agent(backend_id)
        except KeyError:
            agent = None
        if agent:
            try:
                from cache_agent_client import get_file_size
                result = await get_file_size(agent.base_url, key)
                if result and result.get("exists", False):
                    return result.get("size", 0)
            except Exception as e:
                log.debug("Cache-agent size lookup failed for '%s' on '%s': %s", key, backend_id, e)

        # Fallback: local filesystem stat
        if cache_dir and os.path.isdir(cache_dir):
            cache_path = os.path.join(cache_dir, key)
            if os.path.exists(cache_path):
                return os.stat(cache_path).st_size

        return 0

    def get_last_used_time(self, key: str, backend_id: str, cache_dir: str = CACHE_DIR) -> float:
        """Return last-used timestamp for a cache file."""
        return _get_last_used_time(key, META_DIR, cache_dir, backend_id)

    def get_blocks(self, key: str, backend_id: str) -> Optional[List[str]]:
        """Return blocks list for a key, or None if not found."""
        meta = self.read_meta(key, backend_id)
        if meta:
            return meta.get("blocks")
        return None

    def read_meta(self, key: str, backend_id: str) -> Optional[Dict]:
        """Read and return the full meta dict for a key, or None on failure."""
        path = meta_file_path(key, backend_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def list_keys(self, backend_id: str) -> List[str]:
        """Return list of cache keys for a backend (filenames without suffix)."""
        backend_path = meta_dir(backend_id)
        if not os.path.isdir(backend_path):
            return []
        keys = []
        for f in os.listdir(backend_path):
            if f.endswith(hs.META_SUFFIX):
                keys.append(f.removesuffix(hs.META_SUFFIX))
        return sorted(keys)

    def write_meta(
        self,
        key: str,
        n_tokens: int,
        blocks: List[str],
        wpb: int,
        model_id: str,
        backend_id: str,
        cache_size: int = 0,
    ) -> None:
        """Write/overwrite meta file for a key."""
        write_meta(key, n_tokens, blocks, wpb, model_id, backend_id, cache_size)

    def delete_meta_file(self, key: str) -> bool:
        """Delete meta file for a key across all backends. Returns True if deleted."""
        return delete_meta_file(key)

    def find_restore_candidate(
        self,
        key: str,
        req_blocks: List[str],
        wpb: int,
        th: float,
        backend_id: str,
    ) -> Optional[Tuple[str, float]]:
        """Check if a specific key is a valid restore candidate."""
        return find_restore_candidate(key, wpb, th, req_blocks, backend_id)

    def find_best_restore_candidate(
        self,
        req_blocks: List[str],
        wpb: int,
        th: float,
        model_id: str,
        backend_id: str,
    ) -> Optional[Tuple[str, float]]:
        """Find the best restore candidate for a model+backend."""
        return find_best_restore_candidate(req_blocks, wpb, th, model_id, backend_id)

    async def reconcile(
        self,
        cache_dir: str = CACHE_DIR,
        backend_keys: Optional[List[str]] = None,
        backend_agents: Optional[Dict[str, str]] = None,
    ) -> int:
        """Reconcile meta files with cache directory. Returns count deleted."""
        return await reconcile_meta(META_DIR, cache_dir, backend_keys, backend_agents)

    async def total_meta_size(self, backend_id: str, cache_dir: str = CACHE_DIR) -> int:
        """Return total size of all cache files for a backend."""
        total = 0
        for key in self.list_keys(backend_id):
            total += await self.get_cache_size(key, backend_id, cache_dir)
        return total


# Singleton instance
kv_meta = KVMetaManager()
