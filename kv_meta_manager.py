# kv_meta_manager.py

# -*- coding: utf-8 -*-

"""
KV Meta Manager: single interface for all kv-meta operations.

All code lives in this class — no module-level functions.
"""

import os
import json
import glob
import time
import asyncio
import logging
from typing import List, Dict, Optional, Tuple

import hashing as hs
from config import META_DIR
from backend_manager import backend_manager

log = logging.getLogger(__name__)


class KVMetaManager:
    """Manages kv-meta files: read, write, list, delete, reconcile."""

    def meta_file_path(self, key: str, backend_key: Optional[str] = None) -> str:
        """Full path to a meta file."""
        if backend_key:
            return os.path.join(META_DIR, hs.sanitize_backend_dir(backend_key), f"{key}{hs.META_SUFFIX}")
        return os.path.join(META_DIR, f"{key}{hs.META_SUFFIX}")

    def meta_dir(self, backend_key: str) -> str:
        """Path to a backend's meta directory."""
        return os.path.join(META_DIR, hs.sanitize_backend_dir(backend_key))

    def scan_all_meta(self, backend_key: Optional[str] = None) -> List[Dict]:
        """Scan meta files for a backend (or all backends) and return list of meta dicts."""
        if backend_key:
            search_dir = self.meta_dir(backend_key)
            if not os.path.isdir(search_dir):
                log.warn("Meta directory missing for backend '%s'", backend_key)
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
                    metas.append(json.load(fd))
            except Exception as e:
                log.warning("Failed to read meta file %s: %s", f, e)
        log.warn("Meta scan found %d entries", len(metas))
        return metas

    def read_meta(self, key: str, backend_id: str) -> Optional[Dict]:
        """Read and return the full meta dict for a key, or None on failure."""
        path = self.meta_file_path(key, backend_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def get_blocks(self, key: str, backend_id: str) -> Optional[List[str]]:
        """Return blocks list for a key, or None if not found."""
        meta = self.read_meta(key, backend_id)
        if meta:
            return meta.get("blocks")
        return None

    def list_keys(self, backend_id: str) -> List[str]:
        """Return list of cache keys for a backend (filenames without suffix)."""
        backend_path = self.meta_dir(backend_id)
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
        meta = {
            "key": key,
            "model_id": model_id,
            "backend": backend_id,
            "n_tokens": n_tokens,
            "wpb": wpb,
            "blocks": blocks,
            "cache_size": cache_size,
            "recompute_penalty": 0,
            "last_written": time.time(),
        }
        d = self.meta_dir(backend_id)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{key}{hs.META_SUFFIX}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        log.info("Saved cache for key %s (model_id: %s, backend: %s, %d blocks, %d bytes)",
                 key[:16], model_id, backend_id, len(blocks), cache_size)

    def delete_meta_file(self, key: str) -> bool:
        """Delete meta file for a key across all backends. Returns True if deleted."""
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

    def increment_recompute_penalty(self, key: str, backend_id: str) -> None:
        """Increment the recompute_penalty counter on a meta file."""
        path = self.meta_file_path(key, backend_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["recompute_penalty"] = meta.get("recompute_penalty", 0) + 1
            meta["last_updated_penalty"] = time.time()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            log.info("Incremented recompute_penalty for key %s to %d", key[:16], meta["recompute_penalty"])
        except Exception as e:
            log.warning("Failed to increment recompute_penalty for key %s: %s", key[:16], e)

    def find_restore_candidate(
        self,
        key: str,
        wpb: int,
        th: float,
        req_blocks: List[str],
        backend_id: str,
    ) -> Optional[Tuple[str, float]]:
        """Check if a specific key is a valid restore candidate."""
        path = self.meta_file_path(key, backend_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return None

        cand_blocks = meta.get("blocks") or []
        if int(meta.get("wpb") or 0) != wpb:
            return None

        lcp = hs.lcp_blocks(req_blocks, cand_blocks)
        ratio = lcp / max(1, len(req_blocks))

        if ratio >= th:
            return (key, ratio)
        return None

    def find_best_restore_candidate(
        self,
        req_blocks: List[str],
        wpb: int,
        th: float,
        model_id: str,
        backend_id: str,
    ) -> Optional[Tuple[str, float]]:
        """Find the best restore candidate among meta files for the CURRENT model+backend only."""
        metas = self.scan_all_meta(backend_id)
        best_key: Optional[str] = None
        best_ratio = 0.0
        best_score = 0.0

        for meta in metas:
            if meta.get("model_id") != model_id:
                continue
            if meta.get("backend") != backend_id:
                continue
            if int(meta.get("wpb") or 0) != wpb:
                continue

            cand_blocks = meta.get("blocks") or []
            lcp = hs.lcp_blocks(req_blocks, cand_blocks)
            ratio = lcp / max(1, len(req_blocks))
            penalty = meta.get("recompute_penalty", 0)
            score = ratio * max(0, 1 - 0.1 * penalty)

            if ratio >= th and score > best_score:
                best_score = score
                best_ratio = ratio
                best_key = meta.get("key")

        return (best_key, best_ratio) if best_key else None

    async def get_cache_size(self, key: str, backend_id: str) -> int:
        """Return cache size in bytes for a key, or 0 if not found."""
        meta = self.read_meta(key, backend_id)
        if meta:
            size = meta.get("cache_size", 0) or 0
            if size:
                return size

        # Fallback: query backend's cache store (agent or local filesystem)
        return await backend_manager.cache_get_size(backend_id, key)

    def get_last_used_time(self, key: str, backend_id: str) -> float:
        """Return last-used timestamp for a cache file."""
        path = self.meta_file_path(key, backend_id)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                for field in ("last_read", "last_written", "timestamp"):
                    if field in meta:
                        return meta[field]
            except (json.JSONDecodeError, Exception):
                pass

        # Fallback: use local filesystem mtime (agent doesn't provide mtime)
        return backend_manager.cache_get_mtime(backend_id, key)

    async def reconcile(
        self,
        backend_keys: Optional[List[str]] = None,
    ) -> int:
        """Reconcile meta files with cache store. Returns count deleted."""
        deleted = 0
        deleted_backends = set()

        if backend_keys:
            for backend_key in backend_keys:
                backend_dir = self.meta_dir(backend_key)
                if not os.path.isdir(backend_dir):
                    continue

                meta_files = sorted(glob.glob(os.path.join(backend_dir, "*" + hs.META_SUFFIX)))

                # First pass: read all meta files, remove corrupted ones, collect valid keys
                valid_entries = []  # (meta_path, basename, cachename)
                for meta_path in meta_files:
                    basename = os.path.basename(meta_path)
                    cachename = basename.removesuffix(hs.META_SUFFIX)

                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            json.load(f)
                    except (json.JSONDecodeError, Exception):
                        log.warning("Removed corrupted meta file: %s", basename)
                        try:
                            os.remove(meta_path)
                            deleted += 1
                            deleted_backends.add(backend_key)
                        except OSError:
                            pass
                        continue

                    valid_entries.append((meta_path, basename, cachename))

                # Second pass: check cache existence via backend_manager
                for meta_path, basename, cachename in valid_entries:
                    cache_exists = await backend_manager.cache_exists(backend_key, cachename)

                    if not cache_exists:
                        log.info("Removed orphan meta file (no matching cache): %s", basename)
                        try:
                            os.remove(meta_path)
                            deleted += 1
                            deleted_backends.add(backend_key)
                        except OSError:
                            pass

                if backend_key in deleted_backends:
                    try:
                        if not os.listdir(backend_dir):
                            os.rmdir(backend_dir)
                            log.info("Removed empty backend directory: %s", backend_key)
                    except OSError:
                        pass
        else:
            meta_files = sorted(glob.glob(os.path.join(META_DIR, "*" + hs.META_SUFFIX)))
            for meta_path in meta_files:
                basename = os.path.basename(meta_path)
                cachename = basename.removesuffix(hs.META_SUFFIX)

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

                # Without backend_keys, we can't determine the backend for cache existence checks
                # Skip cache existence check for standalone meta files
                log.info("Removed orphan meta file (no backend context): %s", basename)
                try:
                    os.remove(meta_path)
                    deleted += 1
                except OSError:
                    pass

        log.info("Finished reconciling meta files with llama cache directory")
        return deleted

    async def total_meta_size(self, backend_id: str) -> int:
        """Return total size of all cache files for a backend."""
        total = 0
        for key in self.list_keys(backend_id):
            total += await self.get_cache_size(key, backend_id, cache_dir)
        return total


# Singleton instance
kv_meta = KVMetaManager()
