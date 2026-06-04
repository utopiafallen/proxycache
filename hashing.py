# hashing.py

# -*- coding: utf-8 -*-

"""
Raw hashing: raw_prefix strips message roles, concatenates content only, separated by double newlines.

Blocks of 100 words, LCP computed over full SHA256 hashes.
Key = sha256(model_id + "\\n" + raw_prefix), i.e. model is included in the key.

Meta files contain:
- key
- model_id
- prefix_len
- wpb
- blocks
- timestamp
"""

import os
import json
import hashlib
import re
import time
import glob
import logging
from typing import List, Dict, Optional, Tuple

from config import META_DIR, WORDS_PER_BLOCK

META_SUFFIX = ".meta.json"

log = logging.getLogger(__name__)


def sanitize_backend_dir(backend_key: str) -> str:
    """Sanitize backend key for use as a filesystem directory name."""
    return backend_key.replace(":", "-")


def meta_file_path(key: str, backend_key: Optional[str] = None) -> str:
    """Full path to a meta file."""
    if backend_key:
        return os.path.join(META_DIR, sanitize_backend_dir(backend_key), f"{key}{META_SUFFIX}")
    return os.path.join(META_DIR, f"{key}{META_SUFFIX}")


def meta_dir(backend_key: str) -> str:
    """Path to a backend's meta directory."""
    return os.path.join(META_DIR, sanitize_backend_dir(backend_key))


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
        candidate = os.path.join(META_DIR, entry, f"{key}{META_SUFFIX}")
        if os.path.exists(candidate):
            try:
                os.remove(candidate)
                return True
            except OSError:
                return False
    return False


def raw_prefix(messages: List[Dict]) -> str:
    parts = []
    for msg in messages or []:
        content = msg.get("content", "")
        if isinstance(content, str):
            content = content.strip()
        else:
            content = str(content).strip()
        if content:
            parts.append(content)
    text = "\n\n".join(parts).strip()
    log.debug("raw_prefix len_chars=%d", len(text))
    return text


def words_from_text(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def block_hashes_from_text(text: str, wpb: int = WORDS_PER_BLOCK) -> List[str]:
    words = words_from_text(text)
    hashes: List[str] = []
    for i in range(0, len(words), wpb):
        block = " ".join(words[i:i + wpb])
        h = hashlib.sha256(block.encode("utf-8")).hexdigest()
        hashes.append(h)
    log.debug("block_hashes n_blocks=%d wpb=%d", len(hashes), wpb)
    return hashes


def lcp_blocks(blocks1: List[str], blocks2: List[str]) -> int:
    n = min(len(blocks1), len(blocks2))
    i = 0
    while i < n and blocks1[i] == blocks2[i]:
        i += 1
    return i


def prefix_key_sha256(text: str) -> str:
    """
    Basic SHA256 wrapper; for cache we pass model_id + "\\n" + raw_prefix.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def meta_key(canonical_name: str, prefix: str) -> str:
    """sha256(canonical_name + '\n' + prefix)"""
    return prefix_key_sha256(canonical_name + "\n" + prefix)


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

    lcp = lcp_blocks(req_blocks, cand_blocks)
    denom = max(1, min(len(req_blocks), len(cand_blocks)))
    ratio = lcp / denom

    if ratio >= th:
        return (meta_key_str, ratio)
    return None


def scan_all_meta(backend_key: Optional[str] = None) -> List[Dict]:
    if backend_key:
        search_dir = meta_dir(backend_key)
        if not os.path.isdir(search_dir):
            log.debug("scan_meta backend_dir_missing backend=%s", backend_key)
            return []
        files = sorted(
            glob.glob(os.path.join(search_dir, "*" + META_SUFFIX)),
            key=os.path.getmtime,
            reverse=True,
        )
    else:
        files = sorted(
            glob.glob(os.path.join(META_DIR, "*" + META_SUFFIX)),
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
            log.warning("scan_meta_fail %s: %s", f, e)
    log.debug("scan_meta n_found=%d", len(metas))
    return metas


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
        lcp = lcp_blocks(req_blocks, cand_blocks)
        denom = max(1, min(len(req_blocks), len(cand_blocks)))
        ratio = lcp / denom

        if ratio >= th and ratio > best_ratio:
            best_ratio = ratio
            best_key = meta.get("key")

    return (best_key, best_ratio) if best_key else None


def human_readable_time(timestamp: float) -> str:
    """Converts a Unix timestamp to a human-readable format."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def write_meta(
    key: str,
    prefix_text: str,
    blocks: List[str],
    wpb: int,
    model_id: str,
    backend_key: str,
) -> None:
    """
    Write/overwrite meta file for key, bound to a specific model and backend.
    """
    meta = {
        "key": key,
        "model_id": model_id,
        "backend": backend_key,
        "prefix_len": len(prefix_text),
        "wpb": wpb,
        "blocks": blocks,
        "last_written": time.time(),
    }
    d = meta_dir(backend_key)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{key}{META_SUFFIX}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info("Saved cache for key %s (model: %s, backend: %s, %d blocks)", key[:16], model_id, backend_key, len(blocks))


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

            meta_files = sorted(glob.glob(os.path.join(backend_dir, "*" + META_SUFFIX)))

            for meta_path in meta_files:
                basename = os.path.basename(meta_path)
                cachename = basename.removesuffix(META_SUFFIX)

                # Check for corrupted meta files
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        json.load(f)
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
                if agent_url:
                    try:
                        from cache_agent_client import get_file_size
                        result = await get_file_size(agent_url, cachename)
                        if result and result.get("exists", False):
                            cache_exists = True
                    except Exception as e:
                        log.warning("cache_agent_size_fail backend=%s key=%s: %s", backend_key, cachename, e)
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
        meta_files = sorted(glob.glob(os.path.join(meta_base, "*" + META_SUFFIX)))

        for meta_path in meta_files:
            basename = os.path.basename(meta_path)
            cachename = basename.removesuffix(META_SUFFIX)

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

    log.info("Finished reconciling meta state with llama cache dir state")
    return deleted


def _get_last_used_time(basename: str, meta_base: str, cache_dir: str, backend_key: Optional[str] = None) -> float:
    """
    Determines the last-used timestamp for a cache file.
    Priority: last_read -> last_written -> timestamp -> mtime (filesystem fallback).
    """
    if backend_key:
        meta_path = meta_file_path(basename, backend_key)
    else:
        meta_path = os.path.join(meta_base, f"{basename}{META_SUFFIX}")
    
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



