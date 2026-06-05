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

import hashlib
import logging
from typing import List, Dict

from config import WORDS_PER_BLOCK

META_SUFFIX = ".meta.json"

log = logging.getLogger(__name__)


def sanitize_backend_dir(backend_key: str) -> str:
    """Sanitize backend key for use as a filesystem directory name."""
    return backend_key.replace(":", "-")


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
    log.debug("Raw prefix: %d characters", len(text))
    return text


def block_hashes_from_tokens(token_ids: List[int], wpb: int = WORDS_PER_BLOCK) -> List[str]:
    hashes: List[str] = []
    for i in range(0, len(token_ids), wpb):
        chunk = token_ids[i:i + wpb]
        h = hashlib.sha256(",".join(str(t) for t in chunk).encode("utf-8")).hexdigest()
        hashes.append(h)
    log.debug("Block hashes: %d blocks, %d tokens per block", len(hashes), wpb)
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


def meta_key(canonical_name: str, token_ids: List[int]) -> str:
    """sha256(canonical_name + '\n' + ','.join(token_ids))"""
    return prefix_key_sha256(canonical_name + "\n" + ",".join(str(t) for t in token_ids))
