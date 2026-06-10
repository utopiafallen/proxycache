# config.py
# -*- coding: utf-8 -*-

"""
Single configuration source for proxycache:
- BACKENDS: [{"url": "..."}]
- WORDS_PER_BLOCK, LCP_TH
- PORT, REQUEST_TIMEOUT, MODEL_ID
"""

import os
import json
import logging

# Backends
try:
    BACKENDS = json.loads(os.getenv("BACKENDS", "[]"))
except Exception:
    BACKENDS = []

if not BACKENDS:
    BACKENDS = [{"url": "http://127.0.0.1:8000"}]

# Words per block for LCP
WORDS_PER_BLOCK = int(os.getenv("WORDS_PER_BLOCK", "100"))

# LCP threshold (0..1)
LCP_TH = float(os.getenv("LCP_TH", "0.2"))

# Meta dir
META_DIR = os.path.join(os.getcwd(), os.getenv("META_DIR", "kv_meta"))
os.makedirs(META_DIR, exist_ok=True)

# HTTP timeout
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))

# Model id
MODEL_ID = os.getenv("MODEL_ID", "llama.cpp")

# Backend mode: "llama-cpp" or "llama-swap"
BACKEND_MODE = os.getenv("BACKEND_MODE", "llama-cpp")

# Service port
PORT = int(os.getenv("PORT", "8081"))

# Cache cleanup settings
CACHE_DIR = os.getenv("CACHE_DIR", "")  # llama.cpp --slot-save-path directory
CACHE_MAX_AGE_HOURS = int(os.getenv("CACHE_MAX_AGE_HOURS", "168"))  # 7 days default
CACHE_MAX_SIZE_GB = float(os.getenv("CACHE_MAX_SIZE_GB", "25"))

# Default context length used when backend doesn't report n_ctx
DEFAULT_N_CTX = int(os.getenv("DEFAULT_N_CTX", "16384"))

# KV cache skip threshold (0..1) — skip restore if slot KV cache matches >= this
KV_CACHE_SKIP_THRESHOLD = float(os.getenv("KV_CACHE_SKIP_THRESHOLD", "0.9"))

# Cache save ratio threshold (0..1) — only save cache if restore candidate ratio < this
# When a cached prompt already matches well (>= threshold), no need to save a duplicate
CACHE_SAVE_RATIO_THRESHOLD = float(os.getenv("CACHE_SAVE_RATIO_THRESHOLD", "0.8"))


def should_save_cache(best_ratio: float, recompute_happened: bool) -> bool:
    """Decide whether to save a slot's cache to disk.

    Returns True (save) when:
    - Restore happened but ratio < threshold (new cache entry will be more useful)
    - Recompute happened (restore was partial/useless)

    Returns False (skip) when:
    - Ratio >= threshold, and no recompute (old cache still useful)
    """
    return recompute_happened or best_ratio <= CACHE_SAVE_RATIO_THRESHOLD

# Timeout for slot save/restore operations (seconds) — separate from REQUEST_TIMEOUT
# because chat completions can take minutes, but slot operations should fail fast
SLOT_TIMEOUT = float(os.getenv("SLOT_TIMEOUT", "30"))

# Cooldown between slot refresh attempts (seconds) — 300s success, 30s failure
REFRESH_COOLDOWN_SECONDS = int(os.getenv("REFRESH_COOLDOWN_SECONDS", "300"))

# Cache hit wait queue settings
CACHE_HIT_WAIT_EMA_MIN_TIMEOUT = float(os.getenv("CACHE_HIT_WAIT_EMA_MIN_TIMEOUT", "10"))
CACHE_HIT_WAIT_MAX_PENDING_REQS = int(os.getenv("CACHE_HIT_WAIT_MAX_PENDING_REQS", "3"))
CACHE_HIT_WAIT_EMA_ALPHA = float(os.getenv("CACHE_HIT_WAIT_EMA_ALPHA", "0.2"))
CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT = float(os.getenv("CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT", "30"))
CACHE_HIT_WAIT_EMA_MAX_TIMEOUT = float(os.getenv("CACHE_HIT_WAIT_EMA_MAX_TIMEOUT", "300"))

# Logs
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
