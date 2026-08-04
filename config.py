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
    BACKENDS = [{"url": "http://127.0.0.1:8000", "cache_dir": "/tmp/llama-cache"}]

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


# Default context length used when backend doesn't report n_ctx
DEFAULT_N_CTX = int(os.getenv("DEFAULT_N_CTX", "16384"))

# KV cache skip threshold (0..1) — skip restore if slot KV cache matches >= this
KV_CACHE_SKIP_THRESHOLD = float(os.getenv("KV_CACHE_SKIP_THRESHOLD", "0.9"))

# Max block count difference ratio (0..1) for skip-restore — if the slot's tracked
# block count differs from the request by more than this fraction, don't skip.
# e.g. 0.1 means a 46-block request vs 841-block slot (94% diff) won't skip.
KV_CACHE_SKIP_MAX_BLOCK_DIFF_PCT = float(os.getenv("KV_CACHE_SKIP_MAX_BLOCK_DIFF_PCT", "0.1"))

# Cache save ratio threshold (0..1) — only save cache if restore candidate ratio < this
# When a cached prompt already matches well (>= threshold), no need to save a duplicate
CACHE_SAVE_RATIO_THRESHOLD = float(os.getenv("CACHE_SAVE_RATIO_THRESHOLD", "0.8"))

# Cache save context length threshold (0..1) — skip save if request tokens >= this
# fraction of backend's max context. Conversations near max context will likely be
# compacted soon, invalidating the saved KV cache prefix.
CACHE_SAVE_CTX_THRESHOLD = float(os.getenv("CACHE_SAVE_CTX_THRESHOLD", "0.7"))


def should_skip_save_heuristic(prompt_tokens: int, n_ctx: int, messages: list = None) -> bool:
    """Skip save for requests unlikely to produce reusable cache entries.

    Returns True (skip) when:
    - Request is >= CACHE_SAVE_CTX_THRESHOLD of backend's max context (impending compaction)
    - Request is a single user message (likely a one-off summarization, not conversation start)
      (only checked when messages is provided)
    """
    if n_ctx > 0 and isinstance(prompt_tokens, int) and prompt_tokens / n_ctx >= CACHE_SAVE_CTX_THRESHOLD:
        return True
    if messages is not None:
        roles = [m.get("role") for m in messages]
        if roles.count("user") <= 1 and roles.count("assistant") == 0:
            return True
    return False


def should_save_cache(best_ratio: float, recompute_happened: bool) -> bool:
    """Decide whether to save a slot's cache to disk.

    Returns True (save) when:
    - Restore happened but ratio < threshold (new cache entry will be more useful)
    - Recompute happened (restore was partial/useless)

    Returns False (skip) when:
    - Ratio >= threshold, and no recompute (old cache still useful)
    """
    return recompute_happened or best_ratio <= CACHE_SAVE_RATIO_THRESHOLD


# Keywords that strongly indicate summarization intent
_SUMMARIZATION_KEYWORDS = {
    "summarize", "summarise", "summary", "tl;dr", "tl; dr", "key points",
    "bullet points", "overview", "abstract", "digest", "condense",
    "extract", "highlights", "main points", "brief", "in short",
    "give me a summary", "sum it up", "wrap up", "recap",
}

# Patterns that suggest content is being pasted for processing
_PASTE_PATTERNS = (
    "<document>", "</document>", "<context>", "</context>",
    "<article>", "</article>", "<text>", "</text>",
    "<content>", "</content>", "<passage>", "</passage>",
    "<input>", "</input>", "<transcript>", "</transcript>",
)

# Phrases that introduce pasted content
_CONTENT_INTRO_PHRASES = (
    "here is a", "here's a", "here is the", "here's the",
    "below is", "below is a", "below is the",
    "attached is", "following is", "following text",
    "read this", "analyze this", "review this",
    "the following", "consider the following",
)


def classify_request(messages: list, request_json: dict = None) -> dict:
    """Classify a request as summarization-like or conversation-like.

    Returns a dict with:
      - score: float 0..1 (higher = more likely summarization)
      - signals: list of str (which signals fired)

    Heuristics (weights sum to ~1.0):
      - Single user message, no assistant (0.30)
      - Summarization keyword in final user/system message (0.25)
      - One message dominates token count >70% (0.20)
      - Prompt tokens > 4096 (0.10)
      - Structural delimiters / paste markers present (0.05)
      - Content introduction phrases present (0.05)
      - System prompt present (-0.10, slight conversation bias)
    """
    if not messages:
        return {"score": 0.0, "signals": []}

    signals = []
    score = 0.0

    roles = [m.get("role", "") for m in messages]
    n_user = roles.count("user")
    n_assistant = roles.count("assistant")
    n_system = roles.count("system")
    n_total = len(messages)

    # Get text content from messages for keyword/pattern matching
    def _get_text(msg):
        c = msg.get("content", "")
        if isinstance(c, str):
            return c.lower()
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "").lower()
        return ""

    # 1. Single user message, no assistant (0.30)
    if n_user == 1 and n_assistant == 0:
        score += 0.30
        signals.append("single_user_msg")

    # 2. Summarization keyword in final user message or system prompt (0.25)
    final_texts = []
    for m in messages:
        if m.get("role") in ("user", "system"):
            final_texts.append(_get_text(m))
    # Also check the last message specifically
    last_text = _get_text(messages[-1]) if messages else ""

    for text in (last_text,) + tuple(final_texts[:-1]):
        for kw in _SUMMARIZATION_KEYWORDS:
            if kw in text:
                score += 0.25
                signals.append(f"keyword:{kw}")
                break
        else:
            continue
        break

    # 3. One message dominates token count >70% (0.20)
    # Approximate token count by text length (good enough for heuristic)
    msg_lengths = [len(_get_text(m)) for m in messages]
    total_length = sum(msg_lengths)
    if total_length > 0:
        max_ratio = max(msg_lengths) / total_length
        if max_ratio > 0.7:
            score += 0.20
            signals.append(f"dominant_msg:{max_ratio:.0%}")

    # 4. Prompt tokens > 4096 (approximate from text length, ~4 chars/token)
    if total_length > 4096 * 4:
        score += 0.10
        signals.append(f"long_prompt:{total_length // 4}tok_est")

    # 5. Structural delimiters / paste markers (0.05)
    last_lower = last_text
    for pat in _PASTE_PATTERNS:
        if pat in last_lower:
            score += 0.05
            signals.append(f"delimiter:{pat}")
            break

    # 6. Content introduction phrases (0.05)
    for phrase in _CONTENT_INTRO_PHRASES:
        if phrase in last_lower:
            score += 0.05
            signals.append(f"intro:{phrase}")
            break

    # 7. System prompt present — slight negative bias (-0.10)
    # System prompts are common in both, but slightly more common in
    # structured conversations vs one-off summarization dumps
    if n_system > 0:
        score -= 0.10
        signals.append("system_prompt")

    return {
        "score": round(max(0.0, min(1.0, score)), 3),
        "signals": signals,
    }

# Timeout for slot save/restore operations (seconds) — separate from REQUEST_TIMEOUT
# because chat completions can take minutes, but slot operations should fail fast
SLOT_TIMEOUT = float(os.getenv("SLOT_TIMEOUT", "30"))

# Recreate the underlying httpx.AsyncClient after this many requests per backend
# to avoid connection pool degradation (stale/half-closed connections)
CLIENT_RECREATE_INTERVAL = int(os.getenv("CLIENT_RECREATE_INTERVAL", "50"))

# Cache hit wait queue settings
CACHE_HIT_WAIT_EMA_MIN_TIMEOUT = float(os.getenv("CACHE_HIT_WAIT_EMA_MIN_TIMEOUT", "10"))
CACHE_HIT_WAIT_MAX_PENDING_REQS = int(os.getenv("CACHE_HIT_WAIT_MAX_PENDING_REQS", "3"))
CACHE_HIT_WAIT_EMA_ALPHA = float(os.getenv("CACHE_HIT_WAIT_EMA_ALPHA", "0.2"))
CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT = float(os.getenv("CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT", "30"))
CACHE_HIT_WAIT_EMA_MAX_TIMEOUT = float(os.getenv("CACHE_HIT_WAIT_EMA_MAX_TIMEOUT", "300"))

# Metrics retention (single ring buffer for requests + diagnostic events)
METRICS_RETENTION = int(os.getenv("METRICS_RETENTION", "200"))

# Dashboard
DASHBOARD_ENABLED = os.getenv("DASHBOARD_ENABLED", "true").lower() in ("true", "1", "yes")

# Logs
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
