# metrics.py

# -*- coding: utf-8 -*-

"""
In-memory metrics collector for proxycache.

Tracks per-request statistics in a ring buffer and aggregates counters
for cache performance, latency percentiles, and slot utilization.

Two-phase recording:
  1. Request arrival: record() with request_id, status="incomplete"
  2. Request completion: record() with same request_id, status="complete" + metrics
     Updates the existing record in-place (no duplicate entries).

Incomplete requests (status="incomplete") are visible in the dashboard
and excluded from performance metric calculations.
"""

import os
import time
import json
import logging
import threading
from collections import deque
from typing import Dict, List, Optional, Any

from config import METRICS_RETENTION

log = logging.getLogger(__name__)


class MetricsCollector:
    """Thread-safe in-memory metrics collector.

    Ring buffer of recent requests (default 100). Counters are incremented
    on each record(). Performance metrics are computed on-demand from the
    ring buffer.

    Supports two-phase recording: an initial record on request arrival
    (status="incomplete") followed by a completion record (status="complete")
    that updates the same entry in-place via request_id matching.
    """

    def __init__(self, retention: int = 100):
        self._retention = retention
        self._lock = threading.Lock()

        # Ring buffer of request records
        self._requests: deque[Dict[str, Any]] = deque(maxlen=retention)

        # request_id -> index in _requests (for in-place updates)
        self._by_id: Dict[str, int] = {}

        # Counters
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_recomputes = 0
        self._cache_saved = 0
        self._cache_save_skipped = 0
        self._restore_successes = 0
        self._restore_failures = 0

        # Per-model counters
        self._model_counters: Dict[str, Dict[str, int]] = {}

        # Per-backend counters
        self._backend_counters: Dict[str, Dict[str, int]] = {}

        # Start time for uptime calculation
        self._start_time = time.time()

    def record(self, ctx: Dict[str, Any]) -> None:
        """Record or update a request.

        Two-phase usage:
          1. Arrival: record({"request_id": "...", "model": "...", "stream": True,
                              "status": "incomplete"})
          2. Completion: record({"request_id": "...", "model": "...", "latency_ms": ...,
                                 "cache_hit": ..., "status": "complete", ...})
                            -> updates the existing entry in-place

        Args:
            ctx: Dict with keys:
                - request_id: unique ID for matching arrival/completion records
                - t0: start timestamp
                - request_json: the full request body
                - model: model name
                - backend: backend key
                - slot_id: slot ID
                - cache_hit: bool
                - restored: bool or None
                - recompute: bool
                - saved: bool or None
                - latency_ms: float
                - n_tokens: int (optional)
                - cache_size_bytes: int (optional)
                - prompt_preview: str (optional)
                - status: "incomplete" (arrival) or "complete" (completion)
                - stream: bool (optional)
        """
        request_id = ctx.get("request_id")
        model = ctx.get("model", "unknown")
        backend = ctx.get("backend", "unknown")
        cache_hit = bool(ctx.get("cache_hit", False))
        restored = ctx.get("restored")
        recompute = bool(ctx.get("recompute", False))
        saved = ctx.get("saved")
        latency_ms = ctx.get("latency_ms", 0)
        status = ctx.get("status", "complete")
        is_complete = status == "complete"

        # Extract prompt_preview from request_json if not provided
        prompt_preview = ctx.get("prompt_preview", "")
        if not prompt_preview:
            request_json = ctx.get("request_json", {})
            if request_json:
                messages = request_json.get("messages", [])
                for msg in reversed(messages):
                    if msg.get("role") in ("user", "assistant"):
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            prompt_preview = content[:200]
                        elif isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "text":
                                    prompt_preview = c["text"][:200]
                        break

        # Build request record from ctx, overriding with computed values
        record = {
            "timestamp": time.time(),
            "model": model,
            "backend": backend,
            "cache_hit": cache_hit,
            "restored": restored,
            "recompute": recompute,
            "saved": saved,
            "latency_ms": latency_ms,
            "status": status,
        }
        record.update(ctx)
        record["slot_id"] = ctx.get("slot_id", -1)
        record["prompt_preview"] = prompt_preview

        with self._lock:
            if request_id and request_id in self._by_id:
                # Update existing record in-place
                idx = self._by_id[request_id]
                old_record = self._requests[idx]
                old_prompt_preview = old_record.get("prompt_preview", "")
                old_timestamp = old_record.get("timestamp")
                old_record.update(record)
                # Preserve arrival timestamp
                if old_timestamp:
                    old_record["timestamp"] = old_timestamp
                # Preserve fields that may have been set on arrival but not in completion
                if "full_request_json" in old_record:
                    old_record["full_request_json"] = ctx.get("request_json", old_record.get("full_request_json", {}))
                # Restore prompt_preview from arrival record if completion didn't provide one
                if not prompt_preview and old_prompt_preview:
                    old_record["prompt_preview"] = old_prompt_preview
                if is_complete:
                    old_record["status"] = "complete"
                # Increment counters only on completion
                if is_complete:
                    self._total_requests += 1
                    if cache_hit:
                        self._cache_hits += 1
                        if recompute:
                            self._cache_recomputes += 1
                    else:
                        self._cache_misses += 1
                    if saved is True:
                        self._cache_saved += 1
                    elif saved is False:
                        self._cache_save_skipped += 1
                    if restored is True:
                        self._restore_successes += 1
                    elif restored is False:
                        self._restore_failures += 1
                    # Per-model counters
                    if model not in self._model_counters:
                        self._model_counters[model] = {
                            "total": 0, "hits": 0, "misses": 0,
                            "recomputes": 0, "saved": 0, "save_skipped": 0,
                        }
                    mc = self._model_counters[model]
                    mc["total"] += 1
                    if cache_hit:
                        mc["hits"] += 1
                        if recompute:
                            mc["recomputes"] += 1
                    else:
                        mc["misses"] += 1
                    if saved is True:
                        mc["saved"] += 1
                    elif saved is False:
                        mc["save_skipped"] += 1
                    # Per-backend counters
                    if backend not in self._backend_counters:
                        self._backend_counters[backend] = {
                            "total": 0, "hits": 0, "misses": 0,
                            "recomputes": 0, "saved": 0, "save_skipped": 0,
                        }
                    bc = self._backend_counters[backend]
                    bc["total"] += 1
                    if cache_hit:
                        bc["hits"] += 1
                        if recompute:
                            bc["recomputes"] += 1
                    else:
                        bc["misses"] += 1
                    if saved is True:
                        bc["saved"] += 1
                    elif saved is False:
                        bc["save_skipped"] += 1
            else:
                # New request — append to ring buffer (deque auto-evicts oldest if full)
                self._requests.append(record)

                # Rebuild _by_id indices after every append since deque auto-evicts
                # without notifying us, leaving stale indices
                self._by_id = {}
                for i, r in enumerate(self._requests):
                    rid = r.get("request_id")
                    if rid:
                        self._by_id[rid] = i

                # Increment counters only on completion
                if is_complete:
                    self._total_requests += 1
                    if cache_hit:
                        self._cache_hits += 1
                        if recompute:
                            self._cache_recomputes += 1
                    else:
                        self._cache_misses += 1
                    if saved is True:
                        self._cache_saved += 1
                    elif saved is False:
                        self._cache_save_skipped += 1
                    if restored is True:
                        self._restore_successes += 1
                    elif restored is False:
                        self._restore_failures += 1
                    # Per-model counters
                    if model not in self._model_counters:
                        self._model_counters[model] = {
                            "total": 0, "hits": 0, "misses": 0,
                            "recomputes": 0, "saved": 0, "save_skipped": 0,
                        }
                    mc = self._model_counters[model]
                    mc["total"] += 1
                    if cache_hit:
                        mc["hits"] += 1
                        if recompute:
                            mc["recomputes"] += 1
                    else:
                        mc["misses"] += 1
                    if saved is True:
                        mc["saved"] += 1
                    elif saved is False:
                        mc["save_skipped"] += 1
                    # Per-backend counters
                    if backend not in self._backend_counters:
                        self._backend_counters[backend] = {
                            "total": 0, "hits": 0, "misses": 0,
                            "recomputes": 0, "saved": 0, "save_skipped": 0,
                        }
                    bc = self._backend_counters[backend]
                    bc["total"] += 1
                    if cache_hit:
                        bc["hits"] += 1
                        if recompute:
                            bc["recomputes"] += 1
                    else:
                        bc["misses"] += 1
                    if saved is True:
                        bc["saved"] += 1
                    elif saved is False:
                        bc["save_skipped"] += 1

            # Add full_request_json to the record (for last ~20 entries)
            max_full = max(1, self._retention // 5)
            for i, r in enumerate(self._requests):
                if i >= len(self._requests) - max_full:
                    r["full_request_json"] = ctx.get("request_json", {})
                else:
                    r.pop("full_request_json", None)

    def get_requests(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get recent requests with full JSON payload."""
        with self._lock:
            requests = list(self._requests)
        total = len(requests)
        sliced = requests[offset:offset + limit]
        return sliced

    def get_total_count(self) -> int:
        """Return the total number of entries in the ring buffer."""
        with self._lock:
            return len(self._requests)

    def get_requests_summary(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get recent requests without full JSON payload."""
        with self._lock:
            requests = list(self._requests)
        total = len(requests)
        sliced = requests[offset:offset + limit]
        return [{"timestamp": r["timestamp"], "model": r["model"], "backend": r["backend"],
                  "slot_id": r["slot_id"], "cache_hit": r["cache_hit"],
                  "restored": r["restored"], "recompute": r["recompute"],
                  "saved": r["saved"], "latency_ms": r["latency_ms"],
                  "n_tokens": r.get("n_tokens"), "cache_size_bytes": r.get("cache_size_bytes"),
                  "prompt_preview": r["prompt_preview"],
                  "routing_reason": r.get("routing_reason"),
                  "status": r.get("status", "incomplete"),
                  "request_id": r.get("request_id")} for r in sliced]

    def get_performance(self, model: str = None, backend: str = None) -> Dict[str, Any]:
        """Compute performance metrics from the ring buffer.

        Only uses requests with status="complete".

        Args:
            model: Filter by model name (optional)
            backend: Filter by backend key (optional)

        Returns:
            Dict with cache hit rate, mispredict rate, save rate, latency percentiles.
        """
        with self._lock:
            if model or backend:
                requests = [r for r in self._requests
                            if (not model or r["model"] == model)
                            and (not backend or r["backend"] == backend)
                            and r.get("status") == "complete"]
                counters = self._get_filtered_counters(model, backend)
            else:
                requests = [r for r in self._requests if r.get("status") == "complete"]
                counters = self._get_global_counters()

        total = counters["total"]
        if total == 0:
            return self._empty_performance()

        hits = counters["hits"]
        misses = counters["misses"]
        recomputes = counters["recomputes"]
        saved = counters["saved"]
        save_skipped = counters["save_skipped"]

        hit_rate = hits / total if total > 0 else 0
        mispredict_rate = recomputes / hits if hits > 0 else 0
        utility_rate = (hits - recomputes) / total if total > 0 else 0
        total_requests_for_save = hits + misses
        save_rate = saved / total_requests_for_save if total_requests_for_save > 0 else 0
        save_skip_rate = save_skipped / total_requests_for_save if total_requests_for_save > 0 else 0

        # Restore success rate
        restore_successes = 0
        restore_failures = 0
        for r in requests:
            if r["restored"] is True:
                restore_successes += 1
            elif r["restored"] is False:
                restore_failures += 1
        restore_rate = restore_successes / (restore_successes + restore_failures) if (restore_successes + restore_failures) > 0 else 0

        # Latency percentiles
        latencies = sorted([r["latency_ms"] for r in requests if r["latency_ms"] > 0])
        latency_stats = self._compute_percentiles(latencies)

        return {
            "total_requests": total,
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_recomputes": recomputes,
            "cache_saved": saved,
            "cache_save_skipped": save_skipped,
            "cache_hit_rate": round(hit_rate, 4),
            "cache_mispredict_rate": round(mispredict_rate, 4),
            "cache_utility_rate": round(utility_rate, 4),
            "save_rate": round(save_rate, 4),
            "save_skip_rate": round(save_skip_rate, 4),
            "restore_success_rate": round(restore_rate, 4),
            "latency": latency_stats,
        }

    def _get_global_counters(self) -> Dict[str, int]:
        return {
            "total": self._total_requests,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "recomputes": self._cache_recomputes,
            "saved": self._cache_saved,
            "save_skipped": self._cache_save_skipped,
        }

    def _get_filtered_counters(self, model: str = None, backend: str = None) -> Dict[str, int]:
        result = {"total": 0, "hits": 0, "misses": 0, "recomputes": 0, "saved": 0, "save_skipped": 0}
        counters = self._model_counters if model else self._backend_counters
        if model:
            c = counters.get(model)
            if c:
                result.update(c)
        elif backend:
            c = counters.get(backend)
            if c:
                result.update(c)
        return result

    def _empty_performance(self) -> Dict[str, Any]:
        return {
            "total_requests": 0, "cache_hits": 0, "cache_misses": 0,
            "cache_recomputes": 0, "cache_saved": 0, "cache_save_skipped": 0,
            "cache_hit_rate": 0, "cache_mispredict_rate": 0, "cache_utility_rate": 0,
            "save_rate": 0, "save_skip_rate": 0, "restore_success_rate": 0,
            "latency": {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0},
        }

    @staticmethod
    def _compute_percentiles(latencies: List[float]) -> Dict[str, float]:
        if not latencies:
            return {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0}
        n = len(latencies)
        avg = sum(latencies) / n
        p50 = latencies[int(n * 0.50)] if n > 0 else 0
        p95 = latencies[min(int(n * 0.95), n - 1)] if n > 0 else 0
        p99 = latencies[min(int(n * 0.99), n - 1)] if n > 0 else 0
        return {
            "avg_ms": round(avg, 1),
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "p99_ms": round(p99, 1),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a full summary for the dashboard."""
        perf = self.get_performance()
        requests_full = self.get_requests(limit=self._retention // 5)
        requests_summary = self.get_requests_summary(limit=self._retention)

        # Count incomplete requests
        incomplete_count = sum(1 for r in self._requests if r.get("status") != "complete")

        return {
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "performance": perf,
            "requests": requests_full,
            "requests_summary": requests_summary,
            "incomplete_count": incomplete_count,
        }


# Module-level singleton
metrics = MetricsCollector(METRICS_RETENTION)
