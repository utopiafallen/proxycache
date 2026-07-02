#!/usr/bin/env python3
"""Query proxycache request history from metrics endpoint.

Usage:
    python query_metrics.py                  # summary + recent requests
    python query_metrics.py --reason pending_slot_hit   # filter by routing reason
    python query_metrics.py --reason cache_backend_unavailable
    python query_metrics.py --suspicious     # restored=True but cache_hit=False
    python query_metrics.py --top 20         # show more requests
    python query_metrics.py --json           # raw JSON output
"""

import argparse
import json
import sys
import urllib.request
from collections import Counter
from datetime import datetime

BASE_URL = "http://localhost:1235"


def fetch_requests(limit=100, offset=0):
    url = f"{BASE_URL}/metrics/requests?limit={limit}&offset={offset}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def fetch_summary():
    url = f"{BASE_URL}/metrics/summary"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def format_time(ts):
    if not ts:
        return "?"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def print_summary(data):
    requests = [r for r in data["requests"] if r.get("status") == "complete"]
    if not requests:
        print("No completed requests.")
        return

    reasons = Counter(r.get("routing_reason", "MISSING") for r in requests)
    print(f"Completed requests: {len(requests)}")
    print("\nRouting reasons:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")

    hits = sum(1 for r in requests if r.get("cache_hit"))
    print(f"\nCache hits: {hits}, misses: {len(requests) - hits}")

    recomputes = sum(1 for r in requests if r.get("recompute"))
    print(f"Recomputes: {recomputes}")

    # Suspicious entries
    suspicious = [r for r in requests if r.get("restored") and not r.get("cache_hit")]
    if suspicious:
        print(f"\nSuspicious (restored but cache_hit=False): {len(suspicious)}")
        for r in suspicious[:5]:
            print(f"  {format_time(r.get('timestamp'))} {r.get('routing_reason')} {r.get('model', '?')[:40]}")


def print_requests(data, reason_filter=None, suspicious_only=False, top=10):
    requests = [r for r in data["requests"] if r.get("status") == "complete"]

    if reason_filter:
        requests = [r for r in requests if r.get("routing_reason") == reason_filter]
    if suspicious_only:
        requests = [r for r in requests if r.get("restored") and not r.get("cache_hit")]

    print(f"\nShowing {min(top, len(requests))} of {len(requests)} requests:")
    for r in requests[:top]:
        hit_label = "HIT" if r.get("cache_hit") else "miss"
        restored = r.get("restored")
        reason = r.get("routing_reason", "?")
        model = r.get("model", "?")[:35]
        preview = r.get("prompt_preview", "")[:50]
        latency = r.get("latency_ms", 0)
        recompute = " RECOMPUTE" if r.get("recompute") else ""

        print(f"  {format_time(r.get('timestamp'))} {hit_label:4} {reason:28} "
              f"{model} {latency:.0f}ms{recompute}")
        if preview:
            print(f"    {preview}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason", help="Filter by routing reason")
    parser.add_argument("--suspicious", action="store_true",
                        help="Show requests where restored=True but cache_hit=False")
    parser.add_argument("--top", type=int, default=10, help="Number of requests to show")
    parser.add_argument("--json", action="store_true", help="Raw JSON output")
    parser.add_argument("--limit", type=int, default=100, help="Max requests to fetch")
    args = parser.parse_args()

    try:
        data = fetch_requests(limit=args.limit)
    except Exception as e:
        print(f"Error fetching metrics: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(data, indent=2))
        return

    if not args.reason and not args.suspicious:
        print_summary(data)

    print_requests(data, reason_filter=args.reason,
                   suspicious_only=args.suspicious, top=args.top)


if __name__ == "__main__":
    main()
