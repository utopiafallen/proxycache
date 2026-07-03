---
name: metrics-query
description: Querying and parsing proxycache request history from the metrics dashboard. Use when querying metrics endpoints, analyzing request routing, or debugging cache hit/miss behavior.
---

# Metrics Query

Querying and parsing proxycache request history from the metrics dashboard.

## Metrics Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /metrics/requests?limit=N&offset=M` | Full request records including `request_json` |
| `GET /metrics/summary` | Aggregated stats + last 20 requests (via `requests_summary`) |
| `GET /metrics/health` | Backend health and model discovery |
| `GET /metrics/slots` | Per-slot status |
| `GET /metrics/cache` | Per-backend cache utilization |
| `GET /metrics/performance?model=X&backend=Y` | Cache performance metrics |

Default port is from `config.py` (`DASHBOARD_PORT`, defaults to 1235).

## Request Record Fields

Each completed request record contains:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | float | Unix timestamp of request arrival |
| `model` | str | Model name used |
| `backend` | str | Backend key that served the request |
| `slot_id` | int | Slot used (-1 for incomplete) |
| `cache_hit` | bool | Whether a cache hit was found (disk or pending slot) |
| `restored` | bool or None | Whether KV cache was actually restored |
| `recompute` | bool | Whether llama.cpp had to recompute (partial restore) |
| `saved` | bool or None | Whether cache was saved after response |
| `latency_ms` | float | Total request latency |
| `status` | str | `"incomplete"` (arrival only) or `"complete"` |
| `stream` | bool | Whether streaming was used |
| `routing_reason` | str | Why this backend was chosen |
| `request_json` | dict | Full request body (large, omit for summaries) |
| `prompt_preview` | str | First 200 chars of latest user/assistant message |

**Routing reasons:**

| Value | Meaning |
|-------|---------|
| `pending_slot_hit` | In-flight slot had matching KV cache, served from it |
| `cache_hit` | Disk cache hit, served from cache backend |
| `no_cache_entry` | No cache match found, served from any available backend |
| `cache_backend_unavailable` | Cache hit found but cache backend was busy, served from fallback |

## Query Script

`query_metrics.py` (in this skill directory) provides a CLI for common queries:

```bash
python .opencode/skills/metrics-query/query_metrics.py                          # summary + recent requests
python .opencode/skills/metrics-query/query_metrics.py --reason pending_slot_hit  # filter by routing reason
python .opencode/skills/metrics-query/query_metrics.py --reason cache_backend_unavailable
python .opencode/skills/metrics-query/query_metrics.py --suspicious             # restored=True but cache_hit=False
python .opencode/skills/metrics-query/query_metrics.py --top 20                 # show more requests
python .opencode/skills/metrics-query/query_metrics.py --json                   # raw JSON output
python .opencode/skills/metrics-query/query_metrics.py --limit 200              # fetch more records
```

## Common Queries

### Check routing distribution
```bash
python query_metrics.py | head -10
```

### Find pending slot hits
```bash
python query_metrics.py --reason pending_slot_hit
```

### Find misclassified requests (restored but marked as miss)
```bash
python query_metrics.py --suspicious
```

### Raw query with curl
```bash
# Recent completed requests with key fields
curl -s 'localhost:1235/metrics/requests?limit=50' | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data['requests']:
    if r.get('status') != 'complete': continue
    print(f\"{r.get('routing_reason')} hit={r.get('cache_hit')} restored={r.get('restored')} {r.get('model','?')[:30]}\")
"
```

## Gotchas

- **Incomplete records**: Arrival records have `status="incomplete"`, `backend="unknown"`, `slot_id=-1`. Filter them out with `r.get('status') == 'complete'`.
- **`request_json` is large**: Use `requests_summary` from `/metrics/summary` for bulk queries, or omit `request_json` when printing.
- **`cache_hit` vs `restored`**: `cache_hit=True` means a cache match was found. `restored=True` means the KV cache was actually loaded from disk. They can differ: pending slot hits have `cache_hit=True` but `restored=False` (slot already has content).
- **Ring buffer size**: Metrics are kept in memory with a fixed ring buffer size (`METRICS_RETENTION`, default 1000). Old records are evicted when the buffer is full.
- **`routing_reason` not in `request_json`**: It's a top-level field in the metrics record, not inside `request_json`.
