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
| `GET /metrics/diagnostics?request_id=UUID` | Routing diagnostics for a specific request |
| `GET /metrics/diagnostics?liveness=true` | Recent backend liveness change events |
| `GET /metrics/diagnostics?timeline=true` | Unified timeline (requests + events) for post-mortem |

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
| `cached_tokens` | int | Tokens served from llama.cpp's KV cache (0 = no cache used) |
| `latency_ms` | float | Total request latency |
| `status` | str | `"incomplete"` (arrival only) or `"complete"` |
| `stream` | bool | Whether streaming was used |
| `routing_reason` | str | Why this backend was chosen |
| `routing_diagnostics` | dict | Full cache scan trace (see below) |
| `request_json` | dict | Full request body (large, omit for summaries) |
| `prompt_preview` | str | First 200 chars of latest user/assistant message |

**Routing reasons:**

| Value | Meaning |
|-------|---------|
| `pending_slot_hit` | In-flight slot had matching KV cache, served from it |
| `cache_hit` | Disk cache hit, served from cache backend |
| `no_cache_entry` | No cache match found, served from any available backend |
| `cache_backend_unavailable` | Cache hit found but cache backend was busy, served from fallback |

## Routing Diagnostics

The `routing_diagnostics` dict is set during the routing phase and contains the full cache scan trace:

| Field | Type | Description |
|-------|------|-------------|
| `best_ratio` | float | Highest LCP ratio found across all backends |
| `restore_key` | str or None | Cache file key (first 16 chars), or None for pending slot hits |
| `restore_backend` | str or None | Backend with the winning match |
| `restore_info_backend` | str or None | Backend that would get priority in `acquire_for_request` |
| `candidate_backends` | list[str] | Fallback backends (excludes restore backend when `restore_key` is set) |
| `scan` | list[dict] | Per-backend scan results (see below) |

**Per-backend scan entry** (`routing_diagnostics.scan[]`):

| Field | Type | Description |
|-------|------|-------------|
| `model` | str | Model name for this backend |
| `backend` | str | Backend key |
| `status` | str or None | `"unreachable"` if backend was down during scan |
| `n_blocks` | int | Number of blocks in the request for this backend |
| `n_tokens` | int | Number of tokens in the request for this backend |
| `cache_file_key` | str or None | Best cache file key (first 16 chars), or None if no match |
| `cache_file_ratio` | float or None | LCP ratio for best cache file match, or None |
| `pending_slots` | list[dict] | Pending slot matches: `[{slot, lcp_blocks, slot_blocks, ratio}]` |

**Liveness events** (`event="liveness_change"`):

| Field | Type | Description |
|-------|------|-------------|
| `state_changes` | list[dict] | `[{backend, old_state, new_state}]` for each changed backend |
| `discovered_models` | dict | `{model_name: [backend_keys]}` after discovery |

## Diagnostic Queries

### Trace why a request routed to a specific backend
```bash
curl -s 'localhost:1235/metrics/diagnostics?request_id=UUID' | python3 -m json.tool
```

### Check if a backend went down during a time window
```bash
curl -s 'localhost:1235/metrics/diagnostics?liveness=true' | python3 -c "
import json, sys, time
data = json.load(sys.stdin)
for e in data.get('liveness_events', []):
    ts = time.strftime('%H:%M:%S', time.localtime(e.get('timestamp', 0)))
    for sc in e.get('state_changes', []):
        print(f'{ts} {sc[\"backend\"]}: {sc[\"old_state\"]} -> {sc[\"new_state\"]}')
"
```

### Unified timeline for post-mortem analysis
```bash
curl -s 'localhost:1235/metrics/diagnostics?timeline=true' | python3 -c "
import json, sys, time
data = json.load(sys.stdin)
for e in data.get('timeline', []):
    ts = time.strftime('%H:%M:%S', time.localtime(e.get('timestamp', 0)))
    ev = e.get('event')
    if ev:
        for sc in e.get('state_changes', []):
            print(f'{ts} [{ev}] {sc[\"backend\"]}: {sc[\"old_state\"]} -> {sc[\"new_state\"]}')
    else:
        print(f'{ts} [{e.get(\"status\",\"?\")}] {e.get(\"request_id\",\"\")[:8]} be={e.get(\"backend\",\"?\")[:18]} reason={e.get(\"routing_reason\",\"\")}')
"
```

### Compare pending slot ratios across backends for a request
```bash
curl -s 'localhost:1235/metrics/diagnostics?request_id=UUID' | python3 -c "
import json, sys
d = json.load(sys.stdin).get('routing_diagnostics', {})
for s in d.get('scan', []):
    be = s.get('backend', '?')[:20]
    cf = s.get('cache_file_ratio') or 'none'
    ps = [(p['slot'], p['ratio']) for p in s.get('pending_slots', [])]
    print(f'{be}: cache_file={cf}, pending_slots={ps}')
print(f'Winner: backend={d.get(\"restore_backend\")}, ratio={d.get(\"best_ratio\")}, key={d.get(\"restore_key\")}')
print(f'Candidates: {d.get(\"candidate_backends\")}')
"
```

### Find requests where pending slot beat cache file
```bash
curl -s 'localhost:1235/metrics/requests?limit=100' | python3 -c "
import json, sys, time
data = json.load(sys.stdin)
for r in data.get('requests', []):
    if r.get('routing_reason') != 'pending_slot_hit': continue
    diag = r.get('routing_diagnostics', {})
    scan = diag.get('scan', [])
    # Check if any backend had a cache file match
    cache_hits = [s for s in scan if s.get('cache_file_ratio')]
    if cache_hits:
        ts = time.strftime('%H:%M:%S', time.localtime(r.get('timestamp', 0)))
        print(f'{ts} {r.get(\"request_id\",\"\")[:8]} be={r.get(\"backend\",\"?\")[:18]} ratio={diag.get(\"best_ratio\")}')
        for s in cache_hits:
            print(f'  cache_file on {s[\"backend\"]}: ratio={s[\"cache_file_ratio\"]}')
        for s in scan:
            for p in s.get('pending_slots', []):
                if p['ratio'] == diag.get('best_ratio'):
                    print(f'  pending_slot WON on {s[\"backend\"]} slot {p[\"slot\"]}: ratio={p[\"ratio\"]}')
"
```

### Find requests where `cached_tokens=0` (KV cache was useless)
```bash
curl -s 'localhost:1235/metrics/requests?limit=100' | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data.get('requests', []):
    if r.get('status') != 'complete': continue
    ct = r.get('cached_tokens')
    if ct is not None and ct == 0 and r.get('n_tokens', 0) > 100:
        print(f'{r.get(\"request_id\",\"\")[:8]} be={r.get(\"backend\",\"?\")[:18]} reason={r.get(\"routing_reason\")} restored={r.get(\"restored\")} tokens={r.get(\"n_tokens\")}')
"
```

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

## Gotchas

- **Incomplete records**: Arrival records have `status="incomplete"`, `backend="unknown"`, `slot_id=-1`. Filter them out with `r.get('status') == 'complete'`.
- **`request_json` is large**: Use `requests_summary` from `/metrics/summary` for bulk queries, or omit `request_json` when printing.
- **`cache_hit` vs `restored`**: `cache_hit=True` means a cache match was found. `restored=True` means the KV cache was actually loaded from disk. They can differ: pending slot hits have `cache_hit=True` but `restored=False` (slot already has content).
- **`cached_tokens=0` on pending slot hit**: Means `_slot_kv_state` was wrong — the proxy thought the slot had matching blocks, but llama.cpp's actual KV cache didn't match. Common when the slot was evicted or served a different conversation.
- **Ring buffer size**: Single ring buffer for requests + events (`METRICS_RETENTION`, default 200). `get_requests()` filters out events automatically. Use `?timeline=true` to see all entries.
- **`routing_reason` not in `request_json`**: It's a top-level field in the metrics record, not inside `request_json`.
- **`routing_diagnostics.scan` may be missing backends**: If a backend went down during the scan, it appears with `status="unreachable"` and no ratio data. Check liveness events to see if the backend was dropped.
