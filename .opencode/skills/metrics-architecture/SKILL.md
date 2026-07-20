---
name: metrics-architecture
description: Metrics collector with multi-phase request recording, routing diagnostics, skip-restore tracking, ring buffer, liveness events, and dashboard. Use when working with request metrics, routing diagnostics, performance data, cache hit analysis, dashboard, or any proxycache observability.
---

# Metrics Architecture

## Multi-Phase Recording

Requests flow through multiple recording phases, all updating the same ring buffer entry in-place via `request_id` matching.

1. **Arrival**: `chat()` generates `request_id` (UUID), calls `record()` with `status="incomplete"`, `prompt_preview`, and `request_json`. The entry is appended to the ring buffer.
2. **Routing**: After slot acquisition, `record()` updates the entry with resolved `model` (canonical name), `backend`, `slot_id`, `routing_reason`, `cache_hit`, `restored`, and `routing_diagnostics`. Status remains `"incomplete"`.
3. **Terminal**: The request reaches a final status:
   - **`"complete"`** — normal finish (streaming `_cleanup()` with `_stream_complete=True`, or non-streaming success). Includes latency, tokens, save status, recompute.
   - **`"cancelled"`** — streaming request where client disconnected (`_cancelled=True`, `_stream_complete=False`).
   - **`"backend_error"`** — backend timeout, connection error, streaming response non-200, or generic exception. Recorded by error handlers and streaming `_cleanup()` (backend disconnect case).

**Liveness events**: The backend manager's `_liveness_loop()` records `event="liveness_change"` entries when backend state changes (up/down) or models are missing from discovery. These use synthetic `request_id` values (`liveness:<timestamp_ms>`) and `status="complete"`.

**Key behavior:**
- `_by_id: Dict[str, int]` maps request_id → index in ring buffer, rebuilt after every append (deque auto-evicts without notification, leaving stale indices)
- Arrival timestamp, `prompt_preview`, and `full_request_json` are preserved when updating existing records
- Counters only increment on `status="complete"`, not on any other status
- `get_performance()` only uses complete requests; `get_summary()` includes `incomplete_count`
- Defaults to `status="complete"` for backward compatibility with code that doesn't use multi-phase recording

## Routing Reasons

Set after slot acquisition in `app.py`, explains why each request was dispatched to its chosen backend:

| Reason | Meaning |
|--------|---------|
| `cache_hit` | Disk cache hit found, routed to that backend |
| `pending_slot_hit` | In-flight slot found with better ratio, routed to that backend |
| `no_cache_entry` | No cache entry found, first available backend |
| `cache_backend_unavailable` | Cache hit found but desired backend was busy/unavailable, fell back |

The `pending_slot_hit` flag is set when the pending slot scan finds a better ratio than the disk cache hit. It must be reset to `False` when a disk cache hit supersedes it in the per-backend loop.

## Routing Diagnostics

Captured during the routing phase (phase 2) and stored as `routing_diagnostics` on the request record. Provides a full trace of the cache scan for post-hoc analysis.

**Structure:**
```python
{
    "best_ratio": 0.992,
    "restore_key": "abc123..." or None,
    "restore_backend": "backend.lan-1234" or None,
    "restore_info_backend": "backend.lan-1234" or None,
    "candidate_backends": ["other.lan-1234"],
    "skip_restore": {
        "skipped": True,
        "backend": "backend.lan-1234",
        "slot_id": 0,
        "old_kv_blocks": 512,
        "req_blocks": 515,
        "restore_key": "abc123..."
    } or {},
    "scan": [
        {
            "model": "unsloth/...",
            "backend": "backend.lan-1234",
            "n_blocks": 519,
            "n_tokens": 51838,
            "cache_file_key": "abc123..." or None,
            "cache_file_ratio": 0.85 or None,
            "pending_slots": [
                {"slot": 0, "lcp_blocks": 515, "slot_blocks": 515, "ratio": 0.992}
            ]
        }
    ]
}
```

**Key fields:**
- `restore_key` is None when a pending slot hit won (no disk restore needed)
- `restore_info_backend` is None when `restore_key` is None (no priority routing)
- `candidate_backends` excludes the restore backend ONLY when `restore_key` is set (pending slot hits don't exclude)
- `skip_restore` is populated (non-empty dict) when skip-restore fires, with `skipped=True`, block counts, and restore key. Empty dict `{}` means skip-restore did not fire.
- `scan` entries with `status="unreachable"` mean the backend was down during the cache scan
- `pending_slots` lists all matching slots per backend with their LCP details

## Ring Buffer

Single `deque(maxlen=retention)` (default 200) holds both request records and diagnostic events. Events are distinguished by the `event` field (e.g. `"liveness_change"`). When full, the oldest entry is auto-evicted on append.

**Request queries** (events filtered out automatically):
- `get_requests(limit, offset)` — returns request records **newest-first**
- `get_total_count()` — returns request count only (use for pagination)
- `get_requests_summary()` — returns entries without full JSON payload, **newest-first**
- `get_performance()` — computes metrics from complete requests only

**Event queries**:
- `get_events(event_type, limit)` — returns events, optionally filtered by type, **newest-first**
- `get_timeline(limit)` — returns unified timeline (requests + events), **newest-first**

**Recording**:
- `record(ctx)` — if `ctx` has `event` key, appends as event (append-only). Otherwise, treats as request (two-phase with in-place update via `request_id`).

## Diagnostics Endpoint

`GET /metrics/diagnostics` provides routing and liveness diagnostics:

| Query Param | Returns |
|-------------|---------|
| `?request_id=UUID` | `routing_diagnostics` for a specific request |
| `?liveness=true` | Recent `liveness_change` events with `state_changes` and `discovered_models` |
| (no params) | `routing_diagnostics` for all recent requests |

## Dashboard

- **Badge consolidation**: single routing badge per request (`DISK HIT`, `PENDING HIT`, `DISK HIT / RECOMPUTE`, `NO ENTRY`, `BACKEND UNAVAIL`). A conditional status badge (`INCOMPLETE`, `CANCELLED`, `BACKEND ERROR`) is shown only for non-complete requests.
- **Sorting**: explicit timestamp sort (descending) in `_doRenderFilteredRequests()` ensures newest-first after client-side filtering.
- **Pagination**: `currentPage` persisted in `localStorage`. Clamped to last valid page on refresh when ring buffer shrinks. Auto-refresh calls `refreshRequests()` without resetting page.
- **Imports**: `metrics` and `extract_prompt_preview` imported at top of `app.py`, not inline.

## Key Functions

| Function | Location | Role |
|----------|----------|------|
| `extract_prompt_preview()` | `metrics.py` | Extract latest user/assistant message text from request JSON |
| `MetricsCollector.record()` | `metrics.py` | Multi-phase record/update by request_id |
| `MetricsCollector.get_performance()` | `metrics.py` | Compute metrics from complete requests only |
| `MetricsCollector.get_total_count()` | `metrics.py` | Return actual ring buffer size for pagination |
| `MetricsCollector.get_summary()` | `metrics.py` | Full summary including incomplete_count |
| `StreamReader._cleanup()` | `app.py` | Stream lifecycle: save, invalidate_slot, release slot, record metrics with terminal status |
| `BackendManager._liveness_loop()` | `backend_manager.py` | Ping backends every 5s, record state changes as liveness events |

## Gotchas

- **`_by_id` must be rebuilt after every append**: deque auto-evicts without notification, leaving stale indices that cause completion records to create duplicate entries instead of updating in-place
- **Only `"complete"` increments counters**: `cancelled` and `backend_error` statuses update the record but don't affect hit/miss/latency counters
- **All error paths must record metrics**: streaming response non-200, timeout, connect error, and generic exception handlers all call `record()` with `status="backend_error"` — omitting any leaves the record stuck at `"incomplete"`
- **`save_after()` exceptions**: wrapped in try/except in the non-streaming path so they don't bypass metrics recording
- **`pending_slot_hit` flag**: must be reset to `False` when a disk cache hit supersedes it in the per-backend loop
- **Pagination total**: use `get_total_count()`, not `len(requests)` (which is the length of the returned slice)
- **Arrival timestamp preserved**: when updating an existing record, the original timestamp is kept so requests show when they arrived, not when they completed
- **Silent try/except on arrival**: the arrival record is wrapped in try/except to avoid blocking the request; if it fails, the error is logged (not silently swallowed)
- **Prompt preview extraction**: `extract_prompt_preview()` in `metrics.py` — looks for the most recent message with role "user" or "assistant", iterating messages in reverse order, skipping empty content. Called from `record()`, streaming `_cleanup()`, arrival recording, and non-streaming completion.
- **`cached_tokens=0` on pending slot hit**: Indicates `_slot_kv_state` was stale — the proxy's block tracking didn't match llama.cpp's actual KV cache. The slot may have been evicted or served a different conversation.
- **`routing_diagnostics.scan` may skip backends**: If a backend goes down during the cache scan, it appears with `status="unreachable"` and no ratio data. Cross-reference with liveness events to determine if the backend was dropped from the model registry.
- **Pending slot exclusion fix**: `candidate_backends` excludes `restore_backend` only when `restore_key` is truthy. When a pending slot hit overrides a cache hit, `restore_key` becomes None, so the pending-hit backend stays in the candidate list (prevents the "backend excluded but not prioritized" bug).
