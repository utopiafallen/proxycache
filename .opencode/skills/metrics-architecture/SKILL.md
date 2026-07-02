---
name: metrics-architecture
description: Metrics collector with two-phase request recording, routing reasons, and dashboard integration. Use when modifying metrics.py, app.py metrics recording, or dashboard.html metrics display.
---

# Metrics Architecture

## Two-Phase Recording

Requests are recorded immediately on arrival and updated in-place on completion via `request_id` matching.

1. **Arrival**: `chat()` generates `request_id` (UUID), calls `record()` with `status="incomplete"`, `prompt_preview`, and `request_json`. The entry is appended to the ring buffer.
2. **Completion**: Both streaming (`_cleanup()`) and non-streaming paths call `record()` with the same `request_id`, `status="complete"`, and performance metrics. The existing entry is updated in-place.

**Key behavior:**
- `_by_id: Dict[str, int]` maps request_id → index in ring buffer, rebuilt after every append (deque auto-evicts without notification, leaving stale indices)
- Arrival timestamp, `prompt_preview`, and `full_request_json` are preserved when updating existing records
- Counters only increment on completion (status="complete"), not arrival
- `get_performance()` only uses complete requests; `get_summary()` includes `incomplete_count`
- Defaults to `status="complete"` for backward compatibility with code that doesn't use two-phase recording

## Routing Reasons

Set after slot acquisition in `app.py`, explains why each request was dispatched to its chosen backend:

| Reason | Meaning |
|--------|---------|
| `cache_hit` | Disk cache hit found, routed to that backend |
| `pending_slot_hit` | In-flight slot found with better ratio, routed to that backend |
| `no_cache_entry` | No cache entry found, first available backend |
| `cache_backend_unavailable` | Cache hit found but desired backend was busy/unavailable, fell back |

The `pending_slot_hit` flag is set when the pending slot scan finds a better ratio than the disk cache hit. It must be reset to `False` when a disk cache hit supersedes it in the per-backend loop.

## Ring Buffer

`deque(maxlen=retention)` (default 100). When full, the oldest entry is auto-evicted on append.

- `get_requests(limit, offset)` — returns sliced list for pagination
- `get_total_count()` — returns actual ring buffer size (use this for pagination total, not `len(requests)`)
- `get_requests_summary()` — returns entries without full JSON payload, includes `status`, `request_id`, `routing_reason`
- `get_performance()` — computes metrics from complete requests only

## Key Functions

| Function | Location | Role |
|----------|----------|------|
| `MetricsCollector.record()` | `metrics.py` | Two-phase record/update by request_id |
| `MetricsCollector.get_performance()` | `metrics.py` | Compute metrics from complete requests only |
| `MetricsCollector.get_total_count()` | `metrics.py` | Return actual ring buffer size for pagination |
| `MetricsCollector.get_summary()` | `metrics.py` | Full summary including incomplete_count |
| `StreamReader._cleanup()` | `app.py` | Stream lifecycle: save, invalidate_slot, release slot, record metrics |

## Gotchas

- **`_by_id` must be rebuilt after every append**: deque auto-evicts without notification, leaving stale indices that cause completion records to create duplicate entries instead of updating in-place
- **`save_after()` exceptions**: wrapped in try/except in the non-streaming path so they don't bypass metrics recording
- **`pending_slot_hit` flag**: must be reset to `False` when a disk cache hit supersedes it in the per-backend loop
- **Pagination total**: use `get_total_count()`, not `len(requests)` (which is the length of the returned slice)
- **Arrival timestamp preserved**: when updating an existing record, the original timestamp is kept so requests show when they arrived, not when they completed
- **Silent try/except on arrival**: the arrival record is wrapped in try/except to avoid blocking the request; if it fails, the error is logged (not silently swallowed)
