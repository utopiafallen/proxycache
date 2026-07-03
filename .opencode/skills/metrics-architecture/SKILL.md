---
name: metrics-architecture
description: Metrics collector with multi-phase request recording, routing reasons, and dashboard integration. Use when modifying metrics.py, app.py metrics recording, or dashboard.html metrics display.
---

# Metrics Architecture

## Multi-Phase Recording

Requests flow through multiple recording phases, all updating the same ring buffer entry in-place via `request_id` matching.

1. **Arrival**: `chat()` generates `request_id` (UUID), calls `record()` with `status="incomplete"`, `prompt_preview`, and `request_json`. The entry is appended to the ring buffer.
2. **Routing**: After slot acquisition, `record()` updates the entry with resolved `model` (canonical name), `backend`, `slot_id`, `routing_reason`, `cache_hit`, `restored`. Status remains `"incomplete"`.
3. **Terminal**: The request reaches a final status:
   - **`"complete"`** — normal finish (streaming `_cleanup()` with `_stream_complete=True`, or non-streaming success). Includes latency, tokens, save status, recompute.
   - **`"cancelled"`** — streaming request where client disconnected (`_cancelled=True`, `_stream_complete=False`).
   - **`"backend_error"`** — backend timeout, connection error, streaming response non-200, or generic exception. Recorded by error handlers and streaming `_cleanup()` (backend disconnect case).

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

## Ring Buffer

`deque(maxlen=retention)` (default 100). When full, the oldest entry is auto-evicted on append.

- `get_requests(limit, offset)` — returns sliced list **newest-first** (ring buffer is reversed before slicing)
- `get_total_count()` — returns actual ring buffer size (use this for pagination total, not `len(requests)`)
- `get_requests_summary()` — returns entries without full JSON payload, **newest-first**, includes `status`, `request_id`, `routing_reason`
- `get_performance()` — computes metrics from complete requests only

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
