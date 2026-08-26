---
name: metrics-architecture
description: Metrics collector with multi-phase request recording, routing diagnostics, skip-restore tracking, ring buffer, liveness events, liveness diagnostics, and dashboard. Use when working with request metrics, routing diagnostics, performance data, cache hit analysis, dashboard, or any proxycache observability.
---

# Metrics Architecture

## Multi-Phase Recording

Requests flow through multiple recording phases, all updating the same ring buffer entry in-place via `request_id` matching.

1. **Arrival**: `chatHandler()` generates `requestID` (`newRequestID()`, UUID) and calls `Metrics.Record()` with `status="incomplete"`, `prompt_preview`, `request_json`, and the request classification fields. The entry is appended to the ring buffer.
2. **Routing**: After slot acquisition, `Metrics.Record()` updates the entry with resolved `model` (canonical name), `backend`, `slot_id`, `routing_reason`, `cache_hit`, `restored`, and `routing_diagnostics`. Status remains `"incomplete"`.
3. **Terminal**: The request reaches a final status:
   - **`"complete"`** — normal finish (streaming `StreamState.cleanup()` with `streamComplete=true`, or non-streaming success). Includes latency, tokens, save status, recompute.
   - **`"cancelled"`** — streaming request where client disconnected (`cancelled=true`, `streamComplete=false`).
   - **`"backend_error"`** — backend timeout, connection error, streaming response non-200, or generic error. Recorded by error handlers (`recordEarlyError`/`recordChatError`) and streaming `StreamState.cleanup()` (backend disconnect case).

**Liveness events**: The backend manager's `livenessLoop()` records two event types:
- `event="liveness_change"` — when backend state changes (up/down) or models are missing from discovery. Uses synthetic `request_id` (`liveness:<timestamp_ms>`), includes `state_changes` and `discovered_models`.
- `event="liveness_diag"` — on noteworthy liveness iterations (state change, health errors, retries, discover/refresh errors), **rate-limited per backend**: a sustained noteworthy state records at most once per `LIVENESS_DIAG_RECORD_INTERVAL` seconds (default 60) per backend; state transitions always record. Includes per-backend health timing, error names, retry status, discovery timing.

Both event types are rate-limited for a reason: liveness events share the metrics ring buffer with request records. A busy llama.cpp backend intermittently times out its health endpoint (HTTP is blocked during long prefill/generation), so the "fail then retry-succeed" tick would fire every 5s and evict the entire request history (~17 min at retention 200), making the dashboard's request history appear to "periodically empty." The missing-models discovery re-trigger is similarly gated per backend to `MISSING_MODELS_RETRY_INTERVAL` seconds (default 30).

**Key behavior:**
- `byID map[string]int` maps request_id → index in ring buffer, rebuilt after every append (`rebuildByID()`; the ring buffer auto-evicts without notification, leaving stale indices)
- Arrival timestamp, `prompt_preview`, and `request_json` are preserved when updating existing records
- Counters only increment on `status="complete"`, not on any other status
- `GetPerformance()` only uses complete requests; `GetSummary()` includes `incomplete_count`
- Defaults to `status="complete"` for backward compatibility with code that doesn't use multi-phase recording

## Routing Reasons

Set after slot acquisition in `app.go`, explains why each request was dispatched to its chosen backend:

| Reason | Meaning |
|--------|---------|
| `cache_hit` | Disk cache hit found, routed to that backend |
| `pending_slot_hit` | In-flight slot found with better ratio, routed to that backend |
| `no_cache_entry` | No cache entry found, first available backend |
| `cache_backend_unavailable` | Cache hit found but desired backend was busy/unavailable, fell back |

The `pending_slot_hit` flag is set when the pending slot scan finds a better ratio than the disk cache hit. It must be reset to `false` when a disk cache hit supersedes it in the per-backend loop.

## Routing Diagnostics

Captured during the routing phase (phase 2) and stored as `routing_diagnostics` on the request record. Provides a full trace of the cache scan for post-hoc analysis.

**Structure:**
```json
{
    "best_ratio": 0.992,
    "restore_key": "abc123...",
    "restore_backend": "backend.lan-1234",
    "restore_info_backend": "backend.lan-1234",
    "candidate_backends": ["other.lan-1234"],
    "skip_restore": {
        "skipped": true,
        "backend": "backend.lan-1234",
        "slot_id": 0,
        "old_kv_blocks": 512,
        "req_blocks": 515,
        "restore_key": "abc123..."
    },
    "scan": [
        {
            "model": "unsloth/...",
            "backend": "backend.lan-1234",
            "n_blocks": 519,
            "n_tokens": 51838,
            "cache_file_key": "abc123...",
            "cache_file_ratio": 0.85,
            "pending_slots": [
                {"slot": 0, "lcp_blocks": 515, "slot_blocks": 515, "ratio": 0.992}
            ]
        }
    ]
}
```
(`skip_restore` is `{}` when skip-restore did not fire; absent `restore_key`/`restore_backend`/`restore_info_backend`/`cache_file_*` fields mean `null`.)

**Key fields:**
- `restore_key` is absent/`null` when a pending slot hit won (no disk restore needed)
- `restore_info_backend` is absent/`null` when `restore_key` is null (no priority routing)
- `candidate_backends` excludes the restore backend ONLY when `restore_key` is set (pending slot hits don't exclude)
- `skip_restore` is populated (non-empty dict) when skip-restore fires, with `skipped=true`, block counts, and restore key. Empty dict `{}` means skip-restore did not fire.
- `scan` entries with `status="unreachable"` mean the backend was down during the cache scan
- `pending_slots` lists all matching slots per backend with their LCP details

## Ring Buffer

Single ring slice (`metricsRetention`, default 200) holds both request records and diagnostic events. Events are distinguished by the `event` field (e.g. `"liveness_change"`, `"liveness_diag"`). When full, the oldest entry is evicted on append (`trimBuffer()`).

**Request queries** (events filtered out automatically):
- `GetRequests(limit, offset)` — returns request records **newest-first**
- `GetTotalCount()` — returns request count only (use for pagination)
- `GetRequestsSummary(limit, offset)` — returns entries without full JSON payload, **newest-first**
- `GetPerformance(model, backend, reqType)` — computes metrics from complete requests only

**Event queries**:
- `GetEvents(eventType, limit)` — returns events, optionally filtered by type (`"liveness_change"` or `"liveness_diag"`), **newest-first**
- `GetTimeline(limit)` — returns unified timeline (requests + events), **newest-first**

**Recording**:
- `Record(ctx)` — if `ctx` has `event` key, appends as event (append-only). Otherwise, treats as request (multi-phase with in-place update via `request_id`).

## Liveness Diagnostics (`liveness_diag` events)

Recorded when a liveness iteration has state changes, health errors, retries, or discover/refresh errors, gated by `livenessDiagDue()`: state transitions always record; otherwise a record is due only if a noteworthy backend has not been recorded within `LIVENESS_DIAG_RECORD_INTERVAL` seconds. Structure:

```json
{
    "event": "liveness_diag",
    "health": [
        {
            "backend": "10.0.0.1-8000",
            "is_up": true,
            "old_state": false,
            "state_changed": true,
            "ms": 45.2,
            "error": null,
            "recreated": true,
            "retry_succeeded": true
        }
    ],
    "states": {"10.0.0.1-8000": true, "other.lan-9000": false},
    "changed": true,
    "discovered_models": 2,
    "discover_timing": [
        {"backend": "10.0.0.1-8000", "ms": 320.5, "models": 1},
        {"backend": "other.lan-9000", "skipped": true}
    ],
    "discover_ms": 320.5,
    "discover_error": null,
    "slots_ms": 320.5,
    "slots_error": null,
    "total_ms": 410.3
}
```

**Key fields:**
- `health[].retry_succeeded` — the health check failed initially but recovered after client recreation + retry (indicates transient connection issue)
- `health[].recreated` — client was recreated for this backend in this iteration
- `discover_timing[].skipped` — backend was marked down, discovery skipped it
- `discover_ms` / `slots_ms` — timing of concurrent discovery/refresh operations (shared 10s timeout)
- `discover_error` / `slots_error` — error name or `null`. Go's `errName()` deliberately emits Python-flavored names (`"TimeoutError"`, `"ConnectError"`, `"ReadError"`, `"ConnectionError"`, `"HTTPStatusError"`) so old dashboards/logs stay consistent

Query via `GET /metrics/diagnostics?liveness_diag=true`.

## Diagnostics Endpoint

`GET /metrics/diagnostics` provides routing and liveness diagnostics:

| Query Param | Returns |
|-------------|---------|
| `?request_id=UUID` | `routing_diagnostics` for a specific request |
| `?liveness=true` | Recent `liveness_change` events with `state_changes` and `discovered_models` |
| `?liveness_diag=true` | Recent `liveness_diag` events with per-backend health timing, discovery timing, and error details |
| (no params) | `routing_diagnostics` for all recent requests |

## Dashboard

- **Badge consolidation**: single routing badge per request (`DISK HIT`, `PENDING HIT`, `DISK HIT / RECOMPUTE`, `NO ENTRY`, `BACKEND UNAVAIL`). A conditional status badge (`INCOMPLETE`, `CANCELLED`, `BACKEND ERROR`) is shown only for non-complete requests.
- **Sorting**: explicit timestamp sort (descending) in `_doRenderFilteredRequests()` ensures newest-first after client-side filtering.
- **Pagination**: `currentPage` persisted in `localStorage`. Clamped to last valid page on refresh when ring buffer shrinks. Auto-refresh calls `refreshRequests()` without resetting page.
- **Embedding**: `dashboard.html` is embedded into the binary via `//go:embed` in `app.go` and served at `/dashboard`. It has no build step — edit the file and rebuild.

## Key Functions

| Function | Location | Role |
|----------|----------|------|
| `ExtractPromptPreview()` | `metrics.go` | Extract latest user/assistant message text from request JSON |
| `MetricsCollector.Record()` | `metrics.go` | Multi-phase record/update by request_id |
| `MetricsCollector.GetPerformance()` | `metrics.go` | Compute metrics from complete requests only |
| `MetricsCollector.GetTotalCount()` | `metrics.go` | Return actual ring buffer size for pagination |
| `MetricsCollector.GetSummary()` | `metrics.go` | Full summary including incomplete_count |
| `StreamState.cleanup()` | `app.go` | Stream lifecycle: save, invalidate, release slot, record metrics with terminal status |
| `livenessLoop()` | `backendmanager.go` | Ping backends every 5s, record liveness_change + liveness_diag events |
| `livenessDiagDue()` | `backendmanager.go` | Pure gate: should a liveness_diag event record this tick |
| `errName()` | `backendmanager.go` | Map Go errors to Python-flavored exception type names for metrics |

## Gotchas

- **`byID` must be rebuilt after every append**: the ring buffer auto-evicts without notification (`trimBuffer()`), leaving stale indices that cause completion records to create duplicate entries instead of updating in-place. `Record()` always calls `rebuildByID()` after mutating the buffer.
- **Only `"complete"` increments counters**: `cancelled` and `backend_error` statuses update the record but don't affect hit/miss/latency counters
- **All error paths must record metrics**: streaming response non-200, timeout, connect error, and generic error handlers (`recordEarlyError`/`recordChatError` in `app.go`) all call `Metrics.Record()` with `status="backend_error"` — omitting any leaves the record stuck at `"incomplete"`
- **Non-streaming save failures**: `SaveAfter()` errors in the non-streaming path are logged and `saveOK` stays false — the completion record is still emitted with `saved=false` so metrics are never skipped
- **`pending_slot_hit` flag**: must be reset to `false` when a disk cache hit supersedes it in the per-backend loop
- **Pagination total**: use `GetTotalCount()`, not `len(requests)` (which is the length of the returned slice)
- **Arrival timestamp preserved**: when updating an existing record, the original timestamp is kept so requests show when they arrived, not when they completed
- **Prompt preview extraction**: `ExtractPromptPreview()` in `metrics.go` — looks for the most recent message with role "user" or "assistant", iterating messages in reverse order, skipping empty content. Called at arrival, in `StreamState.cleanup()`, and on non-streaming completion.
- **`cached_tokens=0` on pending slot hit**: Indicates `slotKVState` was stale — the proxy's block tracking didn't match llama.cpp's actual KV cache. The slot may have been evicted or served a different conversation.
- **`routing_diagnostics.scan` may skip backends**: If a backend goes down during the cache scan, it appears with `status="unreachable"` and no ratio data. Cross-reference with liveness events to determine if the backend was dropped from the model registry.
- **Liveness events and requests share one ring buffer**: because of this, liveness events are rate-limited per backend (`livenessDiagDue()` / `MissingModelsRetryInterval` gating). If you relax the gating, sustained health-check flapping (common while the backend is busy) will evict all request records and the dashboard history will appear to empty periodically.
- **Pending slot exclusion fix**: `candidate_backends` excludes `restore_backend` only when `restore_key` is truthy. When a pending slot hit overrides a cache hit, `restore_key` becomes None, so the pending-hit backend stays in the candidate list (prevents the "backend excluded but not prioritized" bug).
