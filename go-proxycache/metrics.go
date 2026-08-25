// metrics.go — in-memory metrics collector.
//
// Single ring buffer holds request records and diagnostic events. Events are
// distinguished by the `event` field and filtered out of request-oriented
// queries. Two-phase recording: arrival (status="incomplete") then completion
// (status="complete") updates the same entry in-place via request_id.

package main

import (
	"math"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

const extractPreviewMaxLen = 200

func extractFromContent(content any) string {
	switch c := content.(type) {
	case string:
		stripped := strings.TrimSpace(c)
		if stripped != "" {
			return truncateRune(stripped, extractPreviewMaxLen)
		}
	case []any:
		for _, p := range c {
			pm, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if pm["type"] == "text" {
				if t, ok := pm["text"].(string); ok && strings.TrimSpace(t) != "" {
					return truncateRune(t, extractPreviewMaxLen)
				}
			}
		}
	}
	return ""
}

func truncateRune(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	return string([]rune(s)[:n])
}

// ExtractPromptPreview scans messages in reverse, returning the text content
// of the most recent user/assistant message (content, then reasoning_content).
func ExtractPromptPreview(requestJSON map[string]any) string {
	if len(requestJSON) == 0 {
		return ""
	}
	messages, _ := requestJSON["messages"].([]any)
	for i := len(messages) - 1; i >= 0; i-- {
		msg, ok := messages[i].(map[string]any)
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)
		if role != "user" && role != "assistant" {
			continue
		}
		if p := extractFromContent(msg["content"]); p != "" {
			return p
		}
		if p := extractFromContent(msg["reasoning_content"]); p != "" {
			return p
		}
	}
	return ""
}

func isEvent(record map[string]any) bool {
	ev, ok := record["event"]
	if !ok || ev == nil {
		return false
	}
	if s, ok := ev.(string); ok {
		return s != ""
	}
	return true
}

func getOrNil(r map[string]any, key string) any {
	if v, ok := r[key]; ok {
		return v
	}
	return nil
}

type MetricsCollector struct {
	mu        sync.Mutex
	retention int
	buffer    []map[string]any
	byID      map[string]int

	totalRequests    int
	cacheHits        int
	cacheMisses      int
	cacheRecomputes  int
	cacheSaved       int
	cacheSaveSkipped int
	restoreSuccesses int
	restoreFailures  int

	modelCounters   map[string]map[string]int
	backendCounters map[string]map[string]int
	startTime       time.Time
}

func NewMetricsCollector(retention int) *MetricsCollector {
	return &MetricsCollector{
		retention:       retention,
		byID:            map[string]int{},
		modelCounters:   map[string]map[string]int{},
		backendCounters: map[string]map[string]int{},
		startTime:       time.Now(),
	}
}

// Metrics is the module-level singleton.
var Metrics = NewMetricsCollector(MetricsRetention)

// Record records or updates a request, or appends a diagnostic event.
func (m *MetricsCollector) Record(ctx map[string]any) {
	// Event path: append-only
	if ev, ok := ctx["event"].(string); ok && ev != "" {
		entry := map[string]any{"timestamp": time.Now().Unix()}
		for k, v := range ctx {
			entry[k] = v
		}
		m.mu.Lock()
		m.buffer = append(m.buffer, entry)
		m.trimBuffer()
		m.rebuildByID()
		m.mu.Unlock()
		return
	}

	// Request path
	requestID, _ := ctx["request_id"].(string)
	model, _ := ctx["model"].(string)
	if model == "" {
		model = "unknown"
	}
	backend, _ := ctx["backend"].(string)
	if backend == "" {
		backend = "unknown"
	}
	cacheHit, _ := ctx["cache_hit"].(bool)
	restored := getOrNil(ctx, "restored")
	recompute, _ := ctx["recompute"].(bool)
	saved := getOrNil(ctx, "saved")
	latencyMS, _ := ctx["latency_ms"].(float64)
	status, _ := ctx["status"].(string)
	if status == "" {
		status = "complete"
	}
	isComplete := status == "complete"

	promptPreview, _ := ctx["prompt_preview"].(string)
	if promptPreview == "" {
		if rj, ok := ctx["request_json"].(map[string]any); ok {
			promptPreview = ExtractPromptPreview(rj)
		}
	}

	record := map[string]any{
		"timestamp":  time.Now().Unix(),
		"model":      model,
		"backend":    backend,
		"cache_hit":  cacheHit,
		"restored":   restored,
		"recompute":  recompute,
		"saved":      saved,
		"latency_ms": latencyMS,
		"status":     status,
	}
	for k, v := range ctx {
		record[k] = v
	}
	if sid, ok := ctx["slot_id"]; ok {
		record["slot_id"] = sid
	} else {
		record["slot_id"] = -1
	}
	record["prompt_preview"] = promptPreview

	m.mu.Lock()
	defer m.mu.Unlock()

	if idx, ok := m.byID[requestID]; ok {
		old := m.buffer[idx]
		oldPromptPreview, _ := old["prompt_preview"].(string)
		oldTimestamp := old["timestamp"]
		for k, v := range record {
			old[k] = v
		}
		// Preserve arrival timestamp
		if oldTimestamp != nil {
			old["timestamp"] = oldTimestamp
		}
		// Preserve full_request_json set on arrival if completion didn't provide one
		if _, exists := old["full_request_json"]; exists {
			if rj, ok := ctx["request_json"]; ok {
				old["full_request_json"] = rj
			}
		}
		if promptPreview == "" && oldPromptPreview != "" {
			old["prompt_preview"] = oldPromptPreview
		}
		if isComplete {
			old["status"] = "complete"
			m.incrementCounters(model, backend, cacheHit, recompute, saved, restored)
		}
		return
	}

	// New request — append to ring buffer
	if rj, ok := ctx["request_json"]; ok {
		record["full_request_json"] = rj
	} else {
		record["full_request_json"] = map[string]any{}
	}
	m.buffer = append(m.buffer, record)
	m.trimBuffer()
	m.rebuildByID()

	if isComplete {
		m.incrementCounters(model, backend, cacheHit, recompute, saved, restored)
	}
}

func (m *MetricsCollector) trimBuffer() {
	if len(m.buffer) > m.retention {
		m.buffer = m.buffer[len(m.buffer)-m.retention:]
	}
}

// rebuildByID must be called while holding m.mu.
func (m *MetricsCollector) rebuildByID() {
	m.byID = map[string]int{}
	for i, r := range m.buffer {
		rid, _ := r["request_id"].(string)
		if rid != "" && !isEvent(r) {
			m.byID[rid] = i
		}
	}
}

func (m *MetricsCollector) incrementCounters(model, backend string, cacheHit, recompute bool, saved, restored any) {
	m.totalRequests++
	if cacheHit {
		m.cacheHits++
		if recompute {
			m.cacheRecomputes++
		}
	} else {
		m.cacheMisses++
	}
	switch s := saved.(type) {
	case bool:
		if s {
			m.cacheSaved++
		} else {
			m.cacheSaveSkipped++
		}
	}
	switch r := restored.(type) {
	case bool:
		if r {
			m.restoreSuccesses++
		} else {
			m.restoreFailures++
		}
	}

	bump := func(counters map[string]map[string]int, key string) map[string]int {
		c, ok := counters[key]
		if !ok {
			c = map[string]int{"total": 0, "hits": 0, "misses": 0, "recomputes": 0, "saved": 0, "save_skipped": 0}
			counters[key] = c
		}
		return c
	}
	mc := bump(m.modelCounters, model)
	mc["total"]++
	if cacheHit {
		mc["hits"]++
		if recompute {
			mc["recomputes"]++
		}
	} else {
		mc["misses"]++
	}
	if s, ok := saved.(bool); ok {
		if s {
			mc["saved"]++
		} else {
			mc["save_skipped"]++
		}
	}
	bc := bump(m.backendCounters, backend)
	bc["total"]++
	if cacheHit {
		bc["hits"]++
		if recompute {
			bc["recomputes"]++
		}
	} else {
		bc["misses"]++
	}
	if s, ok := saved.(bool); ok {
		if s {
			bc["saved"]++
		} else {
			bc["save_skipped"]++
		}
	}
}

// ── Request queries (events filtered out) ──────────────────────

// GetRequestByID returns a copy of a single request record.
func (m *MetricsCollector) GetRequestByID(requestID string) map[string]any {
	m.mu.Lock()
	defer m.mu.Unlock()
	idx, ok := m.byID[requestID]
	if !ok {
		return nil
	}
	src := m.buffer[idx]
	dst := make(map[string]any, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func (m *MetricsCollector) reversedNonEvents() []map[string]any {
	entries := make([]map[string]any, 0, len(m.buffer))
	for i := len(m.buffer) - 1; i >= 0; i-- {
		entries = append(entries, m.buffer[i])
	}
	out := make([]map[string]any, 0, len(entries))
	for _, r := range entries {
		if !isEvent(r) {
			out = append(out, copyRecord(r))
		}
	}
	return out
}

func copyRecord(r map[string]any) map[string]any {
	dst := make(map[string]any, len(r))
	for k, v := range r {
		dst[k] = v
	}
	return dst
}

// GetRequests returns recent request records (events excluded), newest first.
func (m *MetricsCollector) GetRequests(limit, offset int) []map[string]any {
	m.mu.Lock()
	requests := m.reversedNonEvents()
	m.mu.Unlock()
	if offset < 0 {
		offset = 0
	}
	if offset > len(requests) {
		return []map[string]any{}
	}
	end := offset + limit
	if end > len(requests) {
		end = len(requests)
	}
	return requests[offset:end]
}

// GetTotalCount returns the number of request entries (events excluded).
func (m *MetricsCollector) GetTotalCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	n := 0
	for _, r := range m.buffer {
		if !isEvent(r) {
			n++
		}
	}
	return n
}

// GetRequestsSummary returns recent requests without the full JSON payload.
func (m *MetricsCollector) GetRequestsSummary(limit, offset int) []map[string]any {
	m.mu.Lock()
	requests := m.reversedNonEvents()
	m.mu.Unlock()
	if offset < 0 {
		offset = 0
	}
	if offset > len(requests) {
		return []map[string]any{}
	}
	end := offset + limit
	if end > len(requests) {
		end = len(requests)
	}
	out := make([]map[string]any, 0, end-offset)
	for _, r := range requests[offset:end] {
		status, _ := r["status"].(string)
		if status == "" {
			status = "incomplete"
		}
		out = append(out, map[string]any{
			"timestamp":             r["timestamp"],
			"model":                 r["model"],
			"backend":               r["backend"],
			"slot_id":               r["slot_id"],
			"cache_hit":             r["cache_hit"],
			"restored":              r["restored"],
			"recompute":             r["recompute"],
			"saved":                 r["saved"],
			"latency_ms":            r["latency_ms"],
			"n_tokens":              getOrNil(r, "n_tokens"),
			"cache_size_bytes":      getOrNil(r, "cache_size_bytes"),
			"cached_tokens":         getOrNil(r, "cached_tokens"),
			"prompt_preview":        r["prompt_preview"],
			"routing_reason":        getOrNil(r, "routing_reason"),
			"routing_diagnostics":   getOrNil(r, "routing_diagnostics"),
			"status":                status,
			"request_id":            getOrNil(r, "request_id"),
			"summarization_score":   getOrNil(r, "summarization_score"),
			"summarization_signals": getOrNil(r, "summarization_signals"),
		})
	}
	return out
}

// ── Event queries ──────────────────────────────────────────────

// GetEvents returns events (newest first), optionally filtered by type.
func (m *MetricsCollector) GetEvents(eventType string, limit int) []map[string]any {
	m.mu.Lock()
	entries := make([]map[string]any, 0, len(m.buffer))
	for i := len(m.buffer) - 1; i >= 0; i-- {
		entries = append(entries, copyRecord(m.buffer[i]))
	}
	m.mu.Unlock()
	events := make([]map[string]any, 0)
	for _, e := range entries {
		if !isEvent(e) {
			continue
		}
		if eventType != "" {
			if et, _ := e["event"].(string); et != eventType {
				continue
			}
		}
		events = append(events, e)
	}
	if limit > 0 && len(events) > limit {
		events = events[:limit]
	}
	return events
}

// GetTimeline returns the unified timeline (requests + events), newest first.
func (m *MetricsCollector) GetTimeline(limit int) []map[string]any {
	m.mu.Lock()
	entries := make([]map[string]any, 0, len(m.buffer))
	for i := len(m.buffer) - 1; i >= 0; i-- {
		entries = append(entries, copyRecord(m.buffer[i]))
	}
	m.mu.Unlock()
	if limit > 0 && len(entries) > limit {
		entries = entries[:limit]
	}
	return entries
}

// ── Performance ────────────────────────────────────────────────

// GetPerformance computes metrics from the buffer. Only uses requests with
// status="complete" (events excluded). model/backend/reqType may be "".
func (m *MetricsCollector) GetPerformance(model, backend, reqType string) map[string]any {
	var requests []map[string]any
	var counters map[string]int

	m.mu.Lock()
	if model != "" || backend != "" {
		for _, r := range m.buffer {
			if isEvent(r) {
				continue
			}
			if model != "" && r["model"] != model {
				continue
			}
			if backend != "" && r["backend"] != backend {
				continue
			}
			if r["status"] != "complete" {
				continue
			}
			requests = append(requests, r)
		}
		counters = map[string]int{"total": 0, "hits": 0, "misses": 0, "recomputes": 0, "saved": 0, "save_skipped": 0}
		src := m.modelCounters
		if model == "" {
			src = m.backendCounters
		}
		key := model
		if model == "" {
			key = backend
		}
		if c, ok := src[key]; ok {
			for k, v := range c {
				counters[k] = v
			}
		}
	} else {
		for _, r := range m.buffer {
			if !isEvent(r) && r["status"] == "complete" {
				requests = append(requests, r)
			}
		}
		counters = map[string]int{
			"total": m.totalRequests, "hits": m.cacheHits, "misses": m.cacheMisses,
			"recomputes": m.cacheRecomputes, "saved": m.cacheSaved, "save_skipped": m.cacheSaveSkipped,
		}
	}
	m.mu.Unlock()

	// Filter by request type
	if reqType == "summarization" {
		filtered := make([]map[string]any, 0, len(requests))
		for _, r := range requests {
			if score := sumScore(r); score >= 0.4 {
				filtered = append(filtered, r)
			}
		}
		requests = filtered
	} else if reqType == "conversation" {
		filtered := make([]map[string]any, 0, len(requests))
		for _, r := range requests {
			if score := sumScore(r); score < 0.4 {
				filtered = append(filtered, r)
			}
		}
		requests = filtered
	}

	total, hits, misses, recomputes, saved, saveSkipped := counters["total"], counters["hits"],
		counters["misses"], counters["recomputes"], counters["saved"], counters["save_skipped"]

	// When filtered by req_type, recompute counters from filtered requests
	if reqType != "" {
		total = len(requests)
		hits, misses, recomputes, saved, saveSkipped = 0, 0, 0, 0, 0
		for _, r := range requests {
			if ch, _ := r["cache_hit"].(bool); ch {
				hits++
			}
			if rc, _ := r["recompute"].(bool); rc {
				recomputes++
			}
			if s, ok := r["saved"].(bool); ok {
				if s {
					saved++
				} else {
					saveSkipped++
				}
			}
		}
		misses = total - hits
	}

	hitRate := 0.0
	if total > 0 {
		hitRate = float64(hits) / float64(total)
	}
	mispredictRate := 0.0
	if hits > 0 {
		mispredictRate = float64(recomputes) / float64(hits)
	}
	utilityRate := 0.0
	if total > 0 {
		utilityRate = float64(hits-recomputes) / float64(total)
	}
	totalRequestsForSave := hits + misses
	saveRate := 0.0
	if totalRequestsForSave > 0 {
		saveRate = float64(saved) / float64(totalRequestsForSave)
	}
	saveSkipRate := 0.0
	if totalRequestsForSave > 0 {
		saveSkipRate = float64(saveSkipped) / float64(totalRequestsForSave)
	}

	restoreSuccesses, restoreFailures := 0, 0
	for _, r := range requests {
		if v, ok := r["restored"].(bool); ok {
			if v {
				restoreSuccesses++
			} else {
				restoreFailures++
			}
		}
	}
	restoreRate := 0.0
	if restoreSuccesses+restoreFailures > 0 {
		restoreRate = float64(restoreSuccesses) / float64(restoreSuccesses+restoreFailures)
	}

	latencies := make([]float64, 0, len(requests))
	for _, r := range requests {
		if l, ok := r["latency_ms"].(float64); ok && l > 0 {
			latencies = append(latencies, l)
		}
	}
	sort.Float64s(latencies)

	return map[string]any{
		"total_requests":       total,
		"cache_hits":           hits,
		"cache_misses":         misses,
		"cache_recomputes":     recomputes,
		"cache_saved":          saved,
		"cache_save_skipped":   saveSkipped,
		"cache_hit_rate":       roundTo(hitRate, 4),
		"cache_mispredict_rate": roundTo(mispredictRate, 4),
		"cache_utility_rate":   roundTo(utilityRate, 4),
		"save_rate":            roundTo(saveRate, 4),
		"save_skip_rate":       roundTo(saveSkipRate, 4),
		"restore_success_rate": roundTo(restoreRate, 4),
		"latency":              computePercentiles(latencies),
	}
}

func sumScore(r map[string]any) float64 {
	s, _ := r["summarization_score"].(float64)
	return s
}

func computePercentiles(latencies []float64) map[string]float64 {
	if len(latencies) == 0 {
		return map[string]float64{"avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0}
	}
	n := len(latencies)
	var sum float64
	for _, v := range latencies {
		sum += v
	}
	avg := sum / float64(n)
	p50 := latencies[int(float64(n)*0.50)]
	p95 := latencies[min(int(float64(n)*0.95), n-1)]
	p99 := latencies[min(int(float64(n)*0.99), n-1)]
	return map[string]float64{
		"avg_ms": roundTo(avg, 1),
		"p50_ms": roundTo(p50, 1),
		"p95_ms": roundTo(p95, 1),
		"p99_ms": roundTo(p99, 1),
	}
}

// GetSummary returns the full dashboard summary.
func (m *MetricsCollector) GetSummary() map[string]any {
	perf := m.GetPerformance("", "", "")
	requestsFull := m.GetRequests(m.retention, 0)
	requestsSummary := m.GetRequestsSummary(m.retention, 0)

	m.mu.Lock()
	incompleteCount := 0
	var complete []map[string]any
	for _, r := range m.buffer {
		if isEvent(r) {
			continue
		}
		if r["status"] != "complete" {
			incompleteCount++
		} else {
			complete = append(complete, r)
		}
	}
	var sumRequests, convRequests []map[string]any
	for _, r := range complete {
		if sumScore(r) >= 0.4 {
			sumRequests = append(sumRequests, r)
		} else {
			convRequests = append(convRequests, r)
		}
	}
	var sumLatencies, convLatencies []float64
	for _, r := range sumRequests {
		if l, ok := r["latency_ms"].(float64); ok && l > 0 {
			sumLatencies = append(sumLatencies, l)
		}
	}
	for _, r := range convRequests {
		if l, ok := r["latency_ms"].(float64); ok && l > 0 {
			convLatencies = append(convLatencies, l)
		}
	}
	m.mu.Unlock()
	sort.Float64s(sumLatencies)
	sort.Float64s(convLatencies)

	return map[string]any{
		"uptime_seconds":    roundTo(time.Since(m.startTime).Seconds(), 1),
		"performance":       perf,
		"requests":          requestsFull,
		"requests_summary":  requestsSummary,
		"incomplete_count":  incompleteCount,
		"summarization": map[string]any{
			"total":                len(sumRequests),
			"conversation_total":   len(convRequests),
			"latency":              computePercentiles(sumLatencies),
			"conversation_latency": computePercentiles(convLatencies),
		},
	}
}

func roundTo(v float64, digits int) float64 {
	switch digits {
	case 1:
		return round1(v)
	case 3:
		return roundTo3(v)
	case 4:
		return round4(v)
	}
	return v
}

func round1(v float64) float64 { return math.Round(v*10) / 10 }
func round4(v float64) float64 { return math.Round(v*10000) / 10000 }
