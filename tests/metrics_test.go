package tests

import (
	"proxycache"
	"reflect"
	"strings"
	"testing"
)

func TestTruncateRune(t *testing.T) {
	if got := proxycache.TruncateRune("abc", 200); got != "abc" {
		t.Errorf("short = %q, want unchanged", got)
	}
	if got := proxycache.TruncateRune(strings.Repeat("a", 200), 200); len(got) != 200 {
		t.Errorf("exact length = %d, want 200", len(got))
	}
	if got := proxycache.TruncateRune(strings.Repeat("a", 250), 200); len(got) != 200 {
		t.Errorf("ascii truncated len = %d, want 200", len(got))
	}
	if got := proxycache.TruncateRune(strings.Repeat("é", 250), 200); utf8RuneCount(got) != 200 {
		t.Errorf("multibyte truncated runes = %d, want 200", utf8RuneCount(got))
	}
}

func utf8RuneCount(s string) int {
	n := 0
	for range s {
		n++
	}
	return n
}

func TestExtractFromContent(t *testing.T) {
	cases := []struct {
		name    string
		content any
		want    string
	}{
		{"string_trimmed", "  hello  ", "hello"},
		{"string_whitespace", "   \t\n ", ""},
		{"nil", nil, ""},
		{"number", 42, ""},
		{"list_text_part", []any{map[string]any{"type": "text", "text": "x"}}, "x"},
		{"list_non_map_skipped", []any{"str", 1}, ""},
		{"list_blank_text_skipped", []any{map[string]any{"type": "text", "text": "   "}}, ""},
		{"list_untrimmed", []any{map[string]any{"type": "text", "text": "  hi  "}}, "  hi  "},
		{"list_non_text_first", []any{map[string]any{"type": "image_url"}, map[string]any{"type": "text", "text": "after"}}, "after"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := proxycache.ExtractFromContent(tc.content); got != tc.want {
				t.Errorf("ExtractFromContent(%v) = %q, want %q", tc.content, got, tc.want)
			}
		})
	}
}

func TestExtractPromptPreview(t *testing.T) {
	if got := proxycache.ExtractPromptPreview(map[string]any{}); got != "" {
		t.Errorf("empty request = %q, want empty", got)
	}
	if got := proxycache.ExtractPromptPreview(nil); got != "" {
		t.Errorf("nil request = %q, want empty", got)
	}
	if got := proxycache.ExtractPromptPreview(map[string]any{"model": "x"}); got != "" {
		t.Errorf("no messages = %q, want empty", got)
	}

	msgs := []any{
		map[string]any{"role": "system", "content": "sys"},
		map[string]any{"role": "user", "content": "hello"},
	}
	if got := proxycache.ExtractPromptPreview(map[string]any{"messages": msgs}); got != "hello" {
		t.Errorf("user content = %q, want hello", got)
	}

	msgs2 := []any{
		map[string]any{"role": "user", "content": "first"},
		map[string]any{"role": "assistant", "content": "second"},
	}
	if got := proxycache.ExtractPromptPreview(map[string]any{"messages": msgs2}); got != "second" {
		t.Errorf("reverse scan = %q, want second (most recent user/assistant)", got)
	}

	msgs3 := []any{
		map[string]any{"role": "system", "content": "sys"},
	}
	if got := proxycache.ExtractPromptPreview(map[string]any{"messages": msgs3}); got != "" {
		t.Errorf("system only = %q, want empty", got)
	}

	msgs4 := []any{
		map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "fromlist"}}},
	}
	if got := proxycache.ExtractPromptPreview(map[string]any{"messages": msgs4}); got != "fromlist" {
		t.Errorf("content list = %q, want fromlist", got)
	}

	msgs5 := []any{
		map[string]any{"role": "user", "content": nil, "reasoning_content": "think"},
	}
	if got := proxycache.ExtractPromptPreview(map[string]any{"messages": msgs5}); got != "think" {
		t.Errorf("reasoning fallback = %q, want think", got)
	}

	msgs6 := []any{
		map[string]any{"role": "user", "content": strings.Repeat("a", 250)},
	}
	if got := proxycache.ExtractPromptPreview(map[string]any{"messages": msgs6}); len(got) != 200 {
		t.Errorf("truncated preview len = %d, want 200", len(got))
	}

	msgs7 := []any{
		map[string]any{"role": "assistant", "content": ""},
		map[string]any{"role": "user", "content": "earlier"},
	}
	if got := proxycache.ExtractPromptPreview(map[string]any{"messages": msgs7}); got != "earlier" {
		t.Errorf("empty content skipped = %q, want earlier", got)
	}
}

func TestMetricsTwoPhaseRecording(t *testing.T) {
	m := proxycache.NewMetricsCollector(200)
	reqJSON := map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "hello"}},
	}
	m.Record(map[string]any{
		"request_id":   "r1",
		"model":        "M",
		"backend":      "B",
		"status":       "incomplete",
		"request_json": reqJSON,
	})

	reqs := m.GetRequests(10, 0)
	if len(reqs) != 1 {
		t.Fatalf("after arrival: %d requests, want 1", len(reqs))
	}
	if reqs[0]["status"] != "incomplete" {
		t.Errorf("status = %v, want incomplete", reqs[0]["status"])
	}
	if reqs[0]["prompt_preview"] != "hello" {
		t.Errorf("prompt_preview = %v, want hello", reqs[0]["prompt_preview"])
	}
	if !reflect.DeepEqual(reqs[0]["full_request_json"], reqJSON) {
		t.Errorf("full_request_json = %v, want the request payload", reqs[0]["full_request_json"])
	}
	if reqs[0]["slot_id"] != -1 {
		t.Errorf("slot_id = %v, want -1 default", reqs[0]["slot_id"])
	}
	arrivalTS := reqs[0]["timestamp"]
	if m.GetTotalCount() != 1 {
		t.Errorf("GetTotalCount = %d, want 1", m.GetTotalCount())
	}
	if perf := m.GetPerformance("", "", ""); perf["total_requests"] != 0 {
		t.Errorf("performance total before completion = %v, want 0 (incomplete excluded)", perf["total_requests"])
	}

	m.Record(map[string]any{
		"request_id": "r1",
		"model":      "M",
		"backend":    "B",
		"status":     "complete",
		"cache_hit":  true,
		"recompute":  false,
		"saved":      true,
		"restored":   true,
		"latency_ms": 123.4,
	})

	reqs = m.GetRequests(10, 0)
	if len(reqs) != 1 {
		t.Fatalf("after completion: %d requests, want 1 (in-place update)", len(reqs))
	}
	if reqs[0]["status"] != "complete" {
		t.Errorf("status = %v, want complete", reqs[0]["status"])
	}
	if reqs[0]["prompt_preview"] != "hello" {
		t.Errorf("prompt_preview = %v, want preserved hello", reqs[0]["prompt_preview"])
	}
	if !reflect.DeepEqual(reqs[0]["full_request_json"], reqJSON) {
		t.Errorf("full_request_json not preserved")
	}
	if reqs[0]["latency_ms"] != 123.4 {
		t.Errorf("latency_ms = %v, want 123.4", reqs[0]["latency_ms"])
	}
	if reqs[0]["timestamp"] != arrivalTS {
		t.Errorf("timestamp changed: %v -> %v (arrival ts must be preserved)", arrivalTS, reqs[0]["timestamp"])
	}

	perf := m.GetPerformance("", "", "")
	if perf["total_requests"] != 1 || perf["cache_hits"] != 1 || perf["cache_misses"] != 0 {
		t.Errorf("perf = %v, want total 1 / hits 1 / misses 0", perf)
	}
	if perf["cache_hit_rate"] != 1.0 || perf["save_rate"] != 1.0 || perf["restore_success_rate"] != 1.0 {
		t.Errorf("rates = hit %v / save %v / restore %v, want 1/1/1", perf["cache_hit_rate"], perf["save_rate"], perf["restore_success_rate"])
	}

	ByID := m.GetRequestByID("r1")
	if ByID == nil {
		t.Fatal("GetRequestByID(r1) = nil")
	}
	ByID["model"] = "HACKED"
	again := m.GetRequestByID("r1")
	if again["model"] != "M" {
		t.Errorf("GetRequestByID returned a shared reference: model = %v", again["model"])
	}
	if m.GetRequestByID("nope") != nil {
		t.Error("GetRequestByID(unknown) = non-nil, want nil")
	}
}

func TestMetricsBasicCounters(t *testing.T) {
	m := proxycache.NewMetricsCollector(200)
	m.Record(map[string]any{"request_id": "r1", "model": "M1", "backend": "B1", "status": "complete", "cache_hit": false, "saved": false, "restored": false})
	m.Record(map[string]any{"request_id": "r2", "model": "M1", "backend": "B1", "status": "complete", "cache_hit": true, "recompute": true, "saved": true, "restored": false})

	perf := m.GetPerformance("", "", "")
	if perf["total_requests"] != 2 || perf["cache_hits"] != 1 || perf["cache_misses"] != 1 {
		t.Errorf("perf totals = %v/%v/%v, want 2/1/1", perf["total_requests"], perf["cache_hits"], perf["cache_misses"])
	}
	if perf["cache_recomputes"] != 1 || perf["cache_saved"] != 1 || perf["cache_save_skipped"] != 1 {
		t.Errorf("perf saved/skipped = %v/%v, want 1/1", perf["cache_saved"], perf["cache_save_skipped"])
	}
	if perf["cache_hit_rate"] != 0.5 || perf["cache_mispredict_rate"] != 1.0 {
		t.Errorf("hit/mispredict = %v/%v, want 0.5/1.0", perf["cache_hit_rate"], perf["cache_mispredict_rate"])
	}
	if perf["cache_utility_rate"] != 0.0 {
		t.Errorf("utility = %v, want 0.0 (hit recomputed)", perf["cache_utility_rate"])
	}
	if perf["save_rate"] != 0.5 || perf["save_skip_rate"] != 0.5 {
		t.Errorf("save rates = %v/%v, want 0.5/0.5", perf["save_rate"], perf["save_skip_rate"])
	}
	if perf["restore_success_rate"] != 0.0 {
		t.Errorf("restore rate = %v, want 0.0 (two failures)", perf["restore_success_rate"])
	}

	byModel := m.GetPerformance("M1", "", "")
	if byModel["total_requests"] != 2 || byModel["cache_hits"] != 1 {
		t.Errorf("model filter = %v, want total 2 / hits 1", byModel)
	}
	byBackend := m.GetPerformance("", "B1", "")
	if byBackend["total_requests"] != 2 || byBackend["cache_misses"] != 1 {
		t.Errorf("backend filter = %v, want total 2 / misses 1", byBackend)
	}
	if other := m.GetPerformance("M9", "", ""); other["total_requests"] != 0 {
		t.Errorf("unknown model filter = %v, want total 0", other)
	}

	reqs := m.GetRequests(10, 0)
	if len(reqs) != 2 {
		t.Fatalf("GetRequests len = %d, want 2", len(reqs))
	}
	if reqs[0]["request_id"] != "r2" || reqs[1]["request_id"] != "r1" {
		t.Errorf("order = %v, %v, want newest first (r2, r1)", reqs[0]["request_id"], reqs[1]["request_id"])
	}
	if one := m.GetRequests(1, 0); len(one) != 1 || one[0]["request_id"] != "r2" {
		t.Errorf("limit 1 = %v, want [r2]", one)
	}
	if page := m.GetRequests(1, 1); len(page) != 1 || page[0]["request_id"] != "r1" {
		t.Errorf("offset 1 = %v, want [r1]", page)
	}
	if over := m.GetRequests(1, 100); len(over) != 0 {
		t.Errorf("offset beyond range = %v, want empty", over)
	}
	if neg := m.GetRequests(1, -5); len(neg) != 1 || neg[0]["request_id"] != "r2" {
		t.Errorf("negative offset = %v, want [r2]", neg)
	}
}

func TestMetricsRingOverflow(t *testing.T) {
	m := proxycache.NewMetricsCollector(5)
	for i := 1; i <= 8; i++ {
		m.Record(map[string]any{
			"request_id": "r" + itoa(i),
			"model":      "M",
			"backend":    "B",
			"status":     "complete",
			"cache_hit":  i%2 == 0,
			"saved":      true,
			"restored":   true,
		})
	}
	if m.GetTotalCount() != 5 {
		t.Errorf("GetTotalCount = %d, want 5 (retention)", m.GetTotalCount())
	}
	reqs := m.GetRequests(10, 0)
	if len(reqs) != 5 {
		t.Fatalf("requests = %d, want 5", len(reqs))
	}
	if reqs[0]["request_id"] != "r8" || reqs[4]["request_id"] != "r4" {
		t.Errorf("overflow window = %v..%v, want r8..r4", reqs[0]["request_id"], reqs[4]["request_id"])
	}
	if m.GetRequestByID("r1") != nil {
		t.Error("evicted r1 still found by ID")
	}
	if m.GetRequestByID("r4") == nil {
		t.Error("r4 (oldest retained) not found by ID")
	}

	m.Record(map[string]any{"request_id": "r1", "model": "M", "backend": "B", "status": "complete", "cache_hit": false, "saved": true, "restored": true})
	if m.GetTotalCount() != 5 {
		t.Errorf("GetTotalCount after re-record = %d, want 5", m.GetTotalCount())
	}
	if m.GetRequestByID("r1") == nil {
		t.Error("re-recorded r1 not found by ID")
	}
}

func TestMetricsEvents(t *testing.T) {
	m := proxycache.NewMetricsCollector(200)
	m.Record(map[string]any{"event": "liveness", "detail": "up"})
	m.Record(map[string]any{"event": "slot_refresh", "backend": "B1"})
	m.Record(map[string]any{"request_id": "r1", "model": "M", "backend": "B", "status": "complete", "cache_hit": false, "saved": true, "restored": true})

	if n := m.GetTotalCount(); n != 1 {
		t.Errorf("GetTotalCount = %d, want 1 (events excluded)", n)
	}
	reqs := m.GetRequests(10, 0)
	if len(reqs) != 1 || reqs[0]["request_id"] != "r1" {
		t.Errorf("GetRequests = %v, want only r1", reqs)
	}

	ev := m.GetEvents("", 10)
	if len(ev) != 2 {
		t.Fatalf("GetEvents len = %d, want 2", len(ev))
	}
	if ev[0]["event"] != "slot_refresh" {
		t.Errorf("newest event = %v, want slot_refresh", ev[0]["event"])
	}
	if _, ok := ev[0]["timestamp"].(int64); !ok {
		t.Errorf("event timestamp type = %T, want int64", ev[0]["timestamp"])
	}
	filtered := m.GetEvents("liveness", 10)
	if len(filtered) != 1 || filtered[0]["detail"] != "up" {
		t.Errorf("GetEvents(liveness) = %v, want 1 entry with detail up", filtered)
	}
	if noLimit := m.GetEvents("liveness", 0); len(noLimit) != 1 {
		t.Errorf("GetEvents limit 0 = %d, want 1 (no limit)", len(noLimit))
	}

	tl := m.GetTimeline(10)
	if len(tl) != 3 {
		t.Fatalf("GetTimeline len = %d, want 3 (requests + events)", len(tl))
	}
	if tl[0]["request_id"] != "r1" {
		t.Errorf("timeline newest = %v, want r1", tl[0]["request_id"])
	}
	if two := m.GetTimeline(2); len(two) != 2 {
		t.Errorf("GetTimeline(2) len = %d, want 2", len(two))
	}
}

func TestMetricsPerformanceLatency(t *testing.T) {
	m := proxycache.NewMetricsCollector(200)
	for i, l := range []float64{300, 100, 400, 200} {
		m.Record(map[string]any{
			"request_id": "r" + itoa(i+1),
			"model":      "M",
			"backend":    "B",
			"status":     "complete",
			"cache_hit":  i%2 == 0,
			"saved":      true,
			"restored":   true,
			"latency_ms": l,
		})
	}
	perf := m.GetPerformance("", "", "")
	lat, _ := perf["latency"].(map[string]float64)
	if lat == nil {
		t.Fatalf("latency = %v, want map", perf["latency"])
	}
	if lat["avg_ms"] != 250 || lat["p50_ms"] != 300 || lat["p95_ms"] != 400 || lat["p99_ms"] != 400 {
		t.Errorf("latency = %v, want avg 250 / p50 300 / p95 400 / p99 400", lat)
	}

	m2 := proxycache.NewMetricsCollector(200)
	empty := m2.GetPerformance("", "", "")
	lat2, _ := empty["latency"].(map[string]float64)
	if lat2["avg_ms"] != 0 || lat2["p95_ms"] != 0 {
		t.Errorf("empty latency = %v, want zeros", lat2)
	}
}

func TestMetricsPerformanceReqType(t *testing.T) {
	m := proxycache.NewMetricsCollector(200)
	m.Record(map[string]any{"request_id": "r1", "model": "M", "backend": "B", "status": "complete", "cache_hit": true, "saved": true, "restored": true, "summarization_score": 0.5})
	m.Record(map[string]any{"request_id": "r2", "model": "M", "backend": "B", "status": "complete", "cache_hit": false, "saved": false, "restored": true, "summarization_score": 0.2})
	m.Record(map[string]any{"request_id": "r3", "model": "M", "backend": "B", "status": "complete", "cache_hit": true, "recompute": true, "saved": true, "restored": true, "summarization_score": 0.4})

	sum := m.GetPerformance("", "", "summarization")
	if sum["total_requests"] != 2 || sum["cache_hits"] != 2 || sum["cache_misses"] != 0 {
		t.Errorf("summarization = %v/%v/%v, want 2/2/0", sum["total_requests"], sum["cache_hits"], sum["cache_misses"])
	}
	if sum["cache_recomputes"] != 1 || sum["cache_saved"] != 2 || sum["cache_save_skipped"] != 0 {
		t.Errorf("summarization saved = %v/%v, want 1 recompute, 2 saved, 0 skipped", sum["cache_recomputes"], sum["cache_saved"])
	}
	if sum["cache_hit_rate"] != 1.0 || sum["cache_mispredict_rate"] != 0.5 || sum["cache_utility_rate"] != 0.5 {
		t.Errorf("summarization rates = %v/%v/%v, want 1/0.5/0.5", sum["cache_hit_rate"], sum["cache_mispredict_rate"], sum["cache_utility_rate"])
	}
	if sum["save_rate"] != 1.0 {
		t.Errorf("summarization save_rate = %v, want 1.0", sum["save_rate"])
	}

	conv := m.GetPerformance("", "", "conversation")
	if conv["total_requests"] != 1 || conv["cache_hits"] != 0 || conv["cache_misses"] != 1 {
		t.Errorf("conversation = %v/%v/%v, want 1/0/1", conv["total_requests"], conv["cache_hits"], conv["cache_misses"])
	}
	if conv["cache_saved"] != 0 || conv["cache_save_skipped"] != 1 {
		t.Errorf("conversation saved = %v/%v, want 0/1", conv["cache_saved"], conv["cache_save_skipped"])
	}
	if conv["save_rate"] != 0.0 || conv["save_skip_rate"] != 1.0 {
		t.Errorf("conversation save rates = %v/%v, want 0/1", conv["save_rate"], conv["save_skip_rate"])
	}
	if conv["restore_success_rate"] != 1.0 {
		t.Errorf("conversation restore rate = %v, want 1.0", conv["restore_success_rate"])
	}
}

func TestGetRequestsSummary(t *testing.T) {
	m := proxycache.NewMetricsCollector(200)
	m.Record(map[string]any{
		"request_id": "r1", "model": "M", "backend": "B", "status": "complete",
		"cache_hit": true, "saved": true, "restored": true, "latency_ms": 50.0,
		"n_tokens": 123, "cached_tokens": 100, "cache_size_bytes": 999,
		"routing_reason": "cache_hit", "summarization_score": 0.5,
	})
	m.Record(map[string]any{
		"request_id": "r2", "model": "M", "backend": "B", "status": "incomplete",
	})

	sum := m.GetRequestsSummary(10, 0)
	if len(sum) != 2 {
		t.Fatalf("summary len = %d, want 2", len(sum))
	}
	s2 := sum[0]
	if s2["request_id"] != "r2" || s2["status"] != "incomplete" {
		t.Errorf("summary[0] = %v/%v, want r2/incomplete (newest first)", s2["request_id"], s2["status"])
	}
	if s2["n_tokens"] != nil || s2["cache_size_bytes"] != nil || s2["routing_reason"] != nil {
		t.Errorf("summary[0] missing fields = %v/%v/%v, want nil", s2["n_tokens"], s2["cache_size_bytes"], s2["routing_reason"])
	}
	if s2["slot_id"] != -1 {
		t.Errorf("summary[0] slot_id = %v, want -1", s2["slot_id"])
	}
	if _, hasJSON := s2["full_request_json"]; hasJSON {
		t.Error("summary contains full_request_json, want stripped")
	}

	s1 := sum[1]
	if s1["request_id"] != "r1" || s1["status"] != "complete" {
		t.Errorf("summary[1] = %v/%v, want r1/complete", s1["request_id"], s1["status"])
	}
	if s1["n_tokens"] != 123 || s1["cached_tokens"] != 100 || s1["cache_size_bytes"] != 999 {
		t.Errorf("summary[1] token fields = %v/%v/%v", s1["n_tokens"], s1["cached_tokens"], s1["cache_size_bytes"])
	}
	if s1["routing_reason"] != "cache_hit" || s1["summarization_score"] != 0.5 {
		t.Errorf("summary[1] routing = %v/%v", s1["routing_reason"], s1["summarization_score"])
	}
}

func TestGetSummary(t *testing.T) {
	m := proxycache.NewMetricsCollector(200)
	m.Record(map[string]any{"request_id": "r1", "model": "M", "backend": "B", "status": "incomplete"})
	m.Record(map[string]any{"request_id": "r2", "model": "M", "backend": "B", "status": "complete", "cache_hit": true, "saved": true, "restored": true, "latency_ms": 100.0, "summarization_score": 0.5})
	m.Record(map[string]any{"request_id": "r3", "model": "M", "backend": "B", "status": "complete", "cache_hit": false, "saved": false, "restored": true, "latency_ms": 200.0, "summarization_score": 0.1})

	s := m.GetSummary()
	if s["incomplete_count"] != 1 {
		t.Errorf("incomplete_count = %v, want 1", s["incomplete_count"])
	}
	perf, _ := s["performance"].(map[string]any)
	if perf["total_requests"] != 2 {
		t.Errorf("performance total = %v, want 2 (complete only)", perf["total_requests"])
	}
	full, _ := s["requests"].([]map[string]any)
	if len(full) != 3 {
		t.Errorf("requests len = %d, want 3 (incomplete included)", len(full))
	}
	summary, _ := s["requests_summary"].([]map[string]any)
	if len(summary) != 3 {
		t.Errorf("requests_summary len = %d, want 3", len(summary))
	}
	up, ok := s["uptime_seconds"].(float64)
	if !ok || up < 0 {
		t.Errorf("uptime_seconds = %v, want >= 0", s["uptime_seconds"])
	}

	sum, _ := s["summarization"].(map[string]any)
	if sum["total"] != 1 || sum["conversation_total"] != 1 {
		t.Errorf("summarization buckets = %v/%v, want 1/1", sum["total"], sum["conversation_total"])
	}
	sumLat, _ := sum["latency"].(map[string]float64)
	if sumLat["avg_ms"] != 100 {
		t.Errorf("summarization latency = %v, want avg 100", sumLat)
	}
	convLat, _ := sum["conversation_latency"].(map[string]float64)
	if convLat["avg_ms"] != 200 {
		t.Errorf("conversation latency = %v, want avg 200", convLat)
	}
}
