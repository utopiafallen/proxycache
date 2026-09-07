package tests

import (
	"proxycache"
	"testing"
)

func candList(ids ...string) []proxycache.CandidateBackend {
	out := make([]proxycache.CandidateBackend, 0, len(ids))
	for _, id := range ids {
		out = append(out, proxycache.CandidateBackend{BackendID: id, ModelName: "m"})
	}
	return out
}

func pendingOfFrom(m map[string]int) func(string) int {
	return func(be string) int { return m[be] }
}

func TestSelectBackendNoHitRoundRobin(t *testing.T) {
	cands := candList("A", "B", "C")
	pending := map[string]int{}
	// No hit: picks the first candidate at/after startIdx with room.
	if i, reason := proxycache.SelectBackend(cands, nil, pendingOfFrom(pending), 0); i != 0 || reason != "fallback_round_robin" {
		t.Errorf("pick(start=0) = (%d, %s), want (0, fallback_round_robin)", i, reason)
	}
	// Pointer advances the start position.
	if i, _ := proxycache.SelectBackend(cands, nil, pendingOfFrom(pending), 1); i != 1 {
		t.Errorf("pick(start=1) = %d, want 1", i)
	}
	// Full backend is skipped.
	pending["B"] = 5
	if i, _ := proxycache.SelectBackend(cands, nil, pendingOfFrom(pending), 1); i != 2 {
		t.Errorf("pick with B full, start=1 = %d, want 2", i)
	}
	// Wrap-around: C(full) -> A(full) -> B.
	pending["A"], pending["C"] = 5, 5
	delete(pending, "B")
	if i, _ := proxycache.SelectBackend(cands, nil, pendingOfFrom(pending), 2); i != 1 {
		t.Errorf("wrap-around pick = %d, want 1", i)
	}
	// All full.
	pending["B"] = 5
	if i, reason := proxycache.SelectBackend(cands, nil, pendingOfFrom(pending), 0); i != -1 || reason != "all_queues_full" {
		t.Errorf("all full = (%d, %s), want (-1, all_queues_full)", i, reason)
	}
}

func TestSelectBackendNoCandidates(t *testing.T) {
	if i, reason := proxycache.SelectBackend(nil, nil, pendingOfFrom(nil), 0); i != -1 || reason != "no_candidates" {
		t.Errorf("empty cands = (%d, %s), want (-1, no_candidates)", i, reason)
	}
}

func TestSelectBackendCacheHit(t *testing.T) {
	cands := candList("A", "B", "C")
	hit := &proxycache.HitInfo{Backend: "B", Canonical: "m"}
	pending := map[string]int{}
	// Hit backend wins while its queue depth is below the limit (2).
	for depth := 0; depth < proxycache.CacheHitQueueLimit; depth++ {
		pending["B"] = depth
		if i, reason := proxycache.SelectBackend(cands, hit, pendingOfFrom(pending), 0); i != 1 || reason != "cache_hit" {
			t.Errorf("hit pick (depth=%d) = (%d, %s), want (1, cache_hit)", depth, i, reason)
		}
	}
	// At the limit: fall back to round-robin starting from the first
	// non-hit candidate at/after startIdx (A), skipping the hit backend.
	pending["B"] = proxycache.CacheHitQueueLimit
	if i, _ := proxycache.SelectBackend(cands, hit, pendingOfFrom(pending), 0); i != 0 {
		t.Errorf("fallback start at/after hit = %d, want 0", i)
	}
	// From a later start position the walk continues in order (C).
	if i, _ := proxycache.SelectBackend(cands, hit, pendingOfFrom(pending), 2); i != 2 {
		t.Errorf("fallback from start=2 = %d, want 2", i)
	}
	// All non-hit backends full, hit backend still has room -> second pass takes it.
	pending["A"], pending["C"] = 5, 5
	if i, _ := proxycache.SelectBackend(cands, hit, pendingOfFrom(pending), 0); i != 1 {
		t.Errorf("second-pass hit pick = %d, want 1", i)
	}
	// Hit backend full too -> nothing available.
	pending["B"] = 5
	if i, reason := proxycache.SelectBackend(cands, hit, pendingOfFrom(pending), 0); i != -1 || reason != "all_queues_full" {
		t.Errorf("saturated all = (%d, %s), want (-1, all_queues_full)", i, reason)
	}
}

func TestSelectBackendCacheHitPrefersCanonicalModel(t *testing.T) {
	cands := []proxycache.CandidateBackend{
		{BackendID: "A", ModelName: "m1"},
		{BackendID: "B", ModelName: "m1"},
		{BackendID: "B", ModelName: "m2"},
	}
	hit := &proxycache.HitInfo{Backend: "B", Canonical: "m2"}
	if i, reason := proxycache.SelectBackend(cands, hit, pendingOfFrom(nil), 0); i != 2 || reason != "cache_hit" {
		t.Errorf("canonical model pick = (%d, %s), want (2, cache_hit)", i, reason)
	}
}

func TestSelectBackendHitBackendNotInCandidates(t *testing.T) {
	// Hit backend went down and dropped out of the candidate list: plain RR.
	cands := candList("A", "C")
	hit := &proxycache.HitInfo{Backend: "B", Canonical: "m"}
	if i, reason := proxycache.SelectBackend(cands, hit, pendingOfFrom(nil), 0); i != 0 || reason != "fallback_round_robin" {
		t.Errorf("missing hit backend pick = (%d, %s), want (0, fallback_round_robin)", i, reason)
	}
}
