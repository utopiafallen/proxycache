package tests

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"proxycache"
	"reflect"
	"strings"
	"testing"
	"time"
)

func blk(prefix string, n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = prefix + string(rune('a'+i))
	}
	return out
}

func withTempMetaDir(t *testing.T) string {
	t.Helper()
	old := proxycache.MetaDir
	proxycache.MetaDir = t.TempDir()
	t.Cleanup(func() { proxycache.MetaDir = old })
	return proxycache.MetaDir
}

func withTestBackend(t *testing.T, cfg []map[string]any) *proxycache.BackendManager {
	t.Helper()
	old := proxycache.GetBackendManager()
	bm := proxycache.NewBackendManager(cfg)
	proxycache.SetBackendManager(bm)
	t.Cleanup(func() {
		bm.Close()
		proxycache.SetBackendManager(old)
	})
	return bm
}

func withEMACfg(t *testing.T, alpha, minT, initial, maxT float64) {
	t.Helper()
	oldA, oldMin, oldI, oldMax := proxycache.CacheHitWaitEMAAlpha, proxycache.CacheHitWaitEMAMinT, proxycache.CacheHitWaitEMAInitialT, proxycache.CacheHitWaitEMAMaxT
	proxycache.CacheHitWaitEMAAlpha, proxycache.CacheHitWaitEMAMinT, proxycache.CacheHitWaitEMAInitialT, proxycache.CacheHitWaitEMAMaxT = alpha, minT, initial, maxT
	t.Cleanup(func() {
		proxycache.CacheHitWaitEMAAlpha, proxycache.CacheHitWaitEMAMinT, proxycache.CacheHitWaitEMAInitialT, proxycache.CacheHitWaitEMAMaxT = oldA, oldMin, oldI, oldMax
	})
}

func backendKeyFromURL(u string) string {
	raw := strings.TrimRight(u, "/")
	if i := strings.LastIndex(raw, "://"); i >= 0 {
		raw = raw[i+3:]
	}
	return proxycache.SanitizeBackendDir(raw)
}

func TestSlotManagerPerModelPools(t *testing.T) {
	bsm := proxycache.NewBackendSlotManager("127.0.0.1-8000")
	if pool := bsm.GetPool("Unknown"); pool != nil {
		t.Errorf("GetPool(Unknown) = %v, want nil", pool)
	}
	bsm.EnsurePool("ModelA", 1)
	bsm.EnsurePool("ModelB", 2)
	if got := bsm.GetPool("ModelA"); !reflect.DeepEqual(got, []int{0}) {
		t.Errorf("GetPool(ModelA) = %v, want [0]", got)
	}
	if got := bsm.GetPool("ModelB"); !reflect.DeepEqual(got, []int{0, 1}) {
		t.Errorf("GetPool(ModelB) = %v, want [0 1]", got)
	}
	if got := bsm.GetPool("Unknown"); got != nil {
		t.Errorf("GetPool(Unknown) = %v, want nil", got)
	}
}

func TestSlotManagerMultipleBackends(t *testing.T) {
	sm := proxycache.NewSlotManager()
	if sm.HasBackend("B1") {
		t.Error("HasBackend(B1) before Get = true, want false")
	}
	b1 := sm.Get("B1")
	b2 := sm.Get("B2")
	if b1 == b2 {
		t.Error("Get returned the same manager for different backends")
	}
	if sm.Get("B1") != b1 {
		t.Error("Get(B1) is not stable")
	}
	if !sm.HasBackend("B1") || !sm.HasBackend("B2") || sm.HasBackend("B3") {
		t.Errorf("HasBackend = %v/%v/%v, want true/true/false", sm.HasBackend("B1"), sm.HasBackend("B2"), sm.HasBackend("B3"))
	}
	if got := sm.Backends(); !reflect.DeepEqual(got, []string{"B1", "B2"}) {
		t.Errorf("Backends() = %v, want [B1 B2]", got)
	}
	states := sm.AllKVStates()
	if len(states) != 0 {
		t.Errorf("AllKVStates() = %v, want empty", states)
	}
	b1.SetKVState(0, blk("a", 3))
	states = sm.AllKVStates()
	if !reflect.DeepEqual(states[proxycache.BackendSlotKey{BackendID: "B1", SlotID: 0}], blk("a", 3)) {
		t.Errorf("AllKVStates()[{B1 0}] = %v, want 3 blocks", states[proxycache.BackendSlotKey{BackendID: "B1", SlotID: 0}])
	}
	if len(states) != 1 {
		t.Errorf("AllKVStates() has %d entries, want 1", len(states))
	}
}

func TestTryAcquireAndRelease(t *testing.T) {
	bsm := proxycache.NewBackendSlotManager("127.0.0.1-8000")
	bsm.EnsurePool("M", 2)
	if got := bsm.TryAcquire("M"); got != 0 {
		t.Errorf("first TryAcquire = %d, want 0", got)
	}
	if got := bsm.TryAcquire("M"); got != 1 {
		t.Errorf("second TryAcquire = %d, want 1", got)
	}
	if got := bsm.TryAcquire("M"); got != -1 {
		t.Errorf("third TryAcquire = %d, want -1 (all in use)", got)
	}
	if got := bsm.CountInUse("M"); got != 2 {
		t.Errorf("CountInUse = %d, want 2", got)
	}
	dur, ok := bsm.Release(0)
	if !ok || dur < 0 {
		t.Errorf("Release(0) = (%v, %v), want (>=0, true)", dur, ok)
	}
	if got := bsm.CountInUse("M"); got != 1 {
		t.Errorf("CountInUse after release = %d, want 1", got)
	}
	if got := bsm.TryAcquire("M"); got != 0 {
		t.Errorf("TryAcquire after release = %d, want 0 (first free ascending)", got)
	}
	if _, ok := bsm.Release(99); ok {
		t.Error("Release(99) = true, want false (never acquired)")
	}
	if got := bsm.TryAcquire("Unknown"); got != -1 {
		t.Errorf("TryAcquire(Unknown) = %d, want -1", got)
	}
}

func TestPoolResize(t *testing.T) {
	bsm := proxycache.NewBackendSlotManager("127.0.0.1-8000")
	bsm.EnsurePool("M", 1)
	bsm.EnsurePool("M", 3)
	if got := bsm.GetPool("M"); !reflect.DeepEqual(got, []int{0, 1, 2}) {
		t.Errorf("pool after resize up = %v, want [0 1 2]", got)
	}
	if got := bsm.TryAcquire("M"); got != 0 {
		t.Errorf("TryAcquire = %d, want 0", got)
	}
	if got := bsm.TryAcquire("M"); got != 1 {
		t.Errorf("TryAcquire = %d, want 1", got)
	}
	if got := bsm.TryAcquire("M"); got != 2 {
		t.Errorf("TryAcquire = %d, want 2", got)
	}

	bsm2 := proxycache.NewBackendSlotManager("127.0.0.1-8000")
	bsm2.EnsurePool("M", 3)
	bsm2.SetKVState(2, blk("a", 3))
	bsm2.MarkSaveSkipped(2, &proxycache.SaveSkipEntry{Key: "k"})
	bsm2.EnsurePool("M", 1)
	if got := bsm2.GetPool("M"); !reflect.DeepEqual(got, []int{0}) {
		t.Errorf("pool after resize down = %v, want [0]", got)
	}
	if got := bsm2.GetKVState(2); got != nil {
		t.Errorf("KV state of freed slot 2 = %v, want nil (cleaned up)", got)
	}
	if got := bsm2.FlushSaveSkipped(2); got != nil {
		t.Errorf("skipped save of freed slot 2 = %v, want nil", got)
	}

	bsm3 := proxycache.NewBackendSlotManager("127.0.0.1-8000")
	bsm3.EnsurePool("M", 3)
	bsm3.SetKVState(2, blk("a", 3))
	for i := 0; i < 3; i++ {
		if s := bsm3.TryAcquire("M"); s != i {
			t.Fatalf("TryAcquire = %d, want %d", s, i)
		}
	}
	bsm3.EnsurePool("M", 1)
	if got := bsm3.GetKVState(2); !reflect.DeepEqual(got, blk("a", 3)) {
		t.Errorf("KV state of busy slot 2 after shrink = %v, want preserved", got)
	}
	if got := bsm3.CountInUse("M"); got != 1 {
		t.Errorf("CountInUse after shrink = %d, want 1 (only slot 0 in pool)", got)
	}
}

func TestGSlotAndCacheHitTypes(t *testing.T) {
	g := proxycache.GSlot{ModelName: "M", BackendID: "127.0.0.1-8000", SlotID: 2}
	if g.ModelName != "M" || g.BackendID != "127.0.0.1-8000" || g.SlotID != 2 {
		t.Errorf("GSlot fields = %+v", g)
	}
	if string(proxycache.CacheHitDiskRestore) != "disk_restore" || string(proxycache.CacheHitSkip) != "skip" {
		t.Errorf("CacheHitType constants = %q/%q", proxycache.CacheHitDiskRestore, proxycache.CacheHitSkip)
	}
}

func TestShouldSkipRestore(t *testing.T) {
	oldTh, oldDiff := proxycache.KVCacheSkipThreshold, proxycache.KVCacheSkipMaxBlockDiff
	proxycache.KVCacheSkipThreshold, proxycache.KVCacheSkipMaxBlockDiff = 0.9, 0.1
	t.Cleanup(func() {
		proxycache.KVCacheSkipThreshold, proxycache.KVCacheSkipMaxBlockDiff = oldTh, oldDiff
	})

	cases := []struct {
		name       string
		setup      func(b *proxycache.BackendSlotManager)
		req        []string
		prev       []string
		prevPassed bool
		want       bool
	}{
		{
			name:  "no_tracked_state",
			setup: func(b *proxycache.BackendSlotManager) { b.EnsurePool("M", 1) },
			req:   blk("a", 5),
			prev:  nil,
			want:  false,
		},
		{
			name:       "fresh_explicit_nil",
			setup:      func(b *proxycache.BackendSlotManager) { b.EnsurePool("M", 1) },
			req:        blk("a", 5),
			prev:       nil,
			prevPassed: true,
			want:       false,
		},
		{
			name:       "perfect_match",
			setup:      func(b *proxycache.BackendSlotManager) { b.EnsurePool("M", 1) },
			req:        blk("a", 5),
			prev:       blk("a", 5),
			prevPassed: true,
			want:       true,
		},
		{
			name:       "req_longer_high_overlap",
			setup:      func(b *proxycache.BackendSlotManager) { b.EnsurePool("M", 1) },
			req:        append(blk("a", 10), "new"),
			prev:       blk("a", 10),
			prevPassed: true,
			want:       true,
		},
		{
			name:       "low_overlap",
			setup:      func(b *proxycache.BackendSlotManager) { b.EnsurePool("M", 1) },
			req:        append(blk("a", 8), "x", "y", "z"),
			prev:       blk("a", 10),
			prevPassed: true,
			want:       false,
		},
		{
			name:       "zero_lcp",
			setup:      func(b *proxycache.BackendSlotManager) { b.EnsurePool("M", 1) },
			req:        blk("w", 4),
			prev:       blk("a", 4),
			prevPassed: true,
			want:       false,
		},
		{
			name:       "req_shorter_than_prev",
			setup:      func(b *proxycache.BackendSlotManager) { b.EnsurePool("M", 1) },
			req:        blk("a", 4),
			prev:       blk("a", 5),
			prevPassed: true,
			want:       false,
		},
		{
			name:       "diff_too_large",
			setup:      func(b *proxycache.BackendSlotManager) { b.EnsurePool("M", 1) },
			req:        append(blk("a", 10), "x", "y", "z"),
			prev:       blk("a", 10),
			prevPassed: true,
			want:       false,
		},
		{
			name:       "multi_slot_pool",
			setup:      func(b *proxycache.BackendSlotManager) { b.EnsurePool("M", 2) },
			req:        blk("a", 5),
			prev:       blk("a", 5),
			prevPassed: true,
			want:       false,
		},
		{
			name:  "kv_state_fallback",
			setup: func(b *proxycache.BackendSlotManager) { b.EnsurePool("M", 1); b.SetKVState(0, blk("a", 5)) },
			req:   blk("a", 5),
			prev:  nil,
			want:  true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			bsm := proxycache.NewBackendSlotManager("127.0.0.1-8000")
			tc.setup(bsm)
			got := bsm.ShouldSkipRestore(0, tc.req, tc.prev, tc.prevPassed)
			if got != tc.want {
				t.Errorf("ShouldSkipRestore = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestSlotDurationEMA(t *testing.T) {
	withEMACfg(t, 0.2, 10, 30, 300)
	bsm := proxycache.NewBackendSlotManager("127.0.0.1-8000")
	bsm.EnsurePool("M", 1)
	if got := bsm.GetSlotDurationEMA(); got != 30 {
		t.Errorf("initial EMA = %v, want 30", got)
	}

	if s := bsm.TryAcquire("M"); s != 0 {
		t.Fatalf("TryAcquire = %d, want 0", s)
	}
	bsm.PoolMu.Lock()
	bsm.SlotAcquiredAt[0] = proxycache.NowFloat() - 5
	bsm.PoolMu.Unlock()
	dur, ok := bsm.Release(0)
	if !ok {
		t.Fatal("Release(0) = false, want true")
	}
	if dur < 4.9 || dur > 6.0 {
		t.Errorf("duration = %v, want ~5", dur)
	}
	if got := bsm.GetSlotDurationEMA(); mathAbs(got-25) > 0.05 {
		t.Errorf("EMA after 5s release = %v, want ~25 (0.2*5 + 0.8*30)", got)
	}

	if s := bsm.TryAcquire("M"); s != 0 {
		t.Fatalf("TryAcquire = %d, want 0", s)
	}
	bsm.PoolMu.Lock()
	bsm.SlotAcquiredAt[0] = proxycache.NowFloat() - 5000
	bsm.PoolMu.Unlock()
	bsm.Release(0)
	if got := bsm.GetSlotDurationEMA(); got != 300 {
		t.Errorf("EMA after 5000s release = %v, want 300 (max clamp)", got)
	}

	bsm2 := proxycache.NewBackendSlotManager("127.0.0.1-8001")
	bsm2.EnsurePool("M", 1)
	withEMACfg(t, 0.9, 10, 30, 300)
	if s := bsm2.TryAcquire("M"); s != 0 {
		t.Fatalf("TryAcquire = %d, want 0", s)
	}
	bsm2.Release(0)
	if got := bsm2.GetSlotDurationEMA(); got != 10 {
		t.Errorf("EMA near-zero release with alpha 0.9 = %v, want 10 (min clamp)", got)
	}
}

func mathAbs(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

func TestKVStateAndInvalidate(t *testing.T) {
	bsm := proxycache.NewBackendSlotManager("127.0.0.1-8000")
	if got := bsm.GetKVState(0); got != nil {
		t.Errorf("GetKVState(0) before set = %v, want nil", got)
	}
	bsm.SetKVState(0, blk("a", 3))
	bsm.SetKVState(1, blk("b", 2))
	if got := bsm.GetKVState(0); !reflect.DeepEqual(got, blk("a", 3)) {
		t.Errorf("GetKVState(0) = %v", got)
	}
	states := bsm.GetKVStates()
	if len(states) != 2 {
		t.Errorf("GetKVStates() = %v, want 2 entries", states)
	}
	bsm.Invalidate(0)
	if got := bsm.GetKVState(0); got != nil {
		t.Errorf("GetKVState(0) after Invalidate = %v, want nil", got)
	}
	if got := bsm.GetKVState(1); !reflect.DeepEqual(got, blk("b", 2)) {
		t.Errorf("GetKVState(1) = %v, want unchanged", got)
	}
	bsm.Invalidate(0)
	bsm.Invalidate(42)
}

func TestSaveSkippedMarkFlush(t *testing.T) {
	bsm := proxycache.NewBackendSlotManager("127.0.0.1-8000")
	if got := bsm.FlushSaveSkipped(0); got != nil {
		t.Errorf("FlushSaveSkipped before mark = %v, want nil", got)
	}
	entry := &proxycache.SaveSkipEntry{Key: "k", NTokens: 42, Recompute: true}
	bsm.MarkSaveSkipped(0, entry)
	if got := bsm.FlushSaveSkipped(0); got != entry {
		t.Errorf("FlushSaveSkipped = %v, want the marked entry", got)
	}
	if got := bsm.FlushSaveSkipped(0); got != nil {
		t.Errorf("second FlushSaveSkipped = %v, want nil", got)
	}
}

func TestCacheWaitPending(t *testing.T) {
	sm := proxycache.NewSlotManager()
	if got := sm.GetCacheWaitPending("B1"); got != 0 {
		t.Errorf("GetCacheWaitPending(new) = %d, want 0", got)
	}
	sm.SetCacheWaitPending("B1", 3)
	if got := sm.GetCacheWaitPending("B1"); got != 3 {
		t.Errorf("GetCacheWaitPending = %d, want 3", got)
	}
	sm.SetCacheWaitPending("B1", 0)
	if got := sm.GetCacheWaitPending("B1"); got != 0 {
		t.Errorf("GetCacheWaitPending after reset = %d, want 0", got)
	}
}

func newSaveServer(t *testing.T, nWritten int, fail bool) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("action") == "save" {
			if fail {
				w.WriteHeader(500)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"n_written": ` + itoa(nWritten) + `}`))
			return
		}
		w.WriteHeader(200)
	}))
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	digits := ""
	for n > 0 {
		digits = string(rune('0'+n%10)) + digits
		n /= 10
	}
	return digits
}

func TestSaveAfterSuccess(t *testing.T) {
	withTempMetaDir(t)
	srv := newSaveServer(t, 123, false)
	defer srv.Close()
	key := backendKeyFromURL(srv.URL)
	withTestBackend(t, []map[string]any{{"url": srv.URL, "cache_dir": t.TempDir()}})

	bsm := proxycache.NewBackendSlotManager(key)
	blocks := blk("a", 4)
	ok, size := bsm.SaveAfter("M", 0, "k1", blocks, 100)
	if !ok || size != 123 {
		t.Fatalf("SaveAfter = (%v, %d), want (true, 123)", ok, size)
	}
	if got := bsm.GetKVState(0); !reflect.DeepEqual(got, blocks) {
		t.Errorf("KV state after save = %v, want %v", got, blocks)
	}
	if got := bsm.GetRingSize(); got != 1 {
		t.Errorf("GetRingSize = %d, want 1", got)
	}
	if got := bsm.GetTotalBytes(); got != 123 {
		t.Errorf("GetTotalBytes = %d, want 123", got)
	}
	oldest, newest, okRing := bsm.RingOldestNewestKeys()
	if !okRing || oldest != "k1" || newest != "k1" {
		t.Errorf("RingOldestNewestKeys = (%q, %q, %v), want (k1, k1, true)", oldest, newest, okRing)
	}

	meta := proxycache.GetKVMeta().ReadMeta("k1", key)
	if meta == nil {
		t.Fatal("ReadMeta(k1) = nil, want meta file")
	}
	if meta["n_tokens"] != float64(100) {
		t.Errorf("meta n_tokens = %v, want 100", meta["n_tokens"])
	}
	if meta["wpb"] != float64(proxycache.WordsPerBlock) {
		t.Errorf("meta wpb = %v, want %d", meta["wpb"], proxycache.WordsPerBlock)
	}
	if meta["model_id"] != "M" || meta["backend"] != key {
		t.Errorf("meta model_id/backend = %v/%v, want M/%v", meta["model_id"], meta["backend"], key)
	}
	if meta["cache_size"] != float64(123) {
		t.Errorf("meta cache_size = %v, want 123", meta["cache_size"])
	}
	if meta["recompute_penalty"] != float64(0) {
		t.Errorf("meta recompute_penalty = %v, want 0", meta["recompute_penalty"])
	}
	if lw, _ := meta["last_written"].(float64); lw <= 0 {
		t.Errorf("meta last_written = %v, want > 0", meta["last_written"])
	}
	var gotBlocks []string
	for _, b := range meta["blocks"].([]any) {
		gotBlocks = append(gotBlocks, b.(string))
	}
	if !reflect.DeepEqual(gotBlocks, blocks) {
		t.Errorf("meta blocks = %v, want %v", gotBlocks, blocks)
	}
	path := filepath.Join(proxycache.MetaDir, proxycache.SanitizeBackendDir(key), "k1"+proxycache.MetaSuffix)
	if _, err := os.Stat(path); err != nil {
		t.Errorf("meta file missing at %s: %v", path, err)
	}
}

func TestSaveAfterNetworkFailure(t *testing.T) {
	withTempMetaDir(t)
	withTestBackend(t, []map[string]any{{"url": "http://127.0.0.1:1", "cache_dir": t.TempDir()}})
	key := "127.0.0.1-1"
	bsm := proxycache.NewBackendSlotManager(key)
	blocks := blk("a", 4)
	ok, size := bsm.SaveAfter("M", 0, "k1", blocks, 100)
	if ok || size != 0 {
		t.Errorf("SaveAfter on dead backend = (%v, %d), want (false, 0)", ok, size)
	}
	if got := bsm.GetKVState(0); !reflect.DeepEqual(got, blocks) {
		t.Errorf("KV state after failed save = %v, want %v (still tracked)", got, blocks)
	}
	if got := bsm.GetRingSize(); got != 0 {
		t.Errorf("GetRingSize = %d, want 0", got)
	}
	if got := bsm.GetTotalBytes(); got != 0 {
		t.Errorf("GetTotalBytes = %d, want 0", got)
	}
}

func TestSaveAfterDisabled(t *testing.T) {
	withTempMetaDir(t)
	withTestBackend(t, []map[string]any{{"url": "http://127.0.0.1:1", "cache_max_size_gb": 0}})
	key := "127.0.0.1-1"
	bsm := proxycache.NewBackendSlotManager(key)
	ok, size := bsm.SaveAfter("M", 0, "k1", blk("a", 4), 100)
	if ok || size != 0 {
		t.Errorf("SaveAfter with cache disabled = (%v, %d), want (false, 0)", ok, size)
	}
	if got := bsm.GetKVState(0); got != nil {
		t.Errorf("KV state with cache disabled = %v, want nil (early return)", got)
	}
}

func TestRestore(t *testing.T) {
	withTempMetaDir(t)
	restoreStatus := 200
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("action") == "restore" {
			w.WriteHeader(restoreStatus)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"n_written": 10}`))
	}))
	defer srv.Close()
	key := backendKeyFromURL(srv.URL)
	withTestBackend(t, []map[string]any{{"url": srv.URL, "cache_dir": t.TempDir()}})

	bsm := proxycache.NewBackendSlotManager(key)
	if ok, _ := bsm.SaveAfter("M", 0, "k1", blk("a", 2), 20); !ok || bsm.GetRingSize() != 1 {
		t.Fatal("setup: SaveAfter failed or ring empty")
	}
	bsm.RingMu.Lock()
	oldTS := bsm.CacheRing[0].TS
	bsm.RingMu.Unlock()
	time.Sleep(20 * time.Millisecond)

	if !bsm.Restore(0, "k1", "M", true) {
		t.Error("Restore with touchRing = false, want true")
	}
	bsm.RingMu.Lock()
	newTS := bsm.CacheRing[0].TS
	bsm.RingMu.Unlock()
	if newTS <= oldTS {
		t.Errorf("ring TS not advanced by touch: old=%v new=%v", oldTS, newTS)
	}

	if !bsm.Restore(0, "k1", "M", false) {
		t.Error("Restore with touchRing=false failed, want true")
	}
	bsm.RingMu.Lock()
	untouchedTS := bsm.CacheRing[0].TS
	bsm.RingMu.Unlock()
	if untouchedTS != newTS {
		t.Errorf("ring TS changed with touchRing=false: %v -> %v", newTS, untouchedTS)
	}

	if !bsm.Restore(0, "missing-key", "M", true) {
		t.Error("Restore of key not in ring failed, want true (touch is a no-op)")
	}

	restoreStatus = 500
	if bsm.Restore(0, "k1", "M", true) {
		t.Error("Restore with 500 = true, want false")
	}
}

func TestRestoreTimeout(t *testing.T) {
	withTempMetaDir(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
			return
		case <-time.After(1 * time.Second):
		}
		w.WriteHeader(200)
	}))
	defer srv.Close()
	key := backendKeyFromURL(srv.URL)
	withTestBackend(t, []map[string]any{{"url": srv.URL, "cache_dir": t.TempDir()}})

	old := proxycache.SlotTimeout
	proxycache.SlotTimeout = 0.3
	t.Cleanup(func() { proxycache.SlotTimeout = old })

	bsm := proxycache.NewBackendSlotManager(key)
	if bsm.Restore(0, "k1", "M", true) {
		t.Error("Restore on slow backend = true, want false (SLOT_TIMEOUT)")
	}
}

func TestRingNoEvictionUnderLimit(t *testing.T) {
	withTempMetaDir(t)
	srv := newSaveServer(t, 400, false)
	defer srv.Close()
	key := backendKeyFromURL(srv.URL)
	withTestBackend(t, []map[string]any{{"url": srv.URL, "cache_dir": t.TempDir(), "cache_max_size_gb": 25}})

	bsm := proxycache.NewBackendSlotManager(key)
	for _, k := range []string{"kA", "kB", "kC"} {
		if ok, size := bsm.SaveAfter("M", 0, k, blk("a", 8), 800); !ok || size != 400 {
			t.Fatalf("SaveAfter(%s) = (%v, %d)", k, ok, size)
		}
	}
	if got := bsm.GetRingSize(); got != 3 {
		t.Errorf("GetRingSize = %d, want 3 (under 25GB limit)", got)
	}
	if got := bsm.GetTotalBytes(); got != 1200 {
		t.Errorf("GetTotalBytes = %d, want 1200", got)
	}
}

func saveThreeEntries(t *testing.T, srvURL string, key string) *proxycache.BackendSlotManager {
	t.Helper()
	withTestBackend(t, []map[string]any{{"url": srvURL, "cache_dir": t.TempDir(), "cache_max_size_gb": 25}})
	bsm := proxycache.NewBackendSlotManager(key)
	for _, tc := range []struct {
		key    string
		blocks []string
	}{
		{"kA", blk("a", 8)},
		{"kB", blk("a", 10)},
		{"kC", blk("c", 8)},
	} {
		if ok, _ := bsm.SaveAfter("M", 0, tc.key, tc.blocks, 800); !ok {
			t.Fatalf("setup: SaveAfter(%s) failed", tc.key)
		}
	}
	if bsm.GetRingSize() != 3 {
		t.Fatal("setup: ring size != 3")
	}
	return bsm
}

func TestRingScoredEviction(t *testing.T) {
	withTempMetaDir(t)
	srv := newSaveServer(t, 400, false)
	defer srv.Close()
	key := backendKeyFromURL(srv.URL)
	bsm := saveThreeEntries(t, srv.URL, key)

	now := proxycache.NowFloat()
	bsm.RingMu.Lock()
	bsm.CacheRing[0].TS = now - 100
	bsm.CacheRing[1].TS = now - 50
	bsm.CacheRing[2].TS = now - 10
	bsm.RingMu.Unlock()

	withTestBackend(t, []map[string]any{{"url": srv.URL, "cache_dir": t.TempDir(), "cache_max_size_gb": 0.000001}})
	bsm.RingMu.Lock()
	bsm.EvictIfNeeded()
	bsm.RingMu.Unlock()

	if got := bsm.GetRingSize(); got != 2 {
		t.Fatalf("GetRingSize after eviction = %d, want 2", got)
	}
	oldest, newest, ok := bsm.RingOldestNewestKeys()
	if !ok || oldest != "kB" || newest != "kC" {
		t.Errorf("remaining ring = (%q, %q), want (kB, kC) — oldest shared entry evicted", oldest, newest)
	}
	if got := bsm.GetTotalBytes(); got != 800 {
		t.Errorf("GetTotalBytes = %d, want 800", got)
	}
	if _, err := os.Stat(filepath.Join(proxycache.MetaDir, proxycache.SanitizeBackendDir(key), "kA"+proxycache.MetaSuffix)); !os.IsNotExist(err) {
		t.Error("kA meta file still present after eviction")
	}
	if _, err := os.Stat(filepath.Join(proxycache.MetaDir, proxycache.SanitizeBackendDir(key), "kB"+proxycache.MetaSuffix)); err != nil {
		t.Errorf("kB meta file missing: %v", err)
	}
}

func TestRingStaleUniqueEvicted(t *testing.T) {
	withTempMetaDir(t)
	srv := newSaveServer(t, 400, false)
	defer srv.Close()
	key := backendKeyFromURL(srv.URL)
	bsm := saveThreeEntries(t, srv.URL, key)

	now := proxycache.NowFloat()
	bsm.RingMu.Lock()
	bsm.CacheRing[0].TS = now - 10
	bsm.CacheRing[1].TS = now - 10
	bsm.CacheRing[2].TS = now - 100
	bsm.RingMu.Unlock()

	withTestBackend(t, []map[string]any{{"url": srv.URL, "cache_dir": t.TempDir(), "cache_max_size_gb": 0.000001}})
	bsm.RingMu.Lock()
	bsm.EvictIfNeeded()
	bsm.RingMu.Unlock()

	if got := bsm.GetRingSize(); got != 2 {
		t.Fatalf("GetRingSize after eviction = %d, want 2", got)
	}
	oldest, newest, ok := bsm.RingOldestNewestKeys()
	if !ok || oldest != "kA" || newest != "kB" {
		t.Errorf("remaining ring = (%q, %q), want (kA, kB) — stale unique entry evicted by age", oldest, newest)
	}
}

func TestRingOrphanEviction(t *testing.T) {
	withTempMetaDir(t)
	srv := newSaveServer(t, 400, false)
	defer srv.Close()
	key := backendKeyFromURL(srv.URL)
	bsm := saveThreeEntries(t, srv.URL, key)

	if err := os.Remove(proxycache.GetKVMeta().MetaFilePath("kA", key)); err != nil {
		t.Fatalf("failed to remove kA meta: %v", err)
	}

	withTestBackend(t, []map[string]any{{"url": srv.URL, "cache_dir": t.TempDir(), "cache_max_size_gb": 0.000001}})
	bsm.RingMu.Lock()
	bsm.EvictIfNeeded()
	bsm.RingMu.Unlock()

	if got := bsm.GetRingSize(); got != 2 {
		t.Fatalf("GetRingSize after orphan eviction = %d, want 2", got)
	}
	oldest, newest, ok := bsm.RingOldestNewestKeys()
	if !ok || oldest != "kB" || newest != "kC" {
		t.Errorf("remaining ring = (%q, %q), want (kB, kC) — orphan dropped", oldest, newest)
	}
	if got := bsm.GetTotalBytes(); got != 800 {
		t.Errorf("GetTotalBytes = %d, want 800", got)
	}
}

func TestInitFromDisk(t *testing.T) {
	withTempMetaDir(t)
	withTestBackend(t, []map[string]any{{"url": "http://127.0.0.1:1", "cache_dir": t.TempDir()}})
	key := "127.0.0.1-1"

	proxycache.GetKVMeta().WriteMeta("k1", 50, blk("z", 4), 100, "M", key, 777)
	proxycache.GetKVMeta().WriteMeta("k2", 50, blk("y", 4), 100, "M", key, 0)

	sm := proxycache.NewSlotManager()
	sm.InitFromDisk()
	bsm := sm.Get(key)
	if got := bsm.GetRingSize(); got != 1 {
		t.Errorf("GetRingSize = %d, want 1 (k2 has zero size)", got)
	}
	if got := bsm.GetTotalBytes(); got != 777 {
		t.Errorf("GetTotalBytes = %d, want 777", got)
	}
	oldest, newest, ok := bsm.RingOldestNewestKeys()
	if !ok || oldest != "k1" || newest != "k1" {
		t.Errorf("ring keys = (%q, %q, %v), want (k1, k1, true)", oldest, newest, ok)
	}
	if ts, _, ok := bsm.RingOldestNewest(); !ok || ts <= 0 {
		t.Errorf("ring TS = %v (ok=%v), want > 0", ts, ok)
	}
}

func TestSlotStatus(t *testing.T) {
	bsm := proxycache.NewBackendSlotManager("127.0.0.1-8000")
	bsm.EnsurePool("M", 1)
	if s := bsm.TryAcquire("M"); s != 0 {
		t.Fatalf("TryAcquire = %d, want 0", s)
	}
	bsm.SetKVState(0, blk("a", 3))
	bsm.MarkSaveSkipped(0, &proxycache.SaveSkipEntry{Key: "some-cache-key-12345678"})

	st := bsm.SlotStatus()
	models, _ := st["models"].(map[string]any)
	m, _ := models["M"].(map[string]any)
	slots, _ := m["slots"].(map[string]any)
	s0, _ := slots["0"].(map[string]any)
	if s0["in_use"] != true {
		t.Errorf("slot 0 in_use = %v, want true", s0["in_use"])
	}
	if s0["kv_blocks"] != 3 {
		t.Errorf("slot 0 kv_blocks = %v, want 3", s0["kv_blocks"])
	}
	if s0["last_restore"] != "some-cache-key-1" {
		t.Errorf("slot 0 last_restore = %v, want truncated key", s0["last_restore"])
	}
	lu, _ := s0["last_used"].(float64)
	if lu <= 0 {
		t.Errorf("slot 0 last_used = %v, want > 0", s0["last_used"])
	}
}

func TestRingOldestNewestOrder(t *testing.T) {
	withTempMetaDir(t)
	srv := newSaveServer(t, 10, false)
	defer srv.Close()
	key := backendKeyFromURL(srv.URL)
	withTestBackend(t, []map[string]any{{"url": srv.URL, "cache_dir": t.TempDir(), "cache_max_size_gb": 25}})

	bsm := proxycache.NewBackendSlotManager(key)
	if _, _, ok := bsm.RingOldestNewest(); ok {
		t.Error("RingOldestNewest on empty ring = ok, want false")
	}
	if ok, _ := bsm.SaveAfter("M", 0, "k1", blk("a", 1), 10); !ok {
		t.Fatal("setup: SaveAfter(k1) failed")
	}
	time.Sleep(20 * time.Millisecond)
	if ok, _ := bsm.SaveAfter("M", 0, "k2", blk("b", 1), 10); !ok {
		t.Fatal("setup: SaveAfter(k2) failed")
	}
	oldest, newest, ok := bsm.RingOldestNewestKeys()
	if !ok || oldest != "k1" || newest != "k2" {
		t.Errorf("RingOldestNewestKeys = (%q, %q), want (k1, k2) insertion order", oldest, newest)
	}
	tsOldest, tsNewest, ok := bsm.RingOldestNewest()
	if !ok || tsNewest <= tsOldest {
		t.Errorf("ring TS = (%v, %v), want newest > oldest", tsOldest, tsNewest)
	}
	time.Sleep(20 * time.Millisecond)
	bsm.TouchRing("k1")
	bsm.RingMu.Lock()
	touched := bsm.CacheRing[0].TS
	bsm.RingMu.Unlock()
	if touched <= tsOldest {
		t.Errorf("TouchRing did not advance TS: %v -> %v", tsOldest, touched)
	}
	bsm.TouchRing("no-such-key")
}
