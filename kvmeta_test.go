package main

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestWriteMetaReadWrite(t *testing.T) {
	withTempMetaDir(t)
	blocks := []string{"b1", "b2", "b3"}
	kvMeta.WriteMeta("key-abc", 10, blocks, 64, "model-x", "be1", 4096)
	path := filepath.Join(MetaDir, "be1", "key-abc"+MetaSuffix)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("meta file not written at %s", path)
	}
	meta := kvMeta.ReadMeta("key-abc", "be1")
	if meta == nil {
		t.Fatal("ReadMeta returned nil for written key")
	}
	if meta["key"] != "key-abc" {
		t.Errorf("key = %v", meta["key"])
	}
	if meta["model_id"] != "model-x" {
		t.Errorf("model_id = %v", meta["model_id"])
	}
	if meta["backend"] != "be1" {
		t.Errorf("backend = %v", meta["backend"])
	}
	if metaFloat(meta, "wpb") != 64 {
		t.Errorf("wpb = %v", meta["wpb"])
	}
	if metaFloat(meta, "n_tokens") != 10 {
		t.Errorf("n_tokens = %v", meta["n_tokens"])
	}
	if metaFloat(meta, "cache_size") != 4096 {
		t.Errorf("cache_size = %v", meta["cache_size"])
	}
	if metaFloat(meta, "recompute_penalty") != 0 {
		t.Errorf("recompute_penalty = %v", meta["recompute_penalty"])
	}
	if _, ok := meta["last_written"].(float64); !ok {
		t.Errorf("last_written = %v, want float64", meta["last_written"])
	}
	got := kvMeta.GetBlocks("key-abc", "be1")
	if len(got) != 3 || got[0] != "b1" || got[2] != "b3" {
		t.Errorf("GetBlocks = %v", got)
	}
	if kvMeta.ReadMeta("key-abc", "be2") != nil {
		t.Error("ReadMeta found key under wrong backend")
	}
	if kvMeta.GetBlocks("missing-key", "be1") != nil {
		t.Error("GetBlocks for missing key should be nil")
	}
}

func TestListKeysSorted(t *testing.T) {
	withTempMetaDir(t)
	for _, k := range []string{"zeta", "alpha", "mid"} {
		kvMeta.WriteMeta(k, 1, []string{"x"}, 64, "m", "be1", 0)
	}
	kvMeta.WriteMeta("other", 1, []string{"x"}, 64, "m", "be2", 0)
	if err := os.WriteFile(filepath.Join(MetaDir, "be1", "notes.txt"), []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(MetaDir, "be1", "subdir"), 0o755); err != nil {
		t.Fatal(err)
	}
	keys := kvMeta.ListKeys("be1")
	want := []string{"alpha", "mid", "zeta"}
	if len(keys) != len(want) {
		t.Fatalf("ListKeys = %v, want %v", keys, want)
	}
	for i := range want {
		if keys[i] != want[i] {
			t.Fatalf("ListKeys = %v, want %v", keys, want)
		}
	}
	if got := kvMeta.ListKeys("missing-be"); len(got) != 0 {
		t.Errorf("ListKeys(missing) = %v, want empty", got)
	}
}

func TestScanAllMetaNewestFirst(t *testing.T) {
	dir := withTempMetaDir(t)
	kvMeta.WriteMeta("old-key", 1, []string{"a"}, 64, "m", "be1", 0)
	kvMeta.WriteMeta("new-key", 1, []string{"a"}, 64, "m", "be1", 0)
	old := time.Now().Add(-100 * time.Second)
	if err := os.Chtimes(filepath.Join(dir, "be1", "old-key"+MetaSuffix), old, old); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "be1", "bad"+MetaSuffix), []byte("not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	metas := kvMeta.ScanAllMeta("be1")
	if len(metas) != 2 {
		t.Fatalf("ScanAllMeta returned %d entries, want 2 (corrupted skipped)", len(metas))
	}
	if metas[0]["key"] != "new-key" {
		t.Errorf("newest-first order violated: first = %v", metas[0]["key"])
	}
	if metas[1]["key"] != "old-key" {
		t.Errorf("second = %v, want old-key", metas[1]["key"])
	}
	if got := kvMeta.ScanAllMeta("missing-be"); len(got) != 0 {
		t.Errorf("ScanAllMeta(missing) = %v, want empty", got)
	}
}

func TestReconcileMetaRemovesOrphans(t *testing.T) {
	cacheDir := t.TempDir()
	withTempMetaDir(t)
	bm := withTestBackend(t, []map[string]any{{"url": "http://10.0.0.1:8000", "cache_dir": cacheDir}})
	key := "10.0.0.1-8000"
	if key != backendKeyFromURL("http://10.0.0.1:8000") {
		t.Fatalf("key mismatch: %s", key)
	}
	if err := os.WriteFile(filepath.Join(cacheDir, "valid_cache_key"), []byte("cache data"), 0o644); err != nil {
		t.Fatal(err)
	}
	kvMeta.WriteMeta("valid_cache_key", 10, []string{}, 100, "test", key, 8)
	kvMeta.WriteMeta("orphan_cache_key", 10, []string{}, 100, "test", key, 8)
	if err := os.WriteFile(filepath.Join(MetaDir, key, "corrupted_cache_key"+MetaSuffix), []byte("not json {{{"), 0o644); err != nil {
		t.Fatal(err)
	}
	deleted := kvMeta.Reconcile([]string{key})
	if deleted != 2 {
		t.Errorf("deleted = %d, want 2 (orphan + corrupted)", deleted)
	}
	if _, err := os.Stat(filepath.Join(MetaDir, key, "valid_cache_key"+MetaSuffix)); err != nil {
		t.Error("valid meta was deleted")
	}
	for _, k := range []string{"orphan_cache_key", "corrupted_cache_key"} {
		if _, err := os.Stat(filepath.Join(MetaDir, key, k+MetaSuffix)); err == nil {
			t.Errorf("%s meta was not deleted", k)
		}
	}
	_ = bm
}

func TestDeleteMetaFile(t *testing.T) {
	withTempMetaDir(t)
	kvMeta.WriteMeta("k1", 5, []string{"a"}, 64, "m", "be1", 100)
	if !kvMeta.DeleteMetaFile("k1") {
		t.Error("DeleteMetaFile should return true when file exists")
	}
	if kvMeta.ReadMeta("k1", "be1") != nil {
		t.Error("meta still readable after delete")
	}
	if kvMeta.DeleteMetaFile("k1") {
		t.Error("second DeleteMetaFile should return false")
	}
	if kvMeta.DeleteMetaFile("never-existed") {
		t.Error("DeleteMetaFile for missing key should return false")
	}
}

func TestIncrementRecomputePenalty(t *testing.T) {
	withTempMetaDir(t)
	kvMeta.WriteMeta("k", 5, []string{"a"}, 64, "m", "be1", 0)
	kvMeta.IncrementRecomputePenalty("k", "be1")
	kvMeta.IncrementRecomputePenalty("k", "be1")
	meta := kvMeta.ReadMeta("k", "be1")
	if meta == nil {
		t.Fatal("meta disappeared after penalty increments")
	}
	if metaFloat(meta, "recompute_penalty") != 2 {
		t.Errorf("recompute_penalty = %v, want 2", meta["recompute_penalty"])
	}
	if _, ok := meta["last_updated_penalty"]; !ok {
		t.Error("last_updated_penalty missing after increment")
	}
	kvMeta.IncrementRecomputePenalty("nope", "be1")
}

func TestIncrementRecomputePenaltyConcurrent(t *testing.T) {
	withTempMetaDir(t)
	kvMeta.WriteMeta("k", 5, []string{"a"}, 64, "m", "be1", 0)
	const n = 20
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			kvMeta.IncrementRecomputePenalty("k", "be1")
		}()
	}
	wg.Wait()
	meta := kvMeta.ReadMeta("k", "be1")
	if meta == nil {
		t.Fatal("meta disappeared after concurrent penalty increments")
	}
	if metaFloat(meta, "recompute_penalty") != n {
		t.Errorf("recompute_penalty = %v, want %d (lost updates)", meta["recompute_penalty"], n)
	}
}

func TestWriteMetaConcurrentWithReads(t *testing.T) {
	withTempMetaDir(t)
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
			}
			blocks := make([]string, 64)
			for b := range blocks {
				blocks[b] = filepath.Join("blk", "x")
			}
			kvMeta.WriteMeta("hot", 1000, blocks, 100, "m", "be1", i)
		}
	}()
	for i := 0; i < 200; i++ {
		if meta := kvMeta.ReadMeta("hot", "be1"); meta != nil {
			if _, ok := meta["key"]; !ok {
				t.Fatalf("corrupt meta read at iteration %d: %v", i, meta)
			}
		}
		kvMeta.ScanAllMeta("be1")
	}
	close(stop)
	wg.Wait()
	if kvMeta.ReadMeta("hot", "be1") == nil {
		t.Fatal("meta file unreadable after concurrent writes")
	}
}

func TestFindRestoreCandidate(t *testing.T) {
	withTempMetaDir(t)
	req := []string{"blk1", "blk2", "blk3", "blk4", "blk5"}
	kvMeta.WriteMeta("full", 100, req, 100, "m", "be1", 0)
	kvMeta.WriteMeta("partial", 100, []string{"blk1", "blk2", "x"}, 100, "m", "be1", 0)
	kvMeta.WriteMeta("low", 100, []string{"blk1", "x", "x"}, 100, "m", "be1", 0)
	kvMeta.WriteMeta("wrongwpb", 100, req, 200, "m", "be1", 0)
	kvMeta.WriteMeta("toolong", 100, []string{"blk1", "blk2", "blk3", "blk4", "blk5", "blk6"}, 100, "m", "be1", 0)

	k, ratio, ok := kvMeta.FindRestoreCandidate("full", 100, 0.2, req, "be1")
	if !ok || k != "full" || ratio != 1.0 {
		t.Errorf("full = (%q, %v, %v), want (full, 1.0, true)", k, ratio, ok)
	}
	k, ratio, ok = kvMeta.FindRestoreCandidate("partial", 100, 0.2, req, "be1")
	if !ok || k != "partial" || ratio != 0.4 {
		t.Errorf("partial = (%q, %v, %v), want (partial, 0.4, true)", k, ratio, ok)
	}
	if _, _, ok = kvMeta.FindRestoreCandidate("partial", 100, 0.5, req, "be1"); ok {
		t.Error("partial at th=0.5 should be rejected")
	}
	k, ratio, ok = kvMeta.FindRestoreCandidate("low", 100, 0.2, req, "be1")
	if !ok || ratio != 0.2 {
		t.Errorf("low = (%q, %v, %v), want ratio 0.2 at boundary", k, ratio, ok)
	}
	if _, _, ok = kvMeta.FindRestoreCandidate("wrongwpb", 100, 0.2, req, "be1"); ok {
		t.Error("wpb mismatch should be rejected")
	}
	if _, _, ok = kvMeta.FindRestoreCandidate("toolong", 100, 0.2, req, "be1"); ok {
		t.Error("candidate longer than request should be rejected")
	}
	if _, _, ok = kvMeta.FindRestoreCandidate("missing", 100, 0.2, req, "be1"); ok {
		t.Error("missing meta should be rejected")
	}
}

func TestFindBestRestoreCandidate(t *testing.T) {
	dir := withTempMetaDir(t)
	req := []string{"blk1", "blk2", "blk3"}
	kvMeta.WriteMeta("best", 100, req, 100, "m1", "be1", 0)
	kvMeta.WriteMeta("penalized", 100, req, 100, "m1", "be1", 0)
	kvMeta.IncrementRecomputePenalty("penalized", "be1")
	kvMeta.WriteMeta("other-model", 100, req, 100, "m2", "be1", 0)
	kvMeta.WriteMeta("other-backend", 100, req, 100, "m1", "be2", 0)
	kvMeta.WriteMeta("wrong-wpb", 100, req, 200, "m1", "be1", 0)

	k, ratio, ok := kvMeta.FindBestRestoreCandidate(req, 100, 0.2, "m1", "be1")
	if !ok || k != "best" || ratio != 1.0 {
		t.Errorf("best = (%q, %v, %v), want (best, 1.0, true)", k, ratio, ok)
	}
	if err := os.Remove(filepath.Join(dir, "be1", "best"+MetaSuffix)); err != nil {
		t.Fatal(err)
	}
	k, ratio, ok = kvMeta.FindBestRestoreCandidate(req, 100, 0.2, "m1", "be1")
	if !ok || k != "penalized" || ratio != 1.0 {
		t.Errorf("penalized = (%q, %v, %v), want (penalized, 1.0, true)", k, ratio, ok)
	}
	for i := 0; i < 10; i++ {
		kvMeta.IncrementRecomputePenalty("penalized", "be1")
	}
	if _, _, ok = kvMeta.FindBestRestoreCandidate(req, 100, 0.2, "m1", "be1"); ok {
		t.Error("penalty >= 10 zeroes the score and disqualifies the candidate")
	}
	if err := os.Remove(filepath.Join(dir, "be1", "penalized"+MetaSuffix)); err != nil {
		t.Fatal(err)
	}
	if _, _, ok = kvMeta.FindBestRestoreCandidate(req, 100, 0.2, "m1", "be1"); ok {
		t.Error("no eligible metas should return false")
	}
}

func TestCacheHitSelectsBestRatio(t *testing.T) {
	withTempMetaDir(t)
	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	keyA := MetaKey("model-a", tokens)
	keyB := MetaKey("model-b", tokens)
	kvMeta.WriteMeta(keyA, 10, []string{"blk1", "blk2", "blk3", "blk4", "blk5"}, 100, "model-a", "be1", 0)
	kvMeta.WriteMeta(keyB, 10, []string{"blk1", "x", "x", "x", "x"}, 100, "model-b", "be2", 0)
	reqBlocks := []string{"blk1", "blk2", "blk3", "blk4", "blk5"}
	bestRatio := 0.0
	bestCanonical := ""
	for name, be := range map[string]string{"model-a": "be1", "model-b": "be2"} {
		mk := MetaKey(name, tokens)
		if _, r, ok := kvMeta.FindRestoreCandidate(mk, 100, 0.2, reqBlocks, be); ok && r > bestRatio {
			bestRatio = r
			bestCanonical = name
		}
	}
	if bestCanonical != "model-a" {
		t.Errorf("best = %q, want model-a (ratio 1.0)", bestCanonical)
	}
	if bestRatio != 1.0 {
		t.Errorf("best ratio = %v, want 1.0", bestRatio)
	}
}

func TestCacheHitAcrossCanonicalModels(t *testing.T) {
	withTempMetaDir(t)
	bm := bmWithModels(t,
		dm("qwen3.6-32b", 32768, "be1"),
		dm("qwen3.6-8b", 8192, "be2"),
	)
	tokens := []int{1, 2, 3}
	kvMeta.WriteMeta(MetaKey("qwen3.6-32b", tokens), 3, []string{"blk1", "blk2", "x"}, 100, "qwen3.6-32b", "be1", 0)
	kvMeta.WriteMeta(MetaKey("qwen3.6-8b", tokens), 3, []string{"blk1", "blk2", "blk3"}, 100, "qwen3.6-8b", "be2", 0)
	reqBlocks := []string{"blk1", "blk2", "blk3"}
	bestRatio := 0.0
	bestCanonical := ""
	for _, d := range bm.GetDiscoveredModels("qwen3.6") {
		mk := MetaKey(d.Name, tokens)
		for _, be := range d.Backends {
			if _, r, ok := kvMeta.FindRestoreCandidate(mk, 100, 0.2, reqBlocks, be); ok && r > bestRatio {
				bestRatio = r
				bestCanonical = d.Name
			}
		}
	}
	if bestCanonical != "qwen3.6-8b" {
		t.Errorf("best = %q, want qwen3.6-8b (higher ratio)", bestCanonical)
	}
	if bestRatio != 1.0 {
		t.Errorf("best ratio = %v, want 1.0", bestRatio)
	}
}

func TestGetCacheSizeMetaAndFallback(t *testing.T) {
	cacheDir := t.TempDir()
	withTempMetaDir(t)
	withTestBackend(t, []map[string]any{{"url": "http://10.0.0.1:8000", "cache_dir": cacheDir}})
	key := "10.0.0.1-8000"
	if err := os.WriteFile(filepath.Join(cacheDir, "k-meta"), []byte("12345678"), 0o644); err != nil {
		t.Fatal(err)
	}
	kvMeta.WriteMeta("k-meta", 10, []string{}, 100, "m", key, 4096)
	if got := kvMeta.GetCacheSize(key, "k-meta"); got != 4096 {
		t.Errorf("GetCacheSize(meta) = %d, want 4096", got)
	}
	if err := os.WriteFile(filepath.Join(cacheDir, "k-file"), []byte("12345678"), 0o644); err != nil {
		t.Fatal(err)
	}
	kvMeta.WriteMeta("k-file", 10, []string{}, 100, "m", key, 0)
	if got := kvMeta.GetCacheSize(key, "k-file"); got != 8 {
		t.Errorf("GetCacheSize(file fallback) = %d, want 8", got)
	}
	if got := kvMeta.GetCacheSize(key, "k-nope"); got != 0 {
		t.Errorf("GetCacheSize(missing) = %d, want 0", got)
	}
}

func TestGetLastUsedTime(t *testing.T) {
	cacheDir := t.TempDir()
	withTempMetaDir(t)
	withTestBackend(t, []map[string]any{{"url": "http://10.0.0.1:8000", "cache_dir": cacheDir}})
	key := "10.0.0.1-8000"

	kvMeta.WriteMeta("k1", 10, []string{}, 100, "m", key, 0)
	meta := kvMeta.ReadMeta("k1", key)
	ts := metaFloat(meta, "last_written")
	if got := kvMeta.GetLastUsedTime("k1", key); got != ts {
		t.Errorf("GetLastUsedTime(k1) = %v, want last_written %v", got, ts)
	}

	if err := os.WriteFile(kvMeta.MetaFilePath("k2", key), []byte(`{"key":"k2","last_read":1234.5,"last_written":111.0}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := kvMeta.GetLastUsedTime("k2", key); got != 1234.5 {
		t.Errorf("GetLastUsedTime(k2) = %v, want last_read 1234.5", got)
	}

	if err := os.WriteFile(filepath.Join(cacheDir, "k3"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	approx := float64(time.Now().UnixNano()) / 1e9
	got := kvMeta.GetLastUsedTime("k3", key)
	if got < approx-10 || got > approx+10 {
		t.Errorf("GetLastUsedTime(k3) = %v, want ~now %v (cache file mtime fallback)", got, approx)
	}
}
