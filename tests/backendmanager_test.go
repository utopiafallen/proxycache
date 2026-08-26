package tests

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"proxycache"
	"strings"
	"testing"
)

func newTestBM(t *testing.T, cfg []map[string]any) *proxycache.BackendManager {
	t.Helper()
	oldMetaDir := proxycache.MetaDir
	proxycache.MetaDir = t.TempDir()
	t.Cleanup(func() { proxycache.MetaDir = oldMetaDir })
	bm := proxycache.NewBackendManager(cfg)
	t.Cleanup(bm.Close)
	return bm
}

func TestGenerateLCPModelsBasic(t *testing.T) {
	bm := &proxycache.BackendManager{}
	names := []string{
		"unsloth/Qwen3.6-27B-MTP-GGUF:Q6_K",
		"unsloth/Qwen3.6-27B-GGUF:Q5_K_S",
	}
	result := bm.GenerateLCPModels(names)
	contains := func(v string) bool {
		for _, r := range result {
			if r == v {
				return true
			}
		}
		return false
	}
	if !contains("Qwen3.6") {
		t.Errorf("expected 'Qwen3.6' in %v", result)
	}
	if !contains("Qwen3.6-27B") {
		t.Errorf("expected 'Qwen3.6-27B' in %v", result)
	}
	for _, p := range result {
		for i := 0; i < len(p); i++ {
			if p[i] == '/' {
				t.Errorf("prefix %q should not contain /", p)
			}
		}
	}
}

func TestGenerateLCPModelsStripsProvider(t *testing.T) {
	bm := &proxycache.BackendManager{}
	names := []string{
		"unsloth/Qwen3.6-27B-MTP-GGUF:Q6_K",
		"unsloth/Llama-3.1-8B-GGUF:Q4_K_M",
	}
	for _, p := range bm.GenerateLCPModels(names) {
		found := false
		for i := 0; i+len("unsloth") <= len(p); i++ {
			if p[i:i+len("unsloth")] == "unsloth" {
				found = true
			}
		}
		if found {
			t.Errorf("provider prefix should be stripped, got %q", p)
		}
	}
}

func TestGenerateLCPModelsStripsTrailingSeparators(t *testing.T) {
	bm := &proxycache.BackendManager{}
	names := []string{
		"unsloth/Qwen3.6-27B-MTP-GGUF:Q6_K",
		"unsloth/Qwen3.6-27B-GGUF:Q5_K_S",
	}
	for _, p := range bm.GenerateLCPModels(names) {
		if len(p) > 0 && (p[len(p)-1] == '-' || p[len(p)-1] == '_') {
			t.Errorf("prefix %q ends with separator", p)
		}
	}
}

func TestGenerateLCPModelsSingleModel(t *testing.T) {
	bm := &proxycache.BackendManager{}
	result := bm.GenerateLCPModels([]string{"unsloth/Qwen3.6-27B-MTP-GGUF:Q6_K"})
	if len(result) != 0 {
		t.Errorf("expected empty for single model, got %v", result)
	}
}

func TestGenerateLCPModelsUnderscoreSeparator(t *testing.T) {
	bm := &proxycache.BackendManager{}
	names := []string{
		"unsloth/Qwen3.6_27B-MTP-GGUF:Q6_K",
		"unsloth/Qwen3.6_27B-GGUF:Q5_K_S",
	}
	result := bm.GenerateLCPModels(names)
	contains := func(v string) bool {
		for _, r := range result {
			if r == v {
				return true
			}
		}
		return false
	}
	if !contains("Qwen3.6") {
		t.Errorf("expected 'Qwen3.6' in %v", result)
	}
	if !contains("Qwen3.6_27B") {
		t.Errorf("expected 'Qwen3.6_27B' in %v", result)
	}
}

func TestBackendCacheDirPerBackend(t *testing.T) {
	bm := newTestBM(t, []map[string]any{
		{"url": "http://10.0.0.1:8000", "cache_dir": "/mnt/cache/b1"},
		{"url": "http://10.0.0.2:8000", "agent_port": 8082},
	})
	be1 := proxycache.SanitizeBackendDir("10.0.0.1:8000")
	be2 := proxycache.SanitizeBackendDir("10.0.0.2:8000")
	keys := bm.Keys()
	if len(keys) != 2 || keys[0] != be1 || keys[1] != be2 {
		t.Errorf("keys = %v, want [%v %v]", keys, be1, be2)
	}
	if bm.GetCacheDir(be1) != "/mnt/cache/b1" {
		t.Errorf("GetCacheDir(be1) = %v", bm.GetCacheDir(be1))
	}
	if bm.GetCacheDir(be2) != "" {
		t.Errorf("GetCacheDir(be2) = %v, want empty", bm.GetCacheDir(be2))
	}
	if !bm.HasCacheConfig(be1) || !bm.HasCacheConfig(be2) {
		t.Error("HasCacheConfig should be true for both")
	}
}

func TestBackendCacheMaxSizeDefaults(t *testing.T) {
	bm := newTestBM(t, []map[string]any{
		{"url": "http://10.0.0.1:8000", "cache_dir": "/mnt/cache/b1"},
		{"url": "http://10.0.0.2:8000", "cache_dir": "/mnt/cache/b2", "cache_max_size_gb": 0},
	})
	be1 := proxycache.SanitizeBackendDir("10.0.0.1:8000")
	be2 := proxycache.SanitizeBackendDir("10.0.0.2:8000")
	if bm.GetCacheMaxSizeGB(be1) != 25.0 {
		t.Errorf("default max size = %v, want 25.0", bm.GetCacheMaxSizeGB(be1))
	}
	if !bm.CacheEnabled(be1) {
		t.Error("CacheEnabled(be1) should be true")
	}
	if bm.CacheEnabled(be2) {
		t.Error("CacheEnabled(be2) should be false with cache_max_size_gb=0")
	}
	if bm.GetCacheMaxSizeGB("unknown") != 25.0 {
		t.Errorf("unknown backend max size = %v, want 25.0", bm.GetCacheMaxSizeGB("unknown"))
	}
}

func assertPanic(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Error("expected panic")
		}
	}()
	fn()
}

func TestBackendCacheDirMutualExclusivity(t *testing.T) {
	assertPanic(t, func() {
		proxycache.NewBackendManager([]map[string]any{{"url": "http://10.0.0.1:8000", "cache_dir": "/mnt/cache", "agent_port": 8082}})
	})
	assertPanic(t, func() {
		proxycache.NewBackendManager([]map[string]any{{"url": "http://10.0.0.1:8000"}})
	})
}

func bmWithModels(t *testing.T, models ...*proxycache.DiscoveredModel) *proxycache.BackendManager {
	t.Helper()
	bm := &proxycache.BackendManager{
		Backends:               map[string]*proxycache.BackendInfo{},
		RefreshState:           map[proxycache.BackendModelKey]proxycache.RefreshEntry{},
		DiscoveredModels:       map[string]*proxycache.DiscoveredModel{},
		BackendState:           map[string]bool{},
		BackendLastUsed:        map[string]float64{},
		BackendLatencyEMA:      map[string]float64{},
		BackendModelLatencyEMA: map[proxycache.ModelEMAKey]float64{},
	}
	for _, m := range models {
		bm.DiscoveredModels[m.Name] = m
		bm.ModelOrder = append(bm.ModelOrder, m.Name)
	}
	return bm
}

func dm(name string, nCtx int, backends ...string) *proxycache.DiscoveredModel {
	return &proxycache.DiscoveredModel{Name: name, NCtx: nCtx, Backends: backends, BackendNCTX: map[string]int{}}
}

func TestResolveExactMatch(t *testing.T) {
	bm := bmWithModels(t, dm("qwen3.6-32b", 32768, "10.0.0.1-8000", "10.0.0.1-9000"))
	result := bm.GetDiscoveredModels("qwen3.6-32b")
	if len(result) != 1 {
		t.Fatalf("got %d results, want 1", len(result))
	}
	if result[0].Name != "qwen3.6-32b" {
		t.Errorf("name = %v", result[0].Name)
	}
	if len(result[0].Backends) != 2 || result[0].Backends[0] != "10.0.0.1-8000" || result[0].Backends[1] != "10.0.0.1-9000" {
		t.Errorf("backends = %v", result[0].Backends)
	}
}

func TestResolveSubstringMatch(t *testing.T) {
	bm := bmWithModels(t, dm("qwen3.6-32b-instruct", 32768, "10.0.0.1-8000"))
	result := bm.GetDiscoveredModels("qwen3.6")
	if len(result) != 1 || result[0].Name != "qwen3.6-32b-instruct" {
		t.Fatalf("got %v, want [qwen3.6-32b-instruct]", result)
	}
}

func TestResolveAmbiguousSubstring(t *testing.T) {
	bm := bmWithModels(t,
		dm("qwen3.6-32b", 32768, "10.0.0.1-8000"),
		dm("qwen3.6-8b", 8192, "10.0.0.1-9000"),
	)
	result := bm.GetDiscoveredModels("qwen3.6")
	if len(result) != 2 {
		t.Fatalf("got %d results, want 2", len(result))
	}
	names := []string{result[0].Name, result[1].Name}
	for _, want := range []string{"qwen3.6-32b", "qwen3.6-8b"} {
		found := false
		for _, n := range names {
			if n == want {
				found = true
			}
		}
		if !found {
			t.Errorf("%q missing from %v", want, names)
		}
	}
}

func TestResolveAny(t *testing.T) {
	bm := bmWithModels(t,
		dm("qwen3.6-32b", 32768, "10.0.0.1-8000"),
		dm("gemma-3-12b", 16384, "10.0.0.1-9000"),
	)
	result := bm.GetDiscoveredModels("any")
	if len(result) != 2 {
		t.Fatalf("got %d results, want 2", len(result))
	}
}

func TestResolveNotFound(t *testing.T) {
	bm := bmWithModels(t)
	if len(bm.GetDiscoveredModels("unknown")) != 0 {
		t.Error("expected empty result for unknown model")
	}
}

func TestResolveAnyNoModels(t *testing.T) {
	bm := bmWithModels(t)
	if len(bm.GetDiscoveredModels("any")) != 0 {
		t.Error("expected empty result for any with no models")
	}
}

func TestGetModelNCtx(t *testing.T) {
	bm := bmWithModels(t, dm("qwen3.6-32b", 32768, "be1"))
	if got := bm.GetModelNCtx("qwen3.6-32b"); got != 32768 {
		t.Errorf("GetModelNCtx = %d, want 32768", got)
	}
	if got := bm.GetModelNCtx("missing"); got != proxycache.DefaultNCtx {
		t.Errorf("GetModelNCtx(missing) = %d, want %d", got, proxycache.DefaultNCtx)
	}
	minCtx := dm("model-a", 8192, "be1", "be2")
	minCtx.BackendNCTX = map[string]int{"be1": 32768, "be2": 8192}
	bm.DiscoveredModels["model-a"] = minCtx
	bm.ModelOrder = append(bm.ModelOrder, "model-a")
	if got := bm.GetModelNCtx("model-a"); got != 8192 {
		t.Errorf("min across backends = %d, want 8192", got)
	}
}

func TestGetBackendNCtx(t *testing.T) {
	minCtx := dm("model-a", 8192, "be1", "be2")
	minCtx.BackendNCTX = map[string]int{"be1": 32768, "be2": 8192}
	bm := bmWithModels(t, minCtx)
	if got := bm.GetBackendNCtx("model-a", "be1"); got != 32768 {
		t.Errorf("be1 = %d, want 32768", got)
	}
	if got := bm.GetBackendNCtx("model-a", "be2"); got != 8192 {
		t.Errorf("be2 = %d, want 8192", got)
	}
	if got := bm.GetBackendNCtx("model-a", "be3"); got != proxycache.DefaultNCtx {
		t.Errorf("be3 = %d, want %d", got, proxycache.DefaultNCtx)
	}
	if got := bm.GetBackendNCtx("unknown-model", "be1"); got != proxycache.DefaultNCtx {
		t.Errorf("unknown-model = %d, want %d", got, proxycache.DefaultNCtx)
	}
}

func TestDiscoverModelsRouterMode(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxycache.WriteJSON(w, 200, map[string]any{"data": []any{
			map[string]any{"id": "model-a", "status": map[string]any{"value": "loaded", "args": []any{"llama-server", "-ctx", "32768"}}},
			map[string]any{"id": "model-b", "status": map[string]any{"value": "loaded", "args": []any{"llama-server", "-c", "8192"}}},
		}})
	}))
	defer srv.Close()
	client := proxycache.NewLlamaClient(srv.URL)
	models, err := client.DiscoverModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(models) != 2 || models[0].Name != "model-a" || models[0].NCtx != 32768 || models[1].Name != "model-b" || models[1].NCtx != 8192 {
		t.Errorf("got %+v", models)
	}
}

func TestDiscoverModelsNonRouterMode(t *testing.T) {
	payload := map[string]any{"data": []any{map[string]any{"id": "llama-3.1", "meta": map[string]any{"n_ctx": 4096}}}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxycache.WriteJSON(w, 200, payload)
	}))
	defer srv.Close()
	client := proxycache.NewLlamaClient(srv.URL)
	models, err := client.DiscoverModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(models) != 1 || models[0].Name != "llama-3.1" || models[0].NCtx != 4096 {
		t.Errorf("got %+v", models)
	}
}

func TestDiscoverModelsCtxNotInArgs(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxycache.WriteJSON(w, 200, map[string]any{"data": []any{
			map[string]any{"id": "model-x", "status": map[string]any{"value": "loaded", "args": []any{"llama-server"}}},
		}})
	}))
	defer srv.Close()
	client := proxycache.NewLlamaClient(srv.URL)
	models, err := client.DiscoverModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(models) != 1 || models[0].NCtx != proxycache.DefaultNCtx {
		t.Errorf("got %+v, want n_ctx=%d", models, proxycache.DefaultNCtx)
	}
}

func TestDiscoverModelsRouterLoadedInfoNCtx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxycache.WriteJSON(w, 200, map[string]any{"data": []any{
			map[string]any{"id": "model-y", "status": map[string]any{
				"value":       "loaded",
				"args":        []any{"llama-server"},
				"loaded_info": map[string]any{"n_ctx": 16384},
			}},
		}})
	}))
	defer srv.Close()
	client := proxycache.NewLlamaClient(srv.URL)
	models, err := client.DiscoverModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(models) != 1 || models[0].NCtx != 16384 {
		t.Errorf("got %+v", models)
	}
}

func TestDiscoverModelsNonRouterMetaNull(t *testing.T) {
	payload := map[string]any{"data": []any{map[string]any{"id": "model-z", "meta": nil}}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxycache.WriteJSON(w, 200, payload)
	}))
	defer srv.Close()
	client := proxycache.NewLlamaClient(srv.URL)
	models, err := client.DiscoverModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(models) != 1 || models[0].Name != "model-z" || models[0].NCtx != proxycache.DefaultNCtx {
		t.Errorf("got %+v, want n_ctx=%d", models, proxycache.DefaultNCtx)
	}
}

func TestDiscoverModelsBothEndpointsFail(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	urlStr := srv.URL
	srv.Close()
	client := proxycache.NewLlamaClient(urlStr)
	models, err := client.DiscoverModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(models) != 0 {
		t.Errorf("got %+v, want empty", models)
	}
}

func TestDiscoverModelsIncludesLCPSynthetics(t *testing.T) {
	payload := map[string]any{"data": []any{
		map[string]any{"id": "unsloth/Qwen3.6-27B-MTP-GGUF:Q6_K", "status": map[string]any{"value": "loaded", "args": []any{"llama-server", "-ctx", "32768"}}},
		map[string]any{"id": "unsloth/Qwen3.6-27B-GGUF:Q5_K_S", "status": map[string]any{"value": "loaded", "args": []any{"llama-server", "-ctx", "32768"}}},
	}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxycache.WriteJSON(w, 200, payload)
	}))
	defer srv.Close()
	bm := newTestBM(t, []map[string]any{{"url": srv.URL, "cache_dir": t.TempDir()}})
	key := backendKeyFromURL(srv.URL)
	bm.BackendState[key] = true
	if err := bm.DiscoverModels(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, want := range []string{
		"unsloth/Qwen3.6-27B-MTP-GGUF:Q6_K",
		"unsloth/Qwen3.6-27B-GGUF:Q5_K_S",
		"Qwen3.6",
		"Qwen3.6-27B",
	} {
		found := false
		for _, m := range bm.SnapshotModels() {
			if m.Name == want {
				found = true
			}
		}
		if !found {
			t.Errorf("model %q missing from registry", want)
		}
	}
	synth := bm.GetDiscoveredModels("Qwen3.6-27B")
	if len(synth) != 1 {
		t.Fatalf("got %d matches for Qwen3.6-27B, want 1", len(synth))
	}
	if synth[0].NCtx != 32768 {
		t.Errorf("synthetic n_ctx = %d, want 32768", synth[0].NCtx)
	}
	if len(synth[0].Backends) != 1 {
		t.Errorf("synthetic backends = %v, want 1", synth[0].Backends)
	}
	if !synth[0].Synthetic {
		t.Error("Qwen3.6-27B should be marked synthetic")
	}
}

func TestBackendCacheDeleteViaLocal(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"test_key", "test_key.ckpt", "test_key.ckpt.0"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("data"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	bm := newTestBM(t, []map[string]any{{"url": "http://10.0.0.1:8000", "cache_dir": dir}})
	be := proxycache.SanitizeBackendDir("10.0.0.1:8000")
	if !bm.CacheDelete(be, "test_key") {
		t.Error("CacheDelete returned false")
	}
	for _, name := range []string{"test_key", "test_key.ckpt", "test_key.ckpt.0"} {
		if _, err := os.Stat(filepath.Join(dir, name)); !os.IsNotExist(err) {
			t.Errorf("%s should be deleted", name)
		}
	}
}

func TestBackendCacheDeleteUnknownBackend(t *testing.T) {
	bm := newTestBM(t, []map[string]any{{"url": "http://10.0.0.1:8000", "cache_dir": t.TempDir()}})
	if bm.CacheDelete("10.9.9.9-8000", "test_key") {
		t.Error("CacheDelete on unknown backend should return false")
	}
}

func TestBackendCacheGetSizeViaLocal(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "test_key"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	data := make([]byte, 1234)
	f, err := os.Create(filepath.Join(dir, "sized_key"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(data); err != nil {
		t.Fatal(err)
	}
	f.Close()
	bm := newTestBM(t, []map[string]any{{"url": "http://10.0.0.1:8000", "cache_dir": dir}})
	be := proxycache.SanitizeBackendDir("10.0.0.1:8000")
	if got := bm.CacheGetSize(be, "sized_key"); got != 1234 {
		t.Errorf("CacheGetSize = %d, want 1234", got)
	}
	if got := bm.CacheGetSize(be, "nonexistent"); got != 0 {
		t.Errorf("CacheGetSize(missing) = %d, want 0", got)
	}
}

func TestBackendCacheExistsViaLocal(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "test_key"), []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	bm := newTestBM(t, []map[string]any{{"url": "http://10.0.0.1:8000", "cache_dir": dir}})
	be := proxycache.SanitizeBackendDir("10.0.0.1:8000")
	if !bm.CacheExists(be, "test_key") {
		t.Error("CacheExists should be true")
	}
	if bm.CacheExists(be, "nonexistent") {
		t.Error("CacheExists should be false for missing file")
	}
}

func TestBackendCacheGetMtimeViaLocal(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_key")
	if err := os.WriteFile(path, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	want := float64(st.ModTime().UnixNano()) / 1e9
	bm := newTestBM(t, []map[string]any{{"url": "http://10.0.0.1:8000", "cache_dir": dir}})
	be := proxycache.SanitizeBackendDir("10.0.0.1:8000")
	got := bm.CacheGetMtime(be, "test_key")
	diff := got - want
	if diff < 0 {
		diff = -diff
	}
	if diff >= 1.0 {
		t.Errorf("mtime = %f, want ~%f", got, want)
	}
}

func TestBackendCacheDeleteViaAgent(t *testing.T) {
	status := 200
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/cache/delete" {
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
		}
		proxycache.WriteJSON(w, status, map[string]any{"ok": true})
	}))
	defer srv.Close()
	hostPort := srv.URL
	if i := strings.LastIndex(hostPort, "/"); i >= 0 {
		hostPort = hostPort[i+1:]
	}
	bm := newTestBM(t, []map[string]any{{"url": "http://" + hostPort, "agent_port": agentPortFromHostPort(t, hostPort)}})
	be := proxycache.SanitizeBackendDir(hostPort)
	if !bm.CacheDelete(be, "test_key") {
		t.Error("agent delete should return true on 200")
	}
	status = 500
	if bm.CacheDelete(be, "test_key") {
		t.Error("agent delete should return false on 500")
	}
}

func agentPortFromHostPort(t *testing.T, hostPort string) int {
	t.Helper()
	for i := len(hostPort) - 1; i >= 0; i-- {
		if hostPort[i] == ':' {
			n := 0
			digits := 0
			for j := i + 1; j < len(hostPort); j++ {
				if hostPort[j] < '0' || hostPort[j] > '9' {
					break
				}
				n = n*10 + int(hostPort[j]-'0')
				digits++
			}
			if digits > 0 {
				return n
			}
		}
	}
	t.Fatalf("no port in %q", hostPort)
	return 0
}

func TestLivenessDiagDue(t *testing.T) {
	// State transitions always record.
	if !proxycache.LivenessDiagDue(nil, nil, true, 100, 60) {
		t.Error("changed=true should always be due")
	}
	// Nothing noteworthy, no change: not due.
	if proxycache.LivenessDiagDue(nil, nil, false, 100, 60) {
		t.Error("nothing noteworthy should not be due")
	}
	// First tick of a noteworthy backend: due (no prior record).
	if !proxycache.LivenessDiagDue([]string{"a"}, map[string]float64{}, false, 100, 60) {
		t.Error("first noteworthy tick should be due")
	}
	// Sustained episode within the interval: suppressed.
	last := map[string]float64{"a": 100}
	if proxycache.LivenessDiagDue([]string{"a"}, last, false, 159.9, 60) {
		t.Error("sustained episode within interval should be suppressed")
	}
	// Due again after the interval elapses.
	if !proxycache.LivenessDiagDue([]string{"a"}, last, false, 160, 60) {
		t.Error("should be due after interval elapses")
	}
	// A second backend's first episode is independent of the first's.
	if !proxycache.LivenessDiagDue([]string{"b"}, last, false, 110, 60) {
		t.Error("unrecorded backend should be due regardless of other backends")
	}
}
