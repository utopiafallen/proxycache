package tests

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"proxycache"
	"strings"
	"sync"
	"testing"
	"time"
)

type mockLlama struct {
	mu            sync.Mutex
	srv           *httptest.Server
	model         string
	nCtx          int
	tokens        []int
	nSlots        int
	chatDelay     time.Duration
	chatBodies    []map[string]any
	saves         []string
	restores      []string
	chatResp      map[string]any
	concurrent    int // chat completions currently being served
	maxConcurrent int // high-water mark of concurrent
}

func newMockLlama(t *testing.T, model string, nCtx int, tokens []int, nSlots int) *mockLlama {
	t.Helper()
	m := &mockLlama{model: model, nCtx: nCtx, tokens: tokens, nSlots: nSlots}
	m.chatResp = map[string]any{
		"object": "chat.completion",
		"choices": []map[string]any{
			{"message": map[string]any{"role": "assistant", "content": "ok"}},
		},
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		proxycache.WriteJSON(w, http.StatusOK, map[string]any{"object": "models"})
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		proxycache.WriteJSON(w, http.StatusOK, map[string]any{
			"data": []map[string]any{
				{"id": m.model, "meta": map[string]any{"n_ctx": m.nCtx}},
			},
		})
	})
	mux.HandleFunc("/apply-template", func(w http.ResponseWriter, r *http.Request) {
		proxycache.WriteJSON(w, http.StatusOK, map[string]any{"prompt": "mock"})
	})
	mux.HandleFunc("/tokenize", func(w http.ResponseWriter, r *http.Request) {
		proxycache.WriteJSON(w, http.StatusOK, map[string]any{"tokens": m.tokens})
	})
	mux.HandleFunc("/slots", func(w http.ResponseWriter, r *http.Request) {
		slots := make([]map[string]any, 0, m.nSlots)
		for i := 0; i < m.nSlots; i++ {
			slots = append(slots, map[string]any{"id": i, "state": 2})
		}
		proxycache.WriteJSON(w, http.StatusOK, slots)
	})
	mux.HandleFunc("/slots/", func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimPrefix(r.URL.Path, "/slots/")
		action := r.URL.Query().Get("action")
		m.mu.Lock()
		if action == "save" {
			m.saves = append(m.saves, id)
		} else if action == "restore" {
			m.restores = append(m.restores, id)
		}
		m.mu.Unlock()
		proxycache.WriteJSON(w, http.StatusOK, map[string]any{"success": true, "n_written": 4096})
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			proxycache.WriteJSON(w, http.StatusBadRequest, map[string]any{"error": "bad json"})
			return
		}
		m.mu.Lock()
		m.chatBodies = append(m.chatBodies, body)
		resp := m.chatResp
		delay := m.chatDelay
		m.concurrent++
		if m.concurrent > m.maxConcurrent {
			m.maxConcurrent = m.concurrent
		}
		m.mu.Unlock()
		if delay > 0 {
			time.Sleep(delay)
		}
		m.mu.Lock()
		m.concurrent--
		m.mu.Unlock()
		proxycache.WriteJSON(w, http.StatusOK, resp)
	})
	m.srv = httptest.NewServer(mux)
	t.Cleanup(m.srv.Close)
	return m
}

func (m *mockLlama) chatCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.chatBodies)
}

func (m *mockLlama) maxChatConcurrency() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.maxConcurrent
}

func (m *mockLlama) lastChatModel() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.chatBodies) == 0 {
		return ""
	}
	v, _ := m.chatBodies[len(m.chatBodies)-1]["model"].(string)
	return v
}

func (m *mockLlama) lastChatOptions() map[string]any {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.chatBodies) == 0 {
		return nil
	}
	v, _ := m.chatBodies[len(m.chatBodies)-1]["options"].(map[string]any)
	return v
}

func postChat(t *testing.T, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	proxycache.ChatHandler(w, req)
	return w
}

func doGet(t *testing.T, h http.HandlerFunc, path string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	w := httptest.NewRecorder()
	h(w, req)
	return w
}

// waitFor polls cond every 10ms until it returns true or the timeout elapses.
func waitFor(t *testing.T, timeout time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("condition not met within %v", timeout)
}

func markBackendsUp(bm *proxycache.BackendManager) {
	bm.Mu.Lock()
	defer bm.Mu.Unlock()
	for _, k := range bm.KeyOrder {
		bm.BackendState[k] = true
	}
}

func injectModels(bm *proxycache.BackendManager, models ...*proxycache.DiscoveredModel) {
	bm.Mu.Lock()
	defer bm.Mu.Unlock()
	for _, m := range models {
		if _, ok := bm.DiscoveredModels[m.Name]; !ok {
			bm.ModelOrder = append(bm.ModelOrder, m.Name)
		}
		bm.DiscoveredModels[m.Name] = m
	}
}

func seq(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i + 1
	}
	return out
}

func offsetSeq(n, start int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = start + i
	}
	return out
}

func TestChatModelNotFound(t *testing.T) {
	withTempMetaDir(t)
	m := newMockLlama(t, "model-a", 32768, seq(10), 1)
	withTestBackend(t, []map[string]any{{"url": m.srv.URL, "cache_dir": t.TempDir()}})
	markBackendsUp(proxycache.GetBackendManager())
	w := postChat(t, `{"model": "unknown-model", "messages": [{"role": "user", "content": "hello"}]}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
	if !strings.Contains(w.Body.String(), "not found") {
		t.Errorf("body = %q, want 'not found'", w.Body.String())
	}
	if m.chatCount() != 0 {
		t.Errorf("chat calls = %d, want 0", m.chatCount())
	}
}

func TestChatPromptTooLong(t *testing.T) {
	withTempMetaDir(t)
	m := newMockLlama(t, "model-a", 4096, seq(4097), 1)
	withTestBackend(t, []map[string]any{{"url": m.srv.URL, "cache_dir": t.TempDir()}})
	markBackendsUp(proxycache.GetBackendManager())
	w := postChat(t, `{"model": "model-a", "messages": [{"role": "user", "content": "hello"}]}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
	if !strings.Contains(w.Body.String(), "too long") {
		t.Errorf("body = %q, want 'too long'", w.Body.String())
	}
}

func TestChatSubstringModelResolution(t *testing.T) {
	withTempMetaDir(t)
	m := newMockLlama(t, "qwen3.6-32b-instruct", 32768, seq(10), 1)
	withTestBackend(t, []map[string]any{{"url": m.srv.URL, "cache_dir": t.TempDir()}})
	markBackendsUp(proxycache.GetBackendManager())
	w := postChat(t, `{"model": "qwen3.6", "messages": [{"role": "user", "content": "hello"}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := m.lastChatModel(); got != "qwen3.6-32b-instruct" {
		t.Errorf("forwarded model = %q, want qwen3.6-32b-instruct", got)
	}
	opts := m.lastChatOptions()
	if opts == nil {
		t.Fatal("no options in forwarded body")
	}
	if v, _ := opts["slot_id"].(float64); v != 0 {
		t.Errorf("options.slot_id = %v, want 0", opts["slot_id"])
	}
}

func TestChatAnyModelRouting(t *testing.T) {
	withTempMetaDir(t)
	m := newMockLlama(t, "model-x", 32768, seq(10), 1)
	withTestBackend(t, []map[string]any{{"url": m.srv.URL, "cache_dir": t.TempDir()}})
	markBackendsUp(proxycache.GetBackendManager())
	w := postChat(t, `{"model": "any", "messages": [{"role": "user", "content": "hello"}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := m.lastChatModel(); got != "model-x" {
		t.Errorf("forwarded model = %q, want model-x", got)
	}
}

func TestChatAnyWithCacheHit(t *testing.T) {
	withTempMetaDir(t)
	tokens := []int{123, 456, 789, 101, 102, 103, 104, 105, 106, 107}
	mA := newMockLlama(t, "model-a", 32768, tokens, 1)
	mB := newMockLlama(t, "model-b", 16384, tokens, 1)
	withTestBackend(t, []map[string]any{
		{"url": mA.srv.URL, "cache_dir": t.TempDir()},
		{"url": mB.srv.URL, "cache_dir": t.TempDir()},
	})
	markBackendsUp(proxycache.GetBackendManager())
	oldWpb := proxycache.WordsPerBlock
	proxycache.WordsPerBlock = 3
	defer func() { proxycache.WordsPerBlock = oldWpb }()
	beA := backendKeyFromURL(mA.srv.URL)
	beB := backendKeyFromURL(mB.srv.URL)
	injectModels(proxycache.GetBackendManager(), dm("model-a", 32768, beA), dm("model-b", 16384, beB))
	reqBlocks := proxycache.BlockHashesFromTokens(tokens, proxycache.WordsPerBlock)
	blocksB := append(append([]string{}, reqBlocks[:2]...), "uniq_b_3", "uniq_b_4")
	proxycache.GetKVMeta().WriteMeta(proxycache.MetaKey("model-a", tokens), 10, reqBlocks, proxycache.WordsPerBlock, "model-a", beA, 1024)
	proxycache.GetKVMeta().WriteMeta(proxycache.MetaKey("model-b", tokens), 10, blocksB, proxycache.WordsPerBlock, "model-b", beB, 1024)
	w := postChat(t, `{"model": "any", "messages": [{"role": "user", "content": "hello"}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := mA.lastChatModel(); got != "model-a" {
		t.Errorf("model-a backend got model %q, want model-a", got)
	}
	if mA.chatCount() != 1 {
		t.Errorf("model-a backend chat count = %d, want 1", mA.chatCount())
	}
	if mB.chatCount() != 0 {
		t.Errorf("model-b backend chat count = %d, want 0", mB.chatCount())
	}
	if len(mA.restores) != 1 {
		t.Errorf("model-a backend restore count = %d, want 1", len(mA.restores))
	}
}

func TestModelsEndpointIncludesAny(t *testing.T) {
	withTempMetaDir(t)
	m := newMockLlama(t, "model-a", 32768, seq(10), 1)
	withTestBackend(t, []map[string]any{{"url": m.srv.URL, "cache_dir": t.TempDir()}})
	markBackendsUp(proxycache.GetBackendManager())
	injectModels(proxycache.GetBackendManager(), dm("model-a", 32768, "be-1"), dm("model-b", 16384, "be-2"))
	w := doGet(t, proxycache.ModelsHandler, "/v1/models")
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	var parsed struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(parsed.Data) != 3 {
		t.Fatalf("data len = %d, want 3", len(parsed.Data))
	}
	ids := []string{parsed.Data[0]["id"].(string), parsed.Data[1]["id"].(string), parsed.Data[2]["id"].(string)}
	if ids[0] != "model-a" || ids[1] != "model-b" || ids[2] != "any" {
		t.Errorf("ids = %v, want [model-a model-b any]", ids)
	}
	if v := parsed.Data[2]["n_ctx"].(float64); int(v) != 16384 {
		t.Errorf("any n_ctx = %v, want 16384", parsed.Data[2]["n_ctx"])
	}
}

func TestModelsEndpointOpenAIFormat(t *testing.T) {
	withTempMetaDir(t)
	m := newMockLlama(t, "model-a", 32768, seq(10), 1)
	withTestBackend(t, []map[string]any{{"url": m.srv.URL, "cache_dir": t.TempDir()}})
	markBackendsUp(proxycache.GetBackendManager())
	injectModels(proxycache.GetBackendManager(), dm("model-a", 32768, "be-1"))
	w := doGet(t, proxycache.ModelsHandler, "/v1/models")
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	var parsed struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, entry := range parsed.Data {
		if entry["object"] != "model" {
			t.Errorf("object = %v, want model", entry["object"])
		}
		if _, ok := entry["owned_by"].(string); !ok {
			t.Errorf("missing owned_by on %v", entry["id"])
		}
	}
	if v := parsed.Data[0]["owned_by"]; v != "backend" {
		t.Errorf("model-a owned_by = %v, want backend", v)
	}
	if v := parsed.Data[len(parsed.Data)-1]["owned_by"]; v != "proxycache" {
		t.Errorf("any owned_by = %v, want proxycache", v)
	}
}

func TestDashboardEndpoint(t *testing.T) {
	old := proxycache.DashboardEnabled
	proxycache.DashboardEnabled = true
	defer func() { proxycache.DashboardEnabled = old }()
	w := doGet(t, proxycache.DashboardHandler, "/dashboard")
	if w.Code != http.StatusOK {
		t.Fatalf("enabled status = %d, want 200", w.Code)
	}
	if w.Body.Len() == 0 {
		t.Error("enabled body empty, want embedded html")
	}
	proxycache.DashboardEnabled = false
	w = doGet(t, proxycache.DashboardHandler, "/dashboard")
	if w.Code != http.StatusNotFound {
		t.Fatalf("disabled status = %d, want 404", w.Code)
	}
}

func TestNoCacheRoundRobinDistribution(t *testing.T) {
	withTempMetaDir(t)
	m1 := newMockLlama(t, "rr-model", 32768, seq(10), 1)
	m2 := newMockLlama(t, "rr-model", 32768, seq(10), 1)
	m3 := newMockLlama(t, "rr-model", 32768, seq(10), 1)
	bm := withTestBackend(t, []map[string]any{
		{"url": m1.srv.URL, "cache_dir": t.TempDir()},
		{"url": m2.srv.URL, "cache_dir": t.TempDir()},
		{"url": m3.srv.URL, "cache_dir": t.TempDir()},
	})
	markBackendsUp(bm)
	be1 := backendKeyFromURL(m1.srv.URL)
	be2 := backendKeyFromURL(m2.srv.URL)
	be3 := backendKeyFromURL(m3.srv.URL)
	injectModels(bm, dm("rr-model", 32768, be1, be2, be3))
	body := `{"model": "rr-model", "messages": [{"role": "user", "content": "hello"}]}`
	// No cache hit: vary the mocked token sequence between requests so each
	// prompt hashes to a new key (identical prompts would make request 2+
	// cache hits on whichever backend served request 1). Requests are spread
	// round-robin across matching backends.
	for i := 0; i < 3; i++ {
		toks := offsetSeq(10, 1000*(i+1))
		m1.tokens, m2.tokens, m3.tokens = toks, toks, toks
		w := postChat(t, body)
		if w.Code != http.StatusOK {
			t.Fatalf("request %d status = %d, want 200", i+1, w.Code)
		}
	}
	if m1.chatCount() != 1 {
		t.Errorf("backend1 chat count = %d, want 1", m1.chatCount())
	}
	if m2.chatCount() != 1 {
		t.Errorf("backend2 chat count = %d, want 1", m2.chatCount())
	}
	if m3.chatCount() != 1 {
		t.Errorf("backend3 chat count = %d, want 1", m3.chatCount())
	}
}

func setupBusyRestoreBackend(t *testing.T, m1, m2 *mockLlama, model string) (string, string) {
	t.Helper()
	be1 := backendKeyFromURL(m1.srv.URL)
	be2 := backendKeyFromURL(m2.srv.URL)
	withTestBackend(t, []map[string]any{
		{"url": m1.srv.URL, "cache_dir": t.TempDir()},
		{"url": m2.srv.URL, "cache_dir": t.TempDir()},
	})
	markBackendsUp(proxycache.GetBackendManager())
	reqTokens := seq(800)
	reqBlocks := proxycache.BlockHashesFromTokens(reqTokens, proxycache.WordsPerBlock)
	proxycache.GetKVMeta().WriteMeta(proxycache.MetaKey(model, reqTokens), 800, reqBlocks, proxycache.WordsPerBlock, model, be1, 2048)
	beSm1 := proxycache.GetSlotManager().Get(be1)
	beSm1.EnsurePool(model, 1)
	if s := beSm1.TryAcquire(model); s != 0 {
		t.Fatalf("pre-acquire = %d, want 0", s)
	}
	return be1, be2
}

// TestCacheHitQueueLimitFallbackP2P verifies the core new routing rule: a
// cache-hit request goes to the hit backend while its queue depth stays under
// CACHE_HIT_QUEUE_LIMIT (2); the next request falls back to another backend,
// which triggers an async P2P transfer of the cache file.
func TestCacheHitQueueLimitFallbackP2P(t *testing.T) {
	withTempMetaDir(t)
	tokens := seq(800)
	m1 := newMockLlama(t, "hitq-model", 32768, tokens, 1)
	m2 := newMockLlama(t, "hitq-model", 32768, tokens, 1)
	m1.chatDelay = 300 * time.Millisecond
	m2.chatDelay = 300 * time.Millisecond
	dir1 := t.TempDir()
	dir2 := t.TempDir()
	be1 := backendKeyFromURL(m1.srv.URL)
	be2 := backendKeyFromURL(m2.srv.URL)
	withTestBackend(t, []map[string]any{
		{"url": m1.srv.URL, "cache_dir": dir1},
		{"url": m2.srv.URL, "cache_dir": dir2},
	})
	markBackendsUp(proxycache.GetBackendManager())
	injectModels(proxycache.GetBackendManager(), dm("hitq-model", 32768, be1, be2))
	reqBlocks := proxycache.BlockHashesFromTokens(tokens, proxycache.WordsPerBlock)
	hitKey := proxycache.MetaKey("hitq-model", tokens)
	proxycache.GetKVMeta().WriteMeta(hitKey, 800, reqBlocks, proxycache.WordsPerBlock, "hitq-model", be1, 2048)
	if err := os.WriteFile(filepath.Join(dir1, hitKey), []byte("kv-cache-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}

	body := `{"model": "hitq-model", "messages": [{"role": "user", "content": "hello"}]}`
	codes := make(chan int, 4)
	var wg sync.WaitGroup
	send := func() {
		defer wg.Done()
		codes <- postChat(t, body).Code
	}

	wg.Add(1)
	go send() // r1: in flight on the hit backend
	waitFor(t, 5*time.Second, func() bool { return m1.chatCount() == 1 })

	wg.Add(1)
	go send() // r2: queued on the hit backend (depth 1)
	waitFor(t, 5*time.Second, func() bool { return proxycache.GetDispatcher().QueueDepth(be1) == 1 })

	wg.Add(1)
	go send() // r3: queued on the hit backend (depth 2 = limit)
	waitFor(t, 5*time.Second, func() bool { return proxycache.GetDispatcher().QueueDepth(be1) == 2 })

	wg.Add(1)
	go send() // r4: hit backend at limit -> fallback to be2 + P2P transfer
	wg.Wait()
	for i := 0; i < 4; i++ {
		if c := <-codes; c != http.StatusOK {
			t.Errorf("request %d status = %d, want 200", i+1, c)
		}
	}
	if m1.chatCount() != 3 {
		t.Errorf("hit backend chat count = %d, want 3", m1.chatCount())
	}
	if m2.chatCount() != 1 {
		t.Errorf("fallback backend chat count = %d, want 1", m2.chatCount())
	}
	// P2P transfer must have copied the cache file to the fallback backend and
	// registered its meta + ring entry there.
	waitFor(t, 5*time.Second, func() bool {
		_, err := os.Stat(filepath.Join(dir2, hitKey))
		return err == nil
	})
	if meta := proxycache.GetKVMeta().ReadMeta(hitKey, be2); meta == nil {
		t.Errorf("no meta written for transferred key on backend %s", be2)
	}
	if got := proxycache.GetSlotManager().Get(be2).GetRingSize(); got != 1 {
		t.Errorf("fallback backend ring size = %d, want 1", got)
	}
	if got := proxycache.GetDispatcher().QueueDepth(be1); got != 0 {
		t.Errorf("hit backend queue depth after drain = %d, want 0", got)
	}
	if got := proxycache.GetDispatcher().QueueDepth(be2); got != 0 {
		t.Errorf("fallback backend queue depth after drain = %d, want 0", got)
	}
}

func withRetryBase(t *testing.T, base float64) {
	t.Helper()
	old := proxycache.SlotAcquireRetryBaseSeconds
	proxycache.SlotAcquireRetryBaseSeconds = base
	t.Cleanup(func() { proxycache.SlotAcquireRetryBaseSeconds = old })
}

func withMigrationAfter(t *testing.T, seconds float64) {
	t.Helper()
	old := proxycache.QueueMigrationAfter
	proxycache.QueueMigrationAfter = seconds
	t.Cleanup(func() { proxycache.QueueMigrationAfter = old })
}

// TestQueueMigrationToIdleBackend verifies the queue-migration monitor: a
// request with a disk cache hit gets queued on its (busy) hit backend; once it
// has waited past QUEUE_MIGRATION_AFTER and another backend serving the model
// is fully idle, the dispatcher moves it there and transfers the disk cache —
// so it runs on the idle backend instead of waiting out the backlog.
func TestQueueMigrationToIdleBackend(t *testing.T) {
	withTempMetaDir(t)
	withMigrationAfter(t, 1.0)
	tokensP := seq(800) // prefix content with a disk cache on A
	tokensQ := offsetSeq(400, 100000)
	mA := newMockLlama(t, "mig-model", 32768, tokensP, 1)
	mB := newMockLlama(t, "mig-model", 32768, tokensP, 1)
	beA := backendKeyFromURL(mA.srv.URL)
	beB := backendKeyFromURL(mB.srv.URL)
	dirA, dirB := t.TempDir(), t.TempDir()
	withTestBackend(t, []map[string]any{
		{"url": mA.srv.URL, "cache_dir": dirA},
		{"url": mB.srv.URL, "cache_dir": dirB},
	})
	bm := proxycache.GetBackendManager()
	markBackendsUp(bm)
	injectModels(bm, dm("mig-model", 32768, beA, beB))

	// Keep B down so the warmup and the busy request route deterministically to A.
	bm.Mu.Lock()
	bm.BackendState[beB] = false
	bm.Mu.Unlock()

	// Warmup: establishes a disk cache for tokensP on A (meta + dummy file —
	// the mock never writes physical cache files itself).
	if w := postChat(t, `{"model": "mig-model", "messages": [{"role": "user", "content": "warm"}]}`); w.Code != http.StatusOK {
		t.Fatalf("warmup status = %d, want 200", w.Code)
	}
	keyP := proxycache.MetaKey("mig-model", tokensP)
	if err := os.WriteFile(filepath.Join(dirA, keyP), []byte("warm-cache-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}

	// A1: holds A's only slot for 3s (no cache anywhere).
	mA.tokens = tokensQ
	mA.chatDelay = 3 * time.Second
	doneA1 := make(chan *httptest.ResponseRecorder, 1)
	go func() { doneA1 <- postChat(t, `{"model": "mig-model", "messages": [{"role": "user", "content": "busy"}]}`) }()
	waitFor(t, 5*time.Second, func() bool { return mA.chatCount() >= 1 })

	// R: disk hit on A while A is busy -> queued on A, then migrated to idle B.
	mA.tokens = tokensP
	bm.Mu.Lock()
	bm.BackendState[beB] = true
	bm.Mu.Unlock()
	doneR := make(chan *httptest.ResponseRecorder, 1)
	go func() { doneR <- postChat(t, `{"model": "mig-model", "messages": [{"role": "user", "content": "hit"}]}`) }()

	wR := <-doneR
	wA1 := <-doneA1
	if wR.Code != http.StatusOK || wA1.Code != http.StatusOK {
		t.Fatalf("statuses: R=%d A1=%d, want 200/200", wR.Code, wA1.Code)
	}
	if got := mA.chatCount(); got != 2 {
		t.Errorf("A chat count = %d, want 2 (warmup + busy request; R should have migrated away)", got)
	}
	if got := mB.chatCount(); got != 1 {
		t.Errorf("B chat count = %d, want 1 (the migrated request)", got)
	}
	mB.mu.Lock()
	restores := len(mB.restores)
	mB.mu.Unlock()
	if restores < 1 {
		t.Error("migrated request did not restore its transferred disk cache on B")
	}
}

// TestQueueMigrationColdWaitsDoubleThreshold verifies the guard for requests
// without a transferable disk key: a cold queued request is NOT migrated at
// QUEUE_MIGRATION_AFTER (migration would trade an imminent serve for a full
// recompute) — it stays on the busy backend and is served there once the slot
// frees before its doubled age threshold elapses.
func TestQueueMigrationColdWaitsDoubleThreshold(t *testing.T) {
	withTempMetaDir(t)
	withMigrationAfter(t, 1.0) // cold requests need >= 2s to be migrated
	tokensP := seq(800)
	tokensR := offsetSeq(400, 200000)
	mA := newMockLlama(t, "cold-model", 32768, tokensP, 1)
	mB := newMockLlama(t, "cold-model", 32768, tokensP, 1)
	beA := backendKeyFromURL(mA.srv.URL)
	beB := backendKeyFromURL(mB.srv.URL)
	withTestBackend(t, []map[string]any{
		{"url": mA.srv.URL, "cache_dir": t.TempDir()},
		{"url": mB.srv.URL, "cache_dir": t.TempDir()},
	})
	bm := proxycache.GetBackendManager()
	markBackendsUp(bm)
	injectModels(bm, dm("cold-model", 32768, beA, beB))

	bm.Mu.Lock()
	bm.BackendState[beB] = false
	bm.Mu.Unlock()

	// A1: holds A's slot for 1.5s — enough to outlast the 1s disk-migration
	// threshold, short of R's 2s cold threshold.
	mA.tokens = tokensP
	mA.chatDelay = 1500 * time.Millisecond
	doneA1 := make(chan *httptest.ResponseRecorder, 1)
	go func() { doneA1 <- postChat(t, `{"model": "cold-model", "messages": [{"role": "user", "content": "busy"}]}`) }()
	waitFor(t, 5*time.Second, func() bool { return mA.chatCount() >= 1 })

	// R: cold (no cache anywhere), queued on A while B stays down.
	mA.tokens = tokensR
	doneR := make(chan *httptest.ResponseRecorder, 1)
	go func() { doneR <- postChat(t, `{"model": "cold-model", "messages": [{"role": "user", "content": "cold"}]}`) }()

	wR := <-doneR
	wA1 := <-doneA1
	if wR.Code != http.StatusOK || wA1.Code != http.StatusOK {
		t.Fatalf("statuses: R=%d A1=%d, want 200/200", wR.Code, wA1.Code)
	}
	if got := mA.chatCount(); got != 2 {
		t.Errorf("A chat count = %d, want 2 (cold request should wait for the slot, not migrate at 1x)", got)
	}
	if got := mB.chatCount(); got != 0 {
		t.Errorf("B chat count = %d, want 0 (nothing should have migrated)", got)
	}
}

// TestQueueMigrationRescansTargetDisk verifies the migration rescan for
// requests whose arrival scan found no disk hit: backend A is down when R
// arrives (so its cache is never scanned) and R queues on B, which has no
// cache for it. A comes back with a warm disk entry before R's doubled
// cold-migration age elapses — the dispatcher migrates R there and the worker
// restores from the target's local entry instead of recomputing.
func TestQueueMigrationRescansTargetDisk(t *testing.T) {
	withTempMetaDir(t)
	withMigrationAfter(t, 1.0) // cold requests need >= 2s to be migrated
	tokensW := seq(400)            // A's warm prefix (true prefix of tokensR)
	tokensR := seq(800)            // R's content
	tokensQ := offsetSeq(400, 100000)
	mA := newMockLlama(t, "mig-local", 32768, tokensW, 1)
	mB := newMockLlama(t, "mig-local", 32768, tokensQ, 1)
	beA := backendKeyFromURL(mA.srv.URL)
	beB := backendKeyFromURL(mB.srv.URL)
	dirA := t.TempDir()
	withTestBackend(t, []map[string]any{
		{"url": mA.srv.URL, "cache_dir": dirA},
		{"url": mB.srv.URL, "cache_dir": t.TempDir()},
	})
	bm := proxycache.GetBackendManager()
	markBackendsUp(bm)
	injectModels(bm, dm("mig-local", 32768, beA, beB))

	// Warmup: establishes a disk cache for tokensW on A while B is down.
	bm.Mu.Lock()
	bm.BackendState[beB] = false
	bm.Mu.Unlock()
	if w := postChat(t, `{"model": "mig-local", "messages": [{"role": "user", "content": "warm"}]}`); w.Code != http.StatusOK {
		t.Fatalf("warmup status = %d, want 200", w.Code)
	}
	keyW := proxycache.MetaKey("mig-local", tokensW)
	if err := os.WriteFile(filepath.Join(dirA, keyW), []byte("warm-cache-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Swap liveness: A (warm disk cache) goes down for R's arrival so its disk
	// is never scanned; B (no cache for R) comes up to absorb the work.
	bm.Mu.Lock()
	bm.BackendState[beA] = false
	bm.BackendState[beB] = true
	bm.Mu.Unlock()

	// B1: holds B's only slot for 4s — past R's 2s cold-migration threshold.
	mB.chatDelay = 4 * time.Second
	doneB1 := make(chan *httptest.ResponseRecorder, 1)
	go func() { doneB1 <- postChat(t, `{"model": "mig-local", "messages": [{"role": "user", "content": "busy"}]}`) }()
	waitFor(t, 5*time.Second, func() bool { return mB.chatCount() >= 1 })

	// R: cold everywhere it can see (A down, B's disk empty) -> queued on B.
	mB.tokens = tokensR
	doneR := make(chan *httptest.ResponseRecorder, 1)
	go func() { doneR <- postChat(t, `{"model": "mig-local", "messages": [{"role": "user", "content": "cold"}]}`) }()

	// Wait until R has actually been scanned and enqueued on B (A still down,
	// so its disk was never seen), only then bring A back up with the warm
	// cache — otherwise R would just get a direct hit at arrival.
	waitFor(t, 5*time.Second, func() bool { return proxycache.GetDispatcher().QueueDepth(beB) >= 1 })
	mA.tokens = tokensR
	bm.Mu.Lock()
	bm.BackendState[beA] = true
	bm.Mu.Unlock()

	wR := <-doneR
	wB1 := <-doneB1
	if wR.Code != http.StatusOK || wB1.Code != http.StatusOK {
		t.Fatalf("statuses: R=%d B1=%d, want 200/200", wR.Code, wB1.Code)
	}
	if got := mA.chatCount(); got != 2 {
		t.Errorf("A chat count = %d, want 2 (warmup + migrated request)", got)
	}
	if got := mB.chatCount(); got != 1 {
		t.Errorf("B chat count = %d, want 1 (the busy request; R should have migrated away)", got)
	}
	mA.mu.Lock()
	restores := len(mA.restores)
	mA.mu.Unlock()
	if restores < 1 {
		t.Error("migrated request did not restore the target's local disk cache")
	}
}

// TestRequeueWhenBackendStopsServingModel verifies that a worker whose backend
// no longer serves the requested model (model disappeared from the registry,
// e.g. after liveness re-discovery) hands the request back to the global
// overflow queue, where the matcher re-routes it to a live backend.
func TestRequeueWhenBackendStopsServingModel(t *testing.T) {
	withTempMetaDir(t)
	// Backend 1 serves the model but reports zero slots (pool never forms);
	// backend 2 will serve it once discovery is updated below.
	m1 := newMockLlama(t, "requeue-model", 32768, seq(10), 0)
	m2 := newMockLlama(t, "requeue-model", 32768, seq(10), 1)
	be1 := backendKeyFromURL(m1.srv.URL)
	be2 := backendKeyFromURL(m2.srv.URL)
	bm := withTestBackend(t, []map[string]any{
		{"url": m1.srv.URL, "cache_dir": t.TempDir()},
		{"url": m2.srv.URL, "cache_dir": t.TempDir()},
	})
	markBackendsUp(bm)
	injectModels(bm, dm("requeue-model", 32768, be1))
	withRetryBase(t, 0.2)
	// Simulate liveness discovery moving the model to backend 2 while the
	// worker is waiting for a slot on backend 1.
	go func() {
		time.Sleep(400 * time.Millisecond)
		bm.Mu.Lock()
		if info, ok := bm.DiscoveredModels["requeue-model"]; ok {
			info.Backends = []string{be2}
		}
		bm.Mu.Unlock()
		proxycache.GetSlotManager().RefreshSlotCounts()
	}()
	w := postChat(t, `{"model": "requeue-model", "messages": [{"role": "user", "content": "hello"}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if m2.chatCount() != 1 {
		t.Errorf("re-routed backend chat count = %d, want 1", m2.chatCount())
	}
	if m1.chatCount() != 0 {
		t.Errorf("departed backend chat count = %d, want 0", m1.chatCount())
	}
}

// TestStalePendingSlotFallsBackToDisk verifies that when the matcher's
// pending-slot hit is stale by the time the worker runs (a different free
// slot is acquired whose tracked KV state does not match), the worker
// restores from the local disk cache instead of recomputing from scratch.
func TestStalePendingSlotFallsBackToDisk(t *testing.T) {
	withTempMetaDir(t)
	tokens := seq(800)
	m := newMockLlama(t, "stale-model", 32768, tokens, 2) // two slots
	be := backendKeyFromURL(m.srv.URL)
	dir := t.TempDir()
	withTestBackend(t, []map[string]any{{"url": m.srv.URL, "cache_dir": dir}})
	markBackendsUp(proxycache.GetBackendManager())
	injectModels(proxycache.GetBackendManager(), dm("stale-model", 32768, be))
	blocks := proxycache.BlockHashesFromTokens(tokens, proxycache.WordsPerBlock)
	key := proxycache.MetaKey("stale-model", tokens)
	proxycache.GetKVMeta().WriteMeta(key, 800, blocks, proxycache.WordsPerBlock, "stale-model", be, 2048)
	if err := os.WriteFile(filepath.Join(dir, key), []byte("kv-cache-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}

	beSm := proxycache.GetSlotManager().Get(be)
	beSm.EnsurePool("stale-model", 2)
	// Simulate history: slot 1 is warm with this prompt (state tracked from a
	// previous completed request), slot 0 holds older, different content.
	beSm.SetKVState(1, blocks)
	beSm.SetKVState(0, blk("old", len(blocks)))

	w := postChat(t, `{"model": "stale-model", "messages": [{"role": "user", "content": "hello"}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	m.mu.Lock()
	restores := append([]string(nil), m.restores...)
	m.mu.Unlock()
	if len(restores) != 1 {
		t.Errorf("restore calls = %v, want 1 (disk fallback after stale pending-slot hit)", restores)
	}
}

// setupFlushScenario runs the common prefix of the clobber-flush tests: A0
// (cold) saves to disk, then A1 (one partial block longer) restores at ratio
// 8/9 > 0.8 so its own save is skipped, leaving its KV only in the slot.
func setupFlushScenario(t *testing.T) (*mockLlama, string, string, string, []int) {
	t.Helper()
	withTempMetaDir(t)
	tokensA0 := seq(800) // 8 blocks
	tokensA1 := seq(810) // 9 blocks: extends A0 by one partial block
	m := newMockLlama(t, "flush-model", 32768, tokensA0, 1)
	be := backendKeyFromURL(m.srv.URL)
	dir := t.TempDir()
	withTestBackend(t, []map[string]any{{"url": m.srv.URL, "cache_dir": dir}})
	markBackendsUp(proxycache.GetBackendManager())
	injectModels(proxycache.GetBackendManager(), dm("flush-model", 32768, be))
	body := `{"model": "flush-model", "messages": [{"role": "user", "content": "hello"}]}`

	m.tokens = tokensA0
	if w := postChat(t, body); w.Code != http.StatusOK {
		t.Fatalf("A0 status = %d, want 200", w.Code)
	}
	keyA0 := proxycache.MetaKey("flush-model", tokensA0)
	if meta := proxycache.GetKVMeta().ReadMeta(keyA0, be); meta == nil {
		t.Fatal("A0 was not saved to disk")
	}
	m.tokens = tokensA1
	if w := postChat(t, body); w.Code != http.StatusOK {
		t.Fatalf("A1 status = %d, want 200", w.Code)
	}
	keyA1 := proxycache.MetaKey("flush-model", tokensA1)
	if meta := proxycache.GetKVMeta().ReadMeta(keyA1, be); meta != nil {
		t.Fatalf("A1 should have skipped its save (high-ratio restore), but meta exists")
	}
	return m, be, dir, keyA0, tokensA1
}

// TestSkippedSaveNotFlushedWhenDiskCovers verifies the policy-consistent case:
// when the on-disk ancestor still covers the skipped content above the save
// threshold, a clobbering request does NOT flush it — saving would grow the
// ring (eviction risk) for a tail that is cheap to recompute.
func TestSkippedSaveNotFlushedWhenDiskCovers(t *testing.T) {
	m, be, _, _, tokensA1 := setupFlushScenario(t)
	keyA1 := proxycache.MetaKey("flush-model", tokensA1)

	m.tokens = offsetSeq(810, 100000) // unrelated prompt B takes the slot
	if w := postChat(t, `{"model": "flush-model", "messages": [{"role": "user", "content": "hello"}]}`); w.Code != http.StatusOK {
		t.Fatalf("B status = %d, want 200", w.Code)
	}
	if meta := proxycache.GetKVMeta().ReadMeta(keyA1, be); meta != nil {
		t.Error("A1 was flushed even though the disk ancestor still covers it (ring growth for a cheap tail)")
	}
	m.mu.Lock()
	saves := len(m.saves)
	m.mu.Unlock()
	if saves != 2 { // A0 save + B save; no clobber-flush
		t.Errorf("save calls = %d, want 2 (A0, B)", saves)
	}
}

// TestSkippedSaveFlushedOnClobberEvictedAncestor verifies the degraded case:
// when the on-disk ancestor has been evicted, a clobbering request must flush
// the skipped save — otherwise only recompute-from-nothing would remain.
func TestSkippedSaveFlushedOnClobberEvictedAncestor(t *testing.T) {
	m, be, _, keyA0, tokensA1 := setupFlushScenario(t)
	keyA1 := proxycache.MetaKey("flush-model", tokensA1)

	// Simulate ring eviction of the ancestor (the mock writes no physical
	// cache files, so dropping the meta is what breaks scan coverage).
	if !proxycache.GetKVMeta().DeleteMetaFile(keyA0) {
		t.Fatalf("failed to delete ancestor meta")
	}

	m.tokens = offsetSeq(810, 100000) // unrelated prompt B takes the slot
	if w := postChat(t, `{"model": "flush-model", "messages": [{"role": "user", "content": "hello"}]}`); w.Code != http.StatusOK {
		t.Fatalf("B status = %d, want 200", w.Code)
	}
	metaA1 := proxycache.GetKVMeta().ReadMeta(keyA1, be)
	if metaA1 == nil {
		t.Fatal("A1's skipped cache was not flushed before the slot was clobbered (ancestor evicted)")
	}
	if got := int(proxycache.MetaFloat(metaA1, "n_tokens")); got != 810 {
		t.Errorf("flushed meta n_tokens = %d, want 810", got)
	}
	m.mu.Lock()
	saves := len(m.saves)
	m.mu.Unlock()
	if saves != 3 { // A0 save + A1 clobber-flush + B save
		t.Errorf("save calls = %d, want 3 (A0, clobber-flush of A1, B)", saves)
	}
}

// TestWorkerAcquiresAfterSlotRelease verifies that a request routed to the
// cache-hit backend waits (retry backoff) in its worker until the slot frees,
// instead of being bounced to another backend.
func TestWorkerAcquiresAfterSlotRelease(t *testing.T) {
	withTempMetaDir(t)
	m1 := newMockLlama(t, "test-model", 32768, seq(800), 1)
	m2 := newMockLlama(t, "test-model", 32768, seq(800), 1)
	be1, _ := setupBusyRestoreBackend(t, m1, m2, "test-model")
	withRetryBase(t, 0.2)
	go func() {
		time.Sleep(300 * time.Millisecond)
		proxycache.GetSlotManager().Get(be1).Release(0)
	}()
	w := postChat(t, `{"model": "test-model", "messages": [{"role": "user", "content": "hello"}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if m1.chatCount() != 1 {
		t.Errorf("restore backend chat count = %d, want 1", m1.chatCount())
	}
	if m2.chatCount() != 0 {
		t.Errorf("fallback backend chat count = %d, want 0", m2.chatCount())
	}
}

// TestQueueCapOverflowGlobalQueue verifies the per-backend queue cap (5) and
// the global overflow queue: with one slow single-slot backend, queued +
// in-flight requests exceed the cap and the excess is parked globally, then
// processed once capacity frees.
func TestQueueCapOverflowGlobalQueue(t *testing.T) {
	withTempMetaDir(t)
	m := newMockLlama(t, "cap-model", 32768, seq(10), 1)
	m.chatDelay = 500 * time.Millisecond
	be := backendKeyFromURL(m.srv.URL)
	withTestBackend(t, []map[string]any{{"url": m.srv.URL, "cache_dir": t.TempDir()}})
	markBackendsUp(proxycache.GetBackendManager())
	injectModels(proxycache.GetBackendManager(), dm("cap-model", 32768, be))
	body := `{"model": "cap-model", "messages": [{"role": "user", "content": "hello"}]}`

	codes := make(chan int, 7)
	var wg sync.WaitGroup
	for i := 0; i < 7; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			codes <- postChat(t, body).Code
		}()
	}

	sawOverflow := false
	for m.chatCount() < 7 {
		if proxycache.GetDispatcher().OverflowLen() > 0 {
			sawOverflow = true
		}
		time.Sleep(20 * time.Millisecond)
	}
	wg.Wait()
	for i := 0; i < 7; i++ {
		if c := <-codes; c != http.StatusOK {
			t.Errorf("request %d status = %d, want 200", i+1, c)
		}
	}
	if !sawOverflow {
		t.Error("global overflow queue was never used, want at least one parked request")
	}
	if got := m.chatCount(); got != 7 {
		t.Errorf("chat count = %d, want 7", got)
	}
	if got := proxycache.GetDispatcher().QueueDepth(be); got != 0 {
		t.Errorf("queue depth after drain = %d, want 0", got)
	}
	if got := proxycache.GetDispatcher().OverflowLen(); got != 0 {
		t.Errorf("overflow queue after drain = %d, want 0", got)
	}
}

// TestMultiSlotConcurrency verifies the pump serves one in-flight request per
// free slot: with a 2-slot backend, a second request dispatched while the
// first is still running overlaps it instead of waiting in its queue.
func TestMultiSlotConcurrency(t *testing.T) {
	withTempMetaDir(t)
	m := newMockLlama(t, "conc-model", 32768, seq(200), 2)
	m.chatDelay = 400 * time.Millisecond
	be := backendKeyFromURL(m.srv.URL)
	withTestBackend(t, []map[string]any{{"url": m.srv.URL, "cache_dir": t.TempDir()}})
	markBackendsUp(proxycache.GetBackendManager())
	injectModels(proxycache.GetBackendManager(), dm("conc-model", 32768, be))

	body1 := `{"model": "conc-model", "messages": [{"role": "user", "content": "one"}]}`
	body2 := `{"model": "conc-model", "messages": [{"role": "user", "content": "two"}]}`
	done1 := make(chan *httptest.ResponseRecorder, 1)
	go func() { done1 <- postChat(t, body1) }()

	// Wait until request 1 is in flight on the backend (holding a slot).
	waitFor(t, 5*time.Second, func() bool { return m.chatCount() >= 1 })

	w2 := postChat(t, body2) // must dispatch to the other free slot in parallel
	w1 := <-done1

	if w1.Code != http.StatusOK || w2.Code != http.StatusOK {
		t.Fatalf("statuses = %d/%d, want 200/200", w1.Code, w2.Code)
	}
	if mc := m.maxChatConcurrency(); mc < 2 {
		t.Fatalf("max concurrent backend chats = %d, want 2 (free slot should be served in parallel)", mc)
	}
}

// TestPumpDiscardDoesNotLeakSlot guards against the dispatcher lockup. A request
// that queues behind an in-flight one and whose client disconnects right before
// its turn causes the pump to pre-acquire the (now free) slot, see the already
// cancelled context, and discard the request. The pump must release the slot it
// just acquired: discarding without doing so leaks the slot (it stays inUse
// forever), so once every slot on a backend has leaked, TryAcquire always fails
// and the backend can never dispatch again — its queue fills to the cap with zero
// in-flight work while idle backends sit unused.
func TestPumpDiscardDoesNotLeakSlot(t *testing.T) {
	withTempMetaDir(t)
	const model = "leak-model"
	m := newMockLlama(t, model, 32768, seq(10), 1) // single slot
	m.chatDelay = time.Second
	be := backendKeyFromURL(m.srv.URL)
	withTestBackend(t, []map[string]any{{"url": m.srv.URL, "cache_dir": t.TempDir()}})
	bm := proxycache.GetBackendManager()
	markBackendsUp(bm)
	injectModels(bm, dm(model, 32768, be))
	beSm := proxycache.GetSlotManager().Get(be)
	// Pre-create the pool so both requests take the normal acquire path (no lazy
	// discovery) and the test is not dependent on liveness timing.
	beSm.EnsurePool(model, 1)
	body := `{"model": "leak-model", "messages": [{"role": "user", "content": "hello"}]}`

	// Request A grabs the only slot and holds it for chatDelay.
	doneA := make(chan int, 1)
	go func() { doneA <- postChat(t, body).Code }()
	waitFor(t, 5*time.Second, func() bool { return m.chatCount() >= 1 })

	// Request B arrives with a cancellable client context and queues behind A.
	ctxB, cancelB := context.WithCancel(context.Background())
	reqB := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	reqB.Header.Set("Content-Type", "application/json")
	reqB = reqB.WithContext(ctxB)
	wB := httptest.NewRecorder()
	doneB := make(chan struct{})
	go func() { proxycache.ChatHandler(wB, reqB); close(doneB) }()
	waitFor(t, 5*time.Second, func() bool {
		return m.chatCount() == 1 && proxycache.GetDispatcher().QueueDepth(be) >= 1
	})

	// B's client disconnects while it is still queued. A then finishes and
	// frees the slot; the pump dispatches B, finds its context already
	// cancelled, and discards it — releasing the slot it just acquired.
	// Note: ChatHandler returns as soon as the client context is cancelled,
	// so doneB fires *before* the dispatcher has processed B's cancellation.
	// The slot is legitimately in use for a few microseconds while the pump
	// pre-acquires-then-releases it, so assert on the settled state (queue
	// drained, nothing in use) rather than racing that window: a real leak
	// would never settle to 0.
	cancelB()
	codeA := <-doneA
	<-doneB
	waitFor(t, 5*time.Second, func() bool {
		return proxycache.GetDispatcher().QueueDepth(be) == 0 && beSm.CountInUse(model) == 0
	})

	if got := beSm.CountInUse(model); got != 0 {
		t.Fatalf("slot leaked after a queued request was discarded: CountInUse=%d, want 0", got)
	}
	if got := m.chatCount(); got != 1 {
		t.Errorf("backend chat count = %d, want 1 (discarded request must not be dispatched)", got)
	}
	if codeA != http.StatusOK {
		t.Errorf("request A status = %d, want 200", codeA)
	}
}

func TestChatSaveSkippedWhenRatioAboveThreshold(t *testing.T) {
	withTempMetaDir(t)
	reqTokens := seq(800)
	m := newMockLlama(t, "test-model", 32768, reqTokens, 1)
	m.chatResp = map[string]any{
		"object":  "chat.completion",
		"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "ok"}}},
		"usage": map[string]any{
			"prompt_tokens":         800,
			"prompt_tokens_details": map[string]any{"cached_tokens": 800},
		},
	}
	be := backendKeyFromURL(m.srv.URL)
	withTestBackend(t, []map[string]any{{"url": m.srv.URL, "cache_dir": t.TempDir()}})
	markBackendsUp(proxycache.GetBackendManager())
	reqBlocks := proxycache.BlockHashesFromTokens(reqTokens, proxycache.WordsPerBlock)
	candBlocks := append(append([]string{}, reqBlocks[:7]...), "different_block")
	candTokens := append(append([]int{}, seq(700)...), offsetSeq(100, 10000)...)
	proxycache.GetKVMeta().WriteMeta(proxycache.MetaKey("test-model", candTokens), 800, candBlocks, proxycache.WordsPerBlock, "test-model", be, 2048)
	w := postChat(t, `{"model": "test-model", "messages": [{"role": "user", "content": "hello"}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if len(m.saves) != 0 {
		t.Errorf("save calls = %d, want 0 (ratio 0.875 > 0.8)", len(m.saves))
	}
	if meta := proxycache.GetKVMeta().ReadMeta(proxycache.MetaKey("test-model", reqTokens), be); meta != nil {
		t.Errorf("new meta written for full request key, want none")
	}
}

func TestChatSavePerformedWhenRatioBelowThreshold(t *testing.T) {
	withTempMetaDir(t)
	reqTokens := seq(800)
	m := newMockLlama(t, "test-model", 32768, reqTokens, 1)
	m.chatResp = map[string]any{
		"object":  "chat.completion",
		"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "ok"}}},
		"usage": map[string]any{
			"prompt_tokens":         800,
			"prompt_tokens_details": map[string]any{"cached_tokens": 800},
		},
	}
	be := backendKeyFromURL(m.srv.URL)
	withTestBackend(t, []map[string]any{{"url": m.srv.URL, "cache_dir": t.TempDir()}})
	markBackendsUp(proxycache.GetBackendManager())
	reqBlocks := proxycache.BlockHashesFromTokens(reqTokens, proxycache.WordsPerBlock)
	candBlocks := append(append([]string{}, reqBlocks[:4]...), blk("cand9", 4)...)
	candTokens := append(append([]int{}, seq(400)...), offsetSeq(400, 10000)...)
	proxycache.GetKVMeta().WriteMeta(proxycache.MetaKey("test-model", candTokens), 800, candBlocks, proxycache.WordsPerBlock, "test-model", be, 2048)
	w := postChat(t, `{"model": "test-model", "messages": [{"role": "user", "content": "hello"}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if len(m.saves) != 1 {
		t.Fatalf("save calls = %d, want 1 (ratio 0.5 <= 0.8)", len(m.saves))
	}
	if meta := proxycache.GetKVMeta().ReadMeta(proxycache.MetaKey("test-model", reqTokens), be); meta == nil {
		t.Error("no meta written for full request key, want one after save")
	}
}

// splitReader delivers its parts one Read per call, to control chunk boundaries.
type splitReader struct {
	parts [][]byte
	idx   int
}

func (r *splitReader) Read(p []byte) (int, error) {
	if r.idx >= len(r.parts) {
		return 0, io.EOF
	}
	n := copy(p, r.parts[r.idx])
	r.idx++
	return n, nil
}

func (r *splitReader) Close() error { return nil }

func TestReadLoopForwardsDoneChunk(t *testing.T) {
	cases := []struct {
		name  string
		parts [][]byte
	}{
		{"done in own chunk", [][]byte{
			[]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"),
			[]byte("data: [DONE]\n\n"),
		}},
		{"done coalesced with content", [][]byte{
			[]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"),
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ss := &proxycache.StreamState{
				Resp:   &http.Response{Body: &splitReader{parts: tc.parts}},
				Chunks: make(chan []byte, 8),
				Done:   make(chan struct{}),
			}
			go ss.ReadLoop()
			<-ss.Done
			// ReadLoop enqueues the [DONE] chunk before closing done, so the
			// channel is fully populated here.
			var all []byte
			for {
				select {
				case c := <-ss.Chunks:
					all = append(all, c...)
				default:
					goto drained
				}
			}
		drained:
			if !strings.Contains(string(all), "[DONE]") {
				t.Errorf("client stream missing data: [DONE]: %q", all)
			}
			if !strings.Contains(string(all), "hi") {
				t.Errorf("client stream missing content: %q", all)
			}
			if !ss.StreamComplete {
				t.Error("StreamComplete = false, want true")
			}
		})
	}
}
