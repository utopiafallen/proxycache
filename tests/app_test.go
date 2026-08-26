package tests

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"proxycache"
	"strings"
	"sync"
	"testing"
	"time"
)

type mockLlama struct {
	mu         sync.Mutex
	srv        *httptest.Server
	model      string
	nCtx       int
	tokens     []int
	nSlots     int
	chatBodies []map[string]any
	saves      []string
	restores   []string
	chatResp   map[string]any
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

func TestNoCacheLRUBackendRouting(t *testing.T) {
	withTempMetaDir(t)
	m1 := newMockLlama(t, "model-a", 32768, seq(10), 1)
	m2 := newMockLlama(t, "model-a", 32768, seq(10), 1)
	m3 := newMockLlama(t, "model-a", 32768, seq(10), 1)
	bm := withTestBackend(t, []map[string]any{
		{"url": m1.srv.URL, "cache_dir": t.TempDir()},
		{"url": m2.srv.URL, "cache_dir": t.TempDir()},
		{"url": m3.srv.URL, "cache_dir": t.TempDir()},
	})
	markBackendsUp(bm)
	be1 := backendKeyFromURL(m1.srv.URL)
	be2 := backendKeyFromURL(m2.srv.URL)
	be3 := backendKeyFromURL(m3.srv.URL)
	injectModels(bm, dm("model-a", 32768, be1, be2, be3))
	bm.TouchBackend(be2)
	time.Sleep(25 * time.Millisecond)
	bm.TouchBackend(be3)
	time.Sleep(25 * time.Millisecond)
	bm.TouchBackend(be1)
	w := postChat(t, `{"model": "model-a", "messages": [{"role": "user", "content": "hello"}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if m2.chatCount() != 1 {
		t.Errorf("oldest backend chat count = %d, want 1", m2.chatCount())
	}
	if m1.chatCount() != 0 {
		t.Errorf("newest backend chat count = %d, want 0", m1.chatCount())
	}
	if m3.chatCount() != 0 {
		t.Errorf("middle backend chat count = %d, want 0", m3.chatCount())
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

func TestCacheBackendBusyFallback(t *testing.T) {
	withTempMetaDir(t)
	m1 := newMockLlama(t, "test-model", 32768, seq(800), 1)
	m2 := newMockLlama(t, "test-model", 32768, seq(800), 1)
	be1, be2 := setupBusyRestoreBackend(t, m1, m2, "test-model")
	withEMACfg(t, 0.2, 1, 1, 2)
	w := postChat(t, `{"model": "test-model", "messages": [{"role": "user", "content": "hello"}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if m2.chatCount() != 1 {
		t.Errorf("fallback backend chat count = %d, want 1", m2.chatCount())
	}
	if m1.chatCount() != 0 {
		t.Errorf("busy backend chat count = %d, want 0", m1.chatCount())
	}
	if got := proxycache.GetSlotManager().GetCacheWaitPending(be1); got != 0 {
		t.Errorf("pending after fallback = %d, want 0", got)
	}
	if got := proxycache.GetSlotManager().GetCacheWaitPending(be2); got != 0 {
		t.Errorf("pending on fallback backend = %d, want 0", got)
	}
}

func TestCacheHitWaitPhase0Success(t *testing.T) {
	withTempMetaDir(t)
	m1 := newMockLlama(t, "test-model", 32768, seq(800), 1)
	m2 := newMockLlama(t, "test-model", 32768, seq(800), 1)
	be1, _ := setupBusyRestoreBackend(t, m1, m2, "test-model")
	beSm1 := proxycache.GetSlotManager().Get(be1)
	go func() {
		time.Sleep(300 * time.Millisecond)
		beSm1.Release(0)
	}()
	withEMACfg(t, 0.2, 1, 1, 2)
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
	if got := proxycache.GetSlotManager().GetCacheWaitPending(be1); got != 0 {
		t.Errorf("pending after acquire = %d, want 0", got)
	}
}

func TestCacheHitWaitPendingCountBlocks(t *testing.T) {
	withTempMetaDir(t)
	m1 := newMockLlama(t, "test-model", 32768, seq(800), 1)
	m2 := newMockLlama(t, "test-model", 32768, seq(800), 1)
	be1, _ := setupBusyRestoreBackend(t, m1, m2, "test-model")
	proxycache.GetSlotManager().SetCacheWaitPending(be1, proxycache.CacheHitWaitMaxPending)
	w := postChat(t, `{"model": "test-model", "messages": [{"role": "user", "content": "hello"}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if m2.chatCount() != 1 {
		t.Errorf("fallback backend chat count = %d, want 1", m2.chatCount())
	}
	if m1.chatCount() != 0 {
		t.Errorf("busy backend chat count = %d, want 0", m1.chatCount())
	}
	if got := proxycache.GetSlotManager().GetCacheWaitPending(be1); got != proxycache.CacheHitWaitMaxPending {
		t.Errorf("pending = %d, want %d (wait skipped)", got, proxycache.CacheHitWaitMaxPending)
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
