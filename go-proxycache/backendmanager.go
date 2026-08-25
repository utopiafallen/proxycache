// backendmanager.go — singleton backend registry: clients, agent clients,
// cache helpers, model discovery, latency EMAs, and the liveness checker.
//
// Key derivation: strips protocol, keeps host:port, colons -> dashes.
// e.g. "http://10.0.0.1:8000" -> "10.0.0.1-8000"

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// DiscoveredModel is one row of the merged model registry.
type DiscoveredModel struct {
	Name           string
	NCtx           int
	Backends       []string
	BackendNCTX    map[string]int
	TotalSlots     int
	LastDiscovered float64
	Synthetic      bool
}

// BackendInfo holds the clients and cache config for one backend.
type BackendInfo struct {
	Client         *LlamaClient
	AgentClient    *CacheAgentClient
	CacheDir       string // "" when the backend uses a remote agent
	CacheMaxSizeGB float64
}

type backendModelKey struct {
	Model   string
	Backend string
}

type refreshEntry struct {
	TS     float64
	OK     bool
	Slots  int
}

type modelEMAKey struct {
	BackendID string
	ModelName string
	ReqType   string
}

// BackendManager is the singleton registry. Backends are configured once at
// startup and never change.
type BackendManager struct {
	mu sync.RWMutex

	backends               map[string]*BackendInfo
	keyOrder               []string
	firstKey               string
	refreshState           map[backendModelKey]refreshEntry
	discoveredModels       map[string]*DiscoveredModel
	modelOrder             []string // first-seen insertion order (dict-order parity)
	backendState           map[string]bool
	backendLastUsed        map[string]float64
	backendLatencyEMA      map[string]float64
	backendModelLatencyEMA map[modelEMAKey]float64
	lastDiscoverTiming     []map[string]any

	livenessCancel context.CancelFunc
}

var lcpSplitRe = regexp.MustCompile(`([-_])`)

// NewBackendManager parses the backend config (same validation as Python:
// cache_dir and agent_port are mutually exclusive; exactly one required when
// cache_max_size_gb > 0).
func NewBackendManager(backendConfig []map[string]any) *BackendManager {
	bm := &BackendManager{
		backends:               map[string]*BackendInfo{},
		refreshState:           map[backendModelKey]refreshEntry{},
		discoveredModels:       map[string]*DiscoveredModel{},
		backendState:           map[string]bool{},
		backendLastUsed:        map[string]float64{},
		backendLatencyEMA:      map[string]float64{},
		backendModelLatencyEMA: map[modelEMAKey]float64{},
	}
	for _, be := range backendConfig {
		urlStr, _ := be["url"].(string)
		u := strings.TrimRight(urlStr, "/")
		rawKey := u
		if i := strings.LastIndex(u, "://"); i >= 0 {
			rawKey = u[i+3:]
		}
		key := SanitizeBackendDir(rawKey)
		client := NewLlamaClient(urlStr)
		var agentClient *CacheAgentClient
		cacheDir, _ := be["cache_dir"].(string)
		_, hasAgentPort := be["agent_port"]
		if hasAgentPort && cacheDir != "" {
			panic(fmt.Sprintf("Backend %s: cache_dir and agent_port are mutually exclusive. "+
				"Use cache_dir for local cache management or agent_port for remote cache-agent.", urlStr))
		}
		cacheMaxSizeGB := 25.0
		if v, ok := toFloat(be["cache_max_size_gb"]); ok {
			cacheMaxSizeGB = v
		}
		if !hasAgentPort && cacheDir == "" && cacheMaxSizeGB > 0 {
			panic(fmt.Sprintf("Backend %s: must specify either cache_dir or agent_port. "+
				"cache_dir for local filesystem cache management, agent_port for remote cache-agent.", urlStr))
		}
		if hasAgentPort {
			host := rawKey
			if i := strings.LastIndex(rawKey, ":"); i >= 0 {
				host = rawKey[:i]
			}
			agentClient = NewCacheAgentClient(fmt.Sprintf("http://%s:%v", host, be["agent_port"]))
		}
		bm.backends[key] = &BackendInfo{
			Client:         client,
			AgentClient:    agentClient,
			CacheDir:       cacheDir,
			CacheMaxSizeGB: cacheMaxSizeGB,
		}
		bm.keyOrder = append(bm.keyOrder, key)
		if bm.firstKey == "" {
			bm.firstKey = key
		}
	}
	logInfo("backend_manager", "Backend manager initialized with %d backends: %s", len(bm.backends), bm.keyOrder)
	bm.LoadLatencyData()
	return bm
}

func init() {
	// Ensure BACKENDS config is loaded (config.go's init may not have run
	// yet — file order is not guaranteed).
	if len(Backends) == 0 {
		initBackends()
	}
	backendManager = NewBackendManager(Backends)
}

// --- Accessors ---

// GetClient returns the LlamaClient for a backend key (panics if unknown,
// mirroring the Python KeyError).
func (bm *BackendManager) GetClient(key string) *LlamaClient {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	be := bm.backends[key]
	if be == nil {
		panic(fmt.Sprintf("Unknown backend key: %s", key))
	}
	return be.Client
}

// GetAgent returns the CacheAgentClient for a backend (nil for local-cache
// backends; panics if the backend is unknown).
func (bm *BackendManager) GetAgent(key string) *CacheAgentClient {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	be := bm.backends[key]
	if be == nil {
		panic(fmt.Sprintf("Unknown backend key: %s", key))
	}
	return be.AgentClient
}

// GetCacheDir returns the local cache dir ("" for agent backends).
func (bm *BackendManager) GetCacheDir(key string) string {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	be := bm.backends[key]
	if be == nil {
		panic(fmt.Sprintf("Unknown backend key: %s", key))
	}
	return be.CacheDir
}

// HasCacheConfig reports whether the backend has any cache management config.
func (bm *BackendManager) HasCacheConfig(key string) bool {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	be := bm.backends[key]
	if be == nil {
		panic(fmt.Sprintf("Unknown backend key: %s", key))
	}
	return be.AgentClient != nil || be.CacheDir != ""
}

// GetCacheMaxSizeGB returns the configured cache budget (25 GB default for
// unknown backends, mirroring the Python getattr default).
func (bm *BackendManager) GetCacheMaxSizeGB(key string) float64 {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	be := bm.backends[key]
	if be == nil {
		return 25.0
	}
	return be.CacheMaxSizeGB
}

// CacheEnabled is false when cache_max_size_gb is 0 (cache management off).
func (bm *BackendManager) CacheEnabled(key string) bool {
	return bm.GetCacheMaxSizeGB(key) > 0
}

// GetBackendState returns the liveness flag for a backend (false if unknown),
// mirroring _backend_state.get(key, False).
func (bm *BackendManager) GetBackendState(key string) bool {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	return bm.backendState[key]
}

// GetRefreshTS returns the last slot-refresh timestamp for a (model, backend)
// pair (0 if never refreshed), mirroring _refresh_state.get((m, k), (0, True, 0))[0].
func (bm *BackendManager) GetRefreshTS(modelName, backendKey string) float64 {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	return bm.refreshState[backendModelKey{Model: modelName, Backend: backendKey}].TS
}

// BackendInfoView is a safe copy of a backend's public config for the health
// endpoint.
type BackendInfoView struct {
	URL        string
	CacheDir   string
	HasAgent   bool
	MaxSizeGB  float64
}

// GetBackendInfo returns a copy of a backend's public config, or nil if unknown.
func (bm *BackendManager) GetBackendInfo(key string) *BackendInfoView {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	be := bm.backends[key]
	if be == nil {
		return nil
	}
	return &BackendInfoView{
		URL:       be.Client.BaseURL(),
		CacheDir:  be.CacheDir,
		HasAgent:  be.AgentClient != nil,
		MaxSizeGB: be.CacheMaxSizeGB,
	}
}

// SnapshotModels returns the discovered-model registry in first-seen insertion
// order, with each row's Backends slice and BackendNCTX map deep-copied so the
// caller can mutate the result freely.
func (bm *BackendManager) SnapshotModels() []DiscoveredModel {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	out := make([]DiscoveredModel, 0, len(bm.modelOrder))
	for _, name := range bm.modelOrder {
		info := bm.discoveredModels[name]
		if info == nil {
			continue
		}
		backends := make([]string, len(info.Backends))
		copy(backends, info.Backends)
		backendNCTX := make(map[string]int, len(info.BackendNCTX))
		for k, v := range info.BackendNCTX {
			backendNCTX[k] = v
		}
		out = append(out, DiscoveredModel{
			Name:           info.Name,
			NCtx:           info.NCtx,
			Backends:       backends,
			BackendNCTX:    backendNCTX,
			TotalSlots:     info.TotalSlots,
			LastDiscovered: info.LastDiscovered,
			Synthetic:      info.Synthetic,
		})
	}
	return out
}

// TouchBackend marks a backend as recently used.
func (bm *BackendManager) TouchBackend(backendID string) {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	bm.backendLastUsed[backendID] = nowFloat()
}

// GetBackendLastUsed returns the last-used timestamp (0 if never used).
func (bm *BackendManager) GetBackendLastUsed(backendID string) float64 {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	return bm.backendLastUsed[backendID]
}

// UpdateBackendLatency updates the per-backend latency EMA and persists it.
func (bm *BackendManager) UpdateBackendLatency(backendID string, latencyMS float64) {
	bm.mu.Lock()
	old := bm.backendLatencyEMA[backendID]
	if old == 0 {
		old = latencyMS
	}
	bm.backendLatencyEMA[backendID] = CacheHitWaitEMAAlpha*latencyMS + (1-CacheHitWaitEMAAlpha)*old
	bm.mu.Unlock()
	bm.saveLatencyData(backendID)
}

// GetBackendLatencyEMA returns the per-backend latency EMA (0 if unset).
func (bm *BackendManager) GetBackendLatencyEMA(backendID string) float64 {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	return bm.backendLatencyEMA[backendID]
}

// UpdateBackendModelLatency updates the per backend/model/type latency EMA.
func (bm *BackendManager) UpdateBackendModelLatency(backendID, modelName string, latencyMS float64, reqType string) {
	k := modelEMAKey{backendID, modelName, reqType}
	bm.mu.Lock()
	old := bm.backendModelLatencyEMA[k]
	if old == 0 {
		old = latencyMS
	}
	bm.backendModelLatencyEMA[k] = CacheHitWaitEMAAlpha*latencyMS + (1-CacheHitWaitEMAAlpha)*old
	bm.mu.Unlock()
	bm.saveLatencyData(backendID)
}

// GetBackendModelLatencyEMA returns the per backend/model/type latency EMA,
// falling back to the per-backend EMA, then 0.
func (bm *BackendManager) GetBackendModelLatencyEMA(backendID, modelName, reqType string) float64 {
	k := modelEMAKey{backendID, modelName, reqType}
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	if v, ok := bm.backendModelLatencyEMA[k]; ok {
		return v
	}
	return bm.backendLatencyEMA[backendID]
}

// GetAllLatencyEMA returns all non-zero per-model EMAs as
// {backend: {model: {reqType: ema_ms}}}.
func (bm *BackendManager) GetAllLatencyEMA() map[string]map[string]map[string]float64 {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	result := map[string]map[string]map[string]float64{}
	for k, ema := range bm.backendModelLatencyEMA {
		if ema <= 0 {
			continue
		}
		if result[k.BackendID] == nil {
			result[k.BackendID] = map[string]map[string]float64{}
		}
		if result[k.BackendID][k.ModelName] == nil {
			result[k.BackendID][k.ModelName] = map[string]float64{}
		}
		result[k.BackendID][k.ModelName][k.ReqType] = round1(ema)
	}
	return result
}

// --- Persisted latency data ---

func (bm *BackendManager) perfFile(backendID string) string {
	beDir := filepath.Join(MetaDir, SanitizeBackendDir(backendID))
	return filepath.Join(beDir, "backend_perf.json")
}

// LoadLatencyData loads persisted latency data for all backends.
func (bm *BackendManager) LoadLatencyData() {
	for _, backendID := range bm.keys() {
		perfFile := bm.perfFile(backendID)
		if fi, err := os.Stat(perfFile); err != nil || fi.IsDir() {
			continue
		}
		data, err := os.ReadFile(perfFile)
		if err != nil {
			logWarn("backend_manager", "Failed to load latency data for backend '%s': %s", backendID, err)
			continue
		}
		var parsed struct {
			LatencyEMA       map[string]float64 `json:"latency_ema"`
			ModelLatencyEMA  map[string]float64 `json:"model_latency_ema"`
		}
		if err := json.Unmarshal(data, &parsed); err != nil {
			logWarn("backend_manager", "Failed to load latency data for backend '%s': %s", backendID, err)
			continue
		}
		bm.mu.Lock()
		if v, ok := parsed.LatencyEMA[backendID]; ok {
			bm.backendLatencyEMA[backendID] = v
		}
		for keyStr, value := range parsed.ModelLatencyEMA {
			parts := strings.Split(keyStr, "|")
			if len(parts) == 3 {
				bm.backendModelLatencyEMA[modelEMAKey{parts[0], parts[1], parts[2]}] = value
			}
		}
		nModelEntries := len(bm.backendModelLatencyEMA)
		beEMA := bm.backendLatencyEMA[backendID]
		bm.mu.Unlock()
		logInfo("backend_manager", "Loaded latency data for backend '%s': ema=%.0fms, %d model-type entries",
			backendID, beEMA, nModelEntries)
	}
}

// saveLatencyData persists latency data for one backend (atomic tmp+rename).
func (bm *BackendManager) saveLatencyData(backendID string) {
	perfFile := bm.perfFile(backendID)
	if err := os.MkdirAll(filepath.Dir(perfFile), 0o755); err != nil {
		logWarn("backend_manager", "Failed to save latency data for backend '%s': %s", backendID, err)
		return
	}
	bm.mu.RLock()
	modelEMA := map[string]float64{}
	for k, v := range bm.backendModelLatencyEMA {
		if k.BackendID == backendID {
			modelEMA[k.BackendID+"|"+k.ModelName+"|"+k.ReqType] = v
		}
	}
	beEMA := bm.backendLatencyEMA[backendID]
	bm.mu.RUnlock()
	data := map[string]any{
		"latency_ema":       map[string]float64{backendID: beEMA},
		"model_latency_ema": modelEMA,
	}
	raw, err := json.Marshal(data)
	if err != nil {
		logWarn("backend_manager", "Failed to save latency data for backend '%s': %s", backendID, err)
		return
	}
	tmpFile := perfFile + ".tmp"
	if err := os.WriteFile(tmpFile, raw, 0o644); err != nil {
		logWarn("backend_manager", "Failed to save latency data for backend '%s': %s", backendID, err)
		return
	}
	if err := os.Rename(tmpFile, perfFile); err != nil {
		logWarn("backend_manager", "Failed to save latency data for backend '%s': %s", backendID, err)
	}
}

// --- Cache helpers (agent or local filesystem) ---

// CacheDelete deletes a cache file (and its .ckpt* sidecars).
func (bm *BackendManager) CacheDelete(backendID, key string) bool {
	bm.mu.RLock()
	be := bm.backends[backendID]
	bm.mu.RUnlock()
	if be == nil {
		return false
	}
	if be.AgentClient != nil {
		return be.AgentClient.Delete(key)
	}
	if be.CacheDir != "" {
		cachePath := filepath.Join(be.CacheDir, key)
		if _, err := os.Stat(cachePath); err == nil {
			os.Remove(cachePath)
			// Delete ckpt sidecar files (<key>.ckpt, <key>.ckpt.0, etc.)
			for _, sidecar := range globSidecars(be.CacheDir, key+".ckpt*") {
				os.Remove(sidecar)
			}
			return true
		}
	}
	return false
}

// CacheGetSize returns the cache file size (0 if missing/unknown backend).
func (bm *BackendManager) CacheGetSize(backendID, key string) int {
	bm.mu.RLock()
	be := bm.backends[backendID]
	bm.mu.RUnlock()
	if be == nil {
		return 0
	}
	if be.AgentClient != nil {
		result := be.AgentClient.GetFileSize(key)
		if result != nil {
			if exists, _ := result["exists"].(bool); exists {
				if v, ok := toFloat(result["size"]); ok {
					return int(v)
				}
			}
		}
		return 0
	}
	if be.CacheDir != "" {
		if st, err := os.Stat(filepath.Join(be.CacheDir, key)); err == nil {
			return int(st.Size())
		}
	}
	return 0
}

// CacheGetMtime returns the cache file mtime. Agent backends have no mtime,
// so they (and missing files) return the current time.
func (bm *BackendManager) CacheGetMtime(backendID, key string) float64 {
	bm.mu.RLock()
	be := bm.backends[backendID]
	bm.mu.RUnlock()
	if be == nil {
		return nowFloat()
	}
	if be.CacheDir != "" {
		if st, err := os.Stat(filepath.Join(be.CacheDir, key)); err == nil {
			return float64(st.ModTime().UnixNano()) / 1e9
		}
	}
	return nowFloat()
}

// CacheExists checks for the cache file.
func (bm *BackendManager) CacheExists(backendID, key string) bool {
	bm.mu.RLock()
	be := bm.backends[backendID]
	bm.mu.RUnlock()
	if be == nil {
		return false
	}
	if be.AgentClient != nil {
		if result := be.AgentClient.GetFileSize(key); result != nil {
			exists, _ := result["exists"].(bool)
			return exists
		}
		return false
	}
	if be.CacheDir != "" {
		if _, err := os.Stat(filepath.Join(be.CacheDir, key)); err == nil {
			return true
		}
	}
	return false
}

func globSidecars(dir, pattern string) []string {
	matches, _ := filepath.Glob(filepath.Join(dir, pattern))
	return matches
}

// --- Registry basics ---

// Keys returns backend keys in config order.
func (bm *BackendManager) Keys() []string {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	out := make([]string, len(bm.keyOrder))
	copy(out, bm.keyOrder)
	return out
}

func (bm *BackendManager) keys() []string {
	return bm.keyOrder
}

// FirstKey returns the first configured backend key.
func (bm *BackendManager) FirstKey() string {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	if bm.firstKey == "" {
		panic("No backends configured")
	}
	return bm.firstKey
}

// NBackends returns the number of configured backends.
func (bm *BackendManager) NBackends() int {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	return len(bm.backends)
}

// Close shuts down all clients.
func (bm *BackendManager) Close() {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	for _, info := range bm.backends {
		info.Client.Close()
		if info.AgentClient != nil {
			info.AgentClient.Close()
		}
	}
}

// --- Synthetic LCP models ---

// generateLCPModels creates synthetic model aliases from chunk-level prefixes:
// strip provider prefix, split on -/_, take 1..N-1 chunk prefixes, keep those
// that substring-match >= 2 real models.
func lcpTokenize(s string) []string {
	pieces := []string{}
	start := 0
	for _, loc := range lcpSplitRe.FindAllStringIndex(s, -1) {
		if loc[0] > start {
			pieces = append(pieces, s[start:loc[0]])
		}
		pieces = append(pieces, s[loc[0]:loc[1]])
		start = loc[1]
	}
	if start < len(s) {
		pieces = append(pieces, s[start:])
	}
	return pieces
}

func (bm *BackendManager) generateLCPModels(modelNames []string) []string {
	if len(modelNames) < 2 {
		return []string{}
	}
	prefixModels := map[string]map[string]bool{}
	for _, name := range modelNames {
		modelPart := name
		if i := strings.Index(name, "/"); i >= 0 {
			modelPart = name[i+1:]
		}
		pieces := lcpTokenize(modelPart)
		for n := 1; n < len(pieces); n++ {
			prefix := strings.TrimRight(strings.Join(pieces[:n], ""), "-_")
			if prefixModels[prefix] == nil {
				prefixModels[prefix] = map[string]bool{}
			}
			prefixModels[prefix][name] = true
		}
	}
	result := []string{}
	for prefix, models := range prefixModels {
		if len(models) >= 2 {
			result = append(result, prefix)
		}
	}
	sort.Strings(result)
	return result
}

// --- Model discovery ---

// DiscoverModels discovers models across all live backends and merges them
// into the registry (including synthetic LCP models).
func (bm *BackendManager) DiscoverModels() error {
	return bm.discoverModelsCtx(context.Background())
}

func (bm *BackendManager) discoverModelsCtx(ctx context.Context) error {
	type beCtx struct {
		be  string
		ctx int
	}
	allDiscovered := map[string][]beCtx{}
	firstSeen := []string{}
	bm.mu.Lock()
	bm.lastDiscoverTiming = []map[string]any{}
	bm.mu.Unlock()

	appendTiming := func(entry map[string]any) {
		bm.mu.Lock()
		bm.lastDiscoverTiming = append(bm.lastDiscoverTiming, entry)
		bm.mu.Unlock()
	}

	for _, backendKey := range bm.Keys() {
		bm.mu.RLock()
		up := bm.backendState[backendKey]
		bm.mu.RUnlock()
		if !up {
			appendTiming(map[string]any{"backend": backendKey, "skipped": true})
			continue
		}
		t0 := time.Now()
		models, err := bm.GetClient(backendKey).DiscoverModels(ctx)
		elapsedMS := float64(time.Since(t0).Nanoseconds()) / 1e6
		// Python logs the raw list of (name, n_ctx) tuples.
		tuples := make([]string, 0, len(models))
		for _, m := range models {
			tuples = append(tuples, fmt.Sprintf("('%s', %d)", m.Name, m.NCtx))
		}
		appendTiming(map[string]any{"backend": backendKey, "ms": round1(elapsedMS), "models": len(models)})
		logInfo("backend_manager", "discover_models on backend '%s': [%s] (%.0fms)",
			backendKey, strings.Join(tuples, ", "), elapsedMS)
		if err != nil {
			// Router JSON decode errors propagate (Python JSONDecodeError is
			// not an HTTPError and is not caught in the client).
			return err
		}
		if len(models) == 0 {
			logWarn("backend_manager", "No models discovered on backend '%s'", backendKey)
			continue
		}
		for _, m := range models {
			if _, exists := allDiscovered[m.Name]; !exists {
				firstSeen = append(firstSeen, m.Name)
			}
			allDiscovered[m.Name] = append(allDiscovered[m.Name], beCtx{backendKey, m.NCtx})
		}
	}

	merged := map[string]*DiscoveredModel{}
	for _, name := range firstSeen {
		entries := allDiscovered[name]
		backends := make([]string, len(entries))
		backendNCTX := map[string]int{}
		minCtx := 0
		for i, e := range entries {
			backends[i] = e.be
			backendNCTX[e.be] = e.ctx
			if i == 0 || e.ctx < minCtx {
				minCtx = e.ctx
			}
		}
		merged[name] = &DiscoveredModel{
			Name: name, NCtx: minCtx, Backends: backends, BackendNCTX: backendNCTX,
			TotalSlots: 0, LastDiscovered: nowFloat(),
		}
	}

	// Add synthetic LCP models.
	lcpNames := bm.generateLCPModels(firstSeen)
	syntheticAdded := []string{}
	for _, lcpName := range lcpNames {
		matching := []string{}
		for _, name := range firstSeen {
			if strings.Contains(strings.ToLower(name), strings.ToLower(lcpName)) {
				matching = append(matching, name)
			}
		}
		if len(matching) >= 2 {
			beSet := map[string]bool{}
			for _, m := range matching {
				for _, be := range merged[m].Backends {
					beSet[be] = true
				}
			}
			lcpBackends := make([]string, 0, len(beSet))
			for be := range beSet {
				lcpBackends = append(lcpBackends, be)
			}
			sort.Strings(lcpBackends)
			lcpBackendNCTX := map[string]int{}
			for _, m := range matching {
				for be, c := range merged[m].BackendNCTX {
					lcpBackendNCTX[be] = c
				}
			}
			lcpNCtx := 0
			for i, m := range matching {
				if i == 0 || merged[m].NCtx < lcpNCtx {
					lcpNCtx = merged[m].NCtx
				}
			}
			merged[lcpName] = &DiscoveredModel{
				Name: lcpName, NCtx: lcpNCtx, Backends: lcpBackends, BackendNCTX: lcpBackendNCTX,
				TotalSlots: 0, LastDiscovered: nowFloat(), Synthetic: true,
			}
			syntheticAdded = append(syntheticAdded, lcpName)
			logInfo("backend_manager", "Synthetic LCP model '%s' matches %d models on backends %s",
				lcpName, len(matching), lcpBackends)
		}
	}

	bm.mu.Lock()
	bm.discoveredModels = merged
	bm.modelOrder = append(append([]string{}, firstSeen...), syntheticAdded...)
	bm.mu.Unlock()

	for _, name := range bm.modelOrder {
		info := merged[name]
		logInfo("backend_manager", "Discovered model '%s' on backends %s with n_ctx=%d",
			name, info.Backends, info.NCtx)
	}
	return nil
}

// GetModelNCtx returns the (min across backends) n_ctx for a model.
func (bm *BackendManager) GetModelNCtx(canonicalName string) int {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	if info, ok := bm.discoveredModels[canonicalName]; ok {
		return info.NCtx
	}
	return DefaultNCtx
}

// GetBackendNCtx returns the per-backend n_ctx for a model.
func (bm *BackendManager) GetBackendNCtx(canonicalName, backendKey string) int {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	if info, ok := bm.discoveredModels[canonicalName]; ok {
		if c, ok2 := info.BackendNCTX[backendKey]; ok2 {
			return c
		}
	}
	return DefaultNCtx
}

// GetDiscoveredModels resolves a client model name to matching DiscoveredModel
// objects: exact match, then substring (case-insensitive), then "any".
func (bm *BackendManager) GetDiscoveredModels(modelName string) []*DiscoveredModel {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	if modelName == "any" {
		if len(bm.discoveredModels) == 0 {
			return []*DiscoveredModel{}
		}
		out := make([]*DiscoveredModel, 0, len(bm.modelOrder))
		for _, name := range bm.modelOrder {
			if info, ok := bm.discoveredModels[name]; ok {
				out = append(out, info)
			}
		}
		return out
	}
	if info, ok := bm.discoveredModels[modelName]; ok {
		return []*DiscoveredModel{info}
	}
	out := []*DiscoveredModel{}
	for _, name := range bm.modelOrder {
		info, ok := bm.discoveredModels[name]
		if !ok {
			continue
		}
		if strings.Contains(strings.ToLower(info.Name), strings.ToLower(modelName)) {
			out = append(out, info)
		}
	}
	return out
}

// --- Slot counts ---

// RefreshSlotCounts queries slot counts for every discovered model+backend
// pair. Returns {backend_key: {model_name: n_slots}}.
func (bm *BackendManager) RefreshSlotCounts() (map[string]map[string]int, error) {
	return bm.refreshSlotCountsCtx(context.Background())
}

func (bm *BackendManager) refreshSlotCountsCtx(ctx context.Context) (map[string]map[string]int, error) {
	backendKeys := bm.Keys()
	bm.mu.RLock()
	nModels := len(bm.discoveredModels)
	modelNames := make([]string, 0, len(bm.discoveredModels))
	for _, name := range bm.modelOrder {
		modelNames = append(modelNames, name)
	}
	bm.mu.RUnlock()

	logInfo("backend_manager", "Refreshing slot counts: %d known models, %d backends", nModels, len(backendKeys))
	if len(backendKeys) == 0 {
		logError("backend_manager", "No backends configured — cannot refresh slot counts")
		return nil, errors.New("No backends configured")
	}

	slotCounts := map[string]map[string]int{}
	refreshedAny := false

	for _, canonicalName := range modelNames {
		bm.mu.RLock()
		info := bm.discoveredModels[canonicalName]
		bm.mu.RUnlock()
		if info == nil {
			continue
		}
		logInfo("backend_manager", "Model '%s' has backends: %s", canonicalName, info.Backends)
		for _, backendKey := range info.Backends {
			bm.mu.RLock()
			_, known := bm.backends[backendKey]
			bm.mu.RUnlock()
			if !known {
				continue
			}
			client := bm.GetClient(backendKey)

			// get_slots_info: connection errors and non-400 HTTP errors are
			// surfaced; they are caught here (Python: except Exception).
			slots, err := client.GetSlotsInfo(ctx, canonicalName)
			if err != nil {
				logWarn("backend_manager", "Failed to get slot info for model '%s' on backend '%s': %s",
					canonicalName, backendKey, err)
				slots = nil
			}

			if slots != nil && len(slots) > 0 {
				var nSlots int
				if _, isRouter := slots[0]["_router_model"]; isRouter {
					for _, s := range slots {
						if s["_router_model"] == canonicalName {
							nSlots++
						}
					}
					logInfo("backend_manager", "Model '%s' on backend '%s' has %d slots (router mode)",
						canonicalName, backendKey, nSlots)
				} else {
					nSlots = len(slots)
					logInfo("backend_manager", "Model '%s' on backend '%s' has %d slots",
						canonicalName, backendKey, nSlots)
				}
				if slotCounts[backendKey] == nil {
					slotCounts[backendKey] = map[string]int{}
				}
				slotCounts[backendKey][canonicalName] = nSlots
				bm.mu.Lock()
				bm.refreshState[backendModelKey{canonicalName, backendKey}] = refreshEntry{nowFloat(), true, nSlots}
				bm.mu.Unlock()
				refreshedAny = true
			} else {
				logWarn("backend_manager", "Model '%s' not loaded on backend '%s'", canonicalName, backendKey)
				bm.mu.Lock()
				bm.refreshState[backendModelKey{canonicalName, backendKey}] = refreshEntry{nowFloat(), false, 0}
				bm.mu.Unlock()
				refreshedAny = true
			}
		}
	}

	if !refreshedAny {
		logWarn("backend_manager", "No backends refreshed: all failed or missing client")
	}

	// Update total_slots on each DiscoveredModel.
	bm.mu.Lock()
	for _, name := range bm.modelOrder {
		info := bm.discoveredModels[name]
		if info == nil {
			continue
		}
		total := 0
		for _, be := range info.Backends {
			total += slotCounts[be][name]
		}
		info.TotalSlots = total
	}
	bm.mu.Unlock()

	return slotCounts, nil
}

// --- Liveness checker ---

// StartLivenessChecker launches the 5s health-check loop.
func (bm *BackendManager) StartLivenessChecker() {
	ctx, cancel := context.WithCancel(context.Background())
	bm.mu.Lock()
	bm.livenessCancel = cancel
	bm.mu.Unlock()
	go bm.livenessLoop(ctx)
}

// StopLivenessChecker stops the health-check loop.
func (bm *BackendManager) StopLivenessChecker() {
	bm.mu.Lock()
	cancel := bm.livenessCancel
	bm.livenessCancel = nil
	bm.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

// livenessLoop pings backends every 5s and triggers discovery + slot refresh
// on state change or missing models.
func (bm *BackendManager) livenessLoop(ctx context.Context) {
	// Initialize all backends as up so discover_models() doesn't skip them
	// before the first health check runs.
	bm.mu.Lock()
	for _, k := range bm.keyOrder {
		if _, ok := bm.backendState[k]; !ok {
			bm.backendState[k] = true
		}
	}
	bm.mu.Unlock()

	for {
		loopT0 := nowFloat()
		select {
		case <-ctx.Done():
			return
		case <-time.After(5 * time.Second):
		}
		changed := false
		stateChanges := []map[string]any{}
		healthResults := []map[string]any{}

		for _, backendKey := range bm.Keys() {
			client := bm.GetClient(backendKey)
			bm.mu.RLock()
			oldState := bm.backendState[backendKey]
			bm.mu.RUnlock()
			isUp := false
			hcT0 := time.Now()
			hcError := ""
			recreated := false
			retrySucceeded := false
			if err := client.HealthCheck(context.Background()); err != nil {
				// Health check failed — recreate client (pool may be poisoned)
				// and retry once before flipping state.
				hcError = errName(err)
				recreated = true
				client.Recreate()
				if err2 := client.HealthCheck(context.Background()); err2 == nil {
					isUp = true
					hcError = ""
					retrySucceeded = true
				} else {
					hcError = errName(err2)
				}
			} else {
				isUp = true
			}
			hcMS := float64(time.Since(hcT0).Nanoseconds()) / 1e6

			stateChanged := false
			if isUp != oldState {
				stateChanged = true
				bm.mu.Lock()
				bm.backendState[backendKey] = isUp
				bm.mu.Unlock()
				changed = true
				stateChanges = append(stateChanges, map[string]any{
					"backend": backendKey, "old_state": oldState, "new_state": isUp,
				})
				// Recreate client on state change — down transition likely
				// poisoned the pool, up transition needs a fresh pool.
				client.Recreate()
				recreated = true
			}

			healthResults = append(healthResults, map[string]any{
				"backend": backendKey, "is_up": isUp, "old_state": oldState,
				"state_changed": stateChanged, "ms": round1(hcMS), "error": hcError,
				"recreated": recreated, "retry_succeeded": retrySucceeded,
			})
		}

		// Also trigger if an up backend has no models in the registry.
		bm.mu.RLock()
		upKeys := map[string]bool{}
		for k, v := range bm.backendState {
			if v {
				upKeys[k] = true
			}
		}
		discoveredBackends := map[string]bool{}
		for _, info := range bm.discoveredModels {
			for _, be := range info.Backends {
				discoveredBackends[be] = true
			}
		}
		missingModels := []string{}
		for k := range upKeys {
			if !discoveredBackends[k] {
				missingModels = append(missingModels, k)
			}
		}
		bm.mu.RUnlock()
		if len(missingModels) > 0 {
			changed = true
			for _, be := range missingModels {
				stateChanges = append(stateChanges, map[string]any{
					"backend": be, "old_state": "up_no_models", "new_state": "up",
				})
			}
		}

		var discMS, slotsMS *float64
		discError := ""
		slotsError := ""
		if changed {
			discT0 := time.Now()
			dctx, dcancel := context.WithTimeout(context.Background(), 10*time.Second)
			var discErr, slotsErr error
			var wg sync.WaitGroup
			wg.Add(2)
			go func() {
				defer wg.Done()
				discErr = bm.discoverModelsCtx(dctx)
			}()
			go func() {
				defer wg.Done()
				_, slotsErr = bm.refreshSlotCountsCtx(dctx)
			}()
			done := make(chan struct{})
			go func() { wg.Wait(); close(done) }()
			timedOut := false
			select {
			case <-done:
			case <-dctx.Done():
				timedOut = true
			}
			dcancel()
			elapsedMS := float64(time.Since(discT0).Nanoseconds()) / 1e6
			if timedOut {
				v := 10000.0
				discMS, slotsMS = &v, &v
				discError = "timeout"
				slotsError = "timeout"
				logWarn("backend_manager", "discovery/refresh timed out during liveness check — recreating clients")
				for _, k := range bm.Keys() {
					bm.GetClient(k).Recreate()
				}
			} else {
				v := elapsedMS
				discMS, slotsMS = &v, &v
				if discErr != nil {
					discError = errName(discErr)
					logError("backend_manager", "discover_models failed: %s", discErr)
				}
				if slotsErr != nil {
					slotsError = errName(slotsErr)
					logError("backend_manager", "refresh_slot_counts failed: %s", slotsErr)
				}
			}

			bm.mu.RLock()
			discoveredModels := map[string][]string{}
			for name, info := range bm.discoveredModels {
				discoveredModels[name] = info.Backends
			}
			bm.mu.RUnlock()
			Metrics.Record(map[string]any{
				"event":             "liveness_change",
				"state_changes":     stateChanges,
				"discovered_models": discoveredModels,
			})
		}

		// Record a diagnostic event only when something noteworthy happened.
		hasErrors := false
		hasRetries := false
		for _, h := range healthResults {
			if h["error"] != "" {
				hasErrors = true
			}
			if h["retry_succeeded"] == true {
				hasRetries = true
			}
		}
		worthRecording := changed || hasErrors || hasRetries || discError != "" || slotsError != ""
		if worthRecording {
			bm.mu.RLock()
			states := map[string]bool{}
			for k, v := range bm.backendState {
				states[k] = v
			}
			nDiscovered := len(bm.discoveredModels)
			timing := make([]map[string]any, len(bm.lastDiscoverTiming))
			copy(timing, bm.lastDiscoverTiming)
			bm.mu.RUnlock()
			record := map[string]any{
				"event":             "liveness_diag",
				"health":            healthResults,
				"states":            states,
				"changed":           changed,
				"discovered_models": nDiscovered,
				"discover_timing":   timing,
				"discover_ms":       nil,
				"discover_error":    discError,
				"slots_ms":          nil,
				"slots_error":       slotsError,
				"total_ms":          round1((nowFloat() - loopT0) * 1000),
			}
			if discMS != nil {
				record["discover_ms"] = round1(*discMS)
			}
			if slotsMS != nil {
				record["slots_ms"] = round1(*slotsMS)
			}
			Metrics.Record(record)
		}
	}
}

// errName produces a Python-flavored exception type name for log/metrics
// diagnostics (mirrors httpx type(e).__name__: "ConnectError", "ReadError",
// "TimeoutError", "RemoteProtocolError", ...).
func errName(err error) string {
	if err == nil {
		return ""
	}
	var statusErr *HTTPStatusError
	if errors.As(err, &statusErr) {
		return "HTTPStatusError"
	}
	var urlErr *url.Error
	if errors.As(err, &urlErr) {
		inner := urlErr.Err
		if errors.Is(inner, context.DeadlineExceeded) || errors.Is(inner, os.ErrDeadlineExceeded) {
			return "TimeoutError"
		}
		var netErr *net.OpError
		if errors.As(inner, &netErr) {
			if netErr.Op == "dial" {
				return "ConnectError"
			}
			return "ReadError"
		}
		return "ConnectionError"
	}
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, ErrTimeout) {
		return "TimeoutError"
	}
	var netErr *net.OpError
	if errors.As(err, &netErr) {
		return "ReadError"
	}
	return "Error"
}

// backendManager is the global singleton, created in init() from BACKENDS.
var backendManager *BackendManager
