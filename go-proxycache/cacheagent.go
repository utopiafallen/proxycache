// cacheagent.go — HTTP client for the remote cache-agent (deletion + file
// size lookups for backends whose cache lives on a remote host).

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

// CacheAgentClient talks to one cache-agent instance.
// Endpoints: POST /cache/delete?key=, GET /cache/files/<key>,
// POST /cache/files/batch. Timeout is SLOT_TIMEOUT.
type CacheAgentClient struct {
	BaseURL string
	client  *http.Client
}

func NewCacheAgentClient(baseURL string) *CacheAgentClient {
	return &CacheAgentClient{
		BaseURL: strings.TrimRight(baseURL, "/"),
		client:  &http.Client{Timeout: seconds(SlotTimeout)},
	}
}

// Delete removes a cache file from the agent's CACHE_DIR.
// Returns true on success, false on any error (agent down, non-200, etc.).
func (a *CacheAgentClient) Delete(key string) bool {
	reqURL := a.BaseURL + "/cache/delete?" + url.Values{"key": {key}}.Encode()
	resp, err := a.client.Post(reqURL, "", nil)
	if err != nil {
		logWarn("cache_agent_client", "Cache agent error on %s for key %s: %s", a.BaseURL, truncateKey(key), err)
		return false
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		logWarn("cache_agent_client", "Cache agent returned status %d for key %s: %s",
			resp.StatusCode, truncateKey(key), truncateString(string(body), 200))
		return false
	}
	return true
}

// GetFileSize returns {"size": int, "exists": bool} for a cache file,
// {"exists": false} on 404, and nil on connection error.
func (a *CacheAgentClient) GetFileSize(key string) map[string]any {
	resp, err := a.client.Get(a.BaseURL + "/cache/files/" + key)
	if err != nil {
		logWarn("cache_agent_client", "Cache agent size check error on %s for key %s: %s", a.BaseURL, truncateKey(key), err)
		return nil
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode == 404 {
		return map[string]any{"exists": false}
	}
	if resp.StatusCode != 200 {
		logWarn("cache_agent_client", "Cache agent size check returned status %d for key %s: %s",
			resp.StatusCode, truncateKey(key), truncateString(string(body), 200))
		return nil
	}
	var data map[string]any
	if err := json.Unmarshal(body, &data); err != nil {
		return nil
	}
	return data
}

// BatchGetFileSizes returns key -> {"size": int, "exists": bool} for many
// keys in one call. Empty map on error.
func (a *CacheAgentClient) BatchGetFileSizes(keys []string) map[string]map[string]any {
	if len(keys) == 0 {
		return map[string]map[string]any{}
	}
	payload, _ := json.Marshal(map[string]any{"keys": keys})
	resp, err := a.client.Post(a.BaseURL+"/cache/files/batch", "application/json", bytes.NewReader(payload))
	if err != nil {
		logWarn("cache_agent_client", "Cache agent batch size check error on %s (%d keys): %s", a.BaseURL, len(keys), err)
		return map[string]map[string]any{}
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		logWarn("cache_agent_client", "Cache agent batch returned status %d for %d keys: %s",
			resp.StatusCode, len(keys), truncateString(string(body), 200))
		return map[string]map[string]any{}
	}
	var data struct {
		Results map[string]map[string]any `json:"results"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return map[string]map[string]any{}
	}
	if data.Results == nil {
		return map[string]map[string]any{}
	}
	return data.Results
}

// Close releases idle connections.
func (a *CacheAgentClient) Close() {
	a.client.CloseIdleConnections()
}

// AgentServer is the embedded cache-agent: byte-for-byte the same behavior as
// the standalone Go cache-agent binary (cache-agent/main.go), embedded so the
// proxycache binary can serve agents for its own backends.
//
//	POST /cache/delete?key=<basename>  -> {"ok": bool, "error": string}
//	GET  /cache/files/<key>            -> {"size": int, "exists": bool}
//	POST /cache/files/batch            -> {"results": {key: {...}}}
type AgentServer struct {
	CacheDir string
	Port     int
}

// NewAgentServer builds the server struct (cacheDir is required).
func NewAgentServer(cacheDir string, port int) *AgentServer {
	return &AgentServer{CacheDir: cacheDir, Port: port}
}

// Start runs the agent HTTP server until ctx is cancelled.
func (s *AgentServer) Start(ctx context.Context) {
	mux := http.NewServeMux()
	mux.HandleFunc("/cache/delete", s.handleDelete)
	mux.HandleFunc("/cache/files/batch", s.handleBatch)
	mux.HandleFunc("/cache/files/", s.handleFile)
	srv := &http.Server{Addr: fmt.Sprintf(":%d", s.Port), Handler: mux}
	go func() {
		<-ctx.Done()
		shutCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutCtx)
	}()
	logInfo("cache_agent", "cache-agent listening on :%d (CACHE_DIR=%s)", s.Port, s.CacheDir)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		logError("cache_agent", "Cache agent server error: %s", err)
	}
}

func (s *AgentServer) handleDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"ok": false, "error": "method not allowed"})
		return
	}
	key := r.URL.Query().Get("key")
	if key == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": "key parameter is required"})
		return
	}
	cachePath := s.CacheDir + "/" + key
	if err := os.Remove(cachePath); err != nil {
		if os.IsNotExist(err) {
			logInfo("cache_agent", "cache delete: file not found: %s", key)
			writeJSON(w, http.StatusNotFound, map[string]any{"ok": false, "error": "file not found"})
			return
		}
		logInfo("cache_agent", "cache delete: failed to remove %s: %v", key, err)
		writeJSON(w, http.StatusInternalServerError, map[string]any{"ok": false, "error": err.Error()})
		return
	}
	// Delete ckpt sidecar files (<key>.ckpt, <key>.ckpt.0, etc.)
	if entries, err := os.ReadDir(s.CacheDir); err == nil {
		prefix := key + ".ckpt"
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			name := entry.Name()
			if name == prefix || (len(name) > len(prefix)+1 && strings.HasPrefix(name, prefix+".")) {
				sidecar := s.CacheDir + "/" + name
				if err := os.Remove(sidecar); err == nil {
					logInfo("cache_agent", "cache delete: sidecar %s", name)
				} else {
					logInfo("cache_agent", "cache delete: failed to remove sidecar %s: %v", name, err)
				}
			}
		}
	}
	logInfo("cache_agent", "cache delete: %s", key)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

func (s *AgentServer) handleFile(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"exists": false})
		return
	}
	basename := strings.TrimPrefix(r.URL.Path, "/cache/files/")
	if basename == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{"exists": false})
		return
	}
	cachePath := s.CacheDir + "/" + basename
	info, err := os.Stat(cachePath)
	if err != nil {
		if os.IsNotExist(err) {
			writeJSON(w, http.StatusNotFound, map[string]any{"exists": false})
			return
		}
		logInfo("cache_agent", "cache file info: failed to stat %s: %v", basename, err)
		writeJSON(w, http.StatusInternalServerError, map[string]any{"exists": false})
		return
	}
	logInfo("cache_agent", "cache file size: %s size=%d", basename, info.Size())
	writeJSON(w, http.StatusOK, map[string]any{"size": info.Size(), "exists": true})
}

func (s *AgentServer) handleBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"results": map[string]map[string]any{}})
		return
	}
	var req struct {
		Keys []string `json:"keys"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"results": map[string]map[string]any{}})
		return
	}
	results := map[string]map[string]any{}
	for _, key := range req.Keys {
		info, err := os.Stat(s.CacheDir + "/" + key)
		if err != nil {
			results[key] = map[string]any{"exists": false}
		} else {
			results[key] = map[string]any{"size": info.Size(), "exists": true}
		}
	}
	logInfo("cache_agent", "batch file size: %d keys queried", len(req.Keys))
	writeJSON(w, http.StatusOK, map[string]any{"results": results})
}

// writeJSON writes a JSON response with the given status code.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
