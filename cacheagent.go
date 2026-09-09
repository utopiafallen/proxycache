// cacheagent.go — HTTP client for the cache-agent (deletion, file size
// lookups, P2P transfer orchestration calls) plus the embedded cache-agent
// server (AgentServer).
//
// AgentServer is an embedded copy of the standalone cache-agent binary
// (cache-agent/main.go — a separate module). It is started for local
// backends that set "agent_serve_port" so this proxycache can serve as the
// cache-agent for its own local llama instances (remote proxies then see
// those backends as ordinary agent_port backends and P2P transfers work).
// Keep the two implementations in sync.
//
// Checkpoint sidecar files (<key>.ckpt, <key>.ckpt.N) are part of a complete
// cache entry when present: they travel with their key on transfers and are
// removed with it on delete/eviction.

package proxycache

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// CacheAgentClient talks to one cache-agent instance.
// Endpoints: POST /cache/delete?key=, GET /cache/files/<key>,
// POST /cache/files/batch, POST /cache/transfer, POST /cache/receive,
// POST /cache/make-space, GET /cache/file. Lookup/delete ops time out at
// SLOT_TIMEOUT; transfer ops use the (larger) CACHE_TRANSFER_TIMEOUT.
type CacheAgentClient struct {
	BaseURL        string
	client         *http.Client
	transferClient *http.Client
}

func NewCacheAgentClient(baseURL string) *CacheAgentClient {
	return &CacheAgentClient{
		BaseURL:        strings.TrimRight(baseURL, "/"),
		client:         &http.Client{Timeout: seconds(SlotTimeout)},
		transferClient: &http.Client{Timeout: seconds(CacheTransferTimeout)},
	}
}

// Sidecars returns the checkpoint sidecar filenames that exist for key
// (e.g. "key.ckpt", "key.ckpt.0"), via the /cache/files/<key> response.
// Empty when the file is missing or the agent has no sidecars for it.
func (a *CacheAgentClient) Sidecars(key string) []string {
	info := a.GetFileSize(key)
	if info == nil {
		return nil
	}
	list, _ := info["sidecars"].([]any)
	out := make([]string, 0, len(list))
	for _, s := range list {
		if str, ok := s.(string); ok {
			out = append(out, str)
		}
	}
	return out
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

// Transfer asks this agent (the cache source) to push `key` to the target
// agent at targetURL, first asking the target to free room for the file
// within the maxBytes budget. Returns true on success.
func (a *CacheAgentClient) Transfer(key, targetURL string, maxBytes int64) bool {
	q := url.Values{}
	q.Set("key", key)
	q.Set("target", targetURL)
	q.Set("max_bytes", strconv.FormatInt(maxBytes, 10))
	resp, err := a.transferClient.Post(a.BaseURL+"/cache/transfer?"+q.Encode(), "", nil)
	if err != nil {
		logWarn("cache_agent_client", "Cache transfer error on %s for key %s: %s", a.BaseURL, truncateKey(key), err)
		return false
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		logWarn("cache_agent_client", "Cache transfer returned status %d for key %s: %s",
			resp.StatusCode, truncateKey(key), truncateString(string(body), 200))
		return false
	}
	var data map[string]any
	if err := json.Unmarshal(body, &data); err == nil {
		if ok, hasOK := data["ok"].(bool); hasOK && !ok {
			logWarn("cache_agent_client", "Cache transfer reported failure for key %s: %v", truncateKey(key), data["error"])
			return false
		}
	}
	return true
}

// UploadFile streams the file at path to the agent's receive endpoint,
// asking it to free room within the maxBytes budget first. The file is
// streamed (fixed-size buffers), never fully buffered in RAM. Content-Length
// is set explicitly so the target can size its make-space. Returns true on
// success.
func (a *CacheAgentClient) UploadFile(key, path string, maxBytes int64) bool {
	f, err := os.Open(path)
	if err != nil {
		logWarn("cache_agent_client", "Cache upload open failed for %s: %v", truncateKey(key), err)
		return false
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		logWarn("cache_agent_client", "Cache upload stat failed for %s: %v", truncateKey(key), err)
		return false
	}
	q := url.Values{}
	q.Set("key", key)
	if maxBytes > 0 {
		q.Set("max_bytes", strconv.FormatInt(maxBytes, 10))
	}
	req, err := http.NewRequest(http.MethodPost, a.BaseURL+"/cache/receive?"+q.Encode(), f)
	if err != nil {
		logWarn("cache_agent_client", "Cache upload error on %s for key %s: %s", a.BaseURL, truncateKey(key), err)
		return false
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	req.ContentLength = fi.Size()
	resp, err := a.transferClient.Do(req)
	if err != nil {
		logWarn("cache_agent_client", "Cache upload error on %s for key %s: %s", a.BaseURL, truncateKey(key), err)
		return false
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		logWarn("cache_agent_client", "Cache upload returned status %d for key %s: %s",
			resp.StatusCode, truncateKey(key), truncateString(string(body), 200))
		return false
	}
	return true
}

// MakeSpace asks the agent to evict old files until `need` bytes fit within
// the maxBytes total budget. Returns (ok, usedBytes).
func (a *CacheAgentClient) MakeSpace(need, maxBytes int64) (bool, int64) {
	q := url.Values{}
	q.Set("need", strconv.FormatInt(need, 10))
	q.Set("max", strconv.FormatInt(maxBytes, 10))
	resp, err := a.transferClient.Post(a.BaseURL+"/cache/make-space?"+q.Encode(), "", nil)
	if err != nil {
		logWarn("cache_agent_client", "Cache make-space error on %s: %s", a.BaseURL, err)
		return false, 0
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return false, 0
	}
	var data struct {
		Ok    bool  `json:"ok"`
		Used  int64 `json:"used"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return false, 0
	}
	return data.Ok, data.Used
}

// FetchStream opens a download of the cache file's raw content from the
// agent and returns the live response body (caller closes it) plus its size
// (from Content-Length; -1 when unknown). Callers stream it straight to disk.
func (a *CacheAgentClient) FetchStream(key string) (io.ReadCloser, int64, error) {
	resp, err := a.transferClient.Get(a.BaseURL + "/cache/file?" + url.Values{"key": {key}}.Encode())
	if err != nil {
		return nil, 0, err
	}
	if resp.StatusCode == http.StatusNotFound {
		resp.Body.Close()
		return nil, 0, os.ErrNotExist
	}
	if resp.StatusCode != 200 {
		resp.Body.Close()
		return nil, 0, fmt.Errorf("fetch file: status %d", resp.StatusCode)
	}
	return resp.Body, resp.ContentLength, nil
}

// FetchFile downloads a cache file's full content from the agent. Returns an
// error when the file is missing or the request fails.
func (a *CacheAgentClient) FetchFile(key string) ([]byte, error) {
	body, _, err := a.FetchStream(key)
	if err != nil {
		return nil, err
	}
	defer body.Close()
	return io.ReadAll(body)
}

// Close releases idle connections.
func (a *CacheAgentClient) Close() {
	a.client.CloseIdleConnections()
	if a.transferClient != nil {
		a.transferClient.CloseIdleConnections()
	}
}

// AgentServer is the embedded cache-agent: same behavior as the standalone Go
// cache-agent binary (cache-agent/main.go), started for local backends that
// set agent_serve_port so this proxycache serves as the cache-agent for its
// own local llama instances. Keep the two implementations in sync.
//
//	POST /cache/delete?key=<basename>  -> {"ok": bool, "error": string}
//	GET  /cache/files/<key>            -> {"size": int, "exists": bool, "sidecars": [string]}
//	POST /cache/files/batch            -> {"results": {key: {...}}}
//	POST /cache/transfer               -> source-side P2P push (main + sidecars) to a target agent
//	POST /cache/receive?key=<key>      -> raw file upload (target side; make-space when max_bytes given)
//	POST /cache/make-space             -> evict oldest files to fit a budget
//	GET  /cache/file?key=<key>         -> raw file download (source pull)
type AgentServer struct {
	CacheDir string
	Port     int

	mu       sync.Mutex
	listener net.Listener
}

// NewAgentServer builds the server struct (cacheDir is required). Port 0 picks
// an ephemeral port (see Addr).
func NewAgentServer(cacheDir string, port int) *AgentServer {
	return &AgentServer{CacheDir: cacheDir, Port: port}
}

// Addr returns the bound listen address once Start has bound the socket
// ("" before that).
func (s *AgentServer) Addr() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.listener == nil {
		return ""
	}
	return s.listener.Addr().String()
}

// Start runs the agent HTTP server until ctx is cancelled.
func (s *AgentServer) Start(ctx context.Context) {
	mux := http.NewServeMux()
	mux.HandleFunc("/cache/delete", s.handleDelete)
	mux.HandleFunc("/cache/files/batch", s.handleBatch)
	mux.HandleFunc("/cache/files/", s.handleFile)
	mux.HandleFunc("/cache/transfer", s.handleTransfer)
	mux.HandleFunc("/cache/receive", s.handleReceive)
	mux.HandleFunc("/cache/make-space", s.handleMakeSpace)
	mux.HandleFunc("/cache/file", s.handleFileContent)
	ln, err := net.Listen("tcp", fmt.Sprintf(":%d", s.Port))
	if err != nil {
		logError("cache_agent", "Cache agent listen error: %s", err)
		return
	}
	s.mu.Lock()
	s.listener = ln
	s.mu.Unlock()
	srv := &http.Server{Handler: mux}
	go func() {
		<-ctx.Done()
		shutCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutCtx)
	}()
	logInfo("cache_agent", "cache-agent listening on %s (CACHE_DIR=%s)", ln.Addr(), s.CacheDir)
	if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
		logError("cache_agent", "Cache agent server error: %s", err)
	}
}

func (s *AgentServer) handleDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		WriteJSON(w, http.StatusMethodNotAllowed, map[string]any{"ok": false, "error": "method not allowed"})
		return
	}
	key := r.URL.Query().Get("key")
	if key == "" {
		WriteJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": "key parameter is required"})
		return
	}
	cachePath := s.CacheDir + "/" + key
	if err := os.Remove(cachePath); err != nil {
		if os.IsNotExist(err) {
			logInfo("cache_agent", "cache delete: file not found: %s", key)
			WriteJSON(w, http.StatusNotFound, map[string]any{"ok": false, "error": "file not found"})
			return
		}
		logInfo("cache_agent", "cache delete: failed to remove %s: %v", key, err)
		WriteJSON(w, http.StatusInternalServerError, map[string]any{"ok": false, "error": err.Error()})
		return
	}
	// Delete ckpt sidecar files (<key>.ckpt, <key>.ckpt.0, etc.)
	if entries, err := os.ReadDir(s.CacheDir); err == nil {
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			name := entry.Name()
			if isSidecarName(name, key) {
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
	WriteJSON(w, http.StatusOK, map[string]any{"ok": true})
}

func (s *AgentServer) handleFile(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		WriteJSON(w, http.StatusMethodNotAllowed, map[string]any{"exists": false})
		return
	}
	basename := strings.TrimPrefix(r.URL.Path, "/cache/files/")
	if basename == "" {
		WriteJSON(w, http.StatusBadRequest, map[string]any{"exists": false})
		return
	}
	cachePath := s.CacheDir + "/" + basename
	info, err := os.Stat(cachePath)
	if err != nil {
		if os.IsNotExist(err) {
			WriteJSON(w, http.StatusNotFound, map[string]any{"exists": false})
			return
		}
		logInfo("cache_agent", "cache file info: failed to stat %s: %v", basename, err)
		WriteJSON(w, http.StatusInternalServerError, map[string]any{"exists": false})
		return
	}
	logInfo("cache_agent", "cache file size: %s size=%d", basename, info.Size())
	WriteJSON(w, http.StatusOK, map[string]any{"size": info.Size(), "exists": true, "sidecars": listLocalSidecars(s.CacheDir, basename)})
}

func (s *AgentServer) handleBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		WriteJSON(w, http.StatusMethodNotAllowed, map[string]any{"results": map[string]map[string]any{}})
		return
	}
	var req struct {
		Keys []string `json:"keys"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteJSON(w, http.StatusBadRequest, map[string]any{"results": map[string]map[string]any{}})
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
	WriteJSON(w, http.StatusOK, map[string]any{"results": results})
}

// --- P2P transfer endpoints ---

// transferHTTPClient bounds agent-to-agent transfer calls.
var transferHTTPClient = &http.Client{Timeout: seconds(CacheTransferTimeout)}

// p2pCopyBufSize is the fixed copy buffer for streaming transfers (no
// full-file RAM buffering, so multi-GB files cost a few buffers, not one
// full copy). Larger than needed for throughput — disk/network paths are
// kernel-buffer-bound, not syscall-bound — but cheap: one 16 MiB buffer per
// concurrent transfer. (net/http's own per-connection buffers are fixed at
// 4 KiB internally; the kernel TCP buffers do the real work.)
const p2pCopyBufSize = 16 << 20 // 16 MiB

// fmtRate formats bytes transferred over a duration as a human rate.
func fmtRate(n int64, d time.Duration) string {
	if d <= 0 {
		return "n/a"
	}
	v := float64(n) / d.Seconds() / (1 << 20)
	if v >= 1024 {
		return fmt.Sprintf("%.1f GiB/s", v/1024)
	}
	return fmt.Sprintf("%.1f MiB/s", v)
}

var p2pSidecarRe = regexp.MustCompile(`\.ckpt(\.\d+)?$`)

// isSidecarName reports whether name is a checkpoint sidecar of key
// (key.ckpt or key.ckpt.N).
func isSidecarName(name, key string) bool {
	prefix := key + ".ckpt"
	return name == prefix || (len(name) > len(prefix)+1 && strings.HasPrefix(name, prefix+".") && p2pSidecarRe.MatchString(name))
}

// listLocalSidecars returns the names of existing checkpoint sidecar files
// for key in dir.
func listLocalSidecars(dir, key string) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if isSidecarName(e.Name(), key) {
			out = append(out, e.Name())
		}
	}
	return out
}

// handleTransfer (source side): pushes key's main file plus any existing
// checkpoint sidecars to the target agent, asking the target to free room for
// the full entry first when max_bytes > 0.
func (s *AgentServer) handleTransfer(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		WriteJSON(w, http.StatusMethodNotAllowed, map[string]any{"ok": false, "error": "method not allowed"})
		return
	}
	q := r.URL.Query()
	key := q.Get("key")
	target := q.Get("target")
	if key == "" || target == "" {
		WriteJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": "key and target parameters are required"})
		return
	}
	var maxBytes int64
	if v, err := strconv.ParseInt(q.Get("max_bytes"), 10, 64); err == nil {
		maxBytes = v
	}
	cachePath := s.CacheDir + "/" + key
	fi, err := os.Stat(cachePath)
	if err != nil || fi.IsDir() {
		WriteJSON(w, http.StatusNotFound, map[string]any{"ok": false, "error": "file not found"})
		return
	}
	size := fi.Size()
	sidecars := listLocalSidecars(s.CacheDir, key)
	total := size
	for _, sc := range sidecars {
		if info, err := os.Stat(s.CacheDir + "/" + sc); err == nil {
			total += info.Size()
		}
	}
	fail := func(msg string) {
		logWarn("cache_agent", "cache transfer: %s", msg)
		WriteJSON(w, http.StatusOK, map[string]any{"ok": false, "error": msg})
	}
	if maxBytes > 0 {
		ok, _ := agentMakeSpaceHTTP(target, total, maxBytes)
		if !ok {
			fail(fmt.Sprintf("target %s cannot make space for %d bytes", target, total))
			return
		}
	}
	push := func(name string) (bool, string, int64, time.Duration) {
		f, err := os.Open(s.CacheDir + "/" + name)
		if err != nil {
			return false, err.Error(), 0, 0
		}
		defer f.Close()
		fi, _ := f.Stat()
		start := time.Now()
		// Explicit Content-Length: a bare *os.File body would go chunked and
		// the target's receive handler could not see the size for make-space.
		req, err := http.NewRequest(http.MethodPost, strings.TrimRight(target, "/")+"/cache/receive?"+url.Values{"key": {name}}.Encode(), f)
		if err != nil {
			return false, err.Error(), 0, time.Since(start)
		}
		req.Header.Set("Content-Type", "application/octet-stream")
		req.ContentLength = fi.Size()
		resp, err := transferHTTPClient.Do(req)
		if err != nil {
			return false, err.Error(), 0, time.Since(start)
		}
		defer resp.Body.Close()
		rb, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			return false, fmt.Sprintf("target status %d: %s", resp.StatusCode, strings.TrimSpace(string(rb))), 0, time.Since(start)
		}
		return true, "", fi.Size(), time.Since(start)
	}
	ok, errMsg, n, d := push(key)
	logInfo("cache_agent", "cache transfer: pushed %s (%d bytes, %.0fms, %s) to %s", key, n, float64(d.Milliseconds()), fmtRate(n, d), target)
	if !ok {
		fail(fmt.Sprintf("push of %s to %s failed: %s", key, target, errMsg))
		return
	}
	for _, sc := range sidecars {
		ok, errMsg, n, d := push(sc)
		logInfo("cache_agent", "cache transfer: pushed %s (%d bytes, %.0fms, %s) to %s", sc, n, float64(d.Milliseconds()), fmtRate(n, d), target)
		if !ok {
			fail(fmt.Sprintf("push of sidecar %s to %s failed: %s", sc, target, errMsg))
			return
		}
	}
	logInfo("cache_agent", "cache transfer: transferred %s (%d bytes total incl. %d sidecar(s)) to %s", key, total, len(sidecars), target)
	WriteJSON(w, http.StatusOK, map[string]any{"ok": true, "bytes": size, "sidecars": len(sidecars)})
}

// agentMakeSpaceHTTP asks a target agent to free room for need bytes within the budget.
func agentMakeSpaceHTTP(target string, need, maxBytes int64) (bool, int64) {
	q := url.Values{}
	q.Set("need", strconv.FormatInt(need, 10))
	q.Set("max", strconv.FormatInt(maxBytes, 10))
	resp, err := transferHTTPClient.Post(strings.TrimRight(target, "/")+"/cache/make-space?"+q.Encode(), "", nil)
	if err != nil {
		logWarn("cache_agent", "cache transfer: make-space on %s failed: %v", target, err)
		return false, 0
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return false, 0
	}
	var data struct {
		Ok   bool  `json:"ok"`
		Used int64 `json:"used"`
	}
	if json.Unmarshal(body, &data) != nil {
		return false, 0
	}
	return data.Ok, data.Used
}

// handleReceive (target side): writes the raw request body to the cache dir
// atomically (tmp + rename). When max_bytes is given and the body length is
// known, frees room for it within the budget first.
func (s *AgentServer) handleReceive(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		WriteJSON(w, http.StatusMethodNotAllowed, map[string]any{"ok": false, "error": "method not allowed"})
		return
	}
	key := r.URL.Query().Get("key")
	if key == "" {
		WriteJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": "key parameter is required"})
		return
	}
	var maxBytes int64
	if v, err := strconv.ParseInt(r.URL.Query().Get("max_bytes"), 10, 64); err == nil {
		maxBytes = v
	}
	if n := r.ContentLength; n > 0 && maxBytes > 0 {
		ok, _ := s.evictUntil(n, maxBytes)
		if !ok {
			WriteJSON(w, http.StatusInsufficientStorage, map[string]any{"ok": false, "error": "insufficient space and no room could be made"})
			return
		}
	}
	tmp := s.CacheDir + "/" + key + ".p2p.tmp"
	out, err := os.Create(tmp)
	if err != nil {
		WriteJSON(w, http.StatusInternalServerError, map[string]any{"ok": false, "error": err.Error()})
		return
	}
	start := time.Now()
	buf := make([]byte, p2pCopyBufSize)
	n, err := io.CopyBuffer(out, r.Body, buf)
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	d := time.Since(start)
	if err != nil {
		os.Remove(tmp)
		WriteJSON(w, http.StatusInternalServerError, map[string]any{"ok": false, "error": err.Error()})
		return
	}
	if err := os.Rename(tmp, s.CacheDir+"/"+key); err != nil {
		os.Remove(tmp)
		WriteJSON(w, http.StatusInternalServerError, map[string]any{"ok": false, "error": err.Error()})
		return
	}
	logInfo("cache_agent", "cache receive: wrote %s (%d bytes, %.0fms, %s)", key, n, float64(d.Milliseconds()), fmtRate(n, d))
	WriteJSON(w, http.StatusOK, map[string]any{"ok": true, "bytes": n})
}

type dirFile struct {
	name string
	size int64
	mt   time.Time
}

// cacheDirFiles lists regular files in the cache dir (skipping in-flight
// transfer temp files).
func cacheDirFiles(dir string) []dirFile {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return []dirFile{}
	}
	out := make([]dirFile, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(name, ".p2p.tmp") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		out = append(out, dirFile{name: name, size: info.Size(), mt: info.ModTime()})
	}
	return out
}

// evictUntil removes oldest files (by mtime) until `need` bytes fit within the
// maxBytes budget. Sidecars of an evicted key are removed with it (they are
// never eviction candidates themselves). Returns (fits, used).
func (s *AgentServer) evictUntil(need, maxBytes int64) (bool, int64) {
	evicted := 0
	for {
		files := cacheDirFiles(s.CacheDir)
		var used int64
		for _, f := range files {
			used += f.size
		}
		if used <= maxBytes-need {
			break
		}
		idx := -1
		for i, f := range files {
			if p2pSidecarRe.MatchString(f.name) {
				continue
			}
			if idx == -1 || f.mt.Before(files[idx].mt) {
				idx = i
			}
		}
		if idx == -1 {
			break // nothing left to evict
		}
		victim := files[idx]
		if err := os.Remove(s.CacheDir + "/" + victim.name); err != nil && !os.IsNotExist(err) {
			logWarn("cache_agent", "cache make-space: failed to remove %s: %v", victim.name, err)
		}
		for _, f := range files {
			if isSidecarName(f.name, victim.name) {
				if err := os.Remove(s.CacheDir + "/" + f.name); err != nil && !os.IsNotExist(err) {
					logWarn("cache_agent", "cache make-space: failed to remove sidecar %s: %v", f.name, err)
				}
			}
		}
		evicted++
		logInfo("cache_agent", "cache make-space: evicted %s (%d bytes)", victim.name, victim.size)
	}
	files := cacheDirFiles(s.CacheDir)
	var used int64
	for _, f := range files {
		used += f.size
	}
	ok := used <= maxBytes-need
	logInfo("cache_agent", "cache make-space: need=%d max=%d used=%d evicted=%d ok=%v", need, maxBytes, used, evicted, ok)
	return ok, used
}

// handleMakeSpace (target side): evicts oldest files (by mtime) until `need`
// bytes fit within the `max` total budget. Sidecars (.ckpt*) of an evicted
// key are removed with it.
func (s *AgentServer) handleMakeSpace(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		WriteJSON(w, http.StatusMethodNotAllowed, map[string]any{"ok": false, "error": "method not allowed"})
		return
	}
	q := r.URL.Query()
	need, errNeed := strconv.ParseInt(q.Get("need"), 10, 64)
	maxBytes, errMax := strconv.ParseInt(q.Get("max"), 10, 64)
	if errNeed != nil || errMax != nil || need <= 0 || maxBytes <= 0 {
		WriteJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": "need and max parameters are required"})
		return
	}
	ok, used := s.evictUntil(need, maxBytes)
	WriteJSON(w, http.StatusOK, map[string]any{"ok": ok, "used": used})
}

// handleFileContent (source side): streams a cache file's raw content.
func (s *AgentServer) handleFileContent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		WriteJSON(w, http.StatusMethodNotAllowed, map[string]any{"ok": false, "error": "method not allowed"})
		return
	}
	key := r.URL.Query().Get("key")
	if key == "" {
		WriteJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": "key parameter is required"})
		return
	}
	cachePath := s.CacheDir + "/" + key
	f, err := os.Open(cachePath)
	if err != nil {
		if os.IsNotExist(err) {
			WriteJSON(w, http.StatusNotFound, map[string]any{"ok": false, "error": "file not found"})
			return
		}
		WriteJSON(w, http.StatusInternalServerError, map[string]any{"ok": false, "error": err.Error()})
		return
	}
	defer f.Close()
	fi, _ := f.Stat()
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Length", strconv.FormatInt(fi.Size(), 10))
	w.WriteHeader(http.StatusOK)
	start := time.Now()
	buf := make([]byte, p2pCopyBufSize)
	_, _ = io.CopyBuffer(w, f, buf)
	d := time.Since(start)
	logInfo("cache_agent", "cache file download: %s (%d bytes, %.0fms, %s)", key, fi.Size(), float64(d.Milliseconds()), fmtRate(fi.Size(), d))
}

// WriteJSON writes a JSON response with the given status code.
func WriteJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
