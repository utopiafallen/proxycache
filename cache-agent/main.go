// cache-agent — standalone cache-agent binary. Serves a llama.cpp slot-save
// directory over HTTP so a proxycache instance can manage (size-check,
// delete) and migrate (P2P transfer) its KV cache files.
//
// Endpoints:
//
//	POST /cache/delete?key=<basename>  -> {"ok": bool, "error": string}
//	GET  /cache/files/<key>            -> {"size": int, "exists": bool, "sidecars": [string]}
//	POST /cache/files/batch            -> {"results": {key: {...}}}
//	POST /cache/transfer               -> source-side P2P push (main file + sidecars) to a target agent
//	POST /cache/receive?key=<key>      -> raw file upload (target side)
//	GET  /cache/file?key=<key>         -> raw file download (source pull)
//
// The agent NEVER evicts files on its own. It has no view of the kv-meta
// files that live on the owning proxycache, so autonomous deletion would
// desynchronize the two. Budget enforcement belongs to the proxycache ring
// buffer, which evicts by issuing explicit /cache/delete calls for keys it
// picked itself (and removes the corresponding meta).
//
// Checkpoint sidecar files (<key>.ckpt, <key>.ckpt.N) are part of a complete
// cache entry when present: they travel with their key on transfers and are
// removed with it on delete.
//
// NOTE: the root proxycache module carries an embedded copy of this server
// (cacheagent.go AgentServer, started for local backends that set
// "agent_serve_port"). Keep the two implementations in sync.

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

var cacheDir string
var port string
var transferTimeout float64

func main() {
	flag.StringVar(&cacheDir, "cache-dir", "", "Path to llama.cpp slot-save-directory")
	flag.StringVar(&port, "port", "8082", "HTTP listen port")
	flag.Float64Var(&transferTimeout, "transfer-timeout", 300, "Timeout in seconds for one P2P transfer exchange")
	flag.Parse()

	if cacheDir == "" {
		flag.Usage()
		os.Exit(1)
	}
	transferHTTPClient = &http.Client{Timeout: time.Duration(transferTimeout * float64(time.Second))}

	mux := http.NewServeMux()
	mux.HandleFunc("/cache/delete", handleDelete)
	mux.HandleFunc("/cache/files/batch", handleBatchFileSizes)
	mux.HandleFunc("/cache/files/", handleFileSizes)
	mux.HandleFunc("/cache/transfer", handleTransfer)
	mux.HandleFunc("/cache/receive", handleReceive)
	mux.HandleFunc("/cache/file", handleFileContent)

	addr := ":" + port
	log.Printf("cache-agent listening on %s (CACHE_DIR=%s)\n", addr, cacheDir)
	log.Fatal(http.ListenAndServe(addr, mux))
}

// transferHTTPClient bounds agent-to-agent transfer calls (set in main after
// flag parsing).
var transferHTTPClient *http.Client

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

type deleteResponse struct {
	Ok    bool   `json:"ok"`
	Error string `json:"error,omitempty"`
}

type fileResponse struct {
	Size     int64    `json:"size"`
	Exists   bool     `json:"exists"`
	Sidecars []string `json:"sidecars,omitempty"`
}

type batchRequest struct {
	Keys []string `json:"keys"`
}

type batchResponse struct {
	Results map[string]fileResponse `json:"results"`
}

// p2pSidecarRe matches checkpoint sidecar suffixes: .ckpt or .ckpt.N.
var p2pSidecarRe = regexp.MustCompile(`\.ckpt(\.\d+)?$`)

func isSidecarName(name, key string) bool {
	prefix := key + ".ckpt"
	return name == prefix || (len(name) > len(prefix)+1 && strings.HasPrefix(name, prefix+".") && p2pSidecarRe.MatchString(name))
}

// listSidecars returns the names of existing checkpoint sidecar files for key.
func listSidecars(dir, key string) []string {
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

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func handleBatchFileSizes(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(batchResponse{Results: map[string]fileResponse{}})
		return
	}

	var req batchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(batchResponse{Results: map[string]fileResponse{}})
		return
	}

	results := make(map[string]fileResponse)
	for _, key := range req.Keys {
		filepath := cacheDir + "/" + key
		info, err := os.Stat(filepath)
		if err != nil {
			results[key] = fileResponse{Exists: false}
		} else {
			results[key] = fileResponse{Size: info.Size(), Exists: true}
		}
	}

	log.Printf("batch file size: %d keys queried\n", len(req.Keys))
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(batchResponse{Results: results})
}

func handleDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(deleteResponse{Ok: false, Error: "method not allowed"})
		return
	}

	key := r.URL.Query().Get("key")
	if key == "" {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(deleteResponse{Ok: false, Error: "key parameter is required"})
		return
	}

	filepath := cacheDir + "/" + key

	if err := os.Remove(filepath); err != nil {
		if os.IsNotExist(err) {
			log.Printf("cache delete: file not found: %s\n", key)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotFound)
			json.NewEncoder(w).Encode(deleteResponse{Ok: false, Error: "file not found"})
			return
		}
		log.Printf("cache delete: failed to remove %s: %v\n", key, err)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(deleteResponse{Ok: false, Error: err.Error()})
		return
	}

	// Delete ckpt sidecar files (<key>.ckpt, <key>.ckpt.0, etc.)
	if entries, err := os.ReadDir(cacheDir); err == nil {
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			name := entry.Name()
			if isSidecarName(name, key) {
				sidecar := cacheDir + "/" + name
				if err := os.Remove(sidecar); err == nil {
					log.Printf("cache delete: sidecar %s\n", name)
				} else {
					log.Printf("cache delete: failed to remove sidecar %s: %v\n", name, err)
				}
			}
		}
	}

	log.Printf("cache delete: %s\n", key)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(deleteResponse{Ok: true})
}

func handleFileSizes(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(fileResponse{Exists: false})
		return
	}

	basename := r.URL.Path[len("/cache/files/"):]
	if basename == "" {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(fileResponse{Exists: false})
		return
	}

	filepath := cacheDir + "/" + basename

	info, err := os.Stat(filepath)
	if err != nil {
		if os.IsNotExist(err) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotFound)
			json.NewEncoder(w).Encode(fileResponse{Exists: false})
			return
		}
		log.Printf("cache file info: failed to stat %s: %v\n", basename, err)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(fileResponse{Exists: false})
		return
	}

	log.Printf("cache file size: %s size=%d\n", basename, info.Size())
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(fileResponse{Size: info.Size(), Exists: true, Sidecars: listSidecars(cacheDir, basename)})
}

// --- P2P transfer endpoints ---

// handleTransfer (source side): pushes key's main file plus any existing
// checkpoint sidecars to the target agent.
func handleTransfer(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"ok": false, "error": "method not allowed"})
		return
	}
	q := r.URL.Query()
	key := q.Get("key")
	target := q.Get("target")
	if key == "" || target == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": "key and target parameters are required"})
		return
	}
	cachePath := cacheDir + "/" + key
	fi, err := os.Stat(cachePath)
	if err != nil || fi.IsDir() {
		writeJSON(w, http.StatusNotFound, map[string]any{"ok": false, "error": "file not found"})
		return
	}
	size := fi.Size()
	sidecars := listSidecars(cacheDir, key)
	total := size
	for _, s := range sidecars {
		if info, err := os.Stat(cacheDir + "/" + s); err == nil {
			total += info.Size()
		}
	}
	fail := func(msg string) {
		log.Printf("cache transfer: %s\n", msg)
		writeJSON(w, http.StatusOK, map[string]any{"ok": false, "error": msg})
	}
	push := func(name string) (bool, string, int64, time.Duration) {
		f, err := os.Open(cacheDir + "/" + name)
		if err != nil {
			return false, err.Error(), 0, 0
		}
		defer f.Close()
		fi, _ := f.Stat()
		start := time.Now()
		// Explicit Content-Length: a bare *os.File body would go chunked;
		// framing with a known length keeps the receive side simple.
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
	log.Printf("cache transfer: pushed %s (%d bytes, %.0fms, %s) to %s\n", key, n, float64(d.Milliseconds()), fmtRate(n, d), target)
	if !ok {
		fail(fmt.Sprintf("push of %s to %s failed: %s", key, target, errMsg))
		return
	}
	for _, s := range sidecars {
		ok, errMsg, n, d := push(s)
		log.Printf("cache transfer: pushed %s (%d bytes, %.0fms, %s) to %s\n", s, n, float64(d.Milliseconds()), fmtRate(n, d), target)
		if !ok {
			fail(fmt.Sprintf("push of sidecar %s to %s failed: %s", s, target, errMsg))
			return
		}
	}
	log.Printf("cache transfer: transferred %s (%d bytes total incl. %d sidecar(s)) to %s\n", key, total, len(sidecars), target)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "bytes": size, "sidecars": len(sidecars)})
}

// handleReceive (target side): writes the raw request body to the cache dir
// atomically (tmp + rename). The owning proxycache enforces the size budget
// itself (ring eviction via /cache/delete), so the agent just stores.
func handleReceive(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"ok": false, "error": "method not allowed"})
		return
	}
	key := r.URL.Query().Get("key")
	if key == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": "key parameter is required"})
		return
	}
	tmp := cacheDir + "/" + key + ".p2p.tmp"
	out, err := os.Create(tmp)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"ok": false, "error": err.Error()})
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
		writeJSON(w, http.StatusInternalServerError, map[string]any{"ok": false, "error": err.Error()})
		return
	}
	if err := os.Rename(tmp, cacheDir+"/"+key); err != nil {
		os.Remove(tmp)
		writeJSON(w, http.StatusInternalServerError, map[string]any{"ok": false, "error": err.Error()})
		return
	}
	log.Printf("cache receive: wrote %s (%d bytes, %.0fms, %s)\n", key, n, float64(d.Milliseconds()), fmtRate(n, d))
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "bytes": n})
}

// handleFileContent (source side): streams a cache file's (or sidecar's) raw
// content.
func handleFileContent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"ok": false, "error": "method not allowed"})
		return
	}
	key := r.URL.Query().Get("key")
	if key == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": "key parameter is required"})
		return
	}
	f, err := os.Open(cacheDir + "/" + key)
	if err != nil {
		if os.IsNotExist(err) {
			writeJSON(w, http.StatusNotFound, map[string]any{"ok": false, "error": "file not found"})
			return
		}
		writeJSON(w, http.StatusInternalServerError, map[string]any{"ok": false, "error": err.Error()})
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
	log.Printf("cache file download: %s (%d bytes, %.0fms, %s)\n", key, fi.Size(), float64(d.Milliseconds()), fmtRate(fi.Size(), d))
}
