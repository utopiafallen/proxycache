// cache-agent — standalone cache-agent binary. Serves a llama.cpp slot-save
// directory over HTTP so a proxycache instance can manage (size-check,
// delete, evict) and migrate (P2P transfer) its KV cache files.
//
// Endpoints:
//
//	POST /cache/delete?key=<basename>  -> {"ok": bool, "error": string}
//	GET  /cache/files/<key>            -> {"size": int, "exists": bool, "sidecars": [string]}
//	POST /cache/files/batch            -> {"results": {key: {...}}}
//	POST /cache/transfer               -> source-side P2P push (main file + sidecars) to a target agent
//	POST /cache/receive?key=<key>      -> raw file upload (target side; make-space when max_bytes given)
//	POST /cache/make-space             -> evict oldest files to fit a budget
//	GET  /cache/file?key=<key>         -> raw file download (source pull)
//
// Checkpoint sidecar files (<key>.ckpt, <key>.ckpt.N) are part of a complete
// cache entry when present: they travel with their key on transfers and are
// removed with it on delete/eviction.
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
	mux.HandleFunc("/cache/make-space", handleMakeSpace)
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

// evictUntil removes oldest files (by mtime) until `need` extra bytes fit
// within the `maxBytes` budget. Sidecars of an evicted key are removed with
// it (they are never eviction candidates themselves). Returns (fits, used).
func evictUntil(need, maxBytes int64) (bool, int64) {
	evicted := 0
	for {
		files := cacheDirFiles(cacheDir)
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
		if err := os.Remove(cacheDir + "/" + victim.name); err != nil && !os.IsNotExist(err) {
			log.Printf("cache make-space: failed to remove %s: %v\n", victim.name, err)
		}
		for _, f := range files {
			if isSidecarName(f.name, victim.name) {
				if err := os.Remove(cacheDir + "/" + f.name); err != nil && !os.IsNotExist(err) {
					log.Printf("cache make-space: failed to remove sidecar %s: %v\n", f.name, err)
				}
			}
		}
		evicted++
		log.Printf("cache make-space: evicted %s (%d bytes)\n", victim.name, victim.size)
	}
	files := cacheDirFiles(cacheDir)
	var used int64
	for _, f := range files {
		used += f.size
	}
	ok := used <= maxBytes-need
	log.Printf("cache make-space: need=%d max=%d used=%d evicted=%d ok=%v\n", need, maxBytes, used, evicted, ok)
	return ok, used
}

func remoteMakeSpace(target string, need, maxBytes int64) (bool, int64) {
	q := "need=" + strconv.FormatInt(need, 10) + "&max=" + strconv.FormatInt(maxBytes, 10)
	resp, err := transferHTTPClient.Post(strings.TrimRight(target, "/")+"/cache/make-space?"+q, "", nil)
	if err != nil {
		log.Printf("cache transfer: make-space on %s failed: %v\n", target, err)
		return false, 0
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
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

// handleTransfer (source side): pushes key's main file plus any existing
// checkpoint sidecars to the target agent, asking the target to free room for
// the full entry first when max_bytes > 0.
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
	var maxBytes int64
	if v, err := strconv.ParseInt(q.Get("max_bytes"), 10, 64); err == nil {
		maxBytes = v
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
	if maxBytes > 0 {
		ok, _ := remoteMakeSpace(target, total, maxBytes)
		if !ok {
			fail(fmt.Sprintf("target %s cannot make space for %d bytes", target, total))
			return
		}
	}
	push := func(name string) (bool, string, int64, time.Duration) {
		f, err := os.Open(cacheDir + "/" + name)
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
// atomically (tmp + rename). When max_bytes is given and the body length is
// known, frees room for it within the budget first.
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
	var maxBytes int64
	if v, err := strconv.ParseInt(r.URL.Query().Get("max_bytes"), 10, 64); err == nil {
		maxBytes = v
	}
	if n := r.ContentLength; n > 0 && maxBytes > 0 {
		ok, _ := evictUntil(n, maxBytes)
		if !ok {
			writeJSON(w, http.StatusInsufficientStorage, map[string]any{"ok": false, "error": "insufficient space and no room could be made"})
			return
		}
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

func handleMakeSpace(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"ok": false, "error": "method not allowed"})
		return
	}
	q := r.URL.Query()
	need, errNeed := strconv.ParseInt(q.Get("need"), 10, 64)
	maxBytes, errMax := strconv.ParseInt(q.Get("max"), 10, 64)
	if errNeed != nil || errMax != nil || need <= 0 || maxBytes <= 0 {
		writeJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": "need and max parameters are required"})
		return
	}
	ok, used := evictUntil(need, maxBytes)
	writeJSON(w, http.StatusOK, map[string]any{"ok": ok, "used": used})
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
