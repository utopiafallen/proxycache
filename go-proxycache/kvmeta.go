// kvmeta.go — KVMetaManager: single interface for all kv-meta file operations
// (read, write, list, delete, reconcile, restore-candidate search).

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// KVMetaManager manages kv-meta files: read, write, list, delete, reconcile.
// It is stateless; all state lives in files under META_DIR.
//
// writeMu serializes meta file writes: os.WriteFile truncates in place, so a
// concurrent reader (scan/reconcile) could observe a half-written file, and
// the recompute-penalty read-modify-write would lose updates.
type KVMetaManager struct {
	writeMu sync.Mutex
}

// atomicWriteFile writes to path+".tmp" then renames over path, so readers
// never observe a partial file. Callers must hold writeMu.
func (kv *KVMetaManager) atomicWriteFile(path string, data []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

// kvMeta is the singleton instance.
var kvMeta = &KVMetaManager{}

// nowFloat returns the current Unix time as float64 seconds (time.time() parity).
func nowFloat() float64 {
	return float64(time.Now().UnixNano()) / 1e9
}

// MetaFilePath returns the full path to a meta file.
func (kv *KVMetaManager) MetaFilePath(key, backendKey string) string {
	if backendKey != "" {
		return filepath.Join(MetaDir, SanitizeBackendDir(backendKey), key+MetaSuffix)
	}
	return filepath.Join(MetaDir, key+MetaSuffix)
}

// MetaDirPath returns the path to a backend's meta directory.
func (kv *KVMetaManager) MetaDirPath(backendKey string) string {
	return filepath.Join(MetaDir, SanitizeBackendDir(backendKey))
}

// ScanAllMeta scans meta files for a backend, sorted newest-first (mtime desc).
func (kv *KVMetaManager) ScanAllMeta(backendKey string) []map[string]any {
	searchDir := kv.MetaDirPath(backendKey)
	if st, err := os.Stat(searchDir); err != nil || !st.IsDir() {
		logWarn("kv_meta", "Meta directory missing for backend '%s'", backendKey)
		return []map[string]any{}
	}
	matches, _ := filepath.Glob(filepath.Join(searchDir, "*"+MetaSuffix))
	type fileMT struct {
		path string
		mt   time.Time
	}
	files := make([]fileMT, 0, len(matches))
	for _, f := range matches {
		st, err := os.Stat(f)
		mt := time.Time{}
		if err == nil {
			mt = st.ModTime()
		}
		files = append(files, fileMT{path: f, mt: mt})
	}
	// Stable sort by mtime desc (Python: sorted(..., key=getmtime, reverse=True)).
	sort.SliceStable(files, func(i, j int) bool {
		return files[i].mt.After(files[j].mt)
	})

	metas := make([]map[string]any, 0, len(files))
	for _, f := range files {
		data, err := os.ReadFile(f.path)
		if err != nil {
			logWarn("kv_meta", "Failed to read meta file %s: %s", f.path, err)
			continue
		}
		var meta map[string]any
		if err := json.Unmarshal(data, &meta); err != nil {
			logWarn("kv_meta", "Failed to read meta file %s: %s", f.path, err)
			continue
		}
		metas = append(metas, meta)
	}
	logWarn("kv_meta", "Meta scan found %d entries", len(metas))
	return metas
}

// ReadMeta reads and returns the full meta dict for a key, or nil on failure.
func (kv *KVMetaManager) ReadMeta(key, backendID string) map[string]any {
	path := kv.MetaFilePath(key, backendID)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var meta map[string]any
	if err := json.Unmarshal(data, &meta); err != nil {
		return nil
	}
	return meta
}

// GetBlocks returns the blocks list for a key, or nil if not found.
func (kv *KVMetaManager) GetBlocks(key, backendID string) []string {
	meta := kv.ReadMeta(key, backendID)
	if len(meta) > 0 {
		blocks, _ := meta["blocks"].([]any)
		out := make([]string, 0, len(blocks))
		for _, b := range blocks {
			if s, ok := b.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

// ListKeys returns the cache keys for a backend (filenames without suffix), sorted.
func (kv *KVMetaManager) ListKeys(backendID string) []string {
	backendPath := kv.MetaDirPath(backendID)
	entries, err := os.ReadDir(backendPath)
	if err != nil {
		return []string{}
	}
	keys := []string{}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(name, MetaSuffix) {
			keys = append(keys, strings.TrimSuffix(name, MetaSuffix))
		}
	}
	sort.Strings(keys)
	return keys
}

// metaFile is the on-disk meta schema. Field order mirrors the Python writer.
type metaFile struct {
	Key              string   `json:"key"`
	ModelID          string   `json:"model_id"`
	Backend          string   `json:"backend"`
	NTokens          int      `json:"n_tokens"`
	WPB              int      `json:"wpb"`
	Blocks           []string `json:"blocks"`
	CacheSize        int      `json:"cache_size"`
	RecomputePenalty int      `json:"recompute_penalty"`
	LastWritten      float64  `json:"last_written"`
}

// WriteMeta writes/overwrites the meta file for a key.
func (kv *KVMetaManager) WriteMeta(key string, nTokens int, blocks []string, wpb int, modelID, backendID string, cacheSize int) {
	meta := metaFile{
		Key:              key,
		ModelID:          modelID,
		Backend:          backendID,
		NTokens:          nTokens,
		WPB:              wpb,
		Blocks:           blocks,
		CacheSize:        cacheSize,
		RecomputePenalty: 0,
		LastWritten:      nowFloat(),
	}
	d := kv.MetaDirPath(backendID)
	if err := os.MkdirAll(d, 0o755); err != nil {
		logWarn("kv_meta", "Failed to create meta dir %s: %s", d, err)
		return
	}
	data, err := marshalMetaJSON(meta)
	if err != nil {
		logWarn("kv_meta", "Failed to marshal meta for key %s: %s", truncateKey(key), err)
		return
	}
	path := filepath.Join(d, key+MetaSuffix)
	kv.writeMu.Lock()
	defer kv.writeMu.Unlock()
	if err := kv.atomicWriteFile(path, data); err != nil {
		logWarn("kv_meta", "Failed to write meta file %s: %s", path, err)
	}
	logInfo("kv_meta", "Saved cache for key %s (model_id: %s, backend: %s, %d blocks, %d bytes)",
		truncateKey(key), modelID, backendID, len(blocks), cacheSize)
}

// marshalMetaJSON mirrors json.dump(meta, indent=2, ensure_ascii=False):
// 2-space indent, no HTML escaping, no trailing newline.
func marshalMetaJSON(v any) ([]byte, error) {
	var buf strings.Builder
	enc := json.NewEncoder(stringsWriter{&buf})
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	out := buf.String()
	if strings.HasSuffix(out, "\n") {
		out = out[:len(out)-1]
	}
	return []byte(out), nil
}

type stringsWriter struct {
	s *strings.Builder
}

func (w stringsWriter) Write(p []byte) (int, error) {
	return w.s.Write(p)
}

// DeleteMetaFile deletes the meta file for a key across all backends.
// Returns true if a file was deleted.
func (kv *KVMetaManager) DeleteMetaFile(key string) bool {
	if st, err := os.Stat(MetaDir); err != nil || !st.IsDir() {
		return false
	}
	entries, err := os.ReadDir(MetaDir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		candidate := filepath.Join(MetaDir, e.Name(), key+MetaSuffix)
		if _, err := os.Stat(candidate); err == nil {
			if err := os.Remove(candidate); err != nil {
				return false
			}
			return true
		}
	}
	return false
}

// IncrementRecomputePenalty increments the recompute_penalty counter on a meta file.
// The lock spans read-modify-write so concurrent increments don't lose updates.
func (kv *KVMetaManager) IncrementRecomputePenalty(key, backendID string) {
	kv.writeMu.Lock()
	defer kv.writeMu.Unlock()
	path := kv.MetaFilePath(key, backendID)
	data, err := os.ReadFile(path)
	if err != nil {
		logWarn("kv_meta", "Failed to increment recompute_penalty for key %s: %s", truncateKey(key), err)
		return
	}
	var meta map[string]any
	if err := json.Unmarshal(data, &meta); err != nil {
		logWarn("kv_meta", "Failed to increment recompute_penalty for key %s: %s", truncateKey(key), err)
		return
	}
	penalty := metaFloat(meta, "recompute_penalty") + 1
	meta["recompute_penalty"] = penalty
	meta["last_updated_penalty"] = nowFloat()
	out, err := marshalMetaJSON(meta)
	if err != nil {
		logWarn("kv_meta", "Failed to increment recompute_penalty for key %s: %s", truncateKey(key), err)
		return
	}
	if err := kv.atomicWriteFile(path, out); err != nil {
		logWarn("kv_meta", "Failed to increment recompute_penalty for key %s: %s", truncateKey(key), err)
		return
	}
	logInfo("kv_meta", "Incremented recompute_penalty for key %s to %d", truncateKey(key), int(penalty))
}

// FindRestoreCandidate checks whether a specific key is a valid restore candidate.
// Returns (key, ratio, ok).
func (kv *KVMetaManager) FindRestoreCandidate(key string, wpb int, th float64, reqBlocks []string, backendID string) (string, float64, bool) {
	path := kv.MetaFilePath(key, backendID)
	data, err := os.ReadFile(path)
	if err != nil {
		return "", 0, false
	}
	var meta map[string]any
	if err := json.Unmarshal(data, &meta); err != nil {
		return "", 0, false
	}
	candBlocks := metaStringSlice(meta, "blocks")
	if int(metaFloat(meta, "wpb")) != wpb {
		return "", 0, false
	}
	if len(candBlocks) > len(reqBlocks) {
		return "", 0, false
	}
	lcp := LCPBlocks(reqBlocks, candBlocks)
	ratio := float64(lcp) / mathMaxFloat(1, float64(len(reqBlocks)))
	if ratio >= th {
		return key, ratio, true
	}
	return "", 0, false
}

// FindBestRestoreCandidate finds the best restore candidate among meta files
// for the current model+backend only. Returns (key, ratio, ok).
func (kv *KVMetaManager) FindBestRestoreCandidate(reqBlocks []string, wpb int, th float64, modelID, backendID string) (string, float64, bool) {
	metas := kv.ScanAllMeta(backendID)
	var bestKey string
	bestRatio := 0.0
	bestScore := 0.0
	found := false

	for _, meta := range metas {
		if metaString(meta, "model_id") != modelID {
			continue
		}
		if metaString(meta, "backend") != backendID {
			continue
		}
		if int(metaFloat(meta, "wpb")) != wpb {
			continue
		}
		candBlocks := metaStringSlice(meta, "blocks")
		if len(candBlocks) > len(reqBlocks) {
			continue
		}
		lcp := LCPBlocks(reqBlocks, candBlocks)
		ratio := float64(lcp) / mathMaxFloat(1, float64(len(reqBlocks)))
		penalty := metaFloat(meta, "recompute_penalty")
		score := ratio * mathMaxFloat(0, 1-0.1*penalty)
		if ratio >= th && score > bestScore {
			bestScore = score
			bestRatio = ratio
			bestKey = metaString(meta, "key")
			found = true
		}
	}
	if found {
		return bestKey, bestRatio, true
	}
	return "", 0, false
}

// GetCacheSize returns the cache size in bytes for a key, or 0 if not found.
func (kv *KVMetaManager) GetCacheSize(backendID, key string) int {
	meta := kv.ReadMeta(key, backendID)
	if len(meta) > 0 {
		size := int(metaFloat(meta, "cache_size"))
		if size != 0 {
			return size
		}
	}
	return backendManager.CacheGetSize(backendID, key)
}

// GetLastUsedTime returns the last-used timestamp for a cache file.
func (kv *KVMetaManager) GetLastUsedTime(key, backendID string) float64 {
	path := kv.MetaFilePath(key, backendID)
	if _, err := os.Stat(path); err == nil {
		data, err := os.ReadFile(path)
		if err == nil {
			var meta map[string]any
			if json.Unmarshal(data, &meta) == nil {
				for _, field := range []string{"last_read", "last_written", "timestamp"} {
					if v, ok := meta[field]; ok {
						if f, ok := toFloat(v); ok {
							return f
						}
					}
				}
			}
		}
	}
	return backendManager.CacheGetMtime(backendID, key)
}

// Reconcile deletes meta files whose cache files are gone (and corrupted
// meta files). Returns the count deleted.
func (kv *KVMetaManager) Reconcile(backendKeys []string) int {
	deleted := 0
	deletedBackends := map[string]bool{}

	for _, backendKey := range backendKeys {
		backendDir := kv.MetaDirPath(backendKey)
		if st, err := os.Stat(backendDir); err != nil || !st.IsDir() {
			continue
		}
		metaFiles, _ := filepath.Glob(filepath.Join(backendDir, "*"+MetaSuffix))
		sort.Strings(metaFiles)

		type validEntry struct {
			metaPath  string
			basename  string
			cachename string
		}
		var validEntries []validEntry

		// First pass: read all meta files, remove corrupted ones.
		for _, metaPath := range metaFiles {
			basename := filepath.Base(metaPath)
			cachename := strings.TrimSuffix(basename, MetaSuffix)
			data, err := os.ReadFile(metaPath)
			if err != nil || json.Valid(data) == false {
				logWarn("kv_meta", "Removed corrupted meta file: %s", basename)
				if err := os.Remove(metaPath); err == nil {
					deleted++
					deletedBackends[backendKey] = true
				}
				continue
			}
			validEntries = append(validEntries, validEntry{metaPath, basename, cachename})
		}

		// Second pass: check cache existence.
		for _, ve := range validEntries {
			cacheExists := backendManager.CacheExists(backendKey, ve.cachename)
			if !cacheExists {
				logInfo("kv_meta", "Removed orphan meta file (no matching cache): %s", ve.basename)
				if err := os.Remove(ve.metaPath); err == nil {
					deleted++
					deletedBackends[backendKey] = true
				}
			}
		}

		if deletedBackends[backendKey] {
			if entries, err := os.ReadDir(backendDir); err == nil && len(entries) == 0 {
				if err := os.Remove(backendDir); err == nil {
					logInfo("kv_meta", "Removed empty backend directory: %s", backendKey)
				}
			}
		}
	}
	logInfo("kv_meta", "Finished reconciling meta files with llama cache directory")
	return deleted
}

// --- meta access helpers ---------------------------------------------------

func metaString(meta map[string]any, key string) string {
	s, _ := meta[key].(string)
	return s
}

func metaFloat(meta map[string]any, key string) float64 {
	if v, ok := meta[key]; ok {
		if f, ok := toFloat(v); ok {
			return f
		}
	}
	return 0
}

func metaStringSlice(meta map[string]any, key string) []string {
	out := []string{}
	if list, ok := meta[key].([]any); ok {
		for _, v := range list {
			if s, ok := v.(string); ok {
				out = append(out, s)
			}
		}
	}
	return out
}

// toFloat converts a JSON scalar (float64, int, or numeric string) to float64.
func toFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case int:
		return float64(x), true
	case string:
		var f float64
		if _, err := fmt.Sscanf(x, "%f", &f); err == nil {
			return f, true
		}
	}
	return 0, false
}

func mathMaxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
