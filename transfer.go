// transfer.go — P2P cache file transfer between backends. When the matcher
// routes a cache-hit request to a backend that does not hold the cache, it
// asks (async) for the cache file to migrate from the hit backend to the
// serving backend so a restore is possible when the request runs. The
// receiving side may evict old cache entries to make space.
//
// Four backend-type combinations are supported:
//
//	source agent + target agent : source agent pushes to target agent (true P2P)
//	source local + target local : proxy copies the file directly
//	source local + target agent : proxy uploads to the target agent
//	source agent + target local : proxy pulls from the source agent
//
// Checkpoint sidecar files (<key>.ckpt, <key>.ckpt.N) are part of a complete
// cache entry when present: they are transferred with their main file in all
// four paths and any sidecar failure fails the transfer.
//
// On success the meta entry and the target's cache ring are updated so the
// transferred file participates in disk scans and eviction like a native one.

package proxycache

import (
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type transferTracker struct {
	mu       sync.Mutex
	inFlight map[string]bool
}

var theTransferTracker = &transferTracker{inFlight: map[string]bool{}}

func transferIDFor(key, dstBe string) string { return key + "|" + dstBe }

// TransferInFlight reports whether a P2P transfer of key to dstBe is active.
func TransferInFlight(key, dstBe string) bool {
	theTransferTracker.mu.Lock()
	defer theTransferTracker.mu.Unlock()
	return theTransferTracker.inFlight[transferIDFor(key, dstBe)]
}

// RequestCacheTransfer starts an async P2P transfer of cache key from srcBe to
// dstBe. Deduped per (key, dst) and skipped when dst already holds the file.
// Returns true when a new transfer was started.
func RequestCacheTransfer(srcBe, dstBe, key string) bool {
	id := transferIDFor(key, dstBe)
	theTransferTracker.mu.Lock()
	if theTransferTracker.inFlight[id] {
		theTransferTracker.mu.Unlock()
		logInfo("transfer", "P2P transfer of key %s to '%s' already in flight; skipping", key16(key), dstBe)
		return false
	}
	if backendManager.CacheExists(dstBe, key) {
		theTransferTracker.mu.Unlock()
		return false
	}
	theTransferTracker.inFlight[id] = true
	theTransferTracker.mu.Unlock()
	go func() {
		defer func() {
			theTransferTracker.mu.Lock()
			delete(theTransferTracker.inFlight, id)
			theTransferTracker.mu.Unlock()
		}()
		runCacheTransfer(srcBe, dstBe, key)
	}()
	logInfo("transfer", "P2P transfer requested: key '%s' from '%s' to '%s'", key16(key), srcBe, dstBe)
	return true
}

func maxCacheBytes(backendID string) int64 {
	return int64(backendManager.GetCacheMaxSizeGB(backendID) * 1024 * 1024 * 1024)
}

func runCacheTransfer(srcBe, dstBe, key string) {
	t0 := time.Now()
	srcMeta := kvMeta.ReadMeta(key, srcBe)
	size := backendManager.CacheGetSize(srcBe, key)
	blocks := kvMeta.GetBlocks(key, srcBe)
	modelID := ""
	nTokens := 0
	if srcMeta != nil {
		modelID = metaString(srcMeta, "model_id")
		nTokens = int(MetaFloat(srcMeta, "n_tokens"))
	}
	var sidecars []string
	var sidecarSize int64
	fail := func(reason string) {
		recordTransferEvent(srcBe, dstBe, key, size, len(sidecars), t0, false, reason)
		logWarn("transfer", "P2P transfer failed: key '%s' from '%s' to '%s': %s", key16(key), srcBe, dstBe, reason)
	}
	if size <= 0 {
		fail("source file missing or size unknown")
		return
	}
	srcInfo := backendManager.Info(srcBe)
	dstInfo := backendManager.Info(dstBe)
	if srcInfo == nil || dstInfo == nil {
		fail("unknown backend")
		return
	}

	// Checkpoint sidecars (key.ckpt, key.ckpt.N) make the entry complete only
	// when transferred too; a sidecar failure fails the whole transfer.
	if srcInfo.AgentClient != nil {
		sidecars = srcInfo.AgentClient.Sidecars(key)
		if len(sidecars) > 0 {
			for _, info := range srcInfo.AgentClient.BatchGetFileSizes(sidecars) {
				if v, ok := ToFloat(info["size"]); ok {
					sidecarSize += int64(v)
				}
			}
		}
	} else {
		sidecars = listLocalSidecars(srcInfo.CacheDir, key)
		for _, sc := range sidecars {
			if info, err := os.Stat(filepath.Join(srcInfo.CacheDir, sc)); err == nil {
				sidecarSize += info.Size()
			}
		}
	}

	var ok bool
	var detail string
	switch {
	case srcInfo.AgentClient != nil && dstInfo.AgentClient != nil:
		// The source agent pushes the main file plus sidecars and sizes the
		// target make-space for the full entry itself.
		ok = srcInfo.AgentClient.Transfer(key, dstInfo.AgentClient.BaseURL, maxCacheBytes(dstBe))
		if !ok {
			detail = "agent-to-agent transfer reported failure"
		}
	case srcInfo.AgentClient == nil && dstInfo.AgentClient == nil:
		slotManager.Get(dstBe).MakeSpaceFor(int64(size) + sidecarSize)
		ok = copyCacheFileLocal(srcInfo.CacheDir, dstInfo.CacheDir, key)
		if ok {
			for _, sc := range sidecars {
				if !copyCacheFileLocal(srcInfo.CacheDir, dstInfo.CacheDir, sc) {
					ok = false
					detail = "local copy of sidecar " + sc + " failed"
					break
				}
			}
		} else {
			detail = "local copy failed"
		}
	case srcInfo.AgentClient == nil:
		// The target agent makes space per file from its content length.
		ok = dstInfo.AgentClient.UploadFile(key, filepath.Join(srcInfo.CacheDir, key), maxCacheBytes(dstBe))
		if ok {
			for _, sc := range sidecars {
				if !dstInfo.AgentClient.UploadFile(sc, filepath.Join(srcInfo.CacheDir, sc), maxCacheBytes(dstBe)) {
					ok = false
					detail = "upload of sidecar " + sc + " failed"
					break
				}
			}
		} else {
			detail = "upload to target agent failed"
		}
	default:
		body, _, err := srcInfo.AgentClient.FetchStream(key)
		if err != nil {
			fail(err.Error())
			return
		}
		slotManager.Get(dstBe).MakeSpaceFor(int64(size) + sidecarSize)
		ok = streamToCacheFile(dstInfo.CacheDir, key, body)
		body.Close()
		if ok {
			for _, sc := range sidecars {
				scBody, _, err := srcInfo.AgentClient.FetchStream(sc)
				if err != nil {
					ok = false
					detail = "fetch of sidecar " + sc + " failed"
					break
				}
				got := streamToCacheFile(dstInfo.CacheDir, sc, scBody)
				scBody.Close()
				if !got {
					ok = false
					detail = "writing sidecar " + sc + " to local cache dir failed"
					break
				}
			}
		} else {
			detail = "writing to local cache dir failed"
		}
	}

	if ok {
		kvMeta.WriteMeta(key, nTokens, blocks, WordsPerBlock, modelID, dstBe, size)
		slotManager.Get(dstBe).AddTransferredEntry(key, int64(size))
		d := time.Since(t0)
		recordTransferEvent(srcBe, dstBe, key, size, len(sidecars), t0, true, "")
		logInfo("transfer", "P2P transfer complete: key '%s' from '%s' to '%s' (%d bytes + %d sidecar(s), %.0fms, %s)",
			key16(key), srcBe, dstBe, size, len(sidecars), float64(d.Milliseconds()), fmtRate(int64(size)+sidecarSize, d))
		return
	}
	fail(detail)
}

// copyCacheFileLocal copies a cache file between two local dirs by streaming
// (fixed-size buffer, no full-file RAM copy) into a tmp file + rename.
func copyCacheFileLocal(srcDir, dstDir, key string) bool {
	in, err := os.Open(filepath.Join(srcDir, key))
	if err != nil {
		logWarn("transfer", "Local transfer open failed for key %s from %s: %v", key16(key), srcDir, err)
		return false
	}
	defer in.Close()
	return streamToCacheFile(dstDir, key, in)
}

// streamToCacheFile streams r into a local cache dir atomically (tmp + rename)
// using a fixed-size buffer.
func streamToCacheFile(dstDir, key string, r io.Reader) bool {
	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		logWarn("transfer", "Local transfer mkdir failed for %s: %v", dstDir, err)
		return false
	}
	tmp := filepath.Join(dstDir, key+".p2p.tmp")
	out, err := os.Create(tmp)
	if err != nil {
		logWarn("transfer", "Local transfer create failed for key %s in %s: %v", key16(key), dstDir, err)
		return false
	}
	buf := make([]byte, p2pCopyBufSize)
	_, err = io.CopyBuffer(out, r, buf)
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmp)
		logWarn("transfer", "Local transfer write failed for key %s: %v", key16(key), err)
		return false
	}
	if err := os.Rename(tmp, filepath.Join(dstDir, key)); err != nil {
		os.Remove(tmp)
		logWarn("transfer", "Local transfer rename failed for key %s: %v", key16(key), err)
		return false
	}
	return true
}

func recordTransferEvent(srcBe, dstBe, key string, size int, sidecarCount int, t0 time.Time, ok bool, detail string) {
	event := map[string]any{
		"event":    "cache_transfer",
		"src":      srcBe,
		"dst":      dstBe,
		"key":      key16(key),
		"bytes":    size,
		"sidecars": sidecarCount,
		"ok":       ok,
		"ms":       round1(float64(time.Since(t0).Nanoseconds()) / 1e6),
	}
	if detail != "" {
		event["error"] = detail
	}
	Metrics.Record(event)
}
