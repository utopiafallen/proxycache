// slotmanager.go — per-backend slot pools, KV cache state tracking, cache
// ring buffer with eviction, and the global SlotManager coordinator.

package proxycache

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// GSlot identifies a slot acquired for a request:
// (canonical_model_name, backend_id, slot_id).
type GSlot struct {
	ModelName string
	BackendID string
	SlotID    int
}

// CacheHitType — explicit routing decision that tells downstream what to do
// with the slot.
type CacheHitType string

const (
	// CacheHitDiskRestore — found a disk cache file, restore from specific key.
	CacheHitDiskRestore CacheHitType = "disk_restore"
	// CacheHitSkip — pending slot hit, the slot already has matching KV cache.
	CacheHitSkip CacheHitType = "skip"
)

type ringEntry struct {
	Key  string
	Size int64
	TS   float64
}

// SaveSkipEntry records a save that was skipped so it can be flushed when the
// slot is next used. Six fields mirror the Python tuple:
// (key, blocks, n_tokens, hit_type, serving_be_ratio, recompute_happened).
type SaveSkipEntry struct {
	Key          string
	Blocks       []string
	NTokens      int
	HitType      *CacheHitType
	ServingRatio float64
	Recompute    bool
	Legacy       bool // 3-tuple legacy entry — always save on flush
}

// BackendSlotManager tracks slot state for one backend.
//
// Locking: Mu protects all slot state. saveMu protects CacheRing/totalBytes
// (held across the meta write + eviction in SaveAfter). HTTP calls are always
// made with both locks released.
type BackendSlotManager struct {
	BackendID string

	PoolMu sync.Mutex // protects slotPools, inUse, lastUsed, slotKVState, slotSaveSkipped, SlotAcquiredAt, slotDurationEMA
	RingMu sync.Mutex // protects CacheRing, totalBytes

	slotPools       map[string][]int // model -> sorted ascending slot IDs
	inUse           map[int]bool
	lastUsed        map[int]float64
	slotKVState     map[int][]string
	slotSaveSkipped map[int]*SaveSkipEntry
	SlotAcquiredAt  map[int]float64
	slotDurationEMA float64

	CacheRing  []ringEntry // insertion order
	totalBytes int64
}

func NewBackendSlotManager(backendID string) *BackendSlotManager {
	return &BackendSlotManager{
		BackendID:       backendID,
		slotPools:       map[string][]int{},
		inUse:           map[int]bool{},
		lastUsed:        map[int]float64{},
		slotKVState:     map[int][]string{},
		slotSaveSkipped: map[int]*SaveSkipEntry{},
		SlotAcquiredAt:  map[int]float64{},
		CacheRing:       []ringEntry{},
		totalBytes:      0,
	}
}

// EnsurePool creates or updates the slot pool for a model.
func (b *BackendSlotManager) EnsurePool(modelName string, nSlots int) {
	b.PoolMu.Lock()
	defer b.PoolMu.Unlock()
	oldPool, exists := b.slotPools[modelName]
	newPool := make([]int, nSlots)
	for i := 0; i < nSlots; i++ {
		newPool[i] = i
	}
	if !exists {
		for _, s := range newPool {
			b.lastUsed[s] = 0
			b.inUse[s] = false
		}
		b.slotPools[modelName] = newPool
		logInfo("slot_manager", "Created slot pool for model '%s' on backend '%s' with %d slots", modelName, b.BackendID, nSlots)
		return
	}
	oldSet := make(map[int]bool, len(oldPool))
	for _, s := range oldPool {
		oldSet[s] = true
	}
	for _, s := range newPool {
		if !oldSet[s] {
			b.lastUsed[s] = 0
			b.inUse[s] = false
		}
	}
	for s := range oldSet {
		if s >= nSlots {
			if !b.inUse[s] {
				delete(b.lastUsed, s)
				delete(b.inUse, s)
				delete(b.slotKVState, s)
				delete(b.slotSaveSkipped, s)
			}
		}
	}
	oldCount := len(oldPool)
	b.slotPools[modelName] = newPool
	logInfo("slot_manager", "Updated slot pool for model '%s' on backend '%s': %d -> %d slots", modelName, b.BackendID, oldCount, nSlots)
}

// GetPool returns a copy of the slot pool, or nil if the model is unknown.
func (b *BackendSlotManager) GetPool(modelName string) []int {
	b.PoolMu.Lock()
	defer b.PoolMu.Unlock()
	pool, ok := b.slotPools[modelName]
	if !ok {
		return nil
	}
	out := make([]int, len(pool))
	copy(out, pool)
	return out
}

// TryAcquire returns a slot ID for the model, or -1 if none can be acquired.
// First free slot (ascending, matching CPython set(range(n)) iteration order);
// if all are in use, the oldest by last_used (ties: smallest ID).
func (b *BackendSlotManager) TryAcquire(modelName string) int {
	b.PoolMu.Lock()
	defer b.PoolMu.Unlock()
	pool, ok := b.slotPools[modelName]
	if !ok || len(pool) == 0 {
		return -1
	}
	slotID := -1
	for _, s := range pool {
		if !b.inUse[s] {
			slotID = s
			break
		}
	}
	if slotID == -1 {
		for _, s := range pool {
			if slotID == -1 || b.lastUsed[s] < b.lastUsed[slotID] {
				slotID = s
			}
		}
		if b.inUse[slotID] {
			return -1 // all slots in use
		}
	}
	now := NowFloat()
	b.inUse[slotID] = true
	b.lastUsed[slotID] = now
	b.SlotAcquiredAt[slotID] = now
	return slotID
}

// Release marks a slot free and updates the slot-occupancy duration EMA.
// Returns (duration, ok).
func (b *BackendSlotManager) Release(slotID int) (float64, bool) {
	b.PoolMu.Lock()
	defer b.PoolMu.Unlock()
	b.inUse[slotID] = false
	acquiredAt, ok := b.SlotAcquiredAt[slotID]
	if !ok {
		return 0, false
	}
	delete(b.SlotAcquiredAt, slotID)
	duration := NowFloat() - acquiredAt
	old := b.slotDurationEMA
	if old <= 0 {
		old = CacheHitWaitEMAInitialT
	}
	ema := CacheHitWaitEMAAlpha*duration + (1-CacheHitWaitEMAAlpha)*old
	ema = math.Min(ema, CacheHitWaitEMAMaxT)
	if ema < CacheHitWaitEMAMinT {
		ema = CacheHitWaitEMAMinT
	}
	b.slotDurationEMA = ema
	return duration, true
}

// Invalidate clears the KV cache state for a slot.
func (b *BackendSlotManager) Invalidate(slotID int) {
	b.PoolMu.Lock()
	defer b.PoolMu.Unlock()
	if _, ok := b.slotKVState[slotID]; ok {
		delete(b.slotKVState, slotID)
		logInfo("slot_manager", "Invalidated KV cache tracking for backend '%s' slot %d", b.BackendID, slotID)
	}
}

// GetKVState returns the tracked block hashes for a slot (nil if absent).
func (b *BackendSlotManager) GetKVState(slotID int) []string {
	b.PoolMu.Lock()
	defer b.PoolMu.Unlock()
	return b.slotKVState[slotID]
}

// SetKVState sets the tracked block hashes for a slot.
func (b *BackendSlotManager) SetKVState(slotID int, blocks []string) {
	b.PoolMu.Lock()
	defer b.PoolMu.Unlock()
	b.slotKVState[slotID] = blocks
}

// GetKVStates returns a copy of the slot->blocks map.
func (b *BackendSlotManager) GetKVStates() map[int][]string {
	b.PoolMu.Lock()
	defer b.PoolMu.Unlock()
	out := make(map[int][]string, len(b.slotKVState))
	for k, v := range b.slotKVState {
		out[k] = v
	}
	return out
}

// GetSlotDurationEMA returns the slot-occupancy EMA (initial timeout if unset).
func (b *BackendSlotManager) GetSlotDurationEMA() float64 {
	b.PoolMu.Lock()
	defer b.PoolMu.Unlock()
	if b.slotDurationEMA > 0 {
		return b.slotDurationEMA
	}
	return CacheHitWaitEMAInitialT
}

// ShouldSkipRestore decides whether to skip a disk restore because the slot's
// tracked KV cache already matches the request closely enough. Only safe on
// single-slot backends.
//
// prevPassed distinguishes "caller explicitly passed nil" (fresh slot, never
// skip) from "caller did not pass anything" (fall back to _slot_kv_state).
func (b *BackendSlotManager) ShouldSkipRestore(slotID int, reqBlocks []string, prevBlocks []string, prevPassed bool) bool {
	if !prevPassed {
		prevBlocks = b.GetKVState(slotID)
		logWarn("slot_manager", "[diag] should_skip_restore: slot %d, prev_blocks not passed, fell back to _slot_kv_state: %d blocks", slotID, len(prevBlocks))
	} else if prevBlocks == nil {
		logWarn("slot_manager", "[diag] should_skip_restore: slot %d, prev_blocks explicitly None (fresh slot), not skipping", slotID)
		return false
	}
	if len(prevBlocks) == 0 {
		return false
	}
	b.PoolMu.Lock()
	for _, pool := range b.slotPools {
		if len(pool) > 1 {
			b.PoolMu.Unlock()
			return false // multi-slot backend — cannot rely on slot KV state
		}
	}
	b.PoolMu.Unlock()

	nPrev := len(prevBlocks)
	nReq := len(reqBlocks)
	if nPrev > nReq {
		logWarn("slot_manager", "Cannot skip restore for backend '%s' slot %d: prev_blocks (%d) longer than req_blocks (%d)", b.BackendID, slotID, nPrev, nReq)
		return false
	}
	diff := nReq - nPrev
	maxLen := nPrev
	if nReq > maxLen {
		maxLen = nReq
	}
	diffPct := 0.0
	if maxLen > 0 {
		diffPct = float64(diff) / float64(maxLen)
	}
	if diffPct > KVCacheSkipMaxBlockDiff {
		logWarn("slot_manager", "Cannot skip restore for backend '%s' slot %d: block diff %d/%d (%.1f%%) > %.0f%% threshold",
			b.BackendID, slotID, diff, maxLen, diffPct*100, KVCacheSkipMaxBlockDiff*100)
		return false
	}
	lcp := LCPBlocks(reqBlocks, prevBlocks)
	denom := nReq
	if nPrev < denom {
		denom = nPrev
	}
	if denom < 1 {
		denom = 1
	}
	ratio := float64(lcp) / float64(denom)
	logWarn("slot_manager", "Checking skip restore for backend '%s' slot %d: ratio=%.3f, blocks %d->%d",
		b.BackendID, slotID, ratio, nPrev, nReq)
	return ratio >= KVCacheSkipThreshold
}

// MarkSaveSkipped records a skipped save for the slot (6-tuple or legacy 3-tuple).
func (b *BackendSlotManager) MarkSaveSkipped(slotID int, entry *SaveSkipEntry) {
	b.PoolMu.Lock()
	defer b.PoolMu.Unlock()
	b.slotSaveSkipped[slotID] = entry
}

// FlushSaveSkipped returns and clears the skipped-save entry for the slot.
func (b *BackendSlotManager) FlushSaveSkipped(slotID int) *SaveSkipEntry {
	b.PoolMu.Lock()
	defer b.PoolMu.Unlock()
	entry := b.slotSaveSkipped[slotID]
	delete(b.slotSaveSkipped, slotID)
	return entry
}

// Restore calls the backend to restore a KV cache into the slot.
func (b *BackendSlotManager) Restore(slotID int, key, modelName string, touchRing bool) bool {
	client := backendManager.GetClient(b.BackendID)
	restored := client.RestoreSlot(context.Background(), slotID, key, modelName)
	logInfo("slot_manager", "Restore for model '%s' on backend '%s' slot %d (key %s): ok=%v",
		modelName, b.BackendID, slotID, truncateKey(key), restored)
	if restored && touchRing {
		b.TouchRing(key)
	}
	return restored
}

// SaveAfter saves the slot's KV cache to disk after a completed response.
// Returns (ok, size).
func (b *BackendSlotManager) SaveAfter(modelName string, slotID int, key string, blocks []string, nTokens int) (bool, int) {
	if !backendManager.CacheEnabled(b.BackendID) {
		return false, 0
	}
	client := backendManager.GetClient(b.BackendID)
	ok, size, err := client.SaveSlot(context.Background(), slotID, key, modelName)
	if err != nil {
		// Python save_slot swallows all exceptions and returns (False, 0).
		ok, size = false, 0
	}
	logInfo("slot_manager", "Save cache for model '%s' on backend '%s' slot %d: ok=%v, size=%d",
		modelName, b.BackendID, slotID, ok, size)

	if len(blocks) > 0 {
		b.SetKVState(slotID, blocks)
		logWarn("slot_manager", "Updated KV cache state for model '%s' on backend '%s' slot %d: %d blocks",
			modelName, b.BackendID, slotID, len(blocks))
	}

	if ok && size > 0 {
		b.RingMu.Lock()
		kvMeta.WriteMeta(key, nTokens, blocks, WordsPerBlock, modelName, b.BackendID, size)
		b.CacheRing = append(b.CacheRing, ringEntry{Key: key, Size: int64(size), TS: NowFloat()})
		b.totalBytes += int64(size)
		b.EvictIfNeeded()
		b.RingMu.Unlock()
	}
	return ok, size
}

// EvictIfNeeded enforces the backend's cache_max_size_gb budget.
// Must be called with RingMu held.
//
// Scoring: score = age_seconds * (2 - uniqueness), where uniqueness = 1 -
// max LCP ratio against all other ring entries. Higher score = evict first.
func (b *BackendSlotManager) EvictIfNeeded() {
	maxBytes := float64(backendManager.GetCacheMaxSizeGB(b.BackendID)) * 1024 * 1024 * 1024
	if float64(b.totalBytes) <= maxBytes {
		return
	}
	now := NowFloat()

	// Pass 1: drop ring entries whose meta/blocks are gone (orphaned).
	blocksMap := map[string][]string{}
	validRing := make([]ringEntry, 0, len(b.CacheRing))
	for _, entry := range b.CacheRing {
		blocks := kvMeta.GetBlocks(entry.Key, b.BackendID)
		if len(blocks) > 0 {
			blocksMap[entry.Key] = blocks
			validRing = append(validRing, entry)
		} else {
			b.totalBytes -= entry.Size
			logInfo("slot_manager", "Evicting orphaned ring entry '%s' for backend '%s' (%d bytes)",
				truncateKey(entry.Key), b.BackendID, entry.Size)
			b.deleteEntry(entry.Key, "ring_evict_orphan", fmt.Sprintf("(%d bytes)", entry.Size))
		}
	}
	b.CacheRing = validRing

	if len(b.CacheRing) == 0 || float64(b.totalBytes) <= maxBytes {
		return
	}

	type scored struct {
		key   string
		score float64
	}
	ringList := make([]ringEntry, len(b.CacheRing))
	copy(ringList, b.CacheRing)
	scoredList := make([]scored, 0, len(ringList))
	for i, entry := range ringList {
		stale := now - entry.TS
		blocks := blocksMap[entry.Key]
		if len(blocks) == 0 {
			// Dead code path (orphans were removed above), ported for fidelity.
			scoredList = append(scoredList, scored{entry.Key, stale * 2.0})
			continue
		}
		maxLCPRatio := 0.0
		for j, other := range ringList {
			if i == j {
				continue
			}
			otherBlocks := blocksMap[other.Key]
			if len(otherBlocks) == 0 {
				continue
			}
			lcp := LCPBlocks(blocks, otherBlocks)
			denom := float64(len(blocks))
			if denom < 1 {
				denom = 1
			}
			ratio := float64(lcp) / denom
			if ratio > maxLCPRatio {
				maxLCPRatio = ratio
			}
		}
		uniqueness := 1.0 - maxLCPRatio
		scoredList = append(scoredList, scored{entry.Key, stale * (2.0 - uniqueness)})
	}
	sort.SliceStable(scoredList, func(i, j int) bool {
		return scoredList[i].score > scoredList[j].score
	})

	keysToEvict := map[string]bool{}
	projected := b.totalBytes
	for _, s := range scoredList {
		if float64(projected) <= maxBytes {
			break
		}
		keysToEvict[s.key] = true
		for _, entry := range ringList {
			if entry.Key == s.key {
				projected -= entry.Size
				break
			}
		}
	}

	evictedAny := false
	remaining := make([]ringEntry, 0, len(b.CacheRing))
	for _, entry := range b.CacheRing {
		if keysToEvict[entry.Key] {
			b.totalBytes -= entry.Size
			staleHours := (now - entry.TS) / 3600
			scoreVal := 0.0
			for _, s := range scoredList {
				if s.key == entry.Key {
					scoreVal = s.score
					break
				}
			}
			logInfo("slot_manager", "Ring buffer eviction: evicted '%s' for backend '%s' (%d bytes, score=%.0f, age=%.1fh, remaining=%d)",
				truncateKey(entry.Key), b.BackendID, entry.Size, scoreVal, staleHours, b.totalBytes)
			b.deleteEntry(entry.Key, "ring_evict_scored", fmt.Sprintf("(score=%.0f, age=%.1fh)", scoreVal, staleHours))
			evictedAny = true
		} else {
			remaining = append(remaining, entry)
		}
	}
	b.CacheRing = remaining

	if evictedAny {
		logInfo("slot_manager", "Cache ring check for backend '%s': total=%d bytes, max=%d bytes, ring_size=%d",
			b.BackendID, b.totalBytes, int64(maxBytes), len(b.CacheRing))
	}
}

// deleteEntry removes a cache file (and its sidecars) plus its meta file.
func (b *BackendSlotManager) deleteEntry(key, logMsg, logExtra string) {
	ok := backendManager.CacheDelete(b.BackendID, key)
	if ok {
		logInfo("slot_manager", "%s: %s %s", logMsg, truncateKey(key), logExtra)
	} else {
		logWarn("slot_manager", "%s_agent_fail: %s", logMsg, truncateKey(key))
	}
	kvMeta.DeleteMetaFile(key)
}

// InitFromDisk rebuilds the in-memory ring buffer from disk.
func (b *BackendSlotManager) InitFromDisk() {
	backendDir := SanitizeBackendDir(b.BackendID)
	if st, err := os.Stat(MetaDir); err != nil || !st.IsDir() {
		return
	}
	metaPath := filepath.Join(MetaDir, backendDir)
	if st, err := os.Stat(metaPath); err != nil || !st.IsDir() {
		return
	}
	b.RingMu.Lock()
	defer b.RingMu.Unlock()
	for _, key := range kvMeta.ListKeys(backendDir) {
		cacheSize := kvMeta.GetCacheSize(backendDir, key)
		if cacheSize == 0 {
			continue
		}
		lastUsed := kvMeta.GetLastUsedTime(key, backendDir)
		b.CacheRing = append(b.CacheRing, ringEntry{Key: key, Size: int64(cacheSize), TS: lastUsed})
		b.totalBytes += int64(cacheSize)
	}
	b.EvictIfNeeded()
}

// TouchRing updates the timestamp of an existing ring entry.
func (b *BackendSlotManager) TouchRing(key string) {
	b.RingMu.Lock()
	defer b.RingMu.Unlock()
	for i := range b.CacheRing {
		if b.CacheRing[i].Key == key {
			b.CacheRing[i].TS = NowFloat()
			return
		}
	}
}

// GetRingSize returns the number of entries in the cache ring.
func (b *BackendSlotManager) GetRingSize() int {
	b.RingMu.Lock()
	defer b.RingMu.Unlock()
	return len(b.CacheRing)
}

// GetTotalBytes returns the total bytes tracked in the cache ring.
func (b *BackendSlotManager) GetTotalBytes() int64 {
	b.RingMu.Lock()
	defer b.RingMu.Unlock()
	return b.totalBytes
}

// CountInUse returns the number of in-use slots in a model's pool.
func (b *BackendSlotManager) CountInUse(modelName string) int {
	b.PoolMu.Lock()
	defer b.PoolMu.Unlock()
	pool, ok := b.slotPools[modelName]
	if !ok {
		return 0
	}
	n := 0
	for _, s := range pool {
		if b.inUse[s] {
			n++
		}
	}
	return n
}

// RingOldestNewest returns the oldest and newest ring-entry timestamps
// (insertion order, not min/max — matching the Python cache_ring[0] /
// cache_ring[-1] reads). ok is false for an empty ring.
func (b *BackendSlotManager) RingOldestNewest() (float64, float64, bool) {
	b.RingMu.Lock()
	defer b.RingMu.Unlock()
	if len(b.CacheRing) == 0 {
		return 0, 0, false
	}
	return b.CacheRing[0].TS, b.CacheRing[len(b.CacheRing)-1].TS, true
}

// RingOldestNewestKeys returns the oldest and newest ring-entry keys
// (insertion order, not min/max — matching the Python cache_ring[0] /
// cache_ring[-1] reads). ok is false for an empty ring.
func (b *BackendSlotManager) RingOldestNewestKeys() (string, string, bool) {
	b.RingMu.Lock()
	defer b.RingMu.Unlock()
	if len(b.CacheRing) == 0 {
		return "", "", false
	}
	return b.CacheRing[0].Key, b.CacheRing[len(b.CacheRing)-1].Key, true
}

// SlotStatus returns a snapshot of this backend's slot state shaped as
// {"models": {modelName: {"slots": {slotID: {in_use, last_used, kv_blocks,
// last_restore}}}}}. Model names are sorted for deterministic output.
func (b *BackendSlotManager) SlotStatus() map[string]any {
	b.PoolMu.Lock()
	defer b.PoolMu.Unlock()
	names := make([]string, 0, len(b.slotPools))
	for name := range b.slotPools {
		names = append(names, name)
	}
	sort.Strings(names)
	models := map[string]any{}
	for _, name := range names {
		pool := b.slotPools[name]
		slots := map[string]any{}
		for _, slotID := range pool {
			lastRestore := any(nil)
			if entry, ok := b.slotSaveSkipped[slotID]; ok && entry != nil && entry.Key != "" {
				lastRestore = truncateKey(entry.Key)
			}
			slots[strconv.Itoa(slotID)] = map[string]any{
				"in_use":       b.inUse[slotID],
				"last_used":    b.lastUsed[slotID],
				"kv_blocks":    len(b.slotKVState[slotID]),
				"last_restore": lastRestore,
			}
		}
		models[name] = map[string]any{"slots": slots}
	}
	return map[string]any{"models": models}
}

// truncateKey returns the first 16 chars of a cache key for log lines
// (Python key[:16]).
func truncateKey(key string) string {
	if len(key) <= 16 {
		return key
	}
	return key[:16]
}

// BackendSlotKey is a comparable (backend_id, slot_id) pair.
type BackendSlotKey struct {
	BackendID string
	SlotID    int
}

// SlotManager — global coordinator: per-backend managers + cross-backend state.
type SlotManager struct {
	Mu sync.Mutex
	// backends and cacheWaitPending are guarded by Mu.
	backends         map[string]*BackendSlotManager
	backendOrder     []string // first-seen insertion order (dict-order parity)
	cacheWaitPending map[string]int
}

func NewSlotManager() *SlotManager {
	return &SlotManager{
		backends:         map[string]*BackendSlotManager{},
		backendOrder:     []string{},
		cacheWaitPending: map[string]int{},
	}
}

// Get returns (creating if needed) the BackendSlotManager for a backend.
func (s *SlotManager) Get(backendID string) *BackendSlotManager {
	s.Mu.Lock()
	defer s.Mu.Unlock()
	if sm, ok := s.backends[backendID]; ok {
		return sm
	}
	sm := NewBackendSlotManager(backendID)
	s.backends[backendID] = sm
	s.backendOrder = append(s.backendOrder, backendID)
	return sm
}

// HasBackend reports whether a per-backend manager exists.
func (s *SlotManager) HasBackend(backendID string) bool {
	s.Mu.Lock()
	defer s.Mu.Unlock()
	_, ok := s.backends[backendID]
	return ok
}

// Backends returns the backend IDs in first-seen insertion order.
func (s *SlotManager) Backends() []string {
	s.Mu.Lock()
	defer s.Mu.Unlock()
	out := make([]string, len(s.backendOrder))
	copy(out, s.backendOrder)
	return out
}

// AllKVStates returns {(backend_id, slot_id): blocks} across all backends.
func (s *SlotManager) AllKVStates() map[BackendSlotKey][]string {
	s.Mu.Lock()
	defer s.Mu.Unlock()
	out := map[BackendSlotKey][]string{}
	for beID, sm := range s.backends {
		states := sm.GetKVStates()
		for slotID, blocks := range states {
			out[BackendSlotKey{beID, slotID}] = blocks
		}
	}
	return out
}

// InitFromDisk rebuilds ring buffers for all registered backends. Mirrors the
// Python SlotManager.init_from_disk, which iterates backend_manager.keys()
// (NOT the lazily-created per-backend managers, which are empty at startup).
func (s *SlotManager) InitFromDisk() {
	totalEntries := 0
	var totalBytes int64
	seen := make([]*BackendSlotManager, 0)
	for _, key := range backendManager.Keys() {
		sm := s.Get(key)
		sm.InitFromDisk()
		seen = append(seen, sm)
	}
	perBackend := make([]string, 0, len(seen))
	for _, sm := range seen {
		count := sm.GetRingSize()
		bytes := sm.GetTotalBytes()
		totalEntries += count
		totalBytes += bytes
		perBackend = append(perBackend, fmt.Sprintf("%s: %.1f GB (%d files)",
			sm.BackendID, float64(bytes)/(1024*1024*1024), count))
	}
	logInfo("slot_manager", "Loaded %d cache entries from disk (%.1f GB), per-backend: %s",
		totalEntries, float64(totalBytes)/(1024*1024*1024), strings.Join(perBackend, "; "))
}

// RefreshSlotCounts queries slot counts for each discovered model+backend pair
// and updates the pools.
func (s *SlotManager) RefreshSlotCounts() {
	slotCounts, err := backendManager.RefreshSlotCounts()
	if err != nil {
		logWarn("slot_manager", "Failed to refresh slot counts: %s — proceeding with existing pool state", err)
		slotCounts = nil
	}
	for backendID, byModel := range slotCounts {
		beSm := s.Get(backendID)
		for modelName, nSlots := range byModel {
			beSm.EnsurePool(modelName, nSlots)
		}
	}
}

// GetCacheWaitPending returns the current pending-waiter count for a backend.
func (s *SlotManager) GetCacheWaitPending(backendID string) int {
	s.Mu.Lock()
	defer s.Mu.Unlock()
	return s.cacheWaitPending[backendID]
}

// SetCacheWaitPending sets the pending-waiter count for a backend.
func (s *SlotManager) SetCacheWaitPending(backendID string, n int) {
	s.Mu.Lock()
	defer s.Mu.Unlock()
	s.cacheWaitPending[backendID] = n
}

// slotManager is the global instance (created in app startup).
var slotManager = NewSlotManager()

// GetSlotManager returns the global slot manager instance.
func GetSlotManager() *SlotManager { return slotManager }
