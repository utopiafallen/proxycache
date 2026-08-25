// config.go — all configuration from environment variables (no .env file),
// plus the request classifier (summarization vs conversation) and save heuristics.

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode/utf8"
)

// --- Backends ---

var Backends []map[string]any

func initBackends() {
	raw := os.Getenv("BACKENDS")
	if raw == "" {
		raw = "[]"
	}
	var v []map[string]any
	if err := json.Unmarshal([]byte(raw), &v); err != nil {
		v = nil
	}
	if len(v) == 0 {
		v = []map[string]any{
			{"url": "http://127.0.0.1:8000", "cache_dir": "/tmp/llama-cache"},
		}
	}
	Backends = v
}

// --- Numeric / general config ---

var (
	WordsPerBlock            = envInt("WORDS_PER_BLOCK", 100)
	LCPTh                    = envFloat("LCP_TH", 0.2)
	MetaDir                  = filepath.Join(mustGetwd(), envStr("META_DIR", "kv_meta"))
	RequestTimeout           = envFloat("REQUEST_TIMEOUT", 600)
	ModelID                  = envStr("MODEL_ID", "llama.cpp")
	BackendMode              = envStr("BACKEND_MODE", "llama-cpp")
	Port                     = envInt("PORT", 8081)
	DefaultNCtx              = envInt("DEFAULT_N_CTX", 16384)
	KVCacheSkipThreshold     = envFloat("KV_CACHE_SKIP_THRESHOLD", 0.9)
	KVCacheSkipMaxBlockDiff  = envFloat("KV_CACHE_SKIP_MAX_BLOCK_DIFF_PCT", 0.1)
	CacheSaveRatioThreshold  = envFloat("CACHE_SAVE_RATIO_THRESHOLD", 0.8)
	CacheSaveCtxThreshold    = envFloat("CACHE_SAVE_CTX_THRESHOLD", 0.7)
	SlotTimeout              = envFloat("SLOT_TIMEOUT", 30)
	ClientRecreateInterval   = envInt("CLIENT_RECREATE_INTERVAL", 50)
	CacheHitWaitEMAMinT      = envFloat("CACHE_HIT_WAIT_EMA_MIN_TIMEOUT", 10)
	CacheHitWaitMaxPending   = envInt("CACHE_HIT_WAIT_MAX_PENDING_REQS", 3)
	CacheHitWaitEMAAlpha     = envFloat("CACHE_HIT_WAIT_EMA_ALPHA", 0.2)
	CacheHitWaitEMAInitialT  = envFloat("CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT", 30)
	CacheHitWaitEMAMaxT      = envFloat("CACHE_HIT_WAIT_EMA_MAX_TIMEOUT", 300)
	MetricsRetention         = envInt("METRICS_RETENTION", 200)
	DashboardEnabled         = envBool("DASHBOARD_ENABLED", true)
	LogLevel                 = os.Getenv("LOG_LEVEL")
)

func init() {
	initBackends()
	os.MkdirAll(MetaDir, 0o755)
}

func mustGetwd() string {
	wd, err := os.Getwd()
	if err != nil {
		return "."
	}
	return wd
}

func envInt(key string, def int) int {
	if s := os.Getenv(key); s != "" {
		var v int
		if _, err := fmt.Sscanf(s, "%d", &v); err == nil {
			return v
		}
	}
	return def
}

func envFloat(key string, def float64) float64 {
	if s := os.Getenv(key); s != "" {
		var v float64
		if _, err := fmt.Sscanf(s, "%f", &v); err == nil {
			return v
		}
	}
	return def
}

func envStr(key, def string) string {
	if s := os.Getenv(key); s != "" {
		return s
	}
	return def
}

func envBool(key string, def bool) bool {
	s := os.Getenv(key)
	if s == "" {
		return def
	}
	switch strings.ToLower(s) {
	case "true", "1", "yes":
		return true
	default:
		return false
	}
}

// --- Save heuristics ---

// ShouldSkipSaveHeuristic returns true when the request is unlikely to produce
// a reusable cache entry (near-max-context or classified as summarization).
func ShouldSkipSaveHeuristic(promptTokens int, nCtx int, messages []map[string]any, requestJSON map[string]any) bool {
	if nCtx > 0 && float64(promptTokens)/float64(nCtx) >= CacheSaveCtxThreshold {
		return true
	}
	if messages != nil {
		reqClass := ClassifyRequest(messages, requestJSON)
		if reqClass.Score >= 0.4 {
			return true
		}
	}
	return false
}

// ShouldSaveCache returns true when the slot's cache should be saved to disk:
// recompute happened (restore was useless) or the restore candidate ratio is
// below the threshold (new entry will be more useful).
func ShouldSaveCache(bestRatio float64, recomputeHappened bool) bool {
	return recomputeHappened || bestRatio <= CacheSaveRatioThreshold
}

// --- Request classification ---

type RequestClass struct {
	Score   float64
	Signals []string
}

var (
	summarizationPatterns = []struct {
		re     *regexp.Regexp
		weight float64
		label  string
	}{
		// Strong keywords (0.30)
		{regexp.MustCompile(`\bsummar[i][sz](e([dsw]|es?)?|[ed]|ing|e?ing|ation[es]?)?\b`), 0.30, "summarize"},
		{regexp.MustCompile(`\btl;?\s?dr\b`), 0.30, "tl;dr"},
		{regexp.MustCompile(`\bcondens(e|ed|ing|es|ation)\b`), 0.30, "condense"},
		{regexp.MustCompile(`\bgive\s+me\s+a\s+summar[i][sz](e([dsw]|es?)?|[ed]|ing|e?ing|ation[es]?)?\b`), 0.30, "give me a summary"},
		{regexp.MustCompile(`\bsum\s+it\s+up\b`), 0.30, "sum it up"},
		{regexp.MustCompile(`\bwrap\s+up\b`), 0.30, "wrap up"},
		{regexp.MustCompile(`\brecap\b`), 0.30, "recap"},
		{regexp.MustCompile(`\bin\s+short\b`), 0.30, "in short"},
		// Weak keywords (0.15)
		{regexp.MustCompile(`\bsummar(ies|y)\b`), 0.15, "summary"},
		{regexp.MustCompile(`\bkey\s+points?\b`), 0.15, "key points"},
		{regexp.MustCompile(`\bbullet\s+points?\b`), 0.15, "bullet points"},
		{regexp.MustCompile(`\boverview\b`), 0.15, "overview"},
		{regexp.MustCompile(`\babstract\b`), 0.15, "abstract"},
		{regexp.MustCompile(`\bdigest\b`), 0.15, "digest"},
		{regexp.MustCompile(`\bextract\b`), 0.15, "extract"},
		{regexp.MustCompile(`\bhighlights?\b`), 0.15, "highlights"},
		{regexp.MustCompile(`\bmain\s+points?\b`), 0.15, "main points"},
		{regexp.MustCompile(`\bbrief\b`), 0.15, "brief"},
	}
	pastePatterns = []string{
		"<document>", "</document>", "<context>", "</context>",
		"<article>", "</article>", "<text>", "</text>",
		"<content>", "</content>", "<passage>", "</passage>",
		"<input>", "</input>", "<transcript>", "</transcript>",
	}
	contentIntroPhrases = []string{
		"here is a", "here's a", "here is the", "here's the",
		"below is", "below is a", "below is the",
		"attached is", "following is", "following text",
		"read this", "analyze this", "review this",
		"the following", "consider the following",
	}

	reStripDelimited   = regexp.MustCompile(`(?s)<[^>]*>.*?</[^>]*>`)
	reColonNewline     = regexp.MustCompile(`:\s*\n`)
)

func getMsgText(msg map[string]any) string {
	c, ok := msg["content"]
	if !ok {
		return ""
	}
	if s, ok := c.(string); ok {
		return strings.ToLower(s)
	}
	if list, ok := c.([]any); ok {
		for _, p := range list {
			pm, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if pm["type"] == "text" {
				if t, ok := pm["text"].(string); ok {
					return strings.ToLower(t)
				}
				return ""
			}
		}
	}
	return ""
}

func stripDelimited(text string) string {
	return reStripDelimited.ReplaceAllString(text, " ")
}

func extractInstruction(text string) string {
	if idx := reColonNewline.FindIndex([]byte(text)); idx != nil {
		text = text[:idx[0]]
	}
	if len(text) > 100 {
		text = text[:100]
	}
	return text
}

// ClassifyRequest scores a chat request as summarization-like or conversation-like.
// Score 0..1 (higher = more likely summarization); threshold 0.4.
func ClassifyRequest(messages []map[string]any, requestJSON map[string]any) RequestClass {
	if len(messages) == 0 {
		return RequestClass{Score: 0.0, Signals: []string{}}
	}

	signals := []string{}
	score := 0.0

	nUser, nAssistant, nSystem := 0, 0, 0
	for _, m := range messages {
		if r, ok := m["role"].(string); ok {
			switch r {
			case "user":
				nUser++
			case "assistant":
				nAssistant++
			case "system":
				nSystem++
			}
		}
	}
	nTotal := len(messages)

	// 1. Single user message, no assistant (0.10)
	if nUser == 1 && nAssistant == 0 {
		score += 0.10
		signals = append(signals, "single_user_msg")
	}

	// 2. Summarization keyword in last user message instruction
	var lastUser map[string]any
	for i := len(messages) - 1; i >= 0; i-- {
		if r, ok := messages[i]["role"].(string); ok && r == "user" {
			lastUser = messages[i]
			break
		}
	}
	if lastUser != nil {
		text := getMsgText(lastUser)
		instruction := extractInstruction(stripDelimited(text))
		for _, p := range summarizationPatterns {
			if p.re.MatchString(instruction) {
				score += p.weight
				signals = append(signals, "keyword:"+p.label)
				break
			}
		}
	}

	// 3. One message dominates >70% of text (0.15) — only for 2+ messages
	var totalLength int
	if nTotal >= 2 {
		totalLength = 0
		maxLen := 0
		for _, m := range messages {
			l := utf8.RuneCountInString(getMsgText(m))
			totalLength += l
			if l > maxLen {
				maxLen = l
			}
		}
		if totalLength > 0 {
			maxRatio := float64(maxLen) / float64(totalLength)
			if maxRatio > 0.7 {
				score += 0.15
				signals = append(signals, fmt.Sprintf("dominant_msg:%.0f%%", maxRatio*100))
			}
		}
	}

	// 4. Prompt text > 8192 chars (0.10)
	if totalLength == 0 {
		for _, m := range messages {
			totalLength += utf8.RuneCountInString(getMsgText(m))
		}
	}
	if totalLength > 8192 {
		score += 0.10
		signals = append(signals, fmt.Sprintf("long_prompt:%dtok_est", totalLength/4))
	}

	// 5. Structural delimiters / paste markers in last user message (0.05)
	lastUserText := ""
	if lastUser != nil {
		lastUserText = getMsgText(lastUser)
	}
	for _, pat := range pastePatterns {
		if strings.Contains(lastUserText, pat) {
			score += 0.05
			signals = append(signals, "delimiter:"+pat)
			break
		}
	}

	// 6. Content introduction phrases in last user instruction (0.05)
	lastUserInstruction := stripDelimited(lastUserText)
	if len(lastUserInstruction) > 500 {
		lastUserInstruction = lastUserInstruction[:500]
	}
	for _, phrase := range contentIntroPhrases {
		if strings.Contains(lastUserInstruction, phrase) {
			score += 0.05
			signals = append(signals, "intro:"+phrase)
			break
		}
	}

	// 7. System prompt present (-0.10)
	if nSystem > 0 {
		score -= 0.10
		signals = append(signals, "system_prompt")
	}

	return RequestClass{
		Score:   roundTo3(clamp01(score)),
		Signals: signals,
	}
}

func clamp01(v float64) float64 {
	if v < 0.0 {
		return 0.0
	}
	if v > 1.0 {
		return 1.0
	}
	return v
}

func roundTo3(v float64) float64 {
	return math.Round(v*1000) / 1000
}
