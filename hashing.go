// hashing.go — cache key hashing: SHA256 of model name + token IDs,
// with word-block LCP matching over full SHA256 block hashes.

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"
)

const MetaSuffix = ".meta.json"

// SanitizeBackendDir converts a backend key to a safe filesystem directory name
// (colons replaced with dashes: "10.0.0.1:8000" -> "10.0.0.1-8000").
func SanitizeBackendDir(backendKey string) string {
	return strings.ReplaceAll(backendKey, ":", "-")
}

// BlockHashesFromTokens splits tokens into blocks of wpb and hashes each block
// (sha256 of comma-joined token IDs). Logs a warning per call, matching the
// Python version's behavior.
func BlockHashesFromTokens(tokenIDs []int, wpb int) []string {
	hashes := make([]string, 0, (len(tokenIDs)+wpb-1)/wpb)
	for i := 0; i < len(tokenIDs); i += wpb {
		end := i + wpb
		if end > len(tokenIDs) {
			end = len(tokenIDs)
		}
		parts := make([]string, end-i)
		for j, t := range tokenIDs[i:end] {
			parts[j] = strconv.Itoa(t)
		}
		h := sha256.Sum256([]byte(strings.Join(parts, ",")))
		hashes = append(hashes, hex.EncodeToString(h[:]))
	}
	logWarn("hashing", "Block hashes: %d blocks, %d tokens per block", len(hashes), wpb)
	return hashes
}

// BlockHashesFromTokensDefault uses the configured WORDS_PER_BLOCK.
func BlockHashesFromTokensDefault(tokenIDs []int) []string {
	return BlockHashesFromTokens(tokenIDs, WordsPerBlock)
}

// LCPBlocks returns the number of leading blocks shared by two block-hash lists.
func LCPBlocks(blocks1, blocks2 []string) int {
	n := len(blocks1)
	if len(blocks2) < n {
		n = len(blocks2)
	}
	i := 0
	for i < n && blocks1[i] == blocks2[i] {
		i++
	}
	return i
}

// PrefixKeySha256 is a basic SHA256 wrapper over UTF-8 text.
func PrefixKeySha256(text string) string {
	h := sha256.Sum256([]byte(text))
	return hex.EncodeToString(h[:])
}

// MetaKey = sha256(canonical_name + '\n' + ','.join(token_ids)).
func MetaKey(canonicalName string, tokenIDs []int) string {
	parts := make([]string, len(tokenIDs))
	for i, t := range tokenIDs {
		parts[i] = strconv.Itoa(t)
	}
	return PrefixKeySha256(canonicalName + "\n" + strings.Join(parts, ","))
}
