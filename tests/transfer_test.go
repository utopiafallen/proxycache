package tests

import (
	"context"
	"net"
	"os"
	"path/filepath"
	"proxycache"
	"testing"
	"time"
)

// startAgentServer runs an embedded AgentServer on an ephemeral port and
// returns its base URL and raw listen address (host:port).
func startAgentServer(t *testing.T, dir string) (string, string) {
	t.Helper()
	srv := proxycache.NewAgentServer(dir, 0)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go srv.Start(ctx)
	var addr string
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if addr = srv.Addr(); addr != "" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if addr == "" {
		t.Fatal("agent server did not bind in time")
	}
	return "http://" + addr, addr
}

func TestAgentP2PTransferPush(t *testing.T) {
	srcDir, dstDir := t.TempDir(), t.TempDir()
	srcURL, _ := startAgentServer(t, srcDir)
	dstURL, _ := startAgentServer(t, dstDir)
	key := "pushkey"
	if err := os.WriteFile(filepath.Join(srcDir, key), []byte("payload-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	client := proxycache.NewCacheAgentClient(srcURL)
	defer client.Close()
	if !client.Transfer(key, dstURL, 0) {
		t.Fatal("Transfer returned false")
	}
	data, err := os.ReadFile(filepath.Join(dstDir, key))
	if err != nil {
		t.Fatalf("transferred file missing on target: %v", err)
	}
	if string(data) != "payload-bytes" {
		t.Errorf("transferred content = %q, want payload-bytes", data)
	}
}

func TestAgentMakeSpaceEvictsOldest(t *testing.T) {
	dstDir := t.TempDir()
	names := []string{"old1", "old2", "old3"}
	for i, name := range names {
		p := filepath.Join(dstDir, name)
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		mt := time.Now().Add(-time.Duration(3-i) * time.Hour)
		if err := os.Chtimes(p, mt, mt); err != nil {
			t.Fatal(err)
		}
	}
	dstURL, _ := startAgentServer(t, dstDir)
	client := proxycache.NewCacheAgentClient(dstURL)
	defer client.Close()

	// Budget 10 bytes, need 9: must evict down to a single 1-byte file.
	ok, used := client.MakeSpace(9, 10)
	if !ok {
		t.Fatalf("MakeSpace ok=false")
	}
	if used != 1 {
		t.Errorf("used after make-space = %d, want 1", used)
	}
	for _, name := range []string{"old1", "old2"} {
		if _, err := os.Stat(filepath.Join(dstDir, name)); !os.IsNotExist(err) {
			t.Errorf("oldest file %s should be evicted", name)
		}
	}
	if _, err := os.Stat(filepath.Join(dstDir, "old3")); err != nil {
		t.Errorf("newest file old3 should survive: %v", err)
	}

	// A transfer now fits without further eviction.
	srcDir := t.TempDir()
	srcURL, _ := startAgentServer(t, srcDir)
	key := "newkey"
	if err := os.WriteFile(filepath.Join(srcDir, key), []byte("yyyyyyyy"), 0o644); err != nil {
		t.Fatal(err)
	}
	sclient := proxycache.NewCacheAgentClient(srcURL)
	defer sclient.Close()
	if !sclient.Transfer(key, dstURL, 10) {
		t.Fatal("Transfer into constrained target failed")
	}
	if _, err := os.Stat(filepath.Join(dstDir, key)); err != nil {
		t.Errorf("transferred key missing on target: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dstDir, "old3")); err != nil {
		t.Errorf("old3 should still be present after fit transfer: %v", err)
	}
}

func TestAgentTransferMissingSource(t *testing.T) {
	srcURL, _ := startAgentServer(t, t.TempDir())
	dstURL, _ := startAgentServer(t, t.TempDir())
	client := proxycache.NewCacheAgentClient(srcURL)
	defer client.Close()
	if client.Transfer("no-such-key", dstURL, 0) {
		t.Error("Transfer of missing file should return false")
	}
	if _, err := client.FetchFile("no-such-key"); err == nil {
		t.Error("FetchFile of missing file should error")
	}
}

// TestProxyTransferLocalToLocal exercises the proxy-side orchestration for
// two local cache_dir backends: file copy, meta registration, ring entry.
func TestProxyTransferLocalToLocal(t *testing.T) {
	withTempMetaDir(t)
	dir1, dir2 := t.TempDir(), t.TempDir()
	m1 := newMockLlama(t, "xfer-model", 32768, seq(10), 1)
	m2 := newMockLlama(t, "xfer-model", 32768, seq(10), 1)
	be1 := backendKeyFromURL(m1.srv.URL)
	be2 := backendKeyFromURL(m2.srv.URL)
	withTestBackend(t, []map[string]any{
		{"url": m1.srv.URL, "cache_dir": dir1},
		{"url": m2.srv.URL, "cache_dir": dir2},
	})
	markBackendsUp(proxycache.GetBackendManager())
	tokens := seq(10)
	key := proxycache.MetaKey("xfer-model", tokens)
	blocks := proxycache.BlockHashesFromTokens(tokens, proxycache.WordsPerBlock)
	proxycache.GetKVMeta().WriteMeta(key, 10, blocks, proxycache.WordsPerBlock, "xfer-model", be1, 42)
	if err := os.WriteFile(filepath.Join(dir1, key), []byte("src-cache"), 0o644); err != nil {
		t.Fatal(err)
	}

	if !proxycache.RequestCacheTransfer(be1, be2, key) {
		t.Fatal("RequestCacheTransfer returned false")
	}
	waitFor(t, 5*time.Second, func() bool {
		_, err := os.Stat(filepath.Join(dir2, key))
		return err == nil
	})
	data, err := os.ReadFile(filepath.Join(dir2, key))
	if err != nil || string(data) != "src-cache" {
		t.Fatalf("transferred file content = %q (err=%v), want src-cache", data, err)
	}
	if meta := proxycache.GetKVMeta().ReadMeta(key, be2); meta == nil {
		t.Error("no meta registered on target backend")
	}
	if got := proxycache.GetSlotManager().Get(be2).GetRingSize(); got != 1 {
		t.Errorf("target ring size = %d, want 1", got)
	}
	// Second request for the same (key, dst) is a no-op: file already there.
	if proxycache.RequestCacheTransfer(be1, be2, key) {
		t.Error("duplicate transfer should be skipped")
	}
}

// TestProxyTransferLocalToAgent covers the proxy orchestration path where the
// target backend manages its cache through a remote cache-agent.
func TestProxyTransferLocalToAgent(t *testing.T) {
	withTempMetaDir(t)
	dir1 := t.TempDir()
	agentDir := t.TempDir()
	m1 := newMockLlama(t, "xfer2-model", 32768, seq(10), 1)
	agentURL, agentAddr := startAgentServer(t, agentDir)
	_, agentPort, err := net.SplitHostPort(agentAddr)
	if err != nil {
		t.Fatalf("bad agent addr %q: %v", agentAddr, err)
	}
	_ = agentURL
	be1 := backendKeyFromURL(m1.srv.URL)
	withTestBackend(t, []map[string]any{
		{"url": m1.srv.URL, "cache_dir": dir1},
		{"url": "http://127.0.0.1:1", "agent_port": agentPort, "cache_max_size_gb": 0.000001},
	})
	markBackendsUp(proxycache.GetBackendManager())
	be2 := backendKeyFromURL("http://127.0.0.1:1")
	tokens := seq(10)
	key := proxycache.MetaKey("xfer2-model", tokens)
	blocks := proxycache.BlockHashesFromTokens(tokens, proxycache.WordsPerBlock)
	proxycache.GetKVMeta().WriteMeta(key, 10, blocks, proxycache.WordsPerBlock, "xfer2-model", be1, 42)
	if err := os.WriteFile(filepath.Join(dir1, key), []byte("src-cache-2"), 0o644); err != nil {
		t.Fatal(err)
	}

	if !proxycache.RequestCacheTransfer(be1, be2, key) {
		t.Fatal("RequestCacheTransfer returned false")
	}
	waitFor(t, 5*time.Second, func() bool {
		_, err := os.Stat(filepath.Join(agentDir, key))
		return err == nil
	})
	if meta := proxycache.GetKVMeta().ReadMeta(key, be2); meta == nil {
		t.Error("no meta registered on agent target backend")
	}
}
