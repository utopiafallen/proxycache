package tests

import (
	"bytes"
	"context"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"proxycache"
	"strconv"
	"testing"
	"time"
)

// startCacheAgent runs the real standalone cache-agent binary against dir and
// returns its base URL and host:port. The agent tests exercise the exact
// production code path the proxy talks to, so they skip (rather than fall
// back to an in-process fake) when the binary has not been built.
func startCacheAgent(t *testing.T, dir string) (string, string) {
	t.Helper()
	bin := filepath.Join("..", "cache-agent.exe")
	if _, err := os.Stat(bin); err != nil {
		t.Skipf("cache-agent.exe not built; run ./build-cache-agent.sh to enable agent tests (%v)", err)
	}
	port := findFreePort(t)
	var logBuf bytes.Buffer
	cmd := exec.Command(bin, "-cache-dir", dir, "-port", strconv.Itoa(port))
	cmd.Stdout = &logBuf
	cmd.Stderr = &logBuf
	if err := cmd.Start(); err != nil {
		t.Fatalf("starting cache-agent: %v", err)
	}
	t.Cleanup(func() {
		cmd.Process.Kill()
		cmd.Wait()
		if logBuf.Len() > 0 {
			t.Logf("cache-agent log:\n%s", logBuf.String())
		}
	})
	addr := "127.0.0.1:" + strconv.Itoa(port)
	waitFor(t, 10*time.Second, func() bool {
		c, err := net.DialTimeout("tcp4", addr, 100*time.Millisecond)
		if err != nil {
			return false
		}
		c.Close()
		return true
	})
	return "http://" + addr, addr
}

func findFreePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("finding free port: %v", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	return port
}

func TestAgentP2PTransferPush(t *testing.T) {
	srcDir, dstDir := t.TempDir(), t.TempDir()
	srcURL, _ := startCacheAgent(t, srcDir)
	dstURL, _ := startCacheAgent(t, dstDir)
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

func TestAgentP2PTransferPushSidecars(t *testing.T) {
	srcDir, dstDir := t.TempDir(), t.TempDir()
	srcURL, _ := startCacheAgent(t, srcDir)
	dstURL, _ := startCacheAgent(t, dstDir)
	key := "skey"
	files := map[string]string{
		key:           "main-payload",
		key + ".ckpt":  "ckpt-a",
		key + ".ckpt.0": "ckpt-b",
		// Not sidecars of this key — must NOT travel with it.
		key + ".ckptx":    "unrelated",
		"otherkey.ckpt":   "someone-elses",
	}
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(srcDir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	client := proxycache.NewCacheAgentClient(srcURL)
	defer client.Close()
	if !client.Transfer(key, dstURL, 0) {
		t.Fatal("Transfer returned false")
	}
	for name, want := range map[string]string{key: "main-payload", key + ".ckpt": "ckpt-a", key + ".ckpt.0": "ckpt-b"} {
		data, err := os.ReadFile(filepath.Join(dstDir, name))
		if err != nil {
			t.Fatalf("transferred %s missing on target: %v", name, err)
		}
		if string(data) != want {
			t.Errorf("transferred %s content = %q, want %q", name, data, want)
		}
	}
	for _, name := range []string{key + ".ckptx", "otherkey.ckpt"} {
		if _, err := os.Stat(filepath.Join(dstDir, name)); !os.IsNotExist(err) {
			t.Errorf("non-sidecar %s should not have been transferred", name)
		}
	}
	// No sidecars on the source → plain transfer still works.
	key2 := "barekey"
	if err := os.WriteFile(filepath.Join(srcDir, key2), []byte("bare"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !client.Transfer(key2, dstURL, 0) {
		t.Fatal("Transfer of key without sidecars returned false")
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
	dstURL, _ := startCacheAgent(t, dstDir)
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
	srcURL, _ := startCacheAgent(t, srcDir)
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

func TestAgentMakeSpaceRemovesSidecarWithKey(t *testing.T) {
	dstDir := t.TempDir()
	// Oldest entry has a sidecar; evicting it must remove both.
	p1 := filepath.Join(dstDir, "oldkey")
	if err := os.WriteFile(p1, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dstDir, "oldkey.ckpt"), []byte("side"), 0o644); err != nil {
		t.Fatal(err)
	}
	p2 := filepath.Join(dstDir, "newkey")
	if err := os.WriteFile(p2, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	mtOld := time.Now().Add(-2 * time.Hour)
	mtNew := time.Now()
	if err := os.Chtimes(p1, mtOld, mtOld); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(filepath.Join(dstDir, "oldkey.ckpt"), mtOld, mtOld); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(p2, mtNew, mtNew); err != nil {
		t.Fatal(err)
	}
	dstURL, _ := startCacheAgent(t, dstDir)
	client := proxycache.NewCacheAgentClient(dstURL)
	defer client.Close()

	ok, used := client.MakeSpace(9, 10)
	if !ok {
		t.Fatalf("MakeSpace ok=false")
	}
	if used != 1 {
		t.Errorf("used after make-space = %d, want 1", used)
	}
	for _, name := range []string{"oldkey", "oldkey.ckpt"} {
		if _, err := os.Stat(filepath.Join(dstDir, name)); !os.IsNotExist(err) {
			t.Errorf("%s should have been evicted together with its key", name)
		}
	}
}

func TestAgentTransferMissingSource(t *testing.T) {
	srcURL, _ := startCacheAgent(t, t.TempDir())
	dstURL, _ := startCacheAgent(t, t.TempDir())
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
// two local cache_dir backends: file copy (main + sidecars), meta
// registration, ring entry.
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
	if err := os.WriteFile(filepath.Join(dir1, key+".ckpt"), []byte("sidecar-a"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir1, key+".ckpt.1"), []byte("sidecar-b"), 0o644); err != nil {
		t.Fatal(err)
	}

	if !proxycache.RequestCacheTransfer(be1, be2, key) {
		t.Fatal("RequestCacheTransfer returned false")
	}
	// Wait for the ring entry, not just the files: meta rewrite + ring
	// registration happen after the last file lands, so file existence alone
	// is not a completion signal (races under low GOMAXPROCS / load).
	waitFor(t, 5*time.Second, func() bool {
		_, e1 := os.Stat(filepath.Join(dir2, key))
		_, e2 := os.Stat(filepath.Join(dir2, key+".ckpt"))
		_, e3 := os.Stat(filepath.Join(dir2, key+".ckpt.1"))
		return e1 == nil && e2 == nil && e3 == nil &&
			proxycache.GetSlotManager().Get(be2).GetRingSize() == 1
	})
	data, err := os.ReadFile(filepath.Join(dir2, key))
	if err != nil || string(data) != "src-cache" {
		t.Fatalf("transferred file content = %q (err=%v), want src-cache", data, err)
	}
	for name, want := range map[string]string{key + ".ckpt": "sidecar-a", key + ".ckpt.1": "sidecar-b"} {
		data, err := os.ReadFile(filepath.Join(dir2, name))
		if err != nil || string(data) != want {
			t.Errorf("transferred %s = %q (err=%v), want %q", name, data, err, want)
		}
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
	agentURL, agentAddr := startCacheAgent(t, agentDir)
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
	if err := os.WriteFile(filepath.Join(dir1, key+".ckpt.0"), []byte("sc-zero"), 0o644); err != nil {
		t.Fatal(err)
	}

	if !proxycache.RequestCacheTransfer(be1, be2, key) {
		t.Fatal("RequestCacheTransfer returned false")
	}
	waitFor(t, 5*time.Second, func() bool {
		_, e1 := os.Stat(filepath.Join(agentDir, key))
		_, e2 := os.Stat(filepath.Join(agentDir, key+".ckpt.0"))
		return e1 == nil && e2 == nil &&
			proxycache.GetSlotManager().Get(be2).GetRingSize() == 1
	})
	data, err := os.ReadFile(filepath.Join(agentDir, key+".ckpt.0"))
	if err != nil || string(data) != "sc-zero" {
		t.Errorf("transferred sidecar = %q (err=%v), want sc-zero", data, err)
	}
	if meta := proxycache.GetKVMeta().ReadMeta(key, be2); meta == nil {
		t.Error("no meta registered on agent target backend")
	}
}

// TestProxyTransferAgentToLocal covers the proxy orchestration path where the
// source backend manages its cache through a remote cache-agent (pull).
func TestProxyTransferAgentToLocal(t *testing.T) {
	withTempMetaDir(t)
	agentDir := t.TempDir()
	dir2 := t.TempDir()
	agentURL, agentAddr := startCacheAgent(t, agentDir)
	_, agentPort, err := net.SplitHostPort(agentAddr)
	if err != nil {
		t.Fatalf("bad agent addr %q: %v", agentAddr, err)
	}
	_ = agentURL
	be1 := backendKeyFromURL("http://127.0.0.1:3")
	be2 := backendKeyFromURL("http://127.0.0.1:4")
	withTestBackend(t, []map[string]any{
		{"url": "http://127.0.0.1:3", "agent_port": agentPort},
		{"url": "http://127.0.0.1:4", "cache_dir": dir2},
	})
	markBackendsUp(proxycache.GetBackendManager())
	tokens := seq(10)
	key := proxycache.MetaKey("xfer3-model", tokens)
	blocks := proxycache.BlockHashesFromTokens(tokens, proxycache.WordsPerBlock)
	if err := os.WriteFile(filepath.Join(agentDir, key), []byte("agent-cache"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, key+".ckpt"), []byte("agent-sidecar"), 0o644); err != nil {
		t.Fatal(err)
	}
	proxycache.GetKVMeta().WriteMeta(key, 10, blocks, proxycache.WordsPerBlock, "xfer3-model", be1, 11)

	if !proxycache.RequestCacheTransfer(be1, be2, key) {
		t.Fatal("RequestCacheTransfer returned false")
	}
	waitFor(t, 5*time.Second, func() bool {
		_, e1 := os.Stat(filepath.Join(dir2, key))
		_, e2 := os.Stat(filepath.Join(dir2, key+".ckpt"))
		return e1 == nil && e2 == nil &&
			proxycache.GetSlotManager().Get(be2).GetRingSize() == 1
	})
	data, err := os.ReadFile(filepath.Join(dir2, key))
	if err != nil || string(data) != "agent-cache" {
		t.Fatalf("transferred file = %q (err=%v), want agent-cache", data, err)
	}
	data, err = os.ReadFile(filepath.Join(dir2, key + ".ckpt"))
	if err != nil || string(data) != "agent-sidecar" {
		t.Errorf("transferred sidecar = %q (err=%v), want agent-sidecar", data, err)
	}
	if meta := proxycache.GetKVMeta().ReadMeta(key, be2); meta == nil {
		t.Error("no meta registered on local target backend")
	}
}

// TestLocalBackendServesAgentPort covers the embedded cache-agent: a local
// backend that set agent_serve_port must serve the cache-agent API for its
// own cache dir once StartEmbeddedAgentServers runs (the production wiring in
// Main).
func TestLocalBackendServesAgentPort(t *testing.T) {
	withTempMetaDir(t)
	dir := t.TempDir()
	m1 := newMockLlama(t, "agentport-model", 32768, seq(10), 1)
	port := findFreePort(t)
	withTestBackend(t, []map[string]any{
		{"url": m1.srv.URL, "cache_dir": dir, "agent_serve_port": port},
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started, _ := proxycache.StartEmbeddedAgentServers(ctx)
	if started != 1 {
		t.Fatalf("StartEmbeddedAgentServers started %d servers, want 1", started)
	}
	addr := "127.0.0.1:" + strconv.Itoa(port)
	waitFor(t, 5*time.Second, func() bool {
		c, err := net.DialTimeout("tcp4", addr, 100*time.Millisecond)
		if err != nil {
			return false
		}
		c.Close()
		return true
	})
	client := proxycache.NewCacheAgentClient("http://" + addr)
	defer client.Close()
	key := "embedded-key"
	if err := os.WriteFile(filepath.Join(dir, key), []byte("emb-payload"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, key+".ckpt"), []byte("emb-ckpt"), 0o644); err != nil {
		t.Fatal(err)
	}
	info := client.GetFileSize(key)
	if info == nil || info["exists"] != true {
		t.Fatalf("GetFileSize = %v, want exists", info)
	}
	if v, _ := proxycache.ToFloat(info["size"]); v != 11 {
		t.Errorf("embedded agent size = %v, want 11", v)
	}
	scs := client.Sidecars(key)
	if len(scs) != 1 || scs[0] != key+".ckpt" {
		t.Errorf("embedded agent sidecars = %v, want [%s.ckpt]", scs, key)
	}
	if !client.Delete(key) {
		t.Error("Delete through embedded agent returned false")
	}
	if _, err := os.Stat(filepath.Join(dir, key)); !os.IsNotExist(err) {
		t.Error("Delete through embedded agent did not remove the file")
	}
	if _, err := os.Stat(filepath.Join(dir, key + ".ckpt")); !os.IsNotExist(err) {
		t.Error("Delete through embedded agent did not remove the sidecar")
	}
}
