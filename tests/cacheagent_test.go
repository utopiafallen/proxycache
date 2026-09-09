package tests

import (
	"net"
	"os"
	"path/filepath"
	"proxycache"
	"strconv"
	"testing"
)

// Client tests run against the real standalone cache-agent binary so the
// contract is exercised end to end (see startCacheAgent in transfer_test.go).

func TestCacheAgentClientDeleteSuccess(t *testing.T) {
	dir := t.TempDir()
	urlStr, _ := startCacheAgent(t, dir)
	client := proxycache.NewCacheAgentClient(urlStr)
	defer client.Close()
	key := "test_key"
	if err := os.WriteFile(filepath.Join(dir, key), []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, key+".ckpt"), []byte("sc"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !client.Delete(key) {
		t.Error("Delete should return true on 200")
	}
	for _, name := range []string{key, key + ".ckpt"} {
		if _, err := os.Stat(filepath.Join(dir, name)); !os.IsNotExist(err) {
			t.Errorf("%s should have been deleted (with its sidecar)", name)
		}
	}
}

func TestCacheAgentClientDeleteMissing(t *testing.T) {
	urlStr, _ := startCacheAgent(t, t.TempDir())
	client := proxycache.NewCacheAgentClient(urlStr)
	defer client.Close()
	if client.Delete("no-such-key") {
		t.Error("Delete of a missing file should return false (404)")
	}
}

func TestCacheAgentClientConnectError(t *testing.T) {
	l, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	urlStr := "http://" + l.Addr().String()
	l.Close() // port is now dead
	client := proxycache.NewCacheAgentClient(urlStr)
	defer client.Close()
	if client.Delete("test_key") {
		t.Error("Delete should return false on connection error")
	}
	if got := client.GetFileSize("test_key"); got != nil {
		t.Errorf("GetFileSize on connection error = %v, want nil", got)
	}
	if got := client.Sidecars("test_key"); got != nil {
		t.Errorf("Sidecars on connection error = %v, want nil", got)
	}
}

func TestCacheAgentClientUploadEnforcesBudget(t *testing.T) {
	dir := t.TempDir()
	urlStr, _ := startCacheAgent(t, dir)
	client := proxycache.NewCacheAgentClient(urlStr)
	defer client.Close()
	if err := os.WriteFile(filepath.Join(dir, "resident"), make([]byte, 10), 0o644); err != nil {
		t.Fatal(err)
	}
	upPath := filepath.Join(t.TempDir(), "up.bin")
	if err := os.WriteFile(upPath, make([]byte, 5), 0o644); err != nil {
		t.Fatal(err)
	}
	// 10 used, need 5, budget 12 → resident (10 > 12-5) is evicted, upload fits.
	if !client.UploadFile("upload", upPath, 12) {
		t.Fatal("UploadFile within budget returned false")
	}
	if _, err := os.Stat(filepath.Join(dir, "upload")); err != nil {
		t.Errorf("uploaded file missing on target: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "resident")); !os.IsNotExist(err) {
		t.Error("resident should have been evicted to fit the upload")
	}
	// 5 used, need 5, budget 4 → cannot fit even after full eviction → rejected, nothing written.
	if client.UploadFile("big", upPath, 4) {
		t.Error("UploadFile over budget should return false (507)")
	}
	if _, err := os.Stat(filepath.Join(dir, "big")); !os.IsNotExist(err) {
		t.Error("over-budget upload must not be written")
	}
}

func TestCacheAgentClientGetFileSize(t *testing.T) {
	dir := t.TempDir()
	urlStr, _ := startCacheAgent(t, dir)
	client := proxycache.NewCacheAgentClient(urlStr)
	defer client.Close()
	if err := os.WriteFile(filepath.Join(dir, "exists_key"), make([]byte, 1234), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "exists_key.ckpt.2"), []byte("sc"), 0o644); err != nil {
		t.Fatal(err)
	}
	got := client.GetFileSize("exists_key")
	if got == nil || got["exists"] != true {
		t.Fatalf("GetFileSize(exists) = %v", got)
	}
	if v, ok := proxycache.ToFloat(got["size"]); !ok || v != 1234 {
		t.Errorf("GetFileSize size = %v, want 1234", got["size"])
	}
	if scs := client.Sidecars("exists_key"); len(scs) != 1 || scs[0] != "exists_key.ckpt.2" {
		t.Errorf("Sidecars = %v, want [exists_key.ckpt.2]", scs)
	}
	missing := client.GetFileSize("missing_key")
	if missing == nil || missing["exists"] != false {
		t.Errorf("GetFileSize(404) = %v, want {exists:false}", missing)
	}
	if got := strconv.Itoa(len(client.Sidecars("missing_key"))); got != "0" {
		t.Errorf("Sidecars(missing) = %s entries, want 0", got)
	}
}
