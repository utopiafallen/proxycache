package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCacheAgentClientDeleteSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/cache/delete" || r.URL.Query().Get("key") != "test_key" {
			t.Errorf("unexpected request %s %s", r.URL.Path, r.URL.RawQuery)
		}
		writeJSON(w, 200, map[string]any{"ok": true})
	}))
	defer srv.Close()
	client := NewCacheAgentClient(srv.URL)
	defer client.Close()
	if !client.Delete("test_key") {
		t.Error("Delete should return true on 200")
	}
}

func TestCacheAgentClientDeleteFailure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 500, map[string]any{"ok": false, "error": "boom"})
	}))
	defer srv.Close()
	client := NewCacheAgentClient(srv.URL)
	defer client.Close()
	if client.Delete("test_key") {
		t.Error("Delete should return false on non-200")
	}
}

func TestCacheAgentClientConnectError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	urlStr := srv.URL
	srv.Close()
	client := NewCacheAgentClient(urlStr)
	defer client.Close()
	if client.Delete("test_key") {
		t.Error("Delete should return false on connection error")
	}
	if got := client.GetFileSize("test_key"); got != nil {
		t.Errorf("GetFileSize on connection error = %v, want nil", got)
	}
}

func TestCacheAgentClientGetFileSize(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/cache/files/exists_key":
			writeJSON(w, 200, map[string]any{"size": 1234, "exists": true})
		case "/cache/files/missing_key":
			writeJSON(w, 404, map[string]any{})
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
		}
	}))
	defer srv.Close()
	client := NewCacheAgentClient(srv.URL)
	defer client.Close()
	got := client.GetFileSize("exists_key")
	if got == nil || got["exists"] != true {
		t.Errorf("GetFileSize(exists) = %v", got)
	}
	if v, ok := toFloat(got["size"]); !ok || v != 1234 {
		t.Errorf("GetFileSize size = %v, want 1234", got["size"])
	}
	missing := client.GetFileSize("missing_key")
	if missing == nil || missing["exists"] != false {
		t.Errorf("GetFileSize(404) = %v, want {exists:false}", missing)
	}
}
