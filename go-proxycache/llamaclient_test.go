package main

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestSaveSlotResponseParsing(t *testing.T) {
	withTempMetaDir(t)
	t.Run("n_written_extracted", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			writeJSON(w, http.StatusOK, map[string]any{
				"id_slot":   0,
				"filename":  "test_cache",
				"n_saved":   1745,
				"n_written": 14309796,
				"timings":   map[string]any{"save_ms": 49.865},
			})
		}))
		defer srv.Close()
		client := NewLlamaClient(srv.URL)
		ok, n, err := client.SaveSlot(context.Background(), 0, "test_cache", "")
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if !ok {
			t.Error("ok = false, want true")
		}
		if n != 14309796 {
			t.Errorf("n_written = %d, want 14309796", n)
		}
	})
	t.Run("missing_n_written_defaults_zero", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			writeJSON(w, http.StatusOK, map[string]any{"id_slot": 0, "filename": "test"})
		}))
		defer srv.Close()
		client := NewLlamaClient(srv.URL)
		ok, n, err := client.SaveSlot(context.Background(), 0, "test", "")
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if !ok {
			t.Error("ok = false, want true")
		}
		if n != 0 {
			t.Errorf("n_written = %d, want 0", n)
		}
	})
	t.Run("server_error_returns_false", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"error": "save failed"})
		}))
		defer srv.Close()
		client := NewLlamaClient(srv.URL)
		ok, n, err := client.SaveSlot(context.Background(), 0, "test", "")
		if err != nil {
			t.Fatalf("err = %v, want nil (swallowed 500)", err)
		}
		if ok {
			t.Error("ok = true, want false")
		}
		if n != 0 {
			t.Errorf("n_written = %d, want 0", n)
		}
	})
	t.Run("other_status_returns_error", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			writeJSON(w, http.StatusNotFound, map[string]any{"error": "no such slot"})
		}))
		defer srv.Close()
		client := NewLlamaClient(srv.URL)
		_, _, err := client.SaveSlot(context.Background(), 0, "test", "")
		var statusErr *HTTPStatusError
		if !errors.As(err, &statusErr) {
			t.Fatalf("err = %v, want *HTTPStatusError", err)
		}
		if statusErr.StatusCode != http.StatusNotFound {
			t.Errorf("status = %d, want 404", statusErr.StatusCode)
		}
	})
}
