package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
)

var cacheDir string

func main() {
	cacheDir = os.Getenv("CACHE_DIR")
	if cacheDir == "" {
		log.Fatal("CACHE_DIR environment variable is required")
	}

	port := os.Getenv("AGENT_PORT")
	if port == "" {
		port = "8082"
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/cache/delete", handleDelete)

	addr := ":" + port
	log.Printf("cache-agent listening on %s (CACHE_DIR=%s)\n", addr, cacheDir)
	log.Fatal(http.ListenAndServe(addr, mux))
}

type deleteResponse struct {
	Ok    bool   `json:"ok"`
	Error string `json:"error,omitempty"`
}

func handleDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(deleteResponse{Ok: false, Error: "method not allowed"})
		return
	}

	key := r.URL.Query().Get("key")
	if key == "" {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(deleteResponse{Ok: false, Error: "key parameter is required"})
		return
	}

	filepath := cacheDir + "/" + key

	if err := os.Remove(filepath); err != nil {
		if os.IsNotExist(err) {
			log.Printf("cache delete: file not found: %s\n", key)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotFound)
			json.NewEncoder(w).Encode(deleteResponse{Ok: false, Error: "file not found"})
			return
		}
		log.Printf("cache delete: failed to remove %s: %v\n", key, err)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(deleteResponse{Ok: false, Error: err.Error()})
		return
	}

	log.Printf("cache delete: %s\n", key)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(deleteResponse{Ok: true})
}
