package main

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"os"
)

var cacheDir string
var port string

func main() {
	flag.StringVar(&cacheDir, "cache-dir", "", "Path to llama.cpp slot-save-directory")
	flag.StringVar(&port, "port", "8082", "HTTP listen port")
	flag.Parse()

	if cacheDir == "" {
		flag.Usage()
		os.Exit(1)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/cache/delete", handleDelete)
	mux.HandleFunc("/cache/files/batch", handleBatchFileSizes)
	mux.HandleFunc("/cache/files/", handleFileSizes)

	addr := ":" + port
	log.Printf("cache-agent listening on %s (CACHE_DIR=%s)\n", addr, cacheDir)
	log.Fatal(http.ListenAndServe(addr, mux))
}

type deleteResponse struct {
	Ok    bool   `json:"ok"`
	Error string `json:"error,omitempty"`
}

type fileResponse struct {
	Size   int64 `json:"size"`
	Exists bool  `json:"exists"`
}

type batchRequest struct {
	Keys []string `json:"keys"`
}

type batchResponse struct {
	Results map[string]fileResponse `json:"results"`
}

func handleBatchFileSizes(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(batchResponse{Results: map[string]fileResponse{}})
		return
	}

	var req batchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(batchResponse{Results: map[string]fileResponse{}})
		return
	}

	results := make(map[string]fileResponse)
	for _, key := range req.Keys {
		filepath := cacheDir + "/" + key
		info, err := os.Stat(filepath)
		if err != nil {
			results[key] = fileResponse{Exists: false}
		} else {
			results[key] = fileResponse{Size: info.Size(), Exists: true}
		}
	}

	log.Printf("batch file size: %d keys queried\n", len(req.Keys))
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(batchResponse{Results: results})
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

	// Delete ckpt sidecar files (<key>.ckpt, <key>.ckpt.0, etc.)
	if entries, err := os.ReadDir(cacheDir); err == nil {
		prefix := key + ".ckpt"
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			name := entry.Name()
			if name == prefix || len(name) > len(prefix)+1 && name[:len(prefix)+1] == prefix+"."{
				sidecar := cacheDir + "/" + name
				if err := os.Remove(sidecar); err == nil {
					log.Printf("cache delete: sidecar %s\n", name)
				} else {
					log.Printf("cache delete: failed to remove sidecar %s: %v\n", name, err)
				}
			}
		}
	}

	log.Printf("cache delete: %s\n", key)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(deleteResponse{Ok: true})
}

func handleFileSizes(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(fileResponse{Exists: false})
		return
	}

	basename := r.URL.Path[len("/cache/files/"):]
	if basename == "" {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(fileResponse{Exists: false})
		return
	}

	filepath := cacheDir + "/" + basename

	info, err := os.Stat(filepath)
	if err != nil {
		if os.IsNotExist(err) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotFound)
			json.NewEncoder(w).Encode(fileResponse{Exists: false})
			return
		}
		log.Printf("cache file info: failed to stat %s: %v\n", basename, err)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(fileResponse{Exists: false})
		return
	}

	log.Printf("cache file size: %s size=%d\n", basename, info.Size())
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(fileResponse{Size: info.Size(), Exists: true})
}
