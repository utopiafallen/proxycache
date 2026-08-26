package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

func main() {
	slotManager.InitFromDisk()

	backendURLs := make([]string, 0, len(Backends))
	for _, be := range Backends {
		backendURLs = append(backendURLs, strFromAny(be["url"]))
	}
	logInfo("main", "Starting on port %d with %d backends: %v", Port, len(Backends), backendURLs)

	backendKeys := backendManager.Keys()
	if reconciled := kvMeta.Reconcile(backendKeys); reconciled > 0 {
		logInfo("main", "Cleaned up %d orphaned/corrupted meta files at startup", reconciled)
	}

	metaCount := 0
	for _, k := range backendKeys {
		metaCount += len(kvMeta.ListKeys(k))
	}
	cacheFiles := 0
	for _, k := range backendKeys {
		dir := backendManager.GetCacheDir(k)
		if dir == "" {
			continue
		}
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, e := range entries {
			info, statErr := os.Stat(filepath.Join(dir, e.Name()))
			if statErr == nil && info.Mode().IsRegular() {
				cacheFiles++
			}
		}
	}
	logInfo("main", "After startup reconcile: %d meta files, %d cache files on disk", metaCount, cacheFiles)

	backendManager.StartLivenessChecker()

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", modelsHandler)
	mux.HandleFunc("/v1/chat/completions", chatHandler)
	mux.HandleFunc("/metrics/dashboard", metricsDashboardHandler)
	mux.HandleFunc("/metrics/health", metricsHealthHandler)
	mux.HandleFunc("/metrics/slots", metricsSlotsHandler)
	mux.HandleFunc("/metrics/cache", metricsCacheHandler)
	mux.HandleFunc("/metrics/diagnostics", metricsDiagnosticsHandler)
	mux.HandleFunc("/metrics/requests", metricsRequestsHandler)
	mux.HandleFunc("/metrics/request/", metricsRequestByIDHandler)
	mux.HandleFunc("/metrics/performance", metricsPerformanceHandler)
	mux.HandleFunc("/dashboard", dashboardHandler)

	srv := &http.Server{Addr: fmt.Sprintf("0.0.0.0:%d", Port), Handler: mux}
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logError("main", "Server error: %v", err)
			os.Exit(1)
		}
	}()
	logInfo("main", "proxycache listening on 0.0.0.0:%d", Port)

	<-sigCh
	logInfo("main", "Shutting down...")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		logError("main", "Graceful shutdown failed: %v", err)
	}
	backendManager.StopLivenessChecker()
	backendManager.Close()
}
