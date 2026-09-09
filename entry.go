package proxycache

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

// StartEmbeddedAgentServers starts an embedded cache-agent (the same server
// as the standalone cache-agent binary) for every local backend that set
// agent_serve_port, so other proxies can treat those backends as ordinary
// agent_port backends and P2P-transfer caches to/from them. Returns the
// number of servers started and a cancel func that stops them.
func StartEmbeddedAgentServers(ctx context.Context) (int, context.CancelFunc) {
	agentCtx, cancel := context.WithCancel(ctx)
	started := 0
	for _, key := range backendManager.Keys() {
		info := backendManager.Info(key)
		if info == nil || info.AgentServePort <= 0 || info.CacheDir == "" {
			continue
		}
		srv := NewAgentServer(info.CacheDir, info.AgentServePort)
		go srv.Start(agentCtx)
		started++
	}
	return started, cancel
}

func Main() {
	slotManager.InitFromDisk()

	backendURLs := make([]string, 0, len(Backends))
	for _, be := range Backends {
		backendURLs = append(backendURLs, strFromAny(be["url"]))
	}
	logInfo("main", "Starting on port %d with %d backends: %v", Port, len(Backends), backendURLs)

	if n, cancelAgents := StartEmbeddedAgentServers(context.Background()); n > 0 {
		logInfo("main", "Serving embedded cache-agent for %d local backend(s)", n)
		defer cancelAgents()
	}

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
	GetDispatcher().Start()

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", ModelsHandler)
	mux.HandleFunc("/v1/chat/completions", ChatHandler)
	mux.HandleFunc("/metrics/dashboard", metricsDashboardHandler)
	mux.HandleFunc("/metrics/health", metricsHealthHandler)
	mux.HandleFunc("/metrics/slots", metricsSlotsHandler)
	mux.HandleFunc("/metrics/cache", metricsCacheHandler)
	mux.HandleFunc("/metrics/diagnostics", metricsDiagnosticsHandler)
	mux.HandleFunc("/metrics/requests", metricsRequestsHandler)
	mux.HandleFunc("/metrics/request/", metricsRequestByIDHandler)
	mux.HandleFunc("/metrics/performance", metricsPerformanceHandler)
	mux.HandleFunc("/dashboard", DashboardHandler)

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
	GetDispatcher().Stop(10 * time.Second)
	backendManager.StopLivenessChecker()
	backendManager.Close()
}
