package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"lumenvec/internal/api"
	"lumenvec/internal/bootstrap"
	"lumenvec/internal/config"
	"net/http"
	"net/http/pprof"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

var (
	executeFunc = execute
	logFatalf   = log.Fatalf
	logInfof    = log.Println
)

func main() {
	mustExecute(executeFunc, resolveConfigPath(), runServer)
	logInfof("Server stopped")
}

func execute(configPath string, runner func(serverRunner)) error {
	server, err := buildServer(configPath)
	if err != nil {
		return err
	}
	runner(server)
	return nil
}

func buildServer(configPath string) (*api.Server, error) {
	cfg, err := config.Load(configPath)
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	options, err := bootstrap.CommunityServerOptions(cfg)
	if err != nil {
		return nil, err
	}
	server := api.NewServerWithOptions(options)
	return server, nil
}

type serverRunner interface {
	Start()
	Run(context.Context) error
}

func runServer(server serverRunner) {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	stopProfiling := startOptionalProfilingServer(ctx)
	defer stopProfiling()
	if err := server.Run(ctx); err != nil {
		logFatalf("server stopped with error: %v", err)
	}
}

// startOptionalProfilingServer exposes pprof only when explicitly configured.
// Keeping it disabled by default avoids adding an unauthenticated diagnostic
// surface to normal deployments.
func startOptionalProfilingServer(ctx context.Context) func() {
	addr := strings.TrimSpace(os.Getenv("VECTOR_DB_PPROF_ADDR"))
	if addr == "" {
		return func() {}
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	mux.Handle("/debug/pprof/allocs", pprof.Handler("allocs"))
	mux.Handle("/debug/pprof/block", pprof.Handler("block"))
	mux.Handle("/debug/pprof/goroutine", pprof.Handler("goroutine"))
	mux.Handle("/debug/pprof/heap", pprof.Handler("heap"))
	mux.Handle("/debug/pprof/mutex", pprof.Handler("mutex"))
	mux.Handle("/debug/pprof/threadcreate", pprof.Handler("threadcreate"))
	httpServer := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	go func() {
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logInfof("pprof server stopped: %v", err)
		}
	}()
	go func() {
		<-ctx.Done()
		_ = httpServer.Close()
	}()
	return func() { _ = httpServer.Close() }
}

func resolveConfigPath() string {
	defaultPath := "./configs/config.yaml"
	if envPath := strings.TrimSpace(os.Getenv("VECTOR_DB_CONFIG")); envPath != "" {
		defaultPath = envPath
	}

	fs := flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	configPath := fs.String("config", defaultPath, "path to the configuration file")
	_ = fs.Parse(os.Args[1:])
	return *configPath
}

func mustExecute(executor func(string, func(serverRunner)) error, configPath string, runner func(serverRunner)) {
	if err := executor(configPath, runner); err != nil {
		logFatalf("failed to initialize server: %v", err)
	}
}

func serverAddr(cfg config.Config) string {
	port := cfg.Server.Port
	if port == "" {
		port = "19190"
	}
	if port[0] != ':' {
		return ":" + port
	}
	return port
}

func newHTTPServer(addr string, handler http.Handler, readTimeout, writeTimeout time.Duration) *http.Server {
	return &http.Server{
		Addr:         addr,
		Handler:      handler,
		ReadTimeout:  readTimeout,
		WriteTimeout: writeTimeout,
	}
}
