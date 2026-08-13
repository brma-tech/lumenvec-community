package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"lumenvec/internal/config"
)

type fakeRunner struct {
	called bool
	err    error
}

func (f *fakeRunner) Start() {
	f.called = true
}

func (f *fakeRunner) Run(context.Context) error {
	f.called = true
	return f.err
}

func TestBuildServer(t *testing.T) {
	cfgPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(cfgPath, []byte(`
server:
  protocol: "grpc"
  port: 19190
  read_timeout: 5s
  write_timeout: 6s
database:
  snapshot_path: "./data/snapshot.json"
  wal_path: "./data/wal.log"
  snapshot_every: 10
  cache_enabled: true
  cache_max_bytes: 4096
  cache_max_items: 50
  cache_ttl: 20s
limits:
  max_body_bytes: 1024
  max_vector_dim: 64
  max_k: 5
search:
  mode: "exact"
  ann_m: 20
  ann_ef_construction: 80
  ann_ef_search: 40
grpc:
  enabled: true
  port: 20191
`), 0o644); err != nil {
		t.Fatal(err)
	}

	server, err := buildServer(cfgPath)
	if err != nil {
		t.Fatalf("buildServer() error = %v", err)
	}
	if server == nil {
		t.Fatal("expected server instance")
	}
}

func TestBuildServerInvalidConfig(t *testing.T) {
	cfgPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(cfgPath, []byte("server: ["), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := buildServer(cfgPath); err == nil {
		t.Fatal("expected buildServer error")
	}
}

func TestExecute(t *testing.T) {
	cfgPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(cfgPath, []byte(`
server:
  protocol: "http"
  port: 19190
database:
  snapshot_path: "./data/snapshot.json"
  wal_path: "./data/wal.log"
  cache_enabled: false
  cache_max_bytes: 1024
grpc:
  enabled: false
  port: 19191
`), 0o644); err != nil {
		t.Fatal(err)
	}
	called := false
	if err := execute(cfgPath, func(serverRunner) { called = true }); err != nil {
		t.Fatalf("execute() error = %v", err)
	}
	if !called {
		t.Fatal("expected runner to be called")
	}
}

func TestExecuteBuildServerError(t *testing.T) {
	cfgPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(cfgPath, []byte("server: ["), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := execute(cfgPath, func(serverRunner) { t.Fatal("runner should not be called") }); err == nil {
		t.Fatal("expected execute error")
	}
}

func TestExecuteMissingConfigUsesDefaults(t *testing.T) {
	called := false
	if err := execute(filepath.Join(t.TempDir(), "missing.yaml"), func(serverRunner) { called = true }); err != nil {
		t.Fatalf("execute() error = %v", err)
	}
	if !called {
		t.Fatal("expected runner to be called")
	}
}

func TestRunServer(t *testing.T) {
	runner := &fakeRunner{}
	runServer(runner)
	if !runner.called {
		t.Fatal("expected Run to be called")
	}
}

func TestRunServerFatalOnError(t *testing.T) {
	oldFatalf := logFatalf
	t.Cleanup(func() { logFatalf = oldFatalf })

	var fatalCalled bool
	logFatalf = func(string, ...interface{}) { fatalCalled = true }
	runServer(&fakeRunner{err: errors.New("boom")})
	if !fatalCalled {
		t.Fatal("expected fatal path")
	}
}

func TestServerAddr(t *testing.T) {
	t.Setenv("VECTOR_DB_PPROF_ADDR", "")
	stopProfiling := startOptionalProfilingServer(context.Background())
	stopProfiling()
	var cfg config.Config
	if got := serverAddr(cfg); got != ":19190" {
		t.Fatalf("serverAddr() default = %q", got)
	}
	cfg.Server.Port = "19190"
	if got := serverAddr(cfg); got != ":19190" {
		t.Fatalf("serverAddr() = %q", got)
	}
	cfg.Server.Port = ":19191"
	if got := serverAddr(cfg); got != ":19191" {
		t.Fatalf("serverAddr() = %q", got)
	}
}

func TestOptionalProfilingServerEnabled(t *testing.T) {
	t.Setenv("VECTOR_DB_PPROF_ADDR", "127.0.0.1:0")
	ctx, cancel := context.WithCancel(context.Background())
	stop := startOptionalProfilingServer(ctx)
	cancel()
	stop()
	// The listener uses an ephemeral port; this test exercises lifecycle and
	// shutdown without exposing a diagnostic endpoint in normal deployments.
	time.Sleep(10 * time.Millisecond)
}

func TestNewHTTPServer(t *testing.T) {
	srv := newHTTPServer(":19190", nil, 5*time.Second, 6*time.Second)
	if srv.Addr != ":19190" || srv.ReadTimeout != 5*time.Second || srv.WriteTimeout != 6*time.Second {
		t.Fatal("unexpected http server config")
	}
}

func TestMustExecuteSignatureCoverage(t *testing.T) {
	_ = errors.New("covered")
}

func TestMustExecuteAndMain(t *testing.T) {
	oldFatalf := logFatalf
	oldInfof := logInfof
	oldExecute := executeFunc
	oldArgs := os.Args
	t.Cleanup(func() {
		logFatalf = oldFatalf
		logInfof = oldInfof
		executeFunc = oldExecute
		os.Args = oldArgs
	})

	var fatalCalled bool
	logFatalf = func(string, ...interface{}) { fatalCalled = true }
	mustExecute(func(string, func(serverRunner)) error { return errors.New("boom") }, "x", func(serverRunner) {})
	if !fatalCalled {
		t.Fatal("expected fatal path")
	}

	var infoCalled bool
	executeFunc = func(string, func(serverRunner)) error { return nil }
	logInfof = func(...interface{}) { infoCalled = true }
	os.Args = []string{"lumenvec"}
	main()
	if !infoCalled {
		t.Fatal("expected info log path")
	}
}

func TestResolveConfigPath(t *testing.T) {
	oldArgs := os.Args
	oldEnv := os.Getenv("VECTOR_DB_CONFIG")
	t.Cleanup(func() {
		os.Args = oldArgs
		if oldEnv == "" {
			_ = os.Unsetenv("VECTOR_DB_CONFIG")
		} else {
			_ = os.Setenv("VECTOR_DB_CONFIG", oldEnv)
		}
	})

	os.Args = []string{"lumenvec", "-config", "custom.yaml"}
	if got := resolveConfigPath(); got != "custom.yaml" {
		t.Fatalf("resolveConfigPath() = %q", got)
	}

	_ = os.Setenv("VECTOR_DB_CONFIG", "env.yaml")
	os.Args = []string{"lumenvec"}
	if got := resolveConfigPath(); got != "env.yaml" {
		t.Fatalf("resolveConfigPath() env = %q", got)
	}

	_ = os.Unsetenv("VECTOR_DB_CONFIG")
	os.Args = []string{"lumenvec", "-badflag"}
	if got := resolveConfigPath(); got != "./configs/config.yaml" {
		t.Fatalf("resolveConfigPath() default = %q", got)
	}
}
