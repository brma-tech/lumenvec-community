package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLoadDefaultsWhenFileMissing(t *testing.T) {
	cfg, err := Load(filepath.Join(t.TempDir(), "missing.yaml"))
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.Server.Port != "19190" || cfg.Search.Mode != "exact" {
		t.Fatal("expected default config values")
	}
	if cfg.Server.Protocol != "http" {
		t.Fatal("expected default http protocol")
	}
	if cfg.Search.ANNM != 16 || cfg.Search.ANNEfConstruct != 64 || cfg.Search.ANNEfSearch != 64 {
		t.Fatal("expected default ann config values")
	}
	if cfg.Search.ANNProfile != "balanced" {
		t.Fatal("expected default ann profile")
	}
	if cfg.Search.ANNEvalSampleRate != 0 {
		t.Fatal("expected default ann eval sample rate")
	}
	if cfg.Security.Profile != "development" {
		t.Fatal("expected default development security profile")
	}
	if cfg.Security.Storage.DirMode != "0755" || cfg.Security.Storage.FileMode != "0644" {
		t.Fatal("expected relaxed development file modes")
	}
	if cfg.Database.SyncEvery != 1 {
		t.Fatal("expected default sync_every")
	}
	if cfg.Database.ShardCount != 1 || cfg.Database.ShardFanout != 1 {
		t.Fatal("expected single-shard defaults")
	}
	if cfg.Cluster.ReplicationMode != "async" || cfg.Cluster.QueueSize != 1024 || cfg.Cluster.WriteQuorum != 1 {
		t.Fatal("expected cluster defaults")
	}
}

func TestLoadFromFileAndEnv(t *testing.T) {
	cfgPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(cfgPath, []byte(`
server:
  protocol: "grpc"
  port: 20000
  read_timeout: 1s
  write_timeout: 2s
  api_key: "from-file"
  rate_limit_rps: 20
database:
  snapshot_path: "./snap.json"
  wal_path: "./wal.log"
  snapshot_every: 12
  vector_store: "disk"
  vector_path: "./vectors"
  sync_every: 16
  shard_count: 4
  shard_fanout_concurrency: 2
  cache_enabled: true
  cache_max_bytes: 2048
  cache_max_items: 321
  cache_ttl: "30s"
limits:
  max_body_bytes: 123
  max_vector_dim: 88
  max_k: 9
search:
  mode: "ann"
  ann_profile: "fast"
  ann_m: 24
  ann_ef_construction: 96
  ann_ef_search: 48
  ann_eval_sample_rate: 15
grpc:
  enabled: true
  port: 21000
cluster:
  topology_path: "./topology.yaml"
  replication_mode: "quorum"
  replication_queue_size: 500
  write_quorum: 2
security:
  profile: "production"
  auth:
    enabled: true
    api_key: "file-secret"
    grpc_enabled: true
  transport:
    tls_enabled: true
    cert_file: "./cert.pem"
    key_file: "./key.pem"
  proxy:
    trust_forwarded_for: true
    trusted_proxies: ["10.0.0.1", "10.0.0.2"]
  storage:
    strict_file_permissions: true
    dir_mode: "0710"
    file_mode: "0640"
`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VECTOR_DB_PORT", "30000")
	t.Setenv("VECTOR_DB_API_KEY", "from-env")
	t.Setenv("VECTOR_DB_SECURITY_API_KEY", "from-security-env")

	cfg, err := Load(cfgPath)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.Server.Port != "30000" || cfg.Server.APIKey != "from-env" || cfg.Search.Mode != "ann" {
		t.Fatal("expected yaml + env overrides")
	}
	if cfg.Server.Protocol != "grpc" {
		t.Fatal("expected grpc protocol from yaml")
	}
	if cfg.Search.ANNM != 24 || cfg.Search.ANNEfConstruct != 96 || cfg.Search.ANNEfSearch != 48 {
		t.Fatal("expected ann config from yaml")
	}
	if cfg.Search.ANNProfile != "fast" {
		t.Fatal("expected ann profile from yaml")
	}
	if cfg.Search.ANNEvalSampleRate != 15 {
		t.Fatal("expected ann eval sample rate from yaml")
	}
	if !cfg.GRPC.Enabled || cfg.GRPC.Port != "21000" {
		t.Fatal("expected grpc config from yaml")
	}
	if cfg.Cluster.TopologyPath != "./topology.yaml" || cfg.Cluster.ReplicationMode != "quorum" || cfg.Cluster.QueueSize != 500 || cfg.Cluster.WriteQuorum != 2 {
		t.Fatal("expected cluster config from yaml")
	}
	if cfg.Database.VectorStore != "disk" || cfg.Database.VectorPath != "./vectors" {
		t.Fatal("expected vector store config from yaml")
	}
	if cfg.Database.SyncEvery != 16 {
		t.Fatal("expected sync_every config from yaml")
	}
	if cfg.Database.ShardCount != 4 || cfg.Database.ShardFanout != 2 {
		t.Fatal("expected shard config from yaml")
	}
	if !cfg.Database.CacheEnabled || cfg.Database.CacheMaxBytes != 2048 || cfg.Database.CacheMaxItems != 321 || cfg.Database.CacheTTL != "30s" {
		t.Fatal("expected cache config from yaml")
	}
	if cfg.Security.Profile != "production" || !cfg.Security.Auth.Enabled || !cfg.Security.Auth.GRPCEnabled {
		t.Fatal("expected security config from yaml")
	}
	if cfg.Security.Auth.APIKey != "from-security-env" {
		t.Fatal("expected security api key env override")
	}
	if !cfg.Security.Transport.TLSEnabled || cfg.Security.Transport.CertFile != "./cert.pem" || cfg.Security.Transport.KeyFile != "./key.pem" {
		t.Fatal("expected tls config from yaml")
	}
	if !cfg.Security.Proxy.TrustForwardedFor || len(cfg.Security.Proxy.TrustedProxies) != 2 {
		t.Fatal("expected proxy config from yaml")
	}
	if !cfg.Security.Storage.StrictFilePermissions || cfg.Security.Storage.DirMode != "0710" || cfg.Security.Storage.FileMode != "0640" {
		t.Fatal("expected storage security config from yaml")
	}
}

func TestParseDuration(t *testing.T) {
	if got := ParseDuration("5s", time.Second); got != 5*time.Second {
		t.Fatalf("ParseDuration() = %v", got)
	}
	if got := ParseDuration("  ", time.Second); got != time.Second {
		t.Fatalf("ParseDuration() empty fallback = %v", got)
	}
	if got := ParseDuration("bad", time.Second); got != time.Second {
		t.Fatalf("ParseDuration() fallback = %v", got)
	}
}

func TestLoadInvalidYAML(t *testing.T) {
	cfgPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(cfgPath, []byte("server: ["), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(cfgPath); err == nil {
		t.Fatal("expected yaml error")
	}
}

func TestOverrideFromEnvIgnoresInvalidNumericValues(t *testing.T) {
	cfg := defaultConfig()
	t.Setenv("VECTOR_DB_ACCESS_LOG", "not-bool")
	t.Setenv("VECTOR_DB_METRICS_ENABLED", "not-bool")
	t.Setenv("VECTOR_DB_RATE_LIMIT_RPS", "bad")
	t.Setenv("VECTOR_DB_SNAPSHOT_EVERY", "-1")
	t.Setenv("VECTOR_DB_SYNC_EVERY", "0")
	t.Setenv("VECTOR_DB_SHARD_COUNT", "0")
	t.Setenv("VECTOR_DB_SHARD_FANOUT_CONCURRENCY", "bad")
	t.Setenv("VECTOR_DB_CACHE_ENABLED", "not-bool")
	t.Setenv("VECTOR_DB_CACHE_MAX_BYTES", "0")
	t.Setenv("VECTOR_DB_CACHE_MAX_ITEMS", "bad")
	t.Setenv("VECTOR_DB_MAX_BODY_BYTES", "bad")
	t.Setenv("VECTOR_DB_MAX_VECTOR_DIM", "bad")
	t.Setenv("VECTOR_DB_MAX_K", "bad")
	t.Setenv("VECTOR_DB_ANN_M", "bad")
	t.Setenv("VECTOR_DB_ANN_EF_CONSTRUCTION", "0")
	t.Setenv("VECTOR_DB_ANN_EF_SEARCH", "-1")
	t.Setenv("VECTOR_DB_ANN_EVAL_SAMPLE_RATE", "-1")
	t.Setenv("VECTOR_DB_GRPC_ENABLED", "not-bool")
	t.Setenv("VECTOR_DB_REPLICATION_MODE", "invalid")
	t.Setenv("VECTOR_DB_REPLICATION_QUEUE_SIZE", "0")
	t.Setenv("VECTOR_DB_WRITE_QUORUM", "bad")
	t.Setenv("VECTOR_DB_SECURITY_AUTH_ENABLED", "not-bool")
	t.Setenv("VECTOR_DB_SECURITY_GRPC_AUTH_ENABLED", "not-bool")
	t.Setenv("VECTOR_DB_TLS_ENABLED", "not-bool")
	t.Setenv("VECTOR_DB_TRUST_FORWARDED_FOR", "not-bool")
	t.Setenv("VECTOR_DB_STRICT_FILE_PERMISSIONS", "not-bool")
	overrideFromEnv(&cfg)
	if cfg.Server.AccessLog || !cfg.Server.Metrics || cfg.Server.RateLimitRPS != 100 {
		t.Fatal("expected invalid server env values to be ignored")
	}
	if cfg.Database.SnapshotEvery != 25 || cfg.Database.SyncEvery != 1 || cfg.Database.ShardCount != 1 || cfg.Database.ShardFanout != 1 || cfg.Database.CacheEnabled || cfg.Database.CacheMaxBytes != 8<<20 || cfg.Database.CacheMaxItems != 1024 {
		t.Fatal("expected invalid database env values to be ignored")
	}
	if cfg.Limits.MaxK != 100 || cfg.Search.ANNM != 16 || cfg.Search.ANNEfConstruct != 64 || cfg.Search.ANNEfSearch != 64 || cfg.Search.ANNEvalSampleRate != 0 {
		t.Fatal("expected invalid env values to be ignored")
	}
	if cfg.GRPC.Enabled || cfg.Security.Auth.Enabled || cfg.Security.Auth.GRPCEnabled || cfg.Security.Transport.TLSEnabled || cfg.Security.Proxy.TrustForwardedFor || cfg.Security.Storage.StrictFilePermissions {
		t.Fatal("expected invalid bool env values to be ignored")
	}
}

func TestOverrideFromEnvValidValues(t *testing.T) {
	cfg := defaultConfig()
	t.Setenv("VECTOR_DB_READ_TIMEOUT", "9s")
	t.Setenv("VECTOR_DB_ACCESS_LOG", "true")
	t.Setenv("VECTOR_DB_METRICS_ENABLED", "false")
	t.Setenv("VECTOR_DB_PROTOCOL", "grpc")
	t.Setenv("VECTOR_DB_WRITE_TIMEOUT", "11s")
	t.Setenv("VECTOR_DB_RATE_LIMIT_RPS", "50")
	t.Setenv("VECTOR_DB_SNAPSHOT_PATH", "/tmp/snap")
	t.Setenv("VECTOR_DB_WAL_PATH", "/tmp/wal")
	t.Setenv("VECTOR_DB_SNAPSHOT_EVERY", "30")
	t.Setenv("VECTOR_DB_MAX_BODY_BYTES", "2048")
	t.Setenv("VECTOR_DB_MAX_VECTOR_DIM", "256")
	t.Setenv("VECTOR_DB_MAX_K", "20")
	t.Setenv("VECTOR_DB_SEARCH_MODE", "ann")
	t.Setenv("VECTOR_DB_ANN_PROFILE", "quality")
	t.Setenv("VECTOR_DB_ANN_M", "20")
	t.Setenv("VECTOR_DB_ANN_EF_CONSTRUCTION", "70")
	t.Setenv("VECTOR_DB_ANN_EF_SEARCH", "33")
	t.Setenv("VECTOR_DB_ANN_SEGMENT_ROUTING", "true")
	t.Setenv("VECTOR_DB_ANN_DIVERSIFIED_PRUNING", "true")
	t.Setenv("VECTOR_DB_ANN_METRIC_WARMUP", "cosine, inner_product")
	t.Setenv("VECTOR_DB_ANN_STAGED_INGEST_MIN_BATCH", "256")
	t.Setenv("VECTOR_DB_ANN_EVAL_SAMPLE_RATE", "25")
	t.Setenv("VECTOR_DB_CACHE_ENABLED", "true")
	t.Setenv("VECTOR_DB_CACHE_MAX_BYTES", "4096")
	t.Setenv("VECTOR_DB_CACHE_MAX_ITEMS", "222")
	t.Setenv("VECTOR_DB_CACHE_TTL", "45s")
	t.Setenv("VECTOR_DB_VECTOR_STORE", "disk")
	t.Setenv("VECTOR_DB_VECTOR_PATH", "/tmp/vectors")
	t.Setenv("VECTOR_DB_SYNC_EVERY", "32")
	t.Setenv("VECTOR_DB_SHARD_COUNT", "8")
	t.Setenv("VECTOR_DB_SHARD_FANOUT_CONCURRENCY", "3")
	t.Setenv("VECTOR_DB_GRPC_ENABLED", "true")
	t.Setenv("VECTOR_DB_CLUSTER_TOPOLOGY_PATH", "/tmp/topology.yaml")
	t.Setenv("VECTOR_DB_REPLICATION_MODE", "quorum")
	t.Setenv("VECTOR_DB_REPLICATION_QUEUE_SIZE", "200")
	t.Setenv("VECTOR_DB_WRITE_QUORUM", "3")
	t.Setenv("VECTOR_DB_GRPC_PORT", "22000")
	t.Setenv("VECTOR_DB_SECURITY_PROFILE", "production")
	t.Setenv("VECTOR_DB_SECURITY_AUTH_ENABLED", "true")
	t.Setenv("VECTOR_DB_SECURITY_API_KEY", "secret")
	t.Setenv("VECTOR_DB_SECURITY_GRPC_AUTH_ENABLED", "true")
	t.Setenv("VECTOR_DB_TLS_ENABLED", "true")
	t.Setenv("VECTOR_DB_TLS_CERT_FILE", "/tmp/cert.pem")
	t.Setenv("VECTOR_DB_TLS_KEY_FILE", "/tmp/key.pem")
	t.Setenv("VECTOR_DB_TRUST_FORWARDED_FOR", "true")
	t.Setenv("VECTOR_DB_TRUSTED_PROXIES", "10.0.0.1,10.0.0.2")
	t.Setenv("VECTOR_DB_STRICT_FILE_PERMISSIONS", "true")
	t.Setenv("VECTOR_DB_STORAGE_DIR_MODE", "0700")
	t.Setenv("VECTOR_DB_STORAGE_FILE_MODE", "0600")
	overrideFromEnv(&cfg)

	if cfg.Server.Protocol != "grpc" {
		t.Fatal("expected protocol override")
	}
	if cfg.Server.ReadTimeout != "9s" || cfg.Server.WriteTimeout != "11s" {
		t.Fatal("expected timeout overrides")
	}
	if !cfg.Server.AccessLog || cfg.Server.Metrics {
		t.Fatal("expected server boolean overrides")
	}
	if cfg.Server.RateLimitRPS != 50 || cfg.Database.SnapshotEvery != 30 {
		t.Fatal("expected numeric overrides")
	}
	if cfg.Database.SnapshotPath != "/tmp/snap" || cfg.Database.WALPath != "/tmp/wal" {
		t.Fatal("expected path overrides")
	}
	if cfg.Database.VectorStore != "disk" || cfg.Database.VectorPath != "/tmp/vectors" || cfg.Database.SyncEvery != 32 {
		t.Fatal("expected vector store overrides")
	}
	if cfg.Database.ShardCount != 8 || cfg.Database.ShardFanout != 3 {
		t.Fatal("expected shard env overrides")
	}
	if !cfg.Database.CacheEnabled || cfg.Database.CacheMaxBytes != 4096 || cfg.Database.CacheMaxItems != 222 || cfg.Database.CacheTTL != "45s" {
		t.Fatal("expected cache overrides")
	}
	if cfg.Limits.MaxBodyBytes != 2048 || cfg.Limits.MaxVectorDim != 256 || cfg.Limits.MaxK != 20 {
		t.Fatal("expected limit overrides")
	}
	if cfg.Search.Mode != "ann" {
		t.Fatal("expected search mode override")
	}
	if cfg.Search.ANNM != 20 || cfg.Search.ANNEfConstruct != 70 || cfg.Search.ANNEfSearch != 33 {
		t.Fatal("expected ann overrides")
	}
	if !cfg.Search.ANNSegmentRouting {
		t.Fatal("expected ANN segment routing override")
	}
	if !cfg.Search.ANNDiversifiedPruning {
		t.Fatal("expected ANN diversified pruning override")
	}
	if got := cfg.Search.ANNMetricWarmup; len(got) != 2 || got[0] != "cosine" || got[1] != "inner_product" {
		t.Fatalf("metric warmup=%v", got)
	}
	if cfg.Search.ANNStagedIngestMinBatch != 256 {
		t.Fatalf("staged ingest threshold = %d, want 256", cfg.Search.ANNStagedIngestMinBatch)
	}
	if cfg.Search.ANNProfile != "quality" {
		t.Fatal("expected ann profile override")
	}
	if cfg.Search.ANNEvalSampleRate != 25 {
		t.Fatal("expected ann eval sample rate override")
	}
	if !cfg.GRPC.Enabled || cfg.GRPC.Port != "22000" {
		t.Fatal("expected grpc overrides")
	}
	if cfg.Cluster.TopologyPath != "/tmp/topology.yaml" || cfg.Cluster.ReplicationMode != "quorum" || cfg.Cluster.QueueSize != 200 || cfg.Cluster.WriteQuorum != 3 {
		t.Fatal("expected cluster env overrides")
	}
	if cfg.Security.Profile != "production" || !cfg.Security.Auth.Enabled || cfg.Security.Auth.APIKey != "secret" || !cfg.Security.Auth.GRPCEnabled {
		t.Fatal("expected security auth overrides")
	}
	if !cfg.Security.Transport.TLSEnabled || cfg.Security.Transport.CertFile != "/tmp/cert.pem" || cfg.Security.Transport.KeyFile != "/tmp/key.pem" {
		t.Fatal("expected tls overrides")
	}
	if !cfg.Security.Proxy.TrustForwardedFor || len(cfg.Security.Proxy.TrustedProxies) != 2 {
		t.Fatal("expected proxy overrides")
	}
	if !cfg.Security.Storage.StrictFilePermissions || cfg.Security.Storage.DirMode != "0700" || cfg.Security.Storage.FileMode != "0600" {
		t.Fatal("expected storage overrides")
	}
}

func TestOverrideFromEnvAllowsDisablingRateLimit(t *testing.T) {
	cfg := defaultConfig()
	t.Setenv("VECTOR_DB_RATE_LIMIT_RPS", "0")
	overrideFromEnv(&cfg)
	if cfg.Server.RateLimitRPS != 0 {
		t.Fatalf("expected rate limit to be disabled, got %d", cfg.Server.RateLimitRPS)
	}
}

func TestInvalidProtocolFallsBackToHTTP(t *testing.T) {
	cfg := defaultConfig()
	t.Setenv("VECTOR_DB_PROTOCOL", "invalid")
	overrideFromEnv(&cfg)

	if cfg.Server.Protocol != "http" {
		t.Fatal("expected invalid protocol to fall back to http")
	}
	if cfg.GRPC.Enabled {
		t.Fatal("expected grpc disabled when protocol falls back to http")
	}
}

func TestANNProfileDefaultsAndPartialOverride(t *testing.T) {
	cfgPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(cfgPath, []byte(`
search:
  mode: "ann"
  ann_profile: "fast"
  ann_ef_search: 40
`), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(cfgPath)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.Search.ANNProfile != "fast" {
		t.Fatal("expected fast ann profile")
	}
	if cfg.Search.ANNM != 8 || cfg.Search.ANNEfConstruct != 32 || cfg.Search.ANNEfSearch != 40 {
		t.Fatal("expected fast profile defaults with explicit ef_search override")
	}
}

func TestANNAdaptiveEnvironmentConfiguration(t *testing.T) {
	t.Setenv("VECTOR_DB_ANN_ADAPTIVE", "true")
	t.Setenv("VECTOR_DB_ANN_MIN_CANDIDATES", "40")
	t.Setenv("VECTOR_DB_ANN_MAX_PROBE", "64")
	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.Search.ANNAdaptive || cfg.Search.ANNMinCandidates != 40 || cfg.Search.ANNMaxProbe != 64 {
		t.Fatalf("adaptive config=%+v", cfg.Search)
	}
}

func TestANNQualityProfileEnablesAdaptiveIVF(t *testing.T) {
	t.Setenv("VECTOR_DB_ANN_BACKEND", "ivf")
	t.Setenv("VECTOR_DB_ANN_PROFILE", "quality")
	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.Search.ANNAdaptive || cfg.Search.ANNMinCandidates != 2000 {
		t.Fatalf("quality IVF config=%+v", cfg.Search)
	}
}

func TestSecurityProfileDefaults(t *testing.T) {
	cfg := defaultConfig()
	cfg.Security.Profile = "production"
	applySecurityDefaults(&cfg)

	if !cfg.Security.Storage.StrictFilePermissions {
		t.Fatal("expected strict file permissions in production")
	}
	if cfg.Security.Storage.DirMode != "0700" || cfg.Security.Storage.FileMode != "0600" {
		t.Fatal("expected strict production file modes")
	}

	cfg = defaultConfig()
	cfg.Server.APIKey = "legacy-secret"
	cfg.Security.Profile = "production"
	applySecurityDefaults(&cfg)
	if !cfg.Security.Auth.Enabled || !cfg.Security.Auth.GRPCEnabled || cfg.Security.Auth.APIKey != "legacy-secret" {
		t.Fatal("expected legacy server api key to propagate into security auth")
	}

	cfg = defaultConfig()
	cfg.Security.Auth.GRPCEnabled = true
	cfg.Security.Storage.DirMode = ""
	cfg.Security.Storage.FileMode = ""
	applySecurityDefaults(&cfg)
	if cfg.Security.Auth.GRPCEnabled {
		t.Fatal("expected grpc auth disabled when auth is disabled in development")
	}
	if cfg.Security.Storage.DirMode != "0755" || cfg.Security.Storage.FileMode != "0644" {
		t.Fatal("expected development storage defaults for empty modes")
	}
}

func TestProductionSecurityDefaultsWithoutTrustedProxies(t *testing.T) {
	cfg := defaultConfig()
	cfg.Security.Profile = "production"
	cfg.Security.Auth.APIKey = "secret"
	cfg.Security.Proxy.TrustForwardedFor = true
	cfg.Security.Proxy.TrustedProxies = nil
	cfg.Security.Storage.DirMode = ""
	cfg.Security.Storage.FileMode = ""

	applySecurityDefaults(&cfg)

	if !cfg.Security.Auth.Enabled || !cfg.Security.Auth.GRPCEnabled {
		t.Fatal("expected auth enabled from production api key")
	}
	if cfg.Security.Proxy.TrustForwardedFor {
		t.Fatal("expected forwarded-for trust disabled without trusted proxies")
	}
	if cfg.Security.Storage.DirMode != "0700" || cfg.Security.Storage.FileMode != "0600" {
		t.Fatal("expected strict production defaults for empty modes")
	}
}

func TestLoadANNQuantizeSegmentsFromFileAndEnvironment(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte("search:\n  ann_quantize_segments: false\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VECTOR_DB_ANN_QUANTIZE_SEGMENTS", "true")
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.Search.ANNQuantizeSegments {
		t.Fatal("expected environment to enable sealed-segment quantization")
	}
}

func TestLoadLocationIndexCapacityFromEnvironment(t *testing.T) {
	t.Setenv("VECTOR_DB_LOCATION_INDEX_CAPACITY", "750000")
	cfg, err := Load(filepath.Join(t.TempDir(), "missing.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Database.LocationIndexCapacity != 750000 {
		t.Fatalf("location index capacity = %d", cfg.Database.LocationIndexCapacity)
	}
}

func TestLoadIDIndexCapacityFromEnvironment(t *testing.T) {
	t.Setenv("VECTOR_DB_ID_INDEX_CAPACITY", "900000")
	cfg, err := Load(filepath.Join(t.TempDir(), "missing.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Database.IDIndexCapacity != 900000 {
		t.Fatalf("ID index capacity = %d", cfg.Database.IDIndexCapacity)
	}
}
