package config

import (
	"os"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server struct {
		Protocol     string `yaml:"protocol"`
		Port         string `yaml:"port"`
		ReadTimeout  string `yaml:"read_timeout"`
		WriteTimeout string `yaml:"write_timeout"`
		APIKey       string `yaml:"api_key"`
		AccessLog    bool   `yaml:"access_log"`
		AuditLogPath string `yaml:"audit_log_path"`
		Metrics      bool   `yaml:"metrics_enabled"`
		RateLimitRPS int    `yaml:"rate_limit_rps"`
	} `yaml:"server"`
	Database struct {
		SnapshotPath          string   `yaml:"snapshot_path"`
		WALPath               string   `yaml:"wal_path"`
		SnapshotEvery         int      `yaml:"snapshot_every"`
		VectorStore           string   `yaml:"vector_store"`
		VectorPath            string   `yaml:"vector_path"`
		LocationIndexCapacity uint64   `yaml:"location_index_capacity"`
		IDIndexCapacity       uint64   `yaml:"id_index_capacity"`
		SyncEvery             int      `yaml:"sync_every"`
		ShardCount            int      `yaml:"shard_count"`
		ShardFanout           int      `yaml:"shard_fanout_concurrency"`
		ShardRouting          string   `yaml:"shard_routing"`
		ShardIDs              []string `yaml:"shard_ids"`
		CacheEnabled          bool     `yaml:"cache_enabled"`
		CacheMaxBytes         int64    `yaml:"cache_max_bytes"`
		CacheMaxItems         int      `yaml:"cache_max_items"`
		CacheTTL              string   `yaml:"cache_ttl"`
	} `yaml:"database"`
	Limits struct {
		MaxBodyBytes int64 `yaml:"max_body_bytes"`
		MaxVectorDim int   `yaml:"max_vector_dim"`
		MaxK         int   `yaml:"max_k"`
	} `yaml:"limits"`
	Search struct {
		Mode                    string   `yaml:"mode"`
		ANNBackend              string   `yaml:"ann_backend"`
		IVFCentroids            int      `yaml:"ivf_centroids"`
		IVFNProbe               int      `yaml:"ivf_nprobe"`
		ANNProfile              string   `yaml:"ann_profile"`
		ANNM                    int      `yaml:"ann_m"`
		ANNEfConstruct          int      `yaml:"ann_ef_construction"`
		ANNEfSearch             int      `yaml:"ann_ef_search"`
		ANNSegmentMaxNodes      int      `yaml:"ann_segment_max_nodes"`
		ANNStagedIngestMinBatch int      `yaml:"ann_staged_ingest_min_batch"`
		ANNQuantizeSegments     bool     `yaml:"ann_quantize_segments"`
		ANNSegmentRouting       bool     `yaml:"ann_segment_routing"`
		ANNDiversifiedPruning   bool     `yaml:"ann_diversified_pruning"`
		ANNMetricWarmup         []string `yaml:"ann_metric_warmup"`
		ANNEvalSampleRate       int      `yaml:"ann_eval_sample_rate"`
		ANNAdaptive             bool     `yaml:"ann_adaptive"`
		ANNMinCandidates        int      `yaml:"ann_min_candidates"`
		ANNMaxProbe             int      `yaml:"ann_max_probe"`
	} `yaml:"search"`
	GRPC struct {
		Enabled bool   `yaml:"enabled"`
		Port    string `yaml:"port"`
	} `yaml:"grpc"`
	Cluster struct {
		TopologyPath       string   `yaml:"topology_path"`
		NodeID             string   `yaml:"node_id"`
		Members            []string `yaml:"members"`
		StateDir           string   `yaml:"state_dir"`
		ReplicationMode    string   `yaml:"replication_mode"`
		QueueSize          int      `yaml:"replication_queue_size"`
		WriteQuorum        int      `yaml:"write_quorum"`
		ReplicationLogPath string   `yaml:"replication_log_path"`
		FencePath          string   `yaml:"fence_path"`
		HeartbeatInterval  string   `yaml:"heartbeat_interval"`
		ElectionTimeout    string   `yaml:"election_timeout"`
	} `yaml:"cluster"`
	Security struct {
		Profile string `yaml:"profile"`
		Auth    struct {
			Enabled     bool   `yaml:"enabled"`
			APIKey      string `yaml:"api_key"`
			GRPCEnabled bool   `yaml:"grpc_enabled"`
		} `yaml:"auth"`
		Transport struct {
			TLSEnabled   bool   `yaml:"tls_enabled"`
			CertFile     string `yaml:"cert_file"`
			KeyFile      string `yaml:"key_file"`
			ClientCAFile string `yaml:"client_ca_file"`
		} `yaml:"transport"`
		Proxy struct {
			TrustForwardedFor bool     `yaml:"trust_forwarded_for"`
			TrustedProxies    []string `yaml:"trusted_proxies"`
		} `yaml:"proxy"`
		Storage struct {
			StrictFilePermissions bool   `yaml:"strict_file_permissions"`
			DirMode               string `yaml:"dir_mode"`
			FileMode              string `yaml:"file_mode"`
		} `yaml:"storage"`
	} `yaml:"security"`
	Admin struct {
		Enabled    bool `yaml:"enabled"`
		Kubernetes struct {
			Enabled     bool   `yaml:"enabled"`
			Namespace   string `yaml:"namespace"`
			ClusterName string `yaml:"cluster_name"`
		} `yaml:"kubernetes"`
	} `yaml:"admin"`
	Metering struct {
		Enabled          bool   `yaml:"enabled"`
		Endpoint         string `yaml:"endpoint"`
		OutboxPath       string `yaml:"outbox_path"`
		ClientCertFile   string `yaml:"client_cert_file"`
		ClientKeyFile    string `yaml:"client_key_file"`
		ServerCAFile     string `yaml:"server_ca_file"`
		Window           string `yaml:"window"`
		RetryMin         string `yaml:"retry_min"`
		RetryMax         string `yaml:"retry_max"`
		MaxPending       int    `yaml:"max_pending_batches"`
		MaxDeliveryBatch int    `yaml:"max_delivery_batch"`
	} `yaml:"metering"`
}

func Load(path string) (Config, error) {
	cfg := defaultConfig()

	data, err := os.ReadFile(path)
	if err == nil {
		if err := yaml.Unmarshal(data, &cfg); err != nil {
			return Config{}, err
		}
	}

	overrideFromEnv(&cfg)
	return cfg, nil
}

func ParseDuration(raw string, fallback time.Duration) time.Duration {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return fallback
	}
	d, err := time.ParseDuration(raw)
	if err != nil {
		return fallback
	}
	return d
}

func defaultConfig() Config {
	var cfg Config
	cfg.Server.Protocol = "http"
	cfg.Server.Port = "19190"
	cfg.Server.ReadTimeout = "10s"
	cfg.Server.WriteTimeout = "10s"
	cfg.Server.APIKey = ""
	cfg.Server.AccessLog = false
	cfg.Server.Metrics = true
	cfg.Server.RateLimitRPS = 100
	cfg.Database.SnapshotPath = "./data/snapshot.json"
	cfg.Database.WALPath = "./data/wal.log"
	cfg.Database.SnapshotEvery = 25
	cfg.Database.VectorStore = "memory"
	cfg.Database.VectorPath = "./data/vectors"
	cfg.Database.SyncEvery = 1
	cfg.Database.ShardCount = 1
	cfg.Database.ShardFanout = 1
	cfg.Database.ShardRouting = "modulo-v1"
	cfg.Database.CacheEnabled = false
	cfg.Database.CacheMaxBytes = 8 << 20
	cfg.Database.CacheMaxItems = 1024
	cfg.Database.CacheTTL = "15m"
	cfg.Limits.MaxBodyBytes = 1 << 20
	cfg.Limits.MaxVectorDim = 4096
	cfg.Limits.MaxK = 100
	cfg.Search.Mode = "exact"
	cfg.Search.ANNProfile = "balanced"
	cfg.Search.ANNSegmentMaxNodes = 10000
	cfg.Search.ANNEvalSampleRate = 0
	cfg.GRPC.Enabled = false
	cfg.GRPC.Port = "19191"
	cfg.Cluster.ReplicationMode = "async"
	cfg.Cluster.QueueSize = 1024
	cfg.Cluster.WriteQuorum = 1
	cfg.Cluster.ReplicationLogPath = "./data/replication"
	cfg.Cluster.FencePath = "./data/fencing"
	cfg.Cluster.HeartbeatInterval = "1s"
	cfg.Cluster.ElectionTimeout = "5s"
	cfg.Security.Profile = "development"
	cfg.Security.Auth.Enabled = false
	cfg.Security.Auth.APIKey = ""
	cfg.Security.Auth.GRPCEnabled = false
	cfg.Security.Transport.TLSEnabled = false
	cfg.Security.Transport.CertFile = ""
	cfg.Security.Transport.KeyFile = ""
	cfg.Security.Proxy.TrustForwardedFor = false
	cfg.Security.Proxy.TrustedProxies = nil
	cfg.Security.Storage.StrictFilePermissions = false
	cfg.Security.Storage.DirMode = "0755"
	cfg.Security.Storage.FileMode = "0644"
	cfg.Admin.Enabled = false
	cfg.Admin.Kubernetes.Enabled = false
	cfg.Admin.Kubernetes.Namespace = "default"
	cfg.Metering.Window = "1m"
	cfg.Metering.RetryMin = "1s"
	cfg.Metering.RetryMax = "30s"
	cfg.Metering.MaxPending = 10000
	cfg.Metering.MaxDeliveryBatch = 100
	return cfg
}

func overrideFromEnv(cfg *Config) {
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_METERING_ENABLED")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Metering.Enabled = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_METERING_ENDPOINT")); v != "" {
		cfg.Metering.Endpoint = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_METERING_OUTBOX_PATH")); v != "" {
		cfg.Metering.OutboxPath = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_METERING_CLIENT_CERT_FILE")); v != "" {
		cfg.Metering.ClientCertFile = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_METERING_CLIENT_KEY_FILE")); v != "" {
		cfg.Metering.ClientKeyFile = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_METERING_SERVER_CA_FILE")); v != "" {
		cfg.Metering.ServerCAFile = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_METERING_WINDOW")); v != "" {
		cfg.Metering.Window = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_METERING_RETRY_MIN")); v != "" {
		cfg.Metering.RetryMin = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_METERING_RETRY_MAX")); v != "" {
		cfg.Metering.RetryMax = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_METERING_MAX_PENDING_BATCHES")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Metering.MaxPending = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_METERING_MAX_DELIVERY_BATCH")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Metering.MaxDeliveryBatch = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ADMIN_ENABLED")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Admin.Enabled = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ADMIN_KUBERNETES_ENABLED")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Admin.Kubernetes.Enabled = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ADMIN_KUBERNETES_NAMESPACE")); v != "" {
		cfg.Admin.Kubernetes.Namespace = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ADMIN_KUBERNETES_CLUSTER_NAME")); v != "" {
		cfg.Admin.Kubernetes.ClusterName = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_PROTOCOL")); v != "" {
		cfg.Server.Protocol = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_PORT")); v != "" {
		cfg.Server.Port = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_READ_TIMEOUT")); v != "" {
		cfg.Server.ReadTimeout = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_WRITE_TIMEOUT")); v != "" {
		cfg.Server.WriteTimeout = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_API_KEY")); v != "" {
		cfg.Server.APIKey = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ACCESS_LOG")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Server.AccessLog = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_METRICS_ENABLED")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Server.Metrics = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_RATE_LIMIT_RPS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			cfg.Server.RateLimitRPS = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_SNAPSHOT_PATH")); v != "" {
		cfg.Database.SnapshotPath = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_WAL_PATH")); v != "" {
		cfg.Database.WALPath = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_SNAPSHOT_EVERY")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Database.SnapshotEvery = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_VECTOR_STORE")); v != "" {
		cfg.Database.VectorStore = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_VECTOR_PATH")); v != "" {
		cfg.Database.VectorPath = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_LOCATION_INDEX_CAPACITY")); v != "" {
		if n, err := strconv.ParseUint(v, 10, 64); err == nil {
			cfg.Database.LocationIndexCapacity = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ID_INDEX_CAPACITY")); v != "" {
		if n, err := strconv.ParseUint(v, 10, 64); err == nil {
			cfg.Database.IDIndexCapacity = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_SYNC_EVERY")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Database.SyncEvery = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_SHARD_COUNT")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Database.ShardCount = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_SHARD_FANOUT_CONCURRENCY")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Database.ShardFanout = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_SHARD_ROUTING")); v != "" {
		cfg.Database.ShardRouting = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_SHARD_IDS")); v != "" {
		raw := strings.Split(v, ",")
		cfg.Database.ShardIDs = cfg.Database.ShardIDs[:0]
		for _, id := range raw {
			if id = strings.TrimSpace(id); id != "" {
				cfg.Database.ShardIDs = append(cfg.Database.ShardIDs, id)
			}
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_CACHE_ENABLED")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Database.CacheEnabled = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_CACHE_MAX_BYTES")); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
			cfg.Database.CacheMaxBytes = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_CACHE_MAX_ITEMS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Database.CacheMaxItems = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_CACHE_TTL")); v != "" {
		cfg.Database.CacheTTL = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_MAX_BODY_BYTES")); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
			cfg.Limits.MaxBodyBytes = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_MAX_VECTOR_DIM")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Limits.MaxVectorDim = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_MAX_K")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Limits.MaxK = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_SEARCH_MODE")); v != "" {
		cfg.Search.Mode = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ANN_PROFILE")); v != "" {
		cfg.Search.ANNProfile = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ANN_BACKEND")); v != "" {
		cfg.Search.ANNBackend = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ANN_QUANTIZE_SEGMENTS")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Search.ANNQuantizeSegments = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ANN_SEGMENT_ROUTING")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Search.ANNSegmentRouting = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ANN_DIVERSIFIED_PRUNING")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Search.ANNDiversifiedPruning = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ANN_METRIC_WARMUP")); v != "" {
		cfg.Search.ANNMetricWarmup = splitCSV(v)
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_IVF_CENTROIDS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Search.IVFCentroids = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_IVF_NPROBE")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Search.IVFNProbe = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ANN_M")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Search.ANNM = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ANN_EF_CONSTRUCTION")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Search.ANNEfConstruct = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ANN_EF_SEARCH")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Search.ANNEfSearch = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ANN_SEGMENT_MAX_NODES")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Search.ANNSegmentMaxNodes = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ANN_STAGED_INGEST_MIN_BATCH")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			cfg.Search.ANNStagedIngestMinBatch = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ANN_EVAL_SAMPLE_RATE")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			cfg.Search.ANNEvalSampleRate = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ANN_ADAPTIVE")); v != "" {
		cfg.Search.ANNAdaptive = strings.EqualFold(v, "1") || strings.EqualFold(v, "true") || strings.EqualFold(v, "yes")
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ANN_MIN_CANDIDATES")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			cfg.Search.ANNMinCandidates = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_ANN_MAX_PROBE")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			cfg.Search.ANNMaxProbe = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_GRPC_ENABLED")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.GRPC.Enabled = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_CLUSTER_TOPOLOGY_PATH")); v != "" {
		cfg.Cluster.TopologyPath = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_CLUSTER_NODE_ID")); v != "" {
		cfg.Cluster.NodeID = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_CLUSTER_MEMBERS")); v != "" {
		members := strings.Split(v, ",")
		cfg.Cluster.Members = cfg.Cluster.Members[:0]
		for _, member := range members {
			if member = strings.TrimSpace(member); member != "" {
				cfg.Cluster.Members = append(cfg.Cluster.Members, member)
			}
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_CLUSTER_STATE_DIR")); v != "" {
		cfg.Cluster.StateDir = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_CLUSTER_HEARTBEAT_INTERVAL")); v != "" {
		cfg.Cluster.HeartbeatInterval = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_CLUSTER_ELECTION_TIMEOUT")); v != "" {
		cfg.Cluster.ElectionTimeout = v
	}
	if v := strings.ToLower(strings.TrimSpace(os.Getenv("VECTOR_DB_REPLICATION_MODE"))); v == "async" || v == "eventual" || v == "quorum" || v == "strong" {
		cfg.Cluster.ReplicationMode = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_REPLICATION_QUEUE_SIZE")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Cluster.QueueSize = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_WRITE_QUORUM")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Cluster.WriteQuorum = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_GRPC_PORT")); v != "" {
		cfg.GRPC.Port = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_SECURITY_PROFILE")); v != "" {
		cfg.Security.Profile = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_SECURITY_AUTH_ENABLED")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Security.Auth.Enabled = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_SECURITY_API_KEY")); v != "" {
		cfg.Security.Auth.APIKey = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_SECURITY_GRPC_AUTH_ENABLED")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Security.Auth.GRPCEnabled = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_TLS_ENABLED")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Security.Transport.TLSEnabled = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_TLS_CERT_FILE")); v != "" {
		cfg.Security.Transport.CertFile = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_TLS_KEY_FILE")); v != "" {
		cfg.Security.Transport.KeyFile = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_TLS_CLIENT_CA_FILE")); v != "" {
		cfg.Security.Transport.ClientCAFile = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_TRUST_FORWARDED_FOR")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Security.Proxy.TrustForwardedFor = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_TRUSTED_PROXIES")); v != "" {
		parts := strings.Split(v, ",")
		cfg.Security.Proxy.TrustedProxies = cfg.Security.Proxy.TrustedProxies[:0]
		for _, part := range parts {
			part = strings.TrimSpace(part)
			if part != "" {
				cfg.Security.Proxy.TrustedProxies = append(cfg.Security.Proxy.TrustedProxies, part)
			}
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_STRICT_FILE_PERMISSIONS")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Security.Storage.StrictFilePermissions = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_STORAGE_DIR_MODE")); v != "" {
		cfg.Security.Storage.DirMode = v
	}
	if v := strings.TrimSpace(os.Getenv("VECTOR_DB_STORAGE_FILE_MODE")); v != "" {
		cfg.Security.Storage.FileMode = v
	}
	applyTransportDefaults(cfg)
	applyANNProfileDefaults(cfg)
	applySecurityDefaults(cfg)
}

func applyTransportDefaults(cfg *Config) {
	cfg.Server.Protocol = normalizeProtocol(cfg.Server.Protocol)
	cfg.GRPC.Enabled = cfg.Server.Protocol == "grpc"
}

func normalizeProtocol(protocol string) string {
	switch strings.ToLower(strings.TrimSpace(protocol)) {
	case "grpc":
		return "grpc"
	default:
		return "http"
	}
}

func applyANNProfileDefaults(cfg *Config) {
	if strings.TrimSpace(cfg.Search.ANNBackend) == "" {
		cfg.Search.ANNBackend = "hierarchical-hnsw"
	} else {
		cfg.Search.ANNBackend = strings.ToLower(strings.TrimSpace(cfg.Search.ANNBackend))
		if cfg.Search.ANNBackend != "hnsw" && cfg.Search.ANNBackend != "hierarchical-hnsw" && cfg.Search.ANNBackend != "ivf" {
			cfg.Search.ANNBackend = "hierarchical-hnsw"
		}
	}
	if cfg.Search.IVFCentroids <= 0 {
		cfg.Search.IVFCentroids = 64
	}
	if cfg.Search.IVFNProbe <= 0 {
		cfg.Search.IVFNProbe = 4
	}
	profile := normalizeANNProfile(cfg.Search.ANNProfile)
	cfg.Search.ANNProfile = profile
	if cfg.Search.ANNBackend == "ivf" && !cfg.Search.ANNAdaptive {
		switch profile {
		case "balanced":
			cfg.Search.ANNAdaptive = true
			if cfg.Search.ANNMinCandidates <= 0 {
				cfg.Search.ANNMinCandidates = 500
			}
		case "quality":
			cfg.Search.ANNAdaptive = true
			if cfg.Search.ANNMinCandidates <= 0 {
				cfg.Search.ANNMinCandidates = 2000
			}
		}
	}

	m, efConstruct, efSearch := annProfileDefaults(profile)
	if cfg.Search.ANNM <= 0 {
		cfg.Search.ANNM = m
	}
	if cfg.Search.ANNEfConstruct <= 0 {
		cfg.Search.ANNEfConstruct = efConstruct
	}
	if cfg.Search.ANNEfSearch <= 0 {
		cfg.Search.ANNEfSearch = efSearch
	}
	if cfg.Search.ANNSegmentMaxNodes <= 0 {
		cfg.Search.ANNSegmentMaxNodes = 10000
	}
}

func applySecurityDefaults(cfg *Config) {
	profile := normalizeSecurityProfile(cfg.Security.Profile)
	cfg.Security.Profile = profile

	if strings.TrimSpace(cfg.Security.Auth.APIKey) == "" && strings.TrimSpace(cfg.Server.APIKey) != "" {
		cfg.Security.Auth.APIKey = cfg.Server.APIKey
	}

	switch profile {
	case "production":
		if strings.TrimSpace(cfg.Security.Auth.APIKey) != "" {
			cfg.Security.Auth.Enabled = true
		}
		cfg.Security.Auth.GRPCEnabled = cfg.Security.Auth.Enabled
		cfg.Security.Proxy.TrustForwardedFor = cfg.Security.Proxy.TrustForwardedFor && len(cfg.Security.Proxy.TrustedProxies) > 0
		cfg.Security.Storage.StrictFilePermissions = true
		if strings.TrimSpace(cfg.Security.Storage.DirMode) == "" || cfg.Security.Storage.DirMode == "0755" {
			cfg.Security.Storage.DirMode = "0700"
		}
		if strings.TrimSpace(cfg.Security.Storage.FileMode) == "" || cfg.Security.Storage.FileMode == "0644" {
			cfg.Security.Storage.FileMode = "0600"
		}
	default:
		if !cfg.Security.Auth.Enabled {
			cfg.Security.Auth.GRPCEnabled = false
		}
		if strings.TrimSpace(cfg.Security.Storage.DirMode) == "" {
			cfg.Security.Storage.DirMode = "0755"
		}
		if strings.TrimSpace(cfg.Security.Storage.FileMode) == "" {
			cfg.Security.Storage.FileMode = "0644"
		}
	}
}

func normalizeSecurityProfile(profile string) string {
	switch strings.ToLower(strings.TrimSpace(profile)) {
	case "production":
		return "production"
	default:
		return "development"
	}
}

func normalizeANNProfile(profile string) string {
	switch strings.ToLower(strings.TrimSpace(profile)) {
	case "fast":
		return "fast"
	case "quality":
		return "quality"
	default:
		return "balanced"
	}
}

func annProfileDefaults(profile string) (m, efConstruct, efSearch int) {
	switch normalizeANNProfile(profile) {
	case "fast":
		return 8, 32, 32
	case "quality":
		return 24, 96, 96
	default:
		return 16, 64, 64
	}
}

func splitCSV(value string) []string {
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}
