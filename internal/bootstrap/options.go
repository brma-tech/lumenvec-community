// Package bootstrap maps shared configuration into the Community API runtime.
// Edition-specific commands may extend the returned options with adapters.
package bootstrap

import (
	"errors"
	"strings"
	"time"

	"lumenvec/internal/api"
	"lumenvec/internal/config"
)

var ErrBusinessFeaturesRequired = errors.New(
	"distributed cluster configuration requires LumenVec Business",
)

func ServerOptions(cfg config.Config) api.ServerOptions {
	return api.ServerOptions{
		Protocol:                     cfg.Server.Protocol,
		Port:                         cfg.Server.Port,
		ReadTimeout:                  config.ParseDuration(cfg.Server.ReadTimeout, 10*time.Second),
		WriteTimeout:                 config.ParseDuration(cfg.Server.WriteTimeout, 10*time.Second),
		MaxBodyBytes:                 cfg.Limits.MaxBodyBytes,
		MaxVectorDim:                 cfg.Limits.MaxVectorDim,
		MaxK:                         cfg.Limits.MaxK,
		SnapshotPath:                 cfg.Database.SnapshotPath,
		WALPath:                      cfg.Database.WALPath,
		SnapshotEvery:                cfg.Database.SnapshotEvery,
		VectorStore:                  cfg.Database.VectorStore,
		VectorPath:                   cfg.Database.VectorPath,
		SyncEvery:                    cfg.Database.SyncEvery,
		ShardCount:                   cfg.Database.ShardCount,
		ShardFanout:                  cfg.Database.ShardFanout,
		ShardRouting:                 cfg.Database.ShardRouting,
		ShardIDs:                     cfg.Database.ShardIDs,
		ClusterTopologyPath:          cfg.Cluster.TopologyPath,
		ReplicationMode:              cfg.Cluster.ReplicationMode,
		ReplicationQueueSize:         cfg.Cluster.QueueSize,
		ReplicationWriteQuorum:       cfg.Cluster.WriteQuorum,
		ReplicationLogPath:           cfg.Cluster.ReplicationLogPath,
		ReplicationFencePath:         cfg.Cluster.FencePath,
		ReplicationHeartbeatInterval: config.ParseDuration(cfg.Cluster.HeartbeatInterval, time.Second),
		ReplicationElectionTimeout:   config.ParseDuration(cfg.Cluster.ElectionTimeout, 5*time.Second),
		ClusterNodeID:                cfg.Cluster.NodeID,
		ClusterMembers:               cfg.Cluster.Members,
		ClusterStateDir:              cfg.Cluster.StateDir,
		APIKey:                       cfg.Server.APIKey,
		MetricsEnabled:               cfg.Server.Metrics,
		DisableRateLimit:             cfg.Server.RateLimitRPS == 0,
		AccessLogEnabled:             cfg.Server.AccessLog,
		AuditLogPath:                 cfg.Server.AuditLogPath,
		RateLimitRPS:                 cfg.Server.RateLimitRPS,
		SearchMode:                   cfg.Search.Mode,
		ANNBackend:                   cfg.Search.ANNBackend,
		IVFCentroids:                 cfg.Search.IVFCentroids,
		IVFNProbe:                    cfg.Search.IVFNProbe,
		ANNProfile:                   cfg.Search.ANNProfile,
		ANNM:                         cfg.Search.ANNM,
		ANNEfConstruct:               cfg.Search.ANNEfConstruct,
		ANNEfSearch:                  cfg.Search.ANNEfSearch,
		ANNSegmentMaxNodes:           cfg.Search.ANNSegmentMaxNodes,
		ANNStagedIngestMinBatch:      cfg.Search.ANNStagedIngestMinBatch,
		ANNQuantizeSegments:          cfg.Search.ANNQuantizeSegments,
		ANNSegmentRouting:            cfg.Search.ANNSegmentRouting,
		ANNDiversifiedPruning:        cfg.Search.ANNDiversifiedPruning,
		ANNMetricWarmup:              cfg.Search.ANNMetricWarmup,
		ANNEvalSampleRate:            cfg.Search.ANNEvalSampleRate,
		ANNAdaptive:                  cfg.Search.ANNAdaptive,
		ANNMinCandidates:             cfg.Search.ANNMinCandidates,
		ANNMaxProbe:                  cfg.Search.ANNMaxProbe,
		CacheEnabled:                 cfg.Database.CacheEnabled,
		CacheMaxBytes:                cfg.Database.CacheMaxBytes,
		CacheMaxItems:                cfg.Database.CacheMaxItems,
		CacheTTL:                     config.ParseDuration(cfg.Database.CacheTTL, 15*time.Minute),
		LocationIndexCapacity:        cfg.Database.LocationIndexCapacity,
		IDIndexCapacity:              cfg.Database.IDIndexCapacity,
		GRPCEnabled:                  cfg.GRPC.Enabled,
		GRPCPort:                     cfg.GRPC.Port,
		SecurityProfile:              cfg.Security.Profile,
		AuthEnabled:                  cfg.Security.Auth.Enabled,
		AuthAPIKey:                   cfg.Security.Auth.APIKey,
		GRPCAuthEnabled:              cfg.Security.Auth.GRPCEnabled,
		TLSEnabled:                   cfg.Security.Transport.TLSEnabled,
		TLSCertFile:                  cfg.Security.Transport.CertFile,
		TLSKeyFile:                   cfg.Security.Transport.KeyFile,
		TLSClientCAFile:              cfg.Security.Transport.ClientCAFile,
		TrustForwardedFor:            cfg.Security.Proxy.TrustForwardedFor,
		TrustedProxies:               cfg.Security.Proxy.TrustedProxies,
		StrictFilePerms:              cfg.Security.Storage.StrictFilePermissions,
		StorageDirMode:               cfg.Security.Storage.DirMode,
		StorageFileMode:              cfg.Security.Storage.FileMode,
		AdminEnabled:                 cfg.Admin.Enabled,
	}
}

func CommunityServerOptions(cfg config.Config) (api.ServerOptions, error) {
	options := ServerOptions(cfg)
	if strings.TrimSpace(options.ClusterTopologyPath) != "" ||
		strings.TrimSpace(options.ClusterNodeID) != "" ||
		len(options.ClusterMembers) > 0 {
		return api.ServerOptions{}, ErrBusinessFeaturesRequired
	}
	return options, nil
}
