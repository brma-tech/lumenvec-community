package api

import (
	"context"
	"crypto/subtle"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"lumenvec/internal/core"
	"lumenvec/internal/edition"
	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	listenAndServeFunc    = func(server *http.Server) error { return server.ListenAndServe() }
	listenAndServeTLSFunc = func(server *http.Server, certFile, keyFile string) error {
		return server.ListenAndServeTLS(certFile, keyFile)
	}
	logFatalfAPI = log.Fatalf
	logPrintfAPI = log.Printf
)

type Server struct {
	router            http.Handler
	protocol          string
	port              string
	grpcPort          string
	grpcEnabled       bool
	readTimeout       time.Duration
	writeTimeout      time.Duration
	service           core.VectorService
	election          edition.ElectionCoordinator
	electionTransport edition.Closer
	consensusLog      edition.ConsensusLog
	consensusLogs     edition.ConsensusLogProvider
	maxBodyBytes      int64
	listVectorsLimit  int
	apiKey            string
	apiKeyMu          sync.RWMutex
	secretCache       SecretCache
	authEnabled       bool
	grpcAuth          bool
	tlsEnabled        bool
	tlsCertFile       string
	tlsKeyFile        string
	tlsClientCAFile   string
	accessLog         bool
	auditLogPath      string
	auditLogMu        sync.Mutex
	metrics           bool
	snapshotPath      string
	walPath           string
	vectorStore       string
	vectorPath        string
	trustXFF          bool
	trustedCIDRs      []netip.Prefix
	rateLimiter       *rateLimiter
	adminEnabled      bool
	adminStartedAt    time.Time
	adminConfig       AdminConfigController
	adminCluster      AdminClusterController
	usageObserver     UsageObserver

	requestTotal    *prometheus.CounterVec
	requestDuration *prometheus.HistogramVec
	metricsRegistry *prometheus.Registry
}

// UsageObserver receives successful billable operations at the transport
// boundary. Implementations must be non-blocking.
type UsageObserver interface {
	ObserveIngestedVectors(uint64)
	ObserveSearchRequests(uint64)
}

// RotateAPIKey atomically replaces the runtime API key. An empty key disables
// authentication only when the server was configured without auth; callers
// should coordinate policy changes with deployment configuration.
func (s *Server) RotateAPIKey(key string) {
	if strings.TrimSpace(key) == "" {
		s.secretCache.Clear()
	} else {
		_ = s.secretCache.Load(SecretVersion{Value: strings.TrimSpace(key), Version: "runtime"})
	}
	s.apiKeyMu.Lock()
	s.apiKey = strings.TrimSpace(key)
	s.apiKeyMu.Unlock()
}

// LoadAPIKeyFromProvider refreshes authentication from an external secret
// provider. The cache is updated before the legacy field, so readers never
// observe a partially rotated credential.
func (s *Server) LoadAPIKeyFromProvider(ctx context.Context, provider SecretProvider, name string) error {
	if provider == nil {
		return errors.New("secret provider is required")
	}
	secret, err := provider.Read(ctx, name)
	if err != nil {
		return err
	}
	if err := s.secretCache.Load(secret); err != nil {
		return err
	}
	s.apiKeyMu.Lock()
	s.apiKey = strings.TrimSpace(secret.Value)
	s.apiKeyMu.Unlock()
	return nil
}

// StartAPIKeyReload polls a provider and atomically applies newer secret
// versions. Transient provider failures are returned on the channel but do
// not stop subsequent retries; cancellation closes the channel.
func (s *Server) StartAPIKeyReload(ctx context.Context, provider SecretProvider, name string, interval time.Duration) <-chan error {
	errs := make(chan error, 1)
	go func() {
		defer close(errs)
		if interval <= 0 {
			interval = time.Minute
		}
		var version string
		load := func() {
			secret, err := provider.Read(ctx, name)
			if err != nil {
				select {
				case errs <- err:
				default:
				}
				return
			}
			if secret.Version == version {
				return
			}
			if err := s.secretCache.Load(secret); err != nil {
				select {
				case errs <- err:
				default:
				}
				return
			}
			s.apiKeyMu.Lock()
			s.apiKey = strings.TrimSpace(secret.Value)
			s.apiKeyMu.Unlock()
			version = secret.Version
		}
		load()
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				load()
			}
		}
	}()
	return errs
}

func (s *Server) currentAPIKey() string {
	if v := s.secretCache.Current(); v.Value != "" {
		return v.Value
	}
	s.apiKeyMu.RLock()
	key := s.apiKey
	s.apiKeyMu.RUnlock()
	return key
}

type vectorPayload struct {
	ID        string            `json:"id"`
	Values    []float64         `json:"values"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	Namespace string            `json:"namespace,omitempty"`
}

type searchRequest struct {
	Values    []float64         `json:"values"`
	K         int               `json:"k"`
	FilterIDs []string          `json:"filter_ids,omitempty"`
	TextQuery string            `json:"text_query,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	Namespace string            `json:"namespace,omitempty"`
	Metric    string            `json:"metric,omitempty"`
}

const namespaceMetadataKey = "__lumenvec_namespace"

func metadataWithNamespace(metadata map[string]string, namespace string) map[string]string {
	if namespace == "" {
		return metadata
	}
	result := make(map[string]string, len(metadata)+1)
	for key, value := range metadata {
		result[key] = value
	}
	result[namespaceMetadataKey] = namespace
	return result
}

type batchVectorsRequest struct {
	Vectors []vectorPayload `json:"vectors"`
}

type batchSearchQuery struct {
	ID     string    `json:"id"`
	Values []float64 `json:"values"`
	K      int       `json:"k"`
}

type batchSearchRequest struct {
	Queries []batchSearchQuery `json:"queries"`
}

type listVectorsResponse struct {
	Vectors    []vectorPayload `json:"vectors"`
	NextCursor string          `json:"next_cursor,omitempty"`
}

const (
	defaultListVectorsLimit = 100
	// maxListVectorsLimit is deliberately higher than the default page. The
	// effective cap is additionally bounded by the configured vector dimension
	// so a ListVectors response always remains below the normal gRPC message
	// budget. Small vectors can therefore migrate in large, efficient pages
	// without making a high-dimensional public endpoint allocation-prone.
	maxListVectorsLimit           = 10000
	listVectorsResponseByteBudget = 3 << 20
)

type ServerOptions struct {
	Protocol                     string
	Port                         string
	ReadTimeout                  time.Duration
	WriteTimeout                 time.Duration
	MaxBodyBytes                 int64
	MaxVectorDim                 int
	MaxK                         int
	SnapshotPath                 string
	WALPath                      string
	SnapshotEvery                int
	VectorStore                  string
	VectorPath                   string
	SyncEvery                    int
	ShardCount                   int
	ShardFanout                  int
	ShardRouting                 string
	ShardIDs                     []string
	ClusterTopologyPath          string
	ReplicationMode              string
	ReplicationQueueSize         int
	ReplicationWriteQuorum       int
	ReplicationLogPath           string
	ReplicationFencePath         string
	ReplicationHeartbeatInterval time.Duration
	ReplicationElectionTimeout   time.Duration
	ClusterNodeID                string
	ClusterMembers               []string
	ClusterStateDir              string
	APIKey                       string
	MetricsEnabled               bool
	DisableRateLimit             bool
	RateLimitRPS                 int
	SearchMode                   string
	ANNBackend                   string
	IVFCentroids                 int
	IVFNProbe                    int
	ANNProfile                   string
	ANNM                         int
	ANNEfConstruct               int
	ANNEfSearch                  int
	ANNSegmentMaxNodes           int
	ANNStagedIngestMinBatch      int
	ANNQuantizeSegments          bool
	ANNSegmentRouting            bool
	ANNDiversifiedPruning        bool
	ANNMetricWarmup              []string
	ANNEvalSampleRate            int
	ANNAdaptive                  bool
	ANNMinCandidates             int
	ANNMaxProbe                  int
	CacheEnabled                 bool
	CacheMaxBytes                int64
	CacheMaxItems                int
	CacheTTL                     time.Duration
	LocationIndexCapacity        uint64
	IDIndexCapacity              uint64
	GRPCEnabled                  bool
	GRPCPort                     string
	SecurityProfile              string
	AuthEnabled                  bool
	AuthAPIKey                   string
	GRPCAuthEnabled              bool
	TLSEnabled                   bool
	TLSCertFile                  string
	TLSKeyFile                   string
	TLSClientCAFile              string
	AccessLogEnabled             bool
	AuditLogPath                 string
	TrustForwardedFor            bool
	TrustedProxies               []string
	StrictFilePerms              bool
	StorageDirMode               string
	StorageFileMode              string
	AdminEnabled                 bool
	AdminConfig                  AdminConfigController
	AdminCluster                 AdminClusterController
	DistributedRuntimeFactory    edition.DistributedRuntimeFactory
	UsageObserver                UsageObserver
}

var defaultServerOptions = ServerOptions{
	Protocol:                     "http",
	Port:                         ":19190",
	ReadTimeout:                  10 * time.Second,
	WriteTimeout:                 10 * time.Second,
	MaxBodyBytes:                 1 << 20,
	MaxVectorDim:                 4096,
	MaxK:                         100,
	SnapshotPath:                 "./data/snapshot.json",
	WALPath:                      "./data/wal.log",
	SnapshotEvery:                25,
	VectorStore:                  "memory",
	VectorPath:                   "./data/vectors",
	SyncEvery:                    1,
	ShardCount:                   1,
	ShardFanout:                  1,
	ShardRouting:                 core.ShardRoutingModulo,
	ReplicationMode:              "async",
	ReplicationQueueSize:         1024,
	ReplicationLogPath:           "./data/replication",
	ReplicationFencePath:         "./data/fencing",
	ReplicationWriteQuorum:       1,
	ReplicationHeartbeatInterval: time.Second,
	ReplicationElectionTimeout:   5 * time.Second,
	APIKey:                       "",
	MetricsEnabled:               true,
	RateLimitRPS:                 100,
	SearchMode:                   "exact",
		ANNBackend:                   "hierarchical-hnsw",
	IVFCentroids:                 64,
	IVFNProbe:                    4,
	ANNProfile:                   "balanced",
	ANNM:                         16,
	ANNEfConstruct:               64,
	ANNEfSearch:                  64,
	ANNSegmentMaxNodes:           10000,
	ANNEvalSampleRate:            0,
	ANNAdaptive:                  false,
	ANNMinCandidates:             0,
	ANNMaxProbe:                  0,
	CacheEnabled:                 false,
	CacheMaxBytes:                8 << 20,
	CacheMaxItems:                1024,
	CacheTTL:                     15 * time.Minute,
	GRPCEnabled:                  false,
	GRPCPort:                     ":19191",
	AccessLogEnabled:             false,
}

func annBuilder(backend string, centroidCount, nprobe int) func([]index.Vector) (core.ANNIndex, error) {
	if strings.ToLower(strings.TrimSpace(backend)) != "ivf" {
		return nil
	}
	return func(vectors []index.Vector) (core.ANNIndex, error) {
		if len(vectors) == 0 {
			return nil, errors.New("ivf requires training samples")
		}
		samples := make([][]float64, 0, len(vectors))
		for _, v := range vectors {
			samples = append(samples, v.Values)
		}
		centroids, err := ann.TrainIVFCentroidsKMeans(samples, centroidCount, 8, 4096)
		if err != nil {
			return nil, err
		}
		candidate, err := ann.NewIVFANNIndex(centroids, nprobe)
		if err != nil {
			return nil, err
		}
		for n, v := range vectors {
			if err := candidate.AddVector(n, v.Values); err != nil {
				return nil, err
			}
		}
		return candidate, nil
	}
}

func NewServer(port string) *Server {
	opts := defaultServerOptions
	if strings.TrimSpace(port) != "" {
		opts.Port = port
	}
	return NewServerWithOptions(opts)
}

func NewServerWithOptions(opts ServerOptions) *Server {
	opts = applyDefaults(opts)
	coreOpts := core.ServiceOptions{
		MaxVectorDim:  opts.MaxVectorDim,
		MaxK:          opts.MaxK,
		SnapshotPath:  opts.SnapshotPath,
		WALPath:       opts.WALPath,
		SnapshotEvery: opts.SnapshotEvery,
		SearchMode:    opts.SearchMode,
		ANNBackend:    opts.ANNBackend,
		ANNBuilder:    annBuilder(opts.ANNBackend, opts.IVFCentroids, opts.IVFNProbe),
		ANNProfile:    opts.ANNProfile,
		ANNOptions: ann.Options{
			M:                  opts.ANNM,
			EfConstruction:     opts.ANNEfConstruct,
			EfSearch:           opts.ANNEfSearch,
			QuantizeSegments:   opts.ANNQuantizeSegments,
			SegmentRouting:     opts.ANNSegmentRouting,
			DiversifiedPruning: opts.ANNDiversifiedPruning,
		},
		ANNSegmentMaxNodes:      opts.ANNSegmentMaxNodes,
		ANNStagedIngestMinBatch: opts.ANNStagedIngestMinBatch,
		ANNEvalSampleRate:       opts.ANNEvalSampleRate,
		ANNAdaptive:             opts.ANNAdaptive,
		ANNMinCandidates:        opts.ANNMinCandidates,
		ANNMaxProbe:             opts.ANNMaxProbe,
		VectorStore:             opts.VectorStore,
		VectorPath:              opts.VectorPath,
		LocationIndexCapacity:   opts.LocationIndexCapacity,
		IDIndexCapacity:         opts.IDIndexCapacity,
		SyncEvery:               opts.SyncEvery,
		StorageSecurity: core.StorageSecurityOptions{
			StrictFilePermissions: opts.StrictFilePerms,
			DirMode:               core.ParseFileMode(opts.StorageDirMode, os.FileMode(0o755)),
			FileMode:              core.ParseFileMode(opts.StorageFileMode, os.FileMode(0o644)),
		},
		Cache: core.CacheOptions{
			Enabled:  opts.CacheEnabled,
			MaxBytes: opts.CacheMaxBytes,
			MaxItems: opts.CacheMaxItems,
			TTL:      opts.CacheTTL,
		},
	}
	var vectorService core.VectorService
	var distributedRuntime edition.DistributedRuntime
	distributedRequested := strings.TrimSpace(opts.ClusterTopologyPath) != "" ||
		strings.TrimSpace(opts.ClusterNodeID) != "" ||
		len(opts.ClusterMembers) > 0
	if distributedRequested {
		if opts.DistributedRuntimeFactory == nil {
			panic("distributed runtime requested but no edition runtime factory is configured")
		}
		var err error
		distributedRuntime, err = opts.DistributedRuntimeFactory(edition.DistributedRuntimeOptions{
			TopologyPath:                 opts.ClusterTopologyPath,
			FanoutConcurrency:            opts.ShardFanout,
			ReplicationMode:              opts.ReplicationMode,
			ReplicationQueueSize:         opts.ReplicationQueueSize,
			ReplicationWriteQuorum:       opts.ReplicationWriteQuorum,
			ReplicationLogPath:           opts.ReplicationLogPath,
			ReplicationFencePath:         opts.ReplicationFencePath,
			ReplicationHeartbeatInterval: opts.ReplicationHeartbeatInterval,
			ReplicationElectionTimeout:   opts.ReplicationElectionTimeout,
			NodeID:                       opts.ClusterNodeID,
			Members:                      append([]string(nil), opts.ClusterMembers...),
			StateDir:                     opts.ClusterStateDir,
			WALPath:                      opts.WALPath,
			TLSEnabled:                   opts.TLSEnabled,
			TLSCertFile:                  opts.TLSCertFile,
		})
		if err != nil {
			panic(fmt.Errorf("build distributed edition runtime: %w", err))
		}
		vectorService = distributedRuntime.Service
	}
	if vectorService == nil && opts.ShardCount > 1 {
		if opts.ShardRouting == core.ShardRoutingRendezvous {
			if len(opts.ShardIDs) != opts.ShardCount {
				panic(fmt.Errorf("build rendezvous shard router: shard_count=%d does not match shard_ids=%d", opts.ShardCount, len(opts.ShardIDs)))
			}
			router, err := core.NewShardedServiceWithIDs(coreOpts, opts.ShardIDs, opts.ShardFanout)
			if err != nil {
				panic(fmt.Errorf("build rendezvous shard router: %w", err))
			}
			vectorService = router
		} else {
			vectorService = core.NewShardedService(coreOpts, opts.ShardCount, opts.ShardFanout)
		}
	} else if vectorService == nil {
		vectorService = core.NewService(coreOpts)
	}
	if len(opts.ANNMetricWarmup) > 0 {
		warmer, ok := vectorService.(interface {
			WarmMetricANN([]core.DistanceMetric) error
		})
		if !ok {
			panic("configured ANN metric warmup is not supported by this service")
		}
		metrics := make([]core.DistanceMetric, len(opts.ANNMetricWarmup))
		for index, metric := range opts.ANNMetricWarmup {
			metrics[index] = core.DistanceMetric(strings.ToLower(strings.TrimSpace(metric)))
		}
		if err := warmer.WarmMetricANN(metrics); err != nil {
			panic(fmt.Errorf("warm metric ANN: %w", err))
		}
	}
	s := &Server{
		protocol:          opts.Protocol,
		port:              opts.Port,
		grpcPort:          opts.GRPCPort,
		grpcEnabled:       opts.GRPCEnabled,
		readTimeout:       opts.ReadTimeout,
		writeTimeout:      opts.WriteTimeout,
		service:           vectorService,
		election:          distributedRuntime.Election,
		electionTransport: distributedRuntime.ElectionTransport,
		consensusLog:      distributedRuntime.ConsensusLog,
		consensusLogs:     distributedRuntime.ConsensusLogs,
		maxBodyBytes:      opts.MaxBodyBytes,
		listVectorsLimit:  listVectorsLimitForDimension(opts.MaxVectorDim),
		apiKey:            firstNonEmpty(opts.AuthAPIKey, opts.APIKey),
		metrics:           opts.MetricsEnabled,
		authEnabled:       opts.AuthEnabled,
		grpcAuth:          opts.GRPCAuthEnabled,
		tlsEnabled:        opts.TLSEnabled,
		tlsCertFile:       opts.TLSCertFile,
		tlsKeyFile:        opts.TLSKeyFile,
		tlsClientCAFile:   opts.TLSClientCAFile,
		accessLog:         opts.AccessLogEnabled,
		auditLogPath:      opts.AuditLogPath,
		trustXFF:          opts.TrustForwardedFor,
		snapshotPath:      opts.SnapshotPath,
		walPath:           opts.WALPath,
		vectorStore:       opts.VectorStore,
		vectorPath:        opts.VectorPath,
		trustedCIDRs:      parseTrustedProxies(opts.TrustedProxies),
		rateLimiter:       newRateLimiter(opts.RateLimitRPS, time.Second),
		adminEnabled:      opts.AdminEnabled,
		adminStartedAt:    time.Now(),
		adminConfig:       opts.AdminConfig,
		adminCluster:      opts.AdminCluster,
		usageObserver:     opts.UsageObserver,
	}
	if s.apiKey != "" {
		_ = s.secretCache.Load(SecretVersion{Value: s.apiKey, Version: "config"})
	}
	if s.metrics {
		s.requestTotal, s.requestDuration, s.metricsRegistry = newMetricsRegistry(s.service)
	}
	s.routes()
	return s
}

func listVectorsLimitForDimension(maxDimension int) int {
	if maxDimension < 1 {
		maxDimension = 1
	}
	// Avoid integer overflow for a malformed configuration and collapse to the
	// safest possible page in that case.
	if maxDimension > (listVectorsResponseByteBudget-128)/8 {
		return 1
	}
	// Values are protobuf packed doubles (eight bytes each). Keep a small
	// per-vector allowance for IDs and protobuf framing; this is conservative
	// and leaves headroom below gRPC's default four MiB send limit.
	bytesPerVector := maxDimension*8 + 128
	limit := listVectorsResponseByteBudget / bytesPerVector
	if limit < 1 {
		return 1
	}
	if limit > maxListVectorsLimit {
		return maxListVectorsLimit
	}
	return limit
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func parseTrustedProxies(values []string) []netip.Prefix {
	parsed := make([]netip.Prefix, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if prefix, err := netip.ParsePrefix(value); err == nil {
			parsed = append(parsed, prefix)
			continue
		}
		if addr, err := netip.ParseAddr(value); err == nil {
			parsed = append(parsed, netip.PrefixFrom(addr, addr.BitLen()))
		}
	}
	return parsed
}

func (s *Server) isTrustedProxy(remoteAddr string) bool {
	host, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		host = remoteAddr
	}
	addr, err := netip.ParseAddr(strings.TrimSpace(host))
	if err != nil {
		return false
	}
	for _, prefix := range s.trustedCIDRs {
		if prefix.Contains(addr) {
			return true
		}
	}
	return false
}

func validateAPIKey(provided, expected string) bool {
	provided = strings.TrimSpace(provided)
	expected = strings.TrimSpace(expected)
	if provided == "" || expected == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(provided), []byte(expected)) == 1
}

func authKeyFromHTTPRequest(r *http.Request) string {
	key := strings.TrimSpace(r.Header.Get("X-API-Key"))
	if key != "" {
		return key
	}
	auth := strings.TrimSpace(r.Header.Get("Authorization"))
	if strings.HasPrefix(strings.ToLower(auth), "bearer ") {
		return strings.TrimSpace(auth[7:])
	}
	return ""
}

func applyDefaults(opts ServerOptions) ServerOptions {
	opts.Protocol = normalizeServerProtocol(opts.Protocol)
	if strings.TrimSpace(opts.Port) == "" {
		opts.Port = defaultServerOptions.Port
	}
	if !strings.HasPrefix(opts.Port, ":") {
		opts.Port = ":" + opts.Port
	}
	if strings.TrimSpace(opts.GRPCPort) == "" {
		opts.GRPCPort = defaultServerOptions.GRPCPort
	}
	if !strings.HasPrefix(opts.GRPCPort, ":") {
		opts.GRPCPort = ":" + opts.GRPCPort
	}
	if opts.ReadTimeout <= 0 {
		opts.ReadTimeout = defaultServerOptions.ReadTimeout
	}
	if opts.WriteTimeout <= 0 {
		opts.WriteTimeout = defaultServerOptions.WriteTimeout
	}
	if opts.MaxBodyBytes <= 0 {
		opts.MaxBodyBytes = defaultServerOptions.MaxBodyBytes
	}
	if opts.MaxVectorDim <= 0 {
		opts.MaxVectorDim = defaultServerOptions.MaxVectorDim
	}
	if opts.MaxK <= 0 {
		opts.MaxK = defaultServerOptions.MaxK
	}
	if strings.TrimSpace(opts.SnapshotPath) == "" {
		opts.SnapshotPath = defaultServerOptions.SnapshotPath
	}
	if strings.TrimSpace(opts.WALPath) == "" {
		opts.WALPath = defaultServerOptions.WALPath
	}
	if strings.TrimSpace(opts.VectorStore) == "" {
		opts.VectorStore = defaultServerOptions.VectorStore
	}
	if strings.TrimSpace(opts.VectorPath) == "" {
		opts.VectorPath = defaultServerOptions.VectorPath
	}
	if opts.SnapshotEvery <= 0 {
		opts.SnapshotEvery = defaultServerOptions.SnapshotEvery
	}
	if opts.SyncEvery <= 0 {
		opts.SyncEvery = defaultServerOptions.SyncEvery
	}
	if opts.ShardCount <= 0 {
		opts.ShardCount = defaultServerOptions.ShardCount
	}
	if opts.ShardFanout <= 0 {
		opts.ShardFanout = opts.ShardCount
	}
	if opts.ShardFanout > opts.ShardCount {
		if strings.TrimSpace(opts.ClusterTopologyPath) == "" {
			opts.ShardFanout = opts.ShardCount
		}
	}
	if strings.TrimSpace(opts.ShardRouting) == "" {
		opts.ShardRouting = defaultServerOptions.ShardRouting
	}
	if opts.ReplicationQueueSize <= 0 {
		opts.ReplicationQueueSize = defaultServerOptions.ReplicationQueueSize
	}
	if opts.ReplicationWriteQuorum <= 0 {
		opts.ReplicationWriteQuorum = defaultServerOptions.ReplicationWriteQuorum
	}
	if opts.ReplicationHeartbeatInterval <= 0 {
		opts.ReplicationHeartbeatInterval = defaultServerOptions.ReplicationHeartbeatInterval
	}
	if opts.ReplicationElectionTimeout <= opts.ReplicationHeartbeatInterval {
		opts.ReplicationElectionTimeout = defaultServerOptions.ReplicationElectionTimeout
	}
	if strings.TrimSpace(opts.ReplicationLogPath) == "" {
		opts.ReplicationLogPath = defaultServerOptions.ReplicationLogPath
	}
	if strings.TrimSpace(opts.ReplicationFencePath) == "" {
		opts.ReplicationFencePath = defaultServerOptions.ReplicationFencePath
	}
	if opts.ReplicationMode == "eventual" {
		opts.ReplicationMode = "async"
	}
	if opts.ReplicationMode != "quorum" && opts.ReplicationMode != "strong" {
		opts.ReplicationMode = "async"
	}
	if !opts.DisableRateLimit && opts.RateLimitRPS <= 0 {
		opts.RateLimitRPS = defaultServerOptions.RateLimitRPS
	}
	if opts.DisableRateLimit {
		opts.RateLimitRPS = 0
	}
	if strings.TrimSpace(opts.SearchMode) == "" {
		opts.SearchMode = defaultServerOptions.SearchMode
	}
	if strings.TrimSpace(opts.ANNProfile) == "" {
		opts.ANNProfile = defaultServerOptions.ANNProfile
	}
	if opts.ANNM <= 0 {
		opts.ANNM = defaultServerOptions.ANNM
	}
	if opts.ANNEfConstruct <= 0 {
		opts.ANNEfConstruct = defaultServerOptions.ANNEfConstruct
	}
	if opts.ANNEfSearch <= 0 {
		opts.ANNEfSearch = defaultServerOptions.ANNEfSearch
	}
	if opts.ANNSegmentMaxNodes <= 0 {
		opts.ANNSegmentMaxNodes = defaultServerOptions.ANNSegmentMaxNodes
	}
	if opts.ANNEvalSampleRate < 0 {
		opts.ANNEvalSampleRate = defaultServerOptions.ANNEvalSampleRate
	}
	if opts.ANNMinCandidates < 0 {
		opts.ANNMinCandidates = defaultServerOptions.ANNMinCandidates
	}
	if opts.ANNMaxProbe < 0 {
		opts.ANNMaxProbe = defaultServerOptions.ANNMaxProbe
	}
	if opts.CacheMaxItems <= 0 {
		opts.CacheMaxItems = defaultServerOptions.CacheMaxItems
	}
	if opts.CacheMaxBytes <= 0 {
		opts.CacheMaxBytes = defaultServerOptions.CacheMaxBytes
	}
	if opts.CacheTTL <= 0 {
		opts.CacheTTL = defaultServerOptions.CacheTTL
	}
	opts.GRPCEnabled = opts.Protocol == "grpc"
	opts.SearchMode = strings.ToLower(strings.TrimSpace(opts.SearchMode))
	if opts.SearchMode != "ann" {
		opts.SearchMode = "exact"
	}
	return opts
}

func normalizeServerProtocol(protocol string) string {
	switch strings.ToLower(strings.TrimSpace(protocol)) {
	case "grpc":
		return "grpc"
	default:
		return "http"
	}
}

func (s *Server) routes() {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", methodHandler(http.MethodGet, s.HealthHandler))
	mux.HandleFunc("/livez", methodHandler(http.MethodGet, s.LivenessHandler))
	mux.HandleFunc("/readyz", methodHandler(http.MethodGet, s.ReadinessHandler))
	mux.HandleFunc("/v1/health", methodHandler(http.MethodGet, s.HealthHandler))
	mux.HandleFunc("/v1/livez", methodHandler(http.MethodGet, s.LivenessHandler))
	mux.HandleFunc("/v1/readyz", methodHandler(http.MethodGet, s.ReadinessHandler))
	if s.metrics && s.metricsRegistry != nil {
		mux.Handle("/metrics", methodHandler(http.MethodGet, promhttp.HandlerFor(s.metricsRegistry, promhttp.HandlerOpts{}).ServeHTTP))
	}
	s.registerVectorRoutes(mux, "")
	s.registerVectorRoutes(mux, "/v1")
	if s.adminEnabled {
		s.registerAdminRoutes(mux)
	}

	var handler http.Handler = mux
	handler = s.accessLogMiddleware(handler)
	if s.metrics {
		handler = s.metricsMiddleware(handler)
	}
	if s.authEnabled && s.apiKey != "" {
		handler = s.authMiddleware(handler)
	}
	if s.rateLimiter != nil {
		handler = s.rateLimitMiddleware(handler)
	}
	handler = s.requestIDMiddleware(handler)
	s.router = handler
}

func (s *Server) registerVectorRoutes(mux *http.ServeMux, prefix string) {
	mux.HandleFunc(prefix+"/admin/checkpoint", methodHandler(http.MethodPost, s.CheckpointHandler))
	mux.HandleFunc(prefix+"/vectors", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			s.ListVectorsHandler(w, r)
		case http.MethodPost:
			s.AddVectorHandler(w, r)
		default:
			methodNotAllowed(w, r)
		}
	})
	mux.HandleFunc(prefix+"/vectors/batch", methodHandler(http.MethodPost, s.AddVectorsBatchHandler))
	mux.HandleFunc(prefix+"/vectors/stream", methodHandler(http.MethodPost, s.AddVectorsStreamHandler))
	mux.HandleFunc(prefix+"/vectors/search", methodHandler(http.MethodPost, s.SearchVectorsHandler))
	mux.HandleFunc(prefix+"/vectors/search/batch", methodHandler(http.MethodPost, s.SearchVectorsBatchHandler))
	mux.HandleFunc(prefix+"/vectors/", s.vectorByIDHandler)
}

func (s *Server) CheckpointHandler(w http.ResponseWriter, r *http.Request) {
	// Checkpoint duration scales with the persisted graph and can legitimately
	// exceed the normal API write timeout. Override only this authenticated
	// administrative request; regular query/write deadlines remain unchanged.
	_ = http.NewResponseController(w).SetWriteDeadline(time.Now().Add(30 * time.Minute))
	checkpoint, ok := s.service.(interface{ Checkpoint() error })
	if !ok {
		writeError(w, r, http.StatusNotImplemented, "not_supported", "checkpoint is not supported")
		return
	}
	if err := checkpoint.Checkpoint(); err != nil {
		writeError(w, r, http.StatusInternalServerError, "checkpoint_failed", err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

const streamBatchSize = 256

// AddVectorsStreamHandler consumes newline-delimited vectorPayload JSON. It
// bounds memory by committing batches as they fill, while retaining the same
// metadata/namespace semantics as the regular batch endpoint.
func (s *Server) AddVectorsStreamHandler(w http.ResponseWriter, r *http.Request) {
	decoder := json.NewDecoder(io.LimitReader(r.Body, 1<<30))
	batch := make([]vectorPayload, 0, streamBatchSize)
	committed := 0
	flush := func() error {
		if len(batch) == 0 {
			return nil
		}
		vectors := make([]index.Vector, 0, len(batch))
		hasMetadata := false
		for i := range batch {
			batch[i].Metadata = metadataWithNamespace(batch[i].Metadata, batch[i].Namespace)
			vectors = append(vectors, index.Vector{ID: batch[i].ID, Values: batch[i].Values})
			hasMetadata = hasMetadata || len(batch[i].Metadata) > 0
		}
		if hasMetadata {
			metadata := make([]map[string]string, len(batch))
			for i := range batch {
				metadata[i] = batch[i].Metadata
			}
			if enriched, ok := s.service.(interface {
				AddVectorsWithMetadata([]index.Vector, []map[string]string) error
			}); ok {
				if err := enriched.AddVectorsWithMetadata(vectors, metadata); err != nil {
					return err
				}
				committed += len(batch)
				s.observeIngest(uint64(len(batch)))
				batch = batch[:0]
				return nil
			}
			enriched, ok := s.service.(interface {
				AddVectorWithMetadata(string, []float64, map[string]string) error
			})
			if !ok {
				return errors.New("metadata ingestion is not supported")
			}
			for _, vector := range batch {
				if err := enriched.AddVectorWithMetadata(vector.ID, vector.Values, vector.Metadata); err != nil {
					return err
				}
			}
		} else if err := s.service.AddVectors(vectors); err != nil {
			return err
		}
		committed += len(batch)
		s.observeIngest(uint64(len(batch)))
		batch = batch[:0]
		return nil
	}
	for {
		var payload vectorPayload
		err := decoder.Decode(&payload)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			writeError(w, r, http.StatusBadRequest, "invalid_argument", "invalid NDJSON vector")
			return
		}
		if payload.ID == "" {
			writeError(w, r, http.StatusBadRequest, "invalid_argument", "id is required")
			return
		}
		batch = append(batch, payload)
		if len(batch) == streamBatchSize {
			if err := flush(); err != nil {
				writeServiceError(w, r, err)
				return
			}
		}
	}
	if err := flush(); err != nil {
		writeServiceError(w, r, err)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]int{"committed": committed})
}

func (s *Server) Router() http.Handler {
	return s.router
}

func methodHandler(method string, handler http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != method {
			methodNotAllowed(w, r)
			return
		}
		handler(w, r)
	}
}

func methodNotAllowed(w http.ResponseWriter, r *http.Request) {
	writeError(w, r, http.StatusMethodNotAllowed, "method_not_allowed", "method not allowed")
}

func (s *Server) vectorByIDHandler(w http.ResponseWriter, r *http.Request) {
	if vectorIDFromPath(r.URL.Path) == "" {
		writeError(w, r, http.StatusNotFound, "not_found", "vector not found")
		return
	}
	switch r.Method {
	case http.MethodGet:
		s.GetVectorHandler(w, r)
	case http.MethodDelete:
		s.DeleteVectorHandler(w, r)
	default:
		methodNotAllowed(w, r)
	}
}

func (s *Server) HealthHandler(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

func (s *Server) LivenessHandler(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

func (s *Server) ReadinessHandler(w http.ResponseWriter, r *http.Request) {
	if err := s.readinessCheck(); err != nil {
		writeError(w, r, http.StatusServiceUnavailable, "not_ready", err.Error())
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ready"))
}

func (s *Server) readinessCheck() error {
	if s == nil || s.service == nil {
		return errors.New("service is not initialized")
	}
	if s.auditLogPath != "" {
		if err := ensureWritableParent(s.auditLogPath); err != nil {
			return fmt.Errorf("audit log path is not writable: %w", err)
		}
	}
	if stats := s.service.Stats(); stats.ReplicationPending > 0 {
		return fmt.Errorf("replication backlog is not drained: %d pending", stats.ReplicationPending)
	}
	if strings.EqualFold(strings.TrimSpace(s.vectorStore), "disk") {
		if err := ensureWritableDir(s.vectorPath); err != nil {
			return fmt.Errorf("vector path is not writable: %w", err)
		}
		return nil
	}
	if err := ensureWritableParent(s.snapshotPath); err != nil {
		return fmt.Errorf("snapshot path is not writable: %w", err)
	}
	if err := ensureWritableParent(s.walPath); err != nil {
		return fmt.Errorf("wal path is not writable: %w", err)
	}
	return nil
}

func ensureWritableParent(path string) error {
	if strings.TrimSpace(path) == "" {
		return errors.New("path is empty")
	}
	return ensureWritableDir(filepath.Dir(path))
}

func ensureWritableDir(path string) error {
	if strings.TrimSpace(path) == "" {
		return errors.New("directory is empty")
	}
	if err := os.MkdirAll(path, 0o755); err != nil { // #nosec G301 -- data directory permissions are controlled by the deployment user.
		return err
	}
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	if !info.IsDir() {
		return fmt.Errorf("%s is not a directory", path)
	}
	return nil
}

func (s *Server) ListVectorsHandler(w http.ResponseWriter, r *http.Request) {
	// Optional query params:
	// - limit (int): maximum number of vectors to return, capped by maxListVectorsLimit
	// - cursor (string): opaque base64-encoded last ID (exclusive)
	// - after (string): legacy raw id cursor (backwards compat)
	// - ids_only (bool): when true, return only ids (no values)
	q := r.URL.Query()
	limit, ok := s.parseListVectorsLimit(q)
	if !ok {
		writeError(w, r, http.StatusBadRequest, "invalid_argument", "limit must be a positive integer")
		return
	}

	afterID := strings.TrimSpace(q.Get("after"))
	if cursorVal := strings.TrimSpace(q.Get("cursor")); cursorVal != "" {
		decoded, err := decodeListVectorsCursor(cursorVal)
		if err != nil {
			writeError(w, r, http.StatusBadRequest, "invalid_argument", "cursor must be a valid list cursor")
			return
		}
		afterID = decoded
	}

	idsOnly := false
	if v := strings.TrimSpace(q.Get("ids_only")); v != "" {
		if v == "1" || strings.EqualFold(v, "true") || strings.EqualFold(v, "yes") {
			idsOnly = true
		}
	}

	page := s.service.ListVectorsPage(core.ListVectorsOptions{
		AfterID: afterID,
		Limit:   limit,
		IDsOnly: idsOnly,
	})
	var nextCursor string
	if page.NextCursor != "" {
		nextCursor = encodeListVectorsCursor(page.NextCursor)
	}

	if idsOnly {
		payload := struct {
			Vectors []struct {
				ID string `json:"id"`
			} `json:"vectors"`
			NextCursor string `json:"next_cursor,omitempty"`
		}{Vectors: make([]struct {
			ID string `json:"id"`
		}, 0, len(page.Vectors)), NextCursor: nextCursor}
		for _, v := range page.Vectors {
			payload.Vectors = append(payload.Vectors, struct {
				ID string `json:"id"`
			}{ID: v.ID})
		}
		writeJSON(w, 0, payload)
		return
	}

	out := make([]vectorPayload, 0, len(page.Vectors))
	for _, vec := range page.Vectors {
		out = append(out, vectorPayload{ID: vec.ID, Values: vec.Values})
	}
	writeJSON(w, 0, listVectorsResponse{Vectors: out, NextCursor: nextCursor})
}

func parseListVectorsLimit(values url.Values) (int, bool) {
	return parseListVectorsLimitWithCap(values, maxListVectorsLimit)
}

func (s *Server) parseListVectorsLimit(values url.Values) (int, bool) {
	return parseListVectorsLimitWithCap(values, s.effectiveListVectorsLimit())
}

func parseListVectorsLimitWithCap(values url.Values, cap int) (int, bool) {
	raw := strings.TrimSpace(values.Get("limit"))
	if raw == "" {
		return defaultListVectorsLimit, true
	}
	limit, err := strconv.Atoi(raw)
	if err != nil || limit <= 0 {
		return 0, false
	}
	if cap < 1 {
		cap = maxListVectorsLimit
	}
	if limit > cap {
		return cap, true
	}
	return limit, true
}

func (s *Server) effectiveListVectorsLimit() int {
	if s.listVectorsLimit > 0 {
		return s.listVectorsLimit
	}
	return maxListVectorsLimit
}

func decodeListVectorsCursor(cursor string) (string, error) {
	if decoded, err := base64.RawURLEncoding.DecodeString(cursor); err == nil {
		return string(decoded), nil
	}
	decoded, err := base64.URLEncoding.DecodeString(cursor)
	if err != nil {
		return "", err
	}
	return string(decoded), nil
}

func encodeListVectorsCursor(id string) string {
	return base64.RawURLEncoding.EncodeToString([]byte(id))
}

func (s *Server) AddVectorHandler(w http.ResponseWriter, r *http.Request) {
	var payload vectorPayload
	if !s.readJSON(w, r, &payload) {
		return
	}
	if payload.ID == "" {
		writeError(w, r, http.StatusBadRequest, "invalid_argument", "id is required")
		return
	}
	var addErr error
	payload.Metadata = metadataWithNamespace(payload.Metadata, payload.Namespace)
	if len(payload.Metadata) > 0 {
		if enriched, ok := s.service.(interface {
			AddVectorWithMetadata(string, []float64, map[string]string) error
		}); ok {
			addErr = enriched.AddVectorWithMetadata(payload.ID, payload.Values, payload.Metadata)
		} else {
			addErr = errors.New("metadata ingestion is not supported")
		}
	} else {
		addErr = s.service.AddVector(payload.ID, payload.Values)
	}
	if err := addErr; err != nil {
		if errors.Is(err, index.ErrVectorExists) {
			writeError(w, r, http.StatusConflict, "already_exists", err.Error())
			return
		}
		writeServiceError(w, r, err)
		return
	}
	s.observeIngest(1)
	w.WriteHeader(http.StatusCreated)
}

func (s *Server) AddVectorsBatchHandler(w http.ResponseWriter, r *http.Request) {
	var req batchVectorsRequest
	if !s.readJSON(w, r, &req) {
		return
	}
	vectors := make([]index.Vector, 0, len(req.Vectors))
	hasMetadata := false
	for _, vec := range req.Vectors {
		vec.Metadata = metadataWithNamespace(vec.Metadata, vec.Namespace)
		vectors = append(vectors, index.Vector{ID: vec.ID, Values: vec.Values})
		if len(vec.Metadata) > 0 {
			hasMetadata = true
		}
	}
	if hasMetadata {
		metadata := make([]map[string]string, len(req.Vectors))
		for i := range req.Vectors {
			req.Vectors[i].Metadata = metadataWithNamespace(req.Vectors[i].Metadata, req.Vectors[i].Namespace)
			metadata[i] = req.Vectors[i].Metadata
		}
		if enriched, ok := s.service.(interface {
			AddVectorsWithMetadata([]index.Vector, []map[string]string) error
		}); ok {
			if err := enriched.AddVectorsWithMetadata(vectors, metadata); err != nil {
				writeServiceError(w, r, err)
				return
			}
			s.observeIngest(uint64(len(vectors)))
			w.WriteHeader(http.StatusCreated)
			return
		}
		if enriched, ok := s.service.(interface {
			AddVectorWithMetadata(string, []float64, map[string]string) error
		}); ok {
			for _, vec := range req.Vectors {
				if err := enriched.AddVectorWithMetadata(vec.ID, vec.Values, vec.Metadata); err != nil {
					writeServiceError(w, r, err)
					return
				}
				s.observeIngest(1)
			}
			w.WriteHeader(http.StatusCreated)
			return
		}
	}
	if err := s.service.AddVectors(vectors); err != nil {
		if errors.Is(err, index.ErrVectorExists) {
			writeError(w, r, http.StatusConflict, "already_exists", err.Error())
			return
		}
		writeServiceError(w, r, err)
		return
	}
	s.observeIngest(uint64(len(vectors)))
	w.WriteHeader(http.StatusCreated)
}

func (s *Server) GetVectorHandler(w http.ResponseWriter, r *http.Request) {
	id := vectorIDFromPath(r.URL.Path)
	vec, err := s.service.GetVector(id)
	if err != nil {
		if errors.Is(err, index.ErrVectorNotFound) {
			writeError(w, r, http.StatusNotFound, "not_found", err.Error())
			return
		}
		writeServiceError(w, r, err)
		return
	}

	writeJSON(w, 0, vectorPayload{ID: vec.ID, Values: vec.Values})
}

func (s *Server) DeleteVectorHandler(w http.ResponseWriter, r *http.Request) {
	id := vectorIDFromPath(r.URL.Path)
	if err := s.service.DeleteVector(id); err != nil {
		if errors.Is(err, index.ErrVectorNotFound) {
			writeError(w, r, http.StatusNotFound, "not_found", err.Error())
			return
		}
		writeServiceError(w, r, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) SearchVectorsHandler(w http.ResponseWriter, r *http.Request) {
	var req searchRequest
	if !s.readJSON(w, r, &req) {
		return
	}
	var results []core.SearchResult
	var err error
	req.Metadata = metadataWithNamespace(req.Metadata, req.Namespace)
	metric := core.DistanceMetric(strings.ToLower(req.Metric))
	if metric == "" {
		metric = core.MetricL2
	}
	if metric != core.MetricL2 && metric != core.MetricCosine && metric != core.MetricInnerProduct {
		http.Error(w, "unsupported search metric", http.StatusBadRequest)
		return
	}
	if len(req.FilterIDs) > 0 || req.TextQuery != "" || len(req.Metadata) > 0 {
		allowed := make(map[string]struct{}, len(req.FilterIDs))
		for _, id := range req.FilterIDs {
			allowed[id] = struct{}{}
		}
		structured := core.StructuredFilter{IDs: allowed, TextQuery: req.TextQuery, Metadata: req.Metadata}
		if structuredSearch, ok := s.service.(interface {
			SearchStructured([]float64, int, core.StructuredFilter, core.DistanceMetric) ([]core.SearchResult, error)
		}); ok {
			results, err = core.SearchStructuredWithContext(r.Context(), structuredSearch, req.Values, req.K, structured, metric)
		} else if filtered, ok := s.service.(interface {
			SearchFilteredMetric([]float64, int, core.VectorFilter, core.DistanceMetric) ([]core.SearchResult, error)
		}); ok {
			results, err = core.SearchFilteredMetricWithContext(r.Context(), filtered, req.Values, req.K, func(v index.Vector) bool {
				if len(allowed) > 0 {
					if _, ok := allowed[v.ID]; !ok {
						return false
					}
				}
				return structured.Match(v)
			}, metric)
		} else {
			http.Error(w, "filtered search is not supported", http.StatusNotImplemented)
			return
		}
	} else if metric == core.MetricL2 {
		results, err = core.SearchWithContext(r.Context(), s.service, req.Values, req.K)
	} else if metricSearch, ok := s.service.(interface {
		SearchFilteredMetric([]float64, int, core.VectorFilter, core.DistanceMetric) ([]core.SearchResult, error)
	}); ok {
		results, err = core.SearchFilteredMetricWithContext(r.Context(), metricSearch, req.Values, req.K, nil, metric)
	} else {
		http.Error(w, "metric search is not supported", http.StatusNotImplemented)
		return
	}
	if err != nil {
		writeServiceError(w, r, err)
		return
	}
	s.observeSearch(1)
	writeJSON(w, 0, results)
}

func (s *Server) SearchVectorsBatchHandler(w http.ResponseWriter, r *http.Request) {
	var req batchSearchRequest
	if !s.readJSON(w, r, &req) {
		return
	}
	queries := make([]core.BatchSearchQuery, 0, len(req.Queries))
	for _, query := range req.Queries {
		queries = append(queries, core.BatchSearchQuery{ID: query.ID, Values: query.Values, K: query.K})
	}
	results, err := core.SearchBatchWithContext(r.Context(), s.service, queries)
	if err != nil {
		writeServiceError(w, r, err)
		return
	}
	s.observeSearch(uint64(len(queries)))
	writeJSON(w, 0, results)
}

func (s *Server) observeIngest(count uint64) {
	if count > 0 && s.usageObserver != nil {
		s.usageObserver.ObserveIngestedVectors(count)
	}
}

func (s *Server) observeSearch(count uint64) {
	if count > 0 && s.usageObserver != nil {
		s.usageObserver.ObserveSearchRequests(count)
	}
}

func (s *Server) Start() {
	if err := s.Run(context.Background()); err != nil {
		logFatalfAPI("Could not start server: %s", err)
	}
}

func (s *Server) Run(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	defer func() { _ = s.Close() }()

	if s.grpcEnabled {
		logPrintfAPI("Starting gRPC server on port %s", s.grpcPort)
		listener, err := s.grpcListener()
		if err != nil {
			return fmt.Errorf("bind gRPC server: %w", err)
		}
		return s.runGRPC(ctx, listener)
	}

	logPrintfAPI("Starting HTTP server on port %s", s.port)
	return s.runHTTP(ctx)
}

func (s *Server) runHTTP(ctx context.Context) error {
	server := s.httpServer()
	if s.tlsEnabled && s.tlsClientCAFile != "" {
		configuration, err := s.serverTLSConfig()
		if err != nil {
			return err
		}
		server.TLSConfig = configuration
	}
	errCh := make(chan error, 1)
	go func() {
		var err error
		if s.tlsEnabled {
			err = listenAndServeTLSFunc(server, s.tlsCertFile, s.tlsKeyFile)
		} else {
			err = listenAndServeFunc(server)
		}
		errCh <- err
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			return fmt.Errorf("shutdown HTTP server: %w", err)
		}
		if err := <-errCh; err != nil && !errors.Is(err, http.ErrServerClosed) {
			return fmt.Errorf("serve HTTP: %w", err)
		}
		return nil
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			return fmt.Errorf("serve HTTP: %w", err)
		}
		return nil
	}
}

func (s *Server) serverTLSConfig() (*tls.Config, error) {
	certificate, err := tls.LoadX509KeyPair(s.tlsCertFile, s.tlsKeyFile)
	if err != nil {
		return nil, fmt.Errorf("load server TLS identity: %w", err)
	}
	configuration := &tls.Config{MinVersion: tls.VersionTLS13, Certificates: []tls.Certificate{certificate}}
	if s.tlsClientCAFile == "" {
		return configuration, nil
	}
	raw, err := os.ReadFile(s.tlsClientCAFile)
	if err != nil {
		return nil, fmt.Errorf("read client CA: %w", err)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(raw) {
		return nil, errors.New("client CA contains no certificates")
	}
	configuration.ClientCAs, configuration.ClientAuth = pool, tls.RequireAndVerifyClientCert
	return configuration, nil
}

func (s *Server) runGRPC(ctx context.Context, listener net.Listener) error {
	server, err := s.grpcServer()
	if err != nil {
		_ = listener.Close()
		return err
	}
	errCh := make(chan error, 1)
	go func() {
		errCh <- grpcServeFunc(server, listener)
	}()

	select {
	case <-ctx.Done():
		stopped := make(chan struct{})
		go func() {
			server.GracefulStop()
			close(stopped)
		}()
		select {
		case <-stopped:
		case <-time.After(10 * time.Second):
			server.Stop()
		}
		if err := <-errCh; err != nil && !errors.Is(err, net.ErrClosed) {
			return fmt.Errorf("serve gRPC: %w", err)
		}
		return nil
	case err := <-errCh:
		if err != nil && !errors.Is(err, net.ErrClosed) {
			return fmt.Errorf("serve gRPC: %w", err)
		}
		return nil
	}
}

func (s *Server) Close() error {
	if s == nil {
		return nil
	}
	var errs []error
	if s.service != nil {
		errs = append(errs, s.service.Close())
	}
	if s.election != nil {
		errs = append(errs, s.election.Close())
	}
	if s.electionTransport != nil {
		errs = append(errs, s.electionTransport.Close())
	}
	if s.consensusLogs != nil {
		errs = append(errs, s.consensusLogs.Close())
	} else if s.consensusLog != nil {
		errs = append(errs, s.consensusLog.Close())
	}
	return errors.Join(errs...)
}

func (s *Server) httpServer() *http.Server {
	return &http.Server{
		Addr:         s.port,
		Handler:      s.router,
		ReadTimeout:  s.readTimeout,
		WriteTimeout: s.writeTimeout,
	}
}

const contentTypeJSON = "application/json"

func (s *Server) readJSON(w http.ResponseWriter, r *http.Request, dst any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, s.maxBodyBytes)
	if err := json.NewDecoder(r.Body).Decode(dst); err != nil {
		writeError(w, r, http.StatusBadRequest, "invalid_json", "invalid JSON payload")
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", contentTypeJSON)
	if status != 0 {
		w.WriteHeader(status)
	}
	_ = json.NewEncoder(w).Encode(v)
}

type errorResponse struct {
	Error apiError `json:"error"`
}

type apiError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func writeServiceError(w http.ResponseWriter, r *http.Request, err error) {
	writeError(w, r, statusFromServiceError(err), errorCodeFromStatus(statusFromServiceError(err)), err.Error())
}

func writeError(w http.ResponseWriter, r *http.Request, status int, code, message string) {
	if !isV1Request(r) {
		http.Error(w, message, status)
		return
	}
	if strings.TrimSpace(code) == "" {
		code = errorCodeFromStatus(status)
	}
	writeJSON(w, status, errorResponse{Error: apiError{Code: code, Message: message}})
}

func isV1Request(r *http.Request) bool {
	return r != nil && (r.URL.Path == "/v1" || strings.HasPrefix(r.URL.Path, "/v1/"))
}

func vectorIDFromPath(path string) string {
	path = strings.TrimPrefix(path, "/v1")
	return strings.TrimPrefix(path, "/vectors/")
}

func statusFromServiceError(err error) int {
	switch {
	case errors.Is(err, core.ErrInvalidID),
		errors.Is(err, core.ErrInvalidValues),
		errors.Is(err, core.ErrInvalidK),
		errors.Is(err, core.ErrVectorDimTooHigh),
		errors.Is(err, core.ErrKTooHigh),
		errors.Is(err, ann.ErrInvalidVectorDim):
		return http.StatusBadRequest
	default:
		return http.StatusInternalServerError
	}
}

func errorCodeFromStatus(status int) string {
	switch status {
	case http.StatusBadRequest:
		return "invalid_argument"
	case http.StatusUnauthorized:
		return "unauthorized"
	case http.StatusNotFound:
		return "not_found"
	case http.StatusConflict:
		return "already_exists"
	case http.StatusMethodNotAllowed:
		return "method_not_allowed"
	case http.StatusTooManyRequests:
		return "rate_limited"
	default:
		return "internal"
	}
}
