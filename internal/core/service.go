package core

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"
	"lumenvec/internal/vector"
)

var (
	ErrInvalidID           = errors.New("id is required")
	ErrInvalidValues       = errors.New("values are required")
	ErrVectorDimTooHigh    = errors.New("vector dimension exceeds configured max")
	ErrInvalidK            = errors.New("k must be greater than 0")
	ErrKTooHigh            = errors.New("k exceeds configured max")
	ErrInvalidMetric       = errors.New("unsupported search metric")
	ErrFilteredUnsupported = errors.New("filtered search is not supported by a shard")
	errSkipWALOp           = errors.New("skip wal op")
)

const (
	annDeleteRebuildMinDeleted = 1024
	annDeleteRebuildRatio      = 0.25
)

type SearchResult struct {
	ID       string  `json:"id"`
	Distance float64 `json:"distance"`
}

type BatchSearchQuery struct {
	ID       string
	Values   []float64
	Values32 []float32
	K        int
}

type BatchSearchResult struct {
	ID      string         `json:"id"`
	Results []SearchResult `json:"results"`
}

type ServiceOptions struct {
	ANNRebuildSnapshotBudgetBytes uint64
	MaxVectorDim                  int
	MaxK                          int
	SnapshotPath                  string
	WALPath                       string
	SnapshotEvery                 int
	SearchMode                    string
	ANNBackend                    string
	ANNProfile                    string
	ANNOptions                    ann.Options
	ANNSegmentMaxNodes            int
	ANNStagedIngestMinBatch       int
	ANNEvalSampleRate             int
	ANNAdaptive                   bool
	ANNMinCandidates              int
	ANNMaxProbe                   int
	VectorStore                   string
	VectorPath                    string
	LocationIndexCapacity         uint64
	IDIndexCapacity               uint64
	Cache                         CacheOptions
	StorageSecurity               StorageSecurityOptions
	SyncEvery                     int
	WALReplicator                 WALReplicator
	// ANNBuilder optionally constructs an ANN index from the current vectors.
	// Returning an error keeps the compatibility fallback (HNSW) active.
	ANNBuilder func([]index.Vector) (ANNIndex, error)
}

type VectorIndex interface {
	AddVector(vec index.Vector) error
	SearchVector(id string) (index.Vector, error)
	DeleteVector(id string) error
	ListVectors() []index.Vector
	RangeVectors(fn func(index.Vector) bool)
}

type ANNIndex interface {
	AddVector(id int, vector []float64) error
	SearchWithDistancesInto(query []float64, k int, dst []ann.Result) ([]ann.Result, error)
	DeleteVector(id int)
	Stats() ann.Stats
}

type vectorIndex32Ranger interface {
	RangeVectors32(fn func(id string, values []float32) bool)
}

type IDResolver interface {
	Assign(id string) int
	Lookup(internalID int) (string, bool)
	Remove(id string)
}

// errorAwareIDResolver is implemented by persistent resolvers whose lookups
// and mutations can fail. Keeping it additive preserves compatibility for
// embedders that provide the original in-memory-only IDResolver contract.
type errorAwareIDResolver interface {
	AssignID(id string) (int, error)
	LookupID(internalID int) (string, bool, error)
	RemoveID(id string) error
}

type ServiceDeps struct {
	Index       VectorIndex
	VectorStore VectorStore
	ANNIndex    ANNIndex
	IDResolver  IDResolver
	Persistence PersistenceBackend
}

type Service struct {
	annRebuildSnapshotBudgetBytes uint64
	annRebuildFailures            atomic.Uint64
	annRebuildSuccesses           atomic.Uint64
	annRebuildLastError           atomic.Pointer[string]
	metadataMu                    sync.RWMutex
	metadata                      map[string]map[string]string
	textIndex                     fullTextIndex
	textIndexInitMu               sync.Mutex
	textIndexReady                atomic.Bool
	index                         VectorIndex
	annIndex                      ANNIndex
	annMu                         sync.RWMutex
	metricANNMu                   sync.RWMutex
	metricANN                     map[DistanceMetric]ANNIndex
	maxVectorDim                  int
	maxK                          int
	snapshotPath                  string
	walPath                       string
	snapshotEvery                 int
	searchMode                    string
	annProfile                    string
	annBackend                    string
	annOptions                    ann.Options
	annSegmented                  bool
	annSegmentMaxNodes            int
	annStagedIngestMinBatch       int
	annEvalSampleRate             int
	annAdaptive                   bool
	annMinCandidates              int
	annMaxProbe                   int
	annBuilder                    func([]index.Vector) (ANNIndex, error)
	annBuilderActive              bool
	persistOps                    int
	persistMu                     sync.Mutex
	syncEvery                     int
	vectorStore                   VectorStore
	vectorPath                    string
	storageSecurity               StorageSecurityOptions
	idResolver                    IDResolver
	persistence                   PersistenceBackend
	annResultPool                 sync.Pool
	query32Pool                   sync.Pool
	batchQuery32Pool              sync.Pool
	batchQuery64Pool              sync.Pool
	batchPreparedPool             sync.Pool
	stats                         serviceStats
	closeMu                       sync.Mutex
	closed                        bool
	closeErr                      error
	annCheckpointLoaded           atomic.Bool
	annCheckpointDirty            atomic.Bool
}

func (s *Service) assignID(id string) (int, error) {
	if resolver, ok := s.idResolver.(errorAwareIDResolver); ok {
		return resolver.AssignID(id)
	}
	return s.idResolver.Assign(id), nil
}

func (s *Service) lookupID(internalID int) (string, bool, error) {
	if resolver, ok := s.idResolver.(errorAwareIDResolver); ok {
		return resolver.LookupID(internalID)
	}
	id, ok := s.idResolver.Lookup(internalID)
	return id, ok, nil
}

func (s *Service) removeID(id string) error {
	if resolver, ok := s.idResolver.(errorAwareIDResolver); ok {
		return resolver.RemoveID(id)
	}
	s.idResolver.Remove(id)
	return nil
}

// ExportIDMappings returns the stable IDs required to publish a prebuilt ANN
// snapshot on another node. Supplying an empty id slice exports the complete
// resolver, which is intended only for an empty destination generation.
func (s *Service) ExportIDMappings(ids []string) ([]IDMappingEntry, error) {
	s.ensureRuntimeDeps()
	resolver, ok := s.idResolver.(checkpointIDResolver)
	if !ok {
		return nil, errors.New("ID resolver does not support mapping export")
	}
	entries := resolver.snapshotSorted()
	if len(ids) == 0 {
		out := make([]IDMappingEntry, len(entries))
		for i, entry := range entries {
			out[i] = IDMappingEntry{ID: entry.id, InternalID: entry.internalID}
		}
		return out, nil
	}
	wanted := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		wanted[id] = struct{}{}
	}
	out := make([]IDMappingEntry, 0, len(ids))
	for _, entry := range entries {
		if _, ok := wanted[entry.id]; ok {
			out = append(out, IDMappingEntry{ID: entry.id, InternalID: entry.internalID})
			delete(wanted, entry.id)
		}
	}
	if len(wanted) != 0 {
		return nil, errors.New("one or more vector IDs are missing from resolver")
	}
	return out, nil
}

// ImportIDMappings prepares an empty destination to accept vectors and a
// prebuilt ANN snapshot with source-assigned internal IDs. It is idempotent
// for identical retries and rejects conflicting identities atomically.
func (s *Service) ImportIDMappings(entries []IDMappingEntry) error {
	if len(entries) == 0 {
		return nil
	}
	s.ensureRuntimeDeps()
	resolver, ok := s.idResolver.(mergeIDMappingResolver)
	if !ok {
		return errors.New("ID resolver does not support mapping import")
	}
	s.persistMu.Lock()
	defer s.persistMu.Unlock()
	return resolver.MergeIDMappings(entries)
}

// ReserveIDMappings allocates collision-free destination identities before a
// target-specific ANN graph is built. Repeated reservations return the same
// IDs, allowing multiple source shards with overlapping local ID spaces to be
// repartitioned safely into one destination.
func (s *Service) ReserveIDMappings(ids []string) ([]IDMappingEntry, error) {
	if len(ids) == 0 {
		return nil, ErrInvalidID
	}
	s.ensureRuntimeDeps()
	lookup, ok := s.idResolver.(externalIDLookupResolver)
	if !ok {
		return nil, errors.New("ID resolver does not support destination reservations")
	}
	s.persistMu.Lock()
	defer s.persistMu.Unlock()
	seen := make(map[string]struct{}, len(ids))
	out := make([]IDMappingEntry, 0, len(ids))
	added := make([]string, 0, len(ids))
	for _, id := range ids {
		if strings.TrimSpace(id) == "" {
			for _, rollbackID := range added {
				_ = s.removeID(rollbackID)
			}
			return nil, ErrInvalidID
		}
		if _, duplicate := seen[id]; duplicate {
			continue
		}
		seen[id] = struct{}{}
		internalID, exists, err := lookup.LookupExternalID(id)
		if err != nil {
			for _, rollbackID := range added {
				_ = s.removeID(rollbackID)
			}
			return nil, err
		}
		if !exists {
			internalID, err = s.assignID(id)
			if err != nil {
				for _, rollbackID := range added {
					_ = s.removeID(rollbackID)
				}
				return nil, err
			}
			added = append(added, id)
		}
		out = append(out, IDMappingEntry{ID: id, InternalID: internalID})
	}
	return out, nil
}

type ServiceStats struct {
	ANNRebuildFailures          uint64 `json:"ann_rebuild_failures"`
	ANNRebuildSuccesses         uint64 `json:"ann_rebuild_successes"`
	ANNRebuildLastError         string `json:"ann_rebuild_last_error,omitempty"`
	ShardCount                  int    `json:"shard_count"`
	ReplicationCommitted        uint64 `json:"replication_committed_offset"`
	ReplicationApplied          uint64 `json:"replication_applied_offset"`
	ReplicationPending          uint64 `json:"replication_pending"`
	ReplicationFailures         uint64 `json:"replication_failures_total"`
	ReplicationTerm             uint64 `json:"replication_term"`
	ReplicationFailovers        uint64 `json:"replication_failovers"`
	SearchRequestsTotal         uint64 `json:"search_requests_total"`
	ExactSearchesTotal          uint64 `json:"exact_searches_total"`
	ANNSearchesTotal            uint64 `json:"ann_searches_total"`
	ANNSearchHitsTotal          uint64 `json:"ann_search_hits_total"`
	ANNSearchFallbacks          uint64 `json:"ann_search_fallbacks_total"`
	ANNSearchErrorsTotal        uint64 `json:"ann_search_errors_total"`
	ANNCandidatesReturned       uint64 `json:"ann_candidates_returned_total"`
	ANNEvalSamplesTotal         uint64 `json:"ann_eval_samples_total"`
	ANNEvalTop1Matches          uint64 `json:"ann_eval_top1_matches_total"`
	ANNEvalOverlapResults       uint64 `json:"ann_eval_overlap_results_total"`
	ANNEvalComparedResults      uint64 `json:"ann_eval_compared_results_total"`
	ANNNodes                    int    `json:"ann_nodes"`
	ANNDeleted                  int    `json:"ann_deleted"`
	ANNVectorBytes              uint64 `json:"ann_vector_bytes"`
	ANNAdjacencyBytes           uint64 `json:"ann_adjacency_bytes"`
	ANNMapBytes                 uint64 `json:"ann_map_bytes"`
	ANNRouteBytes               uint64 `json:"ann_route_bytes"`
	ANNStructuralBytes          uint64 `json:"ann_structural_bytes"`
	ANNMetricIndexes            int    `json:"ann_metric_indexes"`
	ANNMetricNodes              int    `json:"ann_metric_nodes"`
	ANNMetricStructuralBytes    uint64 `json:"ann_metric_structural_bytes"`
	ANNSegments                 int    `json:"ann_segments"`
	ANNCompactionPending        int    `json:"ann_compaction_pending"`
	ANNCompacting               int    `json:"ann_compacting"`
	ANNCompactionCompactable    int    `json:"ann_compaction_compactable"`
	ANNReaders                  int    `json:"ann_readers"`
	ANNRetiredSegments          int    `json:"ann_retired_segments"`
	ANNReclaimedSegments        uint64 `json:"ann_reclaimed_segments_total"`
	ANNReclaimErrors            uint64 `json:"ann_reclaim_errors_total"`
	ANNCompactionMemoryBudget   uint64 `json:"ann_compaction_memory_budget_bytes"`
	ANNCompactionMemoryReserved uint64 `json:"ann_compaction_memory_reserved_bytes"`
	ANNCompactionMemoryEstimate uint64 `json:"ann_compaction_memory_estimate_bytes"`
	ANNCompactionMemoryDeferred uint64 `json:"ann_compaction_memory_deferred_total"`
	ANNPrimaryBuildActive       bool   `json:"ann_primary_build_active"`
	ANNPrimaryBuildTotal        uint64 `json:"ann_primary_build_vectors_total"`
	ANNPrimaryBuildDone         uint64 `json:"ann_primary_build_vectors_done"`
	ANNPrimaryBuildDurationMs   uint64 `json:"ann_primary_build_duration_ms"`
	ANNPrimaryBuildFailures     uint64 `json:"ann_primary_build_failures_total"`
	ANNSearchBudgetQueries      uint64 `json:"ann_search_budget_queries_total"`
	ANNSearchBudgetSegments     uint64 `json:"ann_search_budget_segments_total"`
	ANNSearchEFBudget           uint64 `json:"ann_search_ef_budget_total"`
	ANNCheckpointLoaded         bool   `json:"ann_checkpoint_loaded"`
	CacheHitsTotal              uint64 `json:"cache_hits_total"`
	CacheMissesTotal            uint64 `json:"cache_misses_total"`
	CacheEvictionsTotal         uint64 `json:"cache_evictions_total"`
	CacheItems                  uint64 `json:"cache_items"`
	CacheBytes                  uint64 `json:"cache_bytes"`
	DiskFileBytes               uint64 `json:"disk_file_bytes"`
	DiskRecords                 uint64 `json:"disk_records"`
	DiskStaleRecords            uint64 `json:"disk_stale_records"`
	DiskCompactionsTotal        uint64 `json:"disk_compactions_total"`
	DiskCompactionActive        bool   `json:"disk_compaction_active"`
	DiskSegments                uint64 `json:"disk_segments"`
	ANNProfile                  string `json:"ann_profile"`
	ANNM                        int    `json:"ann_m"`
	ANNEfConstruction           int    `json:"ann_ef_construction"`
	ANNEfSearch                 int    `json:"ann_ef_search"`
	ANNSearchExecution          string `json:"ann_search_execution"`
	ANNEFBudgetMode             string `json:"ann_ef_budget_mode"`
	ANNEFGlobalPercent          int    `json:"ann_ef_global_percent"`
	ANNPrimaryIndexEnabled      bool   `json:"ann_primary_index_enabled"`
	ANNSegmentRouting           bool   `json:"ann_segment_routing"`
	ANNDiversifiedPruning       bool   `json:"ann_diversified_pruning"`
}

type serviceStats struct {
	searchRequestsTotal    atomic.Uint64
	exactSearchesTotal     atomic.Uint64
	annSearchesTotal       atomic.Uint64
	annSearchHitsTotal     atomic.Uint64
	annSearchFallbacks     atomic.Uint64
	annSearchErrorsTotal   atomic.Uint64
	annCandidatesReturned  atomic.Uint64
	annEvalSamplesTotal    atomic.Uint64
	annEvalTop1Matches     atomic.Uint64
	annEvalOverlapResults  atomic.Uint64
	annEvalComparedResults atomic.Uint64
}

type walOp struct {
	Op              string             `json:"op"`
	ID              string             `json:"id"`
	Values          []float64          `json:"values,omitempty"`
	Metadata        map[string]string  `json:"metadata,omitempty"`
	MetadataEntries []walMetadataEntry `json:"metadata_entries,omitempty"`
}

type walMetadataEntry struct {
	ID       string            `json:"id"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

type preparedBatchQuery struct {
	id     string
	vals   []float64
	vals32 []float32
	acc    topKAccumulator
}

type topKAccumulator struct {
	limit int
	items []SearchResult
}

const exactBatchDistanceWidth = 4

func NewService(opts ServiceOptions) *Service {
	return NewServiceWithDeps(opts, ServiceDeps{})
}

func NewServiceWithDeps(opts ServiceOptions, deps ServiceDeps) *Service {
	serviceStarted := time.Now()
	opts.ANNOptions = applyANNProfile(opts.ANNProfile, opts.ANNOptions)
	annBackend := normalizeANNBackend(opts.ANNBackend)
	annSegmented := strings.EqualFold(strings.TrimSpace(opts.VectorStore), "segment") || strings.EqualFold(strings.TrimSpace(opts.VectorStore), "segmented")
	if annSegmented && strings.TrimSpace(opts.VectorPath) == "" {
		opts.VectorPath = "./data/vectors"
	}
	if deps.Index == nil {
		deps.Index = index.NewIndex()
	}
	if deps.VectorStore == nil {
		started := time.Now()
		deps.VectorStore = newDefaultVectorStoreWithLocationCapacity(opts.VectorStore, opts.VectorPath, opts.StorageSecurity, opts.SyncEvery, opts.LocationIndexCapacity)
		startupTrace("vector-store-open", started)
	}
	deps.VectorStore = newCachedVectorStore(deps.VectorStore, opts.Cache)
	if annSegmented {
		// The persistent segment store is canonical; retaining the legacy
		// in-memory index here would duplicate every vector payload.
		deps.Index = &storeBackedIndex{store: deps.VectorStore}
	}
	annBuilderActive := false
	if deps.ANNIndex == nil && opts.ANNBuilder != nil {
		if candidate, err := opts.ANNBuilder(deps.VectorStore.ListVectors()); err == nil && candidate != nil {
			deps.ANNIndex = candidate
			annBuilderActive = true
		}
	}
	if deps.ANNIndex == nil {
		deps.ANNIndex = newConfiguredANNIndex(annBackend, opts.ANNOptions, annSegmented, opts.ANNSegmentMaxNodes, 0)
	}
	if deps.IDResolver == nil {
		if opts.IDIndexCapacity > 0 {
			directory := filepath.Join(opts.VectorPath, "id-index")
			if err := os.MkdirAll(directory, normalizeStorageSecurityOptions(opts.StorageSecurity).DirMode); err != nil {
				panic(err)
			}
			resolver, err := openDiskIDResolver(directory, opts.IDIndexCapacity)
			if err != nil {
				panic(err)
			}
			deps.IDResolver = resolver
		} else {
			deps.IDResolver = newMemoryIDResolver()
		}
	}
	if deps.Persistence == nil {
		deps.Persistence = newSnapshotWALBackendWithOptions(opts.SnapshotPath, opts.WALPath, opts.StorageSecurity, opts.SyncEvery)
	}
	if opts.WALReplicator != nil {
		deps.Persistence = &replicatedPersistence{backend: deps.Persistence, replicator: opts.WALReplicator}
	}

	svc := &Service{
		annRebuildSnapshotBudgetBytes: opts.ANNRebuildSnapshotBudgetBytes,
		index:                         deps.Index,
		annIndex:                      deps.ANNIndex,
		maxVectorDim:                  opts.MaxVectorDim,
		maxK:                          opts.MaxK,
		snapshotPath:                  opts.SnapshotPath,
		walPath:                       opts.WALPath,
		snapshotEvery:                 opts.SnapshotEvery,
		searchMode:                    normalizeSearchMode(opts.SearchMode),
		annProfile:                    normalizeANNProfile(opts.ANNProfile),
		annBackend:                    annBackend,
		annOptions:                    opts.ANNOptions,
		annSegmented:                  annSegmented,
		annSegmentMaxNodes:            opts.ANNSegmentMaxNodes,
		annStagedIngestMinBatch:       opts.ANNStagedIngestMinBatch,
		annEvalSampleRate:             clampPercent(opts.ANNEvalSampleRate),
		annAdaptive:                   opts.ANNAdaptive,
		annMinCandidates:              opts.ANNMinCandidates,
		annMaxProbe:                   opts.ANNMaxProbe,
		annBuilder:                    opts.ANNBuilder,
		annBuilderActive:              annBuilderActive,
		syncEvery:                     normalizeSyncEvery(opts.SyncEvery),
		vectorStore:                   deps.VectorStore,
		vectorPath:                    opts.VectorPath,
		storageSecurity:               normalizeStorageSecurityOptions(opts.StorageSecurity),
		idResolver:                    deps.IDResolver,
		persistence:                   deps.Persistence,
	}
	svc.annResultPool.New = func() any {
		capHint := 64
		if svc.maxK > 0 && svc.maxK < capHint {
			capHint = svc.maxK
		}
		buf := make([]ann.Result, 0, capHint)
		return &buf
	}
	svc.query32Pool.New = func() any {
		buf := make([]float32, 0, maxIntOrOne(svc.maxVectorDim))
		return &buf
	}
	svc.batchQuery32Pool.New = func() any {
		buf := make([]float32, 0, maxIntOrOne(svc.maxVectorDim))
		return &buf
	}
	svc.batchQuery64Pool.New = func() any {
		buf := make([]float64, 0, maxIntOrOne(svc.maxVectorDim))
		return &buf
	}
	svc.batchPreparedPool.New = func() any {
		buf := make([]preparedBatchQuery, 0, 16)
		return &buf
	}

	started := time.Now()
	_ = svc.restoreState()
	if err := svc.recoverPrebuiltANNTransactions(); err != nil {
		panic(fmt.Errorf("recover prebuilt ANN transactions: %w", err))
	}
	startupTrace("restore-state", started)
	startupTrace("service-total", serviceStarted)
	return svc
}

func startupTrace(phase string, started time.Time) {
	if strings.EqualFold(strings.TrimSpace(os.Getenv("VECTOR_DB_STARTUP_TRACE")), "true") {
		fmt.Fprintf(os.Stderr, "lumenvec startup phase=%s duration=%s\n", phase, time.Since(started))
	}
}

func (s *Service) AddVector(id string, values []float64) error {
	return s.AddVectors([]index.Vector{{ID: id, Values: values}})
}

func (s *Service) hasAnyMetadata() bool {
	s.metadataMu.RLock()
	defer s.metadataMu.RUnlock()
	return len(s.metadata) != 0
}

// AddVectors32 is the internal resharding fast path. It accepts the native
// float32 representation exposed by segment stores, preserving the public
// float64 API while avoiding a float64 source snapshot during migration.
func (s *Service) AddVectors32(vectors []vectorBatch32) error {
	staged := s.annStagedIngestMinBatch > 0 && len(vectors) >= s.annStagedIngestMinBatch
	return s.addVectors32(vectors, staged)
}

// addVectors32Staged persists the logical/vector records through the normal
// transaction, but materializes the ANN batch off to the side and publishes it
// only after construction completes. It is used by online migration drains so
// query readers never observe a partially built mutable graph.
func (s *Service) addVectors32Staged(vectors []vectorBatch32) error {
	return s.addVectors32(vectors, true)
}

func (s *Service) addVectors32(vectors []vectorBatch32, staged bool) error {
	if len(vectors) == 0 {
		return nil
	}
	s.ensureRuntimeDeps()
	if writer, ok := s.vectorStore.(interface{ UpsertVectors32([]vectorBatch32) error }); ok {
		s.persistMu.Lock()
		defer s.persistMu.Unlock()
		added := make([]string, 0, len(vectors))
		nativeBatch, useNativeBatch := s.index.(interface {
			AddVectors32([]index.Vector32) error
		})
		nativeIndex, useNativeIndex := s.index.(interface {
			AddVector32(string, []float32) error
		})
		if useNativeBatch {
			batch := make([]index.Vector32, len(vectors))
			for i, vec := range vectors {
				if strings.TrimSpace(vec.ID) == "" || len(vec.Values) == 0 {
					return ErrInvalidValues
				}
				if len(vec.Values) > s.maxVectorDim {
					return fmt.Errorf("%w (%d)", ErrVectorDimTooHigh, s.maxVectorDim)
				}
				batch[i] = index.Vector32{ID: vec.ID, Values: vec.Values}
			}
			if err := nativeBatch.AddVectors32(batch); err != nil {
				return err
			}
			added = append(added, vectorIDs(vectors)...)
		}
		for _, vec := range vectors {
			if useNativeBatch {
				continue
			}
			if strings.TrimSpace(vec.ID) == "" || len(vec.Values) == 0 {
				s.rollbackAddedVectors(added)
				return ErrInvalidValues
			}
			if len(vec.Values) > s.maxVectorDim {
				s.rollbackAddedVectors(added)
				return fmt.Errorf("%w (%d)", ErrVectorDimTooHigh, s.maxVectorDim)
			}
			var err error
			if useNativeIndex {
				err = nativeIndex.AddVector32(vec.ID, vec.Values)
			} else {
				values64 := make([]float64, len(vec.Values))
				for i, value := range vec.Values {
					values64[i] = float64(value)
				}
				err = s.index.AddVector(index.Vector{ID: vec.ID, Values: values64})
			}
			if err != nil {
				s.rollbackAddedVectors(added)
				return err
			}
			added = append(added, vec.ID)
		}
		if err := writer.UpsertVectors32(vectors); err != nil {
			s.rollbackAddedVectors(added)
			return err
		}
		var annErr error
		if staged {
			annErr = s.buildAndPublishANNVectors32(vectors)
		} else {
			annErr = s.addANNVectors32(vectors)
		}
		if annErr != nil {
			s.rollbackAddedVectors(added)
			return annErr
		}
		if err := s.addMetricANNVectors32Locked(vectors); err != nil {
			s.invalidateMetricANNLocked()
			s.rollbackAddedVectors(added)
			return err
		}
		s.annCheckpointDirty.Store(true)
		return s.maybeSnapshot()
	}
	converted := make([]index.Vector, len(vectors))
	for i, vector := range vectors {
		if len(vector.Values) == 0 {
			return ErrInvalidValues
		}
		values := make([]float64, len(vector.Values))
		for j, value := range vector.Values {
			values[j] = float64(value)
		}
		converted[i] = index.Vector{ID: vector.ID, Values: values}
	}
	return s.AddVectors(converted)
}

// ReplayDeltasSince exposes the durable mutation stream to the resharding
// controller without exposing the WAL implementation to transports.
func (s *Service) ReplayDeltasSince(offset uint64, fn func(DeltaRecord) error) error {
	if s == nil {
		return errors.New("delta WAL is unavailable")
	}
	store, ok := s.vectorStore.(*segmentVectorStore)
	if !ok || store.deltaWAL == nil {
		return errors.New("delta WAL is unavailable")
	}
	return store.deltaWAL.ReplaySince(offset, fn)
}

func (s *Service) CurrentDeltaOffset() uint64 {
	if s == nil {
		return 0
	}
	// The offset is a migration cutover barrier, not merely an informational
	// counter. Waiting on persistMu guarantees that an AddVectors transaction
	// which started before this read has either published its complete WAL
	// batch or failed before the coordinator decides the source is stable.
	s.persistMu.Lock()
	defer s.persistMu.Unlock()
	return s.currentDeltaOffsetUnlocked()
}

func (s *Service) VectorCount() (int, error) {
	if s == nil {
		return 0, nil
	}
	if counter, ok := s.vectorStore.(interface{ VectorCount() (int, error) }); ok {
		return counter.VectorCount()
	}
	return len(s.vectorStore.ListVectors()), nil
}

// currentDeltaOffsetUnlocked requires persistMu to be held by the caller.
// ExportANNState uses it to bind its ANN snapshot and WAL offset to the same
// persistence transaction boundary.
func (s *Service) currentDeltaOffsetUnlocked() uint64 {
	store, ok := s.vectorStore.(*segmentVectorStore)
	if !ok || store.deltaWAL == nil {
		return 0
	}
	return store.deltaWAL.CurrentOffset()
}

func vectorIDs(vectors []vectorBatch32) []string {
	ids := make([]string, len(vectors))
	for i, vec := range vectors {
		ids[i] = vec.ID
	}
	return ids
}

func (s *Service) addANNVectors32(vectors []vectorBatch32) error {
	if segmented, ok := s.currentANNIndex().(interface {
		AddBatch32([]ann.BatchVector32) error
	}); ok {
		batch := make([]ann.BatchVector32, len(vectors))
		for i, vector := range vectors {
			internalID, err := s.assignID(vector.ID)
			if err != nil {
				return err
			}
			batch[i] = ann.BatchVector32{ID: internalID, Values: vector.Values}
		}
		return segmented.AddBatch32(batch)
	}
	for _, vector := range vectors {
		values := make([]float64, len(vector.Values))
		for i, value := range vector.Values {
			values[i] = float64(value)
		}
		internalID, err := s.assignID(vector.ID)
		if err != nil {
			return err
		}
		if err := s.addANNVector(internalID, values); err != nil {
			return err
		}
	}
	return nil
}

// buildAndPublishANNVectors32 materializes a migration drain as immutable ANN
// generations. The vector store/WAL remains the source of truth; only after
// the isolated graph is complete does the segmented index publish it.
func (s *Service) buildAndPublishANNVectors32(vectors []vectorBatch32) error {
	segmented, ok := s.currentANNIndex().(interface {
		BuildAndPublishBatch32([]ann.BatchVector32) error
	})
	if !ok {
		return errors.New("segmented ANN index does not support staged publication")
	}
	batch := make([]ann.BatchVector32, len(vectors))
	for i, vector := range vectors {
		internalID, err := s.assignID(vector.ID)
		if err != nil {
			return err
		}
		batch[i] = ann.BatchVector32{ID: internalID, Values: vector.Values}
	}
	return segmented.BuildAndPublishBatch32(batch)
}

func (s *Service) AddVectors(vectors []index.Vector) error {
	s.ensureRuntimeDeps()
	if len(vectors) == 0 {
		return ErrInvalidValues
	}

	s.persistMu.Lock()
	defer s.persistMu.Unlock()

	addedIDs := make([]string, 0, len(vectors))
	for _, vec := range vectors {
		if strings.TrimSpace(vec.ID) == "" {
			s.rollbackAddedVectors(addedIDs)
			return ErrInvalidID
		}
		if len(vec.Values) == 0 {
			s.rollbackAddedVectors(addedIDs)
			return ErrInvalidValues
		}
		if len(vec.Values) > s.maxVectorDim {
			s.rollbackAddedVectors(addedIDs)
			return fmt.Errorf("%w (%d)", ErrVectorDimTooHigh, s.maxVectorDim)
		}
		// Index first so duplicate IDs return conflict without mutating the vector store.
		if err := s.index.AddVector(index.Vector{ID: vec.ID, Values: vec.Values}); err != nil {
			s.rollbackAddedVectors(addedIDs)
			return err
		}
		addedIDs = append(addedIDs, vec.ID)
	}

	if err := s.upsertVectors(vectors); err != nil {
		s.rollbackAddedVectors(addedIDs)
		return err
	}
	// Do not build the expensive n-gram index for every bulk-ingested ID before
	// structured text search is used. SearchStructured performs one indexed
	// rebuild on demand; after that point incremental writes must be indexed.
	if s.textIndexReady.Load() {
		for _, vec := range vectors {
			s.textIndex.add(vec.ID, vec.ID)
		}
	}

	if err := s.addANNVectors(vectors); err != nil {
		s.rollbackAddedVectors(addedIDs)
		return err
	}
	if err := s.addMetricANNVectorsLocked(vectors); err != nil {
		s.invalidateMetricANNLocked()
		s.rollbackAddedVectors(addedIDs)
		return err
	}
	// An opt-in trained backend cannot be built before the first dimensioned
	// batch exists. Build it once after the initial successful ingest.
	if s.annBuilder != nil && !s.annBuilderActive {
		s.rebuildANNLocked()
	}

	// Persistent vector stores already durably record the batch. Avoid building
	// a second in-memory WAL representation (including every value slice) for
	// large bulk ingests; this previously doubled peak memory for segmented
	// 1M-vector migrations even though appendWALBatch was a no-op.
	if !s.usesPersistentVectorStore() {
		ops := make([]walOp, 0, len(vectors))
		for _, vec := range vectors {
			ops = append(ops, walOp{Op: "upsert", ID: vec.ID, Values: vec.Values})
		}
		if err := s.appendWALBatch(ops); err != nil {
			s.rollbackAddedVectors(addedIDs)
			return err
		}
	}
	s.annCheckpointDirty.Store(true)
	return s.maybeSnapshot()
}

// PersistVectorsForPrebuiltANN writes a validated batch to the canonical
// vector store and exact index without mutating the ANN generation. It is an
// internal reshard primitive: callers import stable ID mappings first, persist
// the vectors through this method, then atomically publish a prebuilt ANN
// snapshot. Public ingestion must continue to use AddVectors.
func (s *Service) PersistVectorsForPrebuiltANN(vectors []index.Vector) error {
	s.ensureRuntimeDeps()
	if len(vectors) == 0 {
		return ErrInvalidValues
	}
	ids := make([]string, len(vectors))
	for i, vec := range vectors {
		ids[i] = vec.ID
	}
	s.persistMu.Lock()
	defer s.persistMu.Unlock()
	if s.currentANNIndex().Stats().Nodes != 0 {
		return errors.New("prebuilt ANN vector staging requires an empty ANN generation")
	}
	if _, err := s.ExportIDMappings(ids); err != nil {
		return fmt.Errorf("prebuilt ANN vector staging requires imported ID mappings: %w", err)
	}
	addedIDs := make([]string, 0, len(vectors))
	for _, vec := range vectors {
		if strings.TrimSpace(vec.ID) == "" {
			s.rollbackAddedVectors(addedIDs)
			return ErrInvalidID
		}
		if len(vec.Values) == 0 {
			s.rollbackAddedVectors(addedIDs)
			return ErrInvalidValues
		}
		if len(vec.Values) > s.maxVectorDim {
			s.rollbackAddedVectors(addedIDs)
			return fmt.Errorf("%w (%d)", ErrVectorDimTooHigh, s.maxVectorDim)
		}
		if err := s.index.AddVector(index.Vector{ID: vec.ID, Values: vec.Values}); err != nil {
			s.rollbackAddedVectors(addedIDs)
			return err
		}
		addedIDs = append(addedIDs, vec.ID)
	}
	if err := s.upsertVectors(vectors); err != nil {
		s.rollbackAddedVectors(addedIDs)
		return err
	}
	// Prebuilt publication replaces the primary graph independently. Derived
	// metric graphs are rebuilt lazily from the canonical store.
	s.invalidateMetricANNLocked()
	if !s.usesPersistentVectorStore() {
		ops := make([]walOp, 0, len(vectors))
		for _, vec := range vectors {
			ops = append(ops, walOp{Op: "upsert", ID: vec.ID, Values: vec.Values})
		}
		if err := s.appendWALBatch(ops); err != nil {
			s.rollbackAddedVectors(addedIDs)
			return err
		}
	}
	return s.maybeSnapshot()
}

// PublishPrebuiltANNSnapshot attaches a previously validated HNSW generation
// after PersistVectorsForPrebuiltANN has made its vectors durable. The method
// refuses non-segmented or already-populated ANN indexes, so it cannot silently
// combine a transferred graph with a graph built by ordinary ingestion.
func (s *Service) PublishPrebuiltANNSnapshot(snapshot []byte) error {
	s.ensureRuntimeDeps()
	s.persistMu.Lock()
	defer s.persistMu.Unlock()
	segmented, ok := s.currentANNIndex().(*ann.SegmentedIndex)
	if !ok {
		return errors.New("prebuilt ANN publication requires a segmented ANN index")
	}
	if segmented.Stats().Nodes != 0 {
		return errors.New("prebuilt ANN publication requires an empty ANN generation")
	}
	if err := segmented.PublishPrebuiltSnapshot(snapshot); err != nil {
		return err
	}
	s.annCheckpointDirty.Store(true)
	return s.maybeSnapshot()
}

// PublishPrebuiltANNSnapshotFile is the file-backed counterpart of
// PublishPrebuiltANNSnapshot. It publishes a SnapshotStager-committed artifact
// without allocating a byte slice proportional to the transferred graph.
func (s *Service) PublishPrebuiltANNSnapshotFile(path string, manifest ann.SegmentManifest) error {
	s.ensureRuntimeDeps()
	s.persistMu.Lock()
	defer s.persistMu.Unlock()
	segmented, ok := s.currentANNIndex().(*ann.SegmentedIndex)
	if !ok {
		return errors.New("prebuilt ANN publication requires a segmented ANN index")
	}
	if segmented.Stats().Nodes != 0 {
		return errors.New("prebuilt ANN publication requires an empty ANN generation")
	}
	ids, err := ann.PrebuiltSnapshotIDsFromFile(path, manifest)
	if err != nil {
		return fmt.Errorf("validate prebuilt ANN snapshot file: %w", err)
	}
	for _, internalID := range ids {
		externalID, ok, err := s.lookupID(internalID)
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("prebuilt ANN snapshot references unmapped internal ID %d", internalID)
		}
		if _, err := s.vectorStore.GetVector(externalID); err != nil {
			return fmt.Errorf("prebuilt ANN snapshot references missing vector %q: %w", externalID, err)
		}
	}
	if err := segmented.PublishPrebuiltSnapshotFile(path, manifest); err != nil {
		return err
	}
	s.annCheckpointDirty.Store(true)
	return s.maybeSnapshot()
}

// AddVectorsStream ingests batches supplied incrementally by next. The source
// must return (nil, nil) to signal end of stream. Each batch is committed
// through the same validation, persistence, WAL and ANN paths as AddVectors,
// then becomes eligible for reclamation before the next batch is requested.
// AddVectors remains the atomic single-batch compatibility API; this method is
// intended for bulk imports whose producer cannot retain the full dataset.
func (s *Service) AddVectorsStream(next func() ([]index.Vector, error)) error {
	if next == nil {
		return ErrInvalidValues
	}
	for {
		batch, err := next()
		if err != nil {
			return err
		}
		if batch == nil {
			return nil
		}
		if len(batch) == 0 {
			continue
		}
		if err := s.AddVectors(batch); err != nil {
			return err
		}
	}
}

// AddVectorsStream32 ingests native float32 batches without widening the
// payload to the public float64 representation. It is intentionally an
// internal bulk path; AddVectorsStream remains the compatibility API.
func (s *Service) AddVectorsStream32(next func() ([]vectorBatch32, error)) error {
	if next == nil {
		return ErrInvalidValues
	}
	for {
		batch, err := next()
		if err != nil {
			return err
		}
		if batch == nil {
			return nil
		}
		if len(batch) == 0 {
			continue
		}
		if err := s.AddVectors32(batch); err != nil {
			return err
		}
	}
}

func (s *Service) GetVector(id string) (index.Vector, error) {
	s.ensureRuntimeDeps()
	if strings.TrimSpace(id) == "" {
		return index.Vector{}, ErrInvalidID
	}
	return s.vectorStore.GetVector(id)
}

func (s *Service) ListVectors() []index.Vector {
	s.ensureRuntimeDeps()
	vecs := s.vectorStore.ListVectors()
	sort.Slice(vecs, func(i, j int) bool { return vecs[i].ID < vecs[j].ID })
	return vecs
}

func (s *Service) DeleteVector(id string) error {
	s.ensureRuntimeDeps()
	if strings.TrimSpace(id) == "" {
		return ErrInvalidID
	}

	s.persistMu.Lock()
	defer s.persistMu.Unlock()

	vec, err := s.vectorStore.GetVector(id)
	if err != nil {
		return err
	}
	if err := s.index.DeleteVector(id); err != nil {
		return err
	}
	if err := s.vectorStore.DeleteVector(id); err != nil {
		_ = s.index.AddVector(vec)
		return err
	}
	internalID, err := s.assignID(id)
	if err != nil {
		_ = s.vectorStore.UpsertVector(vec)
		_ = s.index.AddVector(vec)
		return err
	}
	s.deleteANNVector(internalID)
	s.deleteMetricANNVectorLocked(internalID)
	if err := s.removeID(id); err != nil {
		_ = s.vectorStore.UpsertVector(vec)
		_ = s.index.AddVector(vec)
		_, _ = s.assignID(id)
		s.rebuildANNLocked()
		return err
	}

	if err := s.appendWAL(walOp{Op: "delete", ID: id}); err != nil {
		_ = s.vectorStore.UpsertVector(vec)
		_ = s.index.AddVector(vec)
		_, _ = s.assignID(id)
		s.rebuildANNLocked()
		return err
	}
	s.maybeCompactANNAfterDeleteLocked()
	s.metadataMu.Lock()
	if s.metadata != nil {
		delete(s.metadata, id)
	}
	s.metadataMu.Unlock()
	s.textIndex.remove(id)
	s.annCheckpointDirty.Store(true)
	return s.maybeSnapshot()
}

func (s *Service) Search(values []float64, k int) ([]SearchResult, error) {
	return s.SearchContext(context.Background(), values, k)
}

func (s *Service) SearchContext(ctx context.Context, values []float64, k int) ([]SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.ensureRuntimeDeps()
	if err := s.validateSearchRequest(values, k); err != nil {
		return nil, err
	}
	s.stats.searchRequestsTotal.Add(1)

	if s.searchMode == "ann" {
		results, ok, err := s.searchANNContext(ctx, values, k)
		if err != nil {
			return nil, err
		}
		if ok {
			s.maybeEvaluateANN(values, k, results)
			return results, nil
		}
		s.stats.annSearchFallbacks.Add(1)
	}
	s.stats.exactSearchesTotal.Add(1)
	return s.searchExactContext(ctx, values, k)
}

func (s *Service) SearchBatch(queries []BatchSearchQuery) ([]BatchSearchResult, error) {
	return s.SearchBatchContext(context.Background(), queries)
}

func (s *Service) SearchBatchContext(ctx context.Context, queries []BatchSearchQuery) ([]BatchSearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.ensureRuntimeDeps()
	if len(queries) == 0 {
		return nil, ErrInvalidValues
	}

	preparedBuf := s.getPreparedBatchBuffer(len(queries))
	defer s.putPreparedBatchBuffer(preparedBuf)
	prepared := (*preparedBuf)[:0]
	exactMode := s.searchMode != "ann"
	totalQueryValues := 0
	if exactMode {
		for _, query := range queries {
			if len(query.Values) > 0 {
				totalQueryValues += len(query.Values)
			} else {
				totalQueryValues += len(query.Values32)
			}
		}
	}
	var query32Buf *[]float32
	var query32Block []float32
	if totalQueryValues > 0 {
		query32Buf = s.getBatchQuery32Buffer(totalQueryValues)
		query32Block = (*query32Buf)[:totalQueryValues]
		defer s.putBatchQuery32Buffer(query32Buf)
	}
	query32Offset := 0
	for i, query := range queries {
		valuesLen := len(query.Values)
		if valuesLen == 0 {
			valuesLen = len(query.Values32)
		}
		if valuesLen == 0 || valuesLen > s.maxVectorDim {
			if valuesLen == 0 {
				return nil, ErrInvalidValues
			}
			return nil, fmt.Errorf("%w (%d)", ErrVectorDimTooHigh, s.maxVectorDim)
		}
		if query.K <= 0 {
			return nil, ErrInvalidK
		}
		if query.K > s.maxK {
			return nil, fmt.Errorf("%w (%d)", ErrKTooHigh, s.maxK)
		}
		if len(query.Values) > 0 && len(query.Values32) > 0 && len(query.Values) != len(query.Values32) {
			return nil, ErrInvalidValues
		}
		if err := s.validateSearchRequest(query.Values, query.K); err != nil && len(query.Values32) == 0 {
			return nil, err
		}
		queryID := strings.TrimSpace(query.ID)
		if queryID == "" {
			queryID = fmt.Sprintf("query-%d", i)
		}
		var query32 []float32
		if exactMode {
			query32 = query32Block[query32Offset : query32Offset+valuesLen]
			if len(query.Values32) > 0 {
				copy(query32, query.Values32)
			} else {
				fillQuery32Buffer(query32, query.Values)
			}
			query32Offset += valuesLen
		}
		values := query.Values
		if len(values) == 0 {
			values = *s.getBatchQuery64Buffer(valuesLen)
			for j, value := range query.Values32 {
				values[j] = float64(value)
			}
		}
		prepared = append(prepared, preparedBatchQuery{
			id:     queryID,
			vals:   values,
			vals32: query32,
			acc:    newTopKAccumulator(query.K),
		})
	}

	if s.searchMode == "ann" {
		results, err := s.searchBatchANNContext(ctx, prepared)
		for i, query := range queries {
			if len(query.Values) == 0 && len(prepared[i].vals) > 0 {
				s.putBatchQuery64Buffer(&prepared[i].vals)
			}
		}
		return results, err
	}
	if len(prepared) >= exactBatchDistanceWidth*2 && runtime.GOMAXPROCS(0) > 1 {
		return s.searchBatchExactParallelContext(ctx, prepared), ctx.Err()
	}
	return s.searchBatchExactSerialContext(ctx, prepared), ctx.Err()
}

func (s *Service) searchBatchExactParallel(prepared []preparedBatchQuery) []BatchSearchResult {
	return s.searchBatchExactParallelContext(context.Background(), prepared)
}

func (s *Service) searchBatchExactParallelContext(ctx context.Context, prepared []preparedBatchQuery) []BatchSearchResult {
	workers := min(runtime.GOMAXPROCS(0), len(prepared)/exactBatchDistanceWidth)
	if workers <= 1 {
		return s.searchBatchExactSerialContext(ctx, prepared)
	}
	chunkSize := (len(prepared) + workers - 1) / workers
	if rem := chunkSize % exactBatchDistanceWidth; rem != 0 {
		chunkSize += exactBatchDistanceWidth - rem
	}

	var wg sync.WaitGroup
	for start := 0; start < len(prepared); start += chunkSize {
		end := min(start+chunkSize, len(prepared))
		wg.Add(1)
		go func(batch []preparedBatchQuery) {
			defer wg.Done()
			s.scanExactBatchContext(ctx, batch)
		}(prepared[start:end])
	}
	wg.Wait()
	return batchResultsFromPrepared(prepared)
}

func (s *Service) searchBatchExactSerial(prepared []preparedBatchQuery) []BatchSearchResult {
	return s.searchBatchExactSerialContext(context.Background(), prepared)
}

func (s *Service) searchBatchExactSerialContext(ctx context.Context, prepared []preparedBatchQuery) []BatchSearchResult {
	s.scanExactBatchContext(ctx, prepared)
	return batchResultsFromPrepared(prepared)
}

func (s *Service) scanExactBatch(prepared []preparedBatchQuery) {
	s.scanExactBatchContext(context.Background(), prepared)
}

func (s *Service) scanExactBatchContext(ctx context.Context, prepared []preparedBatchQuery) {
	s.rangeExactVectors(func(id string, values []float32) bool {
		if ctx.Err() != nil {
			return false
		}
		for i := 0; i+exactBatchDistanceWidth <= len(prepared); i += exactBatchDistanceWidth {
			q0 := prepared[i].vals32
			q1 := prepared[i+1].vals32
			q2 := prepared[i+2].vals32
			q3 := prepared[i+3].vals32
			if len(q0) == len(values) && len(q1) == len(values) && len(q2) == len(values) && len(q3) == len(values) {
				d0, d1, d2, d3 := vector.SquaredEuclideanDistance32x4SameLen(q0, q1, q2, q3, values)
				prepared[i].acc.Add(SearchResult{ID: id, Distance: d0})
				prepared[i+1].acc.Add(SearchResult{ID: id, Distance: d1})
				prepared[i+2].acc.Add(SearchResult{ID: id, Distance: d2})
				prepared[i+3].acc.Add(SearchResult{ID: id, Distance: d3})
				continue
			}
			for j := 0; j < exactBatchDistanceWidth; j++ {
				dist := vector.SquaredEuclideanDistance32(prepared[i+j].vals32, values)
				if dist != dist {
					continue
				}
				prepared[i+j].acc.Add(SearchResult{ID: id, Distance: dist})
			}
		}
		for i := len(prepared) - len(prepared)%exactBatchDistanceWidth; i < len(prepared); i++ {
			dist := vector.SquaredEuclideanDistance32(prepared[i].vals32, values)
			if dist != dist {
				continue
			}
			prepared[i].acc.Add(SearchResult{ID: id, Distance: dist})
		}
		return true
	})
}

func batchResultsFromPrepared(prepared []preparedBatchQuery) []BatchSearchResult {
	results := make([]BatchSearchResult, len(prepared))
	for i, query := range prepared {
		results[i] = BatchSearchResult{
			ID:      query.id,
			Results: query.acc.Results(),
		}
	}
	return results
}

func (s *Service) searchBatchANN(prepared []preparedBatchQuery) ([]BatchSearchResult, error) {
	return s.searchBatchANNContext(context.Background(), prepared)
}

func (s *Service) searchBatchANNContext(ctx context.Context, prepared []preparedBatchQuery) ([]BatchSearchResult, error) {
	results := make([]BatchSearchResult, len(prepared))
	workers := min(len(prepared), batchSearchWorkerLimit())
	if workers <= 1 {
		for i, query := range prepared {
			hits, err := s.SearchContext(ctx, query.vals, query.acc.limit)
			if err != nil {
				return nil, err
			}
			results[i] = BatchSearchResult{ID: query.id, Results: hits}
		}
		return results, nil
	}

	jobs := make(chan int)
	var wg sync.WaitGroup
	var errOnce sync.Once
	var firstErr error
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				query := prepared[i]
				hits, err := s.SearchContext(ctx, query.vals, query.acc.limit)
				if err != nil {
					errOnce.Do(func() { firstErr = err })
					continue
				}
				results[i] = BatchSearchResult{ID: query.id, Results: hits}
			}
		}()
	}
	for i := range prepared {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	return results, nil
}

// batchSearchWorkerLimit bounds nested client/server concurrency while
// allowing an eight-vCPU node to use its full ANN search budget. The official
// c4/batch100 gate is materially faster at eight than at two or four workers.
func batchSearchWorkerLimit() int {
	return max(1, min(runtime.GOMAXPROCS(0), 8))
}

func (s *Service) validateSearchRequest(values []float64, k int) error {
	if len(values) == 0 {
		return ErrInvalidValues
	}
	if k <= 0 {
		return ErrInvalidK
	}
	if k > s.maxK {
		return fmt.Errorf("%w (%d)", ErrKTooHigh, s.maxK)
	}
	if len(values) > s.maxVectorDim {
		return fmt.Errorf("%w (%d)", ErrVectorDimTooHigh, s.maxVectorDim)
	}
	return nil
}

func (s *Service) searchExact(values []float64, k int) []SearchResult {
	results, _ := s.searchExactContext(context.Background(), values, k)
	return results
}

func (s *Service) searchExactContext(ctx context.Context, values []float64, k int) ([]SearchResult, error) {
	query32 := s.getQuery32Buffer(values)
	defer s.putQuery32Buffer(query32)

	acc := newTopKAccumulator(k)
	s.rangeExactVectors(func(id string, vecValues []float32) bool {
		if ctx.Err() != nil {
			return false
		}
		if len(vecValues) != len(*query32) {
			return true
		}
		dist := vector.SquaredEuclideanDistance32SameLen(*query32, vecValues)
		if dist == dist {
			acc.Add(SearchResult{ID: id, Distance: dist})
		}
		return true
	})
	return acc.Results(), ctx.Err()
}

func (s *Service) rangeExactVectors(fn func(id string, values []float32) bool) {
	// Persistent segment stores are the canonical payload source. Prefer their
	// native float32 scan so exact search does not traverse the compatibility
	// index (which otherwise keeps a second in-memory payload copy alive).
	if ranger, ok := s.vectorStore.(rangeVector32Reader); ok {
		ranger.RangeVectors32(fn)
		return
	}
	if ranger, ok := s.index.(vectorIndex32Ranger); ok {
		ranger.RangeVectors32(fn)
		return
	}
	s.index.RangeVectors(func(vec index.Vector) bool {
		return fn(vec.ID, vector.ToFloat32(vec.Values))
	})
}

func (s *Service) searchANN(values []float64, k int) ([]SearchResult, bool, error) {
	return s.searchANNContext(context.Background(), values, k)
}

func (s *Service) searchANNContext(ctx context.Context, values []float64, k int) ([]SearchResult, bool, error) {
	s.stats.annSearchesTotal.Add(1)
	annIndex := s.currentANNIndex()
	candidateBuf := s.getANNResultBuffer()
	exactDistances := false
	if exact, ok := annIndex.(interface{ ExactDistances() bool }); ok {
		exactDistances = exact.ExactDistances()
	}
	candidateLimit := k
	if !exactDistances {
		candidateLimit = k * 4
		if candidateLimit < k {
			candidateLimit = k
		}
		if candidateLimit > s.maxK*4 {
			candidateLimit = s.maxK * 4
		}
	}
	if s.annAdaptive && s.annMinCandidates > candidateLimit {
		candidateLimit = s.annMinCandidates
		maxInternalCandidates := s.maxK * 4
		if maxInternalCandidates < k {
			maxInternalCandidates = k
		}
		if candidateLimit > maxInternalCandidates {
			candidateLimit = maxInternalCandidates
		}
	}
	var candidates []ann.Result
	var err error
	if contextual, ok := annIndex.(interface {
		SearchWithDistancesContext(context.Context, []float64, int, []ann.Result) ([]ann.Result, error)
	}); ok && !s.annAdaptive {
		candidates, err = contextual.SearchWithDistancesContext(ctx, values, candidateLimit, *candidateBuf)
	} else if s.annAdaptive {
		if adaptive, ok := annIndex.(interface {
			SearchAdaptiveWithDistancesContext(context.Context, []float64, int, int, int, []ann.Result) ([]ann.Result, error)
		}); ok {
			candidates, err = adaptive.SearchAdaptiveWithDistancesContext(ctx, values, candidateLimit, s.annMinCandidates, s.annMaxProbe, *candidateBuf)
		} else if adaptive, ok := annIndex.(interface {
			SearchAdaptiveWithDistancesInto([]float64, int, int, int, []ann.Result) ([]ann.Result, error)
		}); ok {
			candidates, err = adaptive.SearchAdaptiveWithDistancesInto(values, candidateLimit, s.annMinCandidates, s.annMaxProbe, *candidateBuf)
		} else if contextual, ok := annIndex.(interface {
			SearchWithDistancesContext(context.Context, []float64, int, []ann.Result) ([]ann.Result, error)
		}); ok {
			candidates, err = contextual.SearchWithDistancesContext(ctx, values, candidateLimit, *candidateBuf)
		} else {
			candidates, err = annIndex.SearchWithDistancesInto(values, candidateLimit, *candidateBuf)
		}
	} else {
		candidates, err = annIndex.SearchWithDistancesInto(values, candidateLimit, *candidateBuf)
	}
	defer s.putANNResultBuffer(candidateBuf, candidates)
	if ctx.Err() != nil {
		return nil, false, ctx.Err()
	}
	if err != nil {
		s.stats.annSearchErrorsTotal.Add(1)
		return nil, false, nil
	}
	s.stats.annCandidatesReturned.Add(uint64(len(candidates)))

	var query32 *[]float32
	if !exactDistances {
		query32 = s.getQuery32Buffer(values)
		defer s.putQuery32Buffer(query32)
	}
	acc := newTopKAccumulator(k)
	for _, candidate := range candidates {
		id, ok, err := s.lookupID(candidate.ID)
		if err != nil {
			s.stats.annSearchErrorsTotal.Add(1)
			return nil, false, err
		}
		if !ok {
			continue
		}
		if exactDistances {
			if candidate.Distance == candidate.Distance {
				acc.Add(SearchResult{ID: id, Distance: candidate.Distance})
			}
			continue
		}
		payload, err := s.vectorValues32(id)
		if err != nil || len(payload) != len(*query32) {
			continue
		}
		distance := vector.SquaredEuclideanDistance32SameLen(*query32, payload)
		if distance != distance {
			continue
		}
		acc.Add(SearchResult{ID: id, Distance: distance})
	}
	results := acc.Results()
	if len(results) == 0 {
		return nil, false, nil
	}
	s.stats.annSearchHitsTotal.Add(1)
	return results, true, nil
}

func (s *Service) vectorValues32(id string) ([]float32, error) {
	if reader, ok := s.vectorStore.(readOnlyVector32Reader); ok {
		return reader.GetVectorReadOnly32(id)
	}
	vec, err := s.vectorStore.GetVector(id)
	if err != nil {
		return nil, err
	}
	return vector.ToFloat32(vec.Values), nil
}

func (s *Service) getANNResultBuffer() *[]ann.Result {
	got := s.annResultPool.Get()
	if got == nil {
		buf := make([]ann.Result, 0, minInt(maxIntOrOne(s.maxK), 64))
		return &buf
	}
	buf := got.(*[]ann.Result)
	*buf = (*buf)[:0]
	return buf
}

func (s *Service) putANNResultBuffer(buf *[]ann.Result, results []ann.Result) {
	results = results[:0]
	*buf = results
	s.annResultPool.Put(buf)
}

func (s *Service) getQuery32Buffer(values []float64) *[]float32 {
	got := s.query32Pool.Get()
	if got == nil {
		buf := make([]float32, len(values))
		fillQuery32Buffer(buf, values)
		return &buf
	}
	buf := got.(*[]float32)
	if cap(*buf) < len(values) {
		*buf = make([]float32, len(values))
	} else {
		*buf = (*buf)[:len(values)]
	}
	fillQuery32Buffer(*buf, values)
	return buf
}

func (s *Service) putQuery32Buffer(buf *[]float32) {
	*buf = (*buf)[:0]
	s.query32Pool.Put(buf)
}

func (s *Service) getBatchQuery32Buffer(size int) *[]float32 {
	got := s.batchQuery32Pool.Get()
	if got == nil {
		buf := make([]float32, size)
		return &buf
	}
	buf := got.(*[]float32)
	if cap(*buf) < size {
		*buf = make([]float32, size)
	} else {
		*buf = (*buf)[:size]
	}
	return buf
}

func (s *Service) putBatchQuery32Buffer(buf *[]float32) {
	*buf = (*buf)[:0]
	s.batchQuery32Pool.Put(buf)
}

func (s *Service) getBatchQuery64Buffer(size int) *[]float64 {
	got := s.batchQuery64Pool.Get()
	if got == nil {
		buf := make([]float64, size)
		return &buf
	}
	buf := got.(*[]float64)
	if cap(*buf) < size {
		*buf = make([]float64, size)
	} else {
		*buf = (*buf)[:size]
	}
	return buf
}

func (s *Service) putBatchQuery64Buffer(buf *[]float64) {
	*buf = (*buf)[:0]
	s.batchQuery64Pool.Put(buf)
}

func (s *Service) getPreparedBatchBuffer(size int) *[]preparedBatchQuery {
	got := s.batchPreparedPool.Get()
	if got == nil {
		buf := make([]preparedBatchQuery, 0, size)
		return &buf
	}
	buf := got.(*[]preparedBatchQuery)
	if cap(*buf) < size {
		*buf = make([]preparedBatchQuery, 0, size)
	} else {
		*buf = (*buf)[:0]
	}
	return buf
}

func (s *Service) putPreparedBatchBuffer(buf *[]preparedBatchQuery) {
	for i := range *buf {
		(*buf)[i] = preparedBatchQuery{}
	}
	*buf = (*buf)[:0]
	s.batchPreparedPool.Put(buf)
}

func fillQuery32Buffer(dst []float32, values []float64) {
	for i, value := range values {
		dst[i] = float32(value)
	}
}

func newTopKAccumulator(limit int) topKAccumulator {
	if limit <= 0 {
		return topKAccumulator{}
	}
	return topKAccumulator{
		limit: limit,
		items: make([]SearchResult, 0, limit),
	}
}

func (a *topKAccumulator) Add(item SearchResult) {
	if a.limit <= 0 {
		return
	}
	if len(a.items) < a.limit {
		a.push(item)
		return
	}
	if item.Distance >= a.items[0].Distance {
		return
	}
	a.replaceTop(item)
}

func (a *topKAccumulator) Results() []SearchResult {
	if len(a.items) == 0 {
		return nil
	}
	sort.Slice(a.items, func(i, j int) bool {
		return a.items[i].Distance < a.items[j].Distance
	})
	for i := range a.items {
		a.items[i].Distance = sqrtDistance(a.items[i].Distance)
	}
	return a.items
}

// ResultsRaw returns the heap contents sorted by distance without applying the
// squared-distance conversion used by ANN results. It is used by callers whose
// distance is already in the public metric scale.
func (a *topKAccumulator) ResultsRaw() []SearchResult {
	if len(a.items) == 0 {
		return nil
	}
	sort.Slice(a.items, func(i, j int) bool {
		if a.items[i].Distance == a.items[j].Distance {
			return a.items[i].ID < a.items[j].ID
		}
		return a.items[i].Distance < a.items[j].Distance
	})
	return a.items
}

func (a *topKAccumulator) push(item SearchResult) {
	a.items = append(a.items, item)
	upSearchResultMax(a.items, len(a.items)-1)
}

func (a *topKAccumulator) replaceTop(item SearchResult) {
	a.items[0] = item
	downSearchResultMax(a.items, 0)
}

func upSearchResultMax(h []SearchResult, j int) {
	for {
		i := (j - 1) / 2
		if i == j || h[j].Distance <= h[i].Distance {
			break
		}
		h[i], h[j] = h[j], h[i]
		j = i
	}
}

func downSearchResultMax(h []SearchResult, i int) {
	for {
		left := 2*i + 1
		if left >= len(h) {
			break
		}
		child := left
		right := left + 1
		if right < len(h) && h[right].Distance > h[left].Distance {
			child = right
		}
		if h[i].Distance >= h[child].Distance {
			break
		}
		h[i], h[child] = h[child], h[i]
		i = child
	}
}

func normalizeSearchMode(mode string) string {
	mode = strings.ToLower(strings.TrimSpace(mode))
	if mode != "ann" {
		return "exact"
	}
	return mode
}

func (s *Service) rollbackAddedVectors(ids []string) {
	for _, id := range ids {
		if internalID, err := s.assignID(id); err == nil {
			s.deleteANNVector(internalID)
			s.deleteMetricANNVectorLocked(internalID)
		}
		_ = s.vectorStore.DeleteVector(id)
		_ = s.index.DeleteVector(id)
	}
	// Do not rebuild ANN generations synchronously while unwinding a failed
	// ingest. Tombstones keep failed IDs invisible; normal maintenance can
	// compact them at the configured threshold.
}

func (s *Service) upsertVectors(vectors []index.Vector) error {
	if len(vectors) == 0 {
		return nil
	}
	if writer, ok := s.vectorStore.(batchVectorStore); ok {
		return writer.UpsertVectors(vectors)
	}
	for _, vec := range vectors {
		if err := s.vectorStore.UpsertVector(index.Vector{ID: vec.ID, Values: vec.Values}); err != nil {
			return err
		}
	}
	return nil
}

func (s *Service) restoreState() error {
	s.ensureRuntimeDeps()
	if s.usesPersistentVectorStore() {
		if err := s.loadVectorStoreState(); err != nil {
			return err
		}
		if err := s.loadMetadataSidecar(); err != nil {
			return err
		}
		s.persistOps = 0
		return nil
	}
	if err := s.loadSnapshot(); err != nil {
		return err
	}
	if err := s.replayWAL(); err != nil {
		return err
	}
	if err := s.saveSnapshot(); err != nil {
		return err
	}
	s.metadataMu.Lock()
	if s.snapshotPath != "" {
		if err := s.persistMetadata(); err != nil {
			s.metadataMu.Unlock()
			return err
		}
	}
	s.metadataMu.Unlock()
	if err := s.truncateWAL(); err != nil {
		return err
	}
	s.persistOps = 0
	return nil
}

func (s *Service) saveSnapshot() error {
	s.ensureRuntimeDeps()
	if s.usesPersistentVectorStore() {
		return nil
	}
	return s.persistenceBackend().SaveSnapshot(s.vectorStore.ListVectors())
}

func (s *Service) appendWAL(op walOp) error {
	return s.appendWALBatch([]walOp{op})
}

func (s *Service) appendWALBatch(ops []walOp) error {
	s.ensureRuntimeDeps()
	if len(ops) == 0 {
		return nil
	}
	if s.usesPersistentVectorStore() {
		return nil
	}
	backend := s.persistenceBackend()
	if batcher, ok := backend.(interface{ AppendWALBatch([]walOp) error }); ok {
		return batcher.AppendWALBatch(ops)
	}
	for _, op := range ops {
		if err := backend.AppendWAL(op); err != nil {
			return err
		}
	}
	return nil
}

func (s *Service) replayWAL() error {
	s.ensureRuntimeDeps()
	err := s.persistenceBackend().ReplayWAL(func(op walOp) error {
		switch op.Op {
		case "upsert":
			if op.ID == "" || len(op.Values) == 0 || len(op.Values) > s.maxVectorDim {
				return errSkipWALOp
			}
			vec := index.Vector{ID: op.ID, Values: op.Values}
			if err := s.vectorStore.UpsertVector(vec); err != nil {
				return err
			}
			if err := s.index.AddVector(vec); err != nil {
				if errors.Is(err, index.ErrVectorExists) {
					_ = s.vectorStore.UpsertVector(vec)
					_ = s.index.DeleteVector(op.ID)
					_ = s.index.AddVector(vec)
					return nil
				}
				return err
			}
		case "delete":
			if op.ID == "" {
				return errSkipWALOp
			}
			_ = s.vectorStore.DeleteVector(op.ID)
			_ = s.index.DeleteVector(op.ID)
		case "metadata":
			if op.ID == "" {
				return errSkipWALOp
			}
			s.metadataMu.Lock()
			if s.metadata == nil {
				s.metadata = make(map[string]map[string]string)
			}
			s.metadata[op.ID] = cloneMetadata(op.Metadata)
			s.metadataMu.Unlock()
			s.textIndex.addMetadata(op.ID, op.Metadata)
		case "metadata_batch":
			if len(op.MetadataEntries) == 0 {
				return errSkipWALOp
			}
			s.metadataMu.Lock()
			if s.metadata == nil {
				s.metadata = make(map[string]map[string]string)
			}
			for _, entry := range op.MetadataEntries {
				if entry.ID == "" {
					continue
				}
				s.metadata[entry.ID] = cloneMetadata(entry.Metadata)
				s.textIndex.addMetadata(entry.ID, entry.Metadata)
			}
			s.metadataMu.Unlock()
		}
		return nil
	})
	if err != nil {
		return err
	}
	s.rebuildANNLocked()
	return nil
}

func (s *Service) truncateWAL() error {
	s.ensureRuntimeDeps()
	if s.usesPersistentVectorStore() {
		return nil
	}
	return s.persistenceBackend().TruncateWAL()
}

func (s *Service) maybeSnapshot() error {
	s.ensureRuntimeDeps()
	if s.usesPersistentVectorStore() {
		s.persistOps = 0
		return nil
	}
	s.persistOps++
	if s.persistOps < s.snapshotEvery {
		return nil
	}
	if err := s.syncPersistence(); err != nil {
		return err
	}
	if err := s.saveSnapshot(); err != nil {
		return err
	}
	if err := s.truncateWAL(); err != nil {
		return err
	}
	s.persistOps = 0
	return nil
}

func (s *Service) loadSnapshot() error {
	s.ensureRuntimeDeps()
	payload, err := s.persistenceBackend().LoadSnapshot()
	if err != nil {
		return err
	}
	for id, values := range payload {
		if id == "" || len(values) == 0 || len(values) > s.maxVectorDim {
			continue
		}
		if err := s.vectorStore.UpsertVector(index.Vector{ID: id, Values: values}); err != nil {
			return err
		}
		if err := s.index.AddVector(index.Vector{ID: id, Values: values}); err != nil && !errors.Is(err, index.ErrVectorExists) {
			return err
		}
		internalID, err := s.assignID(id)
		if err != nil {
			return err
		}
		if err := s.addANNVector(internalID, values); err != nil {
			return err
		}
	}
	return nil
}

func (s *Service) rebuildANNLocked() {
	if err := s.rebuildANNCheckedLocked(); err != nil {
		s.annRebuildFailures.Add(1)
		message := err.Error()
		s.annRebuildLastError.Store(&message)
	} else {
		s.annRebuildSuccesses.Add(1)
		s.annRebuildLastError.Store(nil)
	}
}

func (s *Service) rebuildANNCheckedLocked() error {
	s.ensureRuntimeDeps()
	var vectors []index.Vector
	var nextIndex ANNIndex
	builderActive := false
	if s.annBackend == "hierarchical-hnsw" {
		type snapshotVector32 struct {
			id         string
			internalID int
		}
		snapshot := make([]snapshotVector32, 0, s.currentANNIndex().Stats().Nodes)
		var snapshotBytes uint64
		var snapshotErr error
		appendVector32 := func(id string, values []float32, borrowed bool) bool {
			nextBytes := snapshotBytes + uint64(len(values))*4
			if nextBytes < snapshotBytes || s.annRebuildSnapshotBudgetBytes > 0 && nextBytes > s.annRebuildSnapshotBudgetBytes {
				snapshotErr = fmt.Errorf("ANN rebuild snapshot payload exceeds budget %d", s.annRebuildSnapshotBudgetBytes)
				return false
			}
			snapshotBytes = nextBytes
			snapshot = append(snapshot, snapshotVector32{id: id})
			return true
		}
		if ranger, ok := s.index.(interface {
			RangeVectors32(func(string, []float32) bool)
		}); ok {
			ranger.RangeVectors32(func(id string, values []float32) bool { return appendVector32(id, values, true) })
		} else {
			for _, vector := range s.index.ListVectors() {
				values := make([]float32, len(vector.Values))
				for dimension, value := range vector.Values {
					values[dimension] = float32(value)
				}
				if !appendVector32(vector.ID, values, false) {
					break
				}
			}
		}
		// Map-backed indexes have intentionally unspecified iteration order.
		if snapshotErr != nil {
			return snapshotErr
		}
		// Resolve previously unseen IDs only after sorting the immutable
		// snapshot, otherwise identical rebuilds can produce different HNSW
		// insertion orders and materially different recall.
		sort.Slice(snapshot, func(left, right int) bool { return snapshot[left].id < snapshot[right].id })
		for position, vector := range snapshot {
			internalID, err := s.assignID(vector.id)
			if err != nil {
				return err
			}
			snapshot[position].internalID = internalID
		}
		sort.Slice(snapshot, func(left, right int) bool { return snapshot[left].internalID < snapshot[right].internalID })
		// persistMu protects the canonical source throughout this rebuild. Keep
		// only O(N) ID descriptors, not an additional O(N*D) payload snapshot.
		// The private builder copies each transient payload into its final arena.
		var scratch []float32
		candidate, err := ann.BuildHierarchicalIndexFromSource(s.annOptions, len(snapshot), func(position int) (ann.BatchVector32, error) {
			record := snapshot[position]
			if reader, ok := s.index.(interface {
				CopyVector32(string, []float32) ([]float32, error)
			}); ok {
				var err error
				scratch, err = reader.CopyVector32(record.id, scratch)
				return ann.BatchVector32{ID: record.internalID, Values: scratch}, err
			}
			v, err := s.index.SearchVector(record.id)
			if err != nil {
				return ann.BatchVector32{}, err
			}
			if cap(scratch) < len(v.Values) {
				scratch = make([]float32, len(v.Values))
			}
			scratch = scratch[:len(v.Values)]
			for i, value := range v.Values {
				scratch[i] = float32(value)
			}
			return ann.BatchVector32{ID: record.internalID, Values: scratch}, nil
		})
		if err != nil {
			return err
		}
		nextIndex = candidate
	} else {
		vectors = s.index.ListVectors()
	}
	if nextIndex == nil && s.annBuilder != nil {
		if candidate, err := s.annBuilder(vectors); err == nil && candidate != nil {
			nextIndex = candidate
			builderActive = true
		}
	}
	if nextIndex == nil {
		nextIndex = newConfiguredANNIndex(s.annBackend, s.annOptions, s.annSegmented, s.annSegmentMaxNodes, len(vectors))
	}
	// Builders may already populate the candidate. The compatibility indexes
	// need the replay, while a populated custom candidate must not duplicate it.
	if nextIndex.Stats().Nodes == 0 {
		for _, vec := range vectors {
			internalID, err := s.assignID(vec.ID)
			if err != nil {
				return err
			}
			if err := nextIndex.AddVector(internalID, vec.Values); err != nil {
				return err
			}
		}
	}
	if validator, ok := nextIndex.(interface{ Validate() error }); ok {
		if err := validator.Validate(); err != nil {
			return err
		}
	}
	s.annMu.Lock()
	previous := s.annIndex
	s.annIndex = nextIndex
	s.annBuilderActive = builderActive
	s.annMu.Unlock()
	if closer, ok := previous.(interface{ Close() error }); ok {
		_ = closer.Close()
	}
	return nil
}

func (s *Service) loadVectorStoreState() error {
	s.ensureRuntimeDeps()
	if s.loadANNCheckpointFromStore() {
		return nil
	}
	if recovery, ok := s.vectorStore.(interface {
		RecoveryIndexLoaded() bool
		RebuildRecoveryIndex() error
	}); ok && recovery.RecoveryIndexLoaded() {
		// A sidecar is only a startup accelerator. If the canonical vector
		// fingerprint does not agree with it, rebuild offsets from immutable
		// segments and retry before considering the ANN checkpoint unusable.
		if err := recovery.RebuildRecoveryIndex(); err != nil {
			return err
		}
		if s.loadANNCheckpointFromStore() {
			return nil
		}
	}
	vectors := s.vectorStore.ListVectors()
	checkpointLoaded := s.loadANNCheckpoint(vectors)
	for _, vec := range vectors {
		if vec.ID == "" || len(vec.Values) == 0 || len(vec.Values) > s.maxVectorDim {
			continue
		}
		if err := s.index.AddVector(vec); err != nil && !errors.Is(err, index.ErrVectorExists) {
			return err
		}
		if !checkpointLoaded {
			internalID, err := s.assignID(vec.ID)
			if err != nil {
				continue
			}
			if err := s.addANNVector(internalID, vec.Values); err != nil {
				continue
			}
		}
	}
	return nil
}

func (s *Service) Stats() ServiceStats {
	annStats := s.currentANNIndex().Stats()
	stats := ServiceStats{
		ANNRebuildFailures:     s.annRebuildFailures.Load(),
		ANNRebuildSuccesses:    s.annRebuildSuccesses.Load(),
		ShardCount:             1,
		SearchRequestsTotal:    s.stats.searchRequestsTotal.Load(),
		ExactSearchesTotal:     s.stats.exactSearchesTotal.Load(),
		ANNSearchesTotal:       s.stats.annSearchesTotal.Load(),
		ANNSearchHitsTotal:     s.stats.annSearchHitsTotal.Load(),
		ANNSearchFallbacks:     s.stats.annSearchFallbacks.Load(),
		ANNSearchErrorsTotal:   s.stats.annSearchErrorsTotal.Load(),
		ANNCandidatesReturned:  s.stats.annCandidatesReturned.Load(),
		ANNEvalSamplesTotal:    s.stats.annEvalSamplesTotal.Load(),
		ANNEvalTop1Matches:     s.stats.annEvalTop1Matches.Load(),
		ANNEvalOverlapResults:  s.stats.annEvalOverlapResults.Load(),
		ANNEvalComparedResults: s.stats.annEvalComparedResults.Load(),
		ANNNodes:               annStats.Nodes,
		ANNDeleted:             annStats.Deleted,
		ANNProfile:             normalizeANNProfile(s.annProfile),
		ANNM:                   s.annOptions.M,
		ANNEfConstruction:      s.annOptions.EfConstruction,
		ANNEfSearch:            s.annOptions.EfSearch,
		ANNSegmentRouting:      s.annOptions.SegmentRouting,
		ANNDiversifiedPruning:  s.annOptions.DiversifiedPruning,
		ANNCheckpointLoaded:    s.annCheckpointLoaded.Load(),
	}
	if message := s.annRebuildLastError.Load(); message != nil {
		stats.ANNRebuildLastError = *message
	}
	if memoryReader, ok := s.currentANNIndex().(interface{ MemoryStats() ann.MemoryStats }); ok {
		memory := memoryReader.MemoryStats()
		stats.ANNVectorBytes = memory.VectorBytes
		stats.ANNAdjacencyBytes = memory.AdjacencyBytes
		stats.ANNMapBytes = memory.MapBytes
		stats.ANNRouteBytes = memory.RouteBytes
		stats.ANNStructuralBytes = memory.TotalBytes
	}
	s.metricANNMu.RLock()
	stats.ANNMetricIndexes = len(s.metricANN)
	for _, metricIndex := range s.metricANN {
		metricStats := metricIndex.Stats()
		stats.ANNMetricNodes += metricStats.Nodes - metricStats.Deleted
		if memoryReader, ok := metricIndex.(interface{ MemoryStats() ann.MemoryStats }); ok {
			memory := memoryReader.MemoryStats()
			stats.ANNVectorBytes += memory.VectorBytes
			stats.ANNAdjacencyBytes += memory.AdjacencyBytes
			stats.ANNMapBytes += memory.MapBytes
			stats.ANNRouteBytes += memory.RouteBytes
			stats.ANNStructuralBytes += memory.TotalBytes
			stats.ANNMetricStructuralBytes += memory.TotalBytes
		}
	}
	s.metricANNMu.RUnlock()
	if segmented, ok := s.currentANNIndex().(interface{ SegmentCount() int }); ok {
		stats.ANNSegments = segmented.SegmentCount()
	} else if annStats.Nodes > 0 {
		stats.ANNSegments = 1
	}
	if maintenance, ok := s.currentANNIndex().(interface {
		MaintenanceState() (bool, bool, bool)
	}); ok {
		pending, compacting, compactable := maintenance.MaintenanceState()
		if pending {
			stats.ANNCompactionPending = 1
		}
		if compacting {
			stats.ANNCompacting = 1
		}
		if compactable {
			stats.ANNCompactionCompactable = 1
		}
	}
	if retirement, ok := s.currentANNIndex().(interface {
		RetirementState() (int, int, uint64, uint64)
	}); ok {
		stats.ANNReaders, stats.ANNRetiredSegments, stats.ANNReclaimedSegments, stats.ANNReclaimErrors = retirement.RetirementState()
	}
	if memoryBudget, ok := s.currentANNIndex().(interface {
		CompactionMemoryState() (uint64, uint64, uint64, uint64)
	}); ok {
		stats.ANNCompactionMemoryBudget, stats.ANNCompactionMemoryReserved, stats.ANNCompactionMemoryEstimate, stats.ANNCompactionMemoryDeferred = memoryBudget.CompactionMemoryState()
	}
	if searchBudget, ok := s.currentANNIndex().(interface {
		SearchBudgetState() (uint64, uint64, uint64)
	}); ok {
		stats.ANNSearchBudgetQueries, stats.ANNSearchBudgetSegments, stats.ANNSearchEFBudget = searchBudget.SearchBudgetState()
	}
	if searchConfig, ok := s.currentANNIndex().(interface {
		SearchConfigState() (string, string, int, bool)
	}); ok {
		stats.ANNSearchExecution, stats.ANNEFBudgetMode, stats.ANNEFGlobalPercent, stats.ANNPrimaryIndexEnabled = searchConfig.SearchConfigState()
	}
	if primaryBuild, ok := s.currentANNIndex().(interface {
		PrimaryBuildState() (bool, uint64, uint64, uint64, uint64)
	}); ok {
		stats.ANNPrimaryBuildActive, stats.ANNPrimaryBuildTotal, stats.ANNPrimaryBuildDone, stats.ANNPrimaryBuildDurationMs, stats.ANNPrimaryBuildFailures = primaryBuild.PrimaryBuildState()
	}
	if cacheStatsReader, ok := s.vectorStore.(interface{ Stats() CacheStats }); ok {
		cacheStats := cacheStatsReader.Stats()
		stats.CacheHitsTotal = cacheStats.Hits
		stats.CacheMissesTotal = cacheStats.Misses
		stats.CacheEvictionsTotal = cacheStats.Evictions
		stats.CacheItems = cacheStats.Items
		stats.CacheBytes = cacheStats.Bytes
	}
	if diskStatsReader, ok := s.vectorStore.(diskStatsReader); ok {
		diskStats := diskStatsReader.DiskStats()
		stats.DiskFileBytes = diskStats.FileBytes
		stats.DiskRecords = diskStats.Records
		stats.DiskStaleRecords = diskStats.StaleRecords
		stats.DiskCompactionsTotal = diskStats.Compactions
		stats.DiskCompactionActive = diskStats.CompactionActive
		stats.DiskSegments = diskStats.Segments
	}
	return stats
}

// ANNReady reports whether the published ANN generation contains the expected
// number of live vectors. It is used by benchmark adapters to separate
// durable ingestion from searchable index readiness.
func (s *Service) ANNReady(expected int) bool {
	if expected <= 0 {
		return false
	}
	stats := s.currentANNIndex().Stats()
	return stats.Nodes-stats.Deleted >= expected
}

func (s *Service) Close() error {
	return s.close(true)
}

// Checkpoint creates a durable recovery point while the service remains
// available. Writers are briefly serialized; searches continue against the
// immutable ANN generations.
func (s *Service) Checkpoint() error {
	s.persistMu.Lock()
	defer s.persistMu.Unlock()
	if err := s.syncPersistence(); err != nil {
		return err
	}
	if err := s.saveANNCheckpointFromStore(); err != nil {
		return err
	}
	s.annCheckpointDirty.Store(false)
	return nil
}

// CloseRetired closes a data plane that has already been replaced by a
// durably committed topology. Its persisted vector segments remain intact,
// but an ANN checkpoint is unnecessary because this instance will not be
// reopened as the active generation.
func (s *Service) CloseRetired() error {
	return s.close(false)
}

func (s *Service) close(writeANNCheckpoint bool) error {
	s.closeMu.Lock()
	defer s.closeMu.Unlock()
	if s.closed {
		return s.closeErr
	}
	s.closed = true
	syncErr := s.syncPersistence()
	var checkpointErr error
	if writeANNCheckpoint && s.annCheckpointDirty.Load() {
		checkpointErr = s.saveANNCheckpointFromStore()
		if checkpointErr == nil {
			s.annCheckpointDirty.Store(false)
		}
	}
	// A mapped ANN checkpoint must remain readable until the replacement
	// checkpoint has been fully written. Closing it first unmaps the immutable
	// columns and makes Stats/serialization invalid during graceful shutdown.
	var annErr error
	if closer, ok := s.currentANNIndex().(interface{ Close() error }); ok {
		annErr = closer.Close()
	}
	metricANNErr := s.closeMetricANN()
	var storeErr error
	var resolverErr error
	if closer, ok := s.idResolver.(interface{ Close() error }); ok {
		resolverErr = closer.Close()
	}
	if closer, ok := s.vectorStore.(interface{ Close() error }); ok {
		storeErr = closer.Close()
	}
	s.closeErr = errors.Join(syncErr, annErr, metricANNErr, checkpointErr, resolverErr, storeErr)
	return s.closeErr
}

func (s *Service) syncPersistence() error {
	var metadataErr error
	if s.snapshotPath != "" {
		if metadataErr = s.loadMetadataSidecar(); metadataErr == nil {
			s.metadataMu.Lock()
			metadataErr = s.persistMetadata()
			s.metadataMu.Unlock()
		}
	}
	if metadataErr != nil {
		return metadataErr
	}
	if syncer, ok := s.persistence.(interface{ Sync() error }); ok {
		return syncer.Sync()
	}
	return nil
}

func (s *Service) ensureRuntimeDeps() {
	if s.index == nil {
		s.index = index.NewIndex()
	}
	if s.vectorStore == nil {
		s.vectorStore = newMemoryVectorStore()
	}
	_ = s.currentANNIndex()
	if s.idResolver == nil {
		s.idResolver = newMemoryIDResolver()
	}
	if s.persistence == nil {
		s.persistence = newSnapshotWALBackend(s.snapshotPath, s.walPath)
	}
}

func (s *Service) currentANNIndex() ANNIndex {
	s.annMu.RLock()
	idx := s.annIndex
	s.annMu.RUnlock()
	if idx != nil {
		return idx
	}

	s.annMu.Lock()
	defer s.annMu.Unlock()
	if s.annIndex == nil {
		s.annIndex = newConfiguredANNIndex(s.annBackend, s.annOptions, s.annSegmented, s.annSegmentMaxNodes, 0)
	}
	return s.annIndex
}

func normalizeANNBackend(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "hierarchical-hnsw" {
		return value
	}
	return "hnsw"
}

func newConfiguredANNIndex(backend string, options ann.Options, segmented bool, segmentMaxNodes, capacity int) ANNIndex {
	if normalizeANNBackend(backend) == "hierarchical-hnsw" {
		return ann.NewHierarchicalIndexWithCapacity(options, capacity)
	}
	if segmented {
		return ann.NewSegmentedIndex(options, segmentMaxNodes)
	}
	return ann.NewAnnIndexWithOptions(options)
}

func (s *Service) addANNVector(internalID int, values []float64) error {
	return s.currentANNIndex().AddVector(internalID, values)
}

func (s *Service) addANNVectors(vectors []index.Vector) error {
	if s.annStagedIngestMinBatch > 0 && len(vectors) >= s.annStagedIngestMinBatch {
		if segmented, ok := s.currentANNIndex().(interface {
			BuildAndPublishBatch([]ann.BatchVector) error
		}); ok {
			batch := make([]ann.BatchVector, len(vectors))
			for i, vector := range vectors {
				internalID, err := s.assignID(vector.ID)
				if err != nil {
					return err
				}
				batch[i] = ann.BatchVector{ID: internalID, Values: vector.Values}
			}
			return segmented.BuildAndPublishBatch(batch)
		}
	}
	if segmented, ok := s.currentANNIndex().(interface{ AddBatch([]ann.BatchVector) error }); ok {
		chunkSize := s.annSegmentMaxNodes
		if chunkSize <= 0 {
			chunkSize = 10000
		}
		for start := 0; start < len(vectors); start += chunkSize {
			end := start + chunkSize
			if end > len(vectors) {
				end = len(vectors)
			}
			batch := make([]ann.BatchVector, end-start)
			for i, vector := range vectors[start:end] {
				internalID, err := s.assignID(vector.ID)
				if err != nil {
					return err
				}
				batch[i] = ann.BatchVector{ID: internalID, Values: vector.Values}
			}
			if err := segmented.AddBatch(batch); err != nil {
				return err
			}
		}
		return nil
	}
	for _, vector := range vectors {
		internalID, err := s.assignID(vector.ID)
		if err != nil {
			return err
		}
		if err := s.addANNVector(internalID, vector.Values); err != nil {
			return err
		}
	}
	return nil
}

func (s *Service) deleteANNVector(internalID int) {
	s.currentANNIndex().DeleteVector(internalID)
}

func (s *Service) buildMetricANNLocked(metric DistanceMetric) (ANNIndex, error) {
	options := s.annOptions
	options.Metric = string(metric)
	if s.annBackend == "hierarchical-hnsw" {
		batch := make([]ann.BatchVector32, 0, 10000)
		var buildErr error
		s.rangeExactVectors(func(id string, values []float32) bool {
			internalID, err := s.assignID(id)
			if err == nil {
				batch = append(batch, ann.BatchVector32{ID: internalID, Values: append([]float32(nil), values...)})
			}
			if err != nil {
				buildErr = err
				return false
			}
			return true
		})
		if buildErr != nil {
			return nil, buildErr
		}
		sort.Slice(batch, func(left, right int) bool { return batch[left].ID < batch[right].ID })
		return ann.BuildHierarchicalIndex(options, batch)
	}
	options.QuantizeSegments = true
	options.DiversifiedPruning = true
	options.SegmentRouting = s.annOptions.SegmentRouting
	segmentMaxNodes := s.annSegmentMaxNodes
	if segmentMaxNodes <= 0 {
		segmentMaxNodes = 10000
	}
	candidate := ann.NewSegmentedIndex(options, segmentMaxNodes)
	batch := make([]ann.BatchVector32, 0, segmentMaxNodes)
	flush := func() error {
		if len(batch) == 0 {
			return nil
		}
		if err := candidate.AddBatch32(batch); err != nil {
			return err
		}
		batch = batch[:0]
		return nil
	}
	appendVector := func(id string, values []float32) error {
		internalID, err := s.assignID(id)
		if err != nil {
			return err
		}
		batch = append(batch, ann.BatchVector32{ID: internalID, Values: append([]float32(nil), values...)})
		if len(batch) >= segmentMaxNodes {
			return flush()
		}
		return nil
	}
	var buildErr error
	if reader, ok := s.vectorStore.(rangeVector32Reader); ok {
		reader.RangeVectors32(func(id string, values []float32) bool {
			if err := appendVector(id, values); err != nil {
				buildErr = err
				return false
			}
			return true
		})
	} else {
		for _, vector := range s.vectorStore.ListVectors() {
			values := make([]float32, len(vector.Values))
			for dimension, value := range vector.Values {
				values[dimension] = float32(value)
			}
			if err := appendVector(vector.ID, values); err != nil {
				buildErr = err
				break
			}
		}
	}
	if buildErr == nil {
		buildErr = flush()
	}
	if buildErr != nil {
		_ = candidate.Close()
		return nil, buildErr
	}
	return candidate, nil
}

func (s *Service) metricANNIndex(metric DistanceMetric) (ANNIndex, error) {
	s.metricANNMu.RLock()
	existing := s.metricANN[metric]
	s.metricANNMu.RUnlock()
	if existing != nil {
		return existing, nil
	}
	s.persistMu.Lock()
	defer s.persistMu.Unlock()
	s.metricANNMu.RLock()
	existing = s.metricANN[metric]
	s.metricANNMu.RUnlock()
	if existing != nil {
		return existing, nil
	}
	candidate, err := s.buildMetricANNLocked(metric)
	if err != nil {
		return nil, err
	}
	s.metricANNMu.Lock()
	if s.metricANN == nil {
		s.metricANN = make(map[DistanceMetric]ANNIndex, 2)
	}
	if existing = s.metricANN[metric]; existing == nil {
		s.metricANN[metric] = candidate
		existing = candidate
		candidate = nil
	}
	s.metricANNMu.Unlock()
	if candidate != nil {
		if closer, ok := candidate.(interface{ Close() error }); ok {
			_ = closer.Close()
		}
	}
	return existing, nil
}

// WarmMetricANN builds configured derived metric generations before the
// service is marked ready, preventing the first user query from paying the
// construction cost. L2 uses the primary generation and needs no warmup.
func (s *Service) WarmMetricANN(metrics []DistanceMetric) error {
	if s.searchMode != "ann" {
		return nil
	}
	for _, metric := range metrics {
		if err := validateDistanceMetric(metric); err != nil {
			return err
		}
		if metric == MetricL2 {
			continue
		}
		if _, err := s.metricANNIndex(metric); err != nil {
			return fmt.Errorf("warm metric ANN %s: %w", metric, err)
		}
	}
	return nil
}

func (s *Service) addMetricANNVectorsLocked(vectors []index.Vector) error {
	s.metricANNMu.RLock()
	defer s.metricANNMu.RUnlock()
	if len(s.metricANN) == 0 {
		return nil
	}
	batch := make([]ann.BatchVector, len(vectors))
	for position, vector := range vectors {
		internalID, err := s.assignID(vector.ID)
		if err != nil {
			return err
		}
		batch[position] = ann.BatchVector{ID: internalID, Values: vector.Values}
	}
	for _, metricIndex := range s.metricANN {
		if batcher, ok := metricIndex.(interface{ AddBatch([]ann.BatchVector) error }); ok {
			if err := batcher.AddBatch(batch); err != nil {
				return err
			}
			continue
		}
		for _, vector := range batch {
			if err := metricIndex.AddVector(vector.ID, vector.Values); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *Service) addMetricANNVectors32Locked(vectors []vectorBatch32) error {
	s.metricANNMu.RLock()
	defer s.metricANNMu.RUnlock()
	if len(s.metricANN) == 0 {
		return nil
	}
	batch := make([]ann.BatchVector32, len(vectors))
	for position, vector := range vectors {
		internalID, err := s.assignID(vector.ID)
		if err != nil {
			return err
		}
		batch[position] = ann.BatchVector32{ID: internalID, Values: vector.Values}
	}
	for _, metricIndex := range s.metricANN {
		if batcher, ok := metricIndex.(interface {
			AddBatch32([]ann.BatchVector32) error
		}); ok {
			if err := batcher.AddBatch32(batch); err != nil {
				return err
			}
			continue
		}
		for _, vector := range batch {
			values := make([]float64, len(vector.Values))
			for dimension, value := range vector.Values {
				values[dimension] = float64(value)
			}
			if err := metricIndex.AddVector(vector.ID, values); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *Service) deleteMetricANNVectorLocked(internalID int) {
	s.metricANNMu.RLock()
	defer s.metricANNMu.RUnlock()
	for _, metricIndex := range s.metricANN {
		metricIndex.DeleteVector(internalID)
	}
}

func (s *Service) invalidateMetricANNLocked() {
	s.metricANNMu.Lock()
	indexes := s.metricANN
	s.metricANN = nil
	s.metricANNMu.Unlock()
	for _, metricIndex := range indexes {
		if closer, ok := metricIndex.(interface{ Close() error }); ok {
			_ = closer.Close()
		}
	}
}

func (s *Service) closeMetricANN() error {
	s.metricANNMu.Lock()
	indexes := s.metricANN
	s.metricANN = nil
	s.metricANNMu.Unlock()
	var errs []error
	for _, metricIndex := range indexes {
		if closer, ok := metricIndex.(interface{ Close() error }); ok {
			errs = append(errs, closer.Close())
		}
	}
	return errors.Join(errs...)
}

func (s *Service) maybeCompactANNAfterDeleteLocked() {
	stats := s.currentANNIndex().Stats()
	if stats.Deleted < annDeleteRebuildMinDeleted {
		return
	}
	if stats.Nodes == 0 || float64(stats.Deleted)/float64(stats.Nodes) < annDeleteRebuildRatio {
		return
	}
	s.rebuildANNLocked()
}

func (s *Service) persistenceBackend() PersistenceBackend {
	if backend, ok := s.persistence.(*snapshotWALBackend); ok {
		if backend.snapshotPath != s.snapshotPath || backend.walPath != s.walPath {
			s.persistence = newSnapshotWALBackendWithOptions(s.snapshotPath, s.walPath, backend.security, s.syncEvery)
		}
	}
	if s.persistence == nil {
		s.persistence = newSnapshotWALBackendWithOptions(s.snapshotPath, s.walPath, DefaultStorageSecurityOptions(), s.syncEvery)
	}
	return s.persistence
}

func (s *Service) usesPersistentVectorStore() bool {
	persistent, ok := s.vectorStore.(persistentVectorStore)
	return ok && persistent.IsPersistent()
}

func newDefaultVectorStore(mode, path string, security ...StorageSecurityOptions) VectorStore {
	storeSecurity := DefaultStorageSecurityOptions()
	if len(security) > 0 {
		storeSecurity = normalizeStorageSecurityOptions(security[0])
	}
	return newDefaultVectorStoreWithOptions(mode, path, storeSecurity, 1)
}

func newDefaultVectorStoreWithOptions(mode, path string, security StorageSecurityOptions, syncEvery int) VectorStore {
	return newDefaultVectorStoreWithLocationCapacity(mode, path, security, syncEvery, 0)
}

func newDefaultVectorStoreWithLocationCapacity(mode, path string, security StorageSecurityOptions, syncEvery int, locationCapacity uint64) VectorStore {
	storeSecurity := normalizeStorageSecurityOptions(security)
	syncEvery = normalizeSyncEvery(syncEvery)
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "segment", "segmented":
		if strings.TrimSpace(path) == "" {
			path = "./data/vectors"
		}
		return newSegmentVectorStoreWithLocationCapacity(path, storeSecurity, locationCapacity)
	case "disk", "file":
		if strings.TrimSpace(path) == "" {
			path = "./data/vectors"
		}
		return newFileVectorStoreWithOptions(path, storeSecurity, syncEvery)
	default:
		return newMemoryVectorStore()
	}
}

func normalizeSyncEvery(syncEvery int) int {
	if syncEvery <= 0 {
		return 1
	}
	return syncEvery
}

func storageSecurityOptionsFromStrings(strict bool, dirMode, fileMode string) StorageSecurityOptions {
	opts := DefaultStorageSecurityOptions()
	if strict {
		opts = StrictStorageSecurityOptions()
	}
	opts.DirMode = ParseFileMode(dirMode, opts.DirMode)
	opts.FileMode = ParseFileMode(fileMode, opts.FileMode)
	opts.StrictFilePermissions = strict
	return normalizeStorageSecurityOptions(opts)
}

func storagePathMode(path string) (os.FileMode, error) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	return info.Mode().Perm(), nil
}

func normalizeANNProfile(profile string) string {
	switch strings.ToLower(strings.TrimSpace(profile)) {
	case "fast":
		return "fast"
	case "quality":
		return "quality"
	case "custom":
		return "custom"
	default:
		return "balanced"
	}
}

func applyANNProfile(profile string, options ann.Options) ann.Options {
	if strings.EqualFold(strings.TrimSpace(profile), "custom") {
		return options
	}
	// Profiles provide deterministic defaults while allowing any explicit
	// positive option to override the profile's value.
	// Preserve advanced and backend-specific options. Profiles own only the
	// three tuning defaults below; rebuilding Options with a struct literal
	// silently disabled metric, reciprocal pruning and other opt-in behavior.
	defaults := options
	defaults.M = 16
	defaults.EfConstruction = 64
	defaults.EfSearch = 64
	switch normalizeANNProfile(profile) {
	case "fast":
		defaults.M = 8
		defaults.EfConstruction = 32
		defaults.EfSearch = 32
	case "quality":
		defaults.M = 32
		defaults.EfConstruction = 128
		defaults.EfSearch = 128
	}
	if options.M > 0 {
		defaults.M = options.M
	}
	if options.EfConstruction > 0 {
		defaults.EfConstruction = options.EfConstruction
	}
	if options.EfSearch > 0 {
		defaults.EfSearch = options.EfSearch
	}
	return defaults
}

func clampPercent(v int) int {
	if v < 0 {
		return 0
	}
	if v > 100 {
		return 100
	}
	return v
}

func (s *Service) shouldEvaluateANN() bool {
	if s.annEvalSampleRate <= 0 {
		return false
	}
	n := s.stats.annSearchesTotal.Load()
	return int(n%100) < s.annEvalSampleRate
}

func (s *Service) maybeEvaluateANN(values []float64, k int, annResults []SearchResult) {
	if !s.shouldEvaluateANN() {
		return
	}
	exactResults := s.searchExact(values, k)
	s.stats.annEvalSamplesTotal.Add(1)
	if len(annResults) > 0 && len(exactResults) > 0 && annResults[0].ID == exactResults[0].ID {
		s.stats.annEvalTop1Matches.Add(1)
	}

	exactIDs := make(map[string]struct{}, len(exactResults))
	for _, result := range exactResults {
		exactIDs[result.ID] = struct{}{}
	}
	compared := minInt(len(annResults), len(exactResults))
	overlap := 0
	for _, result := range annResults {
		if _, ok := exactIDs[result.ID]; ok {
			overlap++
		}
	}
	s.stats.annEvalOverlapResults.Add(uint64(overlap))
	s.stats.annEvalComparedResults.Add(nonNegativeUint64(int64(compared)))
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxIntOrOne(value int) int {
	if value < 1 {
		return 1
	}
	return value
}

func sqrtDistance(distance float64) float64 {
	return math.Sqrt(distance)
}
