package ann

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const defaultSegmentMaxNodes = 10000
const defaultCompactThreshold = 8
const defaultCompactDebounce = 500 * time.Millisecond
const defaultCompactionBudget = 250 * time.Millisecond
const (
	defaultRouteMinSegments        = 8
	defaultRouteFullFanoutSegments = 16
)
const maxSegmentSearchWorkers = 4

var globalANNCompactionReserved atomic.Uint64
var primaryBuildGateOnce sync.Once
var primaryBuildGate chan struct{}

func primaryBuildConcurrency() int {
	value, err := strconv.Atoi(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_PRIMARY_BUILD_CONCURRENCY")))
	if err != nil || value < 1 || value > 16 {
		return 1
	}
	return value
}

func acquirePrimaryBuild(ctx context.Context) error {
	primaryBuildGateOnce.Do(func() {
		primaryBuildGate = make(chan struct{}, primaryBuildConcurrency())
	})
	select {
	case primaryBuildGate <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func releasePrimaryBuild() {
	<-primaryBuildGate
}

// SegmentedIndex limits write/query contention by keeping completed ANN graphs
// stable and directing new writes to a small mutable delta graph.
type SegmentedIndex struct {
	mu                         sync.RWMutex
	ingestMu                   sync.Mutex
	segments                   []*AnnIndex
	retired                    []*AnnIndex
	mappedOwner                snapshotMapping
	delta                      *AnnIndex
	deltaNodes                 int
	segmentMaxNodes            int
	options                    Options
	compactThreshold           int
	compactCh                  chan struct{}
	stopCh                     chan struct{}
	stopOnce                   sync.Once
	workerWG                   sync.WaitGroup
	mutationEpoch              uint64
	segmentEpoch               uint64
	primaryBuildRequestedEpoch uint64
	primaryBuildPublishedEpoch uint64
	routeEpoch                 uint64
	routeKeys                  []uint32
	routeOffsets               []uint32
	routeSegments              []uint32
	routePostingBytes          uint64
	closed                     bool
	pending                    atomic.Bool
	compacting                 atomic.Bool
	compactionBudget           time.Duration
	compactionStream           bool
	compactionMemoryBudget     uint64
	compactionMemoryReserved   atomic.Uint64
	compactionMemoryEstimate   atomic.Uint64
	compactionMemoryDeferred   atomic.Uint64
	searchExecution            string
	globalEFSearch             bool
	globalEFPercent            int
	primaryIndexEnabled        bool
	primaryBuildActive         atomic.Bool
	primaryBuildTotal          atomic.Uint64
	primaryBuildDone           atomic.Uint64
	primaryBuildDurationMs     atomic.Uint64
	primaryBuildFailures       atomic.Uint64
	searchQueries              atomic.Uint64
	searchSegments             atomic.Uint64
	searchEFBudget             atomic.Uint64
	searchWorkerPool           sync.Pool
	adaptiveCacheMu            sync.Mutex
	adaptiveCache              map[uint64]int
	readerCond                 *sync.Cond
	activeReaders              int
	reclaimRetired             bool
	reclaimedSegments          atomic.Uint64
	reclaimErrors              atomic.Uint64
}

type BatchVector struct {
	ID     int
	Values []float64
}

// BatchVector32 is the zero-copy-friendly internal representation used when
// moving already materialized ANN payloads between segments.
type BatchVector32 struct {
	ID     int
	Values []float32
}

// ExportSnapshot is a consistent ownership view used by online resharding.
// Sealed segment pointers are immutable; the mutable tail is copied while the
// ingest barrier is held, so a rotation cannot make one vector appear in both
// views.
type ExportSnapshot struct {
	Segments []*AnnIndex
	Tail     []BatchVector32
	Epoch    uint64
	Release  func()
}

// SnapshotExport captures sealed generations and the current delta atomically
// with respect to ingestion and segment rotation. The returned segment graphs
// remain immutable and can be streamed after this method returns.
func (s *SegmentedIndex) SnapshotExport() ExportSnapshot {
	s.ingestMu.Lock()
	defer s.ingestMu.Unlock()
	s.mu.Lock()
	release, ok := s.pinReaderLocked()
	if !ok {
		s.mu.Unlock()
		return ExportSnapshot{Release: func() {}}
	}
	segments := append([]*AnnIndex(nil), s.segments...)
	delta := s.delta
	epoch := s.mutationEpoch
	s.mu.Unlock()
	var tail []BatchVector32
	if delta != nil {
		tail = delta.ExportLiveVectors32()
	}
	return ExportSnapshot{Segments: segments, Tail: tail, Epoch: epoch, Release: release}
}

// ExportLiveVectors32 returns canonical float32 vectors without widening them
// to float64. The returned slices are copies owned by the caller and can be
// transferred to another segment without a precision conversion.
func (a *AnnIndex) ExportLiveVectors32() []BatchVector32 {
	a.mu.RLock()
	defer a.mu.RUnlock()
	count := a.nodeCountLocked()
	out := make([]BatchVector32, 0, count-a.deletedCount)
	for slot := 0; slot < count; slot++ {
		if a.isDeletedLocked(slot) {
			continue
		}
		out = append(out, BatchVector32{ID: a.nodeIDLocked(slot), Values: append([]float32(nil), a.nodeVectorLocked(slot)...)})
	}
	return out
}

// VisitLiveVectors32 streams immutable vectors without widening them or
// allocating a segment-sized result. The values slice is read-only and valid
// only for the duration of the callback.
func (a *AnnIndex) VisitLiveVectors32(visit func(id int, values []float32) error) error {
	if visit == nil {
		return errors.New("live vector visitor is required")
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	count := a.nodeCountLocked()
	var decoded []float32
	if a.quantized {
		decoded = make([]float32, a.dim)
	}
	for slot := 0; slot < count; slot++ {
		if a.isDeletedLocked(slot) {
			continue
		}
		var values []float32
		if a.quantized {
			for dimension, value := range a.qvectorLocked(slot) {
				decoded[dimension] = a.quantMin[dimension] +
					(a.quantMax[dimension]-a.quantMin[dimension])*(float32(value)/255)
			}
			values = decoded
		} else {
			values = a.nodes[slot].vector
		}
		if err := visit(a.nodeIDLocked(slot), values); err != nil {
			return err
		}
	}
	return nil
}

// MemoryStats aggregates the structural ANN estimate across immutable
// segments and the mutable delta. Runtime/transport allocations are excluded.
func (s *SegmentedIndex) MemoryStats() MemoryStats {
	indexes, release, ok := s.pinIndexes(true)
	if !ok {
		return MemoryStats{}
	}
	defer release()
	var total MemoryStats
	for _, idx := range indexes {
		if idx == nil {
			continue
		}
		stats := idx.MemoryStats()
		total.VectorBytes += stats.VectorBytes
		total.QuantizedBytes += stats.QuantizedBytes
		total.NodeBytes += stats.NodeBytes
		total.AdjacencyBytes += stats.AdjacencyBytes
		total.MapBytes += stats.MapBytes
		total.MappedBytes += stats.MappedBytes
		total.RouteBytes += stats.RouteBytes
	}
	s.mu.RLock()
	total.RouteBytes += s.routePostingBytes
	s.mu.RUnlock()
	total.TotalBytes = total.VectorBytes + total.QuantizedBytes + total.NodeBytes + total.AdjacencyBytes + total.MapBytes + total.RouteBytes
	return total
}

func (s *SegmentedIndex) ExactDistances() bool { return !s.options.QuantizeSegments }

func (s *SegmentedIndex) SetSegmentRouting(enabled bool) {
	s.mu.Lock()
	s.options.SegmentRouting = enabled
	s.mu.Unlock()
}

func (s *SegmentedIndex) SetDiversifiedPruning(enabled bool) {
	s.mu.Lock()
	s.options.DiversifiedPruning = enabled
	s.delta.SetDiversifiedPruning(enabled)
	s.mu.Unlock()
}

// EnableSegmentQuantization converts all sealed generations and enables
// quantization for future rotations. The mutable delta remains float32 so
// writes never need to mutate an immutable quantization range.
func (s *SegmentedIndex) EnableSegmentQuantization() error {
	s.ingestMu.Lock()
	defer s.ingestMu.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, segment := range s.segments {
		if err := segment.Quantize(); err != nil {
			return err
		}
	}
	s.options.QuantizeSegments = true
	return nil
}

func NewSegmentedIndex(opts Options, segmentMaxNodes int) *SegmentedIndex {
	if segmentMaxNodes <= 0 {
		segmentMaxNodes = defaultSegmentMaxNodes
	}
	idx := &SegmentedIndex{
		segmentMaxNodes:        segmentMaxNodes,
		options:                opts,
		compactThreshold:       compactThresholdFromEnv(),
		compactionBudget:       compactionBudgetFromEnv(),
		compactionStream:       compactionStreamingFromEnv(),
		compactionMemoryBudget: compactionMemoryBudgetFromEnv(),
		searchExecution:        searchExecutionFromEnv(),
		globalEFSearch:         globalEFSearchFromEnv(),
		globalEFPercent:        globalEFPercentFromEnv(),
		primaryIndexEnabled:    primaryIndexFromEnv(),
		compactCh:              make(chan struct{}, 1),
		stopCh:                 make(chan struct{}),
		reclaimRetired:         reclaimRetiredFromEnv(),
	}
	idx.readerCond = sync.NewCond(&idx.mu)
	idx.delta = NewAnnIndexWithOptions(idx.ingestOptions())
	idx.workerWG.Add(1)
	go idx.compactionWorker()
	return idx
}

func reclaimRetiredFromEnv() bool {
	value := strings.TrimSpace(os.Getenv("LUMENVEC_RECLAIM_RETIRED"))
	if value == "" {
		return true
	}
	enabled, err := strconv.ParseBool(value)
	return err == nil && enabled
}

func (s *SegmentedIndex) pinReaderLocked() (func(), bool) {
	if s.closed {
		return func() {}, false
	}
	s.activeReaders++
	// Every pin is released exactly once by the owning search/snapshot path
	// (normally through defer). Avoid allocating a sync.Once and closure per
	// query; callers must not invoke the returned release more than once.
	return s.releaseReader, true
}

func (s *SegmentedIndex) releaseReader() {
	var reclaim []*AnnIndex
	s.mu.Lock()
	if s.activeReaders > 0 {
		s.activeReaders--
	}
	if s.activeReaders == 0 {
		s.readerCond.Broadcast()
		if s.reclaimRetired && !s.closed {
			reclaim = s.retired
			s.retired = nil
		}
	}
	s.mu.Unlock()
	s.closeReclaimed(reclaim)
}

func (s *SegmentedIndex) closeReclaimed(indexes []*AnnIndex) {
	for _, index := range indexes {
		if index == nil {
			continue
		}
		if err := index.Close(); err != nil {
			s.reclaimErrors.Add(1)
		}
		s.reclaimedSegments.Add(1)
	}
}

func (s *SegmentedIndex) pinIndexes(includeDelta bool) ([]*AnnIndex, func(), bool) {
	s.mu.Lock()
	release, ok := s.pinReaderLocked()
	if !ok {
		s.mu.Unlock()
		return nil, release, false
	}
	indexes := append([]*AnnIndex(nil), s.segments...)
	if includeDelta {
		indexes = append(indexes, s.delta)
	}
	s.mu.Unlock()
	return indexes, release, true
}

// singleIndexPinned returns the common steady-state generation without
// materializing a routed slice. The reader pin keeps the segment alive while
// the caller executes the search; if a mutable delta exists, the general
// routed path is required so writes remain immediately searchable.
func (s *SegmentedIndex) singleIndexPinned() (*AnnIndex, bool) {
	s.mu.Lock()
	release, ok := s.pinReaderLocked()
	if !ok {
		s.mu.Unlock()
		return nil, false
	}
	if len(s.segments) != 1 || s.deltaNodes != 0 || s.segments[0] == nil {
		s.mu.Unlock()
		release()
		return nil, false
	}
	index := s.segments[0]
	s.mu.Unlock()
	return index, true
}

func (s *SegmentedIndex) RetirementState() (readers, retired int, reclaimed, reclaimErrors uint64) {
	s.mu.RLock()
	readers = s.activeReaders
	retired = len(s.retired)
	s.mu.RUnlock()
	return readers, retired, s.reclaimedSegments.Load(), s.reclaimErrors.Load()
}

// compactThresholdFromEnv lets bulk-ingest deployments defer consolidation
// until more immutable generations have accumulated. The lower bound keeps
// accidental values from disabling maintenance altogether; the default
// remains conservative for existing installations.
func compactThresholdFromEnv() int {
	threshold, err := strconv.Atoi(os.Getenv("LUMENVEC_COMPACTION_THRESHOLD"))
	if err != nil || threshold < 2 {
		return defaultCompactThreshold
	}
	return threshold
}

func compactionBudgetFromEnv() time.Duration {
	ms, err := strconv.Atoi(os.Getenv("LUMENVEC_COMPACTION_BUDGET_MS"))
	if err != nil || ms <= 0 {
		return defaultCompactionBudget
	}
	return time.Duration(ms) * time.Millisecond
}

func compactionStreamingFromEnv() bool {
	strategy := strings.ToLower(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_COMPACTION_STRATEGY")))
	return strategy != "copy"
}

func compactionMemoryBudgetFromEnv() uint64 {
	value := strings.TrimSpace(os.Getenv("LUMENVEC_ANN_COMPACTION_MEMORY_BUDGET_BYTES"))
	if value == "" {
		return 0
	}
	budget, err := strconv.ParseUint(value, 10, 64)
	if err != nil {
		return 0
	}
	return budget
}

func searchExecutionFromEnv() string {
	mode := strings.ToLower(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_SEARCH_EXECUTION")))
	switch mode {
	case "adaptive", "inline":
		return mode
	default:
		return "legacy"
	}
}

func adaptiveInitialPercentFromEnv() int {
	percent, err := strconv.Atoi(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_ADAPTIVE_INITIAL_PERCENT")))
	if err == nil && percent >= 10 && percent <= 100 {
		return percent
	}
	return 50
}

func adaptiveInitialEFFromEnv(segmentEF, k int) int {
	minEF := 64
	if value, err := strconv.Atoi(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_DYNAMIC_EF_MIN"))); err == nil && value > 0 {
		minEF = value
	}
	factor := 8
	if value, err := strconv.Atoi(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_DYNAMIC_EF_FACTOR"))); err == nil && value > 0 {
		factor = value
	}
	return max(k, min(segmentEF, max(minEF, k*factor)))
	/* legacy percentage/absolute tuning retained below for compatibility */
	/*
		if ef, err := strconv.Atoi(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_ADAPTIVE_INITIAL_EF"))); err == nil && ef > 0 {
			return max(k, min(segmentEF, ef))
		}
		return max(k, segmentEF*adaptiveInitialPercentFromEnv()/100)
	*/
}

func adaptiveMarginThresholdFromEnv() float64 {
	value, err := strconv.ParseFloat(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_ADAPTIVE_MARGIN_THRESHOLD")), 64)
	if err == nil && value >= 0 && value <= 1 {
		return value
	}
	return 0.02
}

func adaptiveCacheEnabled() bool {
	return false
}

func adaptiveQueryKey(query []float64, epoch uint64) uint64 {
	h := fnv.New64a()
	for _, value := range query {
		_, _ = h.Write([]byte(strconv.FormatFloat(value, 'g', 17, 64)))
	}
	var key uint64 = h.Sum64()
	return key ^ (epoch * 0x9e3779b97f4a7c15)
}

func globalEFSearchFromEnv() bool {
	return strings.EqualFold(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_EF_BUDGET")), "global")
}

func globalEFPercentFromEnv() int {
	percent, percentErr := strconv.Atoi(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_EF_GLOBAL_PERCENT")))
	if percentErr == nil && percent >= 100 && percent <= 1600 {
		return percent
	}
	multiplier, err := strconv.Atoi(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_EF_GLOBAL_MULTIPLIER")))
	if err != nil || multiplier < 1 || multiplier > 16 {
		return 100
	}
	return multiplier * 100
}

func primaryIndexFromEnv() bool {
	value := strings.ToLower(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_PRIMARY_INDEX")))
	switch value {
	case "1", "true", "on", "auto":
		return true
	default:
		return false
	}
}

func primaryBuildOptions(base Options) Options {
	if value, err := strconv.Atoi(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_PRIMARY_M"))); err == nil && value >= 4 && value <= 128 {
		base.M = value
	}
	if value, err := strconv.Atoi(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_PRIMARY_EF_CONSTRUCTION"))); err == nil && value >= 8 && value <= 1000 {
		base.EfConstruction = value
	}
	if raw := strings.ToLower(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_PRIMARY_QUANTIZE"))); raw == "0" || raw == "false" || raw == "off" {
		base.QuantizeSegments = false
	}
	if raw := strings.ToLower(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_PRIMARY_DIVERSIFIED_PRUNING"))); raw == "1" || raw == "true" || raw == "on" {
		base.DiversifiedPruning = true
	}
	return base
}

func primaryEntrypointCount() int {
	value, err := strconv.Atoi(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_PRIMARY_ENTRYPOINTS")))
	if err != nil || value < 2 {
		return 0
	}
	return min(value, 32)
}

func reserveANNCompactionMemory(limit, estimate uint64) (release func(), retry bool, ok bool) {
	if limit == 0 || estimate == 0 {
		return func() {}, false, true
	}
	if estimate > limit {
		return func() {}, false, false
	}
	for {
		reserved := globalANNCompactionReserved.Load()
		if reserved > limit-estimate {
			return func() {}, true, false
		}
		if globalANNCompactionReserved.CompareAndSwap(reserved, reserved+estimate) {
			return func() { globalANNCompactionReserved.Add(^uint64(estimate - 1)) }, false, true
		}
	}
}

// OpenSegmentedIndex loads published immutable segments from a durable catalog.
// payloadPath resolves each manifest to its payload file. The mutable delta is
// always started empty, so interrupted writes cannot become visible on restart.
func OpenSegmentedIndex(opts Options, segmentMaxNodes int, catalog *SegmentCatalog, payloadPath func(SegmentManifest) string) (*SegmentedIndex, error) {
	if catalog == nil || payloadPath == nil {
		return nil, errors.New("catalog and payload resolver are required")
	}
	idx := NewSegmentedIndex(opts, segmentMaxNodes)
	for _, manifest := range catalog.Snapshot() {
		path := payloadPath(manifest)
		if err := manifest.VerifyPayloadFile(path); err != nil {
			idx.Close()
			return nil, err
		}
		f, err := os.Open(path)
		if err != nil {
			idx.Close()
			return nil, err
		}
		segment := NewAnnIndexWithOptions(opts)
		_, err = ReadSegmentPayload(f, func(id int, values []float32) error {
			// Segment payloads are already canonical float32. Keep the
			// restart path native so reopening does not allocate a widened
			// float64 copy for every vector.
			return segment.AddVector32(id, values)
		})
		closeErr := f.Close()
		if err == nil {
			err = closeErr
		}
		if err != nil {
			idx.Close()
			return nil, err
		}
		if opts.QuantizeSegments {
			segment.CompactAdjacency()
			if err := segment.Quantize(); err != nil {
				idx.Close()
				return nil, err
			}
		}
		idx.mu.Lock()
		idx.segments = append(idx.segments, segment)
		idx.mutationEpoch++
		idx.segmentEpoch++
		idx.mu.Unlock()
	}
	return idx, nil
}

// OpenSegmentedIndexWithSnapshots reopens pre-built ANN graphs, avoiding HNSW
// reconstruction during restart. Snapshot files are validated by the caller's
// catalog/checksum policy and must contain an AnnIndex MarshalBinary payload.
func OpenSegmentedIndexWithSnapshots(opts Options, segmentMaxNodes int, catalog *SegmentCatalog, snapshotPath func(SegmentManifest) string) (*SegmentedIndex, error) {
	if catalog == nil || snapshotPath == nil {
		return nil, errors.New("catalog and snapshot resolver are required")
	}
	idx := NewSegmentedIndex(opts, segmentMaxNodes)
	for _, manifest := range catalog.Snapshot() {
		path := snapshotPath(manifest)
		if err := manifest.VerifyArtifactFile(path); err != nil {
			idx.Close()
			return nil, err
		}
		segment, err := OpenCompactBinaryFile(path)
		if err != nil {
			data, readErr := os.ReadFile(path)
			if readErr != nil {
				idx.Close()
				return nil, readErr
			}
			segment, err = UnmarshalBinary(data)
		}
		if err != nil {
			idx.Close()
			return nil, err
		}
		if opts.QuantizeSegments {
			if err := segment.Quantize(); err != nil {
				idx.Close()
				return nil, err
			}
		}
		idx.mu.Lock()
		idx.segments = append(idx.segments, segment)
		idx.mutationEpoch++
		idx.segmentEpoch++
		idx.mu.Unlock()
	}
	return idx, nil
}

func (s *SegmentedIndex) AddVector(id int, values []float64) error {
	s.ingestMu.Lock()
	defer s.ingestMu.Unlock()
	s.mu.RLock()
	delta := s.delta
	s.mu.RUnlock()
	if err := delta.AddVector(id, values); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	// Only rotate the delta that accepted this vector. Another writer may have
	// already rotated it while this goroutine was adding.
	if s.delta != delta {
		return nil
	}
	s.deltaNodes++
	s.mutationEpoch++
	if s.deltaNodes >= s.segmentMaxNodes {
		s.delta.CompactAdjacency()
		if s.options.QuantizeSegments {
			if err := s.delta.Quantize(); err != nil {
				return err
			}
		}
		s.segments = append(s.segments, s.delta)
		s.segmentEpoch++
		s.delta = NewAnnIndexWithOptions(s.ingestOptions())
		s.deltaNodes = 0
		s.requestCompactionLocked()
	}
	return nil
}

// AddBatch appends into the mutable delta until segmentMaxNodes is reached.
// Physical segment layout is therefore independent of client batch size.
func (s *SegmentedIndex) AddBatch(vectors []BatchVector) error {
	if len(vectors) == 0 {
		return nil
	}
	s.ingestMu.Lock()
	defer s.ingestMu.Unlock()
	for start := 0; start < len(vectors); {
		s.mu.RLock()
		delta := s.delta
		remaining := s.segmentMaxNodes - s.deltaNodes
		s.mu.RUnlock()
		end := min(start+remaining, len(vectors))
		for _, vector := range vectors[start:end] {
			if err := delta.AddVector(vector.ID, vector.Values); err != nil {
				return err
			}
		}
		s.mu.Lock()
		s.deltaNodes += end - start
		s.mutationEpoch++
		if s.deltaNodes >= s.segmentMaxNodes {
			delta.CompactAdjacency()
			if s.options.QuantizeSegments {
				if err := delta.Quantize(); err != nil {
					s.mu.Unlock()
					return err
				}
			}
			s.segments = append(s.segments, delta)
			s.segmentEpoch++
			s.delta = NewAnnIndexWithOptions(s.ingestOptions())
			s.deltaNodes = 0
			s.requestCompactionLocked()
		}
		s.mu.Unlock()
		start = end
	}
	return nil
}

// AddBatch32 builds immutable segments directly from float32 payloads,
// avoiding widening conversions during checkpoint or segment transfer.
func (s *SegmentedIndex) AddBatch32(vectors []BatchVector32) error {
	if len(vectors) == 0 {
		return nil
	}
	s.ingestMu.Lock()
	defer s.ingestMu.Unlock()
	for start := 0; start < len(vectors); {
		s.mu.RLock()
		delta := s.delta
		remaining := s.segmentMaxNodes - s.deltaNodes
		s.mu.RUnlock()
		end := min(start+remaining, len(vectors))
		for _, vector := range vectors[start:end] {
			if err := delta.AddVector32(vector.ID, vector.Values); err != nil {
				return err
			}
		}
		s.mu.Lock()
		s.deltaNodes += end - start
		s.mutationEpoch++
		if s.deltaNodes >= s.segmentMaxNodes {
			delta.CompactAdjacency()
			if s.options.QuantizeSegments {
				if err := delta.Quantize(); err != nil {
					s.mu.Unlock()
					return err
				}
			}
			s.segments = append(s.segments, delta)
			s.segmentEpoch++
			s.delta = NewAnnIndexWithOptions(s.ingestOptions())
			s.deltaNodes = 0
			s.requestCompactionLocked()
		}
		s.mu.Unlock()
		start = end
	}
	return nil
}

// BuildAndPublishBatch32 builds immutable ANN generations off to the side and
// publishes them only after each graph is complete.  It is intended for
// reshard/recovery drains: query readers continue to use the previous
// generation while the incoming delta is being materialized, and the batch
// never accumulates in the query-visible mutable graph.
func (s *SegmentedIndex) BuildAndPublishBatch32(vectors []BatchVector32) error {
	if len(vectors) == 0 {
		return nil
	}
	for start := 0; start < len(vectors); {
		end := min(start+s.segmentMaxNodes, len(vectors))
		segment := NewAnnIndexWithOptionsCapacity(s.options, end-start)
		for _, vector := range vectors[start:end] {
			if err := segment.AddVector32(vector.ID, vector.Values); err != nil {
				_ = segment.Close()
				return err
			}
		}
		if err := s.publishPrebuiltSegment(segment); err != nil {
			_ = segment.Close()
			return err
		}
		start = end
	}
	return nil
}

// BuildAndPublishBatch is the float64 ingestion counterpart of
// BuildAndPublishBatch32. Large public bulk writes can build bounded immutable
// generations without repeatedly rewiring an already large mutable HNSW graph.
func (s *SegmentedIndex) BuildAndPublishBatch(vectors []BatchVector) error {
	if len(vectors) == 0 {
		return nil
	}
	for start := 0; start < len(vectors); {
		end := min(start+s.segmentMaxNodes, len(vectors))
		segment := NewAnnIndexWithOptionsCapacity(s.options, end-start)
		for _, vector := range vectors[start:end] {
			if err := segment.AddVector(vector.ID, vector.Values); err != nil {
				_ = segment.Close()
				return err
			}
		}
		if err := s.publishPrebuiltSegment(segment); err != nil {
			_ = segment.Close()
			return err
		}
		start = end
	}
	return nil
}

// PublishPrebuiltSnapshot atomically attaches an already-built immutable ANN
// graph. It is the destination-side primitive for online resharding: graph
// construction happens before this call, while readers continue using the old
// generation until the complete snapshot has been decoded and validated.
//
// The snapshot is accepted in the compact format first, then the legacy JSON
// format for backwards compatibility. A caller must persist and checksum the
// artifact before publication; durable catalog integration deliberately sits
// above this in the data-plane transaction.
func (s *SegmentedIndex) PublishPrebuiltSnapshot(snapshot []byte) error {
	if len(snapshot) == 0 {
		return errors.New("prebuilt ANN snapshot is empty")
	}
	segment, err := UnmarshalCompactBinary(snapshot)
	if err != nil {
		segment, err = UnmarshalBinary(snapshot)
	}
	if err != nil {
		return fmt.Errorf("decode prebuilt ANN snapshot: %w", err)
	}
	return s.publishPrebuiltSegment(segment)
}

// PublishPrebuiltSnapshotFile attaches a staged artifact without first loading
// the whole snapshot into a byte slice. The manifest verification makes this
// safe to call only after SnapshotStager.Commit has atomically exposed path.
func (s *SegmentedIndex) PublishPrebuiltSnapshotFile(path string, manifest SegmentManifest) error {
	if err := manifest.VerifyArtifactFile(path); err != nil {
		return err
	}
	segment, err := OpenCompactBinaryFile(path)
	if err != nil {
		return fmt.Errorf("open prebuilt ANN snapshot: %w", err)
	}
	if err := s.publishPrebuiltSegment(segment); err != nil {
		_ = segment.Close()
		return err
	}
	return nil
}

// PrebuiltSnapshotIDsFromFile validates a staged snapshot and returns its
// internal IDs without retaining its file mapping. Services use it to verify
// that the destination already has matching ID mappings and vector records
// before the graph becomes query-visible.
func PrebuiltSnapshotIDsFromFile(path string, manifest SegmentManifest) ([]int, error) {
	if err := manifest.VerifyArtifactFile(path); err != nil {
		return nil, err
	}
	segment, err := OpenCompactBinaryFile(path)
	if err != nil {
		return nil, err
	}
	defer segment.Close()
	stats := segment.Stats()
	if stats.Nodes != manifest.Nodes || stats.Deleted != 0 {
		return nil, errors.New("prebuilt snapshot cardinality does not match manifest")
	}
	segment.mu.RLock()
	ids := make([]int, 0, len(segment.idToSlot))
	for id := range segment.idToSlot {
		ids = append(ids, id)
	}
	segment.mu.RUnlock()
	sort.Ints(ids)
	return ids, nil
}

func (s *SegmentedIndex) publishPrebuiltSegment(segment *AnnIndex) error {
	if segment == nil {
		return errors.New("prebuilt ANN segment is nil")
	}
	stats := segment.Stats()
	if stats.Nodes == 0 || stats.Deleted != 0 {
		return errors.New("prebuilt ANN snapshot must contain live vectors only")
	}
	if stats.Nodes > s.segmentMaxNodes {
		return fmt.Errorf("prebuilt ANN snapshot has %d nodes; segment limit is %d", stats.Nodes, s.segmentMaxNodes)
	}
	segment.mu.RLock()
	dimension := segment.dim
	incomingIDs := make(map[int]struct{}, len(segment.idToSlot))
	for id := range segment.idToSlot {
		incomingIDs[id] = struct{}{}
	}
	segment.mu.RUnlock()

	s.ingestMu.Lock()
	defer s.ingestMu.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, existing := range append(append([]*AnnIndex(nil), s.segments...), s.delta) {
		if existing == nil {
			continue
		}
		existing.mu.RLock()
		for id := range incomingIDs {
			if slot, exists := existing.idToSlot[id]; exists && !existing.isDeletedLocked(slot) {
				existing.mu.RUnlock()
				return fmt.Errorf("prebuilt ANN snapshot duplicates vector ID %d", id)
			}
		}
		existing.mu.RUnlock()
	}
	if s.deltaNodes > 0 {
		s.delta.mu.RLock()
		deltaDimension := s.delta.dim
		s.delta.mu.RUnlock()
		if deltaDimension > 0 && dimension != deltaDimension {
			return fmt.Errorf("prebuilt ANN dimension %d does not match delta dimension %d", dimension, deltaDimension)
		}
	}
	segment.mu.RLock()
	quantized := segment.quantized
	segment.mu.RUnlock()
	if s.options.QuantizeSegments && !quantized {
		if err := segment.Quantize(); err != nil {
			return err
		}
	}
	segment.CompactAdjacency()
	s.segments = append(s.segments, segment)
	s.mutationEpoch++
	s.segmentEpoch++
	s.requestCompactionLocked()
	return nil
}

func (s *SegmentedIndex) ingestOptions() Options {
	options := s.options
	if options.M <= 0 {
		options.M = 16
	}
	if options.EfConstruction <= 0 {
		options.EfConstruction = 64
	}
	if options.EfSearch <= 0 {
		options.EfSearch = 32
	}
	return options
}

func (s *SegmentedIndex) requestCompactionLocked() {
	if s.closed || (len(s.segments) < s.compactThreshold && !(s.primaryIndexEnabled && len(s.segments) > 1)) {
		return
	}
	s.requestConsolidationLocked()
}

func (s *SegmentedIndex) requestConsolidationLocked() {
	if s.closed {
		return
	}
	// A primary-index build is also useful when two full segments cannot be
	// merged under the normal segment-size limit. Queue the same maintenance
	// worker so the HTTP/service path does not need to call WaitForMaintenance
	// explicitly before the opt-in generation is built.
	if !s.canCompactLocked() && !(s.primaryIndexEnabled && len(s.segments) > 1) {
		return
	}
	if s.primaryIndexEnabled && len(s.segments) > 1 && s.primaryBuildRequestedEpoch == s.mutationEpoch {
		return
	}
	if s.primaryIndexEnabled && len(s.segments) > 1 {
		s.primaryBuildRequestedEpoch = s.mutationEpoch
	}
	s.pending.Store(true)
	select {
	case s.compactCh <- struct{}{}:
	default:
	}
}

func (s *SegmentedIndex) canCompactLocked() bool {
	if len(s.segments) < 2 {
		return false
	}
	first := s.segments[0].Stats()
	second := s.segments[1].Stats()
	return first.Nodes-first.Deleted+second.Nodes-second.Deleted <= s.segmentMaxNodes
}

func (s *SegmentedIndex) RequestConsolidation() {
	s.mu.Lock()
	s.requestConsolidationLocked()
	s.mu.Unlock()
}

func (s *SegmentedIndex) compactionWorker() {
	defer s.workerWG.Done()
	for {
		select {
		case <-s.compactCh:
			if !s.waitForQuietPeriod() {
				return
			}
			s.pending.Store(false)
			s.compacting.Store(true)
			s.compactGeneration()
			if s.primaryIndexEnabled {
				_ = s.BuildPrimaryIndex(context.Background())
			}
			s.compacting.Store(false)
		case <-s.stopCh:
			return
		}
	}
}

func (s *SegmentedIndex) waitForQuietPeriod() bool {
	timer := time.NewTimer(defaultCompactDebounce)
	defer timer.Stop()
	for {
		select {
		case <-s.compactCh:
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			timer.Reset(defaultCompactDebounce)
		case <-timer.C:
			return true
		case <-s.stopCh:
			return false
		}
	}
}

func (s *SegmentedIndex) compactGeneration() {
	started := time.Now()
	s.mu.Lock()
	if s.closed || len(s.segments) <= 1 {
		s.mu.Unlock()
		return
	}
	release, ok := s.pinReaderLocked()
	if !ok {
		s.mu.Unlock()
		return
	}
	epoch := s.mutationEpoch
	segments := append([]*AnnIndex(nil), s.segments...)
	s.mu.Unlock()
	defer release()

	// Never create a graph larger than the configured immutable-segment
	// budget. Repeatedly merging full segments used to rebuild an ever-growing
	// HNSW graph during ingestion, turning nominally incremental work into a
	// near-quadratic background cost.
	mergeLimit := 0
	liveNodes := 0
	for _, segment := range segments {
		stats := segment.Stats()
		next := stats.Nodes - stats.Deleted
		if mergeLimit > 0 && liveNodes+next > s.segmentMaxNodes {
			break
		}
		liveNodes += next
		mergeLimit++
	}
	if mergeLimit < 2 {
		return
	}
	segments = segments[:mergeLimit]
	estimatedBytes := estimateCompactionMemory(segments)
	s.compactionMemoryEstimate.Store(estimatedBytes)
	releaseMemory, retryReservation, reserved := reserveANNCompactionMemory(s.compactionMemoryBudget, estimatedBytes)
	if !reserved {
		s.compactionMemoryDeferred.Add(1)
		if retryReservation {
			s.mu.Lock()
			s.requestConsolidationLocked()
			s.mu.Unlock()
		}
		return
	}
	s.compactionMemoryReserved.Store(estimatedBytes)
	defer func() {
		s.compactionMemoryReserved.Store(0)
		releaseMemory()
	}()

	merged := NewAnnIndexWithOptions(s.options)
	mergedSegments := 0
mergeLoop:
	for _, segment := range segments {
		if err := s.copyLiveVectorsForCompaction(segment, merged); err != nil {
			_ = merged.Close()
			return
		}
		mergedSegments++
		// Publish progress at segment boundaries. The previous implementation
		// abandoned the complete rebuild when the budget elapsed, then retried
		// from segment zero forever when a generation could not fit in one
		// budget window.
		if mergedSegments >= 2 && s.compactionBudget > 0 && time.Since(started) >= s.compactionBudget {
			break mergeLoop
		}
	}

	merged.CompactAdjacency()
	if s.options.QuantizeSegments {
		if err := merged.Quantize(); err != nil {
			return
		}
	}
	var reclaim []*AnnIndex
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		_ = merged.Close()
		return
	}
	if s.mutationEpoch != epoch || len(s.segments) < mergedSegments {
		s.requestCompactionLocked()
		s.mu.Unlock()
		_ = merged.Close()
		return
	}
	for i := 0; i < mergedSegments; i++ {
		if s.segments[i] != segments[i] {
			s.requestCompactionLocked()
			s.mu.Unlock()
			_ = merged.Close()
			return
		}
	}
	tail := append([]*AnnIndex(nil), s.segments[mergedSegments:]...)
	s.segments = append([]*AnnIndex{merged}, tail...)
	s.segmentEpoch++
	// The compactor itself holds a reader pin while publishing, so replaced
	// generations enter retired and are reclaimed by the final release below.
	// Other concurrent readers extend that lifetime without retaining the
	// generations until process shutdown.
	s.retired = append(s.retired, segments[:mergedSegments]...)
	if s.reclaimRetired && s.activeReaders == 0 {
		reclaim = s.retired
		s.retired = nil
	}
	if len(s.segments) >= s.compactThreshold {
		s.requestCompactionLocked()
	}
	s.mu.Unlock()
	s.closeReclaimed(reclaim)
}

func estimateCompactionMemory(segments []*AnnIndex) uint64 {
	var estimate uint64
	for _, segment := range segments {
		if segment == nil {
			continue
		}
		bytes := segment.MemoryStats().TotalBytes
		if ^uint64(0)-estimate < bytes {
			return ^uint64(0)
		}
		estimate += bytes
	}
	return estimate
}

// estimatePrimaryBuildMemory accounts for both the immutable sources retained
// by the reader pin and the destination graph being built. The destination
// estimate is intentionally conservative: vectors, node metadata and the
// configured adjacency budget are all live until the atomic publication.
func estimatePrimaryBuildMemory(sources []*AnnIndex, liveNodes int, options Options) uint64 {
	estimate := estimateCompactionMemory(sources)
	dimension := 0
	for _, source := range sources {
		if source == nil {
			continue
		}
		source.mu.RLock()
		dimension = source.dim
		source.mu.RUnlock()
		if dimension > 0 {
			break
		}
	}
	if liveNodes <= 0 || dimension <= 0 {
		return estimate
	}
	m := options.M
	if m <= 0 {
		m = 16
	}
	perNode := uint64(dimension*4 + m*8 + 64)
	additional := uint64(liveNodes) * perNode
	if ^uint64(0)-estimate < additional {
		return ^uint64(0)
	}
	return estimate + additional
}

// CompactionMemoryState exposes the configured process-wide build budget and
// this shard's current reservation/last estimate. A zero budget is observation
// mode and never defers maintenance.
func (s *SegmentedIndex) CompactionMemoryState() (budget, reserved, estimated, deferred uint64) {
	return s.compactionMemoryBudget, s.compactionMemoryReserved.Load(), s.compactionMemoryEstimate.Load(), s.compactionMemoryDeferred.Load()
}

func (s *SegmentedIndex) copyLiveVectorsForCompaction(source, destination *AnnIndex) error {
	add := func(position, id int, values []float32) error {
		// Shutdown must not wait for an entire HNSW segment rebuild. No
		// partially merged graph has been published yet, so aborting here
		// preserves the previous immutable generation for checkpointing.
		if position&63 == 0 {
			s.mu.RLock()
			closed := s.closed
			s.mu.RUnlock()
			if closed {
				return errors.New("ANN compaction canceled")
			}
		}
		return destination.AddVector32(id, values)
	}
	if !s.compactionStream {
		for position, vector := range source.ExportLiveVectors32() {
			if err := add(position, vector.ID, vector.Values); err != nil {
				return err
			}
		}
		return nil
	}
	position := 0
	return source.VisitLiveVectors32(func(id int, values []float32) error {
		err := add(position, id, values)
		position++
		return err
	})
}

func (s *SegmentedIndex) WaitForMaintenance(ctx context.Context) error {
	s.RequestConsolidation()
	ticker := time.NewTicker(5 * time.Millisecond)
	defer ticker.Stop()
	for {
		s.mu.RLock()
		closed := s.closed
		compactable := s.canCompactLocked()
		s.mu.RUnlock()
		if closed || (!compactable && !s.pending.Load() && !s.compacting.Load()) {
			if closed || !s.primaryIndexEnabled {
				return nil
			}
			return s.BuildPrimaryIndex(ctx)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

// BuildPrimaryIndex replaces a collection of immutable segments with one
// immutable HNSW generation. New writes are still accepted in a fresh delta
// and are searched alongside the primary graph after the atomic swap. This is
// the high-cardinality escape hatch for workloads where segment routing falls
// back to full fanout.
//
// The operation is opt-in through LUMENVEC_ANN_PRIMARY_INDEX. It aborts and
// leaves the current generations untouched when a write or delete races the
// build, preserving the existing segmented fallback semantics.
func (s *SegmentedIndex) BuildPrimaryIndex(ctx context.Context) error {
	if !s.primaryIndexEnabled {
		return nil
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return errors.New("segmented index is closed")
	}
	immutable := append([]*AnnIndex(nil), s.segments...)
	delta := s.delta
	deltaNodes := s.deltaNodes
	if len(immutable) <= 1 && deltaNodes == 0 {
		s.mu.Unlock()
		return nil
	}
	release, ok := s.pinReaderLocked()
	epoch := s.mutationEpoch
	s.mu.Unlock()
	if !ok {
		return errors.New("segmented index is closed")
	}
	defer release()

	sources := append([]*AnnIndex(nil), immutable...)
	if deltaNodes > 0 && delta != nil {
		sources = append(sources, delta)
	}
	liveNodes := 0
	for _, source := range sources {
		if source != nil {
			stats := source.Stats()
			liveNodes += stats.Nodes - stats.Deleted
		}
	}
	if liveNodes == 0 {
		return nil
	}
	if !s.primaryBuildActive.CompareAndSwap(false, true) {
		return nil
	}
	s.primaryBuildTotal.Store(uint64(liveNodes))
	s.primaryBuildDone.Store(0)
	started := time.Now()
	published := false
	defer func() {
		s.mu.Lock()
		if !published && s.primaryBuildRequestedEpoch == epoch {
			s.primaryBuildRequestedEpoch = 0
		}
		s.mu.Unlock()
		s.primaryBuildDurationMs.Store(uint64(time.Since(started) / time.Millisecond))
		s.primaryBuildActive.Store(false)
	}()
	if err := ctx.Err(); err != nil {
		s.primaryBuildFailures.Add(1)
		return err
	}
	if err := acquirePrimaryBuild(ctx); err != nil {
		s.primaryBuildFailures.Add(1)
		return err
	}
	defer releasePrimaryBuild()

	s.mu.RLock()
	buildOptions := primaryBuildOptions(s.options)
	s.mu.RUnlock()
	estimatedBytes := estimatePrimaryBuildMemory(sources, liveNodes, buildOptions)
	releaseMemory, retryReservation, reserved := reserveANNCompactionMemory(s.compactionMemoryBudget, estimatedBytes)
	if !reserved {
		if retryReservation {
			s.mu.Lock()
			s.requestConsolidationLocked()
			s.mu.Unlock()
		}
		return nil
	}
	defer releaseMemory()

	merged := NewAnnIndexWithOptionsCapacity(buildOptions, liveNodes)
	// Hold the destination write lock once for the complete generation build.
	// Source visitors remain independently pinned, so readers never observe a
	// partially built graph and the hot insertion path pays no lock transition
	// per vector.
	merged.mu.Lock()
	for _, source := range sources {
		if source == nil {
			continue
		}
		if err := source.VisitLiveVectors32(func(id int, values []float32) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := merged.addVector32Locked(id, values); err != nil {
				return err
			}
			s.primaryBuildDone.Add(1)
			return nil
		}); err != nil {
			merged.mu.Unlock()
			_ = merged.Close()
			s.primaryBuildFailures.Add(1)
			return err
		}
	}
	merged.mu.Unlock()
	merged.CompactAdjacency()
	if buildOptions.QuantizeSegments {
		if err := merged.Quantize(); err != nil {
			_ = merged.Close()
			s.primaryBuildFailures.Add(1)
			return err
		}
	}
	if count := primaryEntrypointCount(); count > 0 && liveNodes > 0 {
		entries := make([]int, 0, count)
		for i := 0; i < count; i++ {
			entries = append(entries, i*liveNodes/count)
		}
		merged.SetSearchEntrypoints(entries)
	}

	var reclaim []*AnnIndex
	s.mu.Lock()
	unchanged := s.mutationEpoch == epoch && len(s.segments) == len(immutable) && s.delta == delta
	if unchanged {
		for position, segment := range immutable {
			if s.segments[position] != segment {
				unchanged = false
				break
			}
		}
	}
	if !unchanged || s.closed {
		if !s.closed && s.primaryBuildRequestedEpoch == epoch {
			s.primaryBuildRequestedEpoch = 0
		}
		s.mu.Unlock()
		_ = merged.Close()
		return nil
	}
	old := append([]*AnnIndex(nil), s.segments...)
	if deltaNodes > 0 && delta != nil {
		old = append(old, delta)
	}
	s.segments = []*AnnIndex{merged}
	s.delta = NewAnnIndexWithOptions(s.ingestOptions())
	s.deltaNodes = 0
	s.primaryBuildPublishedEpoch = epoch
	published = true
	s.segmentEpoch++
	s.retired = append(s.retired, old...)
	if s.reclaimRetired && s.activeReaders == 0 {
		reclaim = s.retired
		s.retired = nil
	}
	s.mu.Unlock()
	s.closeReclaimed(reclaim)
	return nil
}

// MaintenanceState exposes whether background consolidation still has work
// which can change the query-visible generation. Full immutable segments may
// legitimately remain numerous; segment count alone is not a readiness signal.
func (s *SegmentedIndex) MaintenanceState() (pending, compacting, compactable bool) {
	s.mu.Lock()
	if !s.closed && s.primaryIndexEnabled && len(s.segments) > 1 {
		// Readiness/metrics polling is also a maintenance trigger for the
		// opt-in primary layout. This covers bulk-ingest paths that publish
		// segments below the legacy compaction threshold.
		s.requestConsolidationLocked()
	}
	compactable = !s.closed && (s.canCompactLocked() || (s.primaryIndexEnabled && len(s.segments) > 1))
	s.mu.Unlock()
	return s.pending.Load(), s.compacting.Load(), compactable
}

// PrimaryBuildState reports progress of the opt-in bulk generation build.
// Duration is the most recent build duration in milliseconds.
func (s *SegmentedIndex) PrimaryBuildState() (active bool, total, done, durationMs, failures uint64) {
	return s.primaryBuildActive.Load(), s.primaryBuildTotal.Load(), s.primaryBuildDone.Load(), s.primaryBuildDurationMs.Load(), s.primaryBuildFailures.Load()
}

func (s *SegmentedIndex) Close() error {
	s.stopOnce.Do(func() {
		s.mu.Lock()
		s.closed = true
		s.mu.Unlock()
		close(s.stopCh)
	})
	s.workerWG.Wait()
	s.mu.Lock()
	for s.activeReaders > 0 {
		s.readerCond.Wait()
	}
	indexes := append([]*AnnIndex(nil), s.segments...)
	indexes = append(indexes, s.retired...)
	indexes = append(indexes, s.delta)
	s.segments = nil
	s.retired = nil
	s.delta = nil
	s.mu.Unlock()
	var firstErr error
	for _, idx := range indexes {
		if idx != nil {
			if err := idx.Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}
	if s.mappedOwner != nil {
		if err := s.mappedOwner.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		s.mappedOwner = nil
	}
	return firstErr
}

func (s *SegmentedIndex) SearchWithDistancesInto(query []float64, k int, dst []Result) ([]Result, error) {
	return s.SearchWithDistancesContext(context.Background(), query, k, dst)
}

func (s *SegmentedIndex) SearchWithDistancesContext(ctx context.Context, query []float64, k int, dst []Result) ([]Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, ErrInvalidK
	}
	if index, ok := s.singleIndexPinned(); ok {
		defer s.releaseReader()
		s.mu.RLock()
		configuredEF := s.options.EfSearch
		s.mu.RUnlock()
		segmentEF := s.segmentSearchEF(configuredEF, k, 1)
		s.searchQueries.Add(1)
		s.searchSegments.Add(1)
		s.searchEFBudget.Add(uint64(segmentEF))
		return index.SearchWithDistancesEfContext(ctx, query, k, segmentEF, dst)
	}
	indexes, release, ok := s.routedIndexesPinned(query)
	if !ok {
		return nil, errors.New("segmented index is closed")
	}
	defer release()
	s.mu.RLock()
	configuredEF := s.options.EfSearch
	s.mu.RUnlock()
	nonEmpty := indexes[:0]
	for _, index := range indexes {
		if index != nil && !index.IsEmpty() {
			nonEmpty = append(nonEmpty, index)
		}
	}
	indexes = nonEmpty
	if len(indexes) == 0 {
		s.mu.RLock()
		configuredEF := s.options.EfSearch
		s.mu.RUnlock()
		// No graph was searched; exact fallback must not fabricate ANN work.
		_ = configuredEF
		return dst[:0], nil
	}
	segmentEF := s.segmentSearchEF(configuredEF, k, len(indexes))
	s.searchQueries.Add(1)
	s.searchSegments.Add(uint64(len(indexes)))
	s.searchEFBudget.Add(uint64(segmentEF * len(indexes)))
	if s.searchExecution == "inline" {
		return s.searchSequential(indexes, query, k, segmentEF, dst, ctx)
	}
	if s.searchExecution == "adaptive" && len(indexes) == 1 {
		return s.searchAdaptive(indexes, query, k, segmentEF, dst, ctx)
	}
	return s.searchLegacy(indexes, query, k, segmentEF, dst, ctx)
}

func searchContext(contexts []context.Context) context.Context {
	if len(contexts) > 0 {
		return contexts[0]
	}
	return context.Background()
}

func (s *SegmentedIndex) searchLegacy(indexes []*AnnIndex, query []float64, k, segmentEF int, dst []Result, contexts ...context.Context) ([]Result, error) {
	ctx := searchContext(contexts)
	workers := min(len(indexes), maxSegmentSearchWorkers, max(1, runtime.GOMAXPROCS(0)))
	perWorker := make([][]Result, workers)
	workerBuffers := make([]*segmentSearchWorker, workers)
	jobs := make(chan int, len(indexes))
	errCh := make(chan error, workers)
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		worker := worker
		wg.Add(1)
		go func() {
			defer wg.Done()
			buffers, _ := s.searchWorkerPool.Get().(*segmentSearchWorker)
			if buffers == nil {
				buffers = &segmentSearchWorker{}
			}
			if cap(buffers.local) < k {
				buffers.local = make([]Result, 0, k)
			} else {
				buffers.local = buffers.local[:0]
			}
			if cap(buffers.spare) < k {
				buffers.spare = make([]Result, 0, k)
			} else {
				buffers.spare = buffers.spare[:0]
			}
			if cap(buffers.scratch) < k {
				buffers.scratch = make([]Result, 0, k)
			} else {
				buffers.scratch = buffers.scratch[:0]
			}
			local, spare, scratch := buffers.local, buffers.spare, buffers.scratch
			for position := range jobs {
				idx := indexes[position]
				// Preserve the configured quality profile across immutable
				// segments. The old hard-coded k*8 ceiling silently reduced an
				// efSearch=512 service to ef=80 and lost recall at 100k.
				results, err := idx.SearchWithDistancesEfContext(ctx, query, k, segmentEF, scratch[:0])
				if err != nil {
					select {
					case errCh <- err:
					default:
					}
					continue
				}
				spare = mergeSortedTopK(local, results, k, spare[:0])
				local, spare = spare, local
				scratch = results[:0]
			}
			buffers.local, buffers.spare, buffers.scratch = local, spare, scratch
			workerBuffers[worker] = buffers
			perWorker[worker] = local
		}()
	}
	for position := range indexes {
		jobs <- position
	}
	close(jobs)
	wg.Wait()
	close(errCh)
	defer func() {
		for _, buffers := range workerBuffers {
			if buffers != nil {
				s.searchWorkerPool.Put(buffers)
			}
		}
	}()
	if err := <-errCh; err != nil {
		return nil, err
	}
	out := dst[:0]
	if cap(out) < k {
		out = make([]Result, 0, k)
	}
	spare := make([]Result, 0, k)
	for _, results := range perWorker {
		spare = mergeSortedTopK(out, results, k, spare[:0])
		out, spare = spare, out[:0]
	}
	if len(out) == 0 {
		return dst[:0], nil
	}
	return out, nil
}

func (s *SegmentedIndex) searchAdaptive(indexes []*AnnIndex, query []float64, k, segmentEF int, dst []Result, contexts ...context.Context) ([]Result, error) {
	ctx := searchContext(contexts)
	if len(indexes) != 1 {
		return s.searchLegacy(indexes, query, k, segmentEF, dst, ctx)
	}
	buffers, _ := s.searchWorkerPool.Get().(*segmentSearchWorker)
	if buffers == nil {
		buffers = &segmentSearchWorker{}
	}
	s.mu.RLock()
	epoch := s.segmentEpoch
	s.mu.RUnlock()
	cacheKey := adaptiveQueryKey(query, epoch)
	if adaptiveCacheEnabled() {
		s.adaptiveCacheMu.Lock()
		if cached, ok := s.adaptiveCache[cacheKey]; ok {
			segmentEF = min(segmentEF, cached)
		}
		s.adaptiveCacheMu.Unlock()
	}
	probeK := k
	buffers.scratch = resizeResultBuffer(buffers.scratch, probeK)
	initial := adaptiveInitialEFFromEnv(segmentEF, k)
	ef := initial
	var results []Result
	var err error
	for {
		results, err = indexes[0].SearchWithDistancesEfContext(ctx, query, probeK, ef, buffers.scratch[:0])
		if err != nil || len(results) >= k || ef >= segmentEF {
			break
		}
		ef = min(segmentEF, ef*2)
	}
	if err != nil {
		s.searchWorkerPool.Put(buffers)
		return nil, err
	}
	if len(results) > k {
		results = results[:k]
	}
	if adaptiveCacheEnabled() {
		s.adaptiveCacheMu.Lock()
		if s.adaptiveCache == nil {
			s.adaptiveCache = make(map[uint64]int)
		}
		if len(s.adaptiveCache) < 10000 {
			s.adaptiveCache[cacheKey] = ef
		}
		s.adaptiveCacheMu.Unlock()
	}
	out := dst[:0]
	if cap(out) < len(results) {
		out = make([]Result, len(results))
	} else {
		out = out[:len(results)]
	}
	copy(out, results)
	buffers.scratch = results[:0]
	s.searchWorkerPool.Put(buffers)
	return out, nil
}

func (s *SegmentedIndex) searchSequential(indexes []*AnnIndex, query []float64, k, segmentEF int, dst []Result, contexts ...context.Context) ([]Result, error) {
	ctx := searchContext(contexts)
	buffers, _ := s.searchWorkerPool.Get().(*segmentSearchWorker)
	if buffers == nil {
		buffers = &segmentSearchWorker{}
	}
	buffers.local = resizeResultBuffer(buffers.local, k)
	buffers.spare = resizeResultBuffer(buffers.spare, k)
	buffers.scratch = resizeResultBuffer(buffers.scratch, k)
	local, spare, scratch := buffers.local, buffers.spare, buffers.scratch
	for _, index := range indexes {
		results, err := index.SearchWithDistancesEfContext(ctx, query, k, segmentEF, scratch[:0])
		if err != nil {
			buffers.local, buffers.spare, buffers.scratch = local[:0], spare[:0], scratch[:0]
			s.searchWorkerPool.Put(buffers)
			return nil, err
		}
		spare = mergeSortedTopK(local, results, k, spare[:0])
		local, spare = spare, local
		scratch = results[:0]
	}
	out := dst[:0]
	if cap(out) < len(local) {
		out = make([]Result, len(local))
	} else {
		out = out[:len(local)]
	}
	copy(out, local)
	buffers.local, buffers.spare, buffers.scratch = local[:0], spare[:0], scratch[:0]
	s.searchWorkerPool.Put(buffers)
	return out, nil
}

func (s *SegmentedIndex) segmentSearchEF(configuredEF, k, fanout int) int {
	minimum := max(64, k*8)
	if value, err := strconv.Atoi(strings.TrimSpace(os.Getenv("LUMENVEC_ANN_MIN_SEGMENT_EF"))); err == nil && value > 0 {
		minimum = max(k, value)
	}
	if !s.globalEFSearch || fanout <= 1 {
		return max(minimum, configuredEF)
	}
	globalBudgetPercent := configuredEF * s.globalEFPercent
	return max(minimum, (globalBudgetPercent+100*fanout-1)/(100*fanout))
}

// SearchBudgetState reports cumulative query fanout and actual exploration
// budget so a global-budget rollout can be compared with the per-segment
// fallback without sampling individual requests.
func (s *SegmentedIndex) SearchBudgetState() (queries, segments, efBudget uint64) {
	return s.searchQueries.Load(), s.searchSegments.Load(), s.searchEFBudget.Load()
}

// SearchWorkStats aggregates per-index ANN work for diagnostics. It is
// intentionally read-only and is not used by the search policy.
func (s *SegmentedIndex) SearchWorkStats() SearchWorkStats {
	s.mu.RLock()
	indexes := append([]*AnnIndex(nil), s.segments...)
	if s.delta != nil {
		indexes = append(indexes, s.delta)
	}
	s.mu.RUnlock()
	var out SearchWorkStats
	for _, index := range indexes {
		if index == nil {
			continue
		}
		stats := index.SearchWorkStats()
		out.Queries += stats.Queries
		out.Visited += stats.Visited
		out.DistanceCalls += stats.DistanceCalls
		if stats.MaxFrontier > out.MaxFrontier {
			out.MaxFrontier = stats.MaxFrontier
		}
	}
	return out
}

// SearchConfigState reports the immutable search policy captured when this
// index was created. It is used to prove benchmark configuration rather than
// inferring container state from host environment variables.
func (s *SegmentedIndex) SearchConfigState() (execution, efBudget string, globalPercent int, primaryIndex bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	efBudget = "per_segment"
	if s.globalEFSearch {
		efBudget = "global"
	}
	return s.searchExecution, efBudget, s.globalEFPercent, s.primaryIndexEnabled
}

func resizeResultBuffer(buffer []Result, capacity int) []Result {
	if cap(buffer) < capacity {
		return make([]Result, 0, capacity)
	}
	return buffer[:0]
}

type routedSegment struct {
	index    *AnnIndex
	position int
	distance int
}

type segmentSearchWorker struct {
	local   []Result
	spare   []Result
	scratch []Result
}

type routePosting struct {
	signature uint32
	segment   uint32
}

func (s *SegmentedIndex) rebuildRouteDirectoryLocked() {
	if uint64(len(s.segments)) > uint64(^uint32(0)) {
		// The compact posting representation cannot address more than 2^32-1
		// physical segments. Keep routing safe by publishing an empty
		// directory, which makes every query fall back to full fanout.
		s.routeKeys = nil
		s.routeOffsets = nil
		s.routeSegments = nil
		s.routePostingBytes = 0
		s.routeEpoch = s.segmentEpoch
		return
	}
	postings := make([]routePosting, 0)
	for segmentIndex, segment := range s.segments {
		for _, signature := range segment.routeSignatureSnapshot() {
			postings = append(postings, routePosting{signature: signature, segment: uint32(segmentIndex)})
		}
	}
	sort.Slice(postings, func(left, right int) bool {
		if postings[left].signature == postings[right].signature {
			return postings[left].segment < postings[right].segment
		}
		return postings[left].signature < postings[right].signature
	})
	keys := make([]uint32, 0, len(postings))
	offsets := make([]uint32, 0, len(postings)+1)
	segments := make([]uint32, 0, len(postings))
	for _, posting := range postings {
		if len(keys) == 0 || keys[len(keys)-1] != posting.signature {
			keys = append(keys, posting.signature)
			offsets = append(offsets, uint32(len(segments)))
		}
		segments = append(segments, posting.segment)
	}
	offsets = append(offsets, uint32(len(segments)))
	s.routeKeys = keys
	s.routeOffsets = offsets
	s.routeSegments = segments
	s.routePostingBytes = uint64(cap(keys)+cap(offsets)+cap(segments)) * 4
	s.routeEpoch = s.segmentEpoch
}

func visitHammingNeighbors(signature uint32, distance int, visit func(uint32)) {
	if distance == 0 {
		visit(signature)
		return
	}
	var enumerate func(startBit, remaining int, mask uint32)
	enumerate = func(startBit, remaining int, mask uint32) {
		if remaining == 0 {
			visit(signature ^ mask)
			return
		}
		for bit := startBit; bit <= routeSignatureBits-remaining; bit++ {
			enumerate(bit+1, remaining-1, mask|(uint32(1)<<uint(bit)))
		}
	}
	enumerate(0, distance, 0)
}

// routedIndexes uses an inverted sign-LSH directory to avoid evaluating every
// signature in every immutable graph. A query probes Hamming neighborhoods
// directly and resolves matching segment postings; a global signature scan is
// reserved for unusually sparse directories. Selection semantics remain
// identical to the legacy per-segment scan: all distance 0/1 segments and at
// least eight nearest segments are searched.
func (s *SegmentedIndex) routedIndexesPinned(query []float64) ([]*AnnIndex, func(), bool) {
	s.mu.Lock()
	release, ok := s.pinReaderLocked()
	if !ok {
		s.mu.Unlock()
		return nil, release, false
	}
	if s.options.SegmentRouting && s.routeEpoch != s.segmentEpoch {
		s.rebuildRouteDirectoryLocked()
	}
	segments := append([]*AnnIndex(nil), s.segments...)
	delta := s.delta
	enabled := s.options.SegmentRouting
	keys := s.routeKeys
	offsets := s.routeOffsets
	postings := s.routeSegments
	s.mu.Unlock()

	// Routing away two of ten medium-sized segments saves little work but can
	// lose a true nearest neighbour on unclustered data. Keep full fanout for
	// small generations and reserve the probabilistic directory for layouts
	// where it provides a material reduction in segment work.
	if !enabled || len(segments) <= defaultRouteFullFanoutSegments {
		return append(segments, delta), release, true
	}
	signature := routeSignature64(query)
	routed := make([]routedSegment, len(segments))
	for position, index := range segments {
		routed[position] = routedSegment{
			index:    index,
			position: position,
			distance: routeSignatureBits + 1,
		}
	}
	selected := 0
	visitPostings := func(candidateSignature uint32, visit func(int)) {
		position := sort.Search(len(keys), func(index int) bool { return keys[index] >= candidateSignature })
		if position >= len(keys) || keys[position] != candidateSignature {
			return
		}
		for _, segmentIndex := range postings[offsets[position]:offsets[position+1]] {
			visit(int(segmentIndex))
		}
	}
	const maxDirectProbeDistance = 2
	for distance := 0; distance <= maxDirectProbeDistance; distance++ {
		visitHammingNeighbors(signature, distance, func(candidateSignature uint32) {
			visitPostings(candidateSignature, func(segmentIndex int) {
				if routed[segmentIndex].distance == routeSignatureBits+1 {
					selected++
				}
				if distance < routed[segmentIndex].distance {
					routed[segmentIndex].distance = distance
				}
			})
		})
		if distance >= 1 && selected >= defaultRouteMinSegments {
			break
		}
	}
	if selected < defaultRouteMinSegments {
		// A sparse neighborhood is a low-confidence routing decision. Nearest
		// sign signatures alone are not a safe predictor for unclustered data:
		// reducing fanout here materially lowers recall. Search every immutable
		// segment and preserve quality instead.
		return append(segments, delta), release, true
	}
	sort.Slice(routed, func(left, right int) bool {
		if routed[left].distance == routed[right].distance {
			return routed[left].position < routed[right].position
		}
		return routed[left].distance < routed[right].distance
	})
	limit := min(defaultRouteMinSegments, len(routed))
	for limit < len(routed) && routed[limit].distance <= 1 {
		limit++
	}
	indexes := make([]*AnnIndex, 0, limit+1)
	for _, candidate := range routed[:limit] {
		indexes = append(indexes, candidate.index)
	}
	return append(indexes, delta), release, true
}

// routedIndexes remains a test/diagnostic helper. Query execution uses the
// pinned variant so returned segment pointers cannot be reclaimed in flight.
func (s *SegmentedIndex) routedIndexes(query []float64) []*AnnIndex {
	indexes, release, ok := s.routedIndexesPinned(query)
	if !ok {
		return nil
	}
	release()
	return indexes
}

func mergeSortedTopK(left, right []Result, k int, dst []Result) []Result {
	leftIndex, rightIndex := 0, 0
	for len(dst) < k && (leftIndex < len(left) || rightIndex < len(right)) {
		takeLeft := rightIndex >= len(right)
		if !takeLeft && leftIndex < len(left) {
			l, r := left[leftIndex], right[rightIndex]
			takeLeft = l.Distance < r.Distance || (l.Distance == r.Distance && l.ID <= r.ID)
		}
		if takeLeft {
			dst = append(dst, left[leftIndex])
			leftIndex++
		} else {
			dst = append(dst, right[rightIndex])
			rightIndex++
		}
	}
	return dst
}

func (s *SegmentedIndex) DeleteVector(id int) {
	indexes, release, ok := s.pinIndexes(true)
	if !ok {
		return
	}
	defer release()
	for _, idx := range indexes {
		idx.DeleteVector(id)
	}
	s.mu.Lock()
	s.mutationEpoch++
	s.requestCompactionLocked()
	s.mu.Unlock()
}

func (s *SegmentedIndex) Stats() Stats {
	indexes, release, ok := s.pinIndexes(true)
	if !ok {
		return Stats{}
	}
	defer release()
	var total Stats
	for _, idx := range indexes {
		if idx == nil {
			continue
		}
		stats := idx.Stats()
		total.Nodes += stats.Nodes
		total.Deleted += stats.Deleted
	}
	return total
}

func (s *SegmentedIndex) SegmentCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	count := len(s.segments)
	if s.deltaNodes > 0 {
		count++
	}
	return count
}

// VisitSealedSegments exposes a stable snapshot of immutable generations for
// export. The callback runs after the index lock is released; sealed segments
// are never mutated, so a producer can stream one compact snapshot at a time
// without blocking queries or retaining the complete segmented index.
func (s *SegmentedIndex) VisitSealedSegments(visit func(index int, segment *AnnIndex) error) error {
	if visit == nil {
		return errors.New("sealed segment visitor is required")
	}
	segments, release, ok := s.pinIndexes(false)
	if !ok {
		return errors.New("segmented index is closed")
	}
	defer release()
	for i, segment := range segments {
		if segment == nil {
			continue
		}
		if err := visit(i, segment); err != nil {
			return err
		}
	}
	return nil
}

// SnapshotMutableVectors32 captures only the bounded ingest tail. Sealed
// generations are intentionally excluded because resharding transfers them
// through prebuilt graph artifacts.
func (s *SegmentedIndex) SnapshotMutableVectors32() []BatchVector32 {
	s.ingestMu.Lock()
	defer s.ingestMu.Unlock()
	s.mu.Lock()
	release, ok := s.pinReaderLocked()
	if !ok {
		s.mu.Unlock()
		return nil
	}
	delta := s.delta
	s.mu.Unlock()
	defer release()
	if delta == nil {
		return nil
	}
	return delta.ExportLiveVectors32()
}
