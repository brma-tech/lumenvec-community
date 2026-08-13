package ann

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strconv"
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

// SegmentedIndex limits write/query contention by keeping completed ANN graphs
// stable and directing new writes to a small mutable delta graph.
type SegmentedIndex struct {
	mu                sync.RWMutex
	ingestMu          sync.Mutex
	segments          []*AnnIndex
	retired           []*AnnIndex
	mappedOwner       snapshotMapping
	delta             *AnnIndex
	deltaNodes        int
	segmentMaxNodes   int
	options           Options
	compactThreshold  int
	compactCh         chan struct{}
	stopCh            chan struct{}
	stopOnce          sync.Once
	workerWG          sync.WaitGroup
	mutationEpoch     uint64
	segmentEpoch      uint64
	routeEpoch        uint64
	routeKeys         []uint32
	routeOffsets      []uint32
	routeSegments     []uint32
	routePostingBytes uint64
	closed            bool
	pending           atomic.Bool
	compacting        atomic.Bool
	compactionBudget  time.Duration
	searchWorkerPool  sync.Pool
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
}

// SnapshotExport captures sealed generations and the current delta atomically
// with respect to ingestion and segment rotation. The returned segment graphs
// remain immutable and can be streamed after this method returns.
func (s *SegmentedIndex) SnapshotExport() ExportSnapshot {
	s.ingestMu.Lock()
	defer s.ingestMu.Unlock()
	s.mu.RLock()
	segments := append([]*AnnIndex(nil), s.segments...)
	delta := s.delta
	epoch := s.mutationEpoch
	s.mu.RUnlock()
	var tail []BatchVector32
	if delta != nil {
		tail = delta.ExportLiveVectors32()
	}
	return ExportSnapshot{Segments: segments, Tail: tail, Epoch: epoch}
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
	for slot := 0; slot < count; slot++ {
		if a.isDeletedLocked(slot) {
			continue
		}
		if err := visit(a.nodeIDLocked(slot), a.nodeVectorLocked(slot)); err != nil {
			return err
		}
	}
	return nil
}

// MemoryStats aggregates the structural ANN estimate across immutable
// segments and the mutable delta. Runtime/transport allocations are excluded.
func (s *SegmentedIndex) MemoryStats() MemoryStats {
	s.mu.RLock()
	indexes := append([]*AnnIndex(nil), s.segments...)
	indexes = append(indexes, s.delta)
	s.mu.RUnlock()
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
		segmentMaxNodes:  segmentMaxNodes,
		options:          opts,
		compactThreshold: compactThresholdFromEnv(),
		compactionBudget: compactionBudgetFromEnv(),
		compactCh:        make(chan struct{}, 1),
		stopCh:           make(chan struct{}),
	}
	idx.delta = NewAnnIndexWithOptions(idx.ingestOptions())
	idx.workerWG.Add(1)
	go idx.compactionWorker()
	return idx
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
	if s.closed || len(s.segments) < s.compactThreshold {
		return
	}
	s.requestConsolidationLocked()
}

func (s *SegmentedIndex) requestConsolidationLocked() {
	if s.closed || !s.canCompactLocked() {
		return
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
	s.mu.RLock()
	if s.closed || len(s.segments) <= 1 {
		s.mu.RUnlock()
		return
	}
	epoch := s.mutationEpoch
	segments := append([]*AnnIndex(nil), s.segments...)
	s.mu.RUnlock()

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

	merged := NewAnnIndexWithOptions(s.options)
	mergedSegments := 0
mergeLoop:
	for _, segment := range segments {
		for i, vector := range segment.ExportLiveVectors32() {
			// Shutdown must not wait for an entire HNSW segment rebuild. No
			// partially merged graph has been published yet, so aborting here
			// preserves the previous immutable generation for checkpointing.
			if i&63 == 0 {
				s.mu.RLock()
				closed := s.closed
				s.mu.RUnlock()
				if closed {
					return
				}
			}
			if err := merged.AddVector32(vector.ID, vector.Values); err != nil {
				return
			}
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
	// A concurrent search may still hold a snapshot of the replaced
	// generation. Retain it until Close rather than unmapping underneath it.
	s.retired = append(s.retired, segments[:mergedSegments]...)
	if len(s.segments) >= s.compactThreshold {
		s.requestCompactionLocked()
	}
	s.mu.Unlock()
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
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
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
	if k <= 0 {
		return nil, ErrInvalidK
	}
	indexes := s.routedIndexes(query)
	s.mu.RLock()
	configuredEF := s.options.EfSearch
	s.mu.RUnlock()

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
				if idx.Stats().Nodes == 0 {
					continue
				}
				// Preserve the configured quality profile across immutable
				// segments. The old hard-coded k*8 ceiling silently reduced an
				// efSearch=512 service to ef=80 and lost recall at 100k.
				segmentEF := max(64, k*8, configuredEF)
				results, err := idx.SearchWithDistancesEfInto(query, k, segmentEF, scratch[:0])
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
func (s *SegmentedIndex) routedIndexes(query []float64) []*AnnIndex {
	s.mu.Lock()
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
		return append(segments, delta)
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
		return append(segments, delta)
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
	return append(indexes, delta)
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
	s.mu.RLock()
	indexes := make([]*AnnIndex, 0, len(s.segments)+1)
	indexes = append(indexes, s.segments...)
	indexes = append(indexes, s.delta)
	s.mu.RUnlock()
	for _, idx := range indexes {
		idx.DeleteVector(id)
	}
	s.mu.Lock()
	s.mutationEpoch++
	s.requestCompactionLocked()
	s.mu.Unlock()
}

func (s *SegmentedIndex) Stats() Stats {
	s.mu.RLock()
	indexes := make([]*AnnIndex, 0, len(s.segments)+1)
	indexes = append(indexes, s.segments...)
	indexes = append(indexes, s.delta)
	s.mu.RUnlock()
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
	s.mu.RLock()
	segments := append([]*AnnIndex(nil), s.segments...)
	s.mu.RUnlock()
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
	s.mu.RLock()
	delta := s.delta
	s.mu.RUnlock()
	if delta == nil {
		return nil
	}
	return delta.ExportLiveVectors32()
}
