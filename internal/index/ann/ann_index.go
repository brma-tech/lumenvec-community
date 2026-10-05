package ann

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"math/bits"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"unsafe"

	vectorutil "lumenvec/internal/vector"
)

var (
	ErrInvalidK         = errors.New("k must be greater than 0")
	ErrInvalidVectorDim = errors.New("query dimension mismatch")
	ErrQuantizedIndex   = errors.New("quantized ANN segment is immutable")
)

const (
	MetricL2           = "l2"
	MetricCosine       = "cosine"
	MetricInnerProduct = "inner_product"
)

type node struct {
	id     int
	vector []float32
	// Slot references are bounded by the node count and fit in int32. Keeping
	// them compact halves adjacency storage for large indexes without changing
	// the public ID or distance representation.
	neighbors []int32
}

// AnnIndex is a graph-based ANN index inspired by HNSW/NSW principles.
type AnnIndex struct {
	nodes               []node
	idToSlot            map[int]int
	deleted             []bool
	deletedCount        int
	entrypoint          int
	searchEntrypoints   []int
	hasEntrypoint       bool
	dim                 int
	m                   int
	efConstruction      int
	efSearch            int
	seed                int64
	rnd                 *rand.Rand
	workspacePool       sync.Pool
	pruneScratch        []distancePair
	pruneSelected       []int
	constructionScratch []distancePair
	constructionLinks   []int
	adjacency           []int32
	adjacencyOffsets    []uint32
	mappedAdjacency     []byte
	mappedOffsets       []byte
	mappedOwner         snapshotMapping
	quantMin            []float32
	quantMax            []float32
	quantScale          []float32
	qvectors            []byte
	immutableIDs        []int64
	mappedIDs           []byte
	deletedBits         []uint64
	immutableNodeCount  int
	quantized           bool
	idsSorted           bool
	routeSignatures     []uint32
	metric              string
	diversifiedPruning  bool
	searchQueries       atomic.Uint64
	searchSegments      atomic.Uint64
	searchEFBudget      atomic.Uint64
	searchVisited       atomic.Uint64
	searchDistances     atomic.Uint64
	searchFrontierMax   atomic.Uint64
	mu                  sync.RWMutex
}

// SearchWorkStats reports cumulative work performed by ANN searches.
// Counters are diagnostic and do not affect search decisions.
type SearchWorkStats struct {
	Queries       uint64
	Visited       uint64
	DistanceCalls uint64
	MaxFrontier   uint64
}

// TopologyStats describes graph connectivity for offline build diagnostics.
type TopologyStats struct {
	Nodes          int
	AverageDegree  float64
	MaxDegree      int
	ReachableNodes int
}

func (a *AnnIndex) TopologyStats() TopologyStats {
	a.mu.RLock()
	defer a.mu.RUnlock()
	n := a.nodeCountLocked()
	if n == 0 {
		return TopologyStats{}
	}
	visited := make([]bool, n)
	queue := make([]int, 0, n)
	starts := a.searchEntrypoints
	if len(starts) == 0 {
		starts = []int{a.entrypoint}
	}
	for _, start := range starts {
		if start >= 0 && start < n && !visited[start] {
			visited[start] = true
			queue = append(queue, start)
		}
	}
	total, maxDegree := 0, 0
	for slot := 0; slot < n; slot++ {
		degree := a.neighborCountLocked(slot)
		total += degree
		if degree > maxDegree {
			maxDegree = degree
		}
	}
	for head := 0; head < len(queue); head++ {
		slot := queue[head]
		for pos, count := 0, a.neighborCountLocked(slot); pos < count; pos++ {
			next := a.neighborAtLocked(slot, pos)
			if next >= 0 && next < n && !visited[next] {
				visited[next] = true
				queue = append(queue, next)
			}
		}
	}
	return TopologyStats{Nodes: n, AverageDegree: float64(total) / float64(n), MaxDegree: maxDegree, ReachableNodes: len(queue)}
}

// LayoutState describes the immutable storage representation used by an
// index. It is diagnostic only and does not affect search behavior.
func (a *AnnIndex) LayoutState() (quantized, compacted, mapped bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.quantized, len(a.adjacencyOffsets) == a.nodeCountLocked()+1, len(a.mappedOffsets) != 0
}

func (a *AnnIndex) SearchWorkStats() SearchWorkStats {
	return SearchWorkStats{
		Queries: a.searchQueries.Load(), Visited: a.searchVisited.Load(),
		DistanceCalls: a.searchDistances.Load(), MaxFrontier: a.searchFrontierMax.Load(),
	}
}

// CompactAdjacency freezes the graph's adjacency lists into one contiguous
// buffer. It is intended for immutable segments after construction; callers
// must not add vectors after compacting.
func (a *AnnIndex) CompactAdjacency() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.compactAdjacencyLocked()
	a.rebuildRouteSignaturesLocked()
}

func (a *AnnIndex) SetDiversifiedPruning(enabled bool) {
	a.mu.Lock()
	a.diversifiedPruning = enabled
	a.mu.Unlock()
}

// SetSearchEntrypoints configures optional independent search starts for an
// immutable graph. It is intended for consolidated indexes; an empty list
// retains the historical single-entrypoint behavior.
func (a *AnnIndex) SetSearchEntrypoints(entries []int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.searchEntrypoints = append(a.searchEntrypoints[:0], entries...)
}

func (a *AnnIndex) compactAdjacencyLocked() {
	if len(a.adjacencyOffsets) == a.nodeCountLocked()+1 || len(a.mappedOffsets) != 0 {
		return
	}
	offsets := make([]uint32, len(a.nodes)+1)
	total := 0
	for n := range a.nodes {
		total += len(a.nodes[n].neighbors)
		offsets[n+1] = uint32(total)
	}
	adjacency := make([]int32, total)
	position := 0
	for n := range a.nodes {
		position += copy(adjacency[position:], a.nodes[n].neighbors)
		a.nodes[n].neighbors = nil
	}
	a.adjacency = adjacency
	a.adjacencyOffsets = offsets
}

func (a *AnnIndex) nodeCountLocked() int {
	if a.immutableNodeCount != 0 || a.quantized {
		return a.immutableNodeCount
	}
	return len(a.nodes)
}

func (a *AnnIndex) nodeIDLocked(slot int) int {
	if len(a.mappedIDs) != 0 {
		return int(int64(binary.LittleEndian.Uint64(a.mappedIDs[slot*8:])))
	}
	if len(a.immutableIDs) != 0 {
		return int(a.immutableIDs[slot])
	}
	return a.nodes[slot].id
}

func (a *AnnIndex) isDeletedLocked(slot int) bool {
	if len(a.deletedBits) != 0 {
		return a.deletedBits[slot>>6]&(uint64(1)<<uint(slot&63)) != 0
	}
	return slot < len(a.deleted) && a.deleted[slot]
}

func (a *AnnIndex) setDeletedLocked(slot int, deleted bool) {
	if len(a.deletedBits) != 0 {
		mask := uint64(1) << uint(slot&63)
		if deleted {
			a.deletedBits[slot>>6] |= mask
		} else {
			a.deletedBits[slot>>6] &^= mask
		}
		return
	}
	a.deleted[slot] = deleted
}

type snapshotMapping interface {
	Close() error
}

// Quantize converts an immutable graph from float32 payloads to one byte per
// dimension. Topology and IDs are unchanged. Search distances become
// approximate; callers should rerank candidates against the canonical store.
func (a *AnnIndex) Quantize() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.quantized || len(a.nodes) == 0 {
		return nil
	}
	a.compactAdjacencyLocked()
	a.rebuildRouteSignaturesLocked()
	if a.dim <= 0 {
		return ErrInvalidVectorDim
	}
	minimum := append([]float32(nil), a.nodes[0].vector...)
	maximum := append([]float32(nil), a.nodes[0].vector...)
	for n := 1; n < len(a.nodes); n++ {
		if len(a.nodes[n].vector) != a.dim {
			return ErrInvalidVectorDim
		}
		for d, value := range a.nodes[n].vector {
			if value < minimum[d] {
				minimum[d] = value
			}
			if value > maximum[d] {
				maximum[d] = value
			}
		}
	}
	if uint64(len(a.nodes)) > uint64(^uint(0)>>1)/uint64(a.dim) {
		return errors.New("quantized ANN arena exceeds address space")
	}
	arena := make([]byte, len(a.nodes)*a.dim)
	ids := make([]int64, len(a.nodes))
	deletedBits := make([]uint64, (len(a.nodes)+63)/64)
	idsSorted := true
	for n := range a.nodes {
		offset := n * a.dim
		encoded := arena[offset : offset+a.dim]
		for d, value := range a.nodes[n].vector {
			span := maximum[d] - minimum[d]
			if span > 0 {
				scaled := (value - minimum[d]) / span
				if scaled < 0 {
					scaled = 0
				} else if scaled > 1 {
					scaled = 1
				}
				encoded[d] = byte(scaled*255 + 0.5)
			}
		}
		a.nodes[n].vector = nil
		ids[n] = int64(a.nodes[n].id)
		if a.deleted[n] {
			deletedBits[n>>6] |= uint64(1) << uint(n&63)
		}
		if n > 0 && a.nodes[n-1].id >= a.nodes[n].id {
			idsSorted = false
		}
	}
	a.quantMin = minimum
	a.quantMax = maximum
	a.quantScale = make([]float32, a.dim)
	for d := range a.quantScale {
		a.quantScale[d] = (maximum[d] - minimum[d]) * (1.0 / 255.0)
	}
	a.qvectors = arena
	a.immutableIDs = ids
	a.deletedBits = deletedBits
	a.immutableNodeCount = len(a.nodes)
	a.nodes = nil
	a.deleted = nil
	a.quantized = true
	a.idsSorted = idsSorted
	if idsSorted {
		a.idToSlot = nil
	}
	return nil
}

const routeSignatureBits = 24

func routeSignature64(vector []float64) uint32 {
	var signature uint32
	for dimension, value := range vector[:min(len(vector), routeSignatureBits)] {
		if value >= 0 {
			signature |= uint32(1) << uint(dimension)
		}
	}
	return signature
}

func routeSignature32(vector []float32) uint32 {
	var signature uint32
	for dimension, value := range vector[:min(len(vector), routeSignatureBits)] {
		if value >= 0 {
			signature |= uint32(1) << uint(dimension)
		}
	}
	return signature
}

func (a *AnnIndex) rebuildRouteSignaturesLocked() {
	count := a.nodeCountLocked()
	if count == 0 || a.dim == 0 {
		a.routeSignatures = nil
		return
	}
	signatures := make([]uint32, 0, count)
	for slot := 0; slot < count; slot++ {
		if a.isDeletedLocked(slot) {
			continue
		}
		var signature uint32
		if a.quantized {
			for dimension, value := range a.qvectorLocked(slot)[:min(a.dim, routeSignatureBits)] {
				decoded := a.quantMin[dimension] +
					(a.quantMax[dimension]-a.quantMin[dimension])*float32(value)/255
				if decoded >= 0 {
					signature |= uint32(1) << uint(dimension)
				}
			}
		} else {
			signature = routeSignature32(a.nodes[slot].vector)
		}
		signatures = append(signatures, signature)
	}
	sort.Slice(signatures, func(left, right int) bool { return signatures[left] < signatures[right] })
	write := 0
	for _, signature := range signatures {
		if write == 0 || signatures[write-1] != signature {
			signatures[write] = signature
			write++
		}
	}
	a.routeSignatures = signatures[:write]
}

func (a *AnnIndex) routeDistance(signature uint32) int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	if len(a.routeSignatures) == 0 {
		return routeSignatureBits + 1
	}
	position := sort.Search(len(a.routeSignatures), func(index int) bool {
		return a.routeSignatures[index] >= signature
	})
	if position < len(a.routeSignatures) && a.routeSignatures[position] == signature {
		return 0
	}
	best := routeSignatureBits + 1
	for _, candidate := range a.routeSignatures {
		if distance := bits.OnesCount32(candidate ^ signature); distance < best {
			best = distance
			if best == 1 {
				break
			}
		}
	}
	return best
}

func (a *AnnIndex) routeSignatureSnapshot() []uint32 {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return append([]uint32(nil), a.routeSignatures...)
}

func (a *AnnIndex) qvectorLocked(slot int) []byte {
	offset := slot * a.dim
	return a.qvectors[offset : offset+a.dim]
}

func (a *AnnIndex) nodeVectorLocked(slot int) []float32 {
	if !a.quantized {
		return a.nodes[slot].vector
	}
	decoded := make([]float32, a.dim)
	for d, value := range a.qvectorLocked(slot) {
		decoded[d] = a.quantMin[d] + (a.quantMax[d]-a.quantMin[d])*float32(value)/255
	}
	return decoded
}

func (a *AnnIndex) distanceToNodeLocked(query []float32, slot int) float64 {
	if !a.quantized {
		return metricDistance32(a.metric, query, a.nodes[slot].vector)
	}
	if a.metric == MetricL2 {
		// L2 is the hot path for the vNext benchmark. Keep the arithmetic in
		// float32 until the final accumulation and replace the per-dimension
		// float64 conversion/division used by the old path. Quantized values
		// are already bounded to [0,255], so multiplying by 1/255 is exact
		// enough for the existing approximate-distance contract.
		var distance float32
		scales := a.quantScale
		for d, value := range a.qvectorLocked(slot) {
			span := float32(0)
			if d < len(scales) {
				span = scales[d]
			} else {
				// Snapshots written before quantScale existed remain searchable.
				span = (a.quantMax[d] - a.quantMin[d]) * (1.0 / 255.0)
			}
			decoded := a.quantMin[d] + span*float32(value)
			delta := query[d] - decoded
			distance += delta * delta
		}
		return float64(distance)
	}
	var distance, dot, queryNorm, vectorNorm float64
	for d, value := range a.qvectorLocked(slot) {
		decoded := a.quantMin[d] + (a.quantMax[d]-a.quantMin[d])*float32(value)/255
		q := float64(query[d])
		v := float64(decoded)
		switch a.metric {
		case MetricCosine:
			dot += q * v
			queryNorm += q * q
			vectorNorm += v * v
		case MetricInnerProduct:
			dot += q * v
		default:
			delta := q - v
			distance += delta * delta
		}
	}
	return finishMetricDistance(a.metric, distance, dot, queryNorm, vectorNorm)
}

func (a *AnnIndex) distanceBetweenNodesLocked(left, right int) float64 {
	if !a.quantized {
		return metricDistance32(a.metric, a.nodes[left].vector, a.nodes[right].vector)
	}
	var distance, dot, leftNorm, rightNorm float64
	leftVector := a.qvectorLocked(left)
	rightVector := a.qvectorLocked(right)
	for d, leftValue := range leftVector {
		leftDecoded := a.quantMin[d] + (a.quantMax[d]-a.quantMin[d])*float32(leftValue)/255
		rightDecoded := a.quantMin[d] + (a.quantMax[d]-a.quantMin[d])*float32(rightVector[d])/255
		l := float64(leftDecoded)
		r := float64(rightDecoded)
		switch a.metric {
		case MetricCosine:
			dot += l * r
			leftNorm += l * l
			rightNorm += r * r
		case MetricInnerProduct:
			dot += l * r
		default:
			delta := l - r
			distance += delta * delta
		}
	}
	return finishMetricDistance(a.metric, distance, dot, leftNorm, rightNorm)
}

func (a *AnnIndex) neighborsForSlot(slot int) []int32 {
	if len(a.adjacencyOffsets) == a.nodeCountLocked()+1 {
		return a.adjacency[a.adjacencyOffsets[slot]:a.adjacencyOffsets[slot+1]]
	}
	return a.nodes[slot].neighbors
}

func (a *AnnIndex) neighborCountLocked(slot int) int {
	if len(a.mappedOffsets) != 0 {
		start := binary.LittleEndian.Uint64(a.mappedOffsets[slot*8:])
		end := binary.LittleEndian.Uint64(a.mappedOffsets[(slot+1)*8:])
		return int(end - start)
	}
	return len(a.neighborsForSlot(slot))
}

func (a *AnnIndex) neighborAtLocked(slot, position int) int {
	if len(a.mappedOffsets) != 0 {
		start := binary.LittleEndian.Uint64(a.mappedOffsets[slot*8:])
		offset := (start + uint64(position)) * 4
		return int(int32(binary.LittleEndian.Uint32(a.mappedAdjacency[offset:])))
	}
	return int(a.neighborsForSlot(slot)[position])
}

type Stats struct {
	Nodes   int
	Deleted int
}

// MemoryStats reports an allocation-independent estimate of the ANN's
// resident structural data. It intentionally excludes Go runtime overhead and
// temporary query workspaces, making it suitable for comparing index layouts.
type MemoryStats struct {
	VectorBytes    uint64
	QuantizedBytes uint64
	NodeBytes      uint64
	AdjacencyBytes uint64
	MapBytes       uint64
	MappedBytes    uint64
	RouteBytes     uint64
	TotalBytes     uint64
}

func (a *AnnIndex) MemoryStats() MemoryStats {
	a.mu.RLock()
	defer a.mu.RUnlock()
	var vectors, adjacency uint64
	for _, n := range a.nodes {
		vectors += uint64(len(n.vector)) * 4
		if len(a.adjacencyOffsets) == a.nodeCountLocked()+1 {
			continue
		}
		adjacency += uint64(cap(n.neighbors)) * 4
	}
	if len(a.adjacencyOffsets) == a.nodeCountLocked()+1 {
		adjacency = uint64(cap(a.adjacency)) * 4
		adjacency += uint64(cap(a.adjacencyOffsets)) * 4
	}
	adjacency += uint64(len(a.mappedOffsets) + len(a.mappedAdjacency))
	// This is a stable estimate for the integer map payload; runtime bucket
	// overhead is deliberately excluded so profiles remain comparable.
	mapBytes := uint64(len(a.idToSlot)) * 16
	quantized := uint64(len(a.qvectors)) + uint64(len(a.quantMin)+len(a.quantMax))*4
	mapped := uint64(len(a.mappedOffsets) + len(a.mappedAdjacency))
	nodeBytes := uint64(cap(a.nodes))*uint64(unsafe.Sizeof(node{})) +
		uint64(len(a.immutableIDs))*8 + uint64(len(a.deletedBits))*8 +
		uint64(cap(a.deleted))
	routeBytes := uint64(cap(a.routeSignatures)) * 4
	return MemoryStats{VectorBytes: vectors, QuantizedBytes: quantized, NodeBytes: nodeBytes, AdjacencyBytes: adjacency, MapBytes: mapBytes, MappedBytes: mapped + uint64(len(a.mappedIDs)), RouteBytes: routeBytes, TotalBytes: vectors + quantized + nodeBytes + adjacency + mapBytes + routeBytes}
}

type Result struct {
	ID       int
	Distance float64
}

// ExactDistances reports that this index ranks and returns distances computed
// from the canonical float32 vectors rather than a compressed approximation.
func (a *AnnIndex) ExactDistances() bool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return !a.quantized
}

// Close releases an optional file mapping owned by an index reopened from a
// columnar snapshot. In-memory indexes have nothing to release.
func (a *AnnIndex) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.mappedOwner == nil {
		return nil
	}
	owner := a.mappedOwner
	a.mappedOwner = nil
	a.qvectors = nil
	a.mappedIDs = nil
	a.mappedOffsets = nil
	a.mappedAdjacency = nil
	a.routeSignatures = nil
	return owner.Close()
}

func (a *AnnIndex) exportLiveVectors() []BatchVector {
	a.mu.RLock()
	defer a.mu.RUnlock()
	count := a.nodeCountLocked()
	out := make([]BatchVector, 0, count-a.deletedCount)
	for slot := 0; slot < count; slot++ {
		if a.isDeletedLocked(slot) {
			continue
		}
		out = append(out, BatchVector{ID: a.nodeIDLocked(slot), Values: vectorutil.ToFloat64(a.nodeVectorLocked(slot))})
	}
	return out
}

type Options struct {
	M                  int
	EfConstruction     int
	EfSearch           int
	Seed               int64
	QuantizeSegments   bool
	SegmentRouting     bool
	Metric             string
	DiversifiedPruning bool
	// DiversifiedExistingPruning enables experimental full-list pruning for
	// existing HNSW neighbor arrays. It is disabled by default.
	DiversifiedExistingPruning bool
	// HierarchicalBuildScratchBudgetBytes limits the deterministic upper bound
	// of the hierarchical builder's private neighbor-cache scratch. Zero keeps
	// backward-compatible unlimited behavior. The candidate generation is
	// rejected before graph construction when the bound exceeds this budget.
	HierarchicalBuildScratchBudgetBytes uint64
	// HierarchicalUpperLayerEF controls the optional beam used while descending
	// upper HNSW layers. Values <= 1 preserve classic greedy navigation.
	HierarchicalUpperLayerEF       int
	HierarchicalLevelMultiplier    float64
	HierarchicalMaxLevel           int
	HierarchicalEntrypointPoolSize int
}

func NewAnnIndex() *AnnIndex {
	return NewAnnIndexWithOptions(Options{})
}

func NewAnnIndexWithOptions(opts Options) *AnnIndex {
	return NewAnnIndexWithOptionsCapacity(opts, 0)
}

// NewAnnIndexWithOptionsCapacity is the bulk-ingest constructor. A positive
// capacity reserves node/map storage for an immutable segment, avoiding
// repeated growth while vectors are appended.
func NewAnnIndexWithOptionsCapacity(opts Options, capacity int) *AnnIndex {
	if opts.M <= 0 {
		opts.M = 16
	}
	if opts.EfConstruction <= 0 {
		opts.EfConstruction = 64
	}
	if opts.EfSearch <= 0 {
		opts.EfSearch = 64
	}
	if opts.Seed == 0 {
		opts.Seed = 42
	}
	idx := &AnnIndex{
		idToSlot:           make(map[int]int, capacityOrZero(capacity)),
		nodes:              make([]node, 0, capacityOrZero(capacity)),
		deleted:            make([]bool, 0, capacityOrZero(capacity)),
		m:                  opts.M,
		efConstruction:     opts.EfConstruction,
		efSearch:           opts.EfSearch,
		seed:               opts.Seed,
		metric:             normalizeMetric(opts.Metric),
		diversifiedPruning: opts.DiversifiedPruning,
		rnd:                rand.New(rand.NewSource(opts.Seed)), // #nosec G404 -- deterministic ANN construction, not security randomness
	}
	idx.workspacePool.New = func() any {
		return &searchWorkspace{}
	}
	return idx
}

func normalizeMetric(metric string) string {
	switch metric {
	case MetricCosine, MetricInnerProduct:
		return metric
	default:
		return MetricL2
	}
}

func metricDistance32(metric string, left, right []float32) float64 {
	if metric == MetricL2 {
		return vectorutil.SquaredEuclideanDistance32FastSameLen(left, right)
	}
	var distance, dot, leftNorm, rightNorm float64
	for dimension, leftValue := range left {
		l := float64(leftValue)
		r := float64(right[dimension])
		switch metric {
		case MetricCosine:
			dot += l * r
			leftNorm += l * l
			rightNorm += r * r
		case MetricInnerProduct:
			dot += l * r
		default:
			delta := l - r
			distance += delta * delta
		}
	}
	return finishMetricDistance(metric, distance, dot, leftNorm, rightNorm)
}

func finishMetricDistance(metric string, l2, dot, leftNorm, rightNorm float64) float64 {
	switch metric {
	case MetricCosine:
		if leftNorm == 0 || rightNorm == 0 {
			return 1
		}
		return 1 - dot/(math.Sqrt(leftNorm)*math.Sqrt(rightNorm))
	case MetricInnerProduct:
		return -dot
	default:
		return l2
	}
}

func capacityOrZero(a int) int {
	if a > 0 {
		return a
	}
	return 0
}

func (a *AnnIndex) AddVector(id int, vector []float64) error {
	if len(vector) == 0 {
		return ErrInvalidVectorDim
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	if a.quantized {
		return ErrQuantizedIndex
	}

	if a.dim == 0 {
		a.dim = len(vector)
	}
	if len(vector) != a.dim {
		return ErrInvalidVectorDim
	}

	vecCopy := vectorutil.ToFloat32(vector)

	if slot, ok := a.idToSlot[id]; ok {
		a.nodes[slot].vector = vecCopy
		if a.deleted[slot] {
			a.deleted[slot] = false
			a.deletedCount--
		}
		return nil
	}

	slot := len(a.nodes)
	a.nodes = append(a.nodes, node{
		id:        id,
		vector:    vecCopy,
		neighbors: make([]int32, 0, a.m),
	})
	a.idToSlot[id] = slot
	a.deleted = append(a.deleted, false)

	if !a.hasEntrypoint {
		a.entrypoint = slot
		a.hasEntrypoint = true
		return nil
	}

	candidates := a.searchCandidates32Locked(vecCopy, a.efConstruction)
	if cap(a.constructionLinks) < a.m {
		a.constructionLinks = make([]int, a.m)
	}
	linkSlots := nearestIDsInto(candidates, a.m, a.constructionLinks[:a.m])
	for _, neighborSlot := range linkSlots {
		if neighborSlot == slot {
			continue
		}
		a.nodes[slot].addNeighbor(slot, neighborSlot)
		a.nodes[neighborSlot].addNeighbor(neighborSlot, slot)
		a.pruneNeighborsLocked(neighborSlot)
	}
	a.pruneNeighborsLocked(slot)

	if a.rnd.Intn(100) < 5 {
		a.entrypoint = slot
	}
	return nil
}

// AddVector32 inserts an already normalized float32 vector without widening
// it to float64. It is intended for segment transfer and checkpoint restore.
func (a *AnnIndex) AddVector32(id int, vector []float32) error {
	if len(vector) == 0 {
		return ErrInvalidVectorDim
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.addVector32Locked(id, vector)
}

// addVector32Locked appends one vector while the caller owns a write lock.
// Bulk builders use this primitive to hold the lock once for an entire
// generation, avoiding one mutex transition per vector without changing the
// graph construction semantics of incremental inserts.
func (a *AnnIndex) addVector32Locked(id int, vector []float32) error {
	if len(vector) == 0 {
		return ErrInvalidVectorDim
	}
	if a.quantized {
		return ErrQuantizedIndex
	}
	if a.dim == 0 {
		a.dim = len(vector)
	}
	if len(vector) != a.dim {
		return ErrInvalidVectorDim
	}
	vecCopy := append([]float32(nil), vector...)
	if slot, ok := a.idToSlot[id]; ok {
		a.nodes[slot].vector = vecCopy
		if a.deleted[slot] {
			a.deleted[slot] = false
			a.deletedCount--
		}
		return nil
	}
	slot := len(a.nodes)
	a.nodes = append(a.nodes, node{id: id, vector: vecCopy, neighbors: make([]int32, 0, a.m)})
	a.idToSlot[id] = slot
	a.deleted = append(a.deleted, false)
	if !a.hasEntrypoint {
		a.entrypoint = slot
		a.hasEntrypoint = true
		return nil
	}
	candidates := a.searchCandidates32Locked(vecCopy, a.efConstruction)
	if cap(a.constructionLinks) < a.m {
		a.constructionLinks = make([]int, a.m)
	}
	linkSlots := nearestIDsInto(candidates, a.m, a.constructionLinks[:a.m])
	for _, neighborSlot := range linkSlots {
		if neighborSlot == slot {
			continue
		}
		a.nodes[slot].addNeighbor(slot, neighborSlot)
		a.nodes[neighborSlot].addNeighbor(neighborSlot, slot)
		a.pruneNeighborsLocked(neighborSlot)
	}
	a.pruneNeighborsLocked(slot)
	if a.rnd.Intn(100) < 5 {
		a.entrypoint = slot
	}
	return nil
}

func (a *AnnIndex) Search(query []float64, k int) ([]int, error) {
	if k <= 0 {
		return nil, ErrInvalidK
	}

	a.mu.RLock()
	defer a.mu.RUnlock()

	if len(query) == 0 || len(query) != a.dim {
		return nil, ErrInvalidVectorDim
	}
	if a.nodeCountLocked() == 0 {
		return []int{}, nil
	}

	return a.searchIDs64Locked(query, k, a.efSearch), nil
}

func (a *AnnIndex) SearchWithDistances(query []float64, k int) ([]Result, error) {
	return a.SearchWithDistancesInto(query, k, nil)
}

func (a *AnnIndex) SearchWithDistancesInto(query []float64, k int, dst []Result) ([]Result, error) {
	return a.SearchWithDistancesContext(context.Background(), query, k, dst)
}

func (a *AnnIndex) SearchWithDistancesContext(ctx context.Context, query []float64, k int, dst []Result) ([]Result, error) {
	a.mu.RLock()
	ef := a.efSearch
	a.mu.RUnlock()
	return a.SearchWithDistancesEfContext(ctx, query, k, ef, dst)
}

// SearchWithDistancesEfInto allows a segmented coordinator to distribute a
// bounded probe budget across many immutable graphs. Standalone callers keep
// the configured efSearch through SearchWithDistancesInto.
func (a *AnnIndex) SearchWithDistancesEfInto(query []float64, k, ef int, dst []Result) ([]Result, error) {
	return a.SearchWithDistancesEfContext(context.Background(), query, k, ef, dst)
}

func (a *AnnIndex) SearchWithDistancesEfContext(ctx context.Context, query []float64, k, ef int, dst []Result) ([]Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, ErrInvalidK
	}

	a.mu.RLock()
	defer a.mu.RUnlock()

	if len(query) == 0 || len(query) != a.dim {
		return nil, ErrInvalidVectorDim
	}
	if a.nodeCountLocked() == 0 {
		return dst[:0], nil
	}
	if ef < k {
		ef = k
	}
	if a.efSearch >= k && ef > a.efSearch {
		ef = a.efSearch
	}
	a.searchEFBudget.Add(uint64(ef))
	a.searchSegments.Add(1)
	results := a.searchResults64Locked(query, k, ef, dst, ctx)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return results, nil
}

// SearchBudgetState exposes the same budget contract as SegmentedIndex for
// single-index profiles used by the HTTP benchmark.
func (a *AnnIndex) SearchBudgetState() (queries, segments, efBudget uint64) {
	return a.searchQueries.Load(), a.searchSegments.Load(), a.searchEFBudget.Load()
}

func (a *AnnIndex) DeleteVector(id int) {
	a.mu.Lock()
	defer a.mu.Unlock()

	slot, ok := a.lookupSlotLocked(id)
	if !ok || a.isDeletedLocked(slot) {
		return
	}
	a.setDeletedLocked(slot, true)
	a.deletedCount++
}

func (a *AnnIndex) lookupSlotLocked(id int) (int, bool) {
	if a.idToSlot != nil {
		slot, ok := a.idToSlot[id]
		return slot, ok
	}
	if !a.idsSorted {
		return 0, false
	}
	count := a.nodeCountLocked()
	slot := sort.Search(count, func(slot int) bool { return a.nodeIDLocked(slot) >= id })
	return slot, slot < count && a.nodeIDLocked(slot) == id
}

func (a *AnnIndex) Stats() Stats {
	a.mu.RLock()
	defer a.mu.RUnlock()

	return Stats{
		Nodes:   a.nodeCountLocked(),
		Deleted: a.deletedCount,
	}
}

// IsEmpty reports whether the index has no searchable nodes. It avoids
// materializing the full Stats value on hot fanout paths.
func (a *AnnIndex) IsEmpty() bool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.nodeCountLocked() == 0
}

func (a *AnnIndex) searchCandidates32Locked(query []float32, ef int) []distancePair {
	if ef <= 0 {
		ef = a.efSearch
	}

	ws := a.workspacePool.Get().(*searchWorkspace)
	ws.reset(ef, a.nodeCountLocked())
	defer a.workspacePool.Put(ws)

	candidates := a.searchCandidates32WithWorkspaceLocked(ws, ef, query)
	if cap(a.constructionScratch) < len(candidates) {
		a.constructionScratch = make([]distancePair, len(candidates))
	} else {
		a.constructionScratch = a.constructionScratch[:len(candidates)]
	}
	copy(a.constructionScratch, candidates)
	return a.constructionScratch
}

func (a *AnnIndex) searchIDs64Locked(query []float64, k int, ef int) []int {
	if ef <= 0 {
		ef = a.efSearch
	}

	ws := a.workspacePool.Get().(*searchWorkspace)
	ws.reset(ef, a.nodeCountLocked())
	defer a.workspacePool.Put(ws)

	query32 := ws.query32From64(query)
	candidates := a.searchCandidates32WithWorkspaceLocked(ws, ef, query32)
	candidates = a.liveCandidates(candidates)
	return a.nearestExternalIDs(candidates, k)
}

func (a *AnnIndex) searchResults64Locked(query []float64, k int, ef int, dst []Result, contexts ...context.Context) []Result {
	if ef <= 0 {
		ef = a.efSearch
	}

	ws := a.workspacePool.Get().(*searchWorkspace)
	ws.reset(ef, a.nodeCountLocked())
	if len(contexts) > 0 {
		ws.queryContext = contexts[0]
	}
	defer func() {
		ws.queryContext = nil
		a.workspacePool.Put(ws)
	}()

	query32 := ws.query32From64(query)
	candidates := a.searchCandidates32WithWorkspaceLocked(ws, ef, query32)
	a.searchQueries.Add(1)
	a.searchVisited.Add(ws.visitedCount)
	a.searchDistances.Add(ws.distanceCount)
	for {
		old := a.searchFrontierMax.Load()
		if ws.maxFrontier <= old || a.searchFrontierMax.CompareAndSwap(old, ws.maxFrontier) {
			break
		}
	}
	candidates = a.liveCandidates(candidates)
	return a.nearestResults(candidates, k, dst)
}

func (a *AnnIndex) searchCandidates32WithWorkspaceLocked(ws *searchWorkspace, ef int, query []float32) []distancePair {
	minQ := ws.minQ
	maxQ := ws.maxQ

	starts := a.searchEntrypoints
	if len(starts) == 0 {
		starts = []int{a.entrypoint}
	}
	for _, start := range starts {
		if start < 0 || start >= a.nodeCountLocked() || ws.isVisited(start) {
			continue
		}
		startDist := a.distanceToNodeLocked(query, start)
		ws.distanceCount++
		minQ.Push(distancePair{id: start, distance: startDist})
		maxQ.Push(distancePair{id: start, distance: startDist})
		ws.markVisited(start)
		ws.visitedCount++
	}

	for iteration := 0; minQ.Len() > 0; iteration++ {
		if iteration%64 == 0 && ws.queryContext != nil && ws.queryContext.Err() != nil {
			break
		}
		current := minQ.Pop()

		worst := maxQ.Peek()
		if maxQ.Len() >= ef && current.distance > worst.distance {
			break
		}

		for neighborPosition, neighborCount := 0, a.neighborCountLocked(current.id); neighborPosition < neighborCount; neighborPosition++ {
			nid := a.neighborAtLocked(current.id, neighborPosition)
			if !ws.visit(nid) {
				continue
			}
			ws.visitedCount++

			dist := a.distanceToNodeLocked(query, nid)
			ws.distanceCount++
			dp := distancePair{id: nid, distance: dist}
			if maxQ.Len() < ef {
				minQ.Push(dp)
				maxQ.Push(dp)
				if uint64(minQ.Len()) > ws.maxFrontier {
					ws.maxFrontier = uint64(minQ.Len())
				}
				continue
			}
			if dist < maxQ.Peek().distance {
				minQ.Push(dp)
				maxQ.ReplaceTop(dp)
				if uint64(minQ.Len()) > ws.maxFrontier {
					ws.maxFrontier = uint64(minQ.Len())
				}
			}
		}
	}

	ws.minQ = minQ[:0]
	return ws.drainCandidates(maxQ)
}

func (w *searchWorkspace) drainCandidates(maxQ maxDistHeap) []distancePair {
	out := w.candidates[:0]
	if cap(out) < maxQ.Len() {
		out = make([]distancePair, 0, maxQ.Len())
	}
	out = out[:maxQ.Len()]
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = maxQ.Pop()
	}
	w.maxQ = maxQ[:0]
	w.candidates = out
	return out
}

type searchWorkspace struct {
	queryContext  context.Context
	visitedMarks  []uint32
	visitEpoch    uint32
	minQ          minDistHeap
	maxQ          maxDistHeap
	candidates    []distancePair
	query32       []float32
	visitedCount  uint64
	distanceCount uint64
	maxFrontier   uint64
}

func (w *searchWorkspace) reset(ef int, nodes int) {
	w.visitedCount = 0
	w.distanceCount = 0
	w.maxFrontier = 0
	if cap(w.visitedMarks) < nodes {
		// Grow geometrically. During bulk HNSW construction the number of
		// nodes increases one at a time; allocating exactly `nodes` here would
		// copy the mark table on every insertion and create O(n²) allocation
		// traffic. The spare capacity is private to this pooled workspace and
		// is bounded by the index size, so it does not change query semantics.
		newCap := cap(w.visitedMarks) * 2
		if newCap < 1024 {
			newCap = 1024
		}
		if newCap < nodes {
			newCap = nodes
		}
		w.visitedMarks = make([]uint32, nodes, newCap)
	} else {
		w.visitedMarks = w.visitedMarks[:nodes]
	}
	w.visitEpoch++
	if w.visitEpoch == 0 {
		clear(w.visitedMarks)
		w.visitEpoch = 1
	}
	if cap(w.minQ) < ef {
		w.minQ = make(minDistHeap, 0, ef)
	} else {
		w.minQ = w.minQ[:0]
	}
	if cap(w.maxQ) < ef {
		w.maxQ = make(maxDistHeap, 0, ef)
	} else {
		w.maxQ = w.maxQ[:0]
	}
	if cap(w.candidates) < ef {
		w.candidates = make([]distancePair, 0, ef)
	} else {
		w.candidates = w.candidates[:0]
	}
}

func (w *searchWorkspace) isVisited(slot int) bool {
	return slot >= 0 && slot < len(w.visitedMarks) && w.visitedMarks[slot] == w.visitEpoch
}

func (w *searchWorkspace) query32From64(query []float64) []float32 {
	if cap(w.query32) < len(query) {
		w.query32 = make([]float32, len(query))
	} else {
		w.query32 = w.query32[:len(query)]
	}
	for i, value := range query {
		w.query32[i] = float32(value)
	}
	return w.query32
}

func (w *searchWorkspace) markVisited(slot int) {
	w.visitedMarks[slot] = w.visitEpoch
}

func (w *searchWorkspace) visit(slot int) bool {
	if w.visitedMarks[slot] == w.visitEpoch {
		return false
	}
	w.visitedMarks[slot] = w.visitEpoch
	return true
}

func (a *AnnIndex) liveCandidates(candidates []distancePair) []distancePair {
	if a.deletedCount == 0 {
		return candidates
	}
	out := candidates[:0]
	for _, candidate := range candidates {
		if candidate.id >= 0 && candidate.id < a.nodeCountLocked() && a.isDeletedLocked(candidate.id) {
			continue
		}
		out = append(out, candidate)
	}
	return out
}

func (a *AnnIndex) pruneNeighborsLocked(slot int) {
	n := &a.nodes[slot]
	if len(n.neighbors) <= a.m {
		return
	}
	if cap(a.pruneScratch) < len(n.neighbors) {
		a.pruneScratch = make([]distancePair, 0, len(n.neighbors))
	}
	pairs := a.pruneScratch[:0]
	for _, rawNID := range n.neighbors {
		nid := int(rawNID)
		pairs = append(pairs, distancePair{
			id:       nid,
			distance: a.distanceBetweenNodesLocked(slot, nid),
		})
	}
	if !a.diversifiedPruning {
		keep := min(a.m, len(pairs))
		selectNearest(pairs, keep)
		n.neighbors = n.neighbors[:keep]
		for i := 0; i < keep; i++ {
			n.neighbors[i] = int32(pairs[i].id)
		}
		a.pruneScratch = pairs
		return
	}
	selectNearest(pairs, len(pairs))
	if cap(a.pruneSelected) < a.m {
		a.pruneSelected = make([]int, 0, a.m)
	}
	selected := a.pruneSelected[:0]
	for _, candidate := range pairs {
		diverse := true
		diversityChecks := min(len(selected), 4)
		for _, selectedID := range selected[:diversityChecks] {
			if a.distanceBetweenNodesLocked(candidate.id, selectedID) < candidate.distance {
				diverse = false
				break
			}
		}
		if diverse {
			selected = append(selected, candidate.id)
			if len(selected) == a.m {
				break
			}
		}
	}
	// Highly collinear datasets may not provide m diverse neighbors. Fill the
	// remaining degree with the nearest candidates to preserve connectivity.
	for _, candidate := range pairs {
		if len(selected) == a.m {
			break
		}
		alreadySelected := false
		for _, selectedID := range selected {
			if selectedID == candidate.id {
				alreadySelected = true
				break
			}
		}
		if !alreadySelected {
			selected = append(selected, candidate.id)
		}
	}
	n.neighbors = n.neighbors[:len(selected)]
	for i, selectedID := range selected {
		n.neighbors[i] = int32(selectedID)
	}
	a.pruneScratch = pairs
	a.pruneSelected = selected
}

func (n *node) addNeighbor(self, id int) {
	if id == self {
		return
	}
	for _, existing := range n.neighbors {
		if int(existing) == id {
			return
		}
	}
	n.neighbors = append(n.neighbors, int32(id))
}

type distancePair struct {
	id       int
	distance float64
}

func nearestIDs(candidates []distancePair, k int) []int {
	if k > len(candidates) {
		k = len(candidates)
	}
	if k <= 0 {
		return []int{}
	}

	selectNearest(candidates, k)

	out := make([]int, 0, k)
	for i := 0; i < k; i++ {
		out = append(out, candidates[i].id)
	}
	return out
}

func nearestIDsInto(candidates []distancePair, k int, dst []int) []int {
	if k > len(candidates) {
		k = len(candidates)
	}
	if k <= 0 {
		return dst[:0]
	}
	selectNearest(candidates, k)
	dst = dst[:k]
	for i := 0; i < k; i++ {
		dst[i] = candidates[i].id
	}
	return dst
}

func (a *AnnIndex) nearestExternalIDs(candidates []distancePair, k int) []int {
	if k > len(candidates) {
		k = len(candidates)
	}
	if k <= 0 {
		return []int{}
	}

	selectNearest(candidates, k)

	out := make([]int, 0, k)
	for i := 0; i < k; i++ {
		out = append(out, a.nodeIDLocked(candidates[i].id))
	}
	return out
}

func (a *AnnIndex) nearestResults(candidates []distancePair, k int, dst []Result) []Result {
	if k > len(candidates) {
		k = len(candidates)
	}
	if k <= 0 {
		return dst[:0]
	}

	selectNearest(candidates, k)

	out := dst[:0]
	if cap(out) < k {
		out = make([]Result, 0, k)
	}
	for i := 0; i < k; i++ {
		out = append(out, Result{ID: a.nodeIDLocked(candidates[i].id), Distance: candidates[i].distance})
	}
	return out
}

func selectNearest(candidates []distancePair, k int) {
	// partial selection sort for small-k use.
	for i := 0; i < k; i++ {
		best := i
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].distance < candidates[best].distance {
				best = j
			}
		}
		candidates[i], candidates[best] = candidates[best], candidates[i]
	}
}

func squaredDistance32(a, b []float32) float64 {
	sum := float64(0)
	i := 0
	for limit := len(a) - len(a)%4; i < limit; i += 4 {
		diff0 := float64(a[i] - b[i])
		diff1 := float64(a[i+1] - b[i+1])
		diff2 := float64(a[i+2] - b[i+2])
		diff3 := float64(a[i+3] - b[i+3])
		sum += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
	}
	for ; i < len(a); i++ {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}
	return sum
}

type minDistHeap []distancePair

func (h minDistHeap) Len() int { return len(h) }
func (h minDistHeap) Peek() distancePair {
	return h[0]
}
func (h *minDistHeap) Push(x distancePair) {
	*h = append(*h, x)
	upMin(*h, len(*h)-1)
}
func (h *minDistHeap) Pop() distancePair {
	old := *h
	n := len(old) - 1
	old[0], old[n] = old[n], old[0]
	downMin(old[:n], 0)
	item := old[n]
	*h = old[:n]
	return item
}

type maxDistHeap []distancePair

func (h maxDistHeap) Len() int { return len(h) }
func (h maxDistHeap) Peek() distancePair {
	return h[0]
}
func (h *maxDistHeap) Push(x distancePair) {
	*h = append(*h, x)
	upMax(*h, len(*h)-1)
}
func (h *maxDistHeap) ReplaceTop(x distancePair) {
	old := *h
	old[0] = x
	downMax(old, 0)
}
func (h *maxDistHeap) Pop() distancePair {
	old := *h
	n := len(old) - 1
	old[0], old[n] = old[n], old[0]
	downMax(old[:n], 0)
	item := old[n]
	*h = old[:n]
	return item
}

func upMin(h minDistHeap, j int) {
	for {
		i := (j - 1) / 2
		if i == j || h[j].distance >= h[i].distance {
			break
		}
		h[i], h[j] = h[j], h[i]
		j = i
	}
}

func downMin(h minDistHeap, i int) {
	for {
		left := 2*i + 1
		if left >= len(h) {
			break
		}
		child := left
		right := left + 1
		if right < len(h) && h[right].distance < h[left].distance {
			child = right
		}
		if h[i].distance <= h[child].distance {
			break
		}
		h[i], h[child] = h[child], h[i]
		i = child
	}
}

func upMax(h maxDistHeap, j int) {
	for {
		i := (j - 1) / 2
		if i == j || h[j].distance <= h[i].distance {
			break
		}
		h[i], h[j] = h[j], h[i]
		j = i
	}
}

func downMax(h maxDistHeap, i int) {
	for {
		left := 2*i + 1
		if left >= len(h) {
			break
		}
		child := left
		right := left + 1
		if right < len(h) && h[right].distance > h[left].distance {
			child = right
		}
		if h[i].distance >= h[child].distance {
			break
		}
		h[i], h[child] = h[child], h[i]
		i = child
	}
}
