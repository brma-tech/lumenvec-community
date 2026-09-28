package ann

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"unsafe"

	vectorutil "lumenvec/internal/vector"
)

const hierarchicalMaxLevel = 32

var ErrHierarchicalBuildScratchBudget = errors.New("hierarchical build scratch budget exceeded")

type hierarchicalNode struct {
	id           int
	vectorOffset int
	level        int
	links        [][]int32
	farthest     []farthestLink
}

type farthestLink struct {
	distance float32
	position int32
}

// hierarchicalNeighborCache is private build scratch. pgvector keeps the
// candidate distance and Algorithm 4 "closer" decision beside every HNSW
// connection, which lets reciprocal updates re-evaluate only candidates
// affected by the newly added edge. LumenVec keeps the same information only
// while constructing a private generation and drops it before publication so
// search-time adjacency remains compact.
type hierarchicalNeighborCache struct {
	layers []hierarchicalNeighborLayerCache
}

type hierarchicalNeighborLayerCache struct {
	distances []float32
	closer    []bool
	order     []int
	closerSet bool
}

type diversifiedCandidate struct {
	id       int
	distance float64
	closer   bool
	position int
	selected bool
}

// ConnectivityStats describes the graph topology without mutating it. It is
// intended for post-build diagnostics because recall failures at scale can be
// caused by disconnected components even when every edge is locally valid.
type ConnectivityStats struct {
	Nodes             int
	Reachable         int
	Unreachable       int
	Components        int
	WeakComponents    int
	Edges             int
	ReverseEdges      int
	MaxDegreeLayer0   int
	AverageDegreeZero float64
}

type HierarchicalLayerStats struct {
	Layer         int
	Nodes         int
	Edges         int
	AverageDegree float64
}

// HierarchicalNavigationStats measures the quality of the entrypoint handed
// to layer zero after upper-layer descent. It is an offline O(Q*N*D)
// diagnostic and is never used by the production search path.
type HierarchicalNavigationStats struct {
	Queries               int
	AverageEntrypointRank float64
	AverageDistanceRatio  float64
	BestRank              int
	WorstRank             int
}

// HierarchicalBuildStats records private construction memory independently
// from steady ANN memory. NeighborCacheUpperBoundBytes is a deterministic,
// architecture-specific upper bound for the cache released before publication.
type HierarchicalBuildStats struct {
	NeighborCacheUpperBoundBytes uint64
	ScratchBudgetBytes           uint64
}

// HierarchicalIndex is an experimental in-memory HNSW backend. Unlike
// AnnIndex, it stores a probabilistic hierarchy and uses upper-layer greedy
// descent before the bounded level-zero search. Service integration remains
// opt-in; recovery rebuilds it from canonical vectors rather than interpreting
// a compatibility ANN checkpoint as a hierarchical graph.
type HierarchicalIndex struct {
	structuralLayerBytes         uint64
	structuralEdgeBytes          uint64
	mu                           sync.RWMutex
	nodes                        []hierarchicalNode
	vectors                      []float32
	capacityHint                 int
	idToSlot                     map[int]int
	deleted                      []bool
	deletedCount                 int
	dim                          int
	m                            int
	m0                           int
	efConstruction               int
	efSearch                     int
	metric                       string
	diversifiedExistingPruning   bool
	entrypoint                   int
	entrypointPool               []int
	entrypointPoolSize           int
	maxLevel                     int
	hasEntrypoint                bool
	levelMult                    float64
	rnd                          *rand.Rand
	workspacePool                sync.Pool
	selectedScratch              []int
	rejectedScratch              []int
	reciprocalCandidateScratch   []distancePair
	reciprocalSelectedScratch    []int
	reciprocalRejectedScratch    []int
	reciprocalAddedScratch       []int
	reciprocalDiversifiedScratch []diversifiedCandidate
	neighborCache                []hierarchicalNeighborCache
	buildCacheActive             bool
	buildStats                   HierarchicalBuildStats
	buildScratchBudgetBytes      uint64
	upperLayerEF                 int
	maxConfiguredLevel           int
	searchQueries                atomic.Uint64
	searchSegments               atomic.Uint64
	searchEFBudget               atomic.Uint64
	disableDiversifiedCache      bool // test-only reference path
}

// ExactDistances reports that the index keeps the canonical float32 payload
// used by the graph search. Callers can therefore consume the returned
// distances directly instead of fetching and reranking candidates from the
// vector store.
func (h *HierarchicalIndex) ExactDistances() bool { return true }

// SetEfSearch changes the default query budget for subsequent searches. It
// does not rebuild the graph, so benchmark sweeps can compare recall and
// latency on the exact same index generation.
func (h *HierarchicalIndex) SetEfSearch(ef int) {
	if ef <= 0 {
		return
	}
	h.mu.Lock()
	h.efSearch = ef
	h.mu.Unlock()
}

func NewHierarchicalIndex(opts Options) *HierarchicalIndex {
	return NewHierarchicalIndexWithCapacity(opts, 0)
}

func NewHierarchicalIndexWithCapacity(opts Options, capacity int) *HierarchicalIndex {
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
	index := &HierarchicalIndex{
		nodes:                      make([]hierarchicalNode, 0, capacityOrZero(capacity)),
		capacityHint:               capacityOrZero(capacity),
		idToSlot:                   make(map[int]int, capacityOrZero(capacity)),
		deleted:                    make([]bool, 0, capacityOrZero(capacity)),
		m:                          opts.M,
		m0:                         opts.M * 2,
		efConstruction:             max(opts.EfConstruction, opts.M),
		efSearch:                   opts.EfSearch,
		metric:                     normalizeMetric(opts.Metric),
		diversifiedExistingPruning: opts.DiversifiedExistingPruning,
		buildScratchBudgetBytes:    opts.HierarchicalBuildScratchBudgetBytes,
		upperLayerEF:               max(1, opts.HierarchicalUpperLayerEF),
		maxConfiguredLevel:         opts.HierarchicalMaxLevel,
		entrypointPoolSize:         max(0, opts.HierarchicalEntrypointPoolSize),
		maxLevel:                   -1,
		levelMult:                  1 / math.Log(float64(opts.M)),
		rnd:                        rand.New(rand.NewSource(opts.Seed)), // #nosec G404 -- deterministic index construction
	}
	if opts.HierarchicalLevelMultiplier > 0 {
		index.levelMult = opts.HierarchicalLevelMultiplier
	}
	if index.maxConfiguredLevel <= 0 || index.maxConfiguredLevel > hierarchicalMaxLevel {
		index.maxConfiguredLevel = hierarchicalMaxLevel
	}
	index.workspacePool.New = func() any { return &searchWorkspace{} }
	return index
}

// BuildHierarchicalIndex constructs a complete private generation from a
// batch and returns it only after every vector has been indexed. Callers can
// validate the returned generation and publish it atomically, avoiding
// partially-built graphs being visible to readers.
func BuildHierarchicalIndex(opts Options, vectors []BatchVector32) (*HierarchicalIndex, error) {
	return BuildHierarchicalIndexFromSource(opts, len(vectors), func(i int) (BatchVector32, error) { return vectors[i], nil })
}

// BuildHierarchicalIndexFromSource copies one vector at a time into the private
// arena. The source must provide a stable, deterministic order for the duration
// of the build; returned payloads may be borrowed until the next source call.
func BuildHierarchicalIndexFromSource(opts Options, count int, source func(int) (BatchVector32, error)) (*HierarchicalIndex, error) {
	if count < 0 || source == nil {
		return nil, fmt.Errorf("invalid bulk source")
	}
	idx := NewHierarchicalIndexWithCapacity(opts, count)
	if err := idx.buildBulkSource(count, source); err != nil {
		return nil, err
	}
	if err := idx.Validate(); err != nil {
		return nil, err
	}
	idx.prepareEntrypointPool()
	return idx, nil
}

func (h *HierarchicalIndex) prepareEntrypointPool() {
	if h.entrypointPoolSize <= 1 || h.maxLevel <= 0 {
		return
	}
	level := h.maxLevel - 1
	for slot := range h.nodes {
		if !h.deleted[slot] && h.nodes[slot].level >= level {
			h.entrypointPool = append(h.entrypointPool, slot)
		}
	}
	if len(h.entrypointPool) > h.entrypointPoolSize {
		h.entrypointPool = h.entrypointPool[:h.entrypointPoolSize]
	}
}

// buildBulk constructs a generation in two phases. The first phase owns and
// validates all payload storage and assigns deterministic levels. The second
// phase builds the graph privately with the configured reciprocal pruning.
// No caller can observe the half-built generation.
func (h *HierarchicalIndex) buildBulk(vectors []BatchVector32) error {
	return h.buildBulkSource(len(vectors), func(i int) (BatchVector32, error) { return vectors[i], nil })
}

func (h *HierarchicalIndex) buildBulkSource(count int, source func(int) (BatchVector32, error)) error {
	if count == 0 {
		return nil
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.nodes) != 0 {
		return fmt.Errorf("bulk build requires an empty index")
	}
	first, err := source(0)
	if err != nil {
		return err
	}
	dim := len(first.Values)
	if dim == 0 {
		return ErrInvalidVectorDim
	}
	if count > int(^uint(0)>>1)/dim {
		return fmt.Errorf("bulk vector arena exceeds address space")
	}
	h.dim = dim
	h.vectors = make([]float32, 0, count*dim)
	h.nodes = make([]hierarchicalNode, 0, count)
	h.deleted = make([]bool, count)
	for position := 0; position < count; position++ {
		vector := first
		if position > 0 {
			vector, err = source(position)
			if err != nil {
				return err
			}
		}
		if len(vector.Values) != dim {
			return ErrInvalidVectorDim
		}
		if _, duplicate := h.idToSlot[vector.ID]; duplicate {
			return fmt.Errorf("duplicate vector id %d", vector.ID)
		}
		level := h.randomLevelLocked()
		links := make([][]int32, level+1)
		farthest := make([]farthestLink, level+1)
		for layer := range links {
			links[layer] = make([]int32, 0, h.layerLimit(layer))
			farthest[layer].position = -1
		}
		slot := len(h.nodes)
		offset := len(h.vectors)
		h.vectors = append(h.vectors, vector.Values...)
		h.nodes = append(h.nodes, hierarchicalNode{id: vector.ID, vectorOffset: offset, level: level, links: links, farthest: farthest})
		h.structuralLayerBytes += uint64(len(links))*24 + uint64(len(farthest))*8
		h.idToSlot[vector.ID] = slot
	}
	workspace := h.workspacePool.Get().(*searchWorkspace)
	defer h.workspacePool.Put(workspace)
	if h.diversifiedExistingPruning {
		neighborCacheBound, err := h.neighborCacheUpperBoundLocked()
		if err != nil {
			return err
		}
		h.buildStats = HierarchicalBuildStats{
			NeighborCacheUpperBoundBytes: neighborCacheBound,
			ScratchBudgetBytes:           h.buildScratchBudgetBytes,
		}
		if h.buildScratchBudgetBytes > 0 && neighborCacheBound > h.buildScratchBudgetBytes {
			return fmt.Errorf("%w: requires at most %d bytes, budget is %d bytes",
				ErrHierarchicalBuildScratchBudget, neighborCacheBound, h.buildScratchBudgetBytes)
		}
		h.neighborCache = make([]hierarchicalNeighborCache, len(h.nodes))
		h.buildCacheActive = true
	}
	for slot := range h.nodes {
		h.insertPreparedSlotLocked(slot, workspace)
	}
	// The cache changes only build complexity, not the published graph. Free it
	// before validation so MemoryStats reports steady
	// ANN memory rather than transient construction scratch.
	h.neighborCache = nil
	h.buildCacheActive = false
	return nil
}

// BuildStats returns construction diagnostics retained after private scratch
// has been released. It does not include the published graph's steady memory;
// callers should combine it with MemoryStats when reporting both figures.
func (h *HierarchicalIndex) BuildStats() HierarchicalBuildStats {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.buildStats
}

// neighborCacheUpperBoundLocked is O(N * L), where L is the number of levels
// assigned to a node (bounded by hierarchicalMaxLevel). It runs before the
// O(N * efConstruction * D) graph build and allocates no cache payload.
func (h *HierarchicalIndex) neighborCacheUpperBoundLocked() (uint64, error) {
	bytes := uint64(len(h.nodes)) * uint64(unsafe.Sizeof(hierarchicalNeighborCache{}))
	layerHeaderBytes := uint64(unsafe.Sizeof(hierarchicalNeighborLayerCache{}))
	perSlotBytes := uint64(unsafe.Sizeof(float32(0))) +
		uint64(unsafe.Sizeof(false)) + uint64(unsafe.Sizeof(int(0)))
	for nodeIndex := range h.nodes {
		layers := h.nodes[nodeIndex].level + 1
		addition := uint64(layers) * layerHeaderBytes
		for layer := 0; layer < layers; layer++ {
			addition += uint64(h.layerLimit(layer)) * perSlotBytes
		}
		if ^uint64(0)-bytes < addition {
			return 0, fmt.Errorf("hierarchical neighbor cache size overflows uint64")
		}
		bytes += addition
	}
	return bytes, nil
}

func (h *HierarchicalIndex) insertPreparedSlotLocked(slot int, workspace *searchWorkspace) {
	level := h.nodes[slot].level
	if !h.hasEntrypoint {
		h.entrypoint, h.maxLevel, h.hasEntrypoint = slot, level, true
		return
	}
	entry := h.entrypoint
	query := h.vectorLocked(slot)
	for layer := h.maxLevel; layer > level; layer-- {
		entry = h.upperLayerClosestLocked(query, entry, layer, min(h.efConstruction, h.upperLayerEF), workspace)
	}
	for layer := min(level, h.maxLevel); layer >= 0; layer-- {
		candidates := h.searchLayerLocked(query, entry, h.efConstruction, layer, workspace)
		// connectLocked may prune an existing endpoint. Keep this iteration
		// independent from every pruning scratch buffer.
		selected := append([]int(nil), h.selectNeighborsLocked(slot, candidates, layer, h.layerLimit(layer))...)
		for _, neighbor := range selected {
			h.connectLocked(slot, neighbor, layer)
		}
		for _, candidate := range candidates {
			if len(h.nodes[slot].links[layer]) >= h.layerLimit(layer) {
				break
			}
			if candidate.id != slot && !containsLink(h.nodes[slot].links[layer], candidate.id) {
				h.connectLocked(slot, candidate.id, layer)
			}
		}
		if len(candidates) > 0 {
			entry = candidates[0].id
		}
	}
	if level > h.maxLevel {
		h.entrypoint, h.maxLevel = slot, level
	}
}

func (h *HierarchicalIndex) AddVector(id int, vector []float64) error {
	if len(vector) == 0 {
		return ErrInvalidVectorDim
	}
	return h.AddVector32(id, vectorutil.ToFloat32(vector))
}

func (h *HierarchicalIndex) AddVector32(id int, vector []float32) error {
	if len(vector) == 0 {
		return ErrInvalidVectorDim
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.dim == 0 {
		h.dim = len(vector)
		maxInt := int(^uint(0) >> 1)
		if h.capacityHint > 0 && h.capacityHint <= maxInt/h.dim {
			h.vectors = make([]float32, 0, h.capacityHint*h.dim)
		}
	}
	if len(vector) != h.dim {
		return ErrInvalidVectorDim
	}
	if slot, exists := h.idToSlot[id]; exists {
		copy(h.vectorLocked(slot), vector)
		h.invalidateDistanceCachesLocked(slot)
		if h.deleted[slot] {
			h.deleted[slot] = false
			h.deletedCount--
		}
		return nil
	}

	level := h.randomLevelLocked()
	slot := len(h.nodes)
	vectorOffset := len(h.vectors)
	h.vectors = append(h.vectors, vector...)
	links := make([][]int32, level+1)
	farthest := make([]farthestLink, level+1)
	for layer := range links {
		links[layer] = make([]int32, 0, h.layerLimit(layer))
		farthest[layer].position = -1
	}
	h.nodes = append(h.nodes, hierarchicalNode{id: id, vectorOffset: vectorOffset, level: level, links: links, farthest: farthest})
	h.structuralLayerBytes += uint64(len(links))*24 + uint64(len(farthest))*8
	h.idToSlot[id] = slot
	h.deleted = append(h.deleted, false)
	if !h.hasEntrypoint {
		h.entrypoint = slot
		h.maxLevel = level
		h.hasEntrypoint = true
		return nil
	}

	entry := h.entrypoint
	query := h.vectorLocked(slot)
	for layer := h.maxLevel; layer > level; layer-- {
		entry = h.greedyClosestLocked(query, entry, layer)
	}

	workspace := h.workspacePool.Get().(*searchWorkspace)
	defer h.workspacePool.Put(workspace)
	for layer := min(level, h.maxLevel); layer >= 0; layer-- {
		candidates := h.searchLayerLocked(query, entry, h.efConstruction, layer, workspace)
		selected := append([]int(nil), h.selectNeighborsLocked(slot, candidates, layer, h.layerLimit(layer))...)
		for _, neighbor := range selected {
			h.connectLocked(slot, neighbor, layer)
		}
		// If reciprocal pruning rejected some selected edges on the neighbor,
		// use the next closest candidates as bounded fallbacks. This keeps the
		// new node connected without forcing an over-capacity edge.
		for _, candidate := range candidates {
			if len(h.nodes[slot].links[layer]) >= h.layerLimit(layer) {
				break
			}
			if candidate.id == slot || containsLink(h.nodes[slot].links[layer], candidate.id) {
				continue
			}
			h.connectLocked(slot, candidate.id, layer)
		}
		if len(candidates) > 0 {
			entry = candidates[0].id
		}
	}
	if level > h.maxLevel {
		h.entrypoint = slot
		h.maxLevel = level
	}
	return nil
}

// AddBatch preserves insertion order so builds remain reproducible for a
// fixed seed. A future parallel builder may replace this path, but publishing
// a partially built graph is deliberately outside this implementation.
func (h *HierarchicalIndex) AddBatch(vectors []BatchVector) error {
	for _, vector := range vectors {
		if err := h.AddVector(vector.ID, vector.Values); err != nil {
			return err
		}
	}
	return nil
}

// AddBatch32 avoids widening the canonical storage representation during bulk
// rebuilds. AddVector32 copies every payload into the index-owned arena.
func (h *HierarchicalIndex) AddBatch32(vectors []BatchVector32) error {
	for _, vector := range vectors {
		if err := h.AddVector32(vector.ID, vector.Values); err != nil {
			return err
		}
	}
	return nil
}

func (h *HierarchicalIndex) invalidateDistanceCachesLocked(slot int) {
	for layer, links := range h.nodes[slot].links {
		h.nodes[slot].farthest[layer].position = -1
		for _, rawNeighbor := range links {
			neighbor := int(rawNeighbor)
			if layer <= h.nodes[neighbor].level {
				h.nodes[neighbor].farthest[layer].position = -1
			}
		}
	}
}

func (h *HierarchicalIndex) Search(query []float64, k int) ([]int, error) {
	results, err := h.SearchWithDistances(query, k)
	if err != nil {
		return nil, err
	}
	ids := make([]int, len(results))
	for index := range results {
		ids[index] = results[index].ID
	}
	return ids, nil
}

func (h *HierarchicalIndex) SearchWithDistances(query []float64, k int) ([]Result, error) {
	return h.SearchWithDistancesInto(query, k, nil)
}

func (h *HierarchicalIndex) SearchWithDistancesInto(query []float64, k int, dst []Result) ([]Result, error) {
	h.mu.RLock()
	ef := h.efSearch
	h.mu.RUnlock()
	return h.SearchWithDistancesEfInto(query, k, ef, dst)
}

func (h *HierarchicalIndex) SearchWithDistancesEfInto(query []float64, k, ef int, dst []Result) ([]Result, error) {
	return h.searchWithContext(context.Background(), query, k, ef, dst)
}

func (h *HierarchicalIndex) SearchWithDistancesContext(ctx context.Context, query []float64, k int, dst []Result) ([]Result, error) {
	h.mu.RLock()
	ef := h.efSearch
	h.mu.RUnlock()
	return h.searchWithContext(ctx, query, k, ef, dst)
}

func (h *HierarchicalIndex) searchWithContext(ctx context.Context, query []float64, k, ef int, dst []Result) ([]Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, ErrInvalidK
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	if len(query) == 0 || len(query) != h.dim {
		return nil, ErrInvalidVectorDim
	}
	if !h.hasEntrypoint || len(h.nodes) == 0 {
		return dst[:0], nil
	}
	if ef < k {
		ef = k
	}
	h.searchQueries.Add(1)
	h.searchSegments.Add(1)
	h.searchEFBudget.Add(uint64(ef))
	workspace := h.workspacePool.Get().(*searchWorkspace)
	workspace.queryContext = ctx
	defer func() { workspace.queryContext = nil; h.workspacePool.Put(workspace) }()
	query32 := workspace.query32From64(query)
	entry := h.entrypoint
	startLayer := h.maxLevel
	if len(h.entrypointPool) > 0 {
		bestDistance := math.Inf(1)
		candidates := h.entrypointPool
		for _, candidate := range candidates {
			distance := h.distanceToNodeLocked(query32, candidate)
			if distance < bestDistance {
				entry, bestDistance = candidate, distance
				startLayer = h.nodes[candidate].level
			}
		}
		if bestDistance == math.Inf(1) {
			startLayer = h.maxLevel
		}
	}
	for layer := startLayer; layer > 0; layer-- {
		entry = h.upperLayerClosestLocked(query32, entry, layer, min(ef, h.upperLayerEF), workspace)
	}
	var candidates []distancePair
	if h.deletedCount == 0 {
		candidates = h.searchLayerLocked(query32, entry, ef, 0, workspace)
	} else {
		candidates = h.searchLiveLayerLocked(query32, entry, ef, workspace)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out := dst[:0]
	if cap(out) < min(k, len(candidates)) {
		out = make([]Result, 0, min(k, len(candidates)))
	}
	for _, candidate := range candidates {
		if h.deleted[candidate.id] {
			continue
		}
		out = append(out, Result{ID: h.nodes[candidate.id].id, Distance: candidate.distance})
		if len(out) == k {
			break
		}
	}
	return out, nil
}

func (h *HierarchicalIndex) SearchBudgetState() (queries, segments, efBudget uint64) {
	return h.searchQueries.Load(), h.searchSegments.Load(), h.searchEFBudget.Load()
}

func (h *HierarchicalIndex) DeleteVector(id int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	slot, exists := h.idToSlot[id]
	if !exists || h.deleted[slot] {
		return
	}
	h.deleted[slot] = true
	h.deletedCount++
}

func (h *HierarchicalIndex) Stats() Stats {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return Stats{Nodes: len(h.nodes), Deleted: h.deletedCount}
}

func (h *HierarchicalIndex) MaxLevel() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.maxLevel
}

func (h *HierarchicalIndex) MemoryStats() MemoryStats {
	h.mu.RLock()
	defer h.mu.RUnlock()
	vectorBytes := uint64(len(h.vectors)) * 4
	// Logical structural estimate, not RSS or allocation capacity. Updated
	// under the writer lock so metrics reads remain O(1).
	adjacencyBytes, nodeBytes := h.structuralEdgeBytes, h.structuralLayerBytes
	mapBytes := uint64(len(h.idToSlot)) * 16
	return MemoryStats{
		VectorBytes: vectorBytes, NodeBytes: nodeBytes, AdjacencyBytes: adjacencyBytes,
		MapBytes: mapBytes, TotalBytes: vectorBytes + nodeBytes + adjacencyBytes + mapBytes,
	}
}

// Validate checks the invariants required before publishing a generation.
// It is intentionally O(N+E), so callers should run it after bulk builds and
// not on every write.
func (h *HierarchicalIndex) Validate() error {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if len(h.nodes) == 0 {
		return nil
	}
	if !h.hasEntrypoint || h.entrypoint < 0 || h.entrypoint >= len(h.nodes) {
		return fmt.Errorf("invalid entrypoint %d", h.entrypoint)
	}
	if h.nodes[h.entrypoint].level != h.maxLevel {
		return fmt.Errorf("entrypoint level %d differs from max level %d", h.nodes[h.entrypoint].level, h.maxLevel)
	}
	for owner, node := range h.nodes {
		for layer, links := range node.links {
			if len(links) > h.layerLimit(layer) {
				return fmt.Errorf("node %d layer %d degree %d exceeds limit %d", owner, layer, len(links), h.layerLimit(layer))
			}
			seen := make(map[int]struct{}, len(links))
			for _, rawNeighbor := range links {
				neighbor := int(rawNeighbor)
				if neighbor < 0 || neighbor >= len(h.nodes) {
					return fmt.Errorf("node %d layer %d points to invalid node %d", owner, layer, neighbor)
				}
				if neighbor == owner {
					return fmt.Errorf("node %d layer %d points to itself", owner, layer)
				}
				if _, duplicate := seen[neighbor]; duplicate {
					return fmt.Errorf("node %d layer %d contains duplicate neighbor %d", owner, layer, neighbor)
				}
				seen[neighbor] = struct{}{}
			}
		}
	}
	return nil
}

// Connectivity returns O(N+E) topology metrics for the current generation.
func (h *HierarchicalIndex) Connectivity() ConnectivityStats {
	h.mu.RLock()
	defer h.mu.RUnlock()
	stats := ConnectivityStats{Nodes: len(h.nodes)}
	if len(h.nodes) == 0 {
		return stats
	}
	for owner, node := range h.nodes {
		stats.Edges += len(node.links[0])
		if len(node.links[0]) > stats.MaxDegreeLayer0 {
			stats.MaxDegreeLayer0 = len(node.links[0])
		}
		for _, rawNeighbor := range node.links[0] {
			if containsLink(h.nodes[int(rawNeighbor)].links[0], owner) {
				stats.ReverseEdges++
			}
		}
	}
	stats.AverageDegreeZero = float64(stats.Edges) / float64(len(h.nodes))
	visited := make([]bool, len(h.nodes))
	queue := []int{h.entrypoint}
	visited[h.entrypoint] = true
	for len(queue) > 0 {
		owner := queue[0]
		queue = queue[1:]
		stats.Reachable++
		for _, rawNeighbor := range h.nodes[owner].links[0] {
			neighbor := int(rawNeighbor)
			if !visited[neighbor] && !h.deleted[neighbor] {
				visited[neighbor] = true
				queue = append(queue, neighbor)
			}
		}
	}
	// Count weakly connected components separately. A directed HNSW graph may
	// contain stale reverse edges, so this metric must traverse both outgoing
	// and incoming adjacency rather than reusing the entrypoint reachability
	// walk above.
	reverse := make([][]int, len(h.nodes))
	for owner, node := range h.nodes {
		for _, rawNeighbor := range node.links[0] {
			neighbor := int(rawNeighbor)
			reverse[neighbor] = append(reverse[neighbor], owner)
		}
	}
	visited = make([]bool, len(h.nodes))
	for start := range h.nodes {
		if visited[start] || h.deleted[start] {
			continue
		}
		stats.WeakComponents++
		queue = []int{start}
		visited[start] = true
		for len(queue) > 0 {
			owner := queue[0]
			queue = queue[1:]
			for _, rawNeighbor := range h.nodes[owner].links[0] {
				neighbor := int(rawNeighbor)
				if !visited[neighbor] && !h.deleted[neighbor] {
					visited[neighbor] = true
					queue = append(queue, neighbor)
				}
			}
			for _, neighbor := range reverse[owner] {
				if !visited[neighbor] && !h.deleted[neighbor] {
					visited[neighbor] = true
					queue = append(queue, neighbor)
				}
			}
		}
	}
	stats.Components = stats.WeakComponents
	stats.Unreachable = stats.Nodes - stats.Reachable - h.deletedCount
	return stats
}

func (h *HierarchicalIndex) LayerStats() []HierarchicalLayerStats {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.maxLevel < 0 {
		return nil
	}
	out := make([]HierarchicalLayerStats, h.maxLevel+1)
	for layer := range out {
		out[layer].Layer = layer
	}
	for slot := range h.nodes {
		for layer := 0; layer <= h.nodes[slot].level; layer++ {
			out[layer].Nodes++
			out[layer].Edges += len(h.nodes[slot].links[layer])
		}
	}
	for layer := range out {
		if out[layer].Nodes > 0 {
			out[layer].AverageDegree = float64(out[layer].Edges) / float64(out[layer].Nodes)
		}
	}
	return out
}

func (h *HierarchicalIndex) NavigationStats(queries [][]float64) (HierarchicalNavigationStats, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	out := HierarchicalNavigationStats{BestRank: int(^uint(0) >> 1)}
	if !h.hasEntrypoint {
		out.BestRank = 0
		return out, nil
	}
	workspace := h.workspacePool.Get().(*searchWorkspace)
	defer h.workspacePool.Put(workspace)
	for _, query := range queries {
		if len(query) != h.dim {
			return HierarchicalNavigationStats{}, ErrInvalidVectorDim
		}
		query32 := workspace.query32From64(query)
		entry := h.entrypoint
		for layer := h.maxLevel; layer > 0; layer-- {
			entry = h.greedyClosestLocked(query32, entry, layer)
		}
		entryDistance := h.distanceToNodeLocked(query32, entry)
		bestDistance := math.Inf(1)
		rank := 1
		for slot := range h.nodes {
			distance := h.distanceToNodeLocked(query32, slot)
			if distance < bestDistance {
				bestDistance = distance
			}
			if distance < entryDistance {
				rank++
			}
		}
		ratio := 1.0
		if bestDistance > 0 {
			ratio = entryDistance / bestDistance
		} else if entryDistance > 0 {
			ratio = math.Inf(1)
		}
		out.Queries++
		out.AverageEntrypointRank += float64(rank)
		out.AverageDistanceRatio += ratio
		out.BestRank = min(out.BestRank, rank)
		out.WorstRank = max(out.WorstRank, rank)
	}
	if out.Queries > 0 {
		out.AverageEntrypointRank /= float64(out.Queries)
		out.AverageDistanceRatio /= float64(out.Queries)
	} else {
		out.BestRank = 0
	}
	return out, nil
}

func containsLink(links []int32, target int) bool {
	for _, raw := range links {
		if int(raw) == target {
			return true
		}
	}
	return false
}

func (h *HierarchicalIndex) randomLevelLocked() int {
	value := h.rnd.Float64()
	if value <= 0 {
		return hierarchicalMaxLevel
	}
	return min(int(-math.Log(value)*h.levelMult), h.maxConfiguredLevel)
}

func (h *HierarchicalIndex) layerLimit(layer int) int {
	if layer == 0 {
		return h.m0
	}
	return h.m
}

func (h *HierarchicalIndex) distanceToNodeLocked(query []float32, slot int) float64 {
	return metricDistance32(h.metric, query, h.vectorLocked(slot))
}

func (h *HierarchicalIndex) distanceBetweenNodesLocked(left, right int) float64 {
	return metricDistance32(h.metric, h.vectorLocked(left), h.vectorLocked(right))
}

func (h *HierarchicalIndex) vectorLocked(slot int) []float32 {
	offset := h.nodes[slot].vectorOffset
	return h.vectors[offset : offset+h.dim]
}

func (h *HierarchicalIndex) greedyClosestLocked(query []float32, entry, layer int) int {
	best := entry
	bestDistance := h.distanceToNodeLocked(query, best)
	for {
		changed := false
		if layer <= h.nodes[best].level {
			for _, rawNeighbor := range h.nodes[best].links[layer] {
				neighbor := int(rawNeighbor)
				distance := h.distanceToNodeLocked(query, neighbor)
				if distance < bestDistance {
					best = neighbor
					bestDistance = distance
					changed = true
				}
			}
		}
		if !changed {
			return best
		}
	}
}

func (h *HierarchicalIndex) upperLayerClosestLocked(query []float32, entry, layer, ef int, workspace *searchWorkspace) int {
	if ef <= 1 {
		return h.greedyClosestLocked(query, entry, layer)
	}
	candidates := h.searchLayerLocked(query, entry, ef, layer, workspace)
	if len(candidates) == 0 {
		return entry
	}
	return candidates[0].id
}

// searchLiveLayerLocked preserves deleted nodes as navigation bridges, but
// only live nodes consume the result budget. The no-deletion SIMD path remains
// unchanged. Visited marks bound exploration even across deleted cycles.
func (h *HierarchicalIndex) searchLiveLayerLocked(query []float32, entry, ef int, workspace *searchWorkspace) []distancePair {
	workspace.reset(ef, len(h.nodes))
	first := distancePair{id: entry, distance: h.distanceToNodeLocked(query, entry)}
	workspace.minQ.Push(first)
	if !h.deleted[entry] {
		workspace.maxQ.Push(first)
	}
	workspace.markVisited(entry)
	iterations := 0
	for workspace.minQ.Len() > 0 {
		iterations++
		if iterations&63 == 0 && workspace.queryContext != nil && workspace.queryContext.Err() != nil {
			break
		}
		current := workspace.minQ.Pop()
		if workspace.maxQ.Len() >= ef && current.distance > workspace.maxQ.Peek().distance {
			break
		}
		for _, raw := range h.nodes[current.id].links[0] {
			id := int(raw)
			if !workspace.visit(id) {
				continue
			}
			candidate := distancePair{id: id, distance: h.distanceToNodeLocked(query, id)}
			if workspace.maxQ.Len() < ef || candidate.distance < workspace.maxQ.Peek().distance {
				workspace.minQ.Push(candidate)
				if !h.deleted[id] {
					if workspace.maxQ.Len() < ef {
						workspace.maxQ.Push(candidate)
					} else {
						workspace.maxQ.ReplaceTop(candidate)
					}
				}
			}
		}
	}
	workspace.minQ = workspace.minQ[:0]
	return workspace.drainCandidates(workspace.maxQ)
}

func (h *HierarchicalIndex) searchLayerLocked(query []float32, entry, ef, layer int, workspace *searchWorkspace) []distancePair {
	workspace.reset(ef, len(h.nodes))
	distance := h.distanceToNodeLocked(query, entry)
	workspace.minQ.Push(distancePair{id: entry, distance: distance})
	workspace.maxQ.Push(distancePair{id: entry, distance: distance})
	workspace.markVisited(entry)
	iterations := 0
	for workspace.minQ.Len() > 0 {
		iterations++
		if iterations&63 == 0 && workspace.queryContext != nil && workspace.queryContext.Err() != nil {
			break
		}
		current := workspace.minQ.Pop()
		if workspace.maxQ.Len() >= ef && current.distance > workspace.maxQ.Peek().distance {
			break
		}
		if layer > h.nodes[current.id].level {
			continue
		}
		links := h.nodes[current.id].links[layer]
		if h.metric == MetricL2 {
			var pending [4]int
			pendingCount := 0
			for _, rawNeighbor := range links {
				neighbor := int(rawNeighbor)
				if !workspace.visit(neighbor) {
					continue
				}
				pending[pendingCount] = neighbor
				pendingCount++
				if pendingCount == len(pending) {
					distance0, distance1, distance2, distance3 := vectorutil.SquaredEuclideanDistance32x4ArenaFastSameLen(
						h.vectors,
						h.nodes[pending[0]].vectorOffset, h.nodes[pending[1]].vectorOffset,
						h.nodes[pending[2]].vectorOffset, h.nodes[pending[3]].vectorOffset,
						h.dim, query,
					)
					h.admitSearchCandidateLocked(workspace, pending[0], distance0, ef)
					h.admitSearchCandidateLocked(workspace, pending[1], distance1, ef)
					h.admitSearchCandidateLocked(workspace, pending[2], distance2, ef)
					h.admitSearchCandidateLocked(workspace, pending[3], distance3, ef)
					pendingCount = 0
				}
			}
			for position := 0; position < pendingCount; position++ {
				neighbor := pending[position]
				h.admitSearchCandidateLocked(workspace, neighbor, h.distanceToNodeLocked(query, neighbor), ef)
			}
			continue
		}
		for _, rawNeighbor := range links {
			neighbor := int(rawNeighbor)
			if !workspace.visit(neighbor) {
				continue
			}
			distance := h.distanceToNodeLocked(query, neighbor)
			h.admitSearchCandidateLocked(workspace, neighbor, distance, ef)
		}
	}
	workspace.minQ = workspace.minQ[:0]
	return workspace.drainCandidates(workspace.maxQ)
}

func (h *HierarchicalIndex) admitSearchCandidateLocked(workspace *searchWorkspace, id int, distance float64, ef int) {
	candidate := distancePair{id: id, distance: distance}
	if workspace.maxQ.Len() < ef {
		workspace.minQ.Push(candidate)
		workspace.maxQ.Push(candidate)
	} else if distance < workspace.maxQ.Peek().distance {
		workspace.minQ.Push(candidate)
		workspace.maxQ.ReplaceTop(candidate)
	}
}

func (h *HierarchicalIndex) selectNeighborsLocked(owner int, candidates []distancePair, layer, limit int) []int {
	selectedCapacity := min(limit, len(candidates))
	if cap(h.selectedScratch) < selectedCapacity {
		h.selectedScratch = make([]int, 0, selectedCapacity)
	}
	if cap(h.rejectedScratch) < len(candidates) {
		h.rejectedScratch = make([]int, 0, len(candidates))
	}
	selected := h.selectedScratch[:0]
	rejected := h.rejectedScratch[:0]
	passes := 2
	if h.diversifiedExistingPruning {
		passes = 1
	}
	for pass := 0; pass < passes && len(selected) < limit; pass++ {
		for _, candidate := range candidates {
			if candidate.id == owner || containsInt(selected, candidate.id) {
				continue
			}
			if !h.diversifiedExistingPruning && pass == 0 && len(h.nodes[candidate.id].links[layer]) >= h.layerLimit(layer) {
				continue
			}
			diverse := h.isDiverseCandidateLocked(candidate.id, candidate.distance, selected)
			if diverse && len(selected) < limit {
				selected = append(selected, candidate.id)
			} else {
				rejected = append(rejected, candidate.id)
			}
			if len(selected) == limit {
				h.selectedScratch = selected
				h.rejectedScratch = rejected
				return selected
			}
		}
	}
	// Keep pruned connections as a bounded fallback. This follows the HNSW
	// paper's optional connectivity safeguard and prevents sparse early layers.
	for _, candidate := range rejected {
		selected = append(selected, candidate)
		if len(selected) == limit {
			break
		}
	}
	h.selectedScratch = selected
	h.rejectedScratch = rejected
	return selected
}

// isDiverseCandidateLocked implements the HNSW diversity predicate. L2 uses
// the same bit-consistent four-way kernel as graph traversal, reducing the
// constant factor while preserving the scalar topology exactly.
func (h *HierarchicalIndex) isDiverseCandidateLocked(candidate int, candidateDistance float64, selected []int) bool {
	if h.metric != MetricL2 || len(selected) < 4 {
		for _, existing := range selected {
			if h.distanceBetweenNodesLocked(candidate, existing) < candidateDistance {
				return false
			}
		}
		return true
	}
	candidateVector := h.vectorLocked(candidate)
	position := 0
	for ; position+3 < len(selected); position += 4 {
		distance0, distance1, distance2, distance3 := vectorutil.SquaredEuclideanDistance32x4ArenaFastSameLen(
			h.vectors,
			h.nodes[selected[position]].vectorOffset, h.nodes[selected[position+1]].vectorOffset,
			h.nodes[selected[position+2]].vectorOffset, h.nodes[selected[position+3]].vectorOffset,
			h.dim, candidateVector,
		)
		if distance0 < candidateDistance || distance1 < candidateDistance ||
			distance2 < candidateDistance || distance3 < candidateDistance {
			return false
		}
	}
	for ; position < len(selected); position++ {
		if h.distanceBetweenNodesLocked(candidate, selected[position]) < candidateDistance {
			return false
		}
	}
	return true
}

func containsInt(values []int, value int) bool {
	for _, v := range values {
		if v == value {
			return true
		}
	}
	return false
}

func (h *HierarchicalIndex) connectLocked(left, right, layer int) {
	if left == right || layer > h.nodes[left].level || layer > h.nodes[right].level {
		return
	}
	distance := h.distanceBetweenNodesLocked(left, right)
	h.addLinkLocked(left, right, layer, distance)
	h.addLinkLocked(right, left, layer, distance)
}

func (h *HierarchicalIndex) addLinkLocked(owner, neighbor, layer int, distance float64) {
	links := h.nodes[owner].links[layer]
	for _, existing := range links {
		if int(existing) == neighbor {
			return
		}
	}
	limit := h.layerLimit(layer)
	if len(links) < limit {
		h.nodes[owner].links[layer] = append(links, int32(neighbor))
		h.structuralEdgeBytes += 4
		if cache := h.cachedNeighborLayerLocked(owner, layer); cache != nil {
			cache.distances = append(cache.distances, float32(distance))
			cache.closer = append(cache.closer, false)
			cache.closerSet = false
		}
		farthest := &h.nodes[owner].farthest[layer]
		if farthest.position < 0 || distance > float64(farthest.distance) {
			*farthest = farthestLink{position: int32(len(links)), distance: float32(distance)}
		}
		return
	}
	if h.diversifiedExistingPruning {
		h.tryDiversifiedReplacementLocked(owner, neighbor, layer, distance, limit)
		return
	}
	farthest := &h.nodes[owner].farthest[layer]
	if farthest.position < 0 {
		h.recomputeFarthestLocked(owner, layer)
	}
	if distance >= float64(farthest.distance) {
		return
	}
	h.nodes[owner].links[layer][int(farthest.position)] = int32(neighbor)
	h.recomputeFarthestLocked(owner, layer)
}

func (h *HierarchicalIndex) tryDiversifiedReplacementLocked(owner, neighbor, layer int, distance float64, limit int) {
	links := h.nodes[owner].links[layer]
	cache := h.ensureNeighborLayerCacheLocked(owner, layer)
	required := len(links) + 1
	if cap(h.reciprocalDiversifiedScratch) < required {
		h.reciprocalDiversifiedScratch = make([]diversifiedCandidate, required)
	}
	candidates := h.reciprocalDiversifiedScratch[:required]
	insertAt := sort.Search(len(cache.order), func(index int) bool {
		position := cache.order[index]
		existingDistance := float64(cache.distances[position])
		return existingDistance > distance || (existingDistance == distance && int(links[position]) >= neighbor)
	})
	for candidatePosition, linkPosition := range cache.order[:insertAt] {
		candidates[candidatePosition] = diversifiedCandidate{
			id: int(links[linkPosition]), distance: float64(cache.distances[linkPosition]),
			closer: cache.closer[linkPosition], position: linkPosition,
		}
	}
	candidates[insertAt] = diversifiedCandidate{id: neighbor, distance: distance, position: -1}
	for orderPosition, linkPosition := range cache.order[insertAt:] {
		candidates[insertAt+1+orderPosition] = diversifiedCandidate{
			id: int(links[linkPosition]), distance: float64(cache.distances[linkPosition]),
			closer: cache.closer[linkPosition], position: linkPosition,
		}
	}
	if cap(h.reciprocalSelectedScratch) < limit {
		h.reciprocalSelectedScratch = make([]int, 0, limit)
	}
	if cap(h.reciprocalRejectedScratch) < len(candidates) {
		h.reciprocalRejectedScratch = make([]int, 0, len(candidates))
	}
	selected := h.reciprocalSelectedScratch[:0]
	rejected := h.reciprocalRejectedScratch[:0]
	added := h.reciprocalAddedScratch[:0]
	mustCalculate := !cache.closerSet
	removedAny := false
	for position := range candidates {
		candidate := &candidates[position]
		switch {
		case mustCalculate:
			candidate.closer = h.isDiverseCandidateLocked(candidate.id, candidate.distance, selected)
		case candidate.position < 0:
			candidate.closer = h.isDiverseCandidateLocked(candidate.id, candidate.distance, selected)
			if candidate.closer {
				added = append(added, candidate.id)
			}
		case candidate.closer:
			if !h.isDiverseCandidateLocked(candidate.id, candidate.distance, added) {
				candidate.closer = false
				removedAny = true
			}
		case removedAny:
			candidate.closer = h.isDiverseCandidateLocked(candidate.id, candidate.distance, selected)
			if candidate.closer {
				added = append(added, candidate.id)
			}
		}
		if candidate.closer && len(selected) < limit {
			selected = append(selected, candidate.id)
			candidate.selected = true
		} else {
			rejected = append(rejected, position)
		}
	}
	for _, candidatePosition := range rejected {
		if len(selected) == limit {
			break
		}
		selected = append(selected, candidates[candidatePosition].id)
		candidates[candidatePosition].selected = true
	}
	for _, candidate := range candidates {
		if candidate.position >= 0 {
			cache.closer[candidate.position] = candidate.closer
		}
	}
	cache.closerSet = true
	newSelected := candidates[insertAt].selected
	if !newSelected {
		h.reciprocalDiversifiedScratch = candidates
		h.reciprocalSelectedScratch = selected
		h.reciprocalRejectedScratch = rejected
		h.reciprocalAddedScratch = added
		return
	}
	prunedPosition := -1
	for _, candidate := range candidates {
		if candidate.position >= 0 && !candidate.selected {
			prunedPosition = candidate.position
			break
		}
	}
	if prunedPosition < 0 {
		return
	}
	links[prunedPosition] = int32(neighbor)
	h.nodes[owner].links[layer] = links
	cache.distances[prunedPosition] = float32(distance)
	cache.closer[prunedPosition] = candidates[insertAt].closer
	orderPosition := 0
	for _, candidate := range candidates {
		switch {
		case candidate.position == prunedPosition:
			continue
		case candidate.position < 0:
			cache.order[orderPosition] = prunedPosition
		default:
			cache.order[orderPosition] = candidate.position
		}
		orderPosition++
	}
	h.recomputeFarthestLocked(owner, layer)
	h.reciprocalSelectedScratch = selected
	h.reciprocalRejectedScratch = rejected
	h.reciprocalAddedScratch = added
	h.reciprocalDiversifiedScratch = candidates
}

func (h *HierarchicalIndex) cachedNeighborLayerLocked(owner, layer int) *hierarchicalNeighborLayerCache {
	if owner < 0 || owner >= len(h.neighborCache) || layer < 0 || layer >= len(h.neighborCache[owner].layers) {
		return nil
	}
	cache := &h.neighborCache[owner].layers[layer]
	if len(cache.distances) != len(h.nodes[owner].links[layer]) || len(cache.closer) != len(h.nodes[owner].links[layer]) {
		return nil
	}
	return cache
}

func (h *HierarchicalIndex) ensureNeighborLayerCacheLocked(owner, layer int) *hierarchicalNeighborLayerCache {
	if !h.buildCacheActive {
		// Online updates after publication must not resurrect build scratch.
		// Full pruning remains correct through an ephemeral one-list cache; the
		// optimized persistent state is reserved for a private bulk generation.
		links := h.nodes[owner].links[layer]
		cache := &hierarchicalNeighborLayerCache{
			distances: make([]float32, len(links)),
			closer:    make([]bool, len(links)),
			order:     make([]int, len(links)),
		}
		for position, rawNeighbor := range links {
			cache.distances[position] = float32(h.distanceBetweenNodesLocked(owner, int(rawNeighbor)))
			cache.order[position] = position
		}
		sort.Slice(cache.order, func(left, right int) bool {
			leftPosition, rightPosition := cache.order[left], cache.order[right]
			leftDistance, rightDistance := cache.distances[leftPosition], cache.distances[rightPosition]
			if leftDistance == rightDistance {
				return int(links[leftPosition]) < int(links[rightPosition])
			}
			return leftDistance < rightDistance
		})
		return cache
	}
	if len(h.neighborCache) < len(h.nodes) {
		h.neighborCache = append(h.neighborCache, make([]hierarchicalNeighborCache, len(h.nodes)-len(h.neighborCache))...)
	}
	nodeCache := &h.neighborCache[owner]
	if len(nodeCache.layers) == 0 {
		// Allocate the final shape once. Growing layer-by-layer retained obsolete
		// backing arrays until the next GC and made build memory less predictable.
		nodeCache.layers = make([]hierarchicalNeighborLayerCache, h.nodes[owner].level+1)
	}
	cache := &nodeCache.layers[layer]
	links := h.nodes[owner].links[layer]
	shapeChanged := len(cache.distances) != len(links) || len(cache.closer) != len(links)
	if shapeChanged {
		cache.distances = make([]float32, len(links))
		cache.closer = make([]bool, len(links))
	}
	if shapeChanged || !cache.closerSet || h.disableDiversifiedCache {
		for position, rawNeighbor := range links {
			cache.distances[position] = float32(h.distanceBetweenNodesLocked(owner, int(rawNeighbor)))
		}
		if h.disableDiversifiedCache {
			clear(cache.closer)
		}
		cache.closerSet = false
	}
	if len(cache.order) != len(links) || shapeChanged || h.disableDiversifiedCache {
		cache.order = make([]int, len(links))
		for position := range cache.order {
			cache.order[position] = position
		}
		sort.Slice(cache.order, func(left, right int) bool {
			leftPosition, rightPosition := cache.order[left], cache.order[right]
			leftDistance, rightDistance := cache.distances[leftPosition], cache.distances[rightPosition]
			if leftDistance == rightDistance {
				return int(links[leftPosition]) < int(links[rightPosition])
			}
			return leftDistance < rightDistance
		})
	}
	return cache
}

func (h *HierarchicalIndex) recomputeFarthestLocked(owner, layer int) {
	links := h.nodes[owner].links[layer]
	if len(links) == 0 {
		h.nodes[owner].farthest[layer] = farthestLink{position: -1}
		return
	}
	cache := h.cachedNeighborLayerLocked(owner, layer)
	distanceAt := func(position int) float64 {
		if cache != nil {
			return float64(cache.distances[position])
		}
		return h.distanceBetweenNodesLocked(owner, int(links[position]))
	}
	position := 0
	distance := distanceAt(0)
	for candidatePosition := 1; candidatePosition < len(links); candidatePosition++ {
		candidateDistance := distanceAt(candidatePosition)
		if candidateDistance > distance {
			position = candidatePosition
			distance = candidateDistance
		}
	}
	h.nodes[owner].farthest[layer] = farthestLink{position: int32(position), distance: float32(distance)}
}
