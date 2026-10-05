package ann

import (
	"errors"
	"math"
	"math/rand"
	"slices"
	"sort"
	"testing"
	"time"
	"unsafe"
)

func TestHierarchicalBuildScratchBudget(t *testing.T) {
	vectors := make([]BatchVector32, 256)
	for id := range vectors {
		vectors[id] = BatchVector32{ID: id, Values: []float32{float32(id), float32(id % 17), float32(id % 7), 1}}
	}
	opts := Options{
		M: 8, EfConstruction: 32, EfSearch: 32, Seed: 42,
		DiversifiedExistingPruning:          true,
		HierarchicalBuildScratchBudgetBytes: 1,
	}
	if _, err := BuildHierarchicalIndex(opts, vectors); !errors.Is(err, ErrHierarchicalBuildScratchBudget) {
		t.Fatalf("BuildHierarchicalIndex error = %v, want scratch budget error", err)
	}

	opts.HierarchicalBuildScratchBudgetBytes = 1 << 20
	index, err := BuildHierarchicalIndex(opts, vectors)
	if err != nil {
		t.Fatalf("BuildHierarchicalIndex with sufficient budget: %v", err)
	}
	stats := index.BuildStats()
	if stats.NeighborCacheUpperBoundBytes == 0 {
		t.Fatal("neighbor-cache upper bound was not recorded")
	}
	if stats.NeighborCacheUpperBoundBytes > stats.ScratchBudgetBytes {
		t.Fatalf("upper bound %d exceeds accepted budget %d", stats.NeighborCacheUpperBoundBytes, stats.ScratchBudgetBytes)
	}
	if index.neighborCache != nil {
		t.Fatal("published index retained private neighbor cache")
	}
}

func TestHierarchicalNavigationDiagnostics10K(t *testing.T) {
	rng := rand.New(rand.NewSource(91))
	vectors := make([]BatchVector32, 10000)
	for id := range vectors {
		values := make([]float32, 32)
		for dim := range values {
			values[dim] = rng.Float32()
		}
		vectors[id] = BatchVector32{ID: id, Values: values}
	}
	index, err := BuildHierarchicalIndex(Options{M: 20, EfConstruction: 128, EfSearch: 256, Seed: 42}, vectors)
	if err != nil {
		t.Fatal(err)
	}
	queries := make([][]float64, 20)
	for query := range queries {
		queries[query] = make([]float64, 32)
		for dim := range queries[query] {
			queries[query][dim] = rng.Float64()
		}
	}
	navigation, err := index.NavigationStats(queries)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("layers=%+v navigation=%+v", index.LayerStats(), navigation)
}

func TestHierarchicalDenseLayerDistribution(t *testing.T) {
	vectors := make([]BatchVector32, 10000)
	for id := range vectors {
		vectors[id] = BatchVector32{ID: id, Values: []float32{float32(id % 97), float32(id % 31), float32(id % 13), 1}}
	}
	index, err := BuildHierarchicalIndex(Options{M: 20, EfConstruction: 128, EfSearch: 256, Seed: 42, HierarchicalLevelMultiplier: 1 / math.Log(16), HierarchicalMaxLevel: 5}, vectors)
	if err != nil {
		t.Fatal(err)
	}
	queries := make([][]float64, 20)
	for i := range queries {
		queries[i] = make([]float64, len(vectors[i].Values))
		for d, value := range vectors[i].Values {
			queries[i][d] = float64(value) + 0.001
		}
	}
	navigation, err := index.NavigationStats(queries)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("dense layers=%+v navigation=%+v", index.LayerStats(), navigation)
}

func TestHierarchicalDenseRecallAndLatency10K(t *testing.T) {
	const n, dims, queryCount, k = 10000, 16, 60, 10
	rng := rand.New(rand.NewSource(177))
	vectors := make([]BatchVector32, n)
	for id := range vectors {
		values := make([]float32, dims)
		for d := range values {
			values[d] = rng.Float32()
		}
		vectors[id] = BatchVector32{ID: id, Values: values}
	}
	queries := make([][]float64, queryCount)
	for i := range queries {
		queries[i] = make([]float64, dims)
		for d, v := range vectors[i].Values {
			queries[i][d] = float64(v) + (rng.Float64()-0.5)*0.01
		}
	}
	exact := make([][]int, queryCount)
	for qi, q := range queries {
		pairs := make([]struct {
			id int
			d  float64
		}, n)
		for id, v := range vectors {
			var d float64
			for j, x := range v.Values {
				delta := float64(x) - q[j]
				d += delta * delta
			}
			pairs[id] = struct {
				id int
				d  float64
			}{id, d}
		}
		sort.Slice(pairs, func(i, j int) bool { return pairs[i].d < pairs[j].d })
		exact[qi] = make([]int, k)
		for i := 0; i < k; i++ {
			exact[qi][i] = pairs[i].id
		}
	}
	configs := []struct {
		name string
		opts Options
	}{
		{"default", Options{M: 20, EfConstruction: 128, EfSearch: 256, Seed: 42}},
		{"dense", Options{M: 20, EfConstruction: 128, EfSearch: 256, Seed: 42, HierarchicalLevelMultiplier: 1 / math.Log(16), HierarchicalMaxLevel: 5}},
	}
	for _, cfg := range configs {
		index, err := BuildHierarchicalIndex(cfg.opts, vectors)
		if err != nil {
			t.Fatal(err)
		}
		latencies := make([]time.Duration, 0, queryCount)
		hits := 0
		for qi, q := range queries {
			start := time.Now()
			results, err := index.SearchWithDistancesEfInto(q, k, 256, nil)
			latencies = append(latencies, time.Since(start))
			if err != nil {
				t.Fatal(err)
			}
			seen := make(map[int]struct{}, len(results))
			for _, r := range results {
				seen[r.ID] = struct{}{}
			}
			for _, id := range exact[qi] {
				if _, ok := seen[id]; ok {
					hits++
				}
			}
		}
		sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
		p95 := latencies[(len(latencies)*95)/100]
		t.Logf("%s layers=%+v recall=%.2f%% p95=%s avg=%s", cfg.name, index.LayerStats(), 100*float64(hits)/float64(queryCount*k), p95, time.Duration(sumDurations(latencies)/int64(len(latencies))))
	}
}

func sumDurations(values []time.Duration) int64 {
	var total int64
	for _, v := range values {
		total += int64(v)
	}
	return total
}

func TestHierarchicalOnlineUpdateDoesNotRetainBuildCache(t *testing.T) {
	index, err := BuildHierarchicalIndex(Options{
		M: 4, EfConstruction: 16, EfSearch: 16, Seed: 42,
		DiversifiedExistingPruning: true,
	}, []BatchVector32{
		{ID: 1, Values: []float32{0, 0}},
		{ID: 2, Values: []float32{1, 0}},
		{ID: 3, Values: []float32{0, 1}},
		{ID: 4, Values: []float32{1, 1}},
		{ID: 5, Values: []float32{2, 1}},
		{ID: 6, Values: []float32{1, 2}},
		{ID: 7, Values: []float32{2, 2}},
		{ID: 8, Values: []float32{3, 2}},
		{ID: 9, Values: []float32{2, 3}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := index.AddVector32(10, []float32{3, 3}); err != nil {
		t.Fatal(err)
	}
	if index.neighborCache != nil || index.buildCacheActive {
		t.Fatal("online update retained private build cache")
	}
	if err := index.Validate(); err != nil {
		t.Fatalf("online update produced invalid graph: %v", err)
	}
}

func TestHierarchicalDiversifiedCachePreservesReferenceTopology(t *testing.T) {
	opts := Options{M: 8, EfConstruction: 32, EfSearch: 64, Seed: 42, DiversifiedExistingPruning: true}
	cached := NewHierarchicalIndex(opts)
	reference := NewHierarchicalIndex(opts)
	reference.disableDiversifiedCache = true
	rng := rand.New(rand.NewSource(42)) // #nosec G404 -- deterministic topology fixture
	for id := 0; id < 600; id++ {
		values := make([]float32, 24)
		for dimension := range values {
			values[dimension] = rng.Float32()
		}
		if err := cached.AddVector32(id, values); err != nil {
			t.Fatal(err)
		}
		if err := reference.AddVector32(id, values); err != nil {
			t.Fatal(err)
		}
	}
	if cached.entrypoint != reference.entrypoint || cached.maxLevel != reference.maxLevel {
		t.Fatalf("entry state differs: cached=(%d,%d) reference=(%d,%d)", cached.entrypoint, cached.maxLevel, reference.entrypoint, reference.maxLevel)
	}
	for owner := range cached.nodes {
		if len(cached.nodes[owner].links) != len(reference.nodes[owner].links) {
			t.Fatalf("node %d layer count differs", owner)
		}
		for layer := range cached.nodes[owner].links {
			if !slices.Equal(cached.nodes[owner].links[layer], reference.nodes[owner].links[layer]) {
				t.Fatalf("node %d layer %d differs: cached=%v reference=%v", owner, layer, cached.nodes[owner].links[layer], reference.nodes[owner].links[layer])
			}
		}
	}
}

func TestHierarchicalIndexUsesCompactFarthestCache(t *testing.T) {
	if size := unsafe.Sizeof(farthestLink{}); size != 8 {
		t.Fatalf("farthest cache grew to %d bytes, want 8", size)
	}
}

func TestHierarchicalIndexVectorArenaReusesStorageOnUpdate(t *testing.T) {
	index := NewHierarchicalIndexWithCapacity(Options{M: 4, EfConstruction: 16, EfSearch: 16, Seed: 3}, 4)
	if err := index.AddVector32(7, []float32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	originalLength := len(index.vectors)
	originalOffset := index.nodes[index.idToSlot[7]].vectorOffset
	if err := index.AddVector32(7, []float32{4, 5, 6}); err != nil {
		t.Fatal(err)
	}
	if len(index.vectors) != originalLength {
		t.Fatalf("update appended to vector arena: length %d, want %d", len(index.vectors), originalLength)
	}
	if offset := index.nodes[index.idToSlot[7]].vectorOffset; offset != originalOffset {
		t.Fatalf("update moved vector from offset %d to %d", originalOffset, offset)
	}
	if vector := index.vectorLocked(index.idToSlot[7]); vector[0] != 4 || vector[1] != 5 || vector[2] != 6 {
		t.Fatalf("arena update was not applied: %v", vector)
	}
}

func TestHierarchicalIndexBuildsMultipleLayers(t *testing.T) {
	index := NewHierarchicalIndex(Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 42})
	for id := 0; id < 2000; id++ {
		if err := index.AddVector(id, []float64{float64(id), float64(id % 17)}); err != nil {
			t.Fatal(err)
		}
	}
	if index.MaxLevel() < 2 {
		t.Fatalf("expected a hierarchy with at least three layers, max level=%d", index.MaxLevel())
	}
	if stats := index.Stats(); stats.Nodes != 2000 || stats.Deleted != 0 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}

func TestHierarchicalIndexSearchDeleteAndRevive(t *testing.T) {
	index := NewHierarchicalIndex(Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 7})
	for id := 0; id < 100; id++ {
		if err := index.AddVector(id, []float64{float64(id), 0}); err != nil {
			t.Fatal(err)
		}
	}
	results, err := index.SearchWithDistances([]float64{42.1, 0}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 3 || results[0].ID != 42 {
		t.Fatalf("unexpected nearest results: %+v", results)
	}
	index.DeleteVector(42)
	results, err = index.SearchWithDistances([]float64{42.1, 0}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 || results[0].ID == 42 {
		t.Fatalf("deleted vector returned: %+v", results)
	}
	if err := index.AddVector(42, []float64{42, 0}); err != nil {
		t.Fatal(err)
	}
	results, err = index.SearchWithDistances([]float64{42.1, 0}, 3)
	if err != nil || len(results) == 0 || results[0].ID != 42 {
		t.Fatalf("revived vector missing: results=%+v err=%v", results, err)
	}
}

func TestHierarchicalIndexUpdateInvalidatesDistanceCaches(t *testing.T) {
	index := NewHierarchicalIndex(Options{M: 4, EfConstruction: 16, EfSearch: 16, Seed: 19})
	for id := 0; id < 100; id++ {
		if err := index.AddVector(id, []float64{float64(id), 0}); err != nil {
			t.Fatal(err)
		}
	}
	if err := index.AddVector(50, []float64{1000, 0}); err != nil {
		t.Fatal(err)
	}
	index.mu.RLock()
	slot := index.idToSlot[50]
	for layer := range index.nodes[slot].farthest {
		if index.nodes[slot].farthest[layer].position >= 0 {
			index.mu.RUnlock()
			t.Fatalf("updated node retained a valid farthest cache at layer %d", layer)
		}
	}
	index.mu.RUnlock()
	if err := index.AddVector(100, []float64{100.5, 0}); err != nil {
		t.Fatal(err)
	}
	if stats := index.Stats(); stats.Nodes != 101 {
		t.Fatalf("unexpected stats after update and insert: %+v", stats)
	}
}

func TestHierarchicalIndexRecallGate(t *testing.T) {
	const vectors, dimensions, queries, topK = 3000, 32, 40, 10
	rng := rand.New(rand.NewSource(91))
	data := make([][]float64, vectors)
	index := NewHierarchicalIndex(Options{M: 16, EfConstruction: 96, EfSearch: 128, Seed: 91})
	for id := range data {
		data[id] = make([]float64, dimensions)
		for dimension := range data[id] {
			data[id][dimension] = rng.Float64()
		}
		if err := index.AddVector(id, data[id]); err != nil {
			t.Fatal(err)
		}
	}
	matched := 0
	for queryIndex := 0; queryIndex < queries; queryIndex++ {
		query := make([]float64, dimensions)
		for dimension := range query {
			query[dimension] = rng.Float64()
		}
		truth := exactTopKForTest(data, query, topK)
		results, err := index.SearchWithDistancesEfInto(query, topK, 128, nil)
		if err != nil {
			t.Fatal(err)
		}
		truthSet := make(map[int]struct{}, topK)
		for _, id := range truth {
			truthSet[id] = struct{}{}
		}
		for _, result := range results {
			if _, ok := truthSet[result.ID]; ok {
				matched++
			}
		}
	}
	recall := float64(matched) / float64(queries*topK)
	if recall < 0.98 {
		t.Fatalf("Recall@10=%.4f, want >=0.98", recall)
	}
}

func TestHierarchicalIndexValidatesStructuralGraph(t *testing.T) {
	index := NewHierarchicalIndex(Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 42})
	rng := rand.New(rand.NewSource(42))
	for id := 0; id < 200; id++ {
		values := make([]float64, 16)
		for dimension := range values {
			values[dimension] = rng.Float64()
		}
		if err := index.AddVector(id, values); err != nil {
			t.Fatal(err)
		}
	}
	if err := index.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	stats := index.Connectivity()
	if stats.Nodes != 200 || stats.Reachable != 200 || stats.Unreachable != 0 || stats.WeakComponents != 1 {
		t.Fatalf("unexpected connectivity stats: %+v", stats)
	}
}

func exactTopKForTest(data [][]float64, query []float64, k int) []int {
	candidates := make([]distancePair, len(data))
	for id, vector := range data {
		var distance float64
		for dimension, value := range vector {
			delta := value - query[dimension]
			distance += delta * delta
		}
		candidates[id] = distancePair{id: id, distance: distance}
	}
	selectNearest(candidates, k)
	ids := make([]int, k)
	for index := 0; index < k; index++ {
		ids[index] = candidates[index].id
	}
	return ids
}

func TestBuildHierarchicalIndexPublishesValidatedGeneration(t *testing.T) {
	vectors := make([]BatchVector32, 32)
	for i := range vectors {
		vectors[i] = BatchVector32{ID: i, Values: []float32{float32(i), float32(i % 3)}}
	}
	idx, err := BuildHierarchicalIndex(Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 42}, vectors)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if err := idx.Validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	got, err := idx.SearchWithDistances([]float64{10, 1}, 3)
	if err != nil || len(got) != 3 {
		t.Fatalf("search result: %v (%d)", err, len(got))
	}
}

func TestBuildHierarchicalIndexRejectsInvalidSnapshot(t *testing.T) {
	_, err := BuildHierarchicalIndex(Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 42}, []BatchVector32{
		{ID: 7, Values: []float32{1, 2}},
		{ID: 7, Values: []float32{3, 4}},
	})
	if err == nil {
		t.Fatal("duplicate snapshot ID was accepted")
	}
	_, err = BuildHierarchicalIndex(Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 42}, []BatchVector32{
		{ID: 1, Values: []float32{1, 2}},
		{ID: 2, Values: []float32{3}},
	})
	if err != ErrInvalidVectorDim {
		t.Fatalf("dimension error = %v, want %v", err, ErrInvalidVectorDim)
	}
}

func TestBuildHierarchicalIndexBulkWeakConnectivityAndRecall(t *testing.T) {
	const vectors, dimensions, queries, topK = 3000, 32, 40, 10
	rng := rand.New(rand.NewSource(91))
	data := make([][]float64, vectors)
	batch := make([]BatchVector32, vectors)
	for id := range data {
		data[id] = make([]float64, dimensions)
		values := make([]float32, dimensions)
		for dimension := range data[id] {
			data[id][dimension] = rng.Float64()
			values[dimension] = float32(data[id][dimension])
		}
		batch[id] = BatchVector32{ID: id, Values: values}
	}
	index, err := BuildHierarchicalIndex(Options{M: 16, EfConstruction: 96, EfSearch: 128, Seed: 91}, batch)
	if err != nil {
		t.Fatal(err)
	}
	connectivity := index.Connectivity()
	if connectivity.WeakComponents != 1 || connectivity.Unreachable > vectors/100 {
		t.Fatalf("bulk graph connectivity is outside its safety bound: %+v", connectivity)
	}
	matched := 0
	for queryIndex := 0; queryIndex < queries; queryIndex++ {
		query := make([]float64, dimensions)
		for dimension := range query {
			query[dimension] = rng.Float64()
		}
		truth := exactTopKForTest(data, query, topK)
		results, searchErr := index.SearchWithDistancesEfInto(query, topK, 128, nil)
		if searchErr != nil {
			t.Fatal(searchErr)
		}
		truthSet := make(map[int]struct{}, topK)
		for _, id := range truth {
			truthSet[id] = struct{}{}
		}
		for _, result := range results {
			if _, ok := truthSet[result.ID]; ok {
				matched++
			}
		}
	}
	recall := float64(matched) / float64(queries*topK)
	if recall < 0.98 {
		t.Fatalf("bulk Recall@10=%.4f, want >=0.98; connectivity=%+v", recall, connectivity)
	}
}
