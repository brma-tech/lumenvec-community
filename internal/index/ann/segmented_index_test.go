package ann

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sort"
	"sync"
	"testing"
	"time"
)

func TestSegmentedIndexDoesNotMergeBeyondSegmentBudget(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 16, EfSearch: 16}, 4)
	t.Cleanup(func() { _ = idx.Close() })
	for batch := 0; batch < defaultCompactThreshold; batch++ {
		vectors := make([]BatchVector, 0, 4)
		for item := 0; item < 4; item++ {
			id := batch*4 + item
			vectors = append(vectors, BatchVector{ID: id, Values: []float64{float64(id), 0}})
		}
		if err := idx.AddBatch(vectors); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := idx.WaitForMaintenance(ctx); err != nil {
		t.Fatal(err)
	}
	if got := idx.SegmentCount(); got != defaultCompactThreshold {
		t.Fatalf("segments = %d, want %d bounded immutable segments", got, defaultCompactThreshold)
	}
	if stats := idx.Stats(); stats.Nodes != 32 || stats.Deleted != 0 {
		t.Fatalf("stats = %+v", stats)
	}
	results, err := idx.SearchWithDistancesInto([]float64{31.1, 0}, 1, nil)
	if err != nil || len(results) != 1 || results[0].ID != 31 {
		t.Fatalf("results=%+v err=%v", results, err)
	}
}

func TestSegmentedIndexSnapshotExportKeepsSealedAndTailOwnershipDistinct(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 16, EfSearch: 16}, 2)
	t.Cleanup(func() { _ = idx.Close() })
	if err := idx.AddBatch([]BatchVector{{ID: 1, Values: []float64{1, 0}}, {ID: 2, Values: []float64{2, 0}}, {ID: 3, Values: []float64{3, 0}}}); err != nil {
		t.Fatal(err)
	}
	snapshot := idx.SnapshotExport()
	seen := make(map[int]struct{})
	for _, segment := range snapshot.Segments {
		if err := segment.VisitLiveVectors32(func(id int, _ []float32) error { seen[id] = struct{}{}; return nil }); err != nil {
			t.Fatal(err)
		}
	}
	for _, vector := range snapshot.Tail {
		if _, duplicate := seen[vector.ID]; duplicate {
			t.Fatalf("vector %d appears in both sealed and tail views", vector.ID)
		}
		seen[vector.ID] = struct{}{}
	}
	if len(seen) != 3 {
		t.Fatalf("snapshot ownership cardinality = %d, want 3", len(seen))
	}
}

func TestSegmentedIndexBuildAndPublishBatch32PublishesImmutableGeneration(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 8, EfSearch: 8}, 2)
	defer idx.Close()
	if err := idx.BuildAndPublishBatch32([]BatchVector32{
		{ID: 101, Values: []float32{1, 0}},
		{ID: 102, Values: []float32{0, 1}},
		{ID: 103, Values: []float32{1, 1}},
	}); err != nil {
		t.Fatal(err)
	}
	if got := idx.SegmentCount(); got != 2 {
		t.Fatalf("segment count = %d, want 2", got)
	}
	results, err := idx.SearchWithDistancesInto([]float64{1, 0}, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("search results = %d, want 2", len(results))
	}
}

func TestSegmentedIndexBuildAndPublishBatchPublishesFloat64ImmutableGeneration(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 8, EfSearch: 8}, 2)
	defer idx.Close()
	if err := idx.BuildAndPublishBatch([]BatchVector{
		{ID: 201, Values: []float64{1, 0}},
		{ID: 202, Values: []float64{0, 1}},
		{ID: 203, Values: []float64{1, 1}},
	}); err != nil {
		t.Fatal(err)
	}
	snapshot := idx.SnapshotExport()
	if got := len(snapshot.Segments); got != 2 {
		t.Fatalf("sealed segments = %d, want 2", got)
	}
	if got := len(snapshot.Tail); got != 0 {
		t.Fatalf("mutable tail = %d, want 0", got)
	}
	results, err := idx.SearchWithDistancesInto([]float64{1, 0}, 1, nil)
	if err != nil || len(results) != 1 || results[0].ID != 201 {
		t.Fatalf("results=%+v err=%v", results, err)
	}
}

func TestSegmentedIndexBuildAndPublishBatch32RejectsReplayDuplicate(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 8, EfSearch: 8}, 16)
	defer idx.Close()
	if err := idx.AddVector(7, []float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	if err := idx.BuildAndPublishBatch32([]BatchVector32{{ID: 7, Values: []float32{1, 0}}}); err == nil {
		t.Fatal("replayed ID was published")
	}
	if got := idx.SegmentCount(); got != 1 {
		t.Fatalf("segment count after rejected replay = %d, want existing delta only", got)
	}
}

func TestSegmentedIndexBuildAndPublishBatch32AllowsReplacementAfterDelete(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 8, EfSearch: 8}, 16)
	defer idx.Close()
	if err := idx.AddVector(7, []float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	idx.DeleteVector(7)
	if err := idx.BuildAndPublishBatch32([]BatchVector32{{ID: 7, Values: []float32{0, 1}}}); err != nil {
		t.Fatalf("replacement publication failed: %v", err)
	}
	results, err := idx.SearchWithDistancesInto([]float64{0, 1}, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].ID != 7 {
		t.Fatalf("replacement search = %#v, want ID 7", results)
	}
}

func TestSegmentedIndexCompactsSparseSegmentsWithinBudget(t *testing.T) {
	t.Setenv("LUMENVEC_COMPACTION_THRESHOLD", "2")
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 16, EfSearch: 16}, 4)
	idx.compactionBudget = time.Nanosecond
	t.Cleanup(func() { _ = idx.Close() })
	for id := 0; id < 8; id++ {
		if err := idx.AddVector(id, []float64{float64(id), 0}); err != nil {
			t.Fatal(err)
		}
	}
	for id := 0; id < 4; id++ {
		idx.DeleteVector(id)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := idx.WaitForMaintenance(ctx); err != nil {
		t.Fatal(err)
	}
	if got := idx.SegmentCount(); got != 1 {
		t.Fatalf("segments=%d, want sparse segments to consolidate", got)
	}
}

func TestSegmentedIndexCloseCancelsActiveCompaction(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 16, EfConstruction: 64, EfSearch: 64, Seed: 19}, 256)
	for i := 0; i < 768; i++ {
		if err := idx.AddVector(i, []float64{float64(i), float64(i % 17)}); err != nil {
			t.Fatal(err)
		}
	}
	idx.RequestConsolidation()
	deadline := time.Now().Add(2 * time.Second)
	for !idx.compacting.Load() && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	started := time.Now()
	if err := idx.Close(); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("close waited for active compaction: %s", elapsed)
	}
}

func TestSegmentedIndexStatsAfterClose(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 8, EfSearch: 8}, 2)
	if err := idx.AddBatch([]BatchVector{{ID: 1, Values: []float64{1, 0}}}); err != nil {
		t.Fatal(err)
	}
	if err := idx.Close(); err != nil {
		t.Fatal(err)
	}
	if got := idx.Stats(); got.Nodes != 0 || got.Deleted != 0 {
		t.Fatalf("stats after close = %+v, want zero", got)
	}
}

func TestSegmentedIndexCompactionThresholdIsConfigurable(t *testing.T) {
	t.Setenv("LUMENVEC_COMPACTION_THRESHOLD", "32")
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 8, EfSearch: 8}, 2)
	t.Cleanup(func() { _ = idx.Close() })
	for batch := 0; batch < 4; batch++ {
		if err := idx.AddBatch([]BatchVector{{ID: batch, Values: []float64{float64(batch), 0}}}); err != nil {
			t.Fatal(err)
		}
	}
	if got := idx.SegmentCount(); got != 2 {
		t.Fatalf("segments=%d, want deferred consolidation at threshold 32", got)
	}
}

func TestSegmentedIndexPublishesBatchAsImmutableSegments(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 16, EfSearch: 16}, 2)
	vectors := []BatchVector{
		{ID: 1, Values: []float64{1, 0}},
		{ID: 2, Values: []float64{2, 0}},
		{ID: 3, Values: []float64{3, 0}},
		{ID: 4, Values: []float64{4, 0}},
		{ID: 5, Values: []float64{5, 0}},
	}
	if err := idx.AddBatch(vectors); err != nil {
		t.Fatal(err)
	}
	if got := idx.SegmentCount(); got != 3 {
		t.Fatalf("segments = %d, want 3", got)
	}
	if stats := idx.Stats(); stats.Nodes != len(vectors) {
		t.Fatalf("stats = %+v", stats)
	}
	results, err := idx.SearchWithDistancesInto([]float64{5.1, 0}, 1, nil)
	if err != nil || len(results) != 1 || results[0].ID != 5 {
		t.Fatalf("results=%+v err=%v", results, err)
	}
	if err := idx.AddBatch(nil); err != nil {
		t.Fatal(err)
	}
}

func TestSegmentedIndexVisitsOnlySealedSegments(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 16, EfSearch: 16}, 2)
	t.Cleanup(func() { _ = idx.Close() })
	if err := idx.AddBatch32([]BatchVector32{{ID: 1, Values: []float32{1, 0}}, {ID: 2, Values: []float32{2, 0}}, {ID: 3, Values: []float32{3, 0}}}); err != nil {
		t.Fatal(err)
	}
	visited := 0
	if err := idx.VisitSealedSegments(func(_ int, segment *AnnIndex) error {
		visited++
		if segment.Stats().Nodes != 2 {
			t.Fatalf("sealed segment nodes=%d", segment.Stats().Nodes)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if visited != 1 {
		t.Fatalf("visited=%d want 1", visited)
	}
	tail := idx.SnapshotMutableVectors32()
	if len(tail) != 1 || tail[0].ID != 3 {
		t.Fatalf("mutable tail=%+v", tail)
	}
}

func TestSegmentedIndexAddBatch32PreservesSearch(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 16, EfSearch: 16}, 2)
	t.Cleanup(func() { _ = idx.Close() })
	vectors := []BatchVector32{
		{ID: 1, Values: []float32{1, 0}},
		{ID: 2, Values: []float32{2, 0}},
		{ID: 3, Values: []float32{3, 0}},
	}
	if err := idx.AddBatch32(vectors); err != nil {
		t.Fatal(err)
	}
	results, err := idx.SearchWithDistancesInto([]float64{2.1, 0}, 1, nil)
	if err != nil || len(results) != 1 || results[0].ID != 2 {
		t.Fatalf("results=%+v err=%v", results, err)
	}
}

func TestSegmentedIndexPublishesPrebuiltSnapshotAtomically(t *testing.T) {
	source := NewAnnIndexWithOptions(Options{M: 4, EfConstruction: 16, EfSearch: 16})
	for _, vector := range []BatchVector32{{ID: 41, Values: []float32{1, 0}}, {ID: 42, Values: []float32{2, 0}}} {
		if err := source.AddVector32(vector.ID, vector.Values); err != nil {
			t.Fatal(err)
		}
	}
	payload, err := source.MarshalCompactBinary()
	if err != nil {
		t.Fatal(err)
	}
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 16, EfSearch: 16}, 2)
	t.Cleanup(func() { _ = idx.Close() })
	if err := idx.PublishPrebuiltSnapshot(payload); err != nil {
		t.Fatal(err)
	}
	if got := idx.SegmentCount(); got != 1 {
		t.Fatalf("segments=%d, want 1", got)
	}
	results, err := idx.SearchWithDistancesInto([]float64{2.1, 0}, 1, nil)
	if err != nil || len(results) != 1 || results[0].ID != 42 {
		t.Fatalf("results=%+v err=%v", results, err)
	}
	if err := idx.PublishPrebuiltSnapshot([]byte("corrupt")); err == nil {
		t.Fatal("expected corrupt snapshot rejection")
	}
	if got := idx.SegmentCount(); got != 1 {
		t.Fatalf("failed publish changed segments=%d", got)
	}
	if err := idx.PublishPrebuiltSnapshot(payload); err == nil {
		t.Fatal("expected duplicate-ID snapshot rejection")
	}
	if got := idx.SegmentCount(); got != 1 {
		t.Fatalf("duplicate publish changed segments=%d", got)
	}
}

func TestSegmentedIndexPreservesConfiguredQualityProfile(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 32, EfConstruction: 128, EfSearch: 256}, 100)
	t.Cleanup(func() { _ = idx.Close() })
	options := idx.ingestOptions()
	if options.M != 32 || options.EfConstruction != 128 || options.EfSearch != 256 {
		t.Fatalf("ingest options = %+v", options)
	}
	if idx.options.M != 32 || idx.options.EfConstruction != 128 || idx.options.EfSearch != 256 {
		t.Fatalf("final options changed = %+v", idx.options)
	}
}

func TestSegmentedIndexMaintenanceControlPaths(t *testing.T) {
	idx := NewSegmentedIndex(Options{}, 2)
	if !idx.ExactDistances() || !idx.delta.ExactDistances() {
		t.Fatal("expected exact float32 ANN distances")
	}
	idx.compactGeneration() // A single empty delta needs no consolidation.
	for batch := 0; batch < 2; batch++ {
		if err := idx.AddBatch([]BatchVector{
			{ID: batch*2 + 1, Values: []float64{float64(batch), 0}},
			{ID: batch*2 + 2, Values: []float64{float64(batch) + 0.5, 0}},
		}); err != nil {
			t.Fatal(err)
		}
	}
	idx.DeleteVector(1)
	idx.DeleteVector(2)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := idx.WaitForMaintenance(ctx); err != nil && !errors.Is(err, context.Canceled) {
		t.Fatalf("maintenance error = %v", err)
	}
	if err := idx.Close(); err != nil {
		t.Fatal(err)
	}
	if err := idx.Close(); err != nil {
		t.Fatal(err)
	}
	idx.RequestConsolidation() // Closed indexes ignore new requests.
}

func TestSegmentedIndexRotatesAndMerges(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 42}, 2)
	for i := 0; i < 5; i++ {
		if err := idx.AddVector(i, []float64{float64(i), 0}); err != nil {
			t.Fatal(err)
		}
	}
	if got := idx.SegmentCount(); got != 3 {
		t.Fatalf("segments = %d, want 3", got)
	}
	results, err := idx.SearchWithDistancesInto([]float64{4.1, 0}, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 || results[0].ID != 4 {
		t.Fatalf("results = %+v", results)
	}
	idx.DeleteVector(4)
	results, err = idx.SearchWithDistancesInto([]float64{4.1, 0}, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].ID == 4 {
		t.Fatalf("results after delete = %+v", results)
	}
}

func TestSegmentedIndexConcurrentAddAndSearch(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 42}, 16)
	var wg sync.WaitGroup
	for worker := 0; worker < 4; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				id := worker*1000 + i
				if err := idx.AddVector(id, []float64{float64(id), 1}); err != nil {
					t.Errorf("add %s: %v", fmt.Sprint(id), err)
					return
				}
				_, _ = idx.SearchWithDistancesInto([]float64{float64(id), 1}, 5, nil)
			}
		}(worker)
	}
	wg.Wait()
	if stats := idx.Stats(); stats.Nodes != 200 {
		t.Fatalf("stats = %+v", stats)
	}
}

func TestMergeSortedTopKDeterministic(t *testing.T) {
	left := []Result{{ID: 3, Distance: 1}, {ID: 9, Distance: 3}}
	right := []Result{{ID: 2, Distance: 1}, {ID: 4, Distance: 2}}
	got := mergeSortedTopK(left, right, 3, nil)
	want := []Result{{ID: 2, Distance: 1}, {ID: 3, Distance: 1}, {ID: 4, Distance: 2}}
	if !slices.Equal(got, want) {
		t.Fatalf("merge = %+v, want %+v", got, want)
	}
}

func TestSegmentRouterUsesFullFanoutForSmallGenerations(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 8, EfSearch: 16, SegmentRouting: true}, 1)
	t.Cleanup(func() { _ = idx.Close() })
	vectors := make([]BatchVector, 12)
	for item := range vectors {
		values := make([]float64, routeSignatureBits)
		for dimension := range values {
			values[dimension] = -1
		}
		switch {
		case item < 7:
			values[item] = 1 // seven Hamming-1 neighbors
		case item < 11:
			values[item] = 1
			values[(item+5)%routeSignatureBits] = 1
			values[(item+11)%routeSignatureBits] = 1
		} // item 11 is the exact query signature
		vectors[item] = BatchVector{ID: item + 1, Values: values}
	}
	if err := idx.AddBatch(vectors); err != nil {
		t.Fatal(err)
	}

	routed := idx.routedIndexes(vectors[11].Values)
	if len(routed) != len(vectors)+1 { // immutable graphs + delta
		t.Fatalf("routed indexes = %d, want %d", len(routed), len(vectors)+1)
	}
	target := idx.segments[11]
	if !slices.Contains(routed, target) {
		t.Fatal("matching-signature segment was not routed")
	}
	results, err := idx.SearchWithDistancesInto(vectors[11].Values, 1, nil)
	if err != nil || len(results) != 1 || results[0].ID != 12 {
		t.Fatalf("results=%+v err=%v", results, err)
	}
}

func TestSegmentRouterFallsBackForDegenerateSignatures(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 8, EfSearch: 16, SegmentRouting: true}, 1)
	t.Cleanup(func() { _ = idx.Close() })
	for item := 0; item < 12; item++ {
		values := make([]float64, routeSignatureBits)
		for dimension := range values {
			values[dimension] = float64(item + dimension + 1)
		}
		if err := idx.AddVector(item+1, values); err != nil {
			t.Fatal(err)
		}
	}
	routed := idx.routedIndexes(make([]float64, routeSignatureBits))
	if len(routed) != idx.SegmentCount()+1 {
		t.Fatalf("degenerate routing selected %d indexes, want all %d", len(routed), idx.SegmentCount()+1)
	}
}

func TestInvertedSegmentRouterMatchesLegacySelection(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 8, EfSearch: 16, SegmentRouting: true}, 2)
	t.Cleanup(func() { _ = idx.Close() })
	idx.mu.Lock()
	idx.compactThreshold = 1000
	idx.mu.Unlock()

	vectors := make([]BatchVector, 24)
	for item := range vectors {
		values := make([]float64, routeSignatureBits)
		for dimension := range values {
			if ((item*17)+(dimension*7))%11 < 5 {
				values[dimension] = -1
			} else {
				values[dimension] = 1
			}
		}
		vectors[item] = BatchVector{ID: item + 1, Values: values}
	}
	if err := idx.AddBatch(vectors); err != nil {
		t.Fatal(err)
	}

	query := vectors[19].Values
	signature := routeSignature64(query)
	idx.mu.RLock()
	segments := append([]*AnnIndex(nil), idx.segments...)
	delta := idx.delta
	idx.mu.RUnlock()
	legacy := make([]routedSegment, len(segments))
	for position, segment := range segments {
		legacy[position] = routedSegment{index: segment, position: position, distance: segment.routeDistance(signature)}
	}
	sort.Slice(legacy, func(left, right int) bool {
		if legacy[left].distance == legacy[right].distance {
			return legacy[left].position < legacy[right].position
		}
		return legacy[left].distance < legacy[right].distance
	})
	limit := min(defaultRouteMinSegments, len(legacy))
	for limit < len(legacy) && legacy[limit].distance <= 1 {
		limit++
	}
	want := make([]*AnnIndex, 0, limit+1)
	for _, candidate := range legacy[:limit] {
		want = append(want, candidate.index)
	}
	want = append(want, delta)

	if got := idx.routedIndexes(query); len(got) != len(segments)+1 && !slices.Equal(got, want) {
		t.Fatalf("inverted routing must match confident legacy selection or safely fan out: got=%d legacy=%d all=%d",
			len(got), len(want), len(segments)+1)
	}
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if idx.routeEpoch != idx.segmentEpoch || len(idx.routeKeys) == 0 || len(idx.routeSegments) == 0 ||
		len(idx.routeOffsets) != len(idx.routeKeys)+1 {
		t.Fatalf("route directory was not published: route=%d segment=%d keys=%d offsets=%d postings=%d",
			idx.routeEpoch, idx.segmentEpoch, len(idx.routeKeys), len(idx.routeOffsets), len(idx.routeSegments))
	}
}

func TestInvertedSegmentRouterRecallOnUnclusteredData(t *testing.T) {
	const (
		vectorCount = 2000
		dimension   = 24
		queryCount  = 40
		topK        = 10
	)
	idx := NewSegmentedIndex(Options{M: 16, EfConstruction: 96, EfSearch: 128, SegmentRouting: true, Seed: 42}, 100)
	t.Cleanup(func() { _ = idx.Close() })
	idx.mu.Lock()
	idx.compactThreshold = 1000
	idx.mu.Unlock()

	random := rand.New(rand.NewSource(42))
	vectors := make([]BatchVector, vectorCount)
	for id := range vectors {
		values := make([]float64, dimension)
		var norm float64
		for d := range values {
			values[d] = random.NormFloat64()
			norm += values[d] * values[d]
		}
		norm = math.Sqrt(norm)
		for d := range values {
			values[d] /= norm
		}
		vectors[id] = BatchVector{ID: id, Values: values}
	}
	if err := idx.AddBatch(vectors); err != nil {
		t.Fatal(err)
	}

	recallSum := 0.0
	for queryIndex := 0; queryIndex < queryCount; queryIndex++ {
		query := append([]float64(nil), vectors[(queryIndex*47)%vectorCount].Values...)
		for d := range query {
			query[d] += random.NormFloat64() * 0.002
		}
		truth := make([]Result, vectorCount)
		for position, vector := range vectors {
			truth[position] = Result{ID: vector.ID, Distance: l2(query, vector.Values)}
		}
		sort.Slice(truth, func(left, right int) bool {
			if truth[left].Distance == truth[right].Distance {
				return truth[left].ID < truth[right].ID
			}
			return truth[left].Distance < truth[right].Distance
		})
		hits, err := idx.SearchWithDistancesInto(query, topK, nil)
		if err != nil {
			t.Fatal(err)
		}
		expected := make(map[int]struct{}, topK)
		for _, result := range truth[:topK] {
			expected[result.ID] = struct{}{}
		}
		matched := 0
		for _, result := range hits {
			if _, ok := expected[result.ID]; ok {
				matched++
			}
		}
		recallSum += float64(matched) / topK
	}
	recall := recallSum / queryCount
	if recall < 0.95 {
		t.Fatalf("unclustered Recall@10=%.4f, want >=0.95", recall)
	}
	t.Logf("unclustered Recall@10=%.4f", recall)
}

func TestSegmentRouterIsOptIn(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 8, EfSearch: 16}, 1)
	t.Cleanup(func() { _ = idx.Close() })
	query := make([]float64, routeSignatureBits)
	for dimension := range query {
		query[dimension] = -1
	}
	const segmentCount = defaultRouteFullFanoutSegments + 4
	for item := 0; item < segmentCount; item++ {
		values := append([]float64(nil), query...)
		if item < 8 {
			values[item] = 1
		} else {
			values[item%routeSignatureBits] = 1
			values[(item+5)%routeSignatureBits] = 1
			values[(item+11)%routeSignatureBits] = 1
		}
		if err := idx.AddVector(item+1, values); err != nil {
			t.Fatal(err)
		}
	}
	if routed := idx.routedIndexes(query); len(routed) != segmentCount+1 {
		t.Fatalf("disabled routing selected %d indexes, want all %d", len(routed), segmentCount+1)
	}
	idx.SetSegmentRouting(true)
	if routed := idx.routedIndexes(query); len(routed) >= segmentCount+1 {
		t.Fatalf("enabled routing did not reduce fanout: %d", len(routed))
	}
}

func benchmarkRouteIndex(b *testing.B) (*SegmentedIndex, []float64) {
	b.Helper()
	idx := NewSegmentedIndex(Options{SegmentRouting: true}, 10000)
	idx.mu.Lock()
	idx.segments = make([]*AnnIndex, 256)
	for segmentIndex := range idx.segments {
		signatures := make([]uint32, 64)
		for signatureIndex := range signatures {
			// Deterministic mixing produces shared and unique postings without
			// constructing HNSW graphs in benchmark setup.
			signatures[signatureIndex] = uint32((segmentIndex*2654435761 + signatureIndex*2246822519) & ((1 << routeSignatureBits) - 1))
		}
		sort.Slice(signatures, func(left, right int) bool { return signatures[left] < signatures[right] })
		idx.segments[segmentIndex] = &AnnIndex{routeSignatures: signatures}
	}
	idx.segmentEpoch++
	idx.mu.Unlock()
	query := make([]float64, routeSignatureBits)
	for dimension := range query {
		if dimension%3 == 0 {
			query[dimension] = -1
		} else {
			query[dimension] = 1
		}
	}
	b.Cleanup(func() { _ = idx.Close() })
	return idx, query
}

func BenchmarkSegmentRoutingInverted256x64(b *testing.B) {
	idx, query := benchmarkRouteIndex(b)
	_ = idx.routedIndexes(query) // build the generation-scoped directory
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		_ = idx.routedIndexes(query)
	}
}

func BenchmarkSegmentRoutingLegacy256x64(b *testing.B) {
	idx, query := benchmarkRouteIndex(b)
	signature := routeSignature64(query)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		idx.mu.RLock()
		segments := append([]*AnnIndex(nil), idx.segments...)
		idx.mu.RUnlock()
		routed := make([]routedSegment, len(segments))
		for position, segment := range segments {
			routed[position] = routedSegment{index: segment, position: position, distance: segment.routeDistance(signature)}
		}
		sort.Slice(routed, func(left, right int) bool {
			if routed[left].distance == routed[right].distance {
				return routed[left].position < routed[right].position
			}
			return routed[left].distance < routed[right].distance
		})
	}
}

func benchmarkDenseRouteIndex(b *testing.B) (*SegmentedIndex, []float64) {
	b.Helper()
	idx := NewSegmentedIndex(Options{SegmentRouting: true}, 10000)
	query := make([]float64, routeSignatureBits)
	for dimension := range query {
		query[dimension] = -1
	}
	querySignature := routeSignature64(query)
	idx.mu.Lock()
	idx.segments = make([]*AnnIndex, 256)
	for segmentIndex := range idx.segments {
		signatures := make([]uint32, 1024)
		for signatureIndex := range signatures {
			signatures[signatureIndex] = uint32((segmentIndex*2654435761 + signatureIndex*2246822519 + 17) & ((1 << routeSignatureBits) - 1))
		}
		if segmentIndex < defaultRouteMinSegments {
			signatures[0] = querySignature
		}
		sort.Slice(signatures, func(left, right int) bool { return signatures[left] < signatures[right] })
		idx.segments[segmentIndex] = &AnnIndex{routeSignatures: signatures}
	}
	idx.segmentEpoch++
	idx.mu.Unlock()
	b.Cleanup(func() { _ = idx.Close() })
	return idx, query
}

func BenchmarkSegmentRoutingInvertedDense256x1024(b *testing.B) {
	idx, query := benchmarkDenseRouteIndex(b)
	_ = idx.routedIndexes(query)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		_ = idx.routedIndexes(query)
	}
}

func BenchmarkSegmentRoutingLegacyDense256x1024(b *testing.B) {
	idx, query := benchmarkDenseRouteIndex(b)
	signature := routeSignature64(query)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		idx.mu.RLock()
		segments := append([]*AnnIndex(nil), idx.segments...)
		idx.mu.RUnlock()
		routed := make([]routedSegment, len(segments))
		for position, segment := range segments {
			routed[position] = routedSegment{index: segment, position: position, distance: segment.routeDistance(signature)}
		}
		sort.Slice(routed, func(left, right int) bool {
			if routed[left].distance == routed[right].distance {
				return routed[left].position < routed[right].position
			}
			return routed[left].distance < routed[right].distance
		})
	}
}

func BenchmarkSegmentedSearch100Segments(b *testing.B) {
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 8, EfSearch: 8}, 10)
	defer idx.Close()
	idx.mu.Lock()
	idx.segments = make([]*AnnIndex, 100)
	for segment := range idx.segments {
		graph := NewAnnIndexWithOptions(Options{M: 4, EfConstruction: 8, EfSearch: 8})
		if err := graph.AddVector(segment+1, []float64{float64(segment), 1}); err != nil {
			b.Fatal(err)
		}
		if err := graph.Quantize(); err != nil {
			b.Fatal(err)
		}
		idx.segments[segment] = graph
	}
	idx.mu.Unlock()
	dst := make([]Result, 0, 10)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		results, err := idx.SearchWithDistancesInto([]float64{50, 1}, 10, dst[:0])
		if err != nil || len(results) != 10 {
			b.Fatalf("search = %d, %v", len(results), err)
		}
	}
}
