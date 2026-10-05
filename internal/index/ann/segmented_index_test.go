package ann

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime/pprof"
	"slices"
	"sort"
	"strconv"
	"sync"
	"testing"
	"time"
)

type countingSnapshotMapping struct {
	closes int
}

func (m *countingSnapshotMapping) Close() error {
	m.closes++
	return nil
}

func TestSegmentedIndexMaintenanceState(t *testing.T) {
	idx := NewSegmentedIndex(Options{}, 4)
	defer idx.Close()
	first := NewAnnIndexWithOptions(Options{})
	second := NewAnnIndexWithOptions(Options{})
	if err := first.AddVector(1, []float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	if err := second.AddVector(2, []float64{2, 0}); err != nil {
		t.Fatal(err)
	}
	idx.mu.Lock()
	idx.segments = []*AnnIndex{first, second}
	idx.mu.Unlock()
	idx.pending.Store(true)
	idx.compacting.Store(true)
	pending, compacting, compactable := idx.MaintenanceState()
	if !pending || !compacting || !compactable {
		t.Fatalf("maintenance state = pending:%v compacting:%v compactable:%v", pending, compacting, compactable)
	}
}

func TestSegmentedIndexReclaimsRetiredGenerationAfterLastReader(t *testing.T) {
	t.Setenv("LUMENVEC_COMPACTION_THRESHOLD", "32")
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 8, EfSearch: 8}, 4)
	t.Cleanup(func() { _ = idx.Close() })
	for batch := 0; batch < 2; batch++ {
		if err := idx.BuildAndPublishBatch([]BatchVector{
			{ID: batch*2 + 1, Values: []float64{float64(batch), 0}},
			{ID: batch*2 + 2, Values: []float64{float64(batch), 1}},
		}); err != nil {
			t.Fatal(err)
		}
	}

	snapshot := idx.SnapshotExport()
	if len(snapshot.Segments) != 2 {
		snapshot.Release()
		t.Fatalf("sealed segments = %d, want 2", len(snapshot.Segments))
	}
	retiredGeneration := append([]*AnnIndex(nil), snapshot.Segments...)
	owners := make([]*countingSnapshotMapping, len(retiredGeneration))
	for position, segment := range retiredGeneration {
		owners[position] = &countingSnapshotMapping{}
		segment.mu.Lock()
		segment.mappedOwner = owners[position]
		segment.mu.Unlock()
	}
	idx.compactGeneration()

	readers, retired, reclaimed, reclaimErrors := idx.RetirementState()
	if readers != 1 || retired != 2 || reclaimed != 0 || reclaimErrors != 0 {
		t.Fatalf("while pinned: readers=%d retired=%d reclaimed=%d errors=%d", readers, retired, reclaimed, reclaimErrors)
	}
	for position, segment := range retiredGeneration {
		if stats := segment.Stats(); stats.Nodes != 2 {
			t.Fatalf("retired segment %d became unreadable while pinned: %+v", position, stats)
		}
		if owners[position].closes != 0 {
			t.Fatalf("retired mmap owner %d closed while reader was pinned", position)
		}
	}

	snapshot.Release()
	snapshot.Release() // Release is intentionally idempotent for error paths.
	readers, retired, reclaimed, reclaimErrors = idx.RetirementState()
	if readers != 0 || retired != 0 || reclaimed != 2 || reclaimErrors != 0 {
		t.Fatalf("after release: readers=%d retired=%d reclaimed=%d errors=%d", readers, retired, reclaimed, reclaimErrors)
	}
	for position, owner := range owners {
		if owner.closes != 1 {
			t.Fatalf("retired mmap owner %d closes = %d, want exactly 1", position, owner.closes)
		}
	}
	if got := idx.SegmentCount(); got != 1 {
		t.Fatalf("published segments = %d, want one merged generation", got)
	}
}

func TestSegmentedIndexRetiredReclamationFallback(t *testing.T) {
	t.Setenv("LUMENVEC_RECLAIM_RETIRED", "false")
	t.Setenv("LUMENVEC_COMPACTION_THRESHOLD", "32")
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 8, EfSearch: 8}, 4)
	for batch := 0; batch < 2; batch++ {
		if err := idx.BuildAndPublishBatch([]BatchVector{
			{ID: batch*2 + 1, Values: []float64{float64(batch), 0}},
			{ID: batch*2 + 2, Values: []float64{float64(batch), 1}},
		}); err != nil {
			t.Fatal(err)
		}
	}
	snapshot := idx.SnapshotExport()
	idx.compactGeneration()
	snapshot.Release()

	readers, retired, reclaimed, reclaimErrors := idx.RetirementState()
	if readers != 0 || retired != 2 || reclaimed != 0 || reclaimErrors != 0 {
		t.Fatalf("fallback state: readers=%d retired=%d reclaimed=%d errors=%d", readers, retired, reclaimed, reclaimErrors)
	}
	if err := idx.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestSegmentedIndexCloseWaitsForPinnedGeneration(t *testing.T) {
	idx := NewSegmentedIndex(Options{}, 4)
	snapshot := idx.SnapshotExport()
	closed := make(chan error, 1)
	go func() { closed <- idx.Close() }()

	deadline := time.Now().Add(time.Second)
	for {
		idx.mu.RLock()
		closing := idx.closed
		idx.mu.RUnlock()
		if closing {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("close did not enter the reader drain")
		}
		time.Sleep(time.Millisecond)
	}
	select {
	case err := <-closed:
		t.Fatalf("close returned before reader release: %v", err)
	case <-time.After(25 * time.Millisecond):
	}

	snapshot.Release()
	select {
	case err := <-closed:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("close did not finish after reader release")
	}
}

func TestSegmentedIndexVisitorErrorReleasesReader(t *testing.T) {
	idx := NewSegmentedIndex(Options{}, 4)
	t.Cleanup(func() { _ = idx.Close() })
	if err := idx.BuildAndPublishBatch([]BatchVector{{ID: 1, Values: []float64{1, 0}}}); err != nil {
		t.Fatal(err)
	}
	wantErr := errors.New("stop export")
	if err := idx.VisitSealedSegments(func(_ int, _ *AnnIndex) error { return wantErr }); !errors.Is(err, wantErr) {
		t.Fatalf("visitor error = %v, want %v", err, wantErr)
	}
	readers, _, _, _ := idx.RetirementState()
	if readers != 0 {
		t.Fatalf("active readers after visitor error = %d, want 0", readers)
	}
	var recovered any
	func() {
		defer func() { recovered = recover() }()
		_ = idx.VisitSealedSegments(func(_ int, _ *AnnIndex) error {
			panic("controlled visitor panic")
		})
	}()
	if recovered != "controlled visitor panic" {
		t.Fatalf("visitor panic = %v", recovered)
	}
	readers, _, _, _ = idx.RetirementState()
	if readers != 0 {
		t.Fatalf("active readers after visitor panic = %d, want 0", readers)
	}
}

func TestSegmentedIndexCompactionStrategiesPreserveLiveVectors(t *testing.T) {
	for _, strategy := range []string{"copy", "stream"} {
		t.Run(strategy, func(t *testing.T) {
			t.Setenv("LUMENVEC_ANN_COMPACTION_STRATEGY", strategy)
			t.Setenv("LUMENVEC_COMPACTION_THRESHOLD", "32")
			idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 8, EfSearch: 16}, 6)
			defer idx.Close()
			for batch := 0; batch < 2; batch++ {
				vectors := make([]BatchVector, 3)
				for item := range vectors {
					id := batch*3 + item
					vectors[item] = BatchVector{ID: id, Values: []float64{float64(id), float64(id % 2)}}
				}
				if err := idx.BuildAndPublishBatch(vectors); err != nil {
					t.Fatal(err)
				}
			}
			idx.DeleteVector(2)
			idx.compactGeneration()
			if stats := idx.Stats(); stats.Nodes != 5 || stats.Deleted != 0 {
				t.Fatalf("compacted stats = %+v", stats)
			}
			snapshot := idx.SnapshotExport()
			defer snapshot.Release()
			seen := make(map[int]struct{}, 5)
			for _, segment := range snapshot.Segments {
				if err := segment.VisitLiveVectors32(func(id int, _ []float32) error {
					seen[id] = struct{}{}
					return nil
				}); err != nil {
					t.Fatal(err)
				}
			}
			if _, found := seen[2]; found || len(seen) != 5 {
				t.Fatalf("live IDs after %s compaction = %v", strategy, seen)
			}
		})
	}
}

func TestSegmentedIndexCompactionMemoryBudgetDefersOversizedBuild(t *testing.T) {
	t.Setenv("LUMENVEC_ANN_COMPACTION_MEMORY_BUDGET_BYTES", "1")
	t.Setenv("LUMENVEC_COMPACTION_THRESHOLD", "32")
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 8, EfSearch: 16}, 4)
	defer idx.Close()
	for batch := 0; batch < 2; batch++ {
		if err := idx.BuildAndPublishBatch([]BatchVector{
			{ID: batch*2 + 1, Values: []float64{float64(batch), 0}},
			{ID: batch*2 + 2, Values: []float64{float64(batch), 1}},
		}); err != nil {
			t.Fatal(err)
		}
	}
	idx.compactGeneration()
	if got := idx.SegmentCount(); got != 2 {
		t.Fatalf("segments after deferred compaction = %d, want 2", got)
	}
	budget, reserved, estimated, deferred := idx.CompactionMemoryState()
	if budget != 1 || reserved != 0 || estimated <= budget || deferred != 1 {
		t.Fatalf("memory state: budget=%d reserved=%d estimated=%d deferred=%d", budget, reserved, estimated, deferred)
	}
	if stats := idx.Stats(); stats.Nodes != 4 || stats.Deleted != 0 {
		t.Fatalf("source generation changed after deferral: %+v", stats)
	}
}

func TestSegmentedIndexAdaptiveSearchMatchesLegacy(t *testing.T) {
	var baseline []Result
	for _, mode := range []string{"legacy", "adaptive", "inline"} {
		t.Run(mode, func(t *testing.T) {
			t.Setenv("LUMENVEC_ANN_SEARCH_EXECUTION", mode)
			idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 16, EfSearch: 32, Seed: 47}, 2)
			defer idx.Close()
			vectors := make([]BatchVector, 8)
			for id := range vectors {
				vectors[id] = BatchVector{ID: id, Values: []float64{float64(id), float64(id % 3)}}
			}
			if err := idx.BuildAndPublishBatch(vectors); err != nil {
				t.Fatal(err)
			}
			results, err := idx.SearchWithDistancesInto([]float64{3.2, 0}, 5, nil)
			if err != nil {
				t.Fatal(err)
			}
			if mode == "legacy" {
				baseline = append([]Result(nil), results...)
			} else if !slices.Equal(results, baseline) {
				t.Fatalf("%s results = %+v, legacy = %+v", mode, results, baseline)
			}
			if _, err := idx.SearchWithDistancesInto([]float64{1}, 2, nil); !errors.Is(err, ErrInvalidVectorDim) {
				t.Fatalf("invalid-dimension error = %v", err)
			}
		})
	}
}

func TestSegmentedIndexGlobalEFSearchBudget(t *testing.T) {
	for _, test := range []struct {
		mode       string
		multiplier string
		percent    string
		wantBudget uint64
	}{
		{mode: "per-segment", wantBudget: 2048},
		{mode: "global", wantBudget: 512},
		{mode: "global", multiplier: "2", wantBudget: 1024},
		{mode: "global", percent: "250", wantBudget: 1280},
	} {
		name := test.mode
		if test.multiplier != "" {
			name += "-" + test.multiplier + "x"
		} else if test.percent != "" {
			name += "-" + test.percent + "pct"
		}
		t.Run(name, func(t *testing.T) {
			t.Setenv("LUMENVEC_ANN_EF_BUDGET", test.mode)
			if test.multiplier != "" {
				t.Setenv("LUMENVEC_ANN_EF_GLOBAL_MULTIPLIER", test.multiplier)
			}
			if test.percent != "" {
				t.Setenv("LUMENVEC_ANN_EF_GLOBAL_PERCENT", test.percent)
			}
			idx := NewSegmentedIndex(Options{M: 8, EfConstruction: 32, EfSearch: 512, Seed: 59}, 64)
			defer idx.Close()
			for segment := 0; segment < 4; segment++ {
				batch := make([]BatchVector, 64)
				for item := range batch {
					id := segment*len(batch) + item
					batch[item] = BatchVector{ID: id, Values: []float64{float64(id % 31), float64(id / 31)}}
				}
				if err := idx.BuildAndPublishBatch(batch); err != nil {
					t.Fatal(err)
				}
			}
			results, err := idx.SearchWithDistancesInto([]float64{7.1, 3.2}, 10, nil)
			if err != nil || len(results) != 10 {
				t.Fatalf("results=%d err=%v", len(results), err)
			}
			queries, segments, budget := idx.SearchBudgetState()
			if queries != 1 || segments != 4 || budget != test.wantBudget {
				t.Fatalf("budget state: queries=%d segments=%d budget=%d, want 1/4/%d", queries, segments, budget, test.wantBudget)
			}
		})
	}
}

func TestANNCompactionMemoryReservationIsProcessWide(t *testing.T) {
	before := globalANNCompactionReserved.Load()
	release, retry, ok := reserveANNCompactionMemory(100, 70)
	if !ok || retry {
		t.Fatalf("first reservation: ok=%v retry=%v", ok, retry)
	}
	if _, retry, ok := reserveANNCompactionMemory(100, 40); ok || !retry {
		release()
		t.Fatalf("contended reservation: ok=%v retry=%v", ok, retry)
	}
	release()
	if after := globalANNCompactionReserved.Load(); after != before {
		t.Fatalf("global reservation leaked: before=%d after=%d", before, after)
	}
}

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
	defer snapshot.Release()
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
	defer snapshot.Release()
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

func TestPrimaryIndexCollapsesHighFanoutGeneration(t *testing.T) {
	t.Setenv("LUMENVEC_ANN_PRIMARY_INDEX", "on")
	idx := NewSegmentedIndex(Options{M: 8, EfConstruction: 32, EfSearch: 64, QuantizeSegments: true}, 1)
	t.Cleanup(func() { _ = idx.Close() })
	for id := 0; id < 12; id++ {
		if err := idx.AddVector(id, []float64{float64(id), 1, 0, 0}); err != nil {
			t.Fatal(err)
		}
	}
	if got := idx.SegmentCount(); got < 2 {
		t.Fatalf("expected high fanout before primary build, got %d", got)
	}
	if err := idx.BuildPrimaryIndex(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := idx.SegmentCount(); got != 1 {
		t.Fatalf("primary build left %d searchable segments, want 1", got)
	}
	results, err := idx.SearchWithDistancesInto([]float64{11, 1, 0, 0}, 1, nil)
	if err != nil || len(results) != 1 || results[0].ID != 11 {
		t.Fatalf("primary search results=%+v err=%v", results, err)
	}
}

func TestPrimaryIndexDisabledPreservesSegmentedFallback(t *testing.T) {
	t.Setenv("LUMENVEC_ANN_PRIMARY_INDEX", "off")
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 8, EfSearch: 16}, 1)
	t.Cleanup(func() { _ = idx.Close() })
	for id := 0; id < 4; id++ {
		if err := idx.AddVector(id, []float64{float64(id), 1}); err != nil {
			t.Fatal(err)
		}
	}
	if err := idx.BuildPrimaryIndex(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := idx.SegmentCount(); got < 2 {
		t.Fatalf("disabled primary unexpectedly collapsed segments: %d", got)
	}
}

func TestPrimaryIndexPreservesRecallOnUnclusteredData(t *testing.T) {
	t.Setenv("LUMENVEC_ANN_PRIMARY_INDEX", "on")
	t.Setenv("LUMENVEC_ANN_PRIMARY_M", "32")
	t.Setenv("LUMENVEC_ANN_PRIMARY_EF_CONSTRUCTION", "64")
	const (
		vectorCount = 1000
		dimension   = 24
		topK        = 10
	)
	idx := NewSegmentedIndex(Options{M: 32, EfConstruction: 128, EfSearch: 512, QuantizeSegments: true, SegmentRouting: true, Seed: 42}, 100)
	t.Cleanup(func() { _ = idx.Close() })
	rng := rand.New(rand.NewSource(42))
	vectors := make([]BatchVector, vectorCount)
	for id := range vectors {
		values := make([]float64, dimension)
		for dimensionIndex := range values {
			values[dimensionIndex] = rng.NormFloat64()
		}
		vectors[id] = BatchVector{ID: id, Values: values}
	}
	if err := idx.AddBatch(vectors); err != nil {
		t.Fatal(err)
	}
	if err := idx.BuildPrimaryIndex(context.Background()); err != nil {
		t.Fatal(err)
	}
	recallSum := 0.0
	for queryIndex := 0; queryIndex < 20; queryIndex++ {
		query := vectors[(queryIndex*47)%vectorCount].Values
		truth := make([]Result, vectorCount)
		for position, vector := range vectors {
			truth[position] = Result{ID: vector.ID, Distance: l2(query, vector.Values)}
		}
		sort.Slice(truth, func(left, right int) bool { return truth[left].Distance < truth[right].Distance })
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
	if recall := recallSum / 20; recall < 0.98 {
		t.Fatalf("primary Recall@10=%.4f, want >=0.98", recall)
	}
}

func TestPrimaryTopologyDifferential(t *testing.T) {
	const count, dim = 2000, 32
	rng := rand.New(rand.NewSource(901))
	vectors := make([]BatchVector, count)
	direct := NewAnnIndexWithOptionsCapacity(Options{M: 16, EfConstruction: 64, EfSearch: 256, Seed: 901}, count)
	segmented := NewSegmentedIndex(Options{M: 16, EfConstruction: 64, EfSearch: 256, Seed: 901}, 500)
	segmented.primaryIndexEnabled = true
	t.Cleanup(func() { _ = direct.Close(); _ = segmented.Close() })
	for id := range vectors {
		values := make([]float64, dim)
		for d := range values {
			values[d] = rng.NormFloat64()
		}
		vectors[id] = BatchVector{ID: id, Values: values}
		if err := direct.AddVector(id, values); err != nil {
			t.Fatal(err)
		}
	}
	if err := segmented.AddBatch(vectors); err != nil {
		t.Fatal(err)
	}
	if err := segmented.BuildPrimaryIndex(context.Background()); err != nil {
		t.Fatal(err)
	}
	directStats, primaryStats := direct.TopologyStats(), func() TopologyStats {
		idx, ok := segmented.singleIndexPinned()
		if !ok {
			t.Fatal("primary index not published")
		}
		stats := idx.TopologyStats()
		segmented.releaseReader()
		return stats
	}()
	truth := make([]Result, count)
	for i, vector := range vectors {
		truth[i] = Result{ID: vector.ID, Distance: l2(vectors[0].Values, vector.Values)}
	}
	sort.Slice(truth, func(i, j int) bool { return truth[i].Distance < truth[j].Distance })
	expected := make(map[int]struct{}, 10)
	for _, result := range truth[:10] {
		expected[result.ID] = struct{}{}
	}
	recall := func(results []Result) float64 {
		hits := 0
		for _, result := range results {
			if _, ok := expected[result.ID]; ok {
				hits++
			}
		}
		return float64(hits) / 10
	}
	directResults, err := direct.SearchWithDistances(vectors[0].Values, 10)
	if err != nil {
		t.Fatal(err)
	}
	primaryResults, err := segmented.SearchWithDistancesInto(vectors[0].Values, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("direct=%+v recall=%.2f primary=%+v recall=%.2f", directStats, recall(directResults), primaryStats, recall(primaryResults))
}

func TestPrimaryIndexAutoConsolidatesPublishedSegments(t *testing.T) {
	t.Setenv("LUMENVEC_ANN_PRIMARY_INDEX", "on")
	idx := NewSegmentedIndex(Options{M: 8, EfConstruction: 16, EfSearch: 32}, 100)
	t.Cleanup(func() { _ = idx.Close() })
	vectors := make([]BatchVector, 200)
	for id := range vectors {
		vectors[id] = BatchVector{ID: id, Values: []float64{float64(id), 1}}
	}
	if err := idx.BuildAndPublishBatch(vectors); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if idx.SegmentCount() == 1 {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("primary worker left %d segments", idx.SegmentCount())
}

func TestPrimaryIndexRespectsBuildMemoryBudget(t *testing.T) {
	t.Setenv("LUMENVEC_ANN_PRIMARY_INDEX", "on")
	t.Setenv("LUMENVEC_ANN_COMPACTION_MEMORY_BUDGET_BYTES", "1")
	idx := NewSegmentedIndex(Options{M: 8, EfConstruction: 16, EfSearch: 32}, 10)
	t.Cleanup(func() { _ = idx.Close() })
	for id := 0; id < 20; id++ {
		if err := idx.AddVector(id, []float64{float64(id), 1}); err != nil {
			t.Fatal(err)
		}
	}
	if err := idx.BuildPrimaryIndex(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := idx.SegmentCount(); got <= 1 {
		t.Fatalf("memory budget unexpectedly published primary index with %d segments", got)
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

func benchmarkLargeSegmentedIndex(b testing.TB, primary bool) (*SegmentedIndex, [][]float64) {
	b.Helper()
	efSearch := 512
	if raw := os.Getenv("LUMENVEC_SEGMENTED_BENCH_EF_SEARCH"); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			efSearch = parsed
		}
	}
	idx := NewSegmentedIndex(Options{
		M: 32, EfConstruction: 128, EfSearch: efSearch,
		QuantizeSegments: true, SegmentRouting: true, Seed: 42,
	}, 10000)
	idx.mu.Lock()
	idx.compactThreshold = 1_000_000
	idx.searchExecution = "inline"
	idx.globalEFSearch = true
	idx.globalEFPercent = 250
	idx.primaryIndexEnabled = primary
	idx.mu.Unlock()
	vectors := make([]BatchVector, 100000)
	rng := rand.New(rand.NewSource(42))
	for id := range vectors {
		values := make([]float64, 128)
		for dimension := range values {
			values[dimension] = rng.Float64()*2 - 1
		}
		vectors[id] = BatchVector{ID: id, Values: values}
	}
	if err := idx.AddBatch(vectors); err != nil {
		b.Fatal(err)
	}
	if primary {
		if err := idx.BuildPrimaryIndex(context.Background()); err != nil {
			b.Fatal(err)
		}
	}
	queries := make([][]float64, 32)
	for i := range queries {
		queries[i] = append([]float64(nil), vectors[i*997].Values...)
	}
	b.Cleanup(func() { _ = idx.Close() })
	return idx, queries
}

func TestProfileSegmentedPrimarySearchOnly(t *testing.T) {
	path := os.Getenv("LUMENVEC_SEGMENTED_SEARCH_PROFILE_PATH")
	if path == "" {
		t.Skip("set LUMENVEC_SEGMENTED_SEARCH_PROFILE_PATH to enable")
	}
	primary := os.Getenv("LUMENVEC_SEGMENTED_SEARCH_PROFILE_PRIMARY") != "0"
	idx, queries := benchmarkLargeSegmentedIndex(t, primary)
	defer idx.Close()
	profile, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := pprof.StartCPUProfile(profile); err != nil {
		profile.Close()
		t.Fatal(err)
	}
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		for _, query := range queries {
			if _, err := idx.SearchWithDistancesInto(query, 10, nil); err != nil {
				pprof.StopCPUProfile()
				profile.Close()
				t.Fatal(err)
			}
		}
	}
	pprof.StopCPUProfile()
	if err := profile.Close(); err != nil {
		t.Fatal(err)
	}
	if heapPath := os.Getenv("LUMENVEC_SEGMENTED_HEAP_PROFILE_PATH"); heapPath != "" {
		heap, err := os.Create(heapPath)
		if err != nil {
			t.Fatal(err)
		}
		if err := pprof.Lookup("heap").WriteTo(heap, 0); err != nil {
			heap.Close()
			t.Fatal(err)
		}
		if err := heap.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func BenchmarkSegmentedSearch100kPrimaryVsFanout(b *testing.B) {
	for _, primary := range []bool{false, true} {
		name := "fanout"
		if primary {
			name = "primary"
		}
		b.Run(name, func(b *testing.B) {
			idx, queries := benchmarkLargeSegmentedIndex(b, primary)
			if index, ok := idx.singleIndexPinned(); ok {
				quantized, compacted, mapped := index.LayoutState()
				b.Logf("layout quantized=%t compacted=%t mapped=%t", quantized, compacted, mapped)
				idx.releaseReader()
			}
			var dst []Result
			reuseResults := os.Getenv("LUMENVEC_SEGMENTED_BENCH_REUSE_DST") == "1"
			// Warm the per-index search workspace so the timed section measures
			// steady-state query cost rather than one-time pool growth.
			for warmup := 0; warmup < 16; warmup++ {
				var err error
				dst, err = idx.SearchWithDistancesInto(queries[warmup%len(queries)], 10, dst)
				if err != nil {
					b.Fatal(err)
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				if reuseResults {
					var err error
					dst, err = idx.SearchWithDistancesInto(queries[iteration%len(queries)], 10, dst[:0])
					if err != nil {
						b.Fatal(err)
					}
					continue
				}
				if _, err := idx.SearchWithDistancesInto(queries[iteration%len(queries)], 10, nil); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			work := idx.SearchWorkStats()
			if work.Queries > 0 {
				b.ReportMetric(float64(work.Visited)/float64(work.Queries), "visited/query")
				b.ReportMetric(float64(work.DistanceCalls)/float64(work.Queries), "distance/query")
			}
			memory := idx.MemoryStats()
			b.ReportMetric(float64(memory.TotalBytes)/(1024*1024), "index-MiB")
			if index, ok := idx.singleIndexPinned(); ok {
				topology := index.TopologyStats()
				b.ReportMetric(topology.AverageDegree, "avg-degree")
				b.ReportMetric(float64(topology.ReachableNodes), "reachable-nodes")
				idx.releaseReader()
			}
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

func BenchmarkSegmentedRetiredGenerationReclamation(b *testing.B) {
	for _, benchmark := range []struct {
		name    string
		reclaim bool
	}{
		{name: "fallback-retain-until-close", reclaim: false},
		{name: "reader-count-reclaim", reclaim: true},
	} {
		b.Run(benchmark.name, func(b *testing.B) {
			b.Setenv("LUMENVEC_COMPACTION_THRESHOLD", "32")
			b.Setenv("LUMENVEC_RECLAIM_RETIRED", strconv.FormatBool(benchmark.reclaim))
			idx := NewSegmentedIndex(Options{M: 8, EfConstruction: 32, EfSearch: 64, Seed: 41}, 512)
			b.Cleanup(func() { _ = idx.Close() })
			for generation := 0; generation < 2; generation++ {
				batch := make([]BatchVector, 256)
				for item := range batch {
					id := generation*len(batch) + item
					values := make([]float64, 32)
					for dimension := range values {
						values[dimension] = float64((id+1)*(dimension+3)%997) / 997
					}
					batch[item] = BatchVector{ID: id, Values: values}
				}
				if err := idx.BuildAndPublishBatch(batch); err != nil {
					b.Fatal(err)
				}
			}
			idx.compactGeneration()
			_, retired, reclaimed, reclaimErrors := idx.RetirementState()
			var retainedBytes uint64
			idx.mu.RLock()
			for _, segment := range idx.retired {
				retainedBytes += segment.MemoryStats().TotalBytes
			}
			idx.mu.RUnlock()
			query := make([]float64, 32)
			for dimension := range query {
				query[dimension] = float64((257)*(dimension+3)%997) / 997
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := idx.SearchWithDistancesInto(query, 10, nil); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(retired), "retired_segments")
			b.ReportMetric(float64(reclaimed), "reclaimed_segments")
			b.ReportMetric(float64(reclaimErrors), "reclaim_errors")
			b.ReportMetric(float64(retainedBytes), "retained_bytes")
		})
	}
}

func BenchmarkSegmentedCompactionStrategy(b *testing.B) {
	for _, strategy := range []string{"copy", "stream"} {
		b.Run(strategy, func(b *testing.B) {
			b.Setenv("LUMENVEC_ANN_COMPACTION_STRATEGY", strategy)
			b.Setenv("LUMENVEC_COMPACTION_THRESHOLD", "32")
			b.ReportAllocs()
			for iteration := 0; iteration < b.N; iteration++ {
				b.StopTimer()
				idx := NewSegmentedIndex(Options{M: 8, EfConstruction: 32, EfSearch: 64, Seed: 43}, 512)
				for generation := 0; generation < 2; generation++ {
					batch := make([]BatchVector, 256)
					for item := range batch {
						id := generation*len(batch) + item
						values := make([]float64, 32)
						for dimension := range values {
							values[dimension] = float64((id+7)*(dimension+5)%991) / 991
						}
						batch[item] = BatchVector{ID: id, Values: values}
					}
					if err := idx.BuildAndPublishBatch(batch); err != nil {
						b.Fatal(err)
					}
				}
				b.StartTimer()
				idx.compactGeneration()
				b.StopTimer()
				if _, retired, _, errors := idx.RetirementState(); retired != 0 || errors != 0 {
					b.Fatalf("retirement state: retired=%d errors=%d", retired, errors)
				}
				if err := idx.Close(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkSegmentedSearchExecution(b *testing.B) {
	for _, fanout := range []int{1, 4, 8, 16} {
		for _, mode := range []string{"legacy", "adaptive", "inline"} {
			b.Run(fmt.Sprintf("fanout-%d/%s", fanout, mode), func(b *testing.B) {
				b.Setenv("LUMENVEC_ANN_SEARCH_EXECUTION", mode)
				idx := NewSegmentedIndex(Options{M: 8, EfConstruction: 32, EfSearch: 64, Seed: 53}, 64)
				b.Cleanup(func() { _ = idx.Close() })
				for segment := 0; segment < fanout; segment++ {
					batch := make([]BatchVector, 64)
					for item := range batch {
						id := segment*len(batch) + item
						values := make([]float64, 32)
						for dimension := range values {
							values[dimension] = float64((id+11)*(dimension+7)%983) / 983
						}
						batch[item] = BatchVector{ID: id, Values: values}
					}
					if err := idx.BuildAndPublishBatch(batch); err != nil {
						b.Fatal(err)
					}
				}
				query := make([]float64, 32)
				for dimension := range query {
					query[dimension] = float64(257*(dimension+7)%983) / 983
				}
				b.ReportAllocs()
				b.ResetTimer()
				for iteration := 0; iteration < b.N; iteration++ {
					if _, err := idx.SearchWithDistancesInto(query, 10, nil); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
