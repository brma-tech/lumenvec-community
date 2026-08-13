package ann

import (
	"context"
	"testing"
	"time"
)

func TestIVFANNIndexAdapterSupportsCoreContract(t *testing.T) {
	idx, err := NewIVFANNIndex([][]float64{{0}, {10}}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.AddVector(1, []float64{0}); err != nil {
		t.Fatal(err)
	}
	if err := idx.AddVector(2, []float64{10}); err != nil {
		t.Fatal(err)
	}
	hits, err := idx.SearchWithDistancesInto([]float64{10}, 1, nil)
	if err != nil || len(hits) != 1 || hits[0].ID != 2 {
		t.Fatalf("hits=%v err=%v", hits, err)
	}
	idx.DeleteVector(2)
	hits, err = idx.SearchWithDistancesInto([]float64{10}, 1, nil)
	if err != nil || len(hits) != 1 || hits[0].ID != 1 {
		t.Fatalf("after delete hits=%v err=%v", hits, err)
	}
	if stats := idx.Stats(); stats.Nodes != 2 || stats.Deleted != 1 {
		t.Fatalf("stats=%+v", stats)
	}
}

func TestIVFANNIndexTrainsFromSamples(t *testing.T) {
	idx, err := NewIVFANNIndexFromSamples([][]float64{{0}, {10}}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.AddVector(1, []float64{10}); err != nil {
		t.Fatal(err)
	}
	hits, err := idx.SearchWithDistancesInto([]float64{10}, 1, nil)
	if err != nil || len(hits) != 1 || hits[0].ID != 1 {
		t.Fatalf("hits=%v err=%v", hits, err)
	}
}

func TestIVFANNIndexRetrainRebuildsPartitions(t *testing.T) {
	idx, _ := NewIVFANNIndexFromSamples([][]float64{{0}, {10}}, 2, 2)
	entries := []IVFVector{{ID: 1, Values: []float64{0}}, {ID: 2, Values: []float64{10}}}
	if err := idx.Retrain(entries, 2, 2); err != nil {
		t.Fatal(err)
	}
	hits, err := idx.SearchWithDistancesInto([]float64{0}, 1, nil)
	if err != nil || len(hits) != 1 || hits[0].ID != 1 {
		t.Fatalf("hits=%v err=%v", hits, err)
	}
}

func TestIVFANNIndexRetrainLoopCancels(t *testing.T) {
	idx, _ := NewIVFANNIndexFromSamples([][]float64{{0}, {10}}, 2, 2)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := idx.RunRetrainLoop(ctx, time.Hour, func() []IVFVector { return []IVFVector{{ID: 1, Values: []float64{0}}} }, 1, 1)
	if err == nil {
		t.Fatal("expected cancellation")
	}
}

func TestIVFANNIndexDriftPolicy(t *testing.T) {
	idx, _ := NewIVFANNIndexFromSamples([][]float64{{0}, {10}}, 2, 2)
	needs, err := idx.NeedsRetrain([]IVFVector{{ID: 1, Values: []float64{5}}}, 1)
	if err != nil || !needs {
		t.Fatalf("needs=%v err=%v", needs, err)
	}
	needs, err = idx.NeedsRetrain([]IVFVector{{ID: 1, Values: []float64{0}}}, 1)
	if err != nil || needs {
		t.Fatalf("needs=%v err=%v", needs, err)
	}
}

func TestIVFANNIndexDriftAwareLoopCancels(t *testing.T) {
	idx, _ := NewIVFANNIndexFromSamples([][]float64{{0}, {10}}, 2, 2)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := idx.RunDriftAwareLoop(ctx, time.Hour, func() []IVFVector { return []IVFVector{{ID: 1, Values: []float64{0}}} }, 1, 2, 2)
	if err == nil {
		t.Fatal("expected cancellation")
	}
}
