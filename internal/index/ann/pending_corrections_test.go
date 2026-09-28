package ann

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

func TestStreamingBuildMatchesBatchAndOwnsPayload(t *testing.T) {
	opts := Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 42}
	batch := make([]BatchVector32, 256)
	for n := range batch {
		batch[n] = BatchVector32{ID: n, Values: []float32{float32(n), float32(n % 13)}}
	}
	want, err := BuildHierarchicalIndex(opts, batch)
	if err != nil {
		t.Fatal(err)
	}
	var scratch [2]float32
	calls := 0
	got, err := BuildHierarchicalIndexFromSource(opts, len(batch), func(n int) (BatchVector32, error) {
		calls++
		copy(scratch[:], batch[n].Values)
		return BatchVector32{ID: n, Values: scratch[:]}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	scratch[0] = -100
	if calls != len(batch) || !reflect.DeepEqual(want.nodes, got.nodes) || !reflect.DeepEqual(want.vectors, got.vectors) {
		t.Fatal("streaming build changed graph or retained borrowed payload")
	}
	sentinel := errors.New("source failure")
	failed, err := BuildHierarchicalIndexFromSource(opts, len(batch), func(n int) (BatchVector32, error) {
		if n == 17 {
			return BatchVector32{}, sentinel
		}
		return batch[n], nil
	})
	if failed != nil || !errors.Is(err, sentinel) {
		t.Fatalf("partial generation escaped: %v", err)
	}
}

func TestLegacySearchCancellationAndInvalidBudget(t *testing.T) {
	a := NewAnnIndexWithOptions(Options{M: 8, EfConstruction: 32, EfSearch: 32})
	for n := 0; n < 128; n++ {
		if err := a.AddVector(n, []float64{float64(n), 1}); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if result, err := a.SearchWithDistancesContext(ctx, []float64{1, 1}, 10, nil); !errors.Is(err, context.Canceled) || result != nil {
		t.Fatalf("%v %v", result, err)
	}
	_, _, before := a.SearchBudgetState()
	_, _ = a.SearchWithDistancesEfInto([]float64{1, 1}, -1, -1, nil)
	_, _, after := a.SearchBudgetState()
	if before != after {
		t.Fatal("invalid request consumed budget")
	}
}
