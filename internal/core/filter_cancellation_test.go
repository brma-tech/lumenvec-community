package core

import (
	"context"
	"errors"
	"lumenvec/internal/index"
	"path/filepath"
	"testing"
)

func TestFilteredSearchCancelsDuringPredicate(t *testing.T) {
	dir := t.TempDir()
	s := NewService(ServiceOptions{SnapshotPath: filepath.Join(dir, "snapshot"), WALPath: filepath.Join(dir, "wal"), MaxVectorDim: 128, MaxK: 10})
	defer s.Close()
	for _, id := range []string{"a", "b", "c"} {
		if err := s.AddVector(id, []float64{1, 2}); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	calls := 0
	results, err := s.SearchFilteredMetricContext(ctx, []float64{1, 2}, 1, func(index.Vector) bool { calls++; cancel(); return true }, MetricL2)
	if !errors.Is(err, context.Canceled) || results != nil || calls != 1 {
		t.Fatalf("results=%v err=%v calls=%d", results, err, calls)
	}
	if results, err := s.SearchStructuredContext(ctx, []float64{1, 2}, 1, StructuredFilter{TextQuery: "a"}, MetricL2); !errors.Is(err, context.Canceled) || results != nil {
		t.Fatalf("%v %v", results, err)
	}
}
