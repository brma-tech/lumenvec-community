package core

import (
	"context"
	"errors"
	"lumenvec/internal/index/ann"
	"path/filepath"
	"testing"
)

func TestSearchContextAlreadyCanceled(t *testing.T) {
	s := NewService(ServiceOptions{MaxVectorDim: 128, MaxK: 10})
	defer s.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.SearchContext(ctx, []float64{1, 2}, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
}

func TestRebuildBudgetPreservesPublishedGeneration(t *testing.T) {
	dir := t.TempDir()
	s := NewService(ServiceOptions{SnapshotPath: filepath.Join(dir, "snapshot"), WALPath: filepath.Join(dir, "wal"), MaxVectorDim: 128, MaxK: 10, SearchMode: "ann", ANNBackend: "hierarchical-hnsw", ANNProfile: "custom", ANNOptions: ann.Options{M: 8, EfConstruction: 32, EfSearch: 32}})
	defer s.Close()
	if err := s.AddVector("one", []float64{1, 2}); err != nil {
		t.Fatal(err)
	}
	previous := s.currentANNIndex()
	s.annRebuildSnapshotBudgetBytes = 1
	s.rebuildANNLocked()
	if s.currentANNIndex() != previous {
		t.Fatal("failed rebuild replaced active generation")
	}
	stats := s.Stats()
	if stats.ANNRebuildFailures != 1 || stats.ANNRebuildLastError == "" {
		t.Fatalf("missing rebuild failure: %+v", stats)
	}
	s.annRebuildSnapshotBudgetBytes = 1024
	s.rebuildANNLocked()
	if s.Stats().ANNRebuildSuccesses == 0 || s.Stats().ANNRebuildLastError != "" {
		t.Fatal("successful rebuild state not cleared")
	}
}
