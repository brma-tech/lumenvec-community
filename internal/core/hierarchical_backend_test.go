package core

import (
	"fmt"
	"path/filepath"
	"reflect"
	"testing"

	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"
)

func TestHierarchicalBackendIsOptInAndRestoresFromCanonicalSnapshot(t *testing.T) {
	base := t.TempDir()
	opts := ServiceOptions{
		MaxVectorDim:  8,
		MaxK:          4,
		SearchMode:    "ann",
		ANNBackend:    "hierarchical-hnsw",
		SnapshotPath:  filepath.Join(base, "snapshot.json"),
		WALPath:       filepath.Join(base, "wal.log"),
		SnapshotEvery: 1,
		ANNOptions: ann.Options{
			M: 4, EfConstruction: 16, EfSearch: 16, Seed: 42,
		},
	}

	service := NewService(opts)
	if _, ok := service.currentANNIndex().(*ann.HierarchicalIndex); !ok {
		t.Fatalf("ANN index type = %T, want *ann.HierarchicalIndex", service.currentANNIndex())
	}
	if err := service.AddVector("near", []float64{0, 0}); err != nil {
		t.Fatalf("AddVector(near) error = %v", err)
	}
	if err := service.AddVector("far", []float64{10, 10}); err != nil {
		t.Fatalf("AddVector(far) error = %v", err)
	}
	results, err := service.Search([]float64{0.1, 0.1}, 1)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 1 || results[0].ID != "near" {
		t.Fatalf("Search() = %+v, want near", results)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	restored := NewService(opts)
	t.Cleanup(func() { _ = restored.Close() })
	if _, ok := restored.currentANNIndex().(*ann.HierarchicalIndex); !ok {
		t.Fatalf("restored ANN index type = %T, want *ann.HierarchicalIndex", restored.currentANNIndex())
	}
	results, err = restored.Search([]float64{0.1, 0.1}, 1)
	if err != nil {
		t.Fatalf("restored Search() error = %v", err)
	}
	if len(results) != 1 || results[0].ID != "near" {
		t.Fatalf("restored Search() = %+v, want near", results)
	}
}

func TestDefaultANNBackendRemainsCompatibilityIndex(t *testing.T) {
	service := NewService(ServiceOptions{MaxVectorDim: 8, MaxK: 4, SearchMode: "ann"})
	t.Cleanup(func() { _ = service.Close() })
	if _, ok := service.currentANNIndex().(*ann.AnnIndex); !ok {
		t.Fatalf("default ANN index type = %T, want *ann.AnnIndex", service.currentANNIndex())
	}
}

func TestHierarchicalRebuildScratchBudgetPreservesPublishedGeneration(t *testing.T) {
	service := NewServiceWithDeps(ServiceOptions{
		MaxVectorDim: 8, MaxK: 4, SearchMode: "ann", ANNBackend: "hierarchical-hnsw",
		ANNOptions: ann.Options{
			M: 4, EfConstruction: 16, EfSearch: 16, Seed: 42,
			DiversifiedExistingPruning:          true,
			HierarchicalBuildScratchBudgetBytes: 1 << 20,
		},
	}, ServiceDeps{Persistence: &noopPersistence{}})
	t.Cleanup(func() { _ = service.Close() })
	for id := 0; id < 64; id++ {
		vector := index.Vector{ID: fmt.Sprintf("v-%03d", id), Values: []float64{float64(id), float64(id % 5), 1}}
		if err := service.index.AddVector(vector); err != nil {
			t.Fatal(err)
		}
		if err := service.upsertVectors([]index.Vector{vector}); err != nil {
			t.Fatal(err)
		}
	}
	before := service.currentANNIndex()
	service.rebuildANNLocked()
	before = service.currentANNIndex()
	if before.Stats().Nodes != 64 {
		t.Fatalf("initial private generation has %d nodes, want 64", before.Stats().Nodes)
	}
	service.annOptions.HierarchicalBuildScratchBudgetBytes = 1
	service.rebuildANNLocked()
	if after := service.currentANNIndex(); after != before {
		t.Fatal("failed private rebuild replaced the published generation")
	}
	results, err := service.Search([]float64{12, 2, 1}, 1)
	if err != nil || len(results) != 1 {
		t.Fatalf("published generation unusable after rejected rebuild: results=%v err=%v", results, err)
	}
}

func TestHierarchicalRebuildIsDeterministicBeforeIDAssignment(t *testing.T) {
	build := func(reverse bool) (*Service, ann.ConnectivityStats, []SearchResult) {
		service := NewServiceWithDeps(ServiceOptions{
			MaxVectorDim: 8, MaxK: 5, SearchMode: "ann", ANNBackend: "hierarchical-hnsw",
			ANNOptions: ann.Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 42},
		}, ServiceDeps{Persistence: &noopPersistence{}})
		for position := 0; position < 300; position++ {
			id := position
			if reverse {
				id = 299 - position
			}
			vector := index.Vector{ID: fmt.Sprintf("v-%04d", id), Values: []float64{float64(id), float64(id % 17), float64(id % 7)}}
			if err := service.index.AddVector(vector); err != nil {
				t.Fatal(err)
			}
			if err := service.upsertVectors([]index.Vector{vector}); err != nil {
				t.Fatal(err)
			}
		}
		service.rebuildANNLocked()
		stats := service.currentANNIndex().(*ann.HierarchicalIndex).Connectivity()
		results, err := service.Search([]float64{123.1, 4, 4}, 5)
		if err != nil {
			t.Fatal(err)
		}
		return service, stats, results
	}
	forward, forwardStats, forwardResults := build(false)
	t.Cleanup(func() { _ = forward.Close() })
	reverse, reverseStats, reverseResults := build(true)
	t.Cleanup(func() { _ = reverse.Close() })
	if !reflect.DeepEqual(forwardStats, reverseStats) {
		t.Fatalf("rebuild topology depends on source order: forward=%+v reverse=%+v", forwardStats, reverseStats)
	}
	if !reflect.DeepEqual(forwardResults, reverseResults) {
		t.Fatalf("rebuild results depend on source order: forward=%+v reverse=%+v", forwardResults, reverseResults)
	}
}
