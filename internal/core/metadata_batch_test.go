package core

import (
	"os"
	"path/filepath"
	"testing"

	"lumenvec/internal/index"
)

func TestAddVectorsWithMetadataPersistsBatch(t *testing.T) {
	dir := t.TempDir()
	options := ServiceOptions{
		MaxVectorDim: 2, MaxK: 10, SnapshotPath: filepath.Join(dir, "snapshot.json"),
		WALPath: filepath.Join(dir, "wal.log"), VectorPath: filepath.Join(dir, "vectors"),
	}
	service := NewService(options)
	vectors := []index.Vector{{ID: "a", Values: []float64{0, 0}}, {ID: "b", Values: []float64{1, 0}}}
	metadata := []map[string]string{{"benchmark_group": "measured"}, {"benchmark_group": "measured"}}
	if err := service.AddVectorsWithMetadata(vectors, metadata); err != nil {
		t.Fatal(err)
	}
	results, err := service.SearchStructured([]float64{0, 0}, 2, StructuredFilter{Metadata: metadata[0]}, MetricL2)
	if err != nil || len(results) != 2 {
		t.Fatalf("results=%v err=%v", results, err)
	}
	if err := service.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(options.SnapshotPath + ".metadata.json"); err != nil {
		t.Fatal(err)
	}

	restored := NewService(options)
	defer restored.Close()
	results, err = restored.SearchStructured([]float64{0, 0}, 2, StructuredFilter{Metadata: metadata[0]}, MetricL2)
	if err != nil || len(results) != 2 {
		t.Fatalf("restored results=%v err=%v", results, err)
	}
}

func TestAddVectorsWithMetadataRejectsMismatchedCounts(t *testing.T) {
	service := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 10})
	defer service.Close()
	if err := service.AddVectorsWithMetadata([]index.Vector{{ID: "a", Values: []float64{0, 0}}}, nil); err == nil {
		t.Fatal("expected mismatched metadata count to fail")
	}
}
