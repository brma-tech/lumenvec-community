package ann

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func testSnapshotManifest(id string, payload []byte) SegmentManifest {
	sum := sha256.Sum256(payload)
	return SegmentManifest{Version: segmentManifestVersion, SegmentID: id, State: SegmentSealed, FirstID: 1, LastID: 1, Nodes: 1, Dimension: 2, PayloadBytes: int64(len(payload)), PayloadSHA256: hex.EncodeToString(sum[:])}
}

func TestSnapshotStagerPublishesIncrementalArtifact(t *testing.T) {
	payload := []byte("prebuilt-ann-snapshot")
	path := filepath.Join(t.TempDir(), "segment.snapshot")
	stager, err := NewSnapshotStager(path, testSnapshotManifest("segment", payload))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stager.Write(payload[:5]); err != nil {
		t.Fatal(err)
	}
	if _, err := stager.Write(payload[5:]); err != nil {
		t.Fatal(err)
	}
	if err := stager.Commit(); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil || string(got) != string(payload) {
		t.Fatalf("artifact=%q err=%v", got, err)
	}
}

func TestSnapshotStagerRejectsInvalidTransferWithoutPublishing(t *testing.T) {
	payload := []byte("expected")
	path := filepath.Join(t.TempDir(), "segment.snapshot")
	stager, err := NewSnapshotStager(path, testSnapshotManifest("segment", payload))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stager.Write([]byte("wrong!!!")); err != nil {
		t.Fatal(err)
	}
	if err := stager.Commit(); err == nil {
		t.Fatal("expected checksum rejection")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("invalid transfer published path err=%v", err)
	}
}

func TestSnapshotStagerRetryIsIdempotentAndRejectsConflictingContent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "segment.snapshot")
	payload := []byte("same snapshot")
	manifest := testSnapshotManifest("segment", payload)
	for attempt := 0; attempt < 2; attempt++ {
		stager, err := NewSnapshotStager(path, manifest)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := stager.Write(payload); err != nil {
			t.Fatal(err)
		}
		if err := stager.Commit(); err != nil {
			t.Fatalf("idempotent attempt %d: %v", attempt, err)
		}
	}
	conflicting := []byte("other content")
	conflictingManifest := testSnapshotManifest("segment", conflicting)
	stager, err := NewSnapshotStager(path, conflictingManifest)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stager.Write(conflicting); err != nil {
		t.Fatal(err)
	}
	if err := stager.Commit(); err == nil {
		t.Fatal("expected conflicting retry to fail")
	}
	got, err := os.ReadFile(path)
	if err != nil || string(got) != string(payload) {
		t.Fatalf("published snapshot changed to %q err=%v", got, err)
	}
}

func TestSnapshotStagerPublishesStagedGraphWithoutByteMaterialization(t *testing.T) {
	graph := NewAnnIndexWithOptions(Options{M: 4, EfConstruction: 16, EfSearch: 16})
	if err := graph.AddVector(7, []float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	payload, err := graph.MarshalCompactBinary()
	if err != nil {
		t.Fatal(err)
	}
	manifest := testSnapshotManifest("segment", payload)
	manifest.FirstID, manifest.LastID, manifest.Nodes, manifest.Dimension = 7, 7, 1, 2
	path := filepath.Join(t.TempDir(), "segment.snapshot")
	stager, err := NewSnapshotStager(path, manifest)
	if err != nil {
		t.Fatal(err)
	}
	for start := 0; start < len(payload); start += 3 {
		end := start + 3
		if end > len(payload) {
			end = len(payload)
		}
		if _, err := stager.Write(payload[start:end]); err != nil {
			t.Fatal(err)
		}
	}
	if err := stager.Commit(); err != nil {
		t.Fatal(err)
	}
	idx := NewSegmentedIndex(Options{M: 4, EfConstruction: 16, EfSearch: 16}, 10)
	t.Cleanup(func() { _ = idx.Close() })
	if err := idx.PublishPrebuiltSnapshotFile(path, manifest); err != nil {
		t.Fatal(err)
	}
	results, err := idx.SearchWithDistancesInto([]float64{1, 0}, 1, nil)
	if err != nil || len(results) != 1 || results[0].ID != 7 {
		t.Fatalf("results=%+v err=%v", results, err)
	}
}

func TestWritePrebuiltSnapshotFileStreamsImmutableSegment(t *testing.T) {
	segment := NewAnnIndexWithOptions(Options{M: 4, EfConstruction: 16, EfSearch: 16})
	for _, vector := range []BatchVector32{{ID: 7, Values: []float32{1, 0}}, {ID: 3, Values: []float32{0, 1}}} {
		if err := segment.AddVector32(vector.ID, vector.Values); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(t.TempDir(), "segment.snapshot")
	manifest, err := WritePrebuiltSnapshotFile(path, "segment-source", "cosine", segment)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Nodes != 2 || manifest.FirstID != 3 || manifest.LastID != 7 || manifest.Metric != "cosine" {
		t.Fatalf("manifest=%+v", manifest)
	}
	if err := manifest.VerifyArtifactFile(path); err != nil {
		t.Fatal(err)
	}
	restored, err := OpenCompactBinaryFile(path)
	if err != nil {
		t.Fatal(err)
	}
	defer restored.Close()
	results, err := restored.Search([]float64{1, 0}, 1)
	if err != nil || len(results) != 1 || results[0] != 7 {
		t.Fatalf("results=%v err=%v", results, err)
	}
}
