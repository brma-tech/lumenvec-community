package ann

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func TestBuildAndPublishSegment(t *testing.T) {
	dir := t.TempDir()
	catalog, err := OpenSegmentCatalog(filepath.Join(dir, "catalog.json"))
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := BuildAndPublishSegment(catalog, filepath.Join(dir, "seg.payload"), filepath.Join(dir, "seg.json"), "seg-1", 2, 8, 32, 1, 2, "l2", func(w *SegmentPayloadWriter) error {
		if err := w.WriteVector(1, []float32{1, 2}); err != nil {
			return err
		}
		return w.WriteVector(2, []float32{3, 4})
	})
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Nodes != 2 || len(catalog.Snapshot()) != 1 {
		t.Fatalf("manifest=%+v", manifest)
	}
}

func TestBuildAndPublishSegmentsPublishesWindow(t *testing.T) {
	dir := t.TempDir()
	catalog, err := OpenSegmentCatalog(filepath.Join(dir, "catalog.json"))
	if err != nil {
		t.Fatal(err)
	}
	specs := []SegmentBuildSpec{
		{PayloadPath: filepath.Join(dir, "a.payload"), ManifestPath: filepath.Join(dir, "a.json"), SegmentID: "a", Dimension: 2, M: 8, EfConstruction: 32, FirstID: 1, LastID: 1, Metric: "l2", Write: func(w *SegmentPayloadWriter) error { return w.WriteVector(1, []float32{1, 0}) }},
		{PayloadPath: filepath.Join(dir, "b.payload"), ManifestPath: filepath.Join(dir, "b.json"), SegmentID: "b", Dimension: 2, M: 8, EfConstruction: 32, FirstID: 2, LastID: 2, Metric: "l2", Write: func(w *SegmentPayloadWriter) error { return w.WriteVector(2, []float32{0, 1}) }},
	}
	manifests, err := BuildAndPublishSegments(catalog, specs)
	if err != nil {
		t.Fatal(err)
	}
	if len(manifests) != 2 || len(catalog.Snapshot()) != 2 {
		t.Fatalf("manifests=%+v catalog=%+v", manifests, catalog.Snapshot())
	}
}

func TestPublishPrebuiltSnapshotsPublishesDurableWindow(t *testing.T) {
	dir := t.TempDir()
	catalog, err := OpenSegmentCatalog(filepath.Join(dir, "catalog.json"))
	if err != nil {
		t.Fatal(err)
	}
	build := func(id int) []byte {
		segment := NewAnnIndexWithOptions(Options{M: 4, EfConstruction: 16, EfSearch: 16})
		if err := segment.AddVector(id, []float64{float64(id), 0}); err != nil {
			t.Fatal(err)
		}
		data, err := segment.MarshalCompactBinary()
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	manifests, err := PublishPrebuiltSnapshots(catalog, []SnapshotBuildSpec{
		{SnapshotPath: filepath.Join(dir, "a.snapshot"), ManifestPath: filepath.Join(dir, "a.json"), SegmentID: "a", Metric: "l2", Snapshot: build(1)},
		{SnapshotPath: filepath.Join(dir, "b.snapshot"), ManifestPath: filepath.Join(dir, "b.json"), SegmentID: "b", Metric: "l2", Snapshot: build(2)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(manifests) != 2 || len(catalog.Snapshot()) != 2 {
		t.Fatalf("manifests=%+v catalog=%+v", manifests, catalog.Snapshot())
	}
	for _, manifest := range manifests {
		if err := manifest.VerifyArtifactFile(filepath.Join(dir, manifest.SegmentID+".snapshot")); err != nil {
			t.Fatalf("verify %s: %v", manifest.SegmentID, err)
		}
	}
	if _, err := PublishPrebuiltSnapshots(catalog, []SnapshotBuildSpec{{SegmentID: "bad", Snapshot: []byte("invalid")}}); err == nil {
		t.Fatal("expected invalid snapshot rejection")
	}
	if got := len(catalog.Snapshot()); got != 2 {
		t.Fatalf("invalid window changed catalog=%d", got)
	}
}

func TestOpenSegmentedIndexLoadsPublishedPayload(t *testing.T) {
	dir := t.TempDir()
	catalog, err := OpenSegmentCatalog(filepath.Join(dir, "catalog.json"))
	if err != nil {
		t.Fatal(err)
	}
	payload := filepath.Join(dir, "seg-1.payload")
	manifestPath := filepath.Join(dir, "seg-1.json")
	_, err = BuildAndPublishSegment(catalog, payload, manifestPath, "seg-1", 2, 8, 32, 1, 2, "l2", func(w *SegmentPayloadWriter) error {
		if err := w.WriteVector(1, []float32{1, 0}); err != nil {
			return err
		}
		return w.WriteVector(2, []float32{0, 1})
	})
	if err != nil {
		t.Fatal(err)
	}
	idx, err := OpenSegmentedIndex(Options{M: 8, EfConstruction: 32, EfSearch: 32}, 10, catalog, func(SegmentManifest) string { return payload })
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	results, err := idx.SearchWithDistancesInto([]float64{1, 0}, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].ID != 1 {
		t.Fatalf("unexpected results: %+v", results)
	}
}

func TestOpenSegmentedIndexWithSnapshots(t *testing.T) {
	dir := t.TempDir()
	catalog, err := OpenSegmentCatalog(filepath.Join(dir, "catalog.json"))
	if err != nil {
		t.Fatal(err)
	}
	seg := NewAnnIndexWithOptions(Options{M: 8, EfConstruction: 32, EfSearch: 32})
	if err := seg.AddVector(1, []float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	snapshot, err := seg.MarshalCompactBinary()
	if err != nil {
		t.Fatal(err)
	}
	snapshotPath := filepath.Join(dir, "seg.snapshot")
	if err := os.WriteFile(snapshotPath, snapshot, 0o644); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(snapshot)
	manifest := SegmentManifest{Version: segmentManifestVersion, SegmentID: "seg", State: SegmentPublished, Nodes: 1, Dimension: 2, FirstID: 1, LastID: 1, PayloadBytes: int64(len(snapshot)), PayloadSHA256: hex.EncodeToString(sum[:])}
	if err := catalog.Publish(manifest); err != nil {
		t.Fatal(err)
	}
	idx, err := OpenSegmentedIndexWithSnapshots(Options{}, 10, catalog, func(SegmentManifest) string { return snapshotPath })
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	results, err := idx.SearchWithDistancesInto([]float64{1, 0}, 1, nil)
	if err != nil || len(results) != 1 || results[0].ID != 1 {
		t.Fatalf("unexpected snapshot results: %+v, %v", results, err)
	}
}

func TestOpenSegmentedIndexRejectsCorruptPayload(t *testing.T) {
	dir := t.TempDir()
	catalog, err := OpenSegmentCatalog(filepath.Join(dir, "catalog.json"))
	if err != nil {
		t.Fatal(err)
	}
	payload := filepath.Join(dir, "seg.payload")
	_, err = BuildAndPublishSegment(catalog, payload, filepath.Join(dir, "seg.json"), "seg", 2, 8, 32, 1, 1, "l2", func(w *SegmentPayloadWriter) error {
		return w.WriteVector(1, []float32{1, 0})
	})
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(payload, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = f.WriteAt([]byte{0xff}, 10); err != nil {
		t.Fatal(err)
	}
	if err = f.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = OpenSegmentedIndex(Options{M: 8, EfConstruction: 32, EfSearch: 32}, 10, catalog, func(SegmentManifest) string { return payload }); err == nil {
		t.Fatal("expected checksum validation failure")
	}
}

func BenchmarkOpenSegmentedIndex(b *testing.B) {
	dir := b.TempDir()
	catalog, err := OpenSegmentCatalog(filepath.Join(dir, "catalog.json"))
	if err != nil {
		b.Fatal(err)
	}
	payload := filepath.Join(dir, "seg.payload")
	if _, err = BuildAndPublishSegment(catalog, payload, filepath.Join(dir, "seg.json"), "seg", 16, 8, 32, 0, 9999, "l2", func(w *SegmentPayloadWriter) error {
		values := make([]float32, 16)
		for i := 0; i < 10000; i++ {
			values[0] = float32(i)
			if err := w.WriteVector(i, values); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, err := OpenSegmentedIndex(Options{M: 8, EfConstruction: 32, EfSearch: 32}, 10000, catalog, func(SegmentManifest) string { return payload })
		if err != nil {
			b.Fatal(err)
		}
		if err := idx.Close(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkOpenSegmentedIndexWithSnapshots(b *testing.B) {
	dir := b.TempDir()
	catalog, err := OpenSegmentCatalog(filepath.Join(dir, "catalog.json"))
	if err != nil {
		b.Fatal(err)
	}
	seg := NewAnnIndexWithOptions(Options{M: 8, EfConstruction: 32, EfSearch: 32})
	values := make([]float64, 16)
	for i := 0; i < 10000; i++ {
		values[0] = float64(i)
		if err := seg.AddVector(i, values); err != nil {
			b.Fatal(err)
		}
	}
	seg.CompactAdjacency()
	if err := seg.Quantize(); err != nil {
		b.Fatal(err)
	}
	raw, err := seg.MarshalCompactBinary()
	if err != nil {
		b.Fatal(err)
	}
	snapshotPath := filepath.Join(dir, "seg.snapshot")
	if err := os.WriteFile(snapshotPath, raw, 0o644); err != nil {
		b.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	manifest := SegmentManifest{Version: segmentManifestVersion, SegmentID: "seg", State: SegmentPublished, Nodes: 10000, Dimension: 16, FirstID: 0, LastID: 9999, PayloadBytes: int64(len(raw)), PayloadSHA256: hex.EncodeToString(sum[:])}
	if err := catalog.Publish(manifest); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, err := OpenSegmentedIndexWithSnapshots(Options{QuantizeSegments: true}, 10000, catalog, func(SegmentManifest) string { return snapshotPath })
		if err != nil {
			b.Fatal(err)
		}
		_ = idx.Close()
	}
}
