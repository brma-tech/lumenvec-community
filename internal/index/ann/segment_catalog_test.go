package ann

import (
	"path/filepath"
	"testing"
)

func TestSegmentCatalogPublishesAndReopens(t *testing.T) {
	path := filepath.Join(t.TempDir(), "catalog.json")
	catalog, err := OpenSegmentCatalog(path)
	if err != nil {
		t.Fatal(err)
	}
	manifest := NewSegmentManifest("seg-1", SegmentSealed, []byte("payload"))
	manifest.Nodes, manifest.Dimension = 2, 3
	if err := catalog.Publish(manifest); err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenSegmentCatalog(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := reopened.Snapshot(); len(got) != 1 || got[0].SegmentID != "seg-1" {
		t.Fatalf("catalog snapshot = %+v", got)
	}
}

func TestSegmentCatalogPublishBatchIsAtomic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "catalog.json")
	catalog, err := OpenSegmentCatalog(path)
	if err != nil {
		t.Fatal(err)
	}
	first := NewSegmentManifest("seg-a", SegmentSealed, []byte("a"))
	first.Nodes, first.Dimension = 1, 2
	second := NewSegmentManifest("seg-b", SegmentPublished, []byte("bb"))
	second.Nodes, second.Dimension = 1, 2
	if err := catalog.PublishBatch([]SegmentManifest{first, second}); err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenSegmentCatalog(path)
	if err != nil {
		t.Fatal(err)
	}
	got := reopened.Snapshot()
	if len(got) != 2 || got[0].SegmentID != "seg-a" || got[1].SegmentID != "seg-b" {
		t.Fatalf("catalog snapshot = %+v", got)
	}
}

func TestSegmentCatalogPublishBatchRejectsBeforeMutation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "catalog.json")
	catalog, err := OpenSegmentCatalog(path)
	if err != nil {
		t.Fatal(err)
	}
	valid := NewSegmentManifest("seg-valid", SegmentSealed, []byte("v"))
	valid.Nodes, valid.Dimension = 1, 1
	if err := catalog.Publish(valid); err != nil {
		t.Fatal(err)
	}
	invalid := NewSegmentManifest("seg-invalid", SegmentBuilding, []byte("x"))
	invalid.Nodes, invalid.Dimension = 1, 1
	if err := catalog.PublishBatch([]SegmentManifest{valid, invalid}); err == nil {
		t.Fatal("expected invalid batch to fail")
	}
	if got := catalog.Snapshot(); len(got) != 1 || got[0].SegmentID != "seg-valid" {
		t.Fatalf("catalog mutated after rejected batch: %+v", got)
	}
}

func TestSegmentCatalogUnpublishBatchRollsBackWindow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "catalog.json")
	catalog, err := OpenSegmentCatalog(path)
	if err != nil {
		t.Fatal(err)
	}
	base := NewSegmentManifest("base", SegmentPublished, []byte("base"))
	base.Nodes, base.Dimension = 1, 1
	a := NewSegmentManifest("new-a", SegmentSealed, []byte("a"))
	a.Nodes, a.Dimension = 1, 1
	b := NewSegmentManifest("new-b", SegmentSealed, []byte("b"))
	b.Nodes, b.Dimension = 1, 1
	if err := catalog.PublishBatch([]SegmentManifest{base, a, b}); err != nil {
		t.Fatal(err)
	}
	if err := catalog.UnpublishBatch([]string{"new-a", "new-b", "missing"}); err != nil {
		t.Fatal(err)
	}
	got := catalog.Snapshot()
	if len(got) != 1 || got[0].SegmentID != "base" {
		t.Fatalf("rollback snapshot = %+v", got)
	}
}
