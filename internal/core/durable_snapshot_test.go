package core

import (
	"lumenvec/internal/index"
	"os"
	"path/filepath"
	"testing"
)

func TestDurableSnapshotReplacementAndPublicationFailure(t *testing.T) {
	dir := t.TempDir()
	b := newSnapshotWALBackend(filepath.Join(dir, "snapshot.json"), filepath.Join(dir, "wal"))
	for _, value := range []float64{1, 2} {
		if err := b.SaveSnapshot([]index.Vector{{ID: "one", Values: []float64{value}}}); err != nil {
			t.Fatal(err)
		}
		v, err := b.LoadSnapshot()
		if err != nil || v["one"][0] != value {
			t.Fatalf("snapshot: %v %v", v, err)
		}
	}
	// Publication must fail, not report completion when the target is a directory.
	b.snapshotPath = filepath.Join(dir, "occupied")
	if err := os.Mkdir(b.snapshotPath, 0700); err != nil {
		t.Fatal(err)
	}
	if err := b.SaveSnapshot([]index.Vector{{ID: "one", Values: []float64{3}}}); err == nil {
		t.Fatal("publication error suppressed")
	}
}
