package ann

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSegmentManifestRoundTripAndChecksum(t *testing.T) {
	dir := t.TempDir()
	payload := []byte("immutable-segment-payload")
	manifest := NewSegmentManifest("seg-0001", SegmentSealed, payload)
	manifest.Nodes = 10
	manifest.Dimension = 4
	manifest.FirstID = 1
	manifest.LastID = 10
	path := filepath.Join(dir, "segment.json")
	if err := manifest.WriteAtomically(path); err != nil {
		t.Fatal(err)
	}
	got, err := ReadSegmentManifest(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := got.VerifyPayload(payload); err != nil {
		t.Fatal(err)
	}
	if err := got.VerifyPayload([]byte("corrupt")); err == nil {
		t.Fatal("expected checksum failure")
	}
}

func TestSegmentManifestRejectsIncompleteManifest(t *testing.T) {
	path := filepath.Join(t.TempDir(), "segment.json")
	if err := os.WriteFile(path, []byte(`{"version":1,"state":"building"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadSegmentManifest(path); err == nil {
		t.Fatal("expected validation failure")
	}
}
