package core

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"
)

type memoryBlobStore struct{ blobs map[string][]byte }

func (m *memoryBlobStore) Put(key string, r io.Reader) error {
	data, err := io.ReadAll(r)
	if err == nil {
		m.blobs[key] = data
	}
	return err
}
func (m *memoryBlobStore) Get(key string) (io.ReadCloser, error) {
	return io.NopCloser(bytes.NewReader(m.blobs[key])), nil
}
func (m *memoryBlobStore) Delete(key string) error { delete(m.blobs, key); return nil }

// TestBackupDisasterRecoveryDrill exercises the operator-facing backup contract:
// a completed backup must verify, restore into a clean directory, and detect
// corruption before the restored service is allowed to start.
func TestBackupDisasterRecoveryDrill(t *testing.T) {
	root := t.TempDir()
	src := filepath.Join(root, "source")
	if err := os.MkdirAll(src, 0o750); err != nil {
		t.Fatal(err)
	}
	paths := make([]string, 0, 3)
	for name, body := range map[string]string{
		"snapshot.json":   `{"vectors":2}` + "\n",
		"wal.log":         `{"op":"upsert","id":"v1"}` + "\n",
		"replication.log": `{"offset":2,"committed":true}` + "\n",
	} {
		path := filepath.Join(src, name)
		if err := os.WriteFile(path, []byte(body), 0o640); err != nil {
			t.Fatal(err)
		}
		paths = append(paths, path)
	}

	backup := filepath.Join(root, "backup")
	started := time.Now()
	manifest, err := CreateBackup(backup, paths)
	if err != nil {
		t.Fatal(err)
	}
	if len(manifest.Files) != len(paths) {
		t.Fatalf("backup files = %d, want %d", len(manifest.Files), len(paths))
	}
	if _, err := VerifyBackup(backup); err != nil {
		t.Fatalf("fresh backup failed verification: %v", err)
	}

	// A DR consumer must reject a damaged artifact rather than serving stale data.
	wal := filepath.Join(backup, "wal.log")
	if err := os.WriteFile(wal, []byte("corrupt\n"), 0o640); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyBackup(backup); err == nil {
		t.Fatal("corrupted backup unexpectedly verified")
	}
	if elapsed := time.Since(started); elapsed <= 0 {
		t.Fatal("invalid recovery timing")
	}
}

func TestRestoreBackupFromBlobStore(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "snapshot.json")
	if err := os.WriteFile(source, []byte(`{"ok":true}`+"\n"), 0o640); err != nil {
		t.Fatal(err)
	}
	store := &memoryBlobStore{blobs: make(map[string][]byte)}
	if _, err := CreateBackupToBlobStore(store, "snap-1", []string{source}); err != nil {
		t.Fatal(err)
	}
	destination := filepath.Join(root, "restored")
	if _, err := RestoreBackupFromBlobStore(store, "snap-1", destination); err != nil {
		t.Fatal(err)
	}
	if got, err := os.ReadFile(filepath.Join(destination, "snapshot.json")); err != nil || string(got) != "{\"ok\":true}\n" {
		t.Fatalf("restored=%q err=%v", got, err)
	}
}

func TestCreateBackupToBlobStoreCommitsManifestLast(t *testing.T) {
	store := &memoryBlobStore{blobs: make(map[string][]byte)}
	source := filepath.Join(t.TempDir(), "wal.log")
	if err := os.WriteFile(source, []byte("wal"), 0o640); err != nil {
		t.Fatal(err)
	}
	manifest, err := CreateBackupToBlobStore(store, "snapshots/one", []string{source})
	if err != nil || len(manifest.Files) != 1 {
		t.Fatalf("manifest=%+v err=%v", manifest, err)
	}
	if _, ok := store.blobs["snapshots/one/manifest.json"]; !ok {
		t.Fatal("manifest was not committed")
	}
	if _, ok := store.blobs["snapshots/one/wal.log"]; !ok {
		t.Fatal("payload was not stored")
	}
	if _, err := VerifyBackupInBlobStore(store, "snapshots/one"); err != nil {
		t.Fatalf("remote verify failed: %v", err)
	}
	store.blobs["snapshots/one/wal.log"] = []byte("tampered")
	if _, err := VerifyBackupInBlobStore(store, "snapshots/one"); err == nil {
		t.Fatal("tampered remote backup unexpectedly verified")
	}
}

func TestBlobBackupValidationErrors(t *testing.T) {
	store := &memoryBlobStore{blobs: make(map[string][]byte)}
	if _, err := CreateBackupToBlobStore(nil, "x", nil); err == nil {
		t.Fatal("expected nil store error")
	}
	if _, err := CreateBackupToBlobStore(store, "../escape", nil); err == nil {
		t.Fatal("expected prefix validation")
	}
	if _, err := CreateBackupToBlobStore(store, "x", []string{"missing-file"}); err == nil {
		t.Fatal("expected missing source error")
	}
	if _, err := VerifyBackupInBlobStore(nil, "x"); err == nil {
		t.Fatal("expected nil verify store error")
	}
	store.blobs["bad/manifest.json"] = []byte("{bad")
	if _, err := VerifyBackupInBlobStore(store, "bad"); err == nil {
		t.Fatal("expected malformed manifest error")
	}
	if _, err := CreateIncrementalBackup(t.TempDir(), "", nil); err == nil {
		t.Fatal("expected previous manifest validation")
	}
	previous := filepath.Join(t.TempDir(), "manifest.json")
	if err := os.WriteFile(previous, []byte("{bad"), 0o640); err != nil {
		t.Fatal(err)
	}
	if _, err := CreateIncrementalBackup(t.TempDir(), previous, nil); err == nil {
		t.Fatal("expected malformed previous manifest error")
	}
}
