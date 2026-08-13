package core

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCreateAndVerifyBackup(t *testing.T) {
	if _, err := CreateBackup("", nil); err == nil {
		t.Fatal("expected destination validation")
	}
	base := t.TempDir()
	source := filepath.Join(base, "source")
	destination := filepath.Join(base, "backup")
	if err := os.WriteFile(source, []byte("durable state"), 0o640); err != nil {
		t.Fatal(err)
	}
	manifest, err := CreateBackup(destination, []string{source})
	if err != nil || len(manifest.Files) != 1 {
		t.Fatalf("manifest=%+v err=%v", manifest, err)
	}
	if _, err := VerifyBackup(destination); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(destination, "source"), []byte("tampered"), 0o640); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyBackup(destination); err == nil {
		t.Fatal("expected checksum failure")
	}
	if _, err := CreateBackup(filepath.Join(base, "missing-source"), []string{filepath.Join(base, "nope")}); err == nil {
		t.Fatal("expected missing source error")
	}
	if _, err := VerifyBackup(filepath.Join(base, "no-backup")); err == nil {
		t.Fatal("expected missing manifest error")
	}
}

func TestRestoreBackupAppliesIncrementalChain(t *testing.T) {
	root := t.TempDir()
	a := filepath.Join(root, "a.data")
	b := filepath.Join(root, "b.data")
	if err := os.WriteFile(a, []byte("a1"), 0o640); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(b, []byte("b1"), 0o640); err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(root, "base")
	if _, err := CreateBackup(base, []string{a, b}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(b, []byte("b2"), 0o640); err != nil {
		t.Fatal(err)
	}
	delta := filepath.Join(root, "delta")
	if _, err := CreateIncrementalBackup(delta, filepath.Join(base, "manifest.json"), []string{a, b}); err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(root, "restore")
	if _, err := RestoreBackup(filepath.Join(delta, "manifest.json"), target); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(filepath.Join(target, "a.data")); string(got) != "a1" {
		t.Fatalf("a=%q", got)
	}
	if got, _ := os.ReadFile(filepath.Join(target, "b.data")); string(got) != "b2" {
		t.Fatalf("b=%q", got)
	}
}

func TestCreateIncrementalBackupCopiesOnlyChangedFiles(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source")
	if err := os.MkdirAll(source, 0o750); err != nil {
		t.Fatal(err)
	}
	a := filepath.Join(source, "a.snapshot")
	b := filepath.Join(source, "b.wal")
	if err := os.WriteFile(a, []byte("a1"), 0o640); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(b, []byte("b1"), 0o640); err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(root, "base")
	if _, err := CreateBackup(base, []string{a, b}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(b, []byte("b2"), 0o640); err != nil {
		t.Fatal(err)
	}
	incremental := filepath.Join(root, "incremental")
	manifest, err := CreateIncrementalBackup(incremental, filepath.Join(base, "manifest.json"), []string{a, b})
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Base == "" || len(manifest.Files) != 1 || manifest.Files[0].Name != filepath.Base(b) {
		t.Fatalf("unexpected incremental manifest: %+v", manifest)
	}
	if _, err := VerifyBackup(incremental); err != nil {
		t.Fatalf("incremental verification failed: %v", err)
	}
}
