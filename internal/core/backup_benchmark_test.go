package core

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkBackupRestoreFull(b *testing.B) {
	root := b.TempDir()
	paths := backupBenchmarkFiles(b, root, 2000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dir := filepath.Join(root, fmt.Sprintf("full-%d", i))
		if _, err := CreateBackup(dir, paths); err != nil {
			b.Fatal(err)
		}
		if _, err := RestoreBackup(filepath.Join(dir, "manifest.json"), filepath.Join(dir, "restore")); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBackupRestoreIncremental(b *testing.B) {
	root := b.TempDir()
	paths := backupBenchmarkFiles(b, root, 2000)
	base := filepath.Join(root, "base")
	if _, err := CreateBackup(base, paths); err != nil {
		b.Fatal(err)
	}
	if err := os.WriteFile(paths[0], []byte("delta\n"), 0o640); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dir := filepath.Join(root, fmt.Sprintf("delta-%d", i))
		if _, err := CreateIncrementalBackup(dir, filepath.Join(base, "manifest.json"), paths); err != nil {
			b.Fatal(err)
		}
		if _, err := RestoreBackup(filepath.Join(dir, "manifest.json"), filepath.Join(dir, "restore")); err != nil {
			b.Fatal(err)
		}
	}
}

func backupBenchmarkFiles(b *testing.B, root string, count int) []string {
	b.Helper()
	source := filepath.Join(root, "source")
	if err := os.MkdirAll(source, 0o750); err != nil {
		b.Fatal(err)
	}
	paths := make([]string, 0, count)
	for i := 0; i < count; i++ {
		path := filepath.Join(source, fmt.Sprintf("segment-%04d.bin", i))
		if err := os.WriteFile(path, []byte(fmt.Sprintf("segment-%d\n", i)), 0o640); err != nil {
			b.Fatal(err)
		}
		paths = append(paths, path)
	}
	return paths
}
