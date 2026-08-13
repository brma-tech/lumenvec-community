package core

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"lumenvec/internal/storage"
)

type BackupFile struct {
	Name   string `json:"name"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

// CreateBackupToBlobStore writes durable files and its manifest to a provider
// neutral blob store. The manifest is written last, so its presence is the
// commit marker for remote backup consumers.
func CreateBackupToBlobStore(store storage.BlobStore, prefix string, paths []string) (BackupManifest, error) {
	if store == nil {
		return BackupManifest{}, fmt.Errorf("blob store is required")
	}
	prefix = filepath.Clean(prefix)
	if prefix == "." || filepath.IsAbs(prefix) || strings.HasPrefix(prefix, "..") {
		return BackupManifest{}, fmt.Errorf("invalid backup prefix")
	}
	manifest := BackupManifest{Version: 1}
	for _, source := range paths {
		info, err := os.Stat(source)
		if err != nil {
			return BackupManifest{}, err
		}
		if !info.Mode().IsRegular() {
			return BackupManifest{}, fmt.Errorf("backup source is not a regular file: %s", source)
		}
		data, err := os.ReadFile(source)
		if err != nil {
			return BackupManifest{}, err
		}
		hash := sha256.Sum256(data)
		key := filepath.ToSlash(filepath.Join(prefix, filepath.Base(source)))
		if err := store.Put(key, bytes.NewReader(data)); err != nil {
			return BackupManifest{}, err
		}
		manifest.Files = append(manifest.Files, BackupFile{Name: key, Bytes: int64(len(data)), SHA256: hex.EncodeToString(hash[:])})
	}
	sort.Slice(manifest.Files, func(i, j int) bool { return manifest.Files[i].Name < manifest.Files[j].Name })
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return BackupManifest{}, err
	}
	if err := store.Put(filepath.ToSlash(filepath.Join(prefix, "manifest.json")), bytes.NewReader(append(data, '\n'))); err != nil {
		return BackupManifest{}, err
	}
	return manifest, nil
}

// VerifyBackupInBlobStore validates a remote manifest and every referenced
// payload before restore or promotion.
func VerifyBackupInBlobStore(store storage.BlobStore, prefix string) (BackupManifest, error) {
	if store == nil {
		return BackupManifest{}, fmt.Errorf("blob store is required")
	}
	prefix = filepath.ToSlash(filepath.Clean(prefix))
	manifestReader, err := store.Get(filepath.ToSlash(filepath.Join(prefix, "manifest.json")))
	if err != nil {
		return BackupManifest{}, err
	}
	data, err := io.ReadAll(manifestReader)
	_ = manifestReader.Close()
	if err != nil {
		return BackupManifest{}, err
	}
	var manifest BackupManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return BackupManifest{}, err
	}
	if manifest.Version != 1 {
		return BackupManifest{}, fmt.Errorf("unsupported backup manifest version %d", manifest.Version)
	}
	for _, file := range manifest.Files {
		reader, err := store.Get(file.Name)
		if err != nil {
			return BackupManifest{}, err
		}
		hash := sha256.New()
		n, copyErr := io.Copy(hash, reader)
		_ = reader.Close()
		if copyErr != nil {
			return BackupManifest{}, copyErr
		}
		if n != file.Bytes || hex.EncodeToString(hash.Sum(nil)) != file.SHA256 {
			return BackupManifest{}, fmt.Errorf("backup checksum mismatch: %s", file.Name)
		}
	}
	return manifest, nil
}

// RestoreBackupFromBlobStore verifies the committed remote manifest and
// materializes its files locally using atomic renames.
func RestoreBackupFromBlobStore(store storage.BlobStore, prefix, destination string) (BackupManifest, error) {
	if store == nil || strings.TrimSpace(destination) == "" {
		return BackupManifest{}, fmt.Errorf("blob store and destination are required")
	}
	manifest, err := VerifyBackupInBlobStore(store, prefix)
	if err != nil {
		return BackupManifest{}, err
	}
	if err := os.MkdirAll(destination, 0o750); err != nil {
		return BackupManifest{}, err
	}
	for _, file := range manifest.Files {
		cleanName := filepath.Clean(filepath.FromSlash(file.Name))
		if filepath.IsAbs(cleanName) || cleanName == ".." || strings.HasPrefix(cleanName, ".."+string(filepath.Separator)) {
			return BackupManifest{}, fmt.Errorf("invalid remote backup file name: %s", file.Name)
		}
		reader, err := store.Get(file.Name)
		if err != nil {
			return BackupManifest{}, err
		}
		tmp := filepath.Join(destination, filepath.Base(file.Name)+".tmp")
		out, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
		if err != nil {
			_ = reader.Close()
			return BackupManifest{}, err
		}
		_, copyErr := io.Copy(out, reader)
		_ = reader.Close()
		closeErr := out.Close()
		if copyErr == nil {
			copyErr = closeErr
		}
		if copyErr != nil {
			_ = os.Remove(tmp)
			return BackupManifest{}, copyErr
		}
		if err := os.Rename(tmp, filepath.Join(destination, filepath.Base(file.Name))); err != nil {
			_ = os.Remove(tmp)
			return BackupManifest{}, err
		}
	}
	return manifest, nil
}

type BackupManifest struct {
	Version int `json:"version"`
	// Base identifies the prior manifest for an incremental backup. A restore
	// must apply Base first, then the files listed here.
	Base  string       `json:"base,omitempty"`
	Files []BackupFile `json:"files"`
}

// CreateBackup copies a set of durable files into a new directory and writes
// a checksum manifest last. Callers should include snapshot, WAL, replication
// journals and ANN checkpoints for every shard.
func CreateBackup(destination string, paths []string) (BackupManifest, error) {
	return createBackup(destination, paths, "")
}

// CreateIncrementalBackup copies only files whose content differs from the
// previous manifest. It is safe to use with a new destination: unchanged
// files remain addressed by Base and are intentionally not duplicated.
func CreateIncrementalBackup(destination, previousManifest string, paths []string) (BackupManifest, error) {
	if previousManifest == "" {
		return BackupManifest{}, fmt.Errorf("previous manifest is required")
	}
	data, err := os.ReadFile(previousManifest)
	if err != nil {
		return BackupManifest{}, err
	}
	var previous BackupManifest
	if err := json.Unmarshal(data, &previous); err != nil {
		return BackupManifest{}, fmt.Errorf("decode previous manifest: %w", err)
	}
	known := make(map[string]string, len(previous.Files))
	for _, file := range previous.Files {
		known[file.Name] = file.SHA256
	}
	changed := make([]string, 0, len(paths))
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return BackupManifest{}, err
		}
		hash := sha256.Sum256(data)
		if known[filepath.Base(path)] != hex.EncodeToString(hash[:]) {
			changed = append(changed, path)
		}
	}
	return createBackup(destination, changed, previousManifest)
}

func createBackup(destination string, paths []string, base string) (BackupManifest, error) {
	if destination == "" {
		return BackupManifest{}, fmt.Errorf("backup destination is required")
	}
	if err := os.MkdirAll(destination, 0o750); err != nil {
		return BackupManifest{}, err
	}
	manifest := BackupManifest{Version: 1, Base: base}
	for _, source := range paths {
		info, err := os.Stat(source)
		if err != nil {
			return BackupManifest{}, err
		}
		if !info.Mode().IsRegular() {
			return BackupManifest{}, fmt.Errorf("backup source is not a regular file: %s", source)
		}
		name := filepath.Base(source)
		target := filepath.Join(destination, name+".tmp")
		out, err := os.Create(target)
		if err != nil {
			return BackupManifest{}, err
		}
		in, err := os.Open(source)
		if err != nil {
			_ = out.Close()
			return BackupManifest{}, err
		}
		hash := sha256.New()
		written, copyErr := io.Copy(io.MultiWriter(out, hash), in)
		_ = in.Close()
		if closeErr := out.Close(); copyErr == nil {
			copyErr = closeErr
		}
		if copyErr != nil {
			_ = os.Remove(target)
			return BackupManifest{}, copyErr
		}
		final := filepath.Join(destination, name)
		if err := os.Rename(target, final); err != nil {
			_ = os.Remove(target)
			return BackupManifest{}, err
		}
		manifest.Files = append(manifest.Files, BackupFile{Name: name, Bytes: written, SHA256: hex.EncodeToString(hash.Sum(nil))})
	}
	sort.Slice(manifest.Files, func(i, j int) bool { return manifest.Files[i].Name < manifest.Files[j].Name })
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return BackupManifest{}, err
	}
	if err := os.WriteFile(filepath.Join(destination, "manifest.json"), append(data, '\n'), 0o600); err != nil {
		return BackupManifest{}, err
	}
	return manifest, nil
}

func VerifyBackup(destination string) (BackupManifest, error) {
	data, err := os.ReadFile(filepath.Join(destination, "manifest.json"))
	if err != nil {
		return BackupManifest{}, err
	}
	var manifest BackupManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return BackupManifest{}, err
	}
	if manifest.Version != 1 {
		return BackupManifest{}, fmt.Errorf("unsupported backup manifest version %d", manifest.Version)
	}
	for _, file := range manifest.Files {
		path := filepath.Join(destination, file.Name)
		info, err := os.Stat(path)
		if err != nil || !info.Mode().IsRegular() {
			return BackupManifest{}, fmt.Errorf("backup file missing: %s", file.Name)
		}
		f, err := os.Open(path)
		if err != nil {
			return BackupManifest{}, err
		}
		hash := sha256.New()
		n, copyErr := io.Copy(hash, f)
		_ = f.Close()
		if copyErr != nil {
			return BackupManifest{}, copyErr
		}
		if n != file.Bytes || hex.EncodeToString(hash.Sum(nil)) != file.SHA256 {
			return BackupManifest{}, fmt.Errorf("backup checksum mismatch: %s", file.Name)
		}
	}
	return manifest, nil
}

// RestoreBackup materializes a full backup from a manifest chain. Base
// manifests are restored first, then changed files override them atomically.
// The source manifest and every payload are verified before they are copied.
func RestoreBackup(manifestPath, destination string) (BackupManifest, error) {
	if manifestPath == "" || destination == "" {
		return BackupManifest{}, fmt.Errorf("manifest and destination are required")
	}
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return BackupManifest{}, err
	}
	var manifest BackupManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return BackupManifest{}, err
	}
	if manifest.Version != 1 {
		return BackupManifest{}, fmt.Errorf("unsupported backup manifest version %d", manifest.Version)
	}
	if manifest.Base != "" {
		if _, err := RestoreBackup(manifest.Base, destination); err != nil {
			return BackupManifest{}, err
		}
	}
	if err := os.MkdirAll(destination, 0o750); err != nil {
		return BackupManifest{}, err
	}
	baseDir := filepath.Dir(manifestPath)
	for _, file := range manifest.Files {
		if filepath.Base(file.Name) != file.Name || file.Name == "." || file.Name == ".." {
			return BackupManifest{}, fmt.Errorf("invalid backup file name: %s", file.Name)
		}
		source := filepath.Join(baseDir, file.Name)
		in, err := os.Open(source)
		if err != nil {
			return BackupManifest{}, err
		}
		hash := sha256.New()
		tmp := filepath.Join(destination, file.Name+".tmp")
		out, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
		if err != nil {
			_ = in.Close()
			return BackupManifest{}, err
		}
		n, copyErr := io.Copy(io.MultiWriter(out, hash), in)
		_ = in.Close()
		closeErr := out.Close()
		if copyErr == nil {
			copyErr = closeErr
		}
		if copyErr != nil {
			_ = os.Remove(tmp)
			return BackupManifest{}, copyErr
		}
		if n != file.Bytes || hex.EncodeToString(hash.Sum(nil)) != file.SHA256 {
			_ = os.Remove(tmp)
			return BackupManifest{}, fmt.Errorf("backup checksum mismatch: %s", file.Name)
		}
		if err := os.Rename(tmp, filepath.Join(destination, file.Name)); err != nil {
			_ = os.Remove(tmp)
			return BackupManifest{}, err
		}
	}
	return manifest, nil
}
