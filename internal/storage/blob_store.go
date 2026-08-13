package storage

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// BlobStore is the minimal object-storage contract used by durable snapshots
// and immutable segments. Implementations may target a local filesystem or an
// object store; callers never depend on provider-specific APIs.
type BlobStore interface {
	Put(key string, r io.Reader) error
	Get(key string) (io.ReadCloser, error)
	Delete(key string) error
}

// LocalBlobStore provides an atomic, crash-safe development backend. Put
// writes a temporary sibling and renames it only after the complete payload
// has been flushed and closed.
type LocalBlobStore struct{ root string }

func NewLocalBlobStore(root string) (*LocalBlobStore, error) {
	if strings.TrimSpace(root) == "" {
		return nil, fmt.Errorf("blob store root is required")
	}
	root, err := filepath.Abs(filepath.Clean(root))
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(root, 0o750); err != nil {
		return nil, err
	}
	return &LocalBlobStore{root: root}, nil
}

func (s *LocalBlobStore) path(key string) (string, error) {
	key = filepath.Clean(key)
	if key == "." || filepath.IsAbs(key) || strings.HasPrefix(key, "/") || strings.HasPrefix(key, "\\") || key == ".." || strings.HasPrefix(key, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("invalid blob key")
	}
	path := filepath.Join(s.root, key)
	rel, err := filepath.Rel(s.root, path)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("invalid blob key")
	}
	return path, nil
}

func (s *LocalBlobStore) Put(key string, r io.Reader) error {
	path, err := s.path(key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".blob-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := io.Copy(tmp, r); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

func (s *LocalBlobStore) Get(key string) (io.ReadCloser, error) {
	path, err := s.path(key)
	if err != nil {
		return nil, err
	}
	// The key is normalized and confined to s.root by path above.
	return os.Open(path) // #nosec G304 -- validated relative key under configured root
}

func (s *LocalBlobStore) Delete(key string) error {
	path, err := s.path(key)
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}
