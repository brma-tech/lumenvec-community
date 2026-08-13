package ann

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"os"
	"path/filepath"
)

// SnapshotStager receives one prebuilt ANN artifact incrementally. It is the
// file-backed boundary used by future gRPC/object-store transports: memory is
// bounded by the incoming chunk while checksum and size are verified before a
// same-directory rename makes the artifact visible to a catalog.
type SnapshotStager struct {
	manifest SegmentManifest
	path     string
	tmpPath  string
	file     *os.File
	hash     hash.Hash
	written  int64
	closed   bool
}

// NewSnapshotStager creates an unpublished temporary artifact alongside path.
// The manifest is the source of truth for the exact expected size and digest.
func NewSnapshotStager(path string, manifest SegmentManifest) (*SnapshotStager, error) {
	if err := manifest.Validate(); err != nil {
		return nil, err
	}
	if path == "" {
		return nil, errors.New("snapshot path is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	file, err := os.CreateTemp(filepath.Dir(path), ".incoming-ann-snapshot-*.tmp")
	if err != nil {
		return nil, err
	}
	return &SnapshotStager{manifest: manifest, path: path, tmpPath: file.Name(), file: file, hash: sha256.New()}, nil
}

func (s *SnapshotStager) Write(chunk []byte) (int, error) {
	if s == nil || s.file == nil || s.closed {
		return 0, errors.New("snapshot stager is closed")
	}
	if int64(len(chunk)) > s.manifest.PayloadBytes-s.written {
		return 0, errors.New("snapshot exceeds manifest size")
	}
	n, err := s.file.Write(chunk)
	if n > 0 {
		_, _ = s.hash.Write(chunk[:n])
		s.written += int64(n)
	}
	if err == nil && n != len(chunk) {
		err = io.ErrShortWrite
	}
	return n, err
}

// Commit fsyncs, verifies and atomically publishes the staged artifact. Any
// invalid or interrupted transfer leaves no destination file behind.
func (s *SnapshotStager) Commit() error {
	if s == nil || s.file == nil || s.closed {
		return errors.New("snapshot stager is closed")
	}
	if s.written != s.manifest.PayloadBytes {
		s.Abort()
		return fmt.Errorf("snapshot size mismatch: got %d want %d", s.written, s.manifest.PayloadBytes)
	}
	if hex.EncodeToString(s.hash.Sum(nil)) != s.manifest.PayloadSHA256 {
		s.Abort()
		return errors.New("snapshot checksum mismatch")
	}
	if err := s.file.Sync(); err != nil {
		s.Abort()
		return err
	}
	if err := s.file.Close(); err != nil {
		s.file = nil
		s.closed = true
		_ = os.Remove(s.tmpPath)
		return err
	}
	s.file = nil
	s.closed = true
	if _, statErr := os.Stat(s.path); statErr == nil {
		if err := s.manifest.VerifyArtifactFile(s.path); err != nil {
			_ = os.Remove(s.tmpPath)
			return errors.New("snapshot destination already exists with different content")
		}
		// A retry of an already committed transfer is complete. Keep the
		// existing durable inode and discard the duplicate temporary file.
		_ = os.Remove(s.tmpPath)
		return nil
	} else if !os.IsNotExist(statErr) {
		_ = os.Remove(s.tmpPath)
		return statErr
	}
	if err := os.Rename(s.tmpPath, s.path); err != nil {
		_ = os.Remove(s.tmpPath)
		return err
	}
	return nil
}

// Abort discards a partial artifact. It is safe to call repeatedly.
func (s *SnapshotStager) Abort() {
	if s == nil || s.closed {
		return
	}
	if s.file != nil {
		_ = s.file.Close()
		s.file = nil
	}
	s.closed = true
	_ = os.Remove(s.tmpPath)
}

var _ io.Writer = (*SnapshotStager)(nil)
