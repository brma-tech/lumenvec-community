package ann

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"path/filepath"
	"time"
)

// WritePrebuiltSnapshotFile serializes one immutable ANN generation straight
// to a same-directory temporary file and atomically publishes it at path. It
// never builds a second in-memory copy of the graph, which makes it suitable
// for resharding large sealed segments.
//
// The caller must pass a sealed, live-only segment. Mutable delta indexes are
// deliberately not accepted as a migration source because their contents can
// change while the artifact is being written.
func WritePrebuiltSnapshotFile(path, segmentID, metric string, segment *AnnIndex) (SegmentManifest, error) {
	if path == "" || segmentID == "" || segment == nil {
		return SegmentManifest{}, errors.New("snapshot path, segment ID and segment are required")
	}
	manifest, err := manifestForPrebuiltSegment(segmentID, metric, segment)
	if err != nil {
		return SegmentManifest{}, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return SegmentManifest{}, err
	}
	file, err := os.CreateTemp(filepath.Dir(path), ".outgoing-ann-snapshot-*.tmp")
	if err != nil {
		return SegmentManifest{}, err
	}
	tmpPath := file.Name()
	defer os.Remove(tmpPath)
	hash := sha256.New()
	written, err := segment.WriteCompactBinary(io.MultiWriter(file, hash))
	if err == nil && written > uint64(^uint64(0)>>1) {
		err = errors.New("snapshot is too large")
	}
	if err == nil {
		err = file.Sync()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return SegmentManifest{}, err
	}
	manifest.PayloadBytes = int64(written)
	manifest.PayloadSHA256 = hex.EncodeToString(hash.Sum(nil))
	manifest.UpdatedAt = time.Now().UTC()
	if err := manifest.Validate(); err != nil {
		return SegmentManifest{}, err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return SegmentManifest{}, err
	}
	return manifest, nil
}

func manifestForPrebuiltSegment(segmentID, metric string, segment *AnnIndex) (SegmentManifest, error) {
	segment.mu.RLock()
	defer segment.mu.RUnlock()
	count := segment.nodeCountLocked()
	if count == 0 || segment.deletedCount != 0 {
		return SegmentManifest{}, errors.New("prebuilt snapshot source must contain live vectors only")
	}
	firstID, lastID := segment.nodeIDLocked(0), segment.nodeIDLocked(0)
	for slot := 1; slot < count; slot++ {
		id := segment.nodeIDLocked(slot)
		if id < firstID {
			firstID = id
		}
		if id > lastID {
			lastID = id
		}
	}
	return SegmentManifest{
		Version:        segmentManifestVersion,
		SegmentID:      segmentID,
		State:          SegmentSealed,
		FirstID:        firstID,
		LastID:         lastID,
		Nodes:          count,
		Dimension:      segment.dim,
		Metric:         metric,
		M:              segment.m,
		EfConstruction: segment.efConstruction,
	}, nil
}
