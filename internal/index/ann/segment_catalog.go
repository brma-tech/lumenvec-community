package ann

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

type segmentCatalogFile struct {
	Version  int               `json:"version"`
	Segments []SegmentManifest `json:"segments"`
}

// SegmentCatalog publishes immutable segment manifests through one atomic
// catalog replacement. Readers always observe a complete catalog version.
type SegmentCatalog struct {
	mu       sync.RWMutex
	path     string
	segments map[string]SegmentManifest
}

func OpenSegmentCatalog(path string) (*SegmentCatalog, error) {
	c := &SegmentCatalog{path: path, segments: make(map[string]SegmentManifest)}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return c, nil
	}
	if err != nil {
		return nil, err
	}
	var file segmentCatalogFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, err
	}
	if file.Version != segmentManifestVersion {
		return nil, fmt.Errorf("unsupported segment catalog version %d", file.Version)
	}
	for _, manifest := range file.Segments {
		if err := manifest.Validate(); err != nil {
			return nil, err
		}
		if manifest.State != SegmentSealed && manifest.State != SegmentPublished {
			continue
		}
		c.segments[manifest.SegmentID] = manifest
	}
	return c, nil
}

func (c *SegmentCatalog) Snapshot() []SegmentManifest {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]SegmentManifest, 0, len(c.segments))
	for _, manifest := range c.segments {
		out = append(out, manifest)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].SegmentID < out[j].SegmentID })
	return out
}

func (c *SegmentCatalog) Publish(manifest SegmentManifest) error {
	return c.PublishBatch([]SegmentManifest{manifest})
}

// PublishBatch publishes a set of immutable segments with one catalog
// replacement. Payloads must already be durable and manifests must be sealed
// (or published). The catalog lock is held for the whole transaction, so
// readers observe either the old catalog or the complete new set, never a
// partially published migration window.
func (c *SegmentCatalog) PublishBatch(manifests []SegmentManifest) error {
	if len(manifests) == 0 {
		return nil
	}
	for _, manifest := range manifests {
		if manifest.State != SegmentSealed && manifest.State != SegmentPublished {
			return errors.New("only sealed segments can be published")
		}
		if err := manifest.Validate(); err != nil {
			return err
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, manifest := range manifests {
		c.segments[manifest.SegmentID] = manifest
	}
	return c.persistLocked()
}

// UnpublishBatch removes a set of segment IDs with one catalog replacement.
// It is used to roll back a failed publication window; unknown IDs are
// ignored, making retries idempotent.
func (c *SegmentCatalog) UnpublishBatch(segmentIDs []string) error {
	if len(segmentIDs) == 0 {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, id := range segmentIDs {
		if id != "" {
			delete(c.segments, id)
		}
	}
	return c.persistLocked()
}

func (c *SegmentCatalog) persistLocked() error {
	segments := make([]SegmentManifest, 0, len(c.segments))
	for _, manifest := range c.segments {
		segments = append(segments, manifest)
	}
	sort.Slice(segments, func(i, j int) bool { return segments[i].SegmentID < segments[j].SegmentID })
	data, err := json.MarshalIndent(segmentCatalogFile{Version: segmentManifestVersion, Segments: segments}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(c.path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(c.path), ".segment-catalog-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err = tmp.Write(data); err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(tmpName, c.path)
}
