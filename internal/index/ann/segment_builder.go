package ann

import (
	"errors"
	"os"
	"path/filepath"
)

// SegmentBuildSpec describes one immutable segment in a publication window.
// The writer must only write through the supplied payload writer.
type SegmentBuildSpec struct {
	PayloadPath    string
	ManifestPath   string
	SegmentID      string
	Dimension      int
	M              int
	EfConstruction int
	FirstID        int
	LastID         int
	Metric         string
	Write          func(*SegmentPayloadWriter) error
}

// SnapshotBuildSpec is a completed immutable ANN graph waiting to be made
// durable. Unlike SegmentBuildSpec, it does not rebuild HNSW from vectors;
// the graph can be created on an isolated worker and transferred as-is.
type SnapshotBuildSpec struct {
	SnapshotPath string
	ManifestPath string
	SegmentID    string
	Metric       string
	Snapshot     []byte
}

// PublishPrebuiltSnapshots validates every graph, fsyncs every artifact and
// manifest, then swaps the full catalog in one operation. A failed window
// leaves the catalog unchanged, so readers never discover a partial reshard
// generation after a crash.
func PublishPrebuiltSnapshots(catalog *SegmentCatalog, specs []SnapshotBuildSpec) ([]SegmentManifest, error) {
	if catalog == nil || len(specs) == 0 {
		return nil, errors.New("catalog and snapshot specs are required")
	}
	manifests := make([]SegmentManifest, 0, len(specs))
	seen := make(map[string]struct{}, len(specs))
	for _, spec := range specs {
		if spec.SegmentID == "" || len(spec.Snapshot) == 0 {
			return nil, errors.New("segment ID and snapshot are required")
		}
		if _, exists := seen[spec.SegmentID]; exists {
			return nil, errors.New("duplicate segment ID in snapshot publication window")
		}
		seen[spec.SegmentID] = struct{}{}
		segment, err := UnmarshalCompactBinary(spec.Snapshot)
		if err != nil {
			segment, err = UnmarshalBinary(spec.Snapshot)
		}
		if err != nil {
			return nil, err
		}
		stats := segment.Stats()
		if stats.Nodes == 0 || stats.Deleted != 0 {
			return nil, errors.New("prebuilt snapshot must contain live vectors only")
		}
		segment.mu.RLock()
		firstID, lastID := 0, 0
		for i, node := range segment.nodes {
			if i == 0 || node.id < firstID {
				firstID = node.id
			}
			if i == 0 || node.id > lastID {
				lastID = node.id
			}
		}
		manifest := SegmentManifest{Version: segmentManifestVersion, SegmentID: spec.SegmentID, State: SegmentSealed,
			FirstID: firstID, LastID: lastID, Nodes: stats.Nodes, Dimension: segment.dim, Metric: spec.Metric,
			M: segment.m, EfConstruction: segment.efConstruction}
		segment.mu.RUnlock()
		if err := writeArtifactAtomically(spec.SnapshotPath, spec.Snapshot); err != nil {
			return nil, err
		}
		info, err := os.Stat(spec.SnapshotPath)
		if err != nil {
			return nil, err
		}
		manifest.PayloadBytes = info.Size()
		// Verify from the durable file rather than trusting the in-memory input.
		raw, err := os.ReadFile(spec.SnapshotPath)
		if err != nil {
			return nil, err
		}
		manifest = NewSegmentManifest(spec.SegmentID, SegmentSealed, raw)
		manifest.FirstID, manifest.LastID, manifest.Nodes, manifest.Dimension = firstID, lastID, stats.Nodes, segment.dim
		manifest.Metric, manifest.M, manifest.EfConstruction = spec.Metric, segment.m, segment.efConstruction
		if err := manifest.WriteAtomically(spec.ManifestPath); err != nil {
			return nil, err
		}
		manifests = append(manifests, manifest)
	}
	if err := catalog.PublishBatch(manifests); err != nil {
		return nil, err
	}
	return manifests, nil
}

func writeArtifactAtomically(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".ann-snapshot-*.tmp")
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
	return os.Rename(tmpName, path)
}

// BuildAndPublishSegment creates a sealed payload and publishes its manifest
// only after both files have been durably written.
func BuildAndPublishSegment(catalog *SegmentCatalog, payloadPath, manifestPath, segmentID string, dimension, m, efConstruction int, firstID, lastID int, metric string, write func(*SegmentPayloadWriter) error) (SegmentManifest, error) {
	if catalog == nil || segmentID == "" || write == nil {
		return SegmentManifest{}, errors.New("catalog, segment ID and writer callback are required")
	}
	header, checksum, err := WriteSegmentPayloadFile(payloadPath, dimension, write)
	if err != nil {
		return SegmentManifest{}, err
	}
	manifest := SegmentManifest{
		Version:        segmentManifestVersion,
		SegmentID:      segmentID,
		State:          SegmentSealed,
		FirstID:        firstID,
		LastID:         lastID,
		Nodes:          int(header.Count),
		Dimension:      int(header.Dimension),
		Metric:         metric,
		M:              m,
		EfConstruction: efConstruction,
		PayloadSHA256:  checksum,
	}
	if info, statErr := os.Stat(payloadPath); statErr != nil {
		return SegmentManifest{}, statErr
	} else {
		manifest.PayloadBytes = info.Size()
	}
	if err := manifest.Validate(); err != nil {
		return SegmentManifest{}, err
	}
	if err := manifest.WriteAtomically(manifestPath); err != nil {
		return SegmentManifest{}, err
	}
	if err := catalog.Publish(manifest); err != nil {
		return SegmentManifest{}, err
	}
	return manifest, nil
}

// BuildAndPublishSegments builds a complete immutable publication window and
// publishes all manifests atomically after every payload and manifest is
// durable. If a build fails, no new segment is added to the catalog.
func BuildAndPublishSegments(catalog *SegmentCatalog, specs []SegmentBuildSpec) ([]SegmentManifest, error) {
	if catalog == nil || len(specs) == 0 {
		return nil, errors.New("catalog and segment specs are required")
	}
	manifests := make([]SegmentManifest, 0, len(specs))
	seen := make(map[string]struct{}, len(specs))
	for _, spec := range specs {
		if spec.SegmentID == "" || spec.Write == nil {
			return nil, errors.New("segment ID and writer callback are required")
		}
		if _, ok := seen[spec.SegmentID]; ok {
			return nil, errors.New("duplicate segment ID in publication window")
		}
		seen[spec.SegmentID] = struct{}{}
		header, checksum, err := WriteSegmentPayloadFile(spec.PayloadPath, spec.Dimension, spec.Write)
		if err != nil {
			return nil, err
		}
		manifest := SegmentManifest{Version: segmentManifestVersion, SegmentID: spec.SegmentID, State: SegmentSealed,
			FirstID: spec.FirstID, LastID: spec.LastID, Nodes: int(header.Count), Dimension: int(header.Dimension),
			Metric: spec.Metric, M: spec.M, EfConstruction: spec.EfConstruction, PayloadSHA256: checksum}
		info, err := os.Stat(spec.PayloadPath)
		if err != nil {
			return nil, err
		}
		manifest.PayloadBytes = info.Size()
		if err := manifest.Validate(); err != nil {
			return nil, err
		}
		if err := manifest.WriteAtomically(spec.ManifestPath); err != nil {
			return nil, err
		}
		manifests = append(manifests, manifest)
	}
	if err := catalog.PublishBatch(manifests); err != nil {
		return nil, err
	}
	return manifests, nil
}
