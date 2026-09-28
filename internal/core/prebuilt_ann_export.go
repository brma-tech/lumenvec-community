package core

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"lumenvec/internal/index/ann"
)

// PrebuiltANNSegmentExport is one immutable graph plus a rewindable stream of
// the stable identities and canonical vectors referenced by that graph.
type PrebuiltANNSegmentExport struct {
	Path     string
	Manifest ann.SegmentManifest
	Replay   PrebuiltANNReplay
}

// PrebuiltANNStateEvent is the atomic export stream consumed by remote
// resharding. Epoch and delta offset identify the exact source view.
type PrebuiltANNStateEvent struct {
	Kind         int // 0 begin, 1 sealed record, 2 end segment, 3 tail record, 4 end
	Epoch        uint64
	DeltaOffset  uint64
	SegmentIndex int
	Record       PrebuiltANNRecord
	Cardinality  uint64
}

// ExportANNState streams one consistent sealed+tail view. SnapshotExport holds
// the ingestion barrier while selecting immutable segments and copying the
// tail, preventing a rotation from duplicating or omitting a vector.
func (s *Service) ExportANNState(visit func(PrebuiltANNStateEvent) error) error {
	if visit == nil {
		return errors.New("ANN state visitor is required")
	}
	s.ensureRuntimeDeps()
	segmented, ok := s.currentANNIndex().(*ann.SegmentedIndex)
	if !ok {
		return errors.New("versioned ANN export requires a segmented ANN index")
	}
	s.persistMu.Lock()
	snapshot := segmented.SnapshotExport()
	offset := s.currentDeltaOffsetUnlocked()
	s.persistMu.Unlock()
	defer snapshot.Release()
	if err := visit(PrebuiltANNStateEvent{Kind: 0, Epoch: snapshot.Epoch, DeltaOffset: offset}); err != nil {
		return err
	}
	var cardinality uint64
	for segmentIndex, segment := range snapshot.Segments {
		if err := segment.VisitLiveVectors32(func(internalID int, values32 []float32) error {
			externalID, found, err := s.lookupID(internalID)
			if err != nil {
				return err
			}
			if !found {
				return fmt.Errorf("sealed ANN segment references unmapped internal ID %d", internalID)
			}
			values := make([]float64, len(values32))
			for i, value := range values32 {
				values[i] = float64(value)
			}
			cardinality++
			return visit(PrebuiltANNStateEvent{Kind: 1, Epoch: snapshot.Epoch, DeltaOffset: offset, SegmentIndex: segmentIndex, Record: PrebuiltANNRecord{ID: externalID, InternalID: internalID, Values: values}})
		}); err != nil {
			return err
		}
		if err := visit(PrebuiltANNStateEvent{Kind: 2, Epoch: snapshot.Epoch, DeltaOffset: offset, SegmentIndex: segmentIndex}); err != nil {
			return err
		}
	}
	for _, vector := range snapshot.Tail {
		externalID, found, err := s.lookupID(vector.ID)
		if err != nil {
			return err
		}
		if !found {
			return fmt.Errorf("mutable ANN tail references unmapped internal ID %d", vector.ID)
		}
		values := make([]float64, len(vector.Values))
		for i, value := range vector.Values {
			values[i] = float64(value)
		}
		cardinality++
		if err := visit(PrebuiltANNStateEvent{Kind: 3, Epoch: snapshot.Epoch, DeltaOffset: offset, Record: PrebuiltANNRecord{ID: externalID, InternalID: vector.ID, Values: values}}); err != nil {
			return err
		}
	}
	return visit(PrebuiltANNStateEvent{Kind: 4, Epoch: snapshot.Epoch, DeltaOffset: offset, Cardinality: cardinality})
}

// VisitSealedANNRecords exposes immutable generations without serializing the
// source graph. Target-specific graphs are built only for records whose owner
// changes under the next routing table.
func (s *Service) VisitSealedANNRecords(visit func(segmentIndex int, replay PrebuiltANNReplay) error) error {
	if visit == nil {
		return errors.New("sealed ANN visitor is required")
	}
	s.ensureRuntimeDeps()
	segmented, ok := s.currentANNIndex().(*ann.SegmentedIndex)
	if !ok {
		return errors.New("sealed ANN export requires a segmented ANN index")
	}
	return segmented.VisitSealedSegments(func(segmentIndex int, segment *ann.AnnIndex) error {
		replay := func(yield func(PrebuiltANNRecord) error) error {
			return segment.VisitLiveVectors32(func(internalID int, values32 []float32) error {
				externalID, found, err := s.lookupID(internalID)
				if err != nil {
					return err
				}
				if !found {
					return fmt.Errorf("sealed ANN segment references unmapped internal ID %d", internalID)
				}
				values := make([]float64, len(values32))
				for i, value := range values32 {
					values[i] = float64(value)
				}
				return yield(PrebuiltANNRecord{ID: externalID, InternalID: internalID, Values: values})
			})
		}
		return visit(segmentIndex, replay)
	})
}

// ExportSealedANNSegments writes each sealed source generation as a compact
// file and exposes its records through a replay callback. Memory is bounded by
// one vector conversion; the complete segment is never copied into a slice.
func (s *Service) ExportSealedANNSegments(dir, generation, metric string, visit func(PrebuiltANNSegmentExport) error) error {
	if strings.TrimSpace(dir) == "" || !safeSegmentGeneration(generation) || visit == nil {
		return errors.New("export directory, safe generation and visitor are required")
	}
	s.ensureRuntimeDeps()
	segmented, ok := s.currentANNIndex().(*ann.SegmentedIndex)
	if !ok {
		return errors.New("sealed ANN export requires a segmented ANN index")
	}
	return segmented.VisitSealedSegments(func(segmentIndex int, segment *ann.AnnIndex) error {
		segmentID := fmt.Sprintf("%s-%06d", generation, segmentIndex)
		path := filepath.Join(dir, segmentID+".snapshot")
		manifest, err := ann.WritePrebuiltSnapshotFile(path, segmentID, metric, segment)
		if err != nil {
			return err
		}
		replay := func(yield func(PrebuiltANNRecord) error) error {
			return segment.VisitLiveVectors32(func(internalID int, values32 []float32) error {
				externalID, found, err := s.lookupID(internalID)
				if err != nil {
					return err
				}
				if !found {
					return fmt.Errorf("sealed ANN segment references unmapped internal ID %d", internalID)
				}
				values := make([]float64, len(values32))
				for i, value := range values32 {
					values[i] = float64(value)
				}
				return yield(PrebuiltANNRecord{ID: externalID, InternalID: internalID, Values: values})
			})
		}
		return visit(PrebuiltANNSegmentExport{Path: path, Manifest: manifest, Replay: replay})
	})
}

// SnapshotMutableANNRecords returns the bounded, unsealed ingest tail using
// stable external IDs. Its size is at most one configured ANN segment.
func (s *Service) SnapshotMutableANNRecords() ([]PrebuiltANNRecord, error) {
	s.ensureRuntimeDeps()
	segmented, ok := s.currentANNIndex().(*ann.SegmentedIndex)
	if !ok {
		return nil, errors.New("mutable ANN snapshot requires a segmented ANN index")
	}
	vectors := segmented.SnapshotMutableVectors32()
	out := make([]PrebuiltANNRecord, 0, len(vectors))
	for _, vector := range vectors {
		externalID, found, err := s.lookupID(vector.ID)
		if err != nil {
			return nil, err
		}
		if !found {
			return nil, fmt.Errorf("mutable ANN tail references unmapped internal ID %d", vector.ID)
		}
		values := make([]float64, len(vector.Values))
		for i, value := range vector.Values {
			values[i] = float64(value)
		}
		out = append(out, PrebuiltANNRecord{ID: externalID, InternalID: vector.ID, Values: values})
	}
	return out, nil
}

// PrebuiltANNMigrationConfig exposes only the local staging directory and ANN
// construction profile needed by the migration coordinator.
func (s *Service) PrebuiltANNMigrationConfig() (string, ann.Options) {
	return filepath.Join(s.vectorPath, "outgoing-ann"), s.annOptions
}

func (s *Service) ANNProfile() ann.Options { return s.annOptions }

func safeSegmentGeneration(value string) bool {
	if value == "" {
		return false
	}
	for _, r := range value {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_') {
			return false
		}
	}
	return true
}
