package core

import (
	"errors"
	"fmt"
	"path/filepath"

	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"
)

var errPrebuiltPublishedUncataloged = errors.New("prebuilt ANN graph published without durable catalog")

// PrebuiltANNRecord is one canonical vector and its stable source identity.
// Values are replayed from a file-backed transport; callers do not need to
// retain a whole segment in memory.
type PrebuiltANNRecord struct {
	ID         string
	InternalID int
	Values     []float64
}

// PrebuiltANNReplay must produce the same finite record sequence on every
// invocation. The service invokes it once for validation and once for commit,
// allowing transports to rewind a staging file instead of materializing a
// segment-sized slice.
type PrebuiltANNReplay func(visit func(PrebuiltANNRecord) error) error

// ApplyPrebuiltANNSnapshotFile commits identities, canonical vectors and one
// immutable ANN graph as a single destination-side reshard window.
//
// This path currently requires a persistent vector store: that store is the
// durable source of truth and does not need a separate WAL append between the
// vector commit and graph publication. Any failure before publication removes
// exactly the vectors and mappings introduced by this call.
func (s *Service) ApplyPrebuiltANNSnapshotFile(path string, manifest ann.SegmentManifest, replay PrebuiltANNReplay) error {
	if replay == nil {
		return errors.New("prebuilt ANN record replay is required")
	}
	s.ensureRuntimeDeps()
	s.persistMu.Lock()
	defer s.persistMu.Unlock()
	if !s.usesPersistentVectorStore() {
		return errors.New("transactional prebuilt ANN application requires a persistent vector store")
	}
	segmented, ok := s.currentANNIndex().(*ann.SegmentedIndex)
	if !ok {
		return errors.New("prebuilt ANN application requires a segmented ANN index")
	}
	snapshotIDs, err := ann.PrebuiltSnapshotIDsFromFile(path, manifest)
	if err != nil {
		return fmt.Errorf("validate prebuilt ANN snapshot: %w", err)
	}
	catalog, err := s.prebuiltANNCommitCatalog()
	if err != nil {
		return err
	}
	if catalog != nil {
		for _, committed := range catalog.Snapshot() {
			if committed.SegmentID != manifest.SegmentID {
				continue
			}
			if committed.PayloadSHA256 != manifest.PayloadSHA256 || committed.PayloadBytes != manifest.PayloadBytes {
				return errors.New("prebuilt ANN segment ID is already committed with different content")
			}
			for _, internalID := range snapshotIDs {
				externalID, ok, err := s.lookupID(internalID)
				if err != nil {
					return err
				}
				if !ok {
					return fmt.Errorf("committed prebuilt segment references unmapped internal ID %d", internalID)
				}
				if _, err := s.vectorStore.GetVector(externalID); err != nil {
					return fmt.Errorf("committed prebuilt segment references missing vector %q: %w", externalID, err)
				}
			}
			return nil
		}
	}
	expected := make(map[int]struct{}, len(snapshotIDs))
	for _, internalID := range snapshotIDs {
		expected[internalID] = struct{}{}
	}
	mappings := make([]IDMappingEntry, 0, len(snapshotIDs))
	seenInternal := make(map[int]struct{}, len(snapshotIDs))
	seenExternal := make(map[string]struct{}, len(snapshotIDs))
	externalByInternal := make(map[int]string, len(snapshotIDs))
	if err := replay(func(record PrebuiltANNRecord) error {
		if record.ID == "" || len(record.Values) == 0 || len(record.Values) != manifest.Dimension {
			return errors.New("invalid prebuilt ANN vector record")
		}
		if _, ok := expected[record.InternalID]; !ok {
			return fmt.Errorf("record internal ID %d is not present in snapshot", record.InternalID)
		}
		if _, duplicate := seenInternal[record.InternalID]; duplicate {
			return fmt.Errorf("duplicate internal ID %d in prebuilt ANN records", record.InternalID)
		}
		if _, duplicate := seenExternal[record.ID]; duplicate {
			return fmt.Errorf("duplicate external ID %q in prebuilt ANN records", record.ID)
		}
		seenInternal[record.InternalID] = struct{}{}
		seenExternal[record.ID] = struct{}{}
		externalByInternal[record.InternalID] = record.ID
		mappings = append(mappings, IDMappingEntry{ID: record.ID, InternalID: record.InternalID})
		return nil
	}); err != nil {
		return err
	}
	if len(seenInternal) != len(expected) {
		return fmt.Errorf("prebuilt ANN record cardinality mismatch: got %d want %d", len(seenInternal), len(expected))
	}
	resolver, ok := s.idResolver.(trackedMergeIDMappingResolver)
	if !ok {
		return errors.New("ID resolver does not support transactional mapping import")
	}
	addedMappings, err := resolver.MergeIDMappingsTracked(mappings)
	if err != nil {
		return err
	}
	rollbackMappings := func() {
		for _, id := range addedMappings {
			_ = s.removeID(id)
		}
	}
	addedVectors := make([]string, 0, len(expected))
	seenCommit := make(map[int]struct{}, len(expected))
	const prebuiltCommitBatchSize = 1000
	nativeWriter, useNativeWriter := s.vectorStore.(interface {
		UpsertVectors32([]vectorBatch32) error
	})
	pendingNative := make([]vectorBatch32, 0, min(prebuiltCommitBatchSize, len(expected)))
	flushNative := func() error {
		if len(pendingNative) == 0 {
			return nil
		}
		if err := nativeWriter.UpsertVectors32(pendingNative); err != nil {
			return err
		}
		pendingNative = pendingNative[:0]
		return nil
	}
	err = replay(func(record PrebuiltANNRecord) error {
		expectedExternal, ok := externalByInternal[record.InternalID]
		if !ok || record.ID != expectedExternal || len(record.Values) != manifest.Dimension {
			return errors.New("prebuilt ANN replay changed between validation and commit")
		}
		if _, duplicate := seenCommit[record.InternalID]; duplicate {
			return errors.New("prebuilt ANN replay contains duplicate records")
		}
		seenCommit[record.InternalID] = struct{}{}
		vec := index.Vector{ID: record.ID, Values: record.Values}
		if err := s.index.AddVector(vec); err != nil {
			return err
		}
		addedVectors = append(addedVectors, record.ID)
		if useNativeWriter {
			values := make([]float32, len(record.Values))
			for i, value := range record.Values {
				values[i] = float32(value)
			}
			pendingNative = append(pendingNative, vectorBatch32{ID: record.ID, Values: values})
			if len(pendingNative) >= prebuiltCommitBatchSize {
				return flushNative()
			}
		} else if err := s.vectorStore.UpsertVector(vec); err != nil {
			return err
		}
		return nil
	})
	if err == nil && useNativeWriter {
		err = flushNative()
	}
	if err == nil && len(seenCommit) != len(expected) {
		err = errors.New("prebuilt ANN replay ended before all records were committed")
	}
	if err != nil {
		s.rollbackAddedVectors(addedVectors)
		rollbackMappings()
		return err
	}
	if err := segmented.PublishPrebuiltSnapshotFile(path, manifest); err != nil {
		s.rollbackAddedVectors(addedVectors)
		rollbackMappings()
		return err
	}
	if catalog != nil {
		committed := manifest
		committed.State = ann.SegmentPublished
		if err := catalog.Publish(committed); err != nil {
			return fmt.Errorf("%w: %v", errPrebuiltPublishedUncataloged, err)
		}
	}
	s.annCheckpointDirty.Store(true)
	return nil
}

func (s *Service) prebuiltANNCommitCatalog() (*ann.SegmentCatalog, error) {
	if s.vectorPath == "" {
		return nil, nil
	}
	return ann.OpenSegmentCatalog(filepath.Join(s.vectorPath, "prebuilt-ann-catalog.json"))
}
