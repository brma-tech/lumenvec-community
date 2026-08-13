package core

import (
	"errors"
	"fmt"
	"path/filepath"

	"lumenvec/internal/index/ann"
)

var ErrEmptyPrebuiltANNPartition = errors.New("prebuilt ANN partition is empty")
var ErrDestinationPrebuiltBuildUnsupported = errors.New("destination-side prebuilt ANN construction is unsupported")

// PrebuiltANNPartitionDestination is implemented by remote shards and keeps
// the control-plane coordinator independent from the transport.
type PrebuiltANNPartitionDestination interface {
	ReservePrebuiltIDs([]string) ([]IDMappingEntry, error)
	UploadPrebuiltSnapshot(ann.SegmentManifest, string) (string, error)
	ApplyPrebuiltSegment(ann.SegmentManifest, PrebuiltANNReplay) error
}

// PrebuiltANNPartitionBuilder is an optional destination capability. Remote
// data nodes implement it to build the graph on the owning node instead of
// consuming coordinator CPU. Returning ErrDestinationPrebuiltBuildUnsupported
// preserves compatibility with older nodes through the legacy upload path.
type PrebuiltANNPartitionBuilder interface {
	BuildPrebuiltSegment(string, string, ann.Options, PrebuiltANNReplay) (ann.SegmentManifest, error)
}

// FilterPrebuiltANNReplay creates a repeatable view over records selected for
// one destination owner. It does not buffer vectors.
func FilterPrebuiltANNReplay(source PrebuiltANNReplay, include func(string) bool) PrebuiltANNReplay {
	return func(visit func(PrebuiltANNRecord) error) error {
		if source == nil || include == nil {
			return errors.New("prebuilt ANN source and filter are required")
		}
		return source(func(record PrebuiltANNRecord) error {
			if !include(record.ID) {
				return nil
			}
			return visit(record)
		})
	}
}

// TransferPrebuiltANNPartition reserves destination identities, builds a graph
// only for the selected/moving records, streams the artifact, and commits the
// matching records. Source-local internal IDs are never published.
func TransferPrebuiltANNPartition(dir, segmentID, metric string, options ann.Options, source PrebuiltANNReplay, destination PrebuiltANNPartitionDestination) (ann.SegmentManifest, error) {
	if dir == "" || !safeSegmentGeneration(segmentID) || source == nil || destination == nil {
		return ann.SegmentManifest{}, errors.New("partition directory, safe segment ID, source and destination are required")
	}
	if builder, ok := destination.(PrebuiltANNPartitionBuilder); ok {
		manifest, err := builder.BuildPrebuiltSegment(segmentID, metric, options, source)
		if err == nil {
			return manifest, nil
		}
		if !errors.Is(err, ErrDestinationPrebuiltBuildUnsupported) {
			return ann.SegmentManifest{}, err
		}
	}
	ids := make([]string, 0)
	seen := make(map[string]struct{})
	if err := source(func(record PrebuiltANNRecord) error {
		if record.ID == "" || len(record.Values) == 0 {
			return errors.New("invalid prebuilt ANN partition record")
		}
		if _, duplicate := seen[record.ID]; duplicate {
			return fmt.Errorf("duplicate partition vector ID %q", record.ID)
		}
		seen[record.ID] = struct{}{}
		ids = append(ids, record.ID)
		return nil
	}); err != nil {
		return ann.SegmentManifest{}, err
	}
	if len(ids) == 0 {
		return ann.SegmentManifest{}, ErrEmptyPrebuiltANNPartition
	}
	reserved, err := destination.ReservePrebuiltIDs(ids)
	if err != nil {
		return ann.SegmentManifest{}, err
	}
	if len(reserved) != len(ids) {
		return ann.SegmentManifest{}, errors.New("destination reservation cardinality mismatch")
	}
	internalByExternal := make(map[string]int, len(reserved))
	seenInternal := make(map[int]string, len(reserved))
	for _, mapping := range reserved {
		if _, requested := seen[mapping.ID]; !requested || mapping.InternalID <= 0 {
			return ann.SegmentManifest{}, errors.New("destination returned an invalid ID reservation")
		}
		if previous, duplicate := seenInternal[mapping.InternalID]; duplicate && previous != mapping.ID {
			return ann.SegmentManifest{}, errors.New("destination returned colliding internal IDs")
		}
		internalByExternal[mapping.ID] = mapping.InternalID
		seenInternal[mapping.InternalID] = mapping.ID
	}
	if len(internalByExternal) != len(ids) {
		return ann.SegmentManifest{}, errors.New("destination omitted one or more ID reservations")
	}

	graph := ann.NewAnnIndexWithOptions(options)
	remapped := func(visit func(PrebuiltANNRecord) error) error {
		return source(func(record PrebuiltANNRecord) error {
			internalID, ok := internalByExternal[record.ID]
			if !ok {
				return fmt.Errorf("missing destination reservation for %q", record.ID)
			}
			return visit(PrebuiltANNRecord{ID: record.ID, InternalID: internalID, Values: record.Values})
		})
	}
	if err := remapped(func(record PrebuiltANNRecord) error {
		return graph.AddVector(record.InternalID, record.Values)
	}); err != nil {
		return ann.SegmentManifest{}, err
	}
	path := filepath.Join(dir, segmentID+".snapshot")
	manifest, err := ann.WritePrebuiltSnapshotFile(path, segmentID, metric, graph)
	if err != nil {
		return ann.SegmentManifest{}, err
	}
	if _, err := destination.UploadPrebuiltSnapshot(manifest, path); err != nil {
		return ann.SegmentManifest{}, err
	}
	if err := destination.ApplyPrebuiltSegment(manifest, remapped); err != nil {
		return ann.SegmentManifest{}, err
	}
	return manifest, nil
}
