package core

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"
)

type localPrebuiltPartitionDestination struct {
	service *Service
	dir     string
}

func (d *localPrebuiltPartitionDestination) ReservePrebuiltIDs(ids []string) ([]IDMappingEntry, error) {
	return d.service.ReserveIDMappings(ids)
}

func (d *localPrebuiltPartitionDestination) UploadPrebuiltSnapshot(manifest ann.SegmentManifest, sourcePath string) (string, error) {
	source, err := os.Open(sourcePath)
	if err != nil {
		return "", err
	}
	defer source.Close()
	path := filepath.Join(d.dir, manifest.SegmentID+".snapshot")
	stager, err := ann.NewSnapshotStager(path, manifest)
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(stager, source); err != nil {
		stager.Abort()
		return "", err
	}
	if err := stager.Commit(); err != nil {
		return "", err
	}
	return path, nil
}

func (d *localPrebuiltPartitionDestination) ApplyPrebuiltSegment(manifest ann.SegmentManifest, replay PrebuiltANNReplay) error {
	return d.service.ApplyPrebuiltANNSnapshotFile(filepath.Join(d.dir, manifest.SegmentID+".snapshot"), manifest, replay)
}

type failingPersistentPrebuiltStore struct {
	vectors   map[string]index.Vector
	failAfter int
	writes    int
}

func (s *failingPersistentPrebuiltStore) IsPersistent() bool { return true }
func (s *failingPersistentPrebuiltStore) UpsertVector(vector index.Vector) error {
	if s.writes >= s.failAfter {
		return errors.New("injected persistent write failure")
	}
	s.writes++
	s.vectors[vector.ID] = vector
	return nil
}
func (s *failingPersistentPrebuiltStore) GetVector(id string) (index.Vector, error) {
	vector, ok := s.vectors[id]
	if !ok {
		return index.Vector{}, errors.New("vector not found")
	}
	return vector, nil
}
func (s *failingPersistentPrebuiltStore) DeleteVector(id string) error {
	delete(s.vectors, id)
	return nil
}
func (s *failingPersistentPrebuiltStore) ListVectors() []index.Vector {
	out := make([]index.Vector, 0, len(s.vectors))
	for _, vector := range s.vectors {
		out = append(out, vector)
	}
	return out
}

func replayPrebuiltRecords(records []PrebuiltANNRecord) PrebuiltANNReplay {
	return func(visit func(PrebuiltANNRecord) error) error {
		for _, record := range records {
			if err := visit(record); err != nil {
				return err
			}
		}
		return nil
	}
}

func writeTransferSnapshot(t *testing.T, dir, segmentID string, records []PrebuiltANNRecord) (string, ann.SegmentManifest) {
	t.Helper()
	graph := ann.NewAnnIndexWithOptions(ann.Options{M: 4, EfConstruction: 16, EfSearch: 16})
	for _, record := range records {
		if err := graph.AddVector(record.InternalID, record.Values); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(dir, segmentID+".snapshot")
	manifest, err := ann.WritePrebuiltSnapshotFile(path, segmentID, "l2", graph)
	if err != nil {
		t.Fatal(err)
	}
	return path, manifest
}

func newPrebuiltTransferTarget(t *testing.T) *Service {
	t.Helper()
	base := t.TempDir()
	service := NewService(ServiceOptions{
		MaxVectorDim:       8,
		MaxK:               5,
		SearchMode:         "ann",
		VectorStore:        "segment",
		VectorPath:         filepath.Join(base, "vectors"),
		SnapshotPath:       filepath.Join(base, "snapshot.json"),
		WALPath:            filepath.Join(base, "wal.log"),
		ANNSegmentMaxNodes: 16,
	})
	t.Cleanup(func() { _ = service.Close() })
	return service
}

func TestApplyPrebuiltANNSnapshotFileCommitsMultipleSegments(t *testing.T) {
	target := newPrebuiltTransferTarget(t)
	dir := t.TempDir()
	first := []PrebuiltANNRecord{
		{ID: "a", InternalID: 11, Values: []float64{1, 0}},
		{ID: "b", InternalID: 12, Values: []float64{0, 1}},
	}
	second := []PrebuiltANNRecord{
		{ID: "c", InternalID: 21, Values: []float64{2, 0}},
		{ID: "d", InternalID: 22, Values: []float64{0, 2}},
	}
	firstPath, firstManifest := writeTransferSnapshot(t, dir, "first", first)
	secondPath, secondManifest := writeTransferSnapshot(t, dir, "second", second)
	if err := target.ApplyPrebuiltANNSnapshotFile(firstPath, firstManifest, replayPrebuiltRecords(first)); err != nil {
		t.Fatal(err)
	}
	if err := target.ApplyPrebuiltANNSnapshotFile(firstPath, firstManifest, replayPrebuiltRecords(first)); err != nil {
		t.Fatalf("idempotent segment retry: %v", err)
	}
	if err := target.ApplyPrebuiltANNSnapshotFile(secondPath, secondManifest, replayPrebuiltRecords(second)); err != nil {
		t.Fatal(err)
	}
	if got := target.currentANNIndex().Stats().Nodes; got != 4 {
		t.Fatalf("ANN nodes=%d want 4", got)
	}
	results, err := target.Search([]float64{2.1, 0}, 1)
	if err != nil || len(results) != 1 || results[0].ID != "c" {
		t.Fatalf("results=%+v err=%v", results, err)
	}
}

func TestApplyPrebuiltANNSnapshotFileRejectsIncompleteWindowWithoutMutation(t *testing.T) {
	target := newPrebuiltTransferTarget(t)
	records := []PrebuiltANNRecord{
		{ID: "a", InternalID: 31, Values: []float64{1, 0}},
		{ID: "b", InternalID: 32, Values: []float64{0, 1}},
	}
	path, manifest := writeTransferSnapshot(t, t.TempDir(), "incomplete", records)
	err := target.ApplyPrebuiltANNSnapshotFile(path, manifest, replayPrebuiltRecords(records[:1]))
	if err == nil {
		t.Fatal("expected incomplete transfer to fail")
	}
	if got := target.currentANNIndex().Stats().Nodes; got != 0 {
		t.Fatalf("failed transfer published %d ANN nodes", got)
	}
	if got := target.ListVectors(); len(got) != 0 {
		t.Fatalf("failed transfer left vectors: %+v", got)
	}
	if _, err := target.ExportIDMappings([]string{"a"}); err == nil {
		t.Fatal("failed transfer left an imported mapping")
	}
}

func TestApplyPrebuiltANNSnapshotFileRetrySurvivesRestart(t *testing.T) {
	base := t.TempDir()
	options := ServiceOptions{
		MaxVectorDim: 8, MaxK: 5, SearchMode: "ann",
		VectorStore: "segment", VectorPath: filepath.Join(base, "vectors"),
		SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal.log"),
		ANNSegmentMaxNodes: 16,
	}
	records := []PrebuiltANNRecord{{ID: "durable", InternalID: 51, Values: []float64{1, 0}}}
	path, manifest := writeTransferSnapshot(t, filepath.Join(base, "incoming"), "durable-segment", records)
	first := NewService(options)
	if err := first.ApplyPrebuiltANNSnapshotFile(path, manifest, replayPrebuiltRecords(records)); err != nil {
		t.Fatal(err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	reopened := NewService(options)
	t.Cleanup(func() { _ = reopened.Close() })
	if err := reopened.ApplyPrebuiltANNSnapshotFile(path, manifest, replayPrebuiltRecords(records)); err != nil {
		t.Fatalf("retry after restart: %v", err)
	}
	results, err := reopened.Search([]float64{1, 0}, 1)
	if err != nil || len(results) != 1 || results[0].ID != "durable" {
		t.Fatalf("results=%+v err=%v", results, err)
	}
}

func TestPrebuiltANNApplyingJournalRecoversPartialCommitAfterRestart(t *testing.T) {
	base := t.TempDir()
	options := ServiceOptions{
		MaxVectorDim: 8, MaxK: 5, SearchMode: "ann",
		VectorStore: "segment", VectorPath: filepath.Join(base, "vectors"),
		SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal.log"),
		ANNSegmentMaxNodes: 16,
	}
	first := NewService(options)
	mappings, err := first.ReserveIDMappings([]string{"recover-a", "recover-b"})
	if err != nil {
		t.Fatal(err)
	}
	records := []PrebuiltANNRecord{
		{ID: mappings[0].ID, InternalID: mappings[0].InternalID, Values: []float64{1, 0}},
		{ID: mappings[1].ID, InternalID: mappings[1].InternalID, Values: []float64{0, 1}},
	}
	snapshotPath, manifest := writeTransferSnapshot(t, filepath.Join(base, "vectors", "incoming-ann"), "recover-segment", records)
	spool, err := NewPrebuiltANNRecordSpool(filepath.Join(base, "vectors", "incoming-ann"), manifest)
	if err != nil {
		t.Fatal(err)
	}
	for _, record := range records {
		if err := spool.Append(record); err != nil {
			t.Fatal(err)
		}
	}
	if err := spool.Seal(); err != nil {
		t.Fatal(err)
	}
	// Model a process failure after the first canonical vector was persisted
	// but before the graph/catalog commit.
	if err := first.PersistVectorsForPrebuiltANN([]index.Vector{{ID: records[0].ID, Values: records[0].Values}}); err != nil {
		t.Fatal(err)
	}
	transaction := prebuiltANNTransaction{
		Version: prebuiltANNTransactionVersion, State: "applying", Manifest: manifest,
		SnapshotPath: snapshotPath, RecordsPath: spool.Path(),
	}
	if err := writePrebuiltANNTransaction(first.prebuiltANNTransactionPath(manifest.SegmentID), transaction); err != nil {
		t.Fatal(err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}

	recovered := NewService(options)
	t.Cleanup(func() { _ = recovered.Close() })
	if got := recovered.ListVectors(); len(got) != 2 {
		t.Fatalf("recovered vectors=%+v", got)
	}
	if got := recovered.currentANNIndex().Stats().Nodes; got != 2 {
		t.Fatalf("recovered ANN nodes=%d want 2", got)
	}
	results, err := recovered.Search([]float64{1, 0}, 1)
	if err != nil || len(results) != 1 || results[0].ID != "recover-a" {
		t.Fatalf("results=%+v err=%v", results, err)
	}
	if _, err := os.Stat(spool.Path()); !os.IsNotExist(err) {
		t.Fatalf("committed recovery retained spool: %v", err)
	}
	if entries, err := os.ReadDir(recovered.prebuiltANNTransactionDir()); err != nil || len(entries) != 0 {
		t.Fatalf("transaction directory entries=%v err=%v", entries, err)
	}
}

func TestApplyPrebuiltANNSnapshotFileRollsBackPersistentWriteFailure(t *testing.T) {
	store := &failingPersistentPrebuiltStore{vectors: make(map[string]index.Vector), failAfter: 1}
	target := NewServiceWithDeps(ServiceOptions{MaxVectorDim: 8, MaxK: 5, SearchMode: "ann"}, ServiceDeps{
		Index:       index.NewIndex(),
		VectorStore: store,
		ANNIndex:    ann.NewSegmentedIndex(ann.Options{M: 4, EfConstruction: 16, EfSearch: 16}, 16),
		IDResolver:  newMemoryIDResolver(),
	})
	t.Cleanup(func() { _ = target.Close() })
	records := []PrebuiltANNRecord{
		{ID: "a", InternalID: 41, Values: []float64{1, 0}},
		{ID: "b", InternalID: 42, Values: []float64{0, 1}},
	}
	path, manifest := writeTransferSnapshot(t, t.TempDir(), "write-failure", records)
	if err := target.ApplyPrebuiltANNSnapshotFile(path, manifest, replayPrebuiltRecords(records)); err == nil {
		t.Fatal("expected injected persistent write failure")
	}
	if len(store.vectors) != 0 || len(target.ListVectors()) != 0 {
		t.Fatalf("failed transfer left vectors: store=%d index=%d", len(store.vectors), len(target.ListVectors()))
	}
	if _, err := target.ExportIDMappings([]string{"a", "b"}); err == nil {
		t.Fatal("failed transfer left imported mappings")
	}
	if got := target.currentANNIndex().Stats().Nodes; got != 0 {
		t.Fatalf("failed transfer published %d ANN nodes", got)
	}
}

func TestExportAndApplySealedANNSegmentsWithoutRebuild(t *testing.T) {
	base := t.TempDir()
	source := NewService(ServiceOptions{
		MaxVectorDim: 8, MaxK: 5, SearchMode: "ann",
		VectorStore: "segment", VectorPath: filepath.Join(base, "source-vectors"),
		SnapshotPath: filepath.Join(base, "source.json"), WALPath: filepath.Join(base, "source.wal"),
		ANNSegmentMaxNodes: 2,
	})
	t.Cleanup(func() { _ = source.Close() })
	for _, record := range []PrebuiltANNRecord{
		{ID: "a", Values: []float64{1, 0}},
		{ID: "b", Values: []float64{0, 1}},
		{ID: "c", Values: []float64{2, 0}},
		{ID: "d", Values: []float64{0, 2}},
	} {
		if err := source.AddVector(record.ID, record.Values); err != nil {
			t.Fatal(err)
		}
	}
	target := newPrebuiltTransferTarget(t)
	exported := 0
	if err := source.ExportSealedANNSegments(filepath.Join(base, "exports"), "generation-a", "l2", func(segment PrebuiltANNSegmentExport) error {
		exported++
		return target.ApplyPrebuiltANNSnapshotFile(segment.Path, segment.Manifest, segment.Replay)
	}); err != nil {
		t.Fatal(err)
	}
	if exported != 2 {
		t.Fatalf("exported segments=%d want 2", exported)
	}
	if got := target.currentANNIndex().Stats().Nodes; got != 4 {
		t.Fatalf("target ANN nodes=%d want 4", got)
	}
	results, err := target.Search([]float64{2.1, 0}, 1)
	if err != nil || len(results) != 1 || results[0].ID != "c" {
		t.Fatalf("results=%+v err=%v", results, err)
	}
}

func TestTransferPrebuiltANNPartitionUsesDestinationReservedIDs(t *testing.T) {
	target := newPrebuiltTransferTarget(t)
	base := t.TempDir()
	destination := &localPrebuiltPartitionDestination{service: target, dir: filepath.Join(base, "remote")}
	sourceRecords := []PrebuiltANNRecord{
		{ID: "stay-a", InternalID: 1, Values: []float64{0, 1}},
		{ID: "move-a", InternalID: 1, Values: []float64{1, 0}},
		{ID: "move-b", InternalID: 1, Values: []float64{2, 0}},
	}
	source := replayPrebuiltRecords(sourceRecords)
	moving := FilterPrebuiltANNReplay(source, func(id string) bool { return len(id) >= 5 && id[:5] == "move-" })
	manifest, err := TransferPrebuiltANNPartition(
		filepath.Join(base, "outgoing"), "partition-a", "l2",
		ann.Options{M: 4, EfConstruction: 16, EfSearch: 16}, moving, destination,
	)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Nodes != 2 {
		t.Fatalf("partition nodes=%d want 2", manifest.Nodes)
	}
	if got := target.ListVectors(); len(got) != 2 {
		t.Fatalf("target vectors=%+v", got)
	}
	if _, err := target.GetVector("stay-a"); err == nil {
		t.Fatal("non-moving vector was transferred")
	}
	mappings, err := target.ExportIDMappings([]string{"move-a", "move-b"})
	if err != nil {
		t.Fatal(err)
	}
	if mappings[0].InternalID == mappings[1].InternalID || mappings[0].InternalID == 1 && mappings[1].InternalID == 1 {
		t.Fatalf("destination mappings collided: %+v", mappings)
	}
	results, err := target.Search([]float64{2.1, 0}, 1)
	if err != nil || len(results) != 1 || results[0].ID != "move-b" {
		t.Fatalf("results=%+v err=%v", results, err)
	}
}

func TestTransferPrebuiltANNPartitionLocalServiceUsesDurableTransaction(t *testing.T) {
	target := newPrebuiltTransferTarget(t)
	records := []PrebuiltANNRecord{
		{ID: "tx-a", InternalID: 1, Values: []float64{1, 0}},
		{ID: "tx-b", InternalID: 2, Values: []float64{0, 1}},
	}
	manifest, err := TransferPrebuiltANNPartition(
		t.TempDir(), "partition-transactional", "l2",
		ann.Options{M: 4, EfConstruction: 16, EfSearch: 16},
		replayPrebuiltRecords(records), target,
	)
	if err != nil {
		t.Fatal(err)
	}
	if got := target.ListVectors(); len(got) != len(records) {
		t.Fatalf("target vectors=%+v", got)
	}
	if entries, err := os.ReadDir(target.prebuiltANNTransactionDir()); err != nil || len(entries) != 0 {
		t.Fatalf("transaction directory entries=%v err=%v", entries, err)
	}
	recordsPath := filepath.Join(target.vectorPath, "incoming-ann", manifest.SegmentID+".records")
	if _, err := os.Stat(recordsPath); !os.IsNotExist(err) {
		t.Fatalf("committed transfer retained record spool: %v", err)
	}
}
