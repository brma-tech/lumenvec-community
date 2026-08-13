package core

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"
)

func TestCurrentDeltaOffsetWaitsForInFlightPersistence(t *testing.T) {
	base := t.TempDir()
	service := NewService(ServiceOptions{
		MaxVectorDim: 2, MaxK: 4, SearchMode: "ann", VectorStore: "segment",
		VectorPath: filepath.Join(base, "vectors"),
	})
	defer service.Close()

	service.persistMu.Lock()
	done := make(chan struct{})
	go func() {
		_ = service.CurrentDeltaOffset()
		close(done)
	}()
	select {
	case <-done:
		service.persistMu.Unlock()
		t.Fatal("delta cutover offset crossed an in-flight persistence transaction")
	case <-time.After(25 * time.Millisecond):
	}
	service.persistMu.Unlock()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("delta cutover offset did not resume after persistence completed")
	}
}

func TestServiceStagedBulkIngestPublishesImmutableANN(t *testing.T) {
	base := t.TempDir()
	service := NewService(ServiceOptions{
		MaxVectorDim: 4, MaxK: 4, SearchMode: "ann", VectorStore: "segment",
		VectorPath: filepath.Join(base, "vectors"), ANNSegmentMaxNodes: 8,
		ANNStagedIngestMinBatch: 2,
		ANNOptions:              ann.Options{M: 4, EfConstruction: 8, EfSearch: 8},
	})
	defer service.Close()
	if err := service.AddVectors([]index.Vector{
		{ID: "staged-a", Values: []float64{1, 0}},
		{ID: "staged-b", Values: []float64{0, 1}},
		{ID: "staged-c", Values: []float64{1, 1}},
	}); err != nil {
		t.Fatal(err)
	}
	segmented, ok := service.currentANNIndex().(*ann.SegmentedIndex)
	if !ok {
		t.Fatal("expected segmented ANN")
	}
	snapshot := segmented.SnapshotExport()
	if len(snapshot.Segments) != 1 || len(snapshot.Tail) != 0 {
		t.Fatalf("segments=%d tail=%d, want immutable-only publication", len(snapshot.Segments), len(snapshot.Tail))
	}
	results, err := service.Search([]float64{1, 0}, 1)
	if err != nil || len(results) != 1 || results[0].ID != "staged-a" {
		t.Fatalf("results=%+v err=%v", results, err)
	}
}

func TestServiceConfiguredStagedFloat32IngestPublishesImmutableANN(t *testing.T) {
	base := t.TempDir()
	service := NewService(ServiceOptions{
		MaxVectorDim: 4, MaxK: 4, SearchMode: "ann", VectorStore: "segment",
		VectorPath: filepath.Join(base, "vectors"), ANNSegmentMaxNodes: 8,
		ANNStagedIngestMinBatch: 2,
		ANNOptions:              ann.Options{M: 4, EfConstruction: 8, EfSearch: 8},
	})
	defer service.Close()
	if err := service.AddVectors32([]VectorBatch32{
		{ID: "native-a", Values: []float32{1, 0}},
		{ID: "native-b", Values: []float32{0, 1}},
		{ID: "native-c", Values: []float32{1, 1}},
	}); err != nil {
		t.Fatal(err)
	}
	segmented := service.currentANNIndex().(*ann.SegmentedIndex)
	snapshot := segmented.SnapshotExport()
	if len(snapshot.Segments) != 1 || len(snapshot.Tail) != 0 {
		t.Fatalf("segments=%d tail=%d, want immutable-only publication", len(snapshot.Segments), len(snapshot.Tail))
	}
}

type stubPersistence struct {
	saveSnapshotFn func([]index.Vector) error
	loadSnapshotFn func() (map[string][]float64, error)
	appendWALFn    func(walOp) error
	replayWALFn    func(func(walOp) error) error
	truncateWALFn  func() error
}

func TestServiceStagedFloat32BatchPersistsAndReopens(t *testing.T) {
	base := t.TempDir()
	opts := ServiceOptions{
		MaxVectorDim: 4, MaxK: 5, SearchMode: "ann", VectorStore: "segment",
		VectorPath: filepath.Join(base, "vectors"), SnapshotPath: filepath.Join(base, "snapshot.json"),
		WALPath: filepath.Join(base, "wal.log"), ANNSegmentMaxNodes: 8,
	}
	service := NewService(opts)
	if err := service.addVectors32Staged([]vectorBatch32{
		{ID: "staged-a", Values: []float32{1, 2}},
		{ID: "staged-b", Values: []float32{3, 4}},
	}); err != nil {
		t.Fatal(err)
	}
	if got, err := service.GetVector("staged-b"); err != nil || got.Values[0] != 3 {
		t.Fatalf("staged vector=%+v err=%v", got, err)
	}
	if err := service.Close(); err != nil {
		t.Fatal(err)
	}

	reopened := NewService(opts)
	defer reopened.Close()
	if got, err := reopened.GetVector("staged-a"); err != nil || got.Values[1] != 2 {
		t.Fatalf("reopened staged vector=%+v err=%v", got, err)
	}
	results, err := reopened.Search([]float64{1, 2}, 2)
	if err != nil || len(results) == 0 || results[0].ID != "staged-a" {
		t.Fatalf("reopened staged search=%+v err=%v", results, err)
	}
}

func TestServiceCanInjectIVFANNIndex(t *testing.T) {
	ivf, err := ann.NewIVFANNIndex([][]float64{{0, 0}, {10, 10}}, 2)
	if err != nil {
		t.Fatal(err)
	}
	base := t.TempDir()
	s := NewServiceWithDeps(ServiceOptions{MaxVectorDim: 2, MaxK: 5, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log")}, ServiceDeps{ANNIndex: ivf})
	defer s.Close()
	if err := s.AddVector("near", []float64{10, 10}); err != nil {
		t.Fatal(err)
	}
	if err := s.AddVector("far", []float64{0, 0}); err != nil {
		t.Fatal(err)
	}
	hits, err := s.Search([]float64{10, 10}, 1)
	if err != nil || len(hits) != 1 || hits[0].ID != "near" {
		t.Fatalf("hits=%+v err=%v", hits, err)
	}
}

func TestServiceExportsAndImportsStableIDMappings(t *testing.T) {
	base := t.TempDir()
	source := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 4, SnapshotPath: filepath.Join(base, "source.json"), WALPath: filepath.Join(base, "source.wal")})
	target := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 4, SnapshotPath: filepath.Join(base, "target.json"), WALPath: filepath.Join(base, "target.wal")})
	t.Cleanup(func() { _ = source.Close(); _ = target.Close() })
	if err := source.AddVectors([]index.Vector{{ID: "z", Values: []float64{0, 1}}, {ID: "a", Values: []float64{1, 0}}}); err != nil {
		t.Fatal(err)
	}
	mapping, err := source.ExportIDMappings([]string{"a", "z"})
	if err != nil {
		t.Fatal(err)
	}
	if err := target.ImportIDMappings(mapping); err != nil {
		t.Fatal(err)
	}
	if err := target.AddVectors([]index.Vector{{ID: "z", Values: []float64{0, 1}}, {ID: "a", Values: []float64{1, 0}}}); err != nil {
		t.Fatal(err)
	}
	results, err := target.Search([]float64{1, 0}, 1)
	if err != nil || len(results) != 1 || results[0].ID != "a" {
		t.Fatalf("translated target search=%+v err=%v", results, err)
	}
}

func TestServiceReservesCollisionFreeDestinationMappings(t *testing.T) {
	base := t.TempDir()
	target := NewService(ServiceOptions{
		MaxVectorDim: 2, MaxK: 4, SearchMode: "ann",
		VectorStore: "segment", VectorPath: filepath.Join(base, "vectors"),
		SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal.log"),
	})
	t.Cleanup(func() { _ = target.Close() })
	first, err := target.ReserveIDMappings([]string{"source-a/doc", "source-b/doc"})
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 2 || first[0].InternalID == first[1].InternalID {
		t.Fatalf("reservations=%+v", first)
	}
	retry, err := target.ReserveIDMappings([]string{"source-b/doc", "source-a/doc", "source-a/doc"})
	if err != nil {
		t.Fatal(err)
	}
	if len(retry) != 2 || retry[0].InternalID != first[1].InternalID || retry[1].InternalID != first[0].InternalID {
		t.Fatalf("retry=%+v first=%+v", retry, first)
	}
	if _, err := target.ReserveIDMappings([]string{""}); !errors.Is(err, ErrInvalidID) {
		t.Fatalf("invalid reservation error=%v", err)
	}
}

func TestPersistVectorsForPrebuiltANNLeavesANNEmpty(t *testing.T) {
	base := t.TempDir()
	service := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 4, SearchMode: "ann", VectorStore: "segment", VectorPath: filepath.Join(base, "vectors"), SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal")})
	t.Cleanup(func() { _ = service.Close() })
	if err := service.ImportIDMappings([]IDMappingEntry{{ID: "a", InternalID: 1}, {ID: "b", InternalID: 2}}); err != nil {
		t.Fatal(err)
	}
	if err := service.PersistVectorsForPrebuiltANN([]index.Vector{{ID: "a", Values: []float64{1, 0}}, {ID: "b", Values: []float64{0, 1}}}); err != nil {
		t.Fatal(err)
	}
	if got := service.currentANNIndex().Stats().Nodes; got != 0 {
		t.Fatalf("staging populated ANN nodes=%d", got)
	}
	if got, err := service.GetVector("b"); err != nil || got.Values[1] != 1 {
		t.Fatalf("staged vector=%+v err=%v", got, err)
	}
}

func TestPersistVectorsForPrebuiltANNRequiresImportedMappings(t *testing.T) {
	base := t.TempDir()
	service := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 4, SearchMode: "ann", VectorStore: "segment", VectorPath: filepath.Join(base, "vectors"), SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal")})
	t.Cleanup(func() { _ = service.Close() })
	err := service.PersistVectorsForPrebuiltANN([]index.Vector{{ID: "missing", Values: []float64{1, 0}}})
	if err == nil || !strings.Contains(err.Error(), "requires imported ID mappings") {
		t.Fatalf("staging error=%v", err)
	}
	if got := service.ListVectors(); len(got) != 0 {
		t.Fatalf("failed staging persisted vectors=%+v", got)
	}
}

func TestServicePublishesPrebuiltANNAfterVectorStaging(t *testing.T) {
	base := t.TempDir()
	source := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 4, SearchMode: "ann", VectorStore: "segment", VectorPath: filepath.Join(base, "source-vectors"), SnapshotPath: filepath.Join(base, "source.json"), WALPath: filepath.Join(base, "source.wal")})
	target := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 4, SearchMode: "ann", VectorStore: "segment", VectorPath: filepath.Join(base, "target-vectors"), SnapshotPath: filepath.Join(base, "target.json"), WALPath: filepath.Join(base, "target.wal")})
	t.Cleanup(func() { _ = source.Close(); _ = target.Close() })
	vectors := []index.Vector{{ID: "a", Values: []float64{1, 0}}, {ID: "b", Values: []float64{0, 1}}}
	if err := source.AddVectors(vectors); err != nil {
		t.Fatal(err)
	}
	mappings, err := source.ExportIDMappings([]string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if err := target.ImportIDMappings(mappings); err != nil {
		t.Fatal(err)
	}
	if err := target.PersistVectorsForPrebuiltANN(vectors); err != nil {
		t.Fatal(err)
	}
	// Publish one immutable graph rather than a whole segmented container.
	// The source has an unsealed delta here, so make a compact graph directly.
	graph := ann.NewAnnIndexWithOptions(ann.Options{M: 16, EfConstruction: 64, EfSearch: 64})
	for _, mapping := range mappings {
		for _, vector := range vectors {
			if vector.ID == mapping.ID {
				if err := graph.AddVector(mapping.InternalID, vector.Values); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	snapshot, err := graph.MarshalCompactBinary()
	if err != nil {
		t.Fatal(err)
	}
	if err := target.PublishPrebuiltANNSnapshot(snapshot); err != nil {
		t.Fatal(err)
	}
	results, err := target.Search([]float64{1, 0}, 1)
	if err != nil || len(results) != 1 || results[0].ID != "a" {
		t.Fatalf("published search=%+v err=%v", results, err)
	}
}

func TestServicePublishesPrebuiltANNFromStagedFile(t *testing.T) {
	base := t.TempDir()
	service := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 4, SearchMode: "ann", VectorStore: "segment", VectorPath: filepath.Join(base, "vectors"), SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal")})
	t.Cleanup(func() { _ = service.Close() })
	vectors := []index.Vector{{ID: "a", Values: []float64{1, 0}}}
	if err := service.ImportIDMappings([]IDMappingEntry{{ID: "a", InternalID: 7}}); err != nil {
		t.Fatal(err)
	}
	if err := service.PersistVectorsForPrebuiltANN(vectors); err != nil {
		t.Fatal(err)
	}
	graph := ann.NewAnnIndexWithOptions(ann.Options{M: 4, EfConstruction: 16, EfSearch: 16})
	if err := graph.AddVector(7, []float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	payload, err := graph.MarshalCompactBinary()
	if err != nil {
		t.Fatal(err)
	}
	manifest := ann.NewSegmentManifest("segment", ann.SegmentSealed, payload)
	manifest.FirstID, manifest.LastID, manifest.Nodes, manifest.Dimension = 7, 7, 1, 2
	path := filepath.Join(base, "segment.snapshot")
	stager, err := ann.NewSnapshotStager(path, manifest)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stager.Write(payload); err != nil {
		t.Fatal(err)
	}
	if err := stager.Commit(); err != nil {
		t.Fatal(err)
	}
	if err := service.PublishPrebuiltANNSnapshotFile(path, manifest); err != nil {
		t.Fatal(err)
	}
	results, err := service.Search([]float64{1, 0}, 1)
	if err != nil || len(results) != 1 || results[0].ID != "a" {
		t.Fatalf("results=%+v err=%v", results, err)
	}
}

func TestServiceRejectsPrebuiltSnapshotWithMissingDestinationIdentity(t *testing.T) {
	base := t.TempDir()
	service := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 4, SearchMode: "ann", VectorStore: "segment", VectorPath: filepath.Join(base, "vectors"), SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal")})
	t.Cleanup(func() { _ = service.Close() })
	graph := ann.NewAnnIndexWithOptions(ann.Options{M: 4, EfConstruction: 16, EfSearch: 16})
	if err := graph.AddVector(7, []float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	payload, err := graph.MarshalCompactBinary()
	if err != nil {
		t.Fatal(err)
	}
	manifest := ann.NewSegmentManifest("segment", ann.SegmentSealed, payload)
	manifest.FirstID, manifest.LastID, manifest.Nodes, manifest.Dimension = 7, 7, 1, 2
	path := filepath.Join(base, "segment.snapshot")
	stager, err := ann.NewSnapshotStager(path, manifest)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stager.Write(payload); err != nil {
		t.Fatal(err)
	}
	if err := stager.Commit(); err != nil {
		t.Fatal(err)
	}
	if err := service.PublishPrebuiltANNSnapshotFile(path, manifest); err == nil || !strings.Contains(err.Error(), "unmapped internal ID") {
		t.Fatalf("publish error=%v", err)
	}
	if got := service.currentANNIndex().Stats().Nodes; got != 0 {
		t.Fatalf("rejected snapshot changed ANN nodes=%d", got)
	}
}

func TestANNProfilesApplyAdaptiveDefaults(t *testing.T) {
	fast := applyANNProfile("fast", ann.Options{})
	quality := applyANNProfile("quality", ann.Options{})
	balanced := applyANNProfile("balanced", ann.Options{})
	if fast.M != 8 || fast.EfSearch != 32 || quality.M != 32 || quality.EfSearch != 128 || balanced.M != 16 {
		t.Fatalf("fast=%+v quality=%+v balanced=%+v", fast, quality, balanced)
	}
	override := applyANNProfile("fast", ann.Options{M: 20, EfSearch: 90, QuantizeSegments: true})
	if override.M != 20 || override.EfSearch != 90 || override.EfConstruction != 32 || !override.QuantizeSegments {
		t.Fatalf("override=%+v", override)
	}
}

func (s *stubPersistence) SaveSnapshot(vectors []index.Vector) error {
	if s.saveSnapshotFn != nil {
		return s.saveSnapshotFn(vectors)
	}
	return nil
}

type stubVectorStore struct {
	upsertFn func(index.Vector) error
	getFn    func(string) (index.Vector, error)
	deleteFn func(string) error
	listFn   func() []index.Vector
}

func (s *stubVectorStore) UpsertVector(vec index.Vector) error {
	if s.upsertFn != nil {
		return s.upsertFn(vec)
	}
	return nil
}

func (s *stubVectorStore) GetVector(id string) (index.Vector, error) {
	if s.getFn != nil {
		return s.getFn(id)
	}
	return index.Vector{}, index.ErrVectorNotFound
}

func (s *stubVectorStore) DeleteVector(id string) error {
	if s.deleteFn != nil {
		return s.deleteFn(id)
	}
	return nil
}

func (s *stubVectorStore) ListVectors() []index.Vector {
	if s.listFn != nil {
		return s.listFn()
	}
	return nil
}

type stubIndex struct {
	addFn    func(index.Vector) error
	getFn    func(string) (index.Vector, error)
	deleteFn func(string) error
	listFn   func() []index.Vector
	rangeFn  func(func(index.Vector) bool)
}

func (s *stubIndex) AddVector(vec index.Vector) error {
	if s.addFn != nil {
		return s.addFn(vec)
	}
	return nil
}

func (s *stubIndex) SearchVector(id string) (index.Vector, error) {
	if s.getFn != nil {
		return s.getFn(id)
	}
	return index.Vector{}, index.ErrVectorNotFound
}

func (s *stubIndex) DeleteVector(id string) error {
	if s.deleteFn != nil {
		return s.deleteFn(id)
	}
	return nil
}

func (s *stubIndex) ListVectors() []index.Vector {
	if s.listFn != nil {
		return s.listFn()
	}
	return nil
}

func (s *stubIndex) RangeVectors(fn func(index.Vector) bool) {
	if s.rangeFn != nil {
		s.rangeFn(fn)
	}
}

func (s *stubPersistence) LoadSnapshot() (map[string][]float64, error) {
	if s.loadSnapshotFn != nil {
		return s.loadSnapshotFn()
	}
	return nil, nil
}

func (s *stubPersistence) AppendWAL(op walOp) error {
	if s.appendWALFn != nil {
		return s.appendWALFn(op)
	}
	return nil
}

func (s *stubPersistence) ReplayWAL(apply func(walOp) error) error {
	if s.replayWALFn != nil {
		return s.replayWALFn(apply)
	}
	return nil
}

func (s *stubPersistence) TruncateWAL() error {
	if s.truncateWALFn != nil {
		return s.truncateWALFn()
	}
	return nil
}

func newCoreService(t *testing.T, mode string) *Service {
	t.Helper()
	base := t.TempDir()
	return NewService(ServiceOptions{
		MaxVectorDim:  8,
		MaxK:          5,
		SnapshotPath:  filepath.Join(base, "snapshot.json"),
		WALPath:       filepath.Join(base, "wal.log"),
		SnapshotEvery: 2,
		SearchMode:    mode,
	})
}

func TestServiceAddGetDelete(t *testing.T) {
	svc := newCoreService(t, "exact")
	if err := svc.AddVector("a", []float64{1, 2, 3}); err != nil {
		t.Fatalf("AddVector() error = %v", err)
	}
	vec, err := svc.GetVector("a")
	if err != nil || vec.ID != "a" {
		t.Fatal("expected stored vector")
	}
	if err := svc.DeleteVector("a"); err != nil {
		t.Fatalf("DeleteVector() error = %v", err)
	}
	if _, err := svc.GetVector("a"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("expected ErrVectorNotFound, got %v", err)
	}
}

func TestServiceAddVectorsStreamProcessesAndReleasesBatches(t *testing.T) {
	svc := newCoreService(t, "exact")
	batches := [][]index.Vector{
		{{ID: "a", Values: []float64{1, 0, 0}}},
		{{ID: "b", Values: []float64{0, 1, 0}}},
	}
	pos := 0
	err := svc.AddVectorsStream(func() ([]index.Vector, error) {
		if pos == len(batches) {
			return nil, nil
		}
		batch := batches[pos]
		pos++
		return batch, nil
	})
	if err != nil {
		t.Fatalf("AddVectorsStream() error = %v", err)
	}
	if pos != len(batches) {
		t.Fatalf("source called %d times, want %d", pos, len(batches))
	}
	for _, id := range []string{"a", "b"} {
		if _, err := svc.GetVector(id); err != nil {
			t.Fatalf("GetVector(%q) error = %v", id, err)
		}
	}
}

func TestServiceAddVectorsStream32ProcessesBatches(t *testing.T) {
	svc := newCoreService(t, "segment")
	pos := 0
	err := svc.AddVectorsStream32(func() ([]vectorBatch32, error) {
		if pos > 0 {
			return nil, nil
		}
		pos++
		return []vectorBatch32{
			{ID: "f32-a", Values: []float32{1, 0, 0}},
			{ID: "f32-b", Values: []float32{0, 1, 0}},
		}, nil
	})
	if err != nil {
		t.Fatalf("AddVectorsStream32() error = %v", err)
	}
	for _, id := range []string{"f32-a", "f32-b"} {
		if _, err := svc.GetVector(id); err != nil {
			t.Fatalf("GetVector(%q) error = %v", id, err)
		}
	}
}

func TestServiceListVectorsEmptySortedAndAfterDelete(t *testing.T) {
	svc := newCoreService(t, "exact")
	if got := svc.ListVectors(); len(got) != 0 {
		t.Fatalf("expected no vectors, got %d", len(got))
	}
	if err := svc.AddVectors([]index.Vector{
		{ID: "charlie", Values: []float64{1, 0, 0}},
		{ID: "alpha", Values: []float64{0, 1, 0}},
		{ID: "bravo", Values: []float64{0, 0, 1}},
	}); err != nil {
		t.Fatalf("AddVectors: %v", err)
	}

	list := svc.ListVectors()
	if len(list) != 3 {
		t.Fatalf("len=%d, want 3", len(list))
	}
	want := []struct {
		id string
		v  []float64
	}{
		{"alpha", []float64{0, 1, 0}},
		{"bravo", []float64{0, 0, 1}},
		{"charlie", []float64{1, 0, 0}},
	}
	for i := range want {
		if list[i].ID != want[i].id {
			t.Fatalf("index %d: id %q, want %q", i, list[i].ID, want[i].id)
		}
		if !slices.Equal(list[i].Values, want[i].v) {
			t.Fatalf("index %d: values %v, want %v", i, list[i].Values, want[i].v)
		}
	}

	if err := svc.DeleteVector("bravo"); err != nil {
		t.Fatalf("DeleteVector: %v", err)
	}
	list = svc.ListVectors()
	if len(list) != 2 || list[0].ID != "alpha" || list[1].ID != "charlie" {
		t.Fatalf("after delete: %+v", list)
	}
	if !slices.Equal(list[0].Values, []float64{0, 1, 0}) || !slices.Equal(list[1].Values, []float64{1, 0, 0}) {
		t.Fatalf("unexpected values after delete: %+v", list)
	}
}

func TestServiceDeleteVectorCompactsANNAfterTombstoneThreshold(t *testing.T) {
	svc := NewServiceWithDeps(ServiceOptions{
		MaxVectorDim:  2,
		MaxK:          5,
		SearchMode:    "ann",
		SnapshotEvery: annDeleteRebuildMinDeleted * 2,
		ANNOptions: ann.Options{
			M:              4,
			EfConstruction: 8,
			EfSearch:       8,
			Seed:           7,
		},
	}, ServiceDeps{Persistence: &noopPersistence{}})

	for i := 0; i < annDeleteRebuildMinDeleted; i++ {
		id := fmt.Sprintf("doc-%04d", i)
		if err := svc.AddVector(id, []float64{float64(i), 0}); err != nil {
			t.Fatalf("AddVector(%q): %v", id, err)
		}
	}

	for i := 0; i < annDeleteRebuildMinDeleted-1; i++ {
		id := fmt.Sprintf("doc-%04d", i)
		if err := svc.DeleteVector(id); err != nil {
			t.Fatalf("DeleteVector(%q): %v", id, err)
		}
	}
	stats := svc.Stats()
	if stats.ANNDeleted != annDeleteRebuildMinDeleted-1 {
		t.Fatalf("ANNDeleted before threshold = %d, want %d", stats.ANNDeleted, annDeleteRebuildMinDeleted-1)
	}

	lastID := fmt.Sprintf("doc-%04d", annDeleteRebuildMinDeleted-1)
	if err := svc.DeleteVector(lastID); err != nil {
		t.Fatalf("DeleteVector(%q): %v", lastID, err)
	}
	stats = svc.Stats()
	if stats.ANNDeleted != 0 || stats.ANNNodes != 0 {
		t.Fatalf("stats after compaction = %+v, want empty ANN without tombstones", stats)
	}
}

func TestServiceListVectorsSortsLexicographic(t *testing.T) {
	svc := NewServiceWithDeps(ServiceOptions{
		MaxVectorDim:  8,
		MaxK:          5,
		SnapshotPath:  filepath.Join(t.TempDir(), "snapshot.json"),
		WALPath:       filepath.Join(t.TempDir(), "wal.log"),
		SnapshotEvery: 2,
		SearchMode:    "exact",
	}, ServiceDeps{
		Index: index.NewIndex(),
		VectorStore: &stubVectorStore{
			listFn: func() []index.Vector {
				return []index.Vector{
					{ID: "zebra", Values: []float64{3}},
					{ID: "alpha", Values: []float64{1}},
					{ID: "mike", Values: []float64{2}},
				}
			},
		},
		ANNIndex:    ann.NewAnnIndex(),
		IDResolver:  newMemoryIDResolver(),
		Persistence: &stubPersistence{},
	})

	got := svc.ListVectors()
	if len(got) != 3 {
		t.Fatalf("len=%d, want 3", len(got))
	}
	if got[0].ID != "alpha" || got[1].ID != "mike" || got[2].ID != "zebra" {
		t.Fatalf("got ids %q, %q, %q, want alpha, mike, zebra", got[0].ID, got[1].ID, got[2].ID)
	}
	if got[0].Values[0] != 1 || got[1].Values[0] != 2 || got[2].Values[0] != 3 {
		t.Fatalf("unexpected values order: %+v", got)
	}
}

func TestServiceDuplicateAddPreservesVectorStore(t *testing.T) {
	svc := newCoreService(t, "exact")
	if err := svc.AddVector("doc-1", []float64{1, 2, 3}); err != nil {
		t.Fatalf("AddVector: %v", err)
	}
	if err := svc.AddVector("doc-1", []float64{9, 9, 9}); !errors.Is(err, index.ErrVectorExists) {
		t.Fatalf("expected ErrVectorExists, got %v", err)
	}
	vec, err := svc.GetVector("doc-1")
	if err != nil || len(vec.Values) != 3 || vec.Values[0] != 1 {
		t.Fatalf("expected original vector still readable, got %+v err=%v", vec, err)
	}
}

func TestServiceAddVectorsRollbackOnConflict(t *testing.T) {
	svc := newCoreService(t, "exact")
	_ = svc.AddVector("existing", []float64{1, 2, 3})
	err := svc.AddVectors([]index.Vector{
		{ID: "fresh", Values: []float64{1, 2, 3}},
		{ID: "existing", Values: []float64{4, 5, 6}},
	})
	if !errors.Is(err, index.ErrVectorExists) {
		t.Fatalf("expected ErrVectorExists, got %v", err)
	}
	if _, err := svc.GetVector("fresh"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatal("expected rollback of fresh vector")
	}
}

func TestServiceSearchExactAndBatch(t *testing.T) {
	svc := newCoreService(t, "exact")
	_ = svc.AddVectors([]index.Vector{
		{ID: "a", Values: []float64{1, 2, 3}},
		{ID: "b", Values: []float64{1, 2, 4}},
	})
	results, err := svc.Search([]float64{1, 2, 3.1}, 1)
	if err != nil || len(results) != 1 || results[0].ID != "a" {
		t.Fatal("unexpected exact search results")
	}
	batch, err := svc.SearchBatch([]BatchSearchQuery{
		{ID: "q1", Values: []float64{1, 2, 3.1}, K: 1},
		{Values: []float64{1, 2, 3.9}, K: 1},
	})
	if err != nil || len(batch) != 2 || batch[0].ID != "q1" || batch[1].ID != "query-1" {
		t.Fatal("unexpected batch search results")
	}
}

func TestServiceSearchANNFallbackAndValidation(t *testing.T) {
	svc := newCoreService(t, "ann")
	_ = svc.AddVector("a", []float64{1, 2, 3})
	if _, err := svc.Search([]float64{}, 1); !errors.Is(err, ErrInvalidValues) {
		t.Fatalf("expected ErrInvalidValues, got %v", err)
	}
	if _, err := svc.Search([]float64{1, 2, 3}, 0); !errors.Is(err, ErrInvalidK) {
		t.Fatalf("expected ErrInvalidK, got %v", err)
	}
	results, err := svc.Search([]float64{1, 2, 3}, 1)
	if err != nil || len(results) == 0 {
		t.Fatal("expected ANN search results")
	}
}

func TestTopKAccumulator(t *testing.T) {
	acc := newTopKAccumulator(2)
	acc.Add(SearchResult{ID: "a", Distance: 5})
	acc.Add(SearchResult{ID: "b", Distance: 3})
	acc.Add(SearchResult{ID: "c", Distance: 1})
	results := acc.Results()
	if len(results) != 2 || results[0].ID != "c" {
		t.Fatal("unexpected accumulator results")
	}
}

func TestServiceValidationAndHelpers(t *testing.T) {
	svc := newCoreService(t, "exact")
	if err := svc.AddVectors(nil); !errors.Is(err, ErrInvalidValues) {
		t.Fatalf("expected ErrInvalidValues, got %v", err)
	}
	if err := svc.AddVector("", []float64{1, 2, 3}); !errors.Is(err, ErrInvalidID) {
		t.Fatalf("expected ErrInvalidID, got %v", err)
	}
	if err := svc.AddVector("a", nil); !errors.Is(err, ErrInvalidValues) {
		t.Fatalf("expected ErrInvalidValues, got %v", err)
	}
	if _, err := svc.GetVector(""); !errors.Is(err, ErrInvalidID) {
		t.Fatalf("expected ErrInvalidID, got %v", err)
	}
	if err := svc.DeleteVector(""); !errors.Is(err, ErrInvalidID) {
		t.Fatalf("expected ErrInvalidID, got %v", err)
	}
	if _, err := svc.Search([]float64{1, 2, 3}, 6); !errors.Is(err, ErrKTooHigh) {
		t.Fatalf("expected ErrKTooHigh, got %v", err)
	}
	if _, err := svc.Search([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1); !errors.Is(err, ErrVectorDimTooHigh) {
		t.Fatalf("expected ErrVectorDimTooHigh, got %v", err)
	}
	if normalizeSearchMode("ANN") != "ann" || normalizeSearchMode("other") != "exact" {
		t.Fatal("unexpected normalizeSearchMode")
	}
	if _, err := svc.SearchBatch(nil); !errors.Is(err, ErrInvalidValues) {
		t.Fatalf("expected ErrInvalidValues, got %v", err)
	}
}

func TestServiceSearchBatchANNAndAccumulatorEdgeCases(t *testing.T) {
	svc := newCoreService(t, "ann")
	_ = svc.AddVectors([]index.Vector{
		{ID: "a", Values: []float64{1, 2, 3}},
		{ID: "b", Values: []float64{2, 3, 4}},
		{ID: "c", Values: []float64{3, 4, 5}},
	})
	queries := []BatchSearchQuery{
		{ID: "q-a", Values: []float64{1, 2, 3}, K: 1},
		{ID: "q-b", Values: []float64{2, 3, 4}, K: 1},
		{ID: "q-c", Values: []float64{3, 4, 5}, K: 1},
	}
	got, err := svc.SearchBatch(queries)
	if err != nil || len(got) != len(queries) {
		t.Fatal("expected ANN batch result")
	}
	for i, query := range queries {
		want, err := svc.Search(query.Values, query.K)
		if err != nil {
			t.Fatalf("Search() error = %v", err)
		}
		if got[i].ID != query.ID {
			t.Fatalf("batch result %d ID = %q, want %q", i, got[i].ID, query.ID)
		}
		if len(got[i].Results) != len(want) || got[i].Results[0].ID != want[0].ID {
			t.Fatalf("batch result %d = %+v, want %+v", i, got[i].Results, want)
		}
	}

	acc := newTopKAccumulator(0)
	acc.Add(SearchResult{ID: "x", Distance: 1})
	if len(acc.Results()) != 0 {
		t.Fatal("expected empty accumulator")
	}
}

func TestServiceSearchBatchExactParallelMatchesSerial(t *testing.T) {
	prev := runtime.GOMAXPROCS(4)
	t.Cleanup(func() { runtime.GOMAXPROCS(prev) })

	svc := newCoreService(t, "exact")
	vectors := make([]index.Vector, 0, 12)
	for i := 0; i < 12; i++ {
		vectors = append(vectors, index.Vector{
			ID:     fmt.Sprintf("vec-%02d", i),
			Values: []float64{float64(i), float64(i % 3), 1},
		})
	}
	if err := svc.AddVectors(vectors); err != nil {
		t.Fatalf("AddVectors() error = %v", err)
	}

	queries := make([]BatchSearchQuery, 0, exactBatchDistanceWidth*2+1)
	for i := 0; i < exactBatchDistanceWidth*2+1; i++ {
		queries = append(queries, BatchSearchQuery{
			ID:     fmt.Sprintf("q-%02d", i),
			Values: []float64{float64(i), float64(i % 3), 1},
			K:      1,
		})
	}

	got, err := svc.SearchBatch(queries)
	if err != nil {
		t.Fatalf("SearchBatch() error = %v", err)
	}
	if len(got) != len(queries) {
		t.Fatalf("result count = %d, want %d", len(got), len(queries))
	}
	for i, query := range queries {
		want, err := svc.Search(query.Values, query.K)
		if err != nil {
			t.Fatalf("Search() error = %v", err)
		}
		if got[i].ID != query.ID {
			t.Fatalf("result %d ID = %q, want %q", i, got[i].ID, query.ID)
		}
		if !slices.Equal(got[i].Results, want) {
			t.Fatalf("result %d = %+v, want %+v", i, got[i].Results, want)
		}
	}
}

func TestServiceSearchBatchANNSerialAndErrorBranches(t *testing.T) {
	prev := runtime.GOMAXPROCS(1)
	t.Cleanup(func() { runtime.GOMAXPROCS(prev) })

	svc := newCoreService(t, "ann")
	if err := svc.AddVectors([]index.Vector{
		{ID: "a", Values: []float64{1, 2, 3}},
		{ID: "b", Values: []float64{2, 3, 4}},
	}); err != nil {
		t.Fatalf("AddVectors() error = %v", err)
	}
	got, err := svc.SearchBatch([]BatchSearchQuery{
		{ID: "q-a", Values: []float64{1, 2, 3}, K: 1},
		{ID: "q-b", Values: []float64{2, 3, 4}, K: 1},
	})
	if err != nil {
		t.Fatalf("SearchBatch() error = %v", err)
	}
	if len(got) != 2 || got[0].ID != "q-a" || got[1].ID != "q-b" {
		t.Fatalf("unexpected serial ANN batch results: %+v", got)
	}

	runtime.GOMAXPROCS(4)
	_, err = svc.searchBatchANN([]preparedBatchQuery{
		{id: "bad", vals: nil, acc: newTopKAccumulator(1)},
		{id: "ok", vals: []float64{1, 2, 3}, acc: newTopKAccumulator(1)},
	})
	if !errors.Is(err, ErrInvalidValues) {
		t.Fatalf("expected ErrInvalidValues, got %v", err)
	}
}

func TestServicePersistenceHelpers(t *testing.T) {
	svc := newCoreService(t, "exact")
	if err := svc.AddVector("a", []float64{1, 2, 3}); err != nil {
		t.Fatalf("AddVector() error = %v", err)
	}
	if err := svc.saveSnapshot(); err != nil {
		t.Fatalf("saveSnapshot() error = %v", err)
	}
	if err := svc.truncateWAL(); err != nil {
		t.Fatalf("truncateWAL() error = %v", err)
	}
	if err := svc.loadSnapshot(); err != nil {
		t.Fatalf("loadSnapshot() error = %v", err)
	}
	if err := svc.replayWAL(); err != nil {
		t.Fatalf("replayWAL() error = %v", err)
	}
	internalID := svc.idResolver.Assign("a")
	if internalID == 0 {
		t.Fatal("expected non-zero internal ID")
	}
	if got, ok := svc.idResolver.Lookup(internalID); !ok || got != "a" {
		t.Fatal("expected internal ID lookup to round-trip")
	}
}

func TestServicePersistenceEdgeCases(t *testing.T) {
	svc := newCoreService(t, "exact")

	if err := os.WriteFile(svc.snapshotPath, []byte("{bad"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := svc.loadSnapshot(); err == nil {
		t.Fatal("expected loadSnapshot error")
	}

	if err := os.WriteFile(svc.walPath, []byte("{bad\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := svc.replayWAL(); err == nil {
		t.Fatal("expected replayWAL error")
	}
}

func TestServiceMaybeSnapshotAndRestoreState(t *testing.T) {
	svc := newCoreService(t, "exact")
	svc.snapshotEvery = 1
	if err := svc.AddVector("a", []float64{1, 2, 3}); err != nil {
		t.Fatalf("AddVector() error = %v", err)
	}
	if svc.persistOps != 0 {
		t.Fatal("expected persistOps reset after snapshot")
	}

	restored := &Service{
		index:         index.NewIndex(),
		annIndex:      svc.annIndex,
		maxVectorDim:  svc.maxVectorDim,
		maxK:          svc.maxK,
		snapshotPath:  svc.snapshotPath,
		walPath:       svc.walPath,
		snapshotEvery: svc.snapshotEvery,
		searchMode:    svc.searchMode,
	}
	if err := restored.restoreState(); err != nil {
		t.Fatalf("restoreState() error = %v", err)
	}
}

func TestServiceJSONFailurePathsAndFallback(t *testing.T) {
	svc := newCoreService(t, "exact")
	if err := svc.index.AddVector(index.Vector{ID: "bad", Values: []float64{math.NaN()}}); err != nil {
		t.Fatalf("index.AddVector() error = %v", err)
	}
	if err := svc.vectorStore.UpsertVector(index.Vector{ID: "bad", Values: []float64{math.NaN()}}); err != nil {
		t.Fatalf("vectorStore.UpsertVector() error = %v", err)
	}
	if err := svc.saveSnapshot(); err == nil {
		t.Fatal("expected saveSnapshot error")
	}
	if err := svc.appendWAL(walOp{Op: "upsert", ID: "bad", Values: []float64{math.NaN()}}); err == nil {
		t.Fatal("expected appendWAL error")
	}

	svc2 := newCoreService(t, "ann")
	_ = svc2.AddVector("a", []float64{1, 2, 3})
	svc2.annIndex = ann.NewAnnIndex()
	results, err := svc2.Search([]float64{1, 2, 3}, 1)
	if err != nil || len(results) != 1 {
		t.Fatal("expected exact fallback when ANN path yields nothing")
	}
}

func TestServiceReplayWALBranches(t *testing.T) {
	svc := newCoreService(t, "exact")
	content := []byte(
		"{\"op\":\"upsert\",\"id\":\"a\",\"values\":[1,2,3]}\n" +
			"{\"op\":\"upsert\",\"id\":\"a\",\"values\":[4,5,6]}\n" +
			"{\"op\":\"delete\",\"id\":\"a\"}\n" +
			"{\"op\":\"upsert\",\"id\":\"\",\"values\":[1]}\n",
	)
	if err := os.WriteFile(svc.walPath, content, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := svc.replayWAL(); err != nil {
		t.Fatalf("replayWAL() error = %v", err)
	}
	if _, err := svc.GetVector("a"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatal("expected delete branch to be applied")
	}
}

func TestServiceMissingPersistenceFiles(t *testing.T) {
	svc := newCoreService(t, "exact")
	if err := svc.loadSnapshot(); err != nil {
		t.Fatalf("loadSnapshot() error = %v", err)
	}
	if err := svc.replayWAL(); err != nil {
		t.Fatalf("replayWAL() error = %v", err)
	}
}

func TestServiceAddVectorsValidationBranches(t *testing.T) {
	svc := newCoreService(t, "exact")
	err := svc.AddVectors([]index.Vector{{ID: "", Values: []float64{1}}})
	if !errors.Is(err, ErrInvalidID) {
		t.Fatalf("expected ErrInvalidID, got %v", err)
	}

	err = svc.AddVectors([]index.Vector{{ID: "a", Values: []float64{}}})
	if !errors.Is(err, ErrInvalidValues) {
		t.Fatalf("expected ErrInvalidValues, got %v", err)
	}

	err = svc.AddVectors([]index.Vector{{ID: "a", Values: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}}})
	if !errors.Is(err, ErrVectorDimTooHigh) {
		t.Fatalf("expected ErrVectorDimTooHigh, got %v", err)
	}
}

func TestServiceFilesystemFailureBranches(t *testing.T) {
	svc := newCoreService(t, "exact")
	if err := svc.AddVector("a", []float64{1, 2, 3}); err != nil {
		t.Fatalf("AddVector() error = %v", err)
	}

	dir := t.TempDir()
	svc.snapshotPath = dir
	if err := svc.saveSnapshot(); err == nil {
		t.Fatal("expected saveSnapshot filesystem error")
	}

	svc.walPath = dir
	if err := svc.appendWAL(walOp{Op: "upsert", ID: "a", Values: []float64{1, 2, 3}}); err == nil {
		t.Fatal("expected appendWAL filesystem error")
	}
	if err := svc.truncateWAL(); err == nil {
		t.Fatal("expected truncateWAL filesystem error")
	}
}

func TestServiceDeleteRollbackAndHelpers(t *testing.T) {
	svc := newCoreService(t, "exact")
	if err := svc.AddVector("a", []float64{1, 2, 3}); err != nil {
		t.Fatalf("AddVector() error = %v", err)
	}
	svc.walPath = t.TempDir()
	if err := svc.DeleteVector("a"); err == nil {
		t.Fatal("expected delete persistence error")
	}
	if _, err := svc.GetVector("a"); err != nil {
		t.Fatal("expected rollback restore of vector")
	}

	acc := topKAccumulator{}
	if len(acc.Results()) != 0 {
		t.Fatal("expected empty accumulator results")
	}

	if err := svc.DeleteVector("missing"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("expected ErrVectorNotFound, got %v", err)
	}
}

func TestServiceSearchANNBranches(t *testing.T) {
	svc := newCoreService(t, "ann")
	if results, ok, _ := svc.searchANN([]float64{1, 2, 3}, 1); ok || results != nil {
		t.Fatal("expected ANN miss on empty index")
	}

	_ = svc.AddVector("a", []float64{1, 2, 3})
	if results, ok, _ := svc.searchANN([]float64{1, 2}, 1); ok || results != nil {
		t.Fatal("expected ANN miss on invalid dimension")
	}

	_ = svc.AddVector("b", []float64{1, 2, 4})
	results, ok, _ := svc.searchANN([]float64{1, 2, 3.1}, 1)
	if !ok || len(results) != 1 {
		t.Fatal("expected ANN hit with truncation")
	}
}

func TestServiceStatsTrackANNHitAndFallback(t *testing.T) {
	svc := newCoreService(t, "ann")
	if err := svc.AddVector("a", []float64{1, 2, 3}); err != nil {
		t.Fatal(err)
	}

	if _, err := svc.Search([]float64{1, 2, 3}, 1); err != nil {
		t.Fatal(err)
	}

	svc.annIndex = ann.NewAnnIndex()
	if _, err := svc.Search([]float64{1, 2, 3}, 1); err != nil {
		t.Fatal(err)
	}

	stats := svc.Stats()
	if stats.SearchRequestsTotal != 2 {
		t.Fatalf("expected 2 search requests, got %d", stats.SearchRequestsTotal)
	}
	if stats.ANNSearchesTotal != 2 {
		t.Fatalf("expected 2 ANN searches, got %d", stats.ANNSearchesTotal)
	}
	if stats.ANNSearchHitsTotal != 1 {
		t.Fatalf("expected 1 ANN hit, got %d", stats.ANNSearchHitsTotal)
	}
	if stats.ANNSearchFallbacks != 1 {
		t.Fatalf("expected 1 ANN fallback, got %d", stats.ANNSearchFallbacks)
	}
	if stats.ExactSearchesTotal != 1 {
		t.Fatalf("expected 1 exact search, got %d", stats.ExactSearchesTotal)
	}
	if stats.ANNCandidatesReturned == 0 {
		t.Fatal("expected candidate accounting for ANN hit")
	}
}

func TestServiceStatsTrackANNError(t *testing.T) {
	svc := newCoreService(t, "ann")
	if err := svc.AddVector("a", []float64{1, 2, 3}); err != nil {
		t.Fatal(err)
	}

	if _, ok, _ := svc.searchANN([]float64{1, 2}, 1); ok {
		t.Fatal("expected ANN error path to return miss")
	}

	stats := svc.Stats()
	if stats.ANNSearchErrorsTotal != 1 {
		t.Fatalf("expected 1 ANN error, got %d", stats.ANNSearchErrorsTotal)
	}
	if stats.ANNSearchesTotal != 1 {
		t.Fatalf("expected 1 ANN search, got %d", stats.ANNSearchesTotal)
	}
}

func TestServiceStatsTrackANNEvaluationSample(t *testing.T) {
	svc := NewService(ServiceOptions{
		MaxVectorDim:      8,
		MaxK:              5,
		SnapshotPath:      filepath.Join(t.TempDir(), "snapshot.json"),
		WALPath:           filepath.Join(t.TempDir(), "wal.log"),
		SnapshotEvery:     2,
		SearchMode:        "ann",
		ANNEvalSampleRate: 100,
	})
	if err := svc.AddVectors([]index.Vector{
		{ID: "a", Values: []float64{1, 2, 3}},
		{ID: "b", Values: []float64{1, 2, 4}},
	}); err != nil {
		t.Fatal(err)
	}

	if _, err := svc.Search([]float64{1, 2, 3.1}, 1); err != nil {
		t.Fatal(err)
	}

	stats := svc.Stats()
	if stats.ANNEvalSamplesTotal != 1 {
		t.Fatalf("expected 1 ann eval sample, got %d", stats.ANNEvalSamplesTotal)
	}
	if stats.ANNEvalTop1Matches != 1 {
		t.Fatalf("expected top1 match to be tracked, got %d", stats.ANNEvalTop1Matches)
	}
	if stats.ANNEvalComparedResults != 1 {
		t.Fatalf("expected compared results to be tracked, got %d", stats.ANNEvalComparedResults)
	}
	if stats.ANNEvalOverlapResults != 1 {
		t.Fatalf("expected overlap results to be tracked, got %d", stats.ANNEvalOverlapResults)
	}
}

func TestServiceRestoreStateFailureBranches(t *testing.T) {
	base := t.TempDir()
	svc := &Service{
		index:         index.NewIndex(),
		annIndex:      ann.NewAnnIndex(),
		maxVectorDim:  8,
		maxK:          5,
		snapshotPath:  filepath.Join(base, "snapshot.json"),
		walPath:       filepath.Join(base, "wal.log"),
		snapshotEvery: 2,
		searchMode:    "exact",
	}

	if err := os.WriteFile(svc.snapshotPath, []byte("{bad"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := svc.restoreState(); err == nil {
		t.Fatal("expected restoreState loadSnapshot error")
	}

	if err := os.WriteFile(svc.snapshotPath, []byte(`{"a":[1,2,3]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(svc.walPath, []byte("{bad\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := svc.restoreState(); err == nil {
		t.Fatal("expected restoreState replayWAL error")
	}
}

func TestServiceMaybeSnapshotErrorBranch(t *testing.T) {
	svc := newCoreService(t, "exact")
	svc.snapshotEvery = 1
	svc.snapshotPath = t.TempDir()
	if err := svc.maybeSnapshot(); err == nil {
		t.Fatal("expected maybeSnapshot save error")
	}
}

func TestServiceLoadSnapshotBranches(t *testing.T) {
	svc := newCoreService(t, "exact")
	if err := svc.AddVector("dup", []float64{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(svc.snapshotPath, []byte(`{"dup":[1,2,3],"ok":[4,5,6],"bad":[],"too_big":[1,2,3,4,5,6,7,8,9]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := svc.loadSnapshot(); err != nil {
		t.Fatalf("loadSnapshot() error = %v", err)
	}
	if _, err := svc.GetVector("ok"); err != nil {
		t.Fatal("expected valid snapshot vector")
	}
}

func TestServiceSearchBatchInvalidQuery(t *testing.T) {
	svc := newCoreService(t, "exact")
	if _, err := svc.SearchBatch([]BatchSearchQuery{{ID: "q", Values: nil, K: 1}}); !errors.Is(err, ErrInvalidValues) {
		t.Fatalf("expected ErrInvalidValues, got %v", err)
	}
}

func TestNewServiceWithDeps(t *testing.T) {
	idx := index.NewIndex()
	store := &stubVectorStore{}
	annIdx := ann.NewAnnIndex()
	resolver := newMemoryIDResolver()
	persistence := &stubPersistence{}

	svc := NewServiceWithDeps(ServiceOptions{
		MaxVectorDim:  8,
		MaxK:          5,
		SnapshotPath:  filepath.Join(t.TempDir(), "snapshot.json"),
		WALPath:       filepath.Join(t.TempDir(), "wal.log"),
		SnapshotEvery: 2,
		SearchMode:    "exact",
	}, ServiceDeps{
		Index:       idx,
		VectorStore: store,
		ANNIndex:    annIdx,
		IDResolver:  resolver,
		Persistence: persistence,
	})

	if svc.index != idx {
		t.Fatal("expected injected index to be used")
	}
	if svc.vectorStore != store {
		t.Fatal("expected injected vector store to be used")
	}
	if svc.idResolver != resolver {
		t.Fatal("expected injected ID resolver to be used")
	}
	if svc.persistence != persistence {
		t.Fatal("expected injected persistence to be used")
	}
	if annIdx == nil || svc.annIndex == nil {
		t.Fatal("expected ANN index instances to be available")
	}
}

func TestServiceANNBuilderIsUsedForRebuild(t *testing.T) {
	var calls atomic.Int32
	svc := NewService(ServiceOptions{
		MaxVectorDim: 4,
		MaxK:         5,
		SearchMode:   "ann",
		SnapshotPath: filepath.Join(t.TempDir(), "snapshot.json"),
		WALPath:      filepath.Join(t.TempDir(), "wal.log"),
		ANNBuilder: func(vectors []index.Vector) (ANNIndex, error) {
			calls.Add(1)
			if len(vectors) == 0 {
				return nil, errors.New("no training samples")
			}
			centroids, err := ann.TrainIVFCentroids(func() [][]float64 {
				out := make([][]float64, 0, len(vectors))
				for _, v := range vectors {
					out = append(out, v.Values)
				}
				return out
			}(), 1)
			if err != nil {
				return nil, err
			}
			candidate, err := ann.NewIVFANNIndex(centroids, 1)
			if err != nil {
				return nil, err
			}
			for _, v := range vectors {
				if err := candidate.AddVector(int(calls.Load())+len(v.ID), v.Values); err != nil {
					return nil, err
				}
			}
			return candidate, nil
		},
	})
	defer svc.Close()
	if err := svc.AddVector("v1", []float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	if calls.Load() == 0 {
		t.Fatal("ANN builder was not called")
	}
}

func TestServiceGetVectorUsesVectorStore(t *testing.T) {
	svc := NewServiceWithDeps(ServiceOptions{
		MaxVectorDim:  8,
		MaxK:          5,
		SnapshotPath:  filepath.Join(t.TempDir(), "snapshot.json"),
		WALPath:       filepath.Join(t.TempDir(), "wal.log"),
		SnapshotEvery: 2,
		SearchMode:    "exact",
	}, ServiceDeps{
		Index: index.NewIndex(),
		VectorStore: &stubVectorStore{
			getFn: func(id string) (index.Vector, error) {
				return index.Vector{ID: id, Values: []float64{9, 8, 7}}, nil
			},
		},
		ANNIndex:    ann.NewAnnIndex(),
		IDResolver:  newMemoryIDResolver(),
		Persistence: &stubPersistence{},
	})

	vec, err := svc.GetVector("from-store")
	if err != nil {
		t.Fatal(err)
	}
	if vec.ID != "from-store" || len(vec.Values) != 3 || vec.Values[0] != 9 {
		t.Fatal("expected vector to come from vector store")
	}
}

type approximateANN struct{ *ann.AnnIndex }

func (a *approximateANN) ExactDistances() bool { return false }

func TestServiceSearchANNRescoresApproximateCandidatesFromVectorStore(t *testing.T) {
	resolver := newMemoryIDResolver()
	storeReads := 0
	approx := &approximateANN{AnnIndex: ann.NewAnnIndex()}

	svc := NewServiceWithDeps(ServiceOptions{
		MaxVectorDim:  8,
		MaxK:          5,
		SnapshotPath:  filepath.Join(t.TempDir(), "snapshot.json"),
		WALPath:       filepath.Join(t.TempDir(), "wal.log"),
		SnapshotEvery: 2,
		SearchMode:    "ann",
	}, ServiceDeps{
		Index: index.NewIndex(),
		VectorStore: &stubVectorStore{
			getFn: func(id string) (index.Vector, error) {
				storeReads++
				return index.Vector{ID: id, Values: []float64{1, 2, 3}}, nil
			},
		},
		ANNIndex:    approx,
		IDResolver:  resolver,
		Persistence: &stubPersistence{},
	})
	svc.annIndex = approx
	internalID := resolver.Assign("ann-store")
	if err := svc.annIndex.AddVector(internalID, []float64{1, 2, 3}); err != nil {
		t.Fatal(err)
	}

	results, ok, _ := svc.searchANN([]float64{1, 2, 3}, 1)
	if !ok {
		t.Fatal("expected ANN search to succeed")
	}
	if len(results) != 1 || results[0].ID != "ann-store" {
		t.Fatal("expected ANN search to return rescored candidate")
	}
	if storeReads != 1 {
		t.Fatalf("vector store reads = %d, want 1", storeReads)
	}
}

func TestServiceSearchANNSkipsStoreRescoreForExactIndex(t *testing.T) {
	resolver := newMemoryIDResolver()
	idx := ann.NewAnnIndex()
	internalID := resolver.Assign("exact-ann")
	if err := idx.AddVector(internalID, []float64{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	svc := NewServiceWithDeps(ServiceOptions{MaxVectorDim: 8, MaxK: 5, SearchMode: "ann"}, ServiceDeps{
		Index: index.NewIndex(),
		VectorStore: &stubVectorStore{getFn: func(string) (index.Vector, error) {
			return index.Vector{}, errors.New("exact ANN must not read payload store")
		}},
		ANNIndex: idx, IDResolver: resolver, Persistence: &stubPersistence{},
	})
	svc.annIndex = idx
	results, ok, _ := svc.searchANN([]float64{1, 2, 3}, 1)
	if !ok || len(results) != 1 || results[0].ID != "exact-ann" || results[0].Distance != 0 {
		t.Fatalf("results=%+v ok=%v", results, ok)
	}
}

func TestServiceWithDiskVectorStore(t *testing.T) {
	base := t.TempDir()
	svc := NewService(ServiceOptions{
		MaxVectorDim:  8,
		MaxK:          5,
		SnapshotPath:  filepath.Join(base, "snapshot.json"),
		WALPath:       filepath.Join(base, "wal.log"),
		SnapshotEvery: 2,
		SearchMode:    "ann",
		VectorStore:   "disk",
		VectorPath:    filepath.Join(base, "vectors"),
	})
	t.Cleanup(func() { _ = svc.Close() })

	if err := svc.AddVector("disk-1", []float64{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	vec, err := svc.GetVector("disk-1")
	if err != nil || vec.ID != "disk-1" {
		t.Fatal("expected vector from disk-backed store")
	}
	results, err := svc.Search([]float64{1, 2, 3.1}, 1)
	if err != nil || len(results) != 1 || results[0].ID != "disk-1" {
		t.Fatal("expected search result from disk-backed store")
	}
}

func TestServiceWithDiskVectorStoreRestoresWithoutSnapshotWAL(t *testing.T) {
	base := t.TempDir()
	vectorPath := filepath.Join(base, "vectors")
	snapshotPath := filepath.Join(base, "snapshot.json")
	walPath := filepath.Join(base, "wal.log")

	svc := NewService(ServiceOptions{
		MaxVectorDim:  8,
		MaxK:          5,
		SnapshotPath:  snapshotPath,
		WALPath:       walPath,
		SnapshotEvery: 1,
		SearchMode:    "ann",
		VectorStore:   "disk",
		VectorPath:    vectorPath,
	})
	t.Cleanup(func() { _ = svc.Close() })
	if err := svc.AddVector("disk-restore", []float64{1, 2, 3}); err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(snapshotPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("expected no snapshot file, got %v", err)
	}
	if _, err := os.Stat(walPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("expected no wal file, got %v", err)
	}

	restored := NewService(ServiceOptions{
		MaxVectorDim:  8,
		MaxK:          5,
		SnapshotPath:  snapshotPath,
		WALPath:       walPath,
		SnapshotEvery: 1,
		SearchMode:    "ann",
		VectorStore:   "disk",
		VectorPath:    vectorPath,
	})
	t.Cleanup(func() { _ = restored.Close() })
	vec, err := restored.GetVector("disk-restore")
	if err != nil || vec.ID != "disk-restore" {
		t.Fatal("expected disk-backed vector after restart")
	}
	results, err := restored.Search([]float64{1, 2, 3.1}, 1)
	if err != nil || len(results) != 1 || results[0].ID != "disk-restore" {
		t.Fatal("expected ANN search result after disk-backed restore")
	}
}

func TestServiceConcurrentReadWrite(t *testing.T) {
	svc := newCoreService(t, "ann")
	var wg sync.WaitGroup

	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			id := fmt.Sprintf("vec-%d", n%6)
			values := []float64{float64(n), float64(n + 1), float64(n + 2)}
			_ = svc.AddVector(id, values)
			_, _ = svc.GetVector(id)
			_, _ = svc.Search(values, 1)
		}(i)
	}

	wg.Wait()
}

func TestServiceStatsHelpersAndConfigNormalization(t *testing.T) {
	base := t.TempDir()
	svc := NewService(ServiceOptions{
		MaxVectorDim:      8,
		MaxK:              5,
		SnapshotPath:      filepath.Join(base, "snapshot.json"),
		WALPath:           filepath.Join(base, "wal.log"),
		SnapshotEvery:     2,
		SearchMode:        "ann",
		ANNProfile:        "custom",
		ANNEvalSampleRate: 150,
		Cache: CacheOptions{
			Enabled:  true,
			MaxBytes: 1024,
			MaxItems: 2,
		},
	})

	if err := svc.AddVector("a", []float64{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if _, err := svc.GetVector("a"); err != nil {
		t.Fatal(err)
	}
	if _, err := svc.GetVector("a"); err != nil {
		t.Fatal(err)
	}

	stats := svc.Stats()
	if stats.ANNProfile != "custom" || stats.CacheHitsTotal == 0 {
		t.Fatal("expected stats to expose profile and cache hits")
	}
	if normalizeANNProfile("fast") != "fast" || normalizeANNProfile("quality") != "quality" || normalizeANNProfile("x") != "balanced" {
		t.Fatal("unexpected ANN profile normalization")
	}
	if clampPercent(-1) != 0 || clampPercent(101) != 100 || clampPercent(42) != 42 {
		t.Fatal("unexpected clampPercent result")
	}
	if minInt(1, 2) != 1 || minInt(3, 2) != 2 {
		t.Fatal("unexpected minInt result")
	}
	if svc.currentANNIndex() == nil {
		t.Fatal("expected current ANN index")
	}
}

func TestServicePersistentRestoreAndHelpers(t *testing.T) {
	base := t.TempDir()
	vectorPath := filepath.Join(base, "vectors")
	svc := NewService(ServiceOptions{
		MaxVectorDim:  8,
		MaxK:          5,
		SnapshotPath:  filepath.Join(base, "snapshot.json"),
		WALPath:       filepath.Join(base, "wal.log"),
		SnapshotEvery: 1,
		SearchMode:    "ann",
		VectorStore:   "disk",
		VectorPath:    vectorPath,
	})
	t.Cleanup(func() { _ = svc.Close() })

	if err := svc.AddVector("disk-1", []float64{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if err := svc.restoreState(); err != nil {
		t.Fatal(err)
	}
	if err := svc.saveSnapshot(); err != nil {
		t.Fatal(err)
	}
	if err := svc.truncateWAL(); err != nil {
		t.Fatal(err)
	}
	if backend := svc.persistenceBackend(); backend == nil {
		t.Fatal("expected persistence backend")
	}
	if err := svc.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestServiceCurrentANNIndexAndEnsureRuntimeDepsPaths(t *testing.T) {
	svc := &Service{
		maxVectorDim:  8,
		maxK:          5,
		snapshotPath:  filepath.Join(t.TempDir(), "snapshot.json"),
		walPath:       filepath.Join(t.TempDir(), "wal.log"),
		snapshotEvery: 1,
	}

	if svc.currentANNIndex() == nil {
		t.Fatal("expected ANN index initialization")
	}
	svc.ensureRuntimeDeps()
	if svc.index == nil || svc.vectorStore == nil || svc.idResolver == nil || svc.persistence == nil {
		t.Fatal("expected runtime deps to be initialized")
	}
}

func TestServiceLoadVectorStoreStateAndFactoryDefaults(t *testing.T) {
	base := t.TempDir()
	vectorPath := filepath.Join(base, "vectors")
	store := newFileVectorStore(vectorPath)
	t.Cleanup(func() { _ = store.Close() })
	if err := store.UpsertVector(index.Vector{ID: "doc-1", Values: []float64{1, 2, 3}}); err != nil {
		t.Fatal(err)
	}

	svc := &Service{
		index:         index.NewIndex(),
		maxVectorDim:  8,
		maxK:          5,
		vectorStore:   store,
		idResolver:    newMemoryIDResolver(),
		annOptions:    ann.Options{},
		persistence:   &stubPersistence{},
		snapshotPath:  filepath.Join(base, "snapshot.json"),
		walPath:       filepath.Join(base, "wal.log"),
		snapshotEvery: 1,
	}
	if err := svc.loadVectorStoreState(); err != nil {
		t.Fatal(err)
	}
	if _, err := svc.index.SearchVector("doc-1"); err != nil {
		t.Fatal("expected vector store state to populate index")
	}

	memStore := newDefaultVectorStore("", "")
	if memStore == nil {
		t.Fatal("expected memory vector store default")
	}
}

func TestServiceLoadVectorStoreStateBranchesAndPersistenceSwap(t *testing.T) {
	base := t.TempDir()
	svc := &Service{
		index:        index.NewIndex(),
		maxVectorDim: 3,
		vectorStore: &stubVectorStore{
			listFn: func() []index.Vector {
				return []index.Vector{
					{ID: "ok", Values: []float64{1, 2, 3}},
					{ID: "skip-empty", Values: nil},
					{ID: "skip-big", Values: []float64{1, 2, 3, 4}},
				}
			},
		},
		idResolver:    newMemoryIDResolver(),
		persistence:   newSnapshotWALBackend(filepath.Join(base, "one.json"), filepath.Join(base, "one.log")),
		snapshotPath:  filepath.Join(base, "two.json"),
		walPath:       filepath.Join(base, "two.log"),
		snapshotEvery: 1,
	}
	if err := svc.loadVectorStoreState(); err != nil {
		t.Fatal(err)
	}
	if _, err := svc.index.SearchVector("ok"); err != nil {
		t.Fatal("expected ok vector to load")
	}
	if svc.persistenceBackend() == nil {
		t.Fatal("expected swapped persistence backend")
	}
}

func TestServiceLoadVectorStoreStateIndexErrorAndANNContinue(t *testing.T) {
	base := t.TempDir()
	svcIndexErr := &Service{
		index: &stubIndex{
			addFn: func(index.Vector) error { return errors.New("index add failed") },
		},
		maxVectorDim: 8,
		vectorStore: &stubVectorStore{
			listFn: func() []index.Vector { return []index.Vector{{ID: "a", Values: []float64{1, 2, 3}}} },
		},
		idResolver:    newMemoryIDResolver(),
		persistence:   &stubPersistence{},
		snapshotPath:  filepath.Join(base, "snapshot-a.json"),
		walPath:       filepath.Join(base, "wal-a.log"),
		snapshotEvery: 1,
	}
	if err := svcIndexErr.loadVectorStoreState(); err == nil {
		t.Fatal("expected loadVectorStoreState index error")
	}

	badANN := ann.NewAnnIndex()
	if err := badANN.AddVector(1, []float64{1, 2}); err != nil {
		t.Fatal(err)
	}
	svcANNContinue := &Service{
		index:        index.NewIndex(),
		annIndex:     badANN,
		maxVectorDim: 8,
		vectorStore: &stubVectorStore{
			listFn: func() []index.Vector {
				return []index.Vector{
					{ID: "a", Values: []float64{1, 2, 3}},
					{ID: "b", Values: []float64{4, 5, 6}},
				}
			},
		},
		idResolver:    newMemoryIDResolver(),
		persistence:   &stubPersistence{},
		snapshotPath:  filepath.Join(base, "snapshot-b.json"),
		walPath:       filepath.Join(base, "wal-b.log"),
		snapshotEvery: 1,
	}
	if err := svcANNContinue.loadVectorStoreState(); err != nil {
		t.Fatal(err)
	}
	if _, err := svcANNContinue.index.SearchVector("a"); err != nil {
		t.Fatal("expected vector indexed even when ann add continues")
	}
}

func TestServiceCloseWithoutCloserAndPersistentRestoreError(t *testing.T) {
	svc := &Service{vectorStore: newMemoryVectorStore()}
	if err := svc.Close(); err != nil {
		t.Fatal(err)
	}

	base := t.TempDir()
	bad := &Service{
		index:        index.NewIndex(),
		annIndex:     ann.NewAnnIndex(),
		maxVectorDim: 8,
		maxK:         5,
		vectorStore: &stubVectorStore{
			listFn: func() []index.Vector {
				return []index.Vector{{ID: "too-big", Values: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}}}
			},
		},
		idResolver:    newMemoryIDResolver(),
		persistence:   &stubPersistence{},
		snapshotPath:  filepath.Join(base, "snapshot.json"),
		walPath:       filepath.Join(base, "wal.log"),
		snapshotEvery: 1,
	}
	if err := bad.restoreState(); err != nil {
		t.Fatal(err)
	}
}

func TestServicePersistenceBackendAndFactoryBranches(t *testing.T) {
	base := t.TempDir()
	svc := &Service{
		snapshotPath: filepath.Join(base, "snapshot.json"),
		walPath:      filepath.Join(base, "wal.log"),
	}
	if backend := svc.persistenceBackend(); backend == nil {
		t.Fatal("expected default backend when nil")
	}

	if diskDefault := newDefaultVectorStore("disk", ""); diskDefault == nil {
		t.Fatal("expected disk default store")
	}
}

func TestStorageSecurityOptionsParsing(t *testing.T) {
	opts := storageSecurityOptionsFromStrings(true, "", "")
	if !opts.StrictFilePermissions || opts.DirMode != 0o700 || opts.FileMode != 0o600 {
		t.Fatal("expected strict defaults")
	}

	opts = storageSecurityOptionsFromStrings(false, "0711", "0640")
	if opts.DirMode != 0o711 || opts.FileMode != 0o640 {
		t.Fatal("expected parsed file modes")
	}

	path := filepath.Join(t.TempDir(), "mode.txt")
	if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	mode, err := storagePathMode(path)
	if err != nil {
		t.Fatal(err)
	}
	if mode == 0 {
		t.Fatal("expected non-zero file mode")
	}
}

func TestSnapshotWALBackendUsesConfiguredPermissions(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("permission bits are not reliable on Windows")
	}
	base := t.TempDir()
	storageDir := filepath.Join(base, "secure")
	snapshotPath := filepath.Join(storageDir, "snapshot.json")
	walPath := filepath.Join(storageDir, "wal.log")
	backend := newSnapshotWALBackendWithSecurity(snapshotPath, walPath, StrictStorageSecurityOptions())

	if err := backend.SaveSnapshot([]index.Vector{{ID: "a", Values: []float64{1, 2, 3}}}); err != nil {
		t.Fatal(err)
	}
	if err := backend.AppendWAL(walOp{Op: "upsert", ID: "a", Values: []float64{1, 2, 3}}); err != nil {
		t.Fatal(err)
	}

	dirMode, err := storagePathMode(storageDir)
	if err != nil {
		t.Fatal(err)
	}
	if dirMode != 0o700 {
		t.Fatalf("expected dir mode 0700, got %o", dirMode)
	}

	snapshotMode, err := storagePathMode(snapshotPath)
	if err != nil {
		t.Fatal(err)
	}
	if snapshotMode != 0o600 {
		t.Fatalf("expected snapshot mode 0600, got %o", snapshotMode)
	}

	walMode, err := storagePathMode(walPath)
	if err != nil {
		t.Fatal(err)
	}
	if walMode != 0o600 {
		t.Fatalf("expected wal mode 0600, got %o", walMode)
	}
}

func TestServiceAddVectorsErrorBranches(t *testing.T) {
	base := t.TempDir()

	svcUpsert := NewServiceWithDeps(ServiceOptions{
		MaxVectorDim:  8,
		MaxK:          5,
		SnapshotPath:  filepath.Join(base, "snapshot-upsert.json"),
		WALPath:       filepath.Join(base, "wal-upsert.log"),
		SnapshotEvery: 2,
		SearchMode:    "exact",
	}, ServiceDeps{
		Index: index.NewIndex(),
		VectorStore: &stubVectorStore{
			upsertFn: func(index.Vector) error { return errors.New("upsert failed") },
		},
		ANNIndex:    ann.NewAnnIndex(),
		IDResolver:  newMemoryIDResolver(),
		Persistence: &stubPersistence{},
	})
	if err := svcUpsert.AddVector("a", []float64{1, 2, 3}); err == nil {
		t.Fatal("expected vector store upsert error")
	}

	svcIndex := NewServiceWithDeps(ServiceOptions{
		MaxVectorDim:  8,
		MaxK:          5,
		SnapshotPath:  filepath.Join(base, "snapshot-index.json"),
		WALPath:       filepath.Join(base, "wal-index.log"),
		SnapshotEvery: 2,
		SearchMode:    "exact",
	}, ServiceDeps{
		Index: &stubIndex{
			addFn: func(index.Vector) error { return errors.New("index add failed") },
		},
		VectorStore: newMemoryVectorStore(),
		ANNIndex:    ann.NewAnnIndex(),
		IDResolver:  newMemoryIDResolver(),
		Persistence: &stubPersistence{},
	})
	if err := svcIndex.AddVector("b", []float64{1, 2, 3}); err == nil {
		t.Fatal("expected index add error")
	}

	svcWAL := NewServiceWithDeps(ServiceOptions{
		MaxVectorDim:  8,
		MaxK:          5,
		SnapshotPath:  filepath.Join(base, "snapshot-wal.json"),
		WALPath:       filepath.Join(base, "wal-wal.log"),
		SnapshotEvery: 2,
		SearchMode:    "exact",
	}, ServiceDeps{
		Index:       index.NewIndex(),
		VectorStore: newMemoryVectorStore(),
		ANNIndex:    ann.NewAnnIndex(),
		IDResolver:  newMemoryIDResolver(),
		Persistence: &stubPersistence{
			appendWALFn: func(walOp) error { return errors.New("wal append failed") },
		},
	})
	if err := svcWAL.AddVector("c", []float64{1, 2, 3}); err == nil {
		t.Fatal("expected wal append error")
	}
}

func TestServiceDeleteVectorErrorBranches(t *testing.T) {
	base := t.TempDir()
	svc := NewServiceWithDeps(ServiceOptions{
		MaxVectorDim:  8,
		MaxK:          5,
		SnapshotPath:  filepath.Join(base, "snapshot.json"),
		WALPath:       filepath.Join(base, "wal.log"),
		SnapshotEvery: 2,
		SearchMode:    "exact",
	}, ServiceDeps{
		Index: &stubIndex{
			deleteFn: func(string) error { return errors.New("index delete failed") },
		},
		VectorStore: &stubVectorStore{
			getFn: func(id string) (index.Vector, error) { return index.Vector{ID: id, Values: []float64{1, 2, 3}}, nil },
		},
		ANNIndex:    ann.NewAnnIndex(),
		IDResolver:  newMemoryIDResolver(),
		Persistence: &stubPersistence{},
	})
	if err := svc.DeleteVector("a"); err == nil {
		t.Fatal("expected index delete error")
	}

	svcStoreDelete := NewServiceWithDeps(ServiceOptions{
		MaxVectorDim:  8,
		MaxK:          5,
		SnapshotPath:  filepath.Join(base, "snapshot-store.json"),
		WALPath:       filepath.Join(base, "wal-store.log"),
		SnapshotEvery: 2,
		SearchMode:    "exact",
	}, ServiceDeps{
		Index: index.NewIndex(),
		VectorStore: &stubVectorStore{
			getFn:    func(id string) (index.Vector, error) { return index.Vector{ID: id, Values: []float64{1, 2, 3}}, nil },
			deleteFn: func(string) error { return errors.New("store delete failed") },
		},
		ANNIndex:    ann.NewAnnIndex(),
		IDResolver:  newMemoryIDResolver(),
		Persistence: &stubPersistence{},
	})
	_ = svcStoreDelete.index.AddVector(index.Vector{ID: "a", Values: []float64{1, 2, 3}})
	if err := svcStoreDelete.DeleteVector("a"); err == nil {
		t.Fatal("expected store delete error")
	}
}

func TestServiceLoadSnapshotAndReplayWALErrorBranches(t *testing.T) {
	base := t.TempDir()
	svcSnapshot := &Service{
		index:        index.NewIndex(),
		vectorStore:  &stubVectorStore{upsertFn: func(index.Vector) error { return errors.New("snapshot upsert failed") }},
		idResolver:   newMemoryIDResolver(),
		persistence:  &stubPersistence{loadSnapshotFn: func() (map[string][]float64, error) { return map[string][]float64{"a": {1, 2, 3}}, nil }},
		maxVectorDim: 8,
		maxK:         5,
		snapshotPath: filepath.Join(base, "snapshot.json"),
		walPath:      filepath.Join(base, "wal.log"),
	}
	if err := svcSnapshot.loadSnapshot(); err == nil {
		t.Fatal("expected snapshot upsert error")
	}

	svcReplay := &Service{
		index: index.NewIndex(),
		vectorStore: &stubVectorStore{
			upsertFn: func(index.Vector) error { return errors.New("wal upsert failed") },
		},
		idResolver: newMemoryIDResolver(),
		persistence: &stubPersistence{replayWALFn: func(apply func(walOp) error) error {
			return apply(walOp{Op: "upsert", ID: "a", Values: []float64{1, 2, 3}})
		}},
		maxVectorDim: 8,
		maxK:         5,
		snapshotPath: filepath.Join(base, "snapshot2.json"),
		walPath:      filepath.Join(base, "wal2.log"),
	}
	if err := svcReplay.replayWAL(); err == nil {
		t.Fatal("expected replay wal apply error")
	}
}
