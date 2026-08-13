package core

import (
	"errors"
	"path/filepath"
	"testing"

	"lumenvec/internal/index"
)

// fakeVectorStore errors on UpsertVector for ID "fail", otherwise behaves like memory store
type fakeVectorStore struct {
	store map[string]index.Vector
}

func newFakeVectorStore() *fakeVectorStore {
	return &fakeVectorStore{store: make(map[string]index.Vector)}
}

func (s *fakeVectorStore) UpsertVector(vec index.Vector) error {
	if vec.ID == "fail" {
		return errors.New("upsert failed")
	}
	s.store[vec.ID] = index.Vector{ID: vec.ID, Values: append([]float64(nil), vec.Values...)}
	return nil
}

func (s *fakeVectorStore) GetVector(id string) (index.Vector, error) {
	v, ok := s.store[id]
	if !ok {
		return index.Vector{}, index.ErrVectorNotFound
	}
	return index.Vector{ID: v.ID, Values: append([]float64(nil), v.Values...)}, nil
}

func (s *fakeVectorStore) DeleteVector(id string) error {
	if _, ok := s.store[id]; !ok {
		return index.ErrVectorNotFound
	}
	delete(s.store, id)
	return nil
}

func (s *fakeVectorStore) ListVectors() []index.Vector {
	out := make([]index.Vector, 0, len(s.store))
	for _, v := range s.store {
		out = append(out, index.Vector{ID: v.ID, Values: append([]float64(nil), v.Values...)})
	}
	return out
}

type noopPersistence struct{}

func (n *noopPersistence) SaveSnapshot(v []index.Vector) error         { return nil }
func (n *noopPersistence) LoadSnapshot() (map[string][]float64, error) { return nil, nil }
func (n *noopPersistence) AppendWAL(op walOp) error                    { return nil }
func (n *noopPersistence) ReplayWAL(fn func(walOp) error) error        { return nil }
func (n *noopPersistence) TruncateWAL() error                          { return nil }

func TestAddVectorsRollbackOnUpsertFailure(t *testing.T) {
	fake := newFakeVectorStore()
	svc := NewServiceWithDeps(ServiceOptions{MaxVectorDim: 16, MaxK: 5}, ServiceDeps{VectorStore: fake, Persistence: &noopPersistence{}})

	// first vector should be upserted, second should fail and cause rollback
	vecs := []index.Vector{{ID: "a", Values: []float64{1, 2, 3}}, {ID: "fail", Values: []float64{4, 5, 6}}}
	if err := svc.AddVectors(vecs); err == nil {
		t.Fatal("expected error from AddVectors due to fake Upsert failure")
	}

	// ensure no vectors remain in the store or index after rollback
	got := svc.ListVectors()
	if len(got) != 0 {
		t.Fatalf("expected 0 vectors after rollback, got %d", len(got))
	}
}

func TestListVectorsOrdering(t *testing.T) {
	svc := NewServiceWithDeps(ServiceOptions{MaxVectorDim: 16, MaxK: 5}, ServiceDeps{Persistence: &noopPersistence{}})
	// add out-of-order ids
	if err := svc.AddVector("z", []float64{1}); err != nil {
		t.Fatal(err)
	}
	if err := svc.AddVector("a", []float64{2}); err != nil {
		t.Fatal(err)
	}
	if err := svc.AddVector("m", []float64{3}); err != nil {
		t.Fatal(err)
	}

	list := svc.ListVectors()
	if len(list) != 3 {
		t.Fatalf("expected 3 vectors, got %d", len(list))
	}
	if list[0].ID != "a" || list[1].ID != "m" || list[2].ID != "z" {
		t.Fatalf("expected sorted ids a,m,z got %+v", list)
	}
}

type pagingFakeStore struct {
	store     map[string]index.Vector
	listCalls int
	pageCalls int
	getCalls  []string
}

func newPagingFakeStore() *pagingFakeStore {
	return &pagingFakeStore{store: make(map[string]index.Vector)}
}

func (s *pagingFakeStore) UpsertVector(vec index.Vector) error {
	s.store[vec.ID] = index.Vector{ID: vec.ID, Values: append([]float64(nil), vec.Values...)}
	return nil
}

func (s *pagingFakeStore) GetVector(id string) (index.Vector, error) {
	s.getCalls = append(s.getCalls, id)
	vec, ok := s.store[id]
	if !ok {
		return index.Vector{}, index.ErrVectorNotFound
	}
	return index.Vector{ID: vec.ID, Values: append([]float64(nil), vec.Values...)}, nil
}

func (s *pagingFakeStore) DeleteVector(id string) error {
	if _, ok := s.store[id]; !ok {
		return index.ErrVectorNotFound
	}
	delete(s.store, id)
	return nil
}

func (s *pagingFakeStore) ListVectors() []index.Vector {
	s.listCalls++
	out := make([]index.Vector, 0, len(s.store))
	for _, vec := range s.store {
		out = append(out, index.Vector{ID: vec.ID, Values: append([]float64(nil), vec.Values...)})
	}
	return out
}

func (s *pagingFakeStore) RangeVectorIDs(fn func(string) bool) {
	for id := range s.store {
		if !fn(id) {
			return
		}
	}
}

func (s *pagingFakeStore) PageVectorIDs(afterID string, limit int) []string {
	s.pageCalls++
	return selectPageIDsFromRange(afterID, limit, s.RangeVectorIDs)
}

func TestListVectorsPageUsesBoundedIDSelection(t *testing.T) {
	store := newPagingFakeStore()
	svc := NewServiceWithDeps(ServiceOptions{MaxVectorDim: 16, MaxK: 5}, ServiceDeps{VectorStore: store, Persistence: &noopPersistence{}})
	for _, vec := range []index.Vector{
		{ID: "d", Values: []float64{4}},
		{ID: "a", Values: []float64{1}},
		{ID: "c", Values: []float64{3}},
		{ID: "b", Values: []float64{2}},
	} {
		if err := svc.AddVector(vec.ID, vec.Values); err != nil {
			t.Fatalf("AddVector(%q): %v", vec.ID, err)
		}
	}
	store.listCalls = 0
	store.getCalls = nil

	page := svc.ListVectorsPage(ListVectorsOptions{AfterID: "a", Limit: 2})
	if store.listCalls != 0 {
		t.Fatalf("ListVectorsPage called ListVectors %d times; want 0", store.listCalls)
	}
	if store.pageCalls != 1 {
		t.Fatalf("ListVectorsPage called PageVectorIDs %d times; want 1", store.pageCalls)
	}
	if len(page.Vectors) != 2 || page.Vectors[0].ID != "b" || page.Vectors[1].ID != "c" {
		t.Fatalf("unexpected page: %+v", page)
	}
	if page.NextCursor != "c" {
		t.Fatalf("NextCursor = %q, want c", page.NextCursor)
	}
	if len(store.getCalls) != 2 {
		t.Fatalf("GetVector calls = %d, want 2", len(store.getCalls))
	}

	store.getCalls = nil
	idsOnly := svc.ListVectorsPage(ListVectorsOptions{AfterID: "b0", Limit: 2, IDsOnly: true})
	if len(idsOnly.Vectors) != 2 || idsOnly.Vectors[0].ID != "c" || idsOnly.Vectors[1].ID != "d" {
		t.Fatalf("unexpected ids_only page: %+v", idsOnly)
	}
	if len(store.getCalls) != 0 {
		t.Fatalf("ids_only GetVector calls = %d, want 0", len(store.getCalls))
	}
}

type listOnlyPagingStore struct {
	vectors   []index.Vector
	listCalls int
	getCalls  []string
}

func (s *listOnlyPagingStore) UpsertVector(vec index.Vector) error {
	s.vectors = append(s.vectors, index.Vector{ID: vec.ID, Values: append([]float64(nil), vec.Values...)})
	return nil
}

func (s *listOnlyPagingStore) GetVector(id string) (index.Vector, error) {
	s.getCalls = append(s.getCalls, id)
	for _, vec := range s.vectors {
		if vec.ID == id {
			return index.Vector{ID: vec.ID, Values: append([]float64(nil), vec.Values...)}, nil
		}
	}
	return index.Vector{}, index.ErrVectorNotFound
}

func (s *listOnlyPagingStore) DeleteVector(id string) error {
	for i, vec := range s.vectors {
		if vec.ID == id {
			s.vectors = append(s.vectors[:i], s.vectors[i+1:]...)
			return nil
		}
	}
	return index.ErrVectorNotFound
}

func (s *listOnlyPagingStore) ListVectors() []index.Vector {
	s.listCalls++
	out := make([]index.Vector, 0, len(s.vectors))
	for _, vec := range s.vectors {
		out = append(out, index.Vector{ID: vec.ID, Values: append([]float64(nil), vec.Values...)})
	}
	return out
}

func TestListVectorsPageFallbackAndZeroLimit(t *testing.T) {
	store := &listOnlyPagingStore{vectors: []index.Vector{
		{ID: "d", Values: []float64{4}},
		{ID: "a", Values: []float64{1}},
		{ID: "c", Values: []float64{3}},
		{ID: "b", Values: []float64{2}},
	}}
	svc := &Service{vectorStore: store}

	if page := svc.ListVectorsPage(ListVectorsOptions{Limit: 0}); len(page.Vectors) != 0 || page.NextCursor != "" {
		t.Fatalf("zero-limit page = %+v, want empty", page)
	}

	page := svc.ListVectorsPage(ListVectorsOptions{Limit: 2})
	if store.listCalls != 1 {
		t.Fatalf("fallback ListVectors calls = %d, want 1", store.listCalls)
	}
	if len(page.Vectors) != 2 || page.Vectors[0].ID != "a" || page.Vectors[1].ID != "b" {
		t.Fatalf("unexpected fallback page: %+v", page)
	}
	if page.NextCursor != "b" {
		t.Fatalf("NextCursor = %q, want b", page.NextCursor)
	}
	if len(store.getCalls) != 2 {
		t.Fatalf("GetVector calls = %d, want 2", len(store.getCalls))
	}
}

func TestVectorStoresRangeVectorIDs(t *testing.T) {
	memory := newMemoryVectorStore()
	if err := memory.UpsertVector(index.Vector{ID: "a", Values: []float64{1}}); err != nil {
		t.Fatal(err)
	}
	if err := memory.UpsertVector(index.Vector{ID: "b", Values: []float64{2}}); err != nil {
		t.Fatal(err)
	}
	seen := 0
	memory.RangeVectorIDs(func(string) bool {
		seen++
		return false
	})
	if seen != 1 {
		t.Fatalf("memory RangeVectorIDs saw %d ids before stop, want 1", seen)
	}
	values32, err := memory.GetVectorReadOnly32("a")
	if err != nil {
		t.Fatal(err)
	}
	if len(values32) != 1 || values32[0] != 1 {
		t.Fatalf("memory GetVectorReadOnly32 = %v", values32)
	}
	readOnly, err := memory.GetVectorReadOnly("a")
	if err != nil {
		t.Fatal(err)
	}
	if len(readOnly.Values) != 1 || readOnly.Values[0] != 1 {
		t.Fatalf("memory GetVectorReadOnly = %+v", readOnly)
	}
	memory.ids = nil
	if got := memory.PageVectorIDs("", 2); len(got) != 2 {
		t.Fatalf("memory PageVectorIDs rebuilt index = %v, want 2 ids", got)
	}

	fileStore := newFileVectorStore(filepath.Join(t.TempDir(), "vectors"))
	t.Cleanup(func() { _ = fileStore.Close() })
	if err := fileStore.UpsertVector(index.Vector{ID: "live", Values: []float64{1}}); err != nil {
		t.Fatal(err)
	}
	fileStore.offsets["deleted"] = fileVectorRecordMeta{deleted: true}
	ids := make(map[string]bool)
	fileStore.RangeVectorIDs(func(id string) bool {
		ids[id] = true
		return true
	})
	if !ids["live"] || ids["deleted"] {
		t.Fatalf("file RangeVectorIDs ids = %+v, want live only", ids)
	}

	closedFileStore := &fileVectorStore{}
	closedFileStore.RangeVectorIDs(func(string) bool {
		t.Fatal("closed file store should not iterate")
		return true
	})
}

func TestVectorStoresPageVectorIDs(t *testing.T) {
	memory := newMemoryVectorStore()
	for _, id := range []string{"d", "a", "c", "b"} {
		if err := memory.UpsertVector(index.Vector{ID: id, Values: []float64{1}}); err != nil {
			t.Fatal(err)
		}
	}
	got := memory.PageVectorIDs("a", 2)
	if len(got) != 2 || got[0] != "b" || got[1] != "c" {
		t.Fatalf("memory PageVectorIDs = %v, want [b c]", got)
	}

	fileStore := newFileVectorStore(filepath.Join(t.TempDir(), "vectors"))
	t.Cleanup(func() { _ = fileStore.Close() })
	for _, id := range []string{"d", "a", "c", "b"} {
		if err := fileStore.UpsertVector(index.Vector{ID: id, Values: []float64{1}}); err != nil {
			t.Fatal(err)
		}
	}
	got = fileStore.PageVectorIDs("b", 3)
	if len(got) != 2 || got[0] != "c" || got[1] != "d" {
		t.Fatalf("file PageVectorIDs = %v, want [c d]", got)
	}
}

func TestCachedVectorStoreRangeVectorIDs(t *testing.T) {
	rangedBackend := newMemoryVectorStore()
	if err := rangedBackend.UpsertVector(index.Vector{ID: "a", Values: []float64{1}}); err != nil {
		t.Fatal(err)
	}
	cached := newCachedVectorStore(rangedBackend, CacheOptions{Enabled: true}).(*cachedVectorStore)
	var rangedIDs []string
	cached.RangeVectorIDs(func(id string) bool {
		rangedIDs = append(rangedIDs, id)
		return true
	})
	if len(rangedIDs) != 1 || rangedIDs[0] != "a" {
		t.Fatalf("cached ranged IDs = %v, want [a]", rangedIDs)
	}

	fallbackBackend := newCountingVectorStore()
	if err := fallbackBackend.UpsertVector(index.Vector{ID: "b", Values: []float64{2}}); err != nil {
		t.Fatal(err)
	}
	cachedFallback := newCachedVectorStore(fallbackBackend, CacheOptions{Enabled: true}).(*cachedVectorStore)
	var fallbackIDs []string
	cachedFallback.RangeVectorIDs(func(id string) bool {
		fallbackIDs = append(fallbackIDs, id)
		return false
	})
	if len(fallbackIDs) != 1 || fallbackIDs[0] != "b" {
		t.Fatalf("cached fallback IDs = %v, want [b]", fallbackIDs)
	}
}

func TestValidateSearchRequestErrors(t *testing.T) {
	svc := NewServiceWithDeps(ServiceOptions{MaxVectorDim: 2, MaxK: 5}, ServiceDeps{Persistence: &noopPersistence{}})

	if err := svc.validateSearchRequest([]float64{}, 1); !errors.Is(err, ErrInvalidValues) {
		t.Fatalf("expected ErrInvalidValues, got %v", err)
	}
	if err := svc.validateSearchRequest([]float64{1}, 0); !errors.Is(err, ErrInvalidK) {
		t.Fatalf("expected ErrInvalidK, got %v", err)
	}
	if err := svc.validateSearchRequest([]float64{1, 2, 3}, 1); !errors.Is(err, ErrVectorDimTooHigh) {
		t.Fatalf("expected ErrVectorDimTooHigh, got %v", err)
	}
	if err := svc.validateSearchRequest([]float64{1, 2}, 10); !errors.Is(err, ErrKTooHigh) {
		t.Fatalf("expected ErrKTooHigh, got %v", err)
	}
}

func TestTopKAccumulatorBehavior(t *testing.T) {
	acc := newTopKAccumulator(2)
	acc.Add(SearchResult{ID: "a", Distance: 5})
	acc.Add(SearchResult{ID: "b", Distance: 3})
	acc.Add(SearchResult{ID: "c", Distance: 4})
	res := acc.Results()
	if len(res) != 2 {
		t.Fatalf("expected 2 results, got %d", len(res))
	}
	if !(res[0].Distance <= res[1].Distance) {
		t.Fatalf("expected results sorted ascending, got %+v", res)
	}

	acc0 := newTopKAccumulator(0)
	acc0.Add(SearchResult{ID: "x", Distance: 1})
	if len(acc0.Results()) != 0 {
		t.Fatalf("expected 0 results for limit 0 accumulator")
	}
}

// store implementing extra interfaces for Stats and disk statistics
type richFakeStore struct {
	mem map[string]index.Vector
}

func newRichFakeStore() *richFakeStore                       { return &richFakeStore{mem: make(map[string]index.Vector)} }
func (s *richFakeStore) UpsertVector(vec index.Vector) error { s.mem[vec.ID] = vec; return nil }
func (s *richFakeStore) GetVector(id string) (index.Vector, error) {
	v, ok := s.mem[id]
	if !ok {
		return index.Vector{}, index.ErrVectorNotFound
	}
	return v, nil
}
func (s *richFakeStore) DeleteVector(id string) error {
	if _, ok := s.mem[id]; !ok {
		return index.ErrVectorNotFound
	}
	delete(s.mem, id)
	return nil
}
func (s *richFakeStore) ListVectors() []index.Vector {
	out := make([]index.Vector, 0, len(s.mem))
	for _, v := range s.mem {
		out = append(out, v)
	}
	return out
}
func (s *richFakeStore) Stats() CacheStats {
	return CacheStats{Hits: 5, Misses: 2, Evictions: 1, Items: uint64(len(s.mem)), Bytes: 123}
}
func (s *richFakeStore) DiskStats() DiskStoreStats {
	return DiskStoreStats{FileBytes: 42, Records: uint64(len(s.mem)), StaleRecords: 0, Compactions: 0}
}
func (s *richFakeStore) IsPersistent() bool { return true }
func (s *richFakeStore) Close() error       { return nil }

func TestServiceStatsAndPersistence(t *testing.T) {
	rich := newRichFakeStore()
	svc := NewServiceWithDeps(ServiceOptions{MaxVectorDim: 8, MaxK: 5}, ServiceDeps{VectorStore: rich, Persistence: &noopPersistence{}})

	// add a vector so stats report non-zero items
	if err := svc.AddVector("x", []float64{1, 2}); err != nil {
		t.Fatal(err)
	}

	stats := svc.Stats()
	if stats.CacheHitsTotal == 0 || stats.CacheMissesTotal == 0 {
		t.Fatalf("expected cache stats to be populated, got %+v", stats)
	}
	if !svc.usesPersistentVectorStore() {
		t.Fatalf("expected store to be treated as persistent")
	}
	if err := svc.Close(); err != nil {
		t.Fatalf("expected close to succeed, got %v", err)
	}
}

type testPersistenceBackend struct {
	saved     bool
	appended  bool
	truncated bool
}

func (p *testPersistenceBackend) SaveSnapshot(v []index.Vector) error         { p.saved = true; return nil }
func (p *testPersistenceBackend) LoadSnapshot() (map[string][]float64, error) { return nil, nil }
func (p *testPersistenceBackend) AppendWAL(op walOp) error                    { p.appended = true; return nil }
func (p *testPersistenceBackend) ReplayWAL(fn func(walOp) error) error        { return nil }
func (p *testPersistenceBackend) TruncateWAL() error                          { p.truncated = true; return nil }

func TestMaybeSnapshotAndPersistenceCalls(t *testing.T) {
	tp := &testPersistenceBackend{}
	svc := NewServiceWithDeps(ServiceOptions{MaxVectorDim: 16, MaxK: 5, SnapshotEvery: 1}, ServiceDeps{Persistence: tp})

	if err := svc.AddVector("p1", []float64{1}); err != nil {
		t.Fatal(err)
	}
	if !tp.appended {
		t.Fatal("expected AppendWAL to be called")
	}
	if !tp.saved {
		t.Fatal("expected SaveSnapshot to be called")
	}
	if !tp.truncated {
		t.Fatal("expected TruncateWAL to be called")
	}
}

type batchRecordingStore struct {
	*memoryVectorStore
	batches int
}

func newBatchRecordingStore() *batchRecordingStore {
	return &batchRecordingStore{memoryVectorStore: newMemoryVectorStore()}
}

func (s *batchRecordingStore) UpsertVectors(vectors []index.Vector) error {
	s.batches++
	return s.memoryVectorStore.UpsertVectors(vectors)
}

type batchRecordingPersistence struct {
	testPersistenceBackend
	appends int
	batches int
	syncs   int
}

func (p *batchRecordingPersistence) AppendWAL(op walOp) error {
	p.appends++
	return nil
}

func (p *batchRecordingPersistence) AppendWALBatch(ops []walOp) error {
	p.batches++
	return nil
}

func (p *batchRecordingPersistence) Sync() error {
	p.syncs++
	return nil
}

func TestServiceAddVectorsUsesBatchStoreAndWALBatch(t *testing.T) {
	store := newBatchRecordingStore()
	persistence := &batchRecordingPersistence{}
	svc := NewServiceWithDeps(ServiceOptions{MaxVectorDim: 16, MaxK: 5}, ServiceDeps{
		VectorStore: store,
		Persistence: persistence,
	})

	if err := svc.AddVectors([]index.Vector{
		{ID: "a", Values: []float64{1}},
		{ID: "b", Values: []float64{2}},
	}); err != nil {
		t.Fatal(err)
	}
	if store.batches != 1 {
		t.Fatalf("store batches = %d, want 1", store.batches)
	}
	if persistence.batches != 1 || persistence.appends != 0 {
		t.Fatalf("wal batches=%d appends=%d, want one batch and zero single appends", persistence.batches, persistence.appends)
	}
	if persistence.syncs == 0 {
		t.Fatal("expected persistence sync before snapshot consolidation")
	}
}
