package core

import (
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"lumenvec/internal/index"
)

func TestFileVectorStoreCRUD(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vectors")
	store := newFileVectorStore(path)
	t.Cleanup(func() { _ = store.Close() })
	vec := index.Vector{ID: "doc-1", Values: []float64{1, 2, 3}}

	if err := store.UpsertVector(vec); err != nil {
		t.Fatal(err)
	}
	got, err := store.GetVector("doc-1")
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != "doc-1" || len(got.Values) != 3 {
		t.Fatal("expected stored vector")
	}
	list := store.ListVectors()
	if len(list) != 1 || list[0].ID != "doc-1" {
		t.Fatal("expected listed vector")
	}
	if err := store.DeleteVector("doc-1"); err != nil {
		t.Fatal(err)
	}
	if _, err := store.GetVector("doc-1"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatal("expected vector to be deleted")
	}

	reopened := newFileVectorStore(path)
	t.Cleanup(func() { _ = reopened.Close() })
	if _, err := reopened.GetVector("doc-1"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatal("expected deleted vector to stay deleted after reopen")
	}
}

func TestFileVectorStoreRebuildsFromAppendOnlyLog(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vectors")
	store := newFileVectorStore(path)
	t.Cleanup(func() { _ = store.Close() })

	if err := store.UpsertVector(index.Vector{ID: "doc-1", Values: []float64{1, 2, 3}}); err != nil {
		t.Fatal(err)
	}
	if err := store.UpsertVector(index.Vector{ID: "doc-1", Values: []float64{4, 5, 6}}); err != nil {
		t.Fatal(err)
	}
	if err := store.UpsertVector(index.Vector{ID: "doc-2", Values: []float64{7, 8, 9}}); err != nil {
		t.Fatal(err)
	}

	reopened := newFileVectorStore(path)
	t.Cleanup(func() { _ = reopened.Close() })
	got, err := reopened.GetVector("doc-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Values) != 3 || got.Values[0] != 4 {
		t.Fatal("expected latest upsert to survive reopen")
	}

	list := reopened.ListVectors()
	if len(list) != 2 {
		t.Fatal("expected both live vectors after rebuild")
	}
}

func TestFileVectorStoreUpsertVectorsPersistsBatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vectors")
	store := newFileVectorStore(path)
	t.Cleanup(func() { _ = store.Close() })

	if err := store.UpsertVectors([]index.Vector{
		{ID: "doc-1", Values: []float64{1, 2, 3}},
		{ID: "doc-2", Values: []float64{4, 5, 6}},
		{ID: "doc-1", Values: []float64{7, 8, 9}},
	}); err != nil {
		t.Fatal(err)
	}

	got, err := store.GetVector("doc-1")
	if err != nil {
		t.Fatal(err)
	}
	if got.Values[0] != 7 {
		t.Fatalf("doc-1 value = %v, want latest batch value", got.Values)
	}
	page := store.PageVectorIDs("", 10)
	if len(page) != 2 || page[0] != "doc-1" || page[1] != "doc-2" {
		t.Fatalf("PageVectorIDs = %v, want [doc-1 doc-2]", page)
	}

	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	reopened := newFileVectorStore(path)
	t.Cleanup(func() { _ = reopened.Close() })
	got, err = reopened.GetVector("doc-2")
	if err != nil {
		t.Fatal(err)
	}
	if got.Values[0] != 4 {
		t.Fatalf("doc-2 value after reopen = %v", got.Values)
	}
}

func TestFileVectorStoreSyncEveryDefersUntilThresholdAndClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vectors")
	store := newFileVectorStoreWithOptions(path, DefaultStorageSecurityOptions(), 3)

	if err := store.UpsertVectors([]index.Vector{
		{ID: "doc-1", Values: []float64{1}},
		{ID: "doc-2", Values: []float64{2}},
	}); err != nil {
		t.Fatal(err)
	}
	if store.pending != 2 {
		t.Fatalf("pending = %d, want 2", store.pending)
	}
	if err := store.UpsertVector(index.Vector{ID: "doc-3", Values: []float64{3}}); err != nil {
		t.Fatal(err)
	}
	if store.pending != 0 {
		t.Fatalf("pending after threshold = %d, want 0", store.pending)
	}
	if err := store.UpsertVector(index.Vector{ID: "doc-4", Values: []float64{4}}); err != nil {
		t.Fatal(err)
	}
	if store.pending != 1 {
		t.Fatalf("pending before close = %d, want 1", store.pending)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if store.pending != 0 {
		t.Fatalf("pending after close = %d, want 0", store.pending)
	}

	reopened := newFileVectorStore(path)
	t.Cleanup(func() { _ = reopened.Close() })
	got, err := reopened.GetVector("doc-4")
	if err != nil {
		t.Fatal(err)
	}
	if got.Values[0] != 4 {
		t.Fatalf("doc-4 after reopen = %v", got.Values)
	}
}

func TestFileVectorStoreHotCacheLifecycle(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vectors")
	store := newFileVectorStore(path)
	store.hotMaxItems = 2
	t.Cleanup(func() { _ = store.Close() })

	if err := store.UpsertVectors([]index.Vector{
		{ID: "doc-1", Values: []float64{1, 2}},
		{ID: "doc-2", Values: []float64{3, 4}},
	}); err != nil {
		t.Fatal(err)
	}
	if got := len(store.hotCache); got != 2 {
		t.Fatalf("hot cache size = %d, want 2", got)
	}
	values, err := store.GetVectorReadOnly32("doc-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 2 || values[0] != 1 {
		t.Fatalf("GetVectorReadOnly32 values = %v", values)
	}

	if err := store.UpsertVector(index.Vector{ID: "doc-3", Values: []float64{5, 6}}); err != nil {
		t.Fatal(err)
	}
	store.hotMu.Lock()
	_, hasDoc1 := store.hotCache["doc-1"]
	_, hasDoc2 := store.hotCache["doc-2"]
	_, hasDoc3 := store.hotCache["doc-3"]
	cacheSize := len(store.hotCache)
	store.hotMu.Unlock()
	if cacheSize != 2 || !hasDoc1 || hasDoc2 || !hasDoc3 {
		t.Fatalf("unexpected hot cache state size=%d doc1=%t doc2=%t doc3=%t", cacheSize, hasDoc1, hasDoc2, hasDoc3)
	}

	if err := store.DeleteVector("doc-1"); err != nil {
		t.Fatal(err)
	}
	store.hotMu.Lock()
	_, hasDoc1 = store.hotCache["doc-1"]
	store.hotMu.Unlock()
	if hasDoc1 {
		t.Fatal("expected delete to evict vector from hot cache")
	}
	if _, err := store.GetVectorReadOnly32("doc-1"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("expected deleted vector not found, got %v", err)
	}
}

func TestNewDefaultVectorStoreDisk(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vectors")
	store := newDefaultVectorStore("disk", path)
	if closer, ok := store.(interface{ Close() error }); ok {
		t.Cleanup(func() { _ = closer.Close() })
	}
	if err := store.UpsertVector(index.Vector{ID: "doc-1", Values: []float64{1, 2, 3}}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.GetVector("doc-1"); err != nil {
		t.Fatal(err)
	}
}

func TestFileVectorStoreCompactsStaleRecords(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vectors")
	store := newFileVectorStore(path)
	t.Cleanup(func() { _ = store.Close() })

	for i := 0; i < 20; i++ {
		if err := store.UpsertVector(index.Vector{ID: "doc-1", Values: []float64{float64(i), 2, 3}}); err != nil {
			t.Fatal(err)
		}
	}

	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	reopened := newFileVectorStore(path)
	t.Cleanup(func() { _ = reopened.Close() })

	info, err := os.Stat(filepath.Join(path, fileVectorStoreDataFile))
	if err != nil {
		t.Fatal(err)
	}
	if got, wantMax := info.Size(), int64(60); got > wantMax {
		t.Fatalf("expected compacted data file, got size %d > %d", got, wantMax)
	}

	vec, err := reopened.GetVector("doc-1")
	if err != nil {
		t.Fatal(err)
	}
	if vec.Values[0] != 19 {
		t.Fatal("expected latest value after compaction and reopen")
	}
}

func TestFileVectorStoreStatsAndClosePaths(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vectors")
	store := newFileVectorStore(path)
	if err := store.UpsertVector(index.Vector{ID: "doc-1", Values: []float64{1, 2, 3}}); err != nil {
		t.Fatal(err)
	}
	if err := store.UpsertVector(index.Vector{ID: "doc-1", Values: []float64{4, 5, 6}}); err != nil {
		t.Fatal(err)
	}

	stats := store.DiskStats()
	if stats.FileBytes == 0 || stats.Records == 0 {
		t.Fatal("expected disk stats to be populated")
	}
	if !store.IsPersistent() {
		t.Fatal("expected persistent vector store")
	}

	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := store.GetVectorReadOnly("doc-1"); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("expected os.ErrClosed, got %v", err)
	}
	if _, err := store.GetVectorReadOnly32("doc-1"); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("expected os.ErrClosed from GetVectorReadOnly32, got %v", err)
	}
	if got := store.ListVectors(); got != nil {
		t.Fatal("expected nil vectors after close")
	}
}

func TestFileVectorStoreHelpersAndDecodeErrors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vectors")
	store := newFileVectorStore(path)
	t.Cleanup(func() { _ = store.Close() })

	record, err := encodeFileVectorRecord(fileVectorStoreOpPut, "doc-1", []float64{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := decodeFileVectorPayload([]byte{1, 2, 3}); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatal("expected short payload error")
	}
	if _, err := decodeFileVectorPayload(record[4 : len(record)-1]); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatal("expected truncated payload error")
	}

	file, err := os.OpenFile(filepath.Join(path, fileVectorStoreDataFile), os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = file.Close() })
	if _, err := file.Write(record[:len(record)-1]); err != nil {
		t.Fatal(err)
	}
	meta := fileVectorRecordMeta{recordOffset: 0, recordLength: uint32(len(record))}
	if _, err := readFileVectorRecordAt(file, meta); !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
		t.Fatalf("expected eof, got %v", err)
	}
}

func TestFileVectorRecordUsesFloat32AndReadsLegacyFloat64(t *testing.T) {
	record, err := encodeFileVectorRecord(fileVectorStoreOpPut, "doc-1", []float64{1.25, 2.5, 3.75})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(record), 30; got != want {
		t.Fatalf("float32 record length = %d, want %d", got, want)
	}
	decoded, err := decodeFileVectorPayload(record[4:])
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded.values) != 3 || decoded.values[0] != 1.25 || decoded.values[2] != 3.75 {
		t.Fatalf("decoded float32 values = %v", decoded.values)
	}
	decoded32, err := decodeFileVectorPayload32(record[4:])
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded32.values) != 3 || decoded32.values[0] != 1.25 || decoded32.values[2] != 3.75 {
		t.Fatalf("decoded float32 values32 = %v", decoded32.values)
	}

	legacy, err := encodeFileVectorRecord(fileVectorStoreOpPutFloat64, "doc-1", []float64{1.25, 2.5, 3.75})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(legacy), 42; got != want {
		t.Fatalf("legacy float64 record length = %d, want %d", got, want)
	}
	decoded, err = decodeFileVectorPayload(legacy[4:])
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded.values) != 3 || decoded.values[0] != 1.25 || decoded.values[2] != 3.75 {
		t.Fatalf("decoded legacy values = %v", decoded.values)
	}
	decoded32, err = decodeFileVectorPayload32(legacy[4:])
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded32.values) != 3 || decoded32.values[0] != 1.25 || decoded32.values[2] != 3.75 {
		t.Fatalf("decoded legacy values32 = %v", decoded32.values)
	}
}

func TestFileVectorStoreReopenAndRebuildHelpers(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vectors")
	store := newFileVectorStore(path)
	t.Cleanup(func() { _ = store.Close() })

	if err := store.UpsertVector(index.Vector{ID: "doc-1", Values: []float64{1, 2, 3}}); err != nil {
		t.Fatal(err)
	}
	if err := store.UpsertVector(index.Vector{ID: "doc-2", Values: []float64{2, 3, 4}}); err != nil {
		t.Fatal(err)
	}
	if err := store.DeleteVector("doc-1"); err != nil {
		t.Fatal(err)
	}

	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if err := store.ensureOpenLocked(); err != nil {
		t.Fatal(err)
	}

	file, err := os.Open(filepath.Join(path, fileVectorStoreDataFile))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = file.Close() }()

	offsets, stats, err := rebuildFileVectorStoreIndex(file)
	if err != nil {
		t.Fatal(err)
	}
	if len(offsets) != 1 || offsets["doc-2"].recordLength == 0 {
		t.Fatal("expected rebuilt index to keep only live vector")
	}
	if stats.totalRecords == 0 || stats.staleRecords == 0 {
		t.Fatal("expected rebuilt stats to track total and stale records")
	}
}

func TestFileVectorStoreOpenLockedFailure(t *testing.T) {
	base := t.TempDir()
	filePath := filepath.Join(base, "not-a-dir")
	if err := os.WriteFile(filePath, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	store := &fileVectorStore{
		basePath: filePath,
		dataPath: filepath.Join(filePath, fileVectorStoreDataFile),
		offsets:  make(map[string]fileVectorRecordMeta),
	}
	if err := store.open(); err == nil {
		t.Fatal("expected open failure when base path is a file")
	}
}

func TestFileVectorStoreUsesConfiguredPermissions(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("permission bits are not reliable on Windows")
	}
	path := filepath.Join(t.TempDir(), "vectors")
	store := newFileVectorStoreWithSecurity(path, StrictStorageSecurityOptions())
	t.Cleanup(func() { _ = store.Close() })

	if err := store.UpsertVector(index.Vector{ID: "doc-1", Values: []float64{1, 2, 3}}); err != nil {
		t.Fatal(err)
	}

	dirMode, err := storagePathMode(path)
	if err != nil {
		t.Fatal(err)
	}
	if dirMode != 0o700 {
		t.Fatalf("expected dir mode 0700, got %o", dirMode)
	}

	fileMode, err := storagePathMode(filepath.Join(path, fileVectorStoreDataFile))
	if err != nil {
		t.Fatal(err)
	}
	if fileMode != 0o600 {
		t.Fatalf("expected file mode 0600, got %o", fileMode)
	}
}

func TestNewFileVectorStoreWithSecurityPanicsOnOpenFailure(t *testing.T) {
	base := t.TempDir()
	filePath := filepath.Join(base, "not-a-dir")
	if err := os.WriteFile(filePath, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	defer func() {
		if recover() == nil {
			t.Fatal("expected constructor panic when base path is a file")
		}
	}()
	_ = newFileVectorStoreWithSecurity(filePath, DefaultStorageSecurityOptions())
}

func TestFileVectorStoreAdditionalErrorBranches(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vectors")
	store := newFileVectorStore(path)
	t.Cleanup(func() { _ = store.Close() })

	if err := store.DeleteVector("missing"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("expected missing delete to return not found, got %v", err)
	}
	store.offsets["deleted"] = fileVectorRecordMeta{deleted: true}
	if err := store.DeleteVector("deleted"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("expected deleted metadata to return not found, got %v", err)
	}

	store.offsets["bad"] = fileVectorRecordMeta{recordOffset: 9999, recordLength: 8}
	if _, err := store.GetVectorReadOnly("bad"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("expected bad metadata to read as not found, got %v", err)
	}
	if _, err := store.GetVectorReadOnly32("bad"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("expected bad metadata to read as not found in float32 path, got %v", err)
	}
	if got := store.ListVectors(); len(got) != 0 {
		t.Fatalf("expected invalid/deleted metadata to be skipped, got %+v", got)
	}

	deleteRecord, err := encodeFileVectorRecord(fileVectorStoreOpDelete, "gone", nil)
	if err != nil {
		t.Fatal(err)
	}
	offset, err := store.file.Seek(0, io.SeekEnd)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.file.Write(deleteRecord); err != nil {
		t.Fatal(err)
	}
	store.offsets["gone"] = fileVectorRecordMeta{recordOffset: offset, recordLength: uint32(len(deleteRecord))}
	if _, err := store.GetVectorReadOnly("gone"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("expected delete record to read as not found, got %v", err)
	}
	if _, err := store.GetVectorReadOnly32("gone"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("expected delete record to read as not found in float32 path, got %v", err)
	}

	if err := (&fileVectorStore{}).compactLocked(); err != nil {
		t.Fatalf("compactLocked nil file error = %v", err)
	}

	tempFile, err := os.CreateTemp(t.TempDir(), "open-file")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = tempFile.Close() })
	blocker := filepath.Join(t.TempDir(), "not-a-dir")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	badCompact := &fileVectorStore{
		dataPath: filepath.Join(blocker, fileVectorStoreDataFile),
		file:     tempFile,
		offsets:  map[string]fileVectorRecordMeta{},
		security: DefaultStorageSecurityOptions(),
	}
	if err := badCompact.compactLocked(); err == nil {
		t.Fatal("expected compactLocked to fail when compact temp path parent is a file")
	}
}

func TestFileVectorStoreDecodeAndRebuildErrorBranches(t *testing.T) {
	base := t.TempDir()

	partialLength := filepath.Join(base, "partial-length.dat")
	if err := os.WriteFile(partialLength, []byte{1, 2}, 0o644); err != nil {
		t.Fatal(err)
	}
	withOpenFile(t, partialLength, func(f *os.File) {
		if _, _, err := rebuildFileVectorStoreIndex(f); err == nil {
			t.Fatal("expected partial length read to fail")
		}
	})

	partialPayload := filepath.Join(base, "partial-payload.dat")
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, 10)
	if err := os.WriteFile(partialPayload, append(buf, []byte{1, 2}...), 0o644); err != nil {
		t.Fatal(err)
	}
	withOpenFile(t, partialPayload, func(f *os.File) {
		if _, _, err := rebuildFileVectorStoreIndex(f); err == nil {
			t.Fatal("expected partial payload read to fail")
		}
	})

	invalidPayload := filepath.Join(base, "invalid-payload.dat")
	buf = make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, 1)
	if err := os.WriteFile(invalidPayload, append(buf, byte(1)), 0o644); err != nil {
		t.Fatal(err)
	}
	withOpenFile(t, invalidPayload, func(f *os.File) {
		if _, _, err := rebuildFileVectorStoreIndex(f); !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatalf("expected decode failure, got %v", err)
		}
	})

	shortRecord := filepath.Join(base, "short-record.dat")
	buf = make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, 100)
	if err := os.WriteFile(shortRecord, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	withOpenFile(t, shortRecord, func(f *os.File) {
		meta := fileVectorRecordMeta{recordOffset: 0, recordLength: 4}
		if _, err := readFileVectorRecordAt(f, meta); !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatalf("expected short record failure, got %v", err)
		}
	})

	payload := make([]byte, 9)
	payload[0] = fileVectorStoreOpPut
	binary.LittleEndian.PutUint32(payload[1:5], 10)
	if _, err := decodeFileVectorPayload(payload); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("expected id length decode failure, got %v", err)
	}
}

func TestFileVectorStoreOpenAndCompactionBranches(t *testing.T) {
	base := t.TempDir()
	blocker := filepath.Join(base, "not-a-dir")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	closed := &fileVectorStore{
		basePath: blocker,
		dataPath: filepath.Join(blocker, fileVectorStoreDataFile),
		offsets:  make(map[string]fileVectorRecordMeta),
		security: DefaultStorageSecurityOptions(),
	}
	if err := closed.UpsertVector(index.Vector{ID: "a", Values: []float64{1}}); err == nil {
		t.Fatal("expected upsert to fail when reopening under file path")
	}
	closed.offsets["a"] = fileVectorRecordMeta{recordOffset: 0, recordLength: 1}
	if err := closed.DeleteVector("a"); err == nil {
		t.Fatal("expected delete to fail when reopening under file path")
	}

	dataDir := filepath.Join(base, "data-dir")
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		t.Fatal(err)
	}
	openFileFailure := &fileVectorStore{
		basePath: base,
		dataPath: dataDir,
		offsets:  make(map[string]fileVectorRecordMeta),
		security: DefaultStorageSecurityOptions(),
	}
	if err := openFileFailure.openLocked(); err == nil {
		t.Fatal("expected openLocked to fail when data path is a directory")
	}

	invalidDataPath := filepath.Join(base, "invalid-records.dat")
	if err := os.WriteFile(invalidDataPath, []byte{1, 2}, 0o644); err != nil {
		t.Fatal(err)
	}
	rebuildFailure := &fileVectorStore{
		basePath: base,
		dataPath: invalidDataPath,
		offsets:  make(map[string]fileVectorRecordMeta),
		security: DefaultStorageSecurityOptions(),
	}
	if err := rebuildFailure.openLocked(); err == nil {
		t.Fatal("expected openLocked to fail on invalid existing data")
	}

	compactPath := filepath.Join(base, "compact.dat")
	compactFile, err := os.OpenFile(compactPath, os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	compactStore := &fileVectorStore{
		dataPath: compactPath,
		file:     compactFile,
		offsets: map[string]fileVectorRecordMeta{
			"bad": {recordOffset: 9999, recordLength: 8},
		},
		security: DefaultStorageSecurityOptions(),
	}
	if err := compactStore.compactLocked(); err != nil {
		t.Fatalf("expected compaction with bad metadata to skip record, got %v", err)
	}
	t.Cleanup(func() { _ = compactStore.Close() })

	dupFilePath := filepath.Join(base, "duplicates.dat")
	first, err := encodeFileVectorRecord(fileVectorStoreOpPut, "dup", []float64{1})
	if err != nil {
		t.Fatal(err)
	}
	second, err := encodeFileVectorRecord(fileVectorStoreOpPut, "dup", []float64{2})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dupFilePath, append(first, second...), 0o644); err != nil {
		t.Fatal(err)
	}
	withOpenFile(t, dupFilePath, func(f *os.File) {
		offsets, stats, err := rebuildFileVectorStoreIndex(f)
		if err != nil {
			t.Fatal(err)
		}
		if len(offsets) != 1 || stats.staleRecords != 1 {
			t.Fatalf("expected duplicate rebuild to mark stale record, offsets=%+v stats=%+v", offsets, stats)
		}
	})

	closedSourcePath := filepath.Join(base, "closed-source.dat")
	closedSource, err := os.OpenFile(closedSourcePath, os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	if err := closedSource.Close(); err != nil {
		t.Fatal(err)
	}
	closedSourceStore := &fileVectorStore{
		dataPath: closedSourcePath,
		file:     closedSource,
		offsets:  map[string]fileVectorRecordMeta{},
		security: DefaultStorageSecurityOptions(),
	}
	if err := closedSourceStore.compactLocked(); err == nil {
		t.Fatal("expected compactLocked to fail when source file is already closed")
	}

	renameTargetDir := filepath.Join(base, "rename-target-dir")
	if err := os.MkdirAll(renameTargetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	renameSource, err := os.CreateTemp(base, "rename-source")
	if err != nil {
		t.Fatal(err)
	}
	renameFailureStore := &fileVectorStore{
		dataPath: renameTargetDir,
		file:     renameSource,
		offsets:  map[string]fileVectorRecordMeta{},
		security: DefaultStorageSecurityOptions(),
	}
	if err := renameFailureStore.compactLocked(); err == nil {
		t.Fatal("expected compactLocked to fail when rename target is a directory")
	}

	closedForRebuild, err := os.Open(dupFilePath)
	if err != nil {
		t.Fatal(err)
	}
	if err := closedForRebuild.Close(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := rebuildFileVectorStoreIndex(closedForRebuild); err == nil {
		t.Fatal("expected rebuild to fail on closed file")
	}
}

func withOpenFile(t *testing.T, path string, fn func(*os.File)) {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = file.Close() }()
	fn(file)
}
