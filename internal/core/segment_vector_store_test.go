package core

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"
)

func TestSegmentVectorStoreCommitRecoverAndDelete(t *testing.T) {
	path := filepath.Join(t.TempDir(), "segments")
	store := newSegmentVectorStore(path, DefaultStorageSecurityOptions())
	if err := store.UpsertVectors([]index.Vector{
		{ID: "a", Values: []float64{1, 2}},
		{ID: "b", Values: []float64{3, 4}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.UpsertVector(index.Vector{ID: "a", Values: []float64{5, 6}}); err != nil {
		t.Fatal(err)
	}
	if err := store.DeleteVector("b"); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	reopened := newSegmentVectorStore(path, DefaultStorageSecurityOptions())
	t.Cleanup(func() { _ = reopened.Close() })
	got, err := reopened.GetVector("a")
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Values) != 2 || got.Values[0] != 5 {
		t.Fatalf("recovered a = %+v", got)
	}
	if _, err := reopened.GetVector("b"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("deleted b error = %v", err)
	}
	if page := reopened.PageVectorIDs("", 10); len(page) != 1 || page[0] != "a" {
		t.Fatalf("page = %v", page)
	}
	if stats := reopened.DiskStats(); stats.BytesWritten == 0 || stats.WriteAmplification <= 0 {
		t.Fatalf("invalid write amplification = %+v", stats)
	}
	if stats := reopened.DiskStats(); stats.Records != 1 || stats.FileBytes == 0 {
		t.Fatalf("stats = %+v", stats)
	}
}

func TestSegmentVectorStorePersistentLocationIndexReopensWithoutRebuild(t *testing.T) {
	path := filepath.Join(t.TempDir(), "segments")
	store := newSegmentVectorStoreWithLocationCapacity(path, DefaultStorageSecurityOptions(), 128)
	vectors := []index.Vector{
		{ID: "alpha", Values: []float64{1, 2}},
		{ID: "beta", Values: []float64{3, 4}},
		{ID: "gamma", Values: []float64{5, 6}},
	}
	if err := store.UpsertVectors(vectors); err != nil {
		t.Fatal(err)
	}
	if err := store.DeleteVector("beta"); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	reopened := newSegmentVectorStoreWithLocationCapacity(path, DefaultStorageSecurityOptions(), 128)
	defer reopened.Close()
	if !reopened.RecoveryIndexLoaded() {
		t.Fatal("persistent location index was not reused")
	}
	if reopened.DiskStats().Records != 2 {
		t.Fatalf("records = %d", reopened.DiskStats().Records)
	}
	if _, err := reopened.GetVector("beta"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("deleted vector error = %v", err)
	}
	for _, id := range []string{"alpha", "gamma"} {
		if vector, err := reopened.GetVector(id); err != nil || vector.ID != id {
			t.Fatalf("GetVector(%s) = %+v, %v", id, vector, err)
		}
	}
}

func TestSegmentVectorStorePersistentLocationIndexRebuildsOnStaleMarker(t *testing.T) {
	path := filepath.Join(t.TempDir(), "segments")
	store := newSegmentVectorStoreWithLocationCapacity(path, DefaultStorageSecurityOptions(), 64)
	if err := store.UpsertVector(index.Vector{ID: "durable", Values: []float64{7, 8}}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(path, "locations.hash.manifest"), []byte("stale"), 0o600); err != nil {
		t.Fatal(err)
	}

	reopened := newSegmentVectorStoreWithLocationCapacity(path, DefaultStorageSecurityOptions(), 64)
	defer reopened.Close()
	if reopened.RecoveryIndexLoaded() {
		t.Fatal("stale persistent index was incorrectly marked as reused")
	}
	vector, err := reopened.GetVector("durable")
	if err != nil || vector.ID != "durable" {
		t.Fatalf("rebuilt GetVector = %+v, %v", vector, err)
	}
	marker, err := os.ReadFile(filepath.Join(path, "locations.hash.manifest"))
	if err != nil || len(marker) != sha256.Size {
		t.Fatalf("rebuilt marker length=%d err=%v", len(marker), err)
	}
}

func TestSegmentVectorStorePersistentLocationCapacityFailsBeforeCommit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "segments")
	store := newSegmentVectorStoreWithLocationCapacity(path, DefaultStorageSecurityOptions(), 8)
	vectors := make([]index.Vector, 8)
	for n := range vectors {
		vectors[n] = index.Vector{ID: fmt.Sprintf("v-%d", n), Values: []float64{float64(n)}}
	}
	if err := store.UpsertVectors(vectors); err != nil {
		t.Fatal(err)
	}
	beforeSegments := len(store.manifest.Segments)
	if err := store.UpsertVector(index.Vector{ID: "overflow", Values: []float64{9}}); !errors.Is(err, ErrLocationIndexFull) {
		t.Fatalf("overflow error = %v", err)
	}
	if len(store.manifest.Segments) != beforeSegments {
		t.Fatal("capacity failure committed a segment")
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	reopened := newSegmentVectorStoreWithLocationCapacity(path, DefaultStorageSecurityOptions(), 8)
	defer reopened.Close()
	if reopened.DiskStats().Records != 8 {
		t.Fatalf("records after capacity failure = %d", reopened.DiskStats().Records)
	}
}

func TestSegmentVectorStoreRangeVectors32StreamsPersistedValues(t *testing.T) {
	store := newSegmentVectorStore(filepath.Join(t.TempDir(), "segments"), DefaultStorageSecurityOptions())
	defer store.Close()
	vectors := []index.Vector{
		{ID: "b", Values: []float64{2, 3}},
		{ID: "a", Values: []float64{1, 4}},
	}
	if err := store.UpsertVectors(vectors); err != nil {
		t.Fatal(err)
	}
	var ids []string
	var first []float32
	store.RangeVectors32(func(id string, values []float32) bool {
		ids = append(ids, id)
		if len(first) == 0 {
			first = append([]float32(nil), values...)
		}
		return true
	})
	if !reflect.DeepEqual(ids, []string{"a", "b"}) {
		t.Fatalf("ids=%v", ids)
	}
	if !reflect.DeepEqual(first, []float32{1, 4}) {
		t.Fatalf("first=%v", first)
	}
}

func TestSegmentVectorStoreRangeVectorsByID32IsCanonicalAcrossSegments(t *testing.T) {
	store := newSegmentVectorStore(filepath.Join(t.TempDir(), "segments"), DefaultStorageSecurityOptions())
	defer store.Close()
	for _, vector := range []index.Vector{
		{ID: "z", Values: []float64{3, 0}},
		{ID: "a", Values: []float64{1, 0}},
		{ID: "m", Values: []float64{2, 0}},
	} {
		if err := store.UpsertVector(vector); err != nil {
			t.Fatal(err)
		}
	}
	var ids []string
	var firstValues []float32
	store.RangeVectorsByID32(func(id string, values []float32) bool {
		ids = append(ids, id)
		firstValues = append(firstValues, values[0])
		return true
	})
	if !reflect.DeepEqual(ids, []string{"a", "m", "z"}) {
		t.Fatalf("ids=%v", ids)
	}
	if !reflect.DeepEqual(firstValues, []float32{1, 2, 3}) {
		t.Fatalf("values=%v", firstValues)
	}
}

func TestSegmentVectorStoreUpsertVectors32RoundTrip(t *testing.T) {
	store := newSegmentVectorStore(t.TempDir(), StorageSecurityOptions{})
	defer store.Close()
	if err := store.UpsertVectors32([]vectorBatch32{{ID: "native", Values: []float32{1.25, -2.5}}}); err != nil {
		t.Fatal(err)
	}
	values, err := store.GetVectorReadOnly32("native")
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 2 || values[0] != 1.25 || values[1] != -2.5 {
		t.Fatalf("unexpected native payload: %v", values)
	}
}

func TestSegmentVectorStoreWritesDeltaWAL(t *testing.T) {
	path := filepath.Join(t.TempDir(), "segments")
	store := newSegmentVectorStore(path, StorageSecurityOptions{})
	if err := store.UpsertVectors32([]vectorBatch32{{ID: "a", Values: []float32{1, 2}}}); err != nil {
		t.Fatal(err)
	}
	if err := store.DeleteVector("a"); err != nil {
		t.Fatal(err)
	}
	var records []DeltaRecord
	if err := store.deltaWAL.ReplaySince(0, func(record DeltaRecord) error { records = append(records, record); return nil }); err != nil {
		t.Fatal(err)
	}
	if len(records) != 2 || records[0].VectorID != "a" || !records[1].Deleted {
		t.Fatalf("delta WAL = %+v", records)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestSegmentVectorStoreWriteAmplificationGate(t *testing.T) {
	store := newSegmentVectorStore(filepath.Join(t.TempDir(), "segments"), DefaultStorageSecurityOptions())
	defer store.Close()
	store.compactThreshold = 1000 // invoke compaction explicitly below
	for i := 0; i < 64; i++ {
		if err := store.UpsertVector(index.Vector{ID: fmt.Sprintf("v-%d", i), Values: []float64{float64(i), 1}}); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.compactOnce(); err != nil {
		t.Fatal(err)
	}
	stats := store.DiskStats()
	if stats.WriteAmplification <= 0 || stats.WriteAmplification > 8 {
		t.Fatalf("write amplification outside gate: %+v", stats)
	}
}

func TestSegmentVectorStoreIgnoresOrphanAndRejectsCorruptManifest(t *testing.T) {
	path := filepath.Join(t.TempDir(), "segments")
	store := newSegmentVectorStore(path, DefaultStorageSecurityOptions())
	if err := store.UpsertVector(index.Vector{ID: "committed", Values: []float64{1}}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	orphan, err := encodeFileVectorRecord(fileVectorStoreOpPut, "orphan", []float64{9})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(path, "segment-99999999999999999999.dat"), orphan, 0o644); err != nil {
		t.Fatal(err)
	}
	reopened := newSegmentVectorStore(path, DefaultStorageSecurityOptions())
	if _, err := reopened.GetVector("orphan"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("orphan error = %v", err)
	}
	_ = reopened.Close()

	if err := os.WriteFile(filepath.Join(path, "manifest.json"), []byte("{"), 0o644); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if recover() == nil {
			t.Fatal("expected corrupt manifest panic")
		}
	}()
	_ = newSegmentVectorStore(path, DefaultStorageSecurityOptions())
}

func TestDefaultSegmentVectorStore(t *testing.T) {
	path := filepath.Join(t.TempDir(), "segments")
	store := newDefaultVectorStore("segment", path)
	closer := store.(interface{ Close() error })
	t.Cleanup(func() { _ = closer.Close() })
	if err := store.UpsertVector(index.Vector{ID: "a", Values: []float64{1}}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.GetVector("a"); err != nil {
		t.Fatal(err)
	}
}

func TestSegmentVectorStoreCompactsInBackground(t *testing.T) {
	path := filepath.Join(t.TempDir(), "segments")
	store := newSegmentVectorStore(path, DefaultStorageSecurityOptions())
	store.compactThreshold = 3
	t.Cleanup(func() { _ = store.Close() })
	for i := 0; i < 3; i++ {
		if err := store.UpsertVector(index.Vector{ID: "a", Values: []float64{float64(i)}}); err != nil {
			t.Fatal(err)
		}
	}
	deadline := time.Now().Add(3 * time.Second)
	for {
		store.mu.RLock()
		segments := len(store.manifest.Segments)
		store.mu.RUnlock()
		if segments == 1 && store.compacts.Load() == 1 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("compaction did not finish: segments=%d compactions=%d", segments, store.compacts.Load())
		}
		time.Sleep(10 * time.Millisecond)
	}
	got, err := store.GetVector("a")
	if err != nil {
		t.Fatal(err)
	}
	if got.Values[0] != 2 {
		t.Fatalf("value after compaction = %v", got.Values)
	}
}

func TestSegmentVectorStoreCompactionBudgetDefersRewrite(t *testing.T) {
	store := newSegmentVectorStore(filepath.Join(t.TempDir(), "segments"), DefaultStorageSecurityOptions())
	defer store.Close()
	store.compactThreshold = 1
	if err := store.UpsertVector(index.Vector{ID: "budget", Values: []float64{1, 2}}); err != nil {
		t.Fatal(err)
	}
	before := store.compacts.Load()
	store.compactBudgetBytes = 1
	if err := store.compactOnce(); err != nil {
		t.Fatal(err)
	}
	if store.compacts.Load() != before {
		t.Fatalf("budget exceeded but compaction ran")
	}
	if stats := store.DiskStats(); stats.CompactionBudgetBytes != 1 {
		t.Fatalf("stats=%+v", stats)
	}
}

func TestSegmentStoreUsesSegmentedANNDelta(t *testing.T) {
	base := t.TempDir()
	svc := NewService(ServiceOptions{
		MaxVectorDim:       8,
		MaxK:               10,
		SearchMode:         "ann",
		VectorStore:        "segment",
		VectorPath:         filepath.Join(base, "vectors"),
		ANNSegmentMaxNodes: 2,
		ANNOptions:         ann.Options{M: 8, EfConstruction: 32, EfSearch: 32},
	})
	t.Cleanup(func() { _ = svc.Close() })
	for i := 0; i < 5; i++ {
		if err := svc.AddVector(string(rune('a'+i)), []float64{float64(i), 0}); err != nil {
			t.Fatal(err)
		}
	}
	segmented, ok := svc.annIndex.(*ann.SegmentedIndex)
	if !ok {
		t.Fatalf("ANN index type = %T", svc.annIndex)
	}
	if got := segmented.SegmentCount(); got != 3 {
		t.Fatalf("ANN segments = %d, want 3", got)
	}
	results, err := svc.Search([]float64{4.1, 0}, 1)
	if err != nil || len(results) != 1 || results[0].ID != "e" {
		t.Fatalf("results=%+v err=%v", results, err)
	}
}

func TestSegmentVectorStoreInterfaceAndClosedPaths(t *testing.T) {
	path := filepath.Join(t.TempDir(), "segments")
	store := newSegmentVectorStore(path, DefaultStorageSecurityOptions())
	if err := store.UpsertVectors(nil); err != nil {
		t.Fatal(err)
	}
	if err := store.DeleteVector("missing"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("missing delete error = %v", err)
	}
	if err := store.UpsertVectors([]index.Vector{
		{ID: "b", Values: []float64{2}},
		{ID: "a", Values: []float64{1}},
	}); err != nil {
		t.Fatal(err)
	}
	vec, err := store.GetVectorReadOnly("a")
	if err != nil || vec.Values[0] != 1 {
		t.Fatalf("readonly vec=%+v err=%v", vec, err)
	}
	list := store.ListVectors()
	if len(list) != 2 || list[0].ID != "a" || list[1].ID != "b" {
		t.Fatalf("list = %+v", list)
	}
	var ids []string
	store.RangeVectorIDs(func(id string) bool {
		ids = append(ids, id)
		return true
	})
	if len(ids) != 2 || ids[0] != "a" || ids[1] != "b" {
		t.Fatalf("range ids = %v", ids)
	}
	if !store.IsPersistent() {
		t.Fatal("expected persistent store")
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if err := store.UpsertVector(index.Vector{ID: "c", Values: []float64{3}}); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("closed upsert error = %v", err)
	}
	if _, err := store.GetVectorReadOnly32("a"); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("closed read error = %v", err)
	}
}

func TestSegmentVectorStoreRejectsUnsupportedAndMissingManifestSegments(t *testing.T) {
	for _, tc := range []struct {
		name     string
		manifest segmentManifest
	}{
		{name: "unsupported", manifest: segmentManifest{Version: 99, NextGeneration: 1}},
		{name: "missing-segment", manifest: segmentManifest{Version: 1, NextGeneration: 2, Segments: []string{"missing.dat"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := t.TempDir()
			data, err := json.Marshal(tc.manifest)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(path, "manifest.json"), data, 0o644); err != nil {
				t.Fatal(err)
			}
			defer func() {
				if recover() == nil {
					t.Fatal("expected constructor panic")
				}
			}()
			_ = newSegmentVectorStore(path, DefaultStorageSecurityOptions())
		})
	}
}

func TestSegmentVectorStoreCompactionSourceFailureAndManifestHelpers(t *testing.T) {
	path := filepath.Join(t.TempDir(), "segments")
	store := newSegmentVectorStore(path, DefaultStorageSecurityOptions())
	store.compactThreshold = 100
	t.Cleanup(func() { _ = store.Close() })
	if err := store.UpsertVector(index.Vector{ID: "a", Values: []float64{1}}); err != nil {
		t.Fatal(err)
	}
	store.mu.RLock()
	location := store.locations["a"]
	store.mu.RUnlock()
	if err := os.Remove(filepath.Join(path, location.segment)); err != nil {
		t.Fatal(err)
	}
	if _, err := store.readLocation32(location); err == nil {
		t.Fatal("expected missing segment read failure")
	}
	store.compactThreshold = 1
	if err := store.compactOnce(); err == nil {
		t.Fatal("expected compaction source failure")
	}

	base := segmentManifest{Version: 1, NextGeneration: 2, Segments: []string{"a"}}
	if !sameSegmentManifest(base, cloneSegmentManifest(base)) {
		t.Fatal("expected equal manifests")
	}
	for _, changed := range []segmentManifest{
		{Version: 2, NextGeneration: 2, Segments: []string{"a"}},
		{Version: 1, NextGeneration: 3, Segments: []string{"a"}},
		{Version: 1, NextGeneration: 2},
		{Version: 1, NextGeneration: 2, Segments: []string{"b"}},
	} {
		if sameSegmentManifest(base, changed) {
			t.Fatalf("expected manifests to differ: %+v", changed)
		}
	}
}

func TestSegmentVectorStoreNormalizesZeroGeneration(t *testing.T) {
	path := t.TempDir()
	data, err := json.Marshal(segmentManifest{Version: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(path, "manifest.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	store := newSegmentVectorStore(path, DefaultStorageSecurityOptions())
	t.Cleanup(func() { _ = store.Close() })
	if store.manifest.NextGeneration != 1 {
		t.Fatalf("next generation = %d", store.manifest.NextGeneration)
	}
}

func TestSegmentVectorStoreRejectsTruncatedSegments(t *testing.T) {
	for _, tc := range []struct {
		name string
		data []byte
	}{
		{name: "header", data: []byte{1, 2}},
		{name: "payload", data: func() []byte {
			buf := make([]byte, 5)
			binary.LittleEndian.PutUint32(buf[:4], 100)
			return buf
		}()},
		{name: "decode", data: func() []byte {
			buf := make([]byte, 7)
			binary.LittleEndian.PutUint32(buf[:4], 3)
			return buf
		}()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := t.TempDir()
			if err := os.WriteFile(filepath.Join(path, "broken.dat"), tc.data, 0o644); err != nil {
				t.Fatal(err)
			}
			manifestData, err := json.Marshal(segmentManifest{Version: 1, NextGeneration: 2, Segments: []string{"broken.dat"}})
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(path, "manifest.json"), manifestData, 0o644); err != nil {
				t.Fatal(err)
			}
			defer func() {
				if recover() == nil {
					t.Fatal("expected truncated segment panic")
				}
			}()
			_ = newSegmentVectorStore(path, DefaultStorageSecurityOptions())
		})
	}
}

func TestSegmentVectorStoreRecoveryIndexFastPathAndFallback(t *testing.T) {
	t.Run("loads-current-index", func(t *testing.T) {
		path := t.TempDir()
		store := newSegmentVectorStore(path, DefaultStorageSecurityOptions())
		store.compactThreshold = 100
		if err := store.UpsertVectors([]index.Vector{
			{ID: "a", Values: []float64{1, 2}},
			{ID: "b", Values: []float64{3, 4}},
		}); err != nil {
			t.Fatal(err)
		}
		if err := store.SaveRecoveryIndex(); err != nil {
			t.Fatal(err)
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}

		restored := newSegmentVectorStore(path, DefaultStorageSecurityOptions())
		defer restored.Close()
		if !restored.recoveryLoaded {
			t.Fatal("expected recovery index fast path")
		}
		if values, err := restored.GetVectorReadOnly32("b"); err != nil || len(values) != 2 || values[1] != 4 {
			t.Fatalf("values=%v err=%v", values, err)
		}
	})

	t.Run("stale-manifest-falls-back", func(t *testing.T) {
		path := t.TempDir()
		store := newSegmentVectorStore(path, DefaultStorageSecurityOptions())
		store.compactThreshold = 100
		if err := store.UpsertVector(index.Vector{ID: "a", Values: []float64{1}}); err != nil {
			t.Fatal(err)
		}
		if err := store.SaveRecoveryIndex(); err != nil {
			t.Fatal(err)
		}
		if err := store.UpsertVector(index.Vector{ID: "b", Values: []float64{2}}); err != nil {
			t.Fatal(err)
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}

		restored := newSegmentVectorStore(path, DefaultStorageSecurityOptions())
		defer restored.Close()
		if restored.recoveryLoaded {
			t.Fatal("stale recovery index must not load")
		}
		if _, err := restored.GetVectorReadOnly32("b"); err != nil {
			t.Fatalf("fallback scan lost latest vector: %v", err)
		}
	})

	t.Run("corrupt-index-falls-back", func(t *testing.T) {
		path := t.TempDir()
		store := newSegmentVectorStore(path, DefaultStorageSecurityOptions())
		store.compactThreshold = 100
		if err := store.UpsertVector(index.Vector{ID: "a", Values: []float64{1}}); err != nil {
			t.Fatal(err)
		}
		if err := store.SaveRecoveryIndex(); err != nil {
			t.Fatal(err)
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
		recoveryPath := filepath.Join(path, "locations.idx")
		data, err := os.ReadFile(recoveryPath)
		if err != nil {
			t.Fatal(err)
		}
		data[len(data)-1] ^= 0xff
		if err := os.WriteFile(recoveryPath, data, 0o644); err != nil {
			t.Fatal(err)
		}

		restored := newSegmentVectorStore(path, DefaultStorageSecurityOptions())
		defer restored.Close()
		if restored.recoveryLoaded {
			t.Fatal("corrupt recovery index must not load")
		}
		if _, err := restored.GetVectorReadOnly32("a"); err != nil {
			t.Fatalf("fallback scan lost vector: %v", err)
		}
	})
}

func TestSegmentVectorStoreRangesSelectedVectors(t *testing.T) {
	store := newSegmentVectorStore(t.TempDir(), DefaultStorageSecurityOptions())
	store.compactThreshold = 100
	defer store.Close()
	for i := 0; i < 6; i++ {
		if err := store.UpsertVector(index.Vector{ID: fmt.Sprintf("v-%d", i), Values: []float64{float64(i), 1}}); err != nil {
			t.Fatal(err)
		}
	}
	candidates := map[string]struct{}{"v-1": {}, "v-4": {}, "missing": {}}
	got := make(map[string]float32)
	store.RangeSelectedVectors32(candidates, func(id string, values []float32) bool {
		got[id] = values[0]
		return true
	})
	if len(got) != 2 || got["v-1"] != 1 || got["v-4"] != 4 {
		t.Fatalf("selected vectors=%v", got)
	}
}

func TestSegmentVectorStoreRangesSelectedVectorIDs(t *testing.T) {
	store := newSegmentVectorStore(t.TempDir(), DefaultStorageSecurityOptions())
	store.compactThreshold = 100
	defer store.Close()
	for i := 0; i < 6; i++ {
		if err := store.UpsertVector(index.Vector{ID: fmt.Sprintf("v-%d", i), Values: []float64{float64(i), 1}}); err != nil {
			t.Fatal(err)
		}
	}
	got := make(map[string]float32)
	store.RangeSelectedVectorIDs32([]string{"v-4", "missing", "v-1"}, func(id string, values []float32) bool {
		got[id] = values[0]
		return true
	})
	if len(got) != 2 || got["v-1"] != 1 || got["v-4"] != 4 {
		t.Fatalf("selected vectors=%v", got)
	}
}

func TestSegmentVectorStoreOpenFailsWhenBaseIsFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if recover() == nil {
			t.Fatal("expected base path failure")
		}
	}()
	_ = newSegmentVectorStore(path, DefaultStorageSecurityOptions())
}
