package core

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"
)

func checkpointServiceOptions(base string) ServiceOptions {
	return ServiceOptions{
		MaxVectorDim:       8,
		MaxK:               10,
		SearchMode:         "ann",
		VectorStore:        "segment",
		VectorPath:         filepath.Join(base, "vectors"),
		ANNSegmentMaxNodes: 2,
		ANNOptions:         ann.Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 7},
		StorageSecurity:    DefaultStorageSecurityOptions(),
	}
}

func TestANNCheckpointRestoresConsolidatedGeneration(t *testing.T) {
	base := t.TempDir()
	opts := checkpointServiceOptions(base)
	opts.ANNSegmentMaxNodes = 128
	svc := NewService(opts)
	for batch := 0; batch < 8; batch++ {
		vectors := make([]index.Vector, 0, 4)
		for item := 0; item < 4; item++ {
			id := batch*4 + item
			vectors = append(vectors, index.Vector{ID: fmt.Sprintf("v-%d", id), Values: []float64{float64(id), 0}})
		}
		if err := svc.AddVectors(vectors); err != nil {
			t.Fatal(err)
		}
	}
	segmented := svc.currentANNIndex().(*ann.SegmentedIndex)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := segmented.WaitForMaintenance(ctx); err != nil {
		t.Fatal(err)
	}
	if segments := segmented.SegmentCount(); segments != 1 {
		t.Fatalf("segments before checkpoint = %d", segments)
	}
	if err := svc.Close(); err != nil {
		t.Fatal(err)
	}

	restored := NewService(opts)
	t.Cleanup(func() { _ = restored.Close() })
	if !restored.Stats().ANNCheckpointLoaded || restored.Stats().ANNSegments != 1 {
		t.Fatalf("restored stats = %+v", restored.Stats())
	}
}

func TestANNCheckpointRoundTripAndStableIDMapping(t *testing.T) {
	base := t.TempDir()
	opts := checkpointServiceOptions(base)
	svc := NewService(opts)
	for _, vec := range []index.Vector{
		{ID: "z", Values: []float64{9, 0}},
		{ID: "a", Values: []float64{1, 0}},
		{ID: "m", Values: []float64{5, 0}},
	} {
		if err := svc.AddVector(vec.ID, vec.Values); err != nil {
			t.Fatal(err)
		}
	}
	beforeIDs := svc.idResolver.(*memoryIDResolver).Snapshot()
	if err := svc.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(opts.VectorPath, "ann-checkpoint.bin")); err != nil {
		t.Fatal(err)
	}

	restored := NewService(opts)
	t.Cleanup(func() { _ = restored.Close() })
	if !restored.Stats().ANNCheckpointLoaded {
		t.Fatal("expected ANN checkpoint to load")
	}
	afterIDs := restored.idResolver.(*memoryIDResolver).Snapshot()
	for id, before := range beforeIDs {
		if afterIDs[id] != before {
			t.Fatalf("internal ID for %s changed: before=%d after=%d", id, before, afterIDs[id])
		}
	}
	results, err := restored.Search([]float64{9.1, 0}, 1)
	if err != nil || len(results) != 1 || results[0].ID != "z" {
		t.Fatalf("results=%+v err=%v", results, err)
	}
}

func TestANNCheckpointBinaryFileParserOwnsAndReleasesMapping(t *testing.T) {
	base := t.TempDir()
	opts := checkpointServiceOptions(base)
	opts.ANNSegmentMaxNodes = 4
	opts.ANNOptions.QuantizeSegments = true
	svc := NewService(opts)
	for id := 0; id < 8; id++ {
		if err := svc.AddVector(fmt.Sprintf("mapped-%d", id), []float64{float64(id), 1}); err != nil {
			t.Fatal(err)
		}
	}
	if err := svc.Close(); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(opts.VectorPath, "ann-checkpoint.bin")
	fingerprint, ids, loaded, err := parseANNCheckpointBinaryFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if fingerprint == "" || len(ids) != 8 || loaded.Stats().Nodes != 8 {
		t.Fatalf("mapped checkpoint fingerprint=%q ids=%d stats=%+v", fingerprint, len(ids), loaded.Stats())
	}
	if err := loaded.Close(); err != nil {
		t.Fatal(err)
	}
	// Windows refuses deletion while a mapped view remains open.
	if err := os.Remove(path); err != nil {
		t.Fatalf("remove released checkpoint mapping: %v", err)
	}
}

func TestDiskIDResolverServiceCheckpointReopenAndCapacity(t *testing.T) {
	base := t.TempDir()
	opts := checkpointServiceOptions(base)
	opts.IDIndexCapacity = 8
	opts.ANNSegmentMaxNodes = 4
	opts.ANNOptions.QuantizeSegments = true
	svc := NewService(opts)
	for id := 0; id < 8; id++ {
		if err := svc.AddVector(fmt.Sprintf("disk-id-%d", id), []float64{float64(id), 1}); err != nil {
			t.Fatal(err)
		}
	}
	if err := svc.AddVector("over-capacity", []float64{99, 1}); !errors.Is(err, ErrLocationIndexFull) {
		t.Fatalf("capacity error = %v", err)
	}
	if _, err := svc.GetVector("over-capacity"); err == nil {
		t.Fatal("resolver capacity failure left vector committed")
	}
	if err := svc.Close(); err != nil {
		t.Fatal(err)
	}

	restored := NewService(opts)
	if !restored.Stats().ANNCheckpointLoaded {
		t.Fatal("disk ID resolver checkpoint was not restored")
	}
	results, err := restored.Search([]float64{7, 1}, 1)
	if err != nil || len(results) != 1 || results[0].ID != "disk-id-7" {
		t.Fatalf("restored results=%+v err=%v", results, err)
	}
	if err := restored.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestExplicitCheckpointClearsDirtyGeneration(t *testing.T) {
	opts := checkpointServiceOptions(t.TempDir())
	svc := NewService(opts)
	if err := svc.AddVector("dirty", []float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	if !svc.annCheckpointDirty.Load() {
		t.Fatal("write did not mark checkpoint dirty")
	}
	if err := svc.Checkpoint(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(opts.VectorPath, "locations.idx")); err != nil {
		t.Fatalf("recovery index missing after checkpoint: %v", err)
	}
	if svc.annCheckpointDirty.Load() {
		t.Fatal("successful checkpoint remained dirty")
	}
	if err := svc.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRetiredServiceSkipsANNCheckpoint(t *testing.T) {
	base := t.TempDir()
	opts := checkpointServiceOptions(base)
	svc := NewService(opts)
	if err := svc.AddVector("retired", []float64{1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := svc.CloseRetired(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(opts.VectorPath, "ann-checkpoint.bin")); !os.IsNotExist(err) {
		t.Fatalf("retired generation wrote ANN checkpoint: %v", err)
	}
	if _, err := os.Stat(filepath.Join(opts.VectorPath, "manifest.json")); err != nil {
		t.Fatalf("retired generation lost canonical segments: %v", err)
	}
}

func TestANNCheckpointMismatchFallsBackToRebuild(t *testing.T) {
	base := t.TempDir()
	opts := checkpointServiceOptions(base)
	svc := NewService(opts)
	if err := svc.AddVector("a", []float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	if err := svc.Close(); err != nil {
		t.Fatal(err)
	}

	store := newSegmentVectorStore(opts.VectorPath, DefaultStorageSecurityOptions())
	if err := store.UpsertVector(index.Vector{ID: "b", Values: []float64{2, 0}}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	restored := NewService(opts)
	t.Cleanup(func() { _ = restored.Close() })
	if restored.Stats().ANNCheckpointLoaded {
		t.Fatal("expected stale checkpoint to be rejected")
	}
	if stats := restored.Stats(); stats.ANNNodes != 2 {
		t.Fatalf("ANN stats after fallback = %+v", stats)
	}
}

func TestMemoryIDResolverCheckpointValidation(t *testing.T) {
	resolver := newMemoryIDResolver()
	if err := resolver.Restore(map[string]int{"a": 4, "b": 7}); err != nil {
		t.Fatal(err)
	}
	if got := resolver.Assign("c"); got != 8 {
		t.Fatalf("next internal ID = %d", got)
	}
	for _, entries := range []map[string]int{
		{"": 1},
		{"a": 0},
		{"a": 1, "b": 1},
	} {
		if err := resolver.Restore(entries); err == nil {
			t.Fatalf("expected resolver validation error for %+v", entries)
		}
	}
}

func TestANNCheckpointDisabledAndCorruptPaths(t *testing.T) {
	plain := &Service{}
	if plain.annCheckpointPath() != "" {
		t.Fatal("expected empty checkpoint path")
	}
	if err := plain.saveANNCheckpoint(nil); err != nil {
		t.Fatal(err)
	}
	if plain.loadANNCheckpoint(nil) {
		t.Fatal("non-segmented service must not load checkpoint")
	}
	segmentedWithoutPath := &Service{annSegmented: true}
	if err := segmentedWithoutPath.saveANNCheckpoint(nil); err != nil {
		t.Fatal(err)
	}
	if segmentedWithoutPath.loadANNCheckpoint(nil) {
		t.Fatal("checkpoint without path must not load")
	}
	nonSegmentedIndex := &Service{
		annSegmented: true,
		vectorPath:   t.TempDir(),
		annIndex:     ann.NewAnnIndex(),
	}
	if err := nonSegmentedIndex.saveANNCheckpoint(nil); err != nil {
		t.Fatal(err)
	}

	base := t.TempDir()
	opts := checkpointServiceOptions(base)
	svc := NewService(opts)
	if err := svc.AddVector("a", []float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	if err := svc.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(opts.VectorPath, "ann-checkpoint.bin"), []byte("corrupt"), 0o644); err != nil {
		t.Fatal(err)
	}
	restored := NewService(opts)
	if restored.Stats().ANNCheckpointLoaded {
		t.Fatal("corrupt checkpoint must be rejected")
	}
	_ = restored.Close()

	if err := os.Remove(filepath.Join(opts.VectorPath, "ann-checkpoint.bin")); err != nil {
		t.Fatal(err)
	}
	versionPath := filepath.Join(opts.VectorPath, "ann-checkpoint.json")
	if err := os.WriteFile(versionPath, []byte(`{"version":99}`), 0o644); err != nil {
		t.Fatal(err)
	}
	probe := &Service{annSegmented: true, vectorPath: opts.VectorPath}
	if probe.loadANNCheckpoint(nil) {
		t.Fatal("unsupported checkpoint version must be rejected")
	}

	resolver := newMemoryIDResolver()
	resolver.Remove("missing")
}

func TestANNCheckpointRebuildsStaleRecoveryOffsets(t *testing.T) {
	opts := checkpointServiceOptions(t.TempDir())
	svc := NewService(opts)
	if err := svc.AddVectors([]index.Vector{
		{ID: "a", Values: []float64{1, 0}},
		{ID: "b", Values: []float64{0, 1}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := svc.Checkpoint(); err != nil {
		t.Fatal(err)
	}
	if err := svc.CloseRetired(); err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(opts.VectorPath, "locations.idx")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	body := data[:len(data)-sha256.Size]
	pos := len(segmentRecoveryIndexMagic) + sha256.Size
	segmentCount := int(binary.LittleEndian.Uint32(body[pos : pos+4]))
	pos += 4 + segmentCount*sha256.Size + 8
	idLength := int(binary.LittleEndian.Uint32(body[pos : pos+4]))
	pos += 4 + idLength + 4
	// Keep the sidecar checksum valid but point the first ID into the middle of
	// its record. Fingerprint validation must reject it and rebuild offsets
	// from canonical segments.
	binary.LittleEndian.PutUint64(body[pos:pos+8], 1)
	sum := sha256.Sum256(body)
	copy(data[len(body):], sum[:])
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}

	restored := NewService(opts)
	defer restored.CloseRetired()
	if !restored.Stats().ANNCheckpointLoaded {
		t.Fatal("checkpoint did not load after safe offset rebuild")
	}
	store := restored.vectorStore.(*segmentVectorStore)
	if store.recoveryLoaded {
		t.Fatal("invalid recovery sidecar remained active")
	}
	if _, err := restored.GetVector("a"); err != nil {
		t.Fatalf("canonical vector unavailable after rebuild: %v", err)
	}
}

func BenchmarkANNCheckpointStreaming(b *testing.B) {
	opts := checkpointServiceOptions(b.TempDir())
	opts.ANNSegmentMaxNodes = 1000
	opts.MaxVectorDim = 32
	svc := NewService(opts)
	defer svc.CloseRetired()
	vectors := make([]index.Vector, 10_000)
	for i := range vectors {
		values := make([]float64, 32)
		for j := range values {
			values[j] = float64((i + j) % 101)
		}
		vectors[i] = index.Vector{ID: fmt.Sprintf("v-%08d", i), Values: values}
	}
	if err := svc.AddVectors(vectors); err != nil {
		b.Fatal(err)
	}
	if err := svc.currentANNIndex().(*ann.SegmentedIndex).Close(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := svc.saveANNCheckpointFromStore(); err != nil {
			b.Fatal(err)
		}
	}
}
