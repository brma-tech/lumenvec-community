package ann

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestAnnIndexQuantizeReducesVectorMemoryAndRemainsSearchable(t *testing.T) {
	const (
		count = 2000
		dim   = 32
	)
	rng := rand.New(rand.NewSource(91))
	idx := NewAnnIndexWithOptions(Options{M: 16, EfConstruction: 96, EfSearch: 128, Seed: 91})
	vectors := make([][]float64, count)
	for id := 0; id < count; id++ {
		vector := make([]float64, dim)
		for d := range vector {
			vector[d] = rng.Float64()*2 - 1
		}
		vectors[id] = vector
		if err := idx.AddVector(id, vector); err != nil {
			t.Fatal(err)
		}
	}
	before := idx.MemoryStats()
	if err := idx.Quantize(); err != nil {
		t.Fatal(err)
	}
	after := idx.MemoryStats()
	if idx.ExactDistances() {
		t.Fatal("quantized index reported exact distances")
	}
	if after.VectorBytes != 0 {
		t.Fatalf("float vectors remain resident: %d bytes", after.VectorBytes)
	}
	if idx.idToSlot != nil {
		t.Fatal("sorted immutable segment retained ID-to-slot map")
	}
	if len(idx.nodes) != 0 || len(idx.immutableIDs) != count || len(idx.deleted) != 0 {
		t.Fatalf("quantized node metadata was not columnar: nodes=%d ids=%d deleted=%d", len(idx.nodes), len(idx.immutableIDs), len(idx.deleted))
	}
	if after.QuantizedBytes >= before.VectorBytes/2 {
		t.Fatalf("quantized payload %d did not materially reduce float payload %d", after.QuantizedBytes, before.VectorBytes)
	}
	for _, id := range []int{0, 311, 1024, count - 1} {
		results, err := idx.SearchWithDistances(vectors[id], 10)
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, result := range results {
			if result.ID == id {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("exact vector %d absent from quantized top-10", id)
		}
	}
	if err := idx.AddVector(count, vectors[0]); !errors.Is(err, ErrQuantizedIndex) {
		t.Fatalf("AddVector error = %v, want ErrQuantizedIndex", err)
	}
	idx.DeleteVector(311)
	if stats := idx.Stats(); stats.Deleted != 1 {
		t.Fatalf("quantized binary-search delete stats = %+v", stats)
	}
}

func TestQuantizedAnnIndexCompactSnapshotRemainsCompatible(t *testing.T) {
	idx := NewAnnIndex()
	for id, values := range [][]float64{{-1, 0}, {0, 1}, {1, 0}} {
		if err := idx.AddVector(id+1, values); err != nil {
			t.Fatal(err)
		}
	}
	idx.CompactAdjacency()
	if err := idx.Quantize(); err != nil {
		t.Fatal(err)
	}
	data, err := idx.MarshalCompactBinary()
	if err != nil {
		t.Fatal(err)
	}
	if len(data) >= 3*2*4+3*2*4+256 {
		t.Fatalf("quantized snapshot unexpectedly large: %d bytes", len(data))
	}
	restored, err := UnmarshalCompactBinary(data)
	if err != nil {
		t.Fatal(err)
	}
	if restored.ExactDistances() {
		t.Fatal("quantized snapshot expanded to exact float vectors")
	}
	if restored.idToSlot != nil {
		t.Fatal("restored sorted quantized segment retained ID map")
	}
	if len(restored.nodes) != 0 || len(restored.mappedIDs) == 0 || len(restored.deletedBits) == 0 {
		t.Fatal("restored v4 metadata was not mapped/bit-packed")
	}
	if restored.MemoryStats().QuantizedBytes == 0 || restored.MemoryStats().VectorBytes != 0 {
		t.Fatalf("quantized representation was not preserved: %+v", restored.MemoryStats())
	}
	results, err := restored.SearchWithDistances([]float64{0, 1}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].ID != 2 {
		t.Fatalf("restored results = %+v", results)
	}
}

func TestQuantizedAnnIndexRetainsMapForUnsortedIDs(t *testing.T) {
	idx := NewAnnIndex()
	if err := idx.AddVector(9, []float64{0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := idx.AddVector(3, []float64{1, 1}); err != nil {
		t.Fatal(err)
	}
	if err := idx.Quantize(); err != nil {
		t.Fatal(err)
	}
	if idx.idToSlot == nil {
		t.Fatal("unsorted segment discarded required ID map")
	}
	idx.DeleteVector(3)
	if stats := idx.Stats(); stats.Deleted != 1 {
		t.Fatalf("unsorted quantized delete stats = %+v", stats)
	}
}

func TestQuantizedV4SnapshotOpensMappedAndReleasesFile(t *testing.T) {
	idx := NewAnnIndexWithOptions(Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 17})
	for id := 0; id < 128; id++ {
		if err := idx.AddVector(id, []float64{float64(id), float64(id % 7), 1, -1}); err != nil {
			t.Fatal(err)
		}
	}
	idx.CompactAdjacency()
	if err := idx.Quantize(); err != nil {
		t.Fatal(err)
	}
	want, err := idx.Search([]float64{64, 1, 1, -1}, 10)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := idx.MarshalCompactBinary()
	if err != nil {
		t.Fatal(err)
	}
	if string(raw[:len(annBinaryMagic)]) != string(annBinaryMagic[:]) {
		t.Fatal("compact snapshot was not encoded as v4")
	}
	path := filepath.Join(t.TempDir(), "segment.ann")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	restored, err := OpenCompactBinaryFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if restored.mappedOwner == nil || len(restored.mappedOffsets) == 0 || len(restored.mappedAdjacency) == 0 {
		t.Fatal("v4 snapshot did not retain mapped columnar sections")
	}
	got, err := restored.Search([]float64{64, 1, 1, -1}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("mapped result count = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("mapped results = %v, want %v", got, want)
		}
	}
	if err := restored.Close(); err != nil {
		t.Fatal(err)
	}
	if err := restored.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
	// On Windows this also proves that the mapping handle/view was released.
	if err := os.Remove(path); err != nil {
		t.Fatalf("remove closed mapped snapshot: %v", err)
	}
}

func TestQuantizedV4SnapshotRejectsCorruptOffsets(t *testing.T) {
	idx := NewAnnIndexWithOptions(Options{M: 4, EfConstruction: 8, EfSearch: 8})
	for id := 0; id < 4; id++ {
		if err := idx.AddVector(id, []float64{float64(id), 1}); err != nil {
			t.Fatal(err)
		}
	}
	idx.CompactAdjacency()
	if err := idx.Quantize(); err != nil {
		t.Fatal(err)
	}
	raw, err := idx.MarshalCompactBinary()
	if err != nil {
		t.Fatal(err)
	}
	// Header, flags, ranges, IDs, tombstones, and vector codes precede
	// the first uint64 adjacency offset.
	count := idx.Stats().Nodes
	offsetsStart := len(annBinaryMagic) + 8*8 + 2 + idx.dim*8 + count*8 + count + count*idx.dim
	raw[offsetsStart] = 1
	if _, err := UnmarshalCompactBinary(raw); err == nil {
		t.Fatal("corrupt first adjacency offset was accepted")
	}
}

func TestQuantizedV3SnapshotRemainsReadable(t *testing.T) {
	idx := NewAnnIndexWithOptions(Options{M: 4, EfConstruction: 8, EfSearch: 8, Seed: 29})
	for id := 0; id < 12; id++ {
		if err := idx.AddVector(id, []float64{float64(id), float64(id % 3)}); err != nil {
			t.Fatal(err)
		}
	}
	idx.DeleteVector(5)
	idx.CompactAdjacency()
	if err := idx.Quantize(); err != nil {
		t.Fatal(err)
	}
	raw := marshalLegacyV3ForTest(t, idx)
	restored, err := UnmarshalCompactBinary(raw)
	if err != nil {
		t.Fatal(err)
	}
	defer restored.Close()
	if got := restored.Stats(); got.Nodes != 12 || got.Deleted != 1 {
		t.Fatalf("v3 stats = %+v", got)
	}
	if got, err := restored.Search([]float64{7, 1}, 3); err != nil || len(got) != 3 {
		t.Fatalf("v3 search = %v, %v", got, err)
	}
}

func marshalLegacyV3ForTest(t *testing.T, idx *AnnIndex) []byte {
	t.Helper()
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	var out bytes.Buffer
	out.Write(annBinaryMagicV3[:])
	writeI64 := func(value int64) { _ = binary.Write(&out, binary.LittleEndian, value) }
	for _, value := range []int64{
		int64(idx.dim), int64(idx.m), int64(idx.efConstruction), int64(idx.efSearch),
		idx.seed, int64(idx.entrypoint), int64(idx.deletedCount), int64(idx.nodeCountLocked()),
	} {
		writeI64(value)
	}
	if idx.hasEntrypoint {
		out.WriteByte(1)
	} else {
		out.WriteByte(0)
	}
	out.WriteByte(1)
	for _, values := range [][]float32{idx.quantMin, idx.quantMax} {
		for _, value := range values {
			_ = binary.Write(&out, binary.LittleEndian, value)
		}
	}
	for slot := 0; slot < idx.nodeCountLocked(); slot++ {
		writeI64(int64(idx.nodeIDLocked(slot)))
		if idx.isDeletedLocked(slot) {
			out.WriteByte(1)
		} else {
			out.WriteByte(0)
		}
		writeI64(int64(idx.dim))
		out.Write(idx.qvectorLocked(slot))
		count := idx.neighborCountLocked(slot)
		writeI64(int64(count))
		for position := 0; position < count; position++ {
			_ = binary.Write(&out, binary.LittleEndian, int32(idx.neighborAtLocked(slot, position)))
		}
	}
	return out.Bytes()
}

func TestSegmentedIndexQuantizesOnlySealedSegments(t *testing.T) {
	idx := NewSegmentedIndex(Options{QuantizeSegments: true, EfSearch: 64}, 100)
	defer idx.Close()
	vectors := make([]BatchVector, 150)
	for id := range vectors {
		vectors[id] = BatchVector{ID: id + 1, Values: []float64{float64(id) / 150, float64(id%7) / 7}}
	}
	if err := idx.AddBatch(vectors); err != nil {
		t.Fatal(err)
	}
	if idx.ExactDistances() {
		t.Fatal("segmented quantized index reported exact distances")
	}
	stats := idx.MemoryStats()
	if stats.QuantizedBytes == 0 || stats.VectorBytes == 0 {
		t.Fatalf("expected sealed quantized payload and mutable float delta: %+v", stats)
	}
	results, err := idx.SearchWithDistancesInto(vectors[120].Values, 10, nil)
	if err != nil || len(results) == 0 {
		t.Fatalf("search failed: results=%+v err=%v", results, err)
	}
}

func BenchmarkQuantizedHNSWSearch10k(b *testing.B) {
	const (
		count = 10_000
		dim   = 128
	)
	rng := rand.New(rand.NewSource(99))
	idx := NewAnnIndexWithOptionsCapacity(Options{M: 16, EfConstruction: 64, EfSearch: 128, Seed: 99}, count)
	var query []float64
	for id := 0; id < count; id++ {
		values := make([]float64, dim)
		for d := range values {
			values[d] = rng.Float64()*2 - 1
		}
		if id == count/2 {
			query = append([]float64(nil), values...)
		}
		if err := idx.AddVector(id, values); err != nil {
			b.Fatal(err)
		}
	}
	idx.CompactAdjacency()
	before := idx.MemoryStats()
	if err := idx.Quantize(); err != nil {
		b.Fatal(err)
	}
	after := idx.MemoryStats()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		if _, err := idx.SearchWithDistancesInto(query, 40, nil); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(before.VectorBytes)/(1<<20), "float-vector-MiB")
	b.ReportMetric(float64(after.QuantizedBytes)/(1<<20), "quantized-vector-MiB")
	b.ReportMetric(float64(after.TotalBytes)/float64(before.TotalBytes), "memory-ratio")
}
