package ann

import (
	"bytes"
	"encoding/json"
	"io"
	"testing"
)

func TestAnnIndexSnapshotRoundTrip(t *testing.T) {
	idx := NewAnnIndexWithOptions(Options{M: 8, EfConstruction: 32, EfSearch: 16, Seed: 7})
	for i := 0; i < 10; i++ {
		if err := idx.AddVector(i, []float64{float64(i), 1}); err != nil {
			t.Fatal(err)
		}
	}
	idx.DeleteVector(3)
	data, err := idx.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := UnmarshalBinary(data)
	if err != nil {
		t.Fatal(err)
	}
	results, err := restored.SearchWithDistances([]float64{9.1, 1}, 3)
	if err != nil || len(results) != 3 || results[0].ID != 9 {
		t.Fatalf("results=%+v err=%v", results, err)
	}
	if stats := restored.Stats(); stats.Nodes != 10 || stats.Deleted != 1 {
		t.Fatalf("stats=%+v", stats)
	}
}

func TestCompactSnapshotRoundTrip(t *testing.T) {
	idx := NewAnnIndexWithOptions(Options{M: 8, EfConstruction: 32, EfSearch: 32})
	if err := idx.AddVector(7, []float64{1, 2}); err != nil {
		t.Fatal(err)
	}
	raw, err := idx.MarshalCompactBinary()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := UnmarshalCompactBinary(raw)
	if err != nil {
		t.Fatal(err)
	}
	if got := restored.Stats().Nodes; got != 1 {
		t.Fatalf("nodes=%d", got)
	}
}

func TestSegmentedSnapshotRoundTrip(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 8, EfConstruction: 32, EfSearch: 16, Seed: 7}, 3)
	for i := 0; i < 8; i++ {
		if err := idx.AddVector(i, []float64{float64(i), 1}); err != nil {
			t.Fatal(err)
		}
	}
	data, err := idx.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := UnmarshalSegmentedBinary(data)
	if err != nil {
		t.Fatal(err)
	}
	if restored.SegmentCount() != idx.SegmentCount() || restored.Stats().Nodes != 8 {
		t.Fatalf("restored segments=%d stats=%+v", restored.SegmentCount(), restored.Stats())
	}
}

func TestSegmentedCompactSnapshotRoundTripAndValidation(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 8, EfConstruction: 32, EfSearch: 16, Seed: 11}, 3)
	defer idx.Close()
	for i := 0; i < 8; i++ {
		if err := idx.AddVector(i, []float64{float64(i), 1}); err != nil {
			t.Fatal(err)
		}
	}
	data, err := idx.MarshalSegmentedCompactBinary()
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(data, []byte(`"nodes"`)) {
		t.Fatal("compact snapshot unexpectedly contains JSON graph fields")
	}
	restored, err := UnmarshalSegmentedCompactBinary(data)
	if err != nil {
		t.Fatal(err)
	}
	defer restored.Close()
	if restored.Stats().Nodes != 8 || restored.SegmentCount() != idx.SegmentCount() {
		t.Fatalf("restored segments=%d stats=%+v", restored.SegmentCount(), restored.Stats())
	}
	results, err := restored.SearchWithDistancesInto([]float64{7.1, 1}, 1, nil)
	if err != nil || len(results) != 1 || results[0].ID != 7 {
		t.Fatalf("results=%+v err=%v", results, err)
	}
	for _, invalid := range [][]byte{data[:7], append(append([]byte(nil), data...), 1)} {
		if _, err := UnmarshalSegmentedCompactBinary(invalid); err == nil {
			t.Fatal("expected compact segmented validation error")
		}
	}
}

func TestCompactSnapshotStreamingMatchesMarshal(t *testing.T) {
	idx := NewSegmentedIndex(Options{M: 8, EfConstruction: 32, EfSearch: 16, Seed: 23}, 4)
	defer idx.Close()
	for i := 0; i < 11; i++ {
		if err := idx.AddVector(i, []float64{float64(i), 1, 2, 3}); err != nil {
			t.Fatal(err)
		}
	}
	marshaled, err := idx.MarshalSegmentedCompactBinary()
	if err != nil {
		t.Fatal(err)
	}
	var streamed bytes.Buffer
	written, err := idx.WriteSegmentedCompactBinary(&streamed)
	if err != nil {
		t.Fatal(err)
	}
	if written != uint64(len(marshaled)) {
		t.Fatalf("written=%d marshaled=%d", written, len(marshaled))
	}
	if !bytes.Equal(streamed.Bytes(), marshaled) {
		t.Fatal("streamed compact snapshot differs from compatibility marshal")
	}
}

type shortSnapshotWriter struct{}

func (shortSnapshotWriter) Write(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	return len(p) - 1, nil
}

func TestCompactSnapshotStreamingPropagatesShortWrite(t *testing.T) {
	idx := NewAnnIndexWithOptions(Options{M: 8, EfConstruction: 32, EfSearch: 16})
	if err := idx.AddVector(1, []float64{1, 2}); err != nil {
		t.Fatal(err)
	}
	if _, err := idx.WriteCompactBinary(shortSnapshotWriter{}); err != io.ErrShortWrite {
		t.Fatalf("err=%v want %v", err, io.ErrShortWrite)
	}
}

var compactSnapshotSink []byte

func BenchmarkSegmentedCompactSnapshotEncoding(b *testing.B) {
	idx := NewSegmentedIndex(Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 29}, 1000)
	defer idx.Close()
	values := make([]float64, 32)
	for i := 0; i < 10_000; i++ {
		for j := range values {
			values[j] = float64((i + j) % 97)
		}
		if err := idx.AddVector(i, values); err != nil {
			b.Fatal(err)
		}
	}
	b.Run("buffered-compatibility", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			data, err := idx.MarshalSegmentedCompactBinary()
			if err != nil {
				b.Fatal(err)
			}
			compactSnapshotSink = data
		}
	})
	b.Run("streaming", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := idx.WriteSegmentedCompactBinary(io.Discard); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkSegmentedCompactSnapshotDecoding(b *testing.B) {
	idx := NewSegmentedIndex(Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 31}, 1000)
	values := make([]float64, 32)
	for i := 0; i < 10_000; i++ {
		for j := range values {
			values[j] = float64((i + j) % 97)
		}
		if err := idx.AddVector(i, values); err != nil {
			b.Fatal(err)
		}
	}
	data, err := idx.MarshalSegmentedCompactBinary()
	if err != nil {
		b.Fatal(err)
	}
	if err := idx.Close(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		restored, err := UnmarshalSegmentedCompactBinary(data)
		if err != nil {
			b.Fatal(err)
		}
		if err := restored.Close(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSegmentedQualityIngest128(b *testing.B) {
	benchmarkSegmentedQualityIngest128(b, false)
}

func BenchmarkSegmentedDiversifiedQualityIngest128(b *testing.B) {
	benchmarkSegmentedQualityIngest128(b, true)
}

func benchmarkSegmentedQualityIngest128(b *testing.B, diversified bool) {
	idx := NewSegmentedIndex(Options{M: 32, EfConstruction: 128, EfSearch: 512, Seed: 37, DiversifiedPruning: diversified}, 10_000)
	defer idx.Close()
	batch := make([]BatchVector32, 1000)
	state := uint32(1)
	for i := range batch {
		values := make([]float32, 128)
		for j := range values {
			state = state*1664525 + 1013904223
			values[j] = float32(state&0xffff) / 65535
		}
		batch[i] = BatchVector32{ID: i + 1, Values: values}
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for j := range batch {
			batch[j].ID = i*len(batch) + j + 1
		}
		if err := idx.AddBatch32(batch); err != nil {
			b.Fatal(err)
		}
	}
}

func TestAnnSnapshotValidation(t *testing.T) {
	invalid := []any{
		indexSnapshot{Version: 99},
		indexSnapshot{Version: 1, Nodes: []nodeSnapshot{{Slot: 2}}, Deleted: []bool{false}},
		indexSnapshot{Version: 1, Nodes: []nodeSnapshot{{Slot: 0, Neighbors: []int{2}}}, Deleted: []bool{false}},
		indexSnapshot{Version: 1, Nodes: []nodeSnapshot{{Slot: 0, ID: 1}, {Slot: 1, ID: 1}}, Deleted: []bool{false, false}},
	}
	for _, value := range invalid {
		data, _ := json.Marshal(value)
		if _, err := UnmarshalBinary(data); err == nil {
			t.Fatalf("expected validation error for %+v", value)
		}
	}
	if _, err := UnmarshalSegmentedBinary([]byte(`{"version":99}`)); err == nil {
		t.Fatal("expected segmented validation error")
	}
}

func TestCompactSnapshotRebuildsSegmentRoutingDirectory(t *testing.T) {
	idx := NewSegmentedIndex(Options{
		M:                4,
		EfConstruction:   8,
		EfSearch:         16,
		QuantizeSegments: true,
		SegmentRouting:   true,
	}, 1)
	for item := 0; item < 12; item++ {
		values := make([]float64, routeSignatureBits)
		for dimension := range values {
			values[dimension] = -1
		}
		values[item] = 1
		if err := idx.AddVector(item+1, values); err != nil {
			t.Fatal(err)
		}
	}
	data, err := idx.MarshalSegmentedCompactBinary()
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Close(); err != nil {
		t.Fatal(err)
	}
	restored, err := UnmarshalSegmentedCompactBinary(data)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = restored.Close() })
	// The compact format remains compatible with v4; Service reapplies the
	// runtime routing policy after loading the checkpoint.
	restored.SetSegmentRouting(true)

	query := make([]float64, routeSignatureBits)
	for dimension := range query {
		query[dimension] = -1
	}
	query[11] = 1
	routed := restored.routedIndexes(query)
	if len(routed) != 13 { // twelve immutable graphs plus the delta
		t.Fatalf("restored routing selected %d indexes, want full small-generation fanout", len(routed))
	}
	results, err := restored.SearchWithDistancesInto(query, 1, nil)
	if err != nil || len(results) != 1 || results[0].ID != 12 {
		t.Fatalf("restored results=%+v err=%v", results, err)
	}
}
