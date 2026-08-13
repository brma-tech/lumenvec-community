package core

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestDeltaWALAppendReplayAndReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "delta.wal")
	wal, err := OpenDeltaWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	if offset, err := wal.Append(DeltaRecord{VectorID: "a", Values: []float32{1, 2}}); err != nil || offset != 1 {
		t.Fatalf("append offset=%d err=%v", offset, err)
	}
	if offset, err := wal.Append(DeltaRecord{VectorID: "b", Deleted: true}); err != nil || offset != 2 {
		t.Fatalf("append offset=%d err=%v", offset, err)
	}
	if err := wal.Close(); err != nil {
		t.Fatal(err)
	}
	wal, err = OpenDeltaWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	defer wal.Close()
	if !wal.checkpointLoaded {
		t.Fatal("expected O(1) WAL tail checkpoint on reopen")
	}
	var got []DeltaRecord
	if err := wal.ReplaySince(0, func(record DeltaRecord) error { got = append(got, record); return nil }); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Offset != 1 || !got[1].Deleted {
		t.Fatalf("replay = %+v", got)
	}
	var resumed []DeltaRecord
	if err := wal.ReplaySince(1, func(record DeltaRecord) error { resumed = append(resumed, record); return nil }); err != nil {
		t.Fatal(err)
	}
	if len(resumed) != 1 || resumed[0].VectorID != "b" {
		t.Fatalf("resumed replay = %+v", resumed)
	}
}

func TestDeltaWALStaleTailCheckpointFallsBackToScan(t *testing.T) {
	path := filepath.Join(t.TempDir(), "delta.wal")
	wal, err := OpenDeltaWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := wal.Append(DeltaRecord{VectorID: "a", Values: []float32{1}}); err != nil {
		t.Fatal(err)
	}
	if err := wal.Close(); err != nil {
		t.Fatal(err)
	}
	record, err := marshalDeltaBinaryRecord(DeltaRecord{Offset: 2, VectorID: "b", Values: []float32{2}})
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o640)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(record); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	wal, err = OpenDeltaWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	defer wal.Close()
	if wal.checkpointLoaded {
		t.Fatal("stale WAL checkpoint must not be trusted")
	}
	if got := wal.CurrentOffset(); got != 2 {
		t.Fatalf("fallback offset=%d want=2", got)
	}
}

func TestDeltaWALReadsAndAppendsLegacyJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "delta.wal")
	first, err := json.Marshal(DeltaRecord{Offset: 1, VectorID: "legacy", Values: []float32{1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(first, '\n'), 0o640); err != nil {
		t.Fatal(err)
	}
	wal, err := OpenDeltaWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	if wal.format != deltaWALLegacyJSON {
		t.Fatalf("format=%v", wal.format)
	}
	if offset, err := wal.Append(DeltaRecord{VectorID: "new", Deleted: true}); err != nil || offset != 2 {
		t.Fatalf("offset=%d err=%v", offset, err)
	}
	var got []DeltaRecord
	if err := wal.ReplaySince(0, func(record DeltaRecord) error {
		got = append(got, record)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].VectorID != "legacy" || got[1].VectorID != "new" {
		t.Fatalf("records=%+v", got)
	}
	if err := wal.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestDeltaWALBinaryChecksumCorruptionFailsClosed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "delta.wal")
	wal, err := OpenDeltaWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := wal.Append(DeltaRecord{VectorID: "safe", Values: []float32{1, 2, 3}}); err != nil {
		t.Fatal(err)
	}
	if err := wal.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)-5] ^= 0xff
	if err := os.WriteFile(path, data, 0o640); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenDeltaWAL(path); err == nil {
		t.Fatal("expected checksum corruption to fail open")
	}
}

func TestDeltaWALBinaryTruncationFailsClosed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "delta.wal")
	wal, err := OpenDeltaWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := wal.Append(DeltaRecord{VectorID: "safe", Values: []float32{1, 2, 3}}); err != nil {
		t.Fatal(err)
	}
	if err := wal.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data[:len(data)-3], 0o640); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenDeltaWAL(path); err == nil {
		t.Fatal("expected truncated binary WAL to fail open")
	}
}

func TestDeltaWALBinaryEncodingIsCompact(t *testing.T) {
	record := DeltaRecord{Offset: 1, VectorID: "vector-000001", Values: make([]float32, 128)}
	for i := range record.Values {
		record.Values[i] = float32(i) / 7
	}
	binaryRecord, err := marshalDeltaBinaryRecord(record)
	if err != nil {
		t.Fatal(err)
	}
	jsonRecord, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	if len(binaryRecord) >= len(jsonRecord)/2 {
		t.Fatalf("binary=%d JSON=%d, expected at least 50%% reduction", len(binaryRecord), len(jsonRecord))
	}
	if bytes.Contains(binaryRecord, []byte(`"values"`)) {
		t.Fatal("binary record unexpectedly contains JSON")
	}
}

func BenchmarkDeltaWALRecordEncoding(b *testing.B) {
	record := DeltaRecord{Offset: 1, VectorID: "vector-000001", Values: make([]float32, 128)}
	for i := range record.Values {
		record.Values[i] = float32(i) / 7
	}
	b.Run("binary-v2", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			data, err := marshalDeltaBinaryRecord(record)
			if err != nil {
				b.Fatal(err)
			}
			b.SetBytes(int64(len(data)))
		}
	})
	b.Run("json-v1", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			data, err := json.Marshal(record)
			if err != nil {
				b.Fatal(err)
			}
			b.SetBytes(int64(len(data)))
		}
	})
}

func TestDeltaWALRejectsInvalidRecords(t *testing.T) {
	wal, err := OpenDeltaWAL(filepath.Join(t.TempDir(), "delta.wal"))
	if err != nil {
		t.Fatal(err)
	}
	defer wal.Close()
	if _, err := wal.Append(DeltaRecord{VectorID: ""}); err == nil {
		t.Fatal("expected invalid record rejection")
	}
	if _, err := wal.Append(DeltaRecord{VectorID: "x", Deleted: true, Values: []float32{1}}); err == nil {
		t.Fatal("expected delete/value rejection")
	}
}

func TestDeltaWALAppendBatchAssignsContiguousOffsets(t *testing.T) {
	wal, err := OpenDeltaWAL(filepath.Join(t.TempDir(), "delta.wal"))
	if err != nil {
		t.Fatal(err)
	}
	defer wal.Close()
	offsets, err := wal.AppendBatch([]DeltaRecord{
		{VectorID: "a", Values: []float32{1}},
		{VectorID: "b", Values: []float32{2}},
		{VectorID: "c", Deleted: true},
	})
	if err != nil || len(offsets) != 3 || offsets[0] != 1 || offsets[1] != 2 || offsets[2] != 3 {
		t.Fatalf("offsets=%v err=%v", offsets, err)
	}
	if got := wal.CurrentOffset(); got != 3 {
		t.Fatalf("current offset=%d", got)
	}
}

func TestDeltaWALReplayUsesAppendBatchPositionAfterCutover(t *testing.T) {
	wal, err := OpenDeltaWAL(filepath.Join(t.TempDir(), "delta.wal"))
	if err != nil {
		t.Fatal(err)
	}
	defer wal.Close()
	if _, err := wal.AppendBatch([]DeltaRecord{
		{VectorID: "old-a", Values: []float32{1}},
		{VectorID: "old-b", Values: []float32{2}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := wal.AppendBatch([]DeltaRecord{
		{VectorID: "new-a", Values: []float32{3}},
		{VectorID: "new-b", Values: []float32{4}},
	}); err != nil {
		t.Fatal(err)
	}
	if len(wal.appendPositions) != 2 {
		t.Fatalf("append positions=%d want 2", len(wal.appendPositions))
	}

	var got []string
	if err := wal.ReplaySince(2, func(record DeltaRecord) error {
		got = append(got, record.VectorID)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []string{"new-a", "new-b"}) {
		t.Fatalf("replayed IDs=%v", got)
	}
}
