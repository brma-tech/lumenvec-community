package ann

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestSegmentPayloadWriterStreamsAndSeals(t *testing.T) {
	var data bytes.Buffer
	w, err := NewSegmentPayloadWriter(&data, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.WriteVector(7, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	header, err := w.Close()
	if err != nil {
		t.Fatal(err)
	}
	if header.Count != 1 || header.Dimension != 2 || data.Len() == 0 {
		t.Fatalf("header=%+v bytes=%d", header, data.Len())
	}
	if len(w.SHA256()) != 64 {
		t.Fatalf("checksum=%q", w.SHA256())
	}
	if err := w.WriteVector(8, []float32{3, 4}); err == nil {
		t.Fatal("expected closed writer failure")
	}
	var seen int
	header, err = ReadSegmentPayload(bytes.NewReader(data.Bytes()), func(id int, values []float32) error {
		if id != 7 || len(values) != 2 {
			t.Fatalf("record=%d values=%v", id, values)
		}
		seen++
		return nil
	})
	if err != nil || seen != 1 || header.Count != 1 {
		t.Fatalf("header=%+v seen=%d err=%v", header, seen, err)
	}
}

func BenchmarkReadSegmentPayload(b *testing.B) {
	var data bytes.Buffer
	w, err := NewSegmentPayloadWriter(&data, 16)
	if err != nil {
		b.Fatal(err)
	}
	values := make([]float32, 16)
	for i := 0; i < 10000; i++ {
		values[0] = float32(i)
		if err := w.WriteVector(i, values); err != nil {
			b.Fatal(err)
		}
	}
	if _, err := w.Close(); err != nil {
		b.Fatal(err)
	}
	raw := append([]byte(nil), data.Bytes()...)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ReadSegmentPayload(bytes.NewReader(raw), func(_ int, _ []float32) error { return nil })
		if err != nil {
			b.Fatal(err)
		}
	}
}

func TestWriteSegmentPayloadFileAtomic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "segment.payload")
	header, checksum, err := WriteSegmentPayloadFile(path, 2, func(w *SegmentPayloadWriter) error {
		return w.WriteVector(11, []float32{1, 2})
	})
	if err != nil {
		t.Fatal(err)
	}
	if header.Count != 1 || len(checksum) != 64 {
		t.Fatalf("header=%+v checksum=%q", header, checksum)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
}
