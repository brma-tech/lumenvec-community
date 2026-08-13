package core

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func openTestDiskLocationIndex(t testing.TB, capacity uint64) (*diskLocationIndex, string, string) {
	t.Helper()
	dir := t.TempDir()
	tablePath := filepath.Join(dir, "locations.hash")
	keysPath := filepath.Join(dir, "locations.keys")
	index, err := openDiskLocationIndex(tablePath, keysPath, capacity)
	if err != nil {
		t.Fatal(err)
	}
	return index, tablePath, keysPath
}

func TestDiskLocationIndexPutGetUpdateDeleteAndReopen(t *testing.T) {
	index, tablePath, keysPath := openTestDiskLocationIndex(t, 32)
	first := diskLocationValue{SegmentGeneration: 4, RecordOffset: 120, RecordLength: 80}
	if err := index.Put("alpha", first); err != nil {
		t.Fatal(err)
	}
	if err := index.Put("beta", diskLocationValue{SegmentGeneration: 5, RecordOffset: 300, RecordLength: 96}); err != nil {
		t.Fatal(err)
	}
	updated := diskLocationValue{SegmentGeneration: 6, RecordOffset: 900, RecordLength: 100}
	if err := index.Put("alpha", updated); err != nil {
		t.Fatal(err)
	}
	if index.Count() != 2 {
		t.Fatalf("count = %d", index.Count())
	}
	if got, ok, err := index.Get("alpha"); err != nil || !ok || got != updated {
		t.Fatalf("Get(alpha) = %+v, %v, %v", got, ok, err)
	}
	if deleted, err := index.Delete("beta"); err != nil || !deleted {
		t.Fatalf("Delete(beta) = %v, %v", deleted, err)
	}
	if _, ok, err := index.Get("beta"); err != nil || ok {
		t.Fatalf("deleted beta returned ok=%v err=%v", ok, err)
	}
	if err := index.Sync(); err != nil {
		t.Fatal(err)
	}
	if err := index.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := openDiskLocationIndex(tablePath, keysPath, 32)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	if reopened.Count() != 1 {
		t.Fatalf("reopened count = %d", reopened.Count())
	}
	if got, ok, err := reopened.Get("alpha"); err != nil || !ok || got != updated {
		t.Fatalf("reopened Get(alpha) = %+v, %v, %v", got, ok, err)
	}
}

func TestDiskLocationIndexExactCollisionHandlingAndTombstoneReuse(t *testing.T) {
	index, _, _ := openTestDiskLocationIndex(t, 8)
	defer index.Close()
	index.hashID = func(string) uint64 { return 3 }
	entries := make([]diskLocationEntry, 0, 3)
	for n, key := range []string{"a", "b", "c"} {
		entries = append(entries, diskLocationEntry{Key: key, Value: diskLocationValue{SegmentGeneration: uint64(n + 1), RecordOffset: int64(n * 10), RecordLength: 10}})
	}
	if err := index.PutBatch(entries); err != nil {
		t.Fatal(err)
	}
	if deleted, err := index.Delete("b"); err != nil || !deleted {
		t.Fatalf("Delete(b) = %v, %v", deleted, err)
	}
	if err := index.Put("d", diskLocationValue{SegmentGeneration: 9, RecordOffset: 99, RecordLength: 11}); err != nil {
		t.Fatal(err)
	}
	for key, generation := range map[string]uint64{"a": 1, "c": 3, "d": 9} {
		got, ok, err := index.Get(key)
		if err != nil || !ok || got.SegmentGeneration != generation {
			t.Fatalf("Get(%s) = %+v, %v, %v", key, got, ok, err)
		}
	}
}

func TestDiskLocationIndexFullAndRange(t *testing.T) {
	index, _, _ := openTestDiskLocationIndex(t, 8)
	defer index.Close()
	for n := 0; n < 8; n++ {
		if err := index.Put(string(rune('a'+n)), diskLocationValue{SegmentGeneration: 1, RecordOffset: int64(n), RecordLength: 1}); err != nil {
			t.Fatal(err)
		}
	}
	if err := index.Put("overflow", diskLocationValue{SegmentGeneration: 1, RecordOffset: 9, RecordLength: 1}); !errors.Is(err, ErrLocationIndexFull) {
		t.Fatalf("full index error = %v", err)
	}
	seen := make(map[string]diskLocationValue)
	if err := index.Range(func(key string, value diskLocationValue) bool {
		seen[key] = value
		return true
	}); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 8 {
		t.Fatalf("Range returned %d entries", len(seen))
	}
}

func TestDiskLocationIndexFailsClosedOnSlotCorruption(t *testing.T) {
	index, tablePath, _ := openTestDiskLocationIndex(t, 16)
	if err := index.Put("corrupt", diskLocationValue{SegmentGeneration: 1, RecordOffset: 4, RecordLength: 8}); err != nil {
		t.Fatal(err)
	}
	slotIndex, found, err := index.findSlotLocked("corrupt", index.hashID("corrupt"), false)
	if err != nil || !found {
		t.Fatalf("find slot = %d, %v, %v", slotIndex, found, err)
	}
	if err := index.Close(); err != nil {
		t.Fatal(err)
	}
	file, err := os.OpenFile(tablePath, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	offset := int64(diskLocationHeaderSize) + int64(slotIndex)*diskLocationSlotSize + 40
	var one [1]byte
	if _, err := file.ReadAt(one[:], offset); err != nil {
		t.Fatal(err)
	}
	one[0] ^= 0xff
	if _, err := file.WriteAt(one[:], offset); err != nil {
		t.Fatal(err)
	}
	_ = file.Close()
	reopened, err := openDiskLocationIndex(tablePath, filepath.Join(filepath.Dir(tablePath), "locations.keys"), 16)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	if _, _, err := reopened.Get("corrupt"); err == nil {
		t.Fatal("corrupt slot was accepted")
	}
}

func TestDiskLocationIndexRejectsCorruptHeader(t *testing.T) {
	index, tablePath, keysPath := openTestDiskLocationIndex(t, 16)
	if err := index.Close(); err != nil {
		t.Fatal(err)
	}
	file, err := os.OpenFile(tablePath, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	var raw [4]byte
	binary.LittleEndian.PutUint32(raw[:], 99)
	if _, err := file.WriteAt(raw[:], 8); err != nil {
		t.Fatal(err)
	}
	_ = file.Close()
	if reopened, err := openDiskLocationIndex(tablePath, keysPath, 16); err == nil {
		_ = reopened.Close()
		t.Fatal("corrupt header was accepted")
	}
}

func BenchmarkLocationIndexBuild100k(b *testing.B) {
	const count = 100_000
	b.Run("disk", func(b *testing.B) {
		for iteration := 0; iteration < b.N; iteration++ {
			index, tablePath, keysPath := openTestDiskLocationIndex(b, 150_000)
			runtime.GC()
			var before runtime.MemStats
			runtime.ReadMemStats(&before)
			for start := 0; start < count; start += 1000 {
				entries := make([]diskLocationEntry, 0, 1000)
				for n := start; n < start+1000; n++ {
					entries = append(entries, diskLocationEntry{
						Key: fmt.Sprintf("vector-%07d", n),
						Value: diskLocationValue{
							SegmentGeneration: uint64(n/1000 + 1),
							RecordOffset:      int64(n * 600),
							RecordLength:      600,
						},
					})
				}
				if err := index.PutBatch(entries); err != nil {
					b.Fatal(err)
				}
			}
			runtime.GC()
			var after runtime.MemStats
			runtime.ReadMemStats(&after)
			tableInfo, _ := os.Stat(tablePath)
			keysInfo, _ := os.Stat(keysPath)
			b.ReportMetric(float64(after.HeapAlloc-before.HeapAlloc)/(1<<20), "live-heap-MiB")
			b.ReportMetric(float64(tableInfo.Size()+keysInfo.Size())/(1<<20), "index-disk-MiB")
			runtime.KeepAlive(index)
			_ = index.Close()
		}
	})
	b.Run("legacy-map", func(b *testing.B) {
		for iteration := 0; iteration < b.N; iteration++ {
			runtime.GC()
			var before runtime.MemStats
			runtime.ReadMemStats(&before)
			locations := make(map[string]segmentLocation, count)
			for n := 0; n < count; n++ {
				id := fmt.Sprintf("vector-%07d", n)
				locations[id] = segmentLocation{
					segment: fmt.Sprintf("segment-%020d.dat", n/1000+1),
					meta: fileVectorRecordMeta{
						recordOffset: int64(n * 600),
						recordLength: 600,
					},
				}
			}
			runtime.GC()
			var after runtime.MemStats
			runtime.ReadMemStats(&after)
			b.ReportMetric(float64(after.HeapAlloc-before.HeapAlloc)/(1<<20), "live-heap-MiB")
			runtime.KeepAlive(locations)
		}
	})
}

func BenchmarkLocationIndexLookup100k(b *testing.B) {
	const count = 100_000
	index, _, _ := openTestDiskLocationIndex(b, 150_000)
	entries := make([]diskLocationEntry, count)
	legacy := make(map[string]segmentLocation, count)
	for n := 0; n < count; n++ {
		id := fmt.Sprintf("vector-%07d", n)
		entries[n] = diskLocationEntry{Key: id, Value: diskLocationValue{SegmentGeneration: 1, RecordOffset: int64(n * 600), RecordLength: 600}}
		legacy[id] = segmentLocation{segment: "segment-00000000000000000001.dat", meta: fileVectorRecordMeta{recordOffset: int64(n * 600), recordLength: 600}}
	}
	if err := index.PutBatch(entries); err != nil {
		b.Fatal(err)
	}
	b.Run("disk", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			id := entries[n%count].Key
			if _, ok, err := index.Get(id); err != nil || !ok {
				b.Fatal(err)
			}
		}
	})
	b.Run("legacy-map", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			if _, ok := legacy[entries[n%count].Key]; !ok {
				b.Fatal("missing ID")
			}
		}
	})
	runtime.KeepAlive(legacy)
	_ = index.Close()
}
