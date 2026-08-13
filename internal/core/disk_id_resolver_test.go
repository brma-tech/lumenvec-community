package core

import (
	"errors"
	"fmt"
	"path/filepath"
	"runtime"
	"testing"
)

func TestDiskIDResolverAssignLookupRemoveAndReopen(t *testing.T) {
	dir := t.TempDir()
	resolver, err := openDiskIDResolver(dir, 32)
	if err != nil {
		t.Fatal(err)
	}
	defer resolver.Close()
	a, err := resolver.AssignID("a")
	if err != nil {
		t.Fatal(err)
	}
	b, err := resolver.AssignID("b")
	if err != nil {
		t.Fatal(err)
	}
	if again, err := resolver.AssignID("a"); err != nil || again != a {
		t.Fatalf("repeat assignment = %d, %v", again, err)
	}
	if got, ok, err := resolver.LookupID(b); err != nil || !ok || got != "b" {
		t.Fatalf("lookup b = %q, %v, %v", got, ok, err)
	}
	if err := resolver.RemoveID("a"); err != nil {
		t.Fatal(err)
	}
	if _, ok, err := resolver.LookupID(a); err != nil || ok {
		t.Fatalf("removed lookup = %v, %v", ok, err)
	}
	if err := resolver.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := openDiskIDResolver(dir, 32)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	if got, ok, err := reopened.LookupID(b); err != nil || !ok || got != "b" {
		t.Fatalf("reopened lookup b = %q, %v, %v", got, ok, err)
	}
	c, err := reopened.AssignID("c")
	if err != nil || c <= b {
		t.Fatalf("monotonic assignment after reopen = %d, %v", c, err)
	}
}

func TestDiskIDResolverPreservesHashCollisions(t *testing.T) {
	resolver, err := openDiskIDResolver(t.TempDir(), 16)
	if err != nil {
		t.Fatal(err)
	}
	defer resolver.Close()
	resolver.forward.hashID = func(string) uint64 { return 11 }
	for _, id := range []string{"alpha", "beta", "gamma"} {
		if _, err := resolver.AssignID(id); err != nil {
			t.Fatal(err)
		}
	}
	for internalID, want := range []string{"alpha", "beta", "gamma"} {
		got, ok, err := resolver.LookupID(internalID + 1)
		if err != nil || !ok || got != want {
			t.Fatalf("collision lookup %d = %q, %v, %v", internalID+1, got, ok, err)
		}
	}
}

func TestDiskIDResolverRestoreIsExactAndCapacityBounded(t *testing.T) {
	dir := t.TempDir()
	resolver, err := openDiskIDResolver(dir, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer resolver.Close()
	if err := resolver.restoreEntries([]idResolverEntry{{id: "sparse", internalID: 7}, {id: "first", internalID: 1}}); err != nil {
		t.Fatal(err)
	}
	if got, ok, err := resolver.LookupID(7); err != nil || !ok || got != "sparse" {
		t.Fatalf("sparse lookup = %q, %v, %v", got, ok, err)
	}
	keyInfo, err := resolver.forward.keys.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if err := resolver.restoreEntries([]idResolverEntry{{id: "sparse", internalID: 7}, {id: "first", internalID: 1}}); err != nil {
		t.Fatal(err)
	}
	reusedInfo, err := resolver.forward.keys.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if reusedInfo.Size() != keyInfo.Size() {
		t.Fatalf("matching restore rebuilt key log: before=%d after=%d", keyInfo.Size(), reusedInfo.Size())
	}
	if err := resolver.restoreEntries([]idResolverEntry{{id: "a", internalID: 1}, {id: "b", internalID: 1}}); err == nil {
		t.Fatal("duplicate internal ID was accepted")
	}
	tooMany := make([]idResolverEntry, 9)
	for i := range tooMany {
		tooMany[i] = idResolverEntry{id: string(rune('a' + i)), internalID: i + 1}
	}
	if err := resolver.restoreEntries(tooMany); !errors.Is(err, ErrLocationIndexFull) {
		t.Fatalf("capacity error = %v", err)
	}
}

func TestDiskLocationKeyReferenceRejectsOutOfBounds(t *testing.T) {
	idx, err := openDiskLocationIndex(
		filepath.Join(t.TempDir(), "table"),
		filepath.Join(t.TempDir(), "keys"),
		8,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	if _, err := idx.ReadKeyReference(diskKeyReference{Offset: 99, Length: 1}); err == nil {
		t.Fatal("out-of-bounds key reference was accepted")
	}
}

func BenchmarkDiskIDResolverLookup100K(b *testing.B) {
	const count = 100_000
	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)
	resolver, err := openDiskIDResolver(b.TempDir(), count*2)
	if err != nil {
		b.Fatal(err)
	}
	defer resolver.Close()
	for i := 0; i < count; i++ {
		if _, err := resolver.AssignID(fmt.Sprintf("vector-%07d", i)); err != nil {
			b.Fatal(err)
		}
	}
	runtime.GC()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	liveHeapMiB := float64(after.HeapAlloc-before.HeapAlloc) / (1 << 20)
	b.ReportAllocs()
	b.ResetTimer()
	b.ReportMetric(liveHeapMiB, "live-heap-MiB")
	for i := 0; i < b.N; i++ {
		if _, ok, err := resolver.LookupID(i%count + 1); err != nil || !ok {
			b.Fatal(err)
		}
	}
}
