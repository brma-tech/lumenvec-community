package core

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"testing"
)

type failingErrorAwareResolver struct {
	*memoryIDResolver
	assignErr error
	lookupErr error
	removeErr error
}

func (r *failingErrorAwareResolver) AssignID(id string) (int, error) {
	if r.assignErr != nil {
		return 0, r.assignErr
	}
	return r.memoryIDResolver.AssignID(id)
}

func (r *failingErrorAwareResolver) LookupID(internalID int) (string, bool, error) {
	if r.lookupErr != nil {
		return "", false, r.lookupErr
	}
	return r.memoryIDResolver.LookupID(internalID)
}

func (r *failingErrorAwareResolver) RemoveID(id string) error {
	if r.removeErr != nil {
		return r.removeErr
	}
	return r.memoryIDResolver.RemoveID(id)
}

func TestMemoryIDResolverCrossesPagesAndReusesExistingID(t *testing.T) {
	resolver := newMemoryIDResolver()
	for n := 1; n <= reverseIDPageSize+2; n++ {
		id := fmt.Sprintf("vector-%d", n)
		if got := resolver.Assign(id); got != n {
			t.Fatalf("Assign(%q) = %d, want %d", id, got, n)
		}
	}
	if got := resolver.Assign("vector-1"); got != 1 {
		t.Fatalf("existing ID reassigned to %d", got)
	}
	for _, internalID := range []int{1, reverseIDPageSize - 1, reverseIDPageSize, reverseIDPageSize + 2} {
		want := fmt.Sprintf("vector-%d", internalID)
		if got, ok := resolver.Lookup(internalID); !ok || got != want {
			t.Fatalf("Lookup(%d) = %q, %v; want %q, true", internalID, got, ok, want)
		}
	}
}

func TestMemoryIDResolverMergeMappingsIsAtomicAndIdempotent(t *testing.T) {
	resolver := newMemoryIDResolver()
	if err := resolver.MergeIDMappings([]IDMappingEntry{{ID: "alpha", InternalID: 7}, {ID: "beta", InternalID: 9}}); err != nil {
		t.Fatal(err)
	}
	if got, ok := resolver.Lookup(7); !ok || got != "alpha" {
		t.Fatalf("mapping lookup=%q,%v", got, ok)
	}
	if err := resolver.MergeIDMappings([]IDMappingEntry{{ID: "alpha", InternalID: 7}}); err != nil {
		t.Fatalf("identical retry: %v", err)
	}
	if err := resolver.MergeIDMappings([]IDMappingEntry{{ID: "gamma", InternalID: 11}, {ID: "gamma", InternalID: 12}}); err == nil {
		t.Fatal("expected conflicting incoming mapping rejection")
	}
	if _, ok := resolver.Lookup(11); ok {
		t.Fatal("failed window mutated resolver")
	}
}

func TestMemoryIDResolverRemoveAndRestoreSparsePages(t *testing.T) {
	resolver := newMemoryIDResolver()
	entries := map[string]int{
		"first":  1,
		"sparse": reverseIDPageSize*3 + 7,
	}
	if err := resolver.Restore(entries); err != nil {
		t.Fatal(err)
	}
	if got, ok := resolver.Lookup(reverseIDPageSize*3 + 7); !ok || got != "sparse" {
		t.Fatalf("sparse lookup = %q, %v", got, ok)
	}
	resolver.Remove("sparse")
	if _, ok := resolver.Lookup(reverseIDPageSize*3 + 7); ok {
		t.Fatal("removed reverse ID is still visible")
	}
	if got := resolver.Assign("next"); got != reverseIDPageSize*3+8 {
		t.Fatalf("next ID = %d", got)
	}
	stats := resolver.MemoryStats()
	if stats.IDs != 2 || stats.ReversePages != 2 {
		t.Fatalf("unexpected memory stats: %+v", stats)
	}
}

func TestMemoryIDResolverRestoreRejectsInvalidOrDuplicateIDs(t *testing.T) {
	for name, entries := range map[string]map[string]int{
		"empty":     {"": 1},
		"zero":      {"zero": 0},
		"too-large": {"large": math.MaxInt32 + 1},
		"duplicate": {"a": 1, "b": 1},
	} {
		t.Run(name, func(t *testing.T) {
			if err := newMemoryIDResolver().Restore(entries); err == nil {
				t.Fatal("Restore unexpectedly succeeded")
			}
		})
	}
}

func TestMemoryIDResolverPreservesCollidingIDs(t *testing.T) {
	resolver := newMemoryIDResolver()
	resolver.hashID = func(string) uint64 { return 7 }
	a := resolver.Assign("a")
	b := resolver.Assign("b")
	c := resolver.Assign("c")
	if resolver.Assign("b") != b {
		t.Fatal("colliding ID was reassigned")
	}
	resolver.Remove("a")
	for id, internalID := range map[string]int{"b": b, "c": c} {
		if got := resolver.Assign(id); got != internalID {
			t.Fatalf("Assign(%q) = %d, want %d", id, got, internalID)
		}
	}
	resolver.Remove("c")
	if got := resolver.Assign("b"); got != b {
		t.Fatalf("remaining collision moved from %d to %d", b, got)
	}
	if got := resolver.Assign("d"); got == a || got == b || got == c {
		t.Fatalf("new colliding ID reused historical internal ID %d", got)
	}
}

func TestServicePropagatesPersistentResolverErrors(t *testing.T) {
	t.Run("assign", func(t *testing.T) {
		svc := newCoreService(t, "ann")
		want := errors.New("resolver assign I/O")
		svc.idResolver = &failingErrorAwareResolver{memoryIDResolver: newMemoryIDResolver(), assignErr: want}
		if err := svc.AddVector("a", []float64{1, 2}); !errors.Is(err, want) {
			t.Fatalf("AddVector error = %v", err)
		}
		if _, err := svc.GetVector("a"); err == nil {
			t.Fatal("failed resolver assignment left vector committed")
		}
	})

	t.Run("lookup", func(t *testing.T) {
		svc := newCoreService(t, "ann")
		resolver := &failingErrorAwareResolver{memoryIDResolver: newMemoryIDResolver()}
		svc.idResolver = resolver
		if err := svc.AddVector("a", []float64{1, 2}); err != nil {
			t.Fatal(err)
		}
		want := errors.New("resolver lookup I/O")
		resolver.lookupErr = want
		if _, err := svc.Search([]float64{1, 2}, 1); !errors.Is(err, want) {
			t.Fatalf("Search error = %v", err)
		}
	})

	t.Run("remove rollback", func(t *testing.T) {
		svc := newCoreService(t, "ann")
		resolver := &failingErrorAwareResolver{memoryIDResolver: newMemoryIDResolver()}
		svc.idResolver = resolver
		if err := svc.AddVector("a", []float64{1, 2}); err != nil {
			t.Fatal(err)
		}
		want := errors.New("resolver remove I/O")
		resolver.removeErr = want
		if err := svc.DeleteVector("a"); !errors.Is(err, want) {
			t.Fatalf("DeleteVector error = %v", err)
		}
		if _, err := svc.GetVector("a"); err != nil {
			t.Fatalf("failed resolver removal did not roll vector back: %v", err)
		}
	})
}

func BenchmarkMemoryIDResolverLookup1M(b *testing.B) {
	resolver := newMemoryIDResolver()
	for n := 0; n < 1_000_000; n++ {
		resolver.Assign(fmt.Sprintf("vector-%07d", n))
	}
	stats := resolver.MemoryStats()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		_, _ = resolver.Lookup(n%1_000_000 + 1)
	}
	b.ReportMetric(float64(stats.ReverseBytes)/(1<<20), "reverse-MiB")
	b.ReportMetric(float64(stats.ReverseBytes)/float64(stats.IDs), "reverse-B/ID")
}

func BenchmarkIDResolverBuild1M(b *testing.B) {
	const count = 1_000_000
	b.Run("paged-fingerprint", func(b *testing.B) {
		for iteration := 0; iteration < b.N; iteration++ {
			runtime.GC()
			var before runtime.MemStats
			runtime.ReadMemStats(&before)
			resolver := newMemoryIDResolver()
			for n := 0; n < count; n++ {
				resolver.Assign(fmt.Sprintf("vector-%07d", n))
			}
			runtime.GC()
			var after runtime.MemStats
			runtime.ReadMemStats(&after)
			b.ReportMetric(float64(after.HeapAlloc-before.HeapAlloc)/(1<<20), "live-heap-MiB")
			runtime.KeepAlive(resolver)
		}
	})
	b.Run("legacy-two-maps", func(b *testing.B) {
		for iteration := 0; iteration < b.N; iteration++ {
			runtime.GC()
			var before runtime.MemStats
			runtime.ReadMemStats(&before)
			direct := make(map[string]int, count)
			reverse := make(map[int]string, count)
			for n := 0; n < count; n++ {
				id := fmt.Sprintf("vector-%07d", n)
				direct[id] = n + 1
				reverse[n+1] = id
			}
			runtime.GC()
			var after runtime.MemStats
			runtime.ReadMemStats(&after)
			b.ReportMetric(float64(after.HeapAlloc-before.HeapAlloc)/(1<<20), "live-heap-MiB")
			runtime.KeepAlive(direct)
			runtime.KeepAlive(reverse)
		}
	})
}
