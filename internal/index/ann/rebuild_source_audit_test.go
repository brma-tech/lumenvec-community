//go:build rebuildaudit

package ann

import (
	"math/rand"
	"os"
	"runtime"
	"testing"
	"time"
)

// Run each mode in a fresh process. This isolates payload preparation from
// unrelated fixes in the preserved historical binary. It is not a service or
// durability benchmark: IDs are already resolved and ordered in both modes.
func TestRebuildSourceAudit(t *testing.T) {
	const n, dim = 10000, 128
	rng := rand.New(rand.NewSource(42))
	canonical := make([]BatchVector32, n)
	for i := range canonical {
		values := make([]float32, dim)
		for d := range values {
			values[d] = rng.Float32()
		}
		canonical[i] = BatchVector32{ID: i, Values: values}
	}
	opts := Options{M: 20, EfConstruction: 128, EfSearch: 256, Seed: 42}
	old, err := BuildHierarchicalIndex(opts, canonical)
	if err != nil {
		t.Fatal(err)
	}
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	start := time.Now()
	var candidate *HierarchicalIndex
	mode := os.Getenv("LUMENVEC_REBUILD_AUDIT_MODE")
	switch mode {
	case "snapshot":
		batch := make([]BatchVector32, n)
		for i, v := range canonical {
			batch[i] = BatchVector32{ID: v.ID, Values: append([]float32(nil), v.Values...)}
		}
		candidate, err = BuildHierarchicalIndex(opts, batch)
	case "stream":
		var scratch [dim]float32
		candidate, err = BuildHierarchicalIndexFromSource(opts, n, func(i int) (BatchVector32, error) {
			copy(scratch[:], canonical[i].Values)
			return BatchVector32{ID: i, Values: scratch[:]}, nil
		})
	default:
		t.Fatal("set LUMENVEC_REBUILD_AUDIT_MODE to snapshot or stream")
	}
	elapsed := time.Since(start)
	if err != nil {
		t.Fatal(err)
	}
	runtime.ReadMemStats(&after)
	t.Logf("mode=%s rebuild_ms=%.3f allocated_bytes=%d heap_after_bytes=%d nodes=%d", mode, float64(elapsed)/float64(time.Millisecond), after.TotalAlloc-before.TotalAlloc, after.HeapAlloc, candidate.Stats().Nodes)
	runtime.KeepAlive(canonical)
	runtime.KeepAlive(old)
	runtime.KeepAlive(candidate)
}
