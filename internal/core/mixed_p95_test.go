package core

import (
	"fmt"
	"path/filepath"
	"sort"
	"sync"
	"testing"
	"time"

	"lumenvec/internal/index"
)

// TestMixedIngestSearchP95SLO keeps the no-pause query guarantee explicit.
// The 250ms SLO is intentionally conservative for this in-process gate; the
// benchmark artifact remains the source for capacity/throughput comparisons.
func TestMixedIngestSearchP95SLO(t *testing.T) {
	base := t.TempDir()
	svc := NewService(ServiceOptions{MaxVectorDim: 64, MaxK: 10, SearchMode: "ann", SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal.log")})
	defer svc.Close()
	vector := make([]float64, 64)
	seed := make([]index.Vector, 64)
	for i := range seed {
		seed[i] = index.Vector{ID: fmt.Sprintf("seed-%d", i), Values: vector}
	}
	if err := svc.AddVectors(seed); err != nil {
		t.Fatal(err)
	}
	times := make([]time.Duration, 0, 100)
	for i := 0; i < 100; i++ {
		start := time.Now()
		var wg sync.WaitGroup
		var addErr error
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			addErr = svc.AddVector(fmt.Sprintf("mixed-%d", n), vector)
		}(i)
		if _, err := svc.SearchBatch([]BatchSearchQuery{{ID: "q", Values: vector, K: 10}}); err != nil {
			t.Fatal(err)
		}
		wg.Wait()
		if addErr != nil {
			t.Fatal(addErr)
		}
		times = append(times, time.Since(start))
	}
	sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
	p95 := times[94]
	if p95 > 250*time.Millisecond {
		t.Fatalf("mixed ingest/search p95=%s exceeds 250ms SLO", p95)
	}
	t.Logf("mixed ingest/search p95=%s", p95)
}
