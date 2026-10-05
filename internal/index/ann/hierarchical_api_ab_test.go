//go:build annapiaudit

package ann

import (
	"context"
	"math/rand"
	"runtime"
	"runtime/pprof"
	"sort"
	"testing"
	"time"
)

// Opt-in diagnostic: run alone, not alongside other tests or benchmarks.
// A reproduces the old measured API; B reproduces the corrected measured API.
// AB/BA ordering balances temporal drift without rebuilding or changing the graph.
func TestHierarchicalAPIPairedAB(t *testing.T) {
	const n, dim, nq, k, operations, pairs = 10000, 128, 100, 10, 200, 120
	rng := rand.New(rand.NewSource(42))
	data := make([][]float64, n)
	batch := make([]BatchVector32, n)
	for id := range batch {
		data[id] = make([]float64, dim)
		values := make([]float32, dim)
		for d := range values {
			data[id][d] = rng.Float64()
			values[d] = float32(data[id][d])
		}
		batch[id] = BatchVector32{ID: id, Values: values}
	}
	queries := make([][]float64, nq)
	truth := make([][]int, nq)
	for i := range queries {
		queries[i] = make([]float64, dim)
		for d := range queries[i] {
			queries[i][d] = rng.Float64()
		}
		truth[i] = exactTopKForTest(data, queries[i], k)
	}
	start := time.Now()
	h, err := BuildHierarchicalIndex(Options{M: 20, EfConstruction: 128, EfSearch: 256, Seed: 42}, batch)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("build_ms=%.3f gomaxprocs=%d go=%s", float64(time.Since(start))/float64(time.Millisecond), runtime.GOMAXPROCS(0), runtime.Version())
	matches := 0
	for i, q := range queries {
		a, err := h.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		b, err := h.SearchWithDistancesEfInto(q, k, 256, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(a) != len(b) {
			t.Fatal("result count differs")
		}
		for j, id := range a {
			if id != b[j].ID {
				t.Fatalf("query %d rank %d differs", i, j)
			}
			for _, expected := range truth[i] {
				if expected == id {
					matches++
					break
				}
			}
		}
	}
	t.Logf("identical_ranked_results=true recall_pct=%.2f connectivity=%+v", float64(matches)*100/(nq*k), h.Connectivity())
	search := func(mode int, q []float64) {
		if mode == 0 {
			if _, err := h.Search(q, k); err != nil {
				t.Fatal(err)
			}
		} else {
			if _, err := h.SearchWithDistancesEfInto(q, k, 256, nil); err != nil {
				t.Fatal(err)
			}
		}
	}
	for mode := 0; mode < 2; mode++ {
		t.Logf("mode=%d allocs_per_query=%.2f", mode, testing.AllocsPerRun(100, func() { search(mode, queries[0]) }))
	}
	var total [2]time.Duration
	for pair := 0; pair < pairs; pair++ {
		for step := 0; step < 2; step++ {
			mode := (pair + step) % 2
			label := "old_api"
			if mode == 1 {
				label = "corrected_api"
			}
			pprof.SetGoroutineLabels(pprof.WithLabels(context.Background(), pprof.Labels("api", label)))
			for _, q := range queries {
				search(mode, q)
			}
			latencies := make([]time.Duration, operations)
			var elapsed time.Duration
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			cpuStart := auditProcessCPU()
			wall := time.Now()
			for i := 0; i < operations; i++ {
				started := time.Now()
				search(mode, queries[i%nq])
				latencies[i] = time.Since(started)
				elapsed += latencies[i]
			}
			wallElapsed := time.Since(wall)
			windowEnd := time.Now().UTC()
			cpuEnd := auditProcessCPU()
			cpuPercent := -1.0
			if cpuStart >= 0 && cpuEnd >= cpuStart {
				cpuPercent = (cpuEnd - cpuStart) / wallElapsed.Seconds() * 100
			}
			runtime.ReadMemStats(&after)
			total[mode] += elapsed
			sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
			t.Logf("pair=%02d mode=%d qps=%.2f wall_qps=%.2f p95_ms=%.4f p99_ms=%.4f gc=%d pause_ms=%.4f", pair, mode, float64(operations)/elapsed.Seconds(), float64(operations)/wallElapsed.Seconds(), float64(latencies[operations*95/100-1])/1e6, float64(latencies[operations*99/100-1])/1e6, after.NumGC-before.NumGC, float64(after.PauseTotalNs-before.PauseTotalNs)/1e6)
			t.Logf("pair=%02d mode=%d process_cpu_pct_one_core=%.2f", pair, mode, cpuPercent)
			t.Logf("window pair=%02d mode=%d start=%s end=%s", pair, mode, wall.UTC().Format(time.RFC3339Nano), windowEnd.Format(time.RFC3339Nano))
		}
	}
	pprof.SetGoroutineLabels(context.Background())
	t.Logf("aggregate_A_qps=%.2f aggregate_B_qps=%.2f B_over_A=%.5f", float64(pairs*operations)/total[0].Seconds(), float64(pairs*operations)/total[1].Seconds(), total[0].Seconds()/total[1].Seconds())
}
