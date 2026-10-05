package ann

import (
	"math"
	"math/rand"
	"sort"
	"testing"
	"time"
)

// BenchmarkHierarchicalBuildModes10K is the focused promotion gate for the
// private bulk builder. Run with -benchtime=500x so every case measures the
// same number of searches and produces a useful p95 sample.
func BenchmarkHierarchicalBuildModes10K(b *testing.B) {
	const vectorCount, dimensions, queryCount, topK = 10_000, 128, 100, 10
	rng := rand.New(rand.NewSource(42))
	data := make([][]float64, vectorCount)
	batch := make([]BatchVector32, vectorCount)
	for id := range data {
		data[id] = make([]float64, dimensions)
		values := make([]float32, dimensions)
		for dimension := range values {
			value := rng.Float64()
			data[id][dimension] = value
			values[dimension] = float32(value)
		}
		batch[id] = BatchVector32{ID: id, Values: values}
	}
	queries := make([][]float64, queryCount)
	truth := make([][]int, queryCount)
	for queryIndex := range queries {
		queries[queryIndex] = make([]float64, dimensions)
		for dimension := range queries[queryIndex] {
			queries[queryIndex][dimension] = rng.Float64()
		}
		truth[queryIndex] = exactTopKForTest(data, queries[queryIndex], topK)
	}
	opts := Options{M: 32, EfConstruction: 128, EfSearch: 256, Seed: 42}
	cases := []struct {
		name  string
		ef    int
		build func() (*HierarchicalIndex, error)
	}{
		{name: "bulk_m32_full_pruning", build: func() (*HierarchicalIndex, error) {
			candidate := opts
			candidate.DiversifiedExistingPruning = true
			return BuildHierarchicalIndex(candidate, batch)
		}},
		{name: "bulk_m28_efc256_full_pruning", ef: 736, build: func() (*HierarchicalIndex, error) {
			candidate := opts
			candidate.M = 28
			candidate.EfConstruction = 256
			candidate.DiversifiedExistingPruning = true
			candidate.HierarchicalBuildScratchBudgetBytes = 128 << 20
			return BuildHierarchicalIndex(candidate, batch)
		}},
		{name: "bulk_m20_full_pruning", build: func() (*HierarchicalIndex, error) {
			candidate := opts
			candidate.M = 20
			candidate.DiversifiedExistingPruning = true
			return BuildHierarchicalIndex(candidate, batch)
		}},
		{name: "bulk_m32_efc256", build: func() (*HierarchicalIndex, error) {
			candidate := opts
			candidate.EfConstruction = 256
			return BuildHierarchicalIndex(candidate, batch)
		}},
		{name: "bulk_m32_efc512", build: func() (*HierarchicalIndex, error) {
			candidate := opts
			candidate.EfConstruction = 512
			return BuildHierarchicalIndex(candidate, batch)
		}},
		{name: "bulk_m64_ef64", ef: 64, build: func() (*HierarchicalIndex, error) {
			candidate := opts
			candidate.M = 64
			candidate.EfSearch = 64
			return BuildHierarchicalIndex(candidate, batch)
		}},
		{name: "bulk_m64_ef128", ef: 128, build: func() (*HierarchicalIndex, error) {
			candidate := opts
			candidate.M = 64
			candidate.EfSearch = 128
			return BuildHierarchicalIndex(candidate, batch)
		}},
		{name: "bulk_m24_full_pruning", build: func() (*HierarchicalIndex, error) {
			candidate := opts
			candidate.M = 24
			candidate.DiversifiedExistingPruning = true
			return BuildHierarchicalIndex(candidate, batch)
		}},
		{name: "bulk_m16_full_pruning", build: func() (*HierarchicalIndex, error) {
			candidate := opts
			candidate.M = 16
			candidate.DiversifiedExistingPruning = true
			return BuildHierarchicalIndex(candidate, batch)
		}},
		{name: "bulk_m16_efc256_full_pruning", build: func() (*HierarchicalIndex, error) {
			candidate := opts
			candidate.M = 16
			candidate.EfConstruction = 256
			candidate.DiversifiedExistingPruning = true
			return BuildHierarchicalIndex(candidate, batch)
		}},
		{name: "bulk_m20", build: func() (*HierarchicalIndex, error) {
			candidate := opts
			candidate.M = 20
			return BuildHierarchicalIndex(candidate, batch)
		}},
		{name: "bulk_m24", build: func() (*HierarchicalIndex, error) {
			candidate := opts
			candidate.M = 24
			return BuildHierarchicalIndex(candidate, batch)
		}},
		{name: "bulk_m16_efc256", build: func() (*HierarchicalIndex, error) {
			m16 := opts
			m16.M = 16
			m16.EfConstruction = 256
			return BuildHierarchicalIndex(m16, batch)
		}},
		{name: "incremental_m16", build: func() (*HierarchicalIndex, error) {
			m16 := opts
			m16.M = 16
			index := NewHierarchicalIndexWithCapacity(m16, len(batch))
			return index, index.AddBatch32(batch)
		}},
		{name: "bulk_m16", build: func() (*HierarchicalIndex, error) {
			m16 := opts
			m16.M = 16
			return BuildHierarchicalIndex(m16, batch)
		}},
		{name: "incremental", build: func() (*HierarchicalIndex, error) {
			index := NewHierarchicalIndexWithCapacity(opts, len(batch))
			return index, index.AddBatch32(batch)
		}},
		{name: "bulk_m32", build: func() (*HierarchicalIndex, error) {
			return BuildHierarchicalIndex(opts, batch)
		}},
	}
	for _, benchmarkCase := range cases {
		b.Run(benchmarkCase.name, func(b *testing.B) {
			searchEf := benchmarkCase.ef
			if searchEf <= 0 {
				searchEf = opts.EfSearch
			}
			started := time.Now()
			index, err := benchmarkCase.build()
			if err != nil {
				b.Fatal(err)
			}
			buildDuration := time.Since(started)
			matches := 0
			for queryIndex, query := range queries {
				results, searchErr := index.SearchWithDistancesEfInto(query, topK, searchEf, nil)
				if searchErr != nil {
					b.Fatal(searchErr)
				}
				expected := make(map[int]struct{}, topK)
				for _, id := range truth[queryIndex] {
					expected[id] = struct{}{}
				}
				for _, result := range results {
					if _, ok := expected[result.ID]; ok {
						matches++
					}
				}
			}
			for _, query := range queries {
				if _, searchErr := index.SearchWithDistancesEfInto(query, topK, searchEf, nil); searchErr != nil {
					b.Fatal(searchErr)
				}
			}
			latencies := make([]float64, 0, b.N)
			var searchDuration time.Duration
			b.ReportAllocs()
			b.ResetTimer()
			for operation := 0; operation < b.N; operation++ {
				started = time.Now()
				if _, searchErr := index.SearchWithDistancesEfInto(queries[operation%len(queries)], topK, searchEf, nil); searchErr != nil {
					b.Fatal(searchErr)
				}
				elapsed := time.Since(started)
				searchDuration += elapsed
				latencies = append(latencies, float64(elapsed)/float64(time.Millisecond))
			}
			b.StopTimer()
			sort.Float64s(latencies)
			p95 := latencies[max(0, int(math.Ceil(float64(len(latencies))*0.95))-1)]
			connectivity := index.Connectivity()
			memory := index.MemoryStats()
			buildStats := index.BuildStats()
			b.ReportMetric(float64(buildDuration)/float64(time.Millisecond), "build_ms")
			b.ReportMetric(float64(matches)*100/float64(queryCount*topK), "recall_pct")
			b.ReportMetric(p95, "p95_ms")
			b.ReportMetric(float64(b.N)/searchDuration.Seconds(), "qps")
			b.ReportMetric(float64(memory.TotalBytes)/(1<<20), "ann_mib")
			b.ReportMetric(float64(buildStats.NeighborCacheUpperBoundBytes)/(1<<20), "build_scratch_bound_mib")
			b.ReportMetric(float64(connectivity.Unreachable), "unreachable")
			b.ReportMetric(float64(connectivity.ReverseEdges)*100/float64(connectivity.Edges), "reverse_edges_pct")
		})
	}
}

// BenchmarkHierarchicalDenseComparison100K compares the current hierarchy
// with the denser funnel using identical data and queries. Use -benchtime=1x
// (or higher) because index construction is intentionally included.
func BenchmarkHierarchicalDenseComparison100K(b *testing.B) {
	const vectorCount, dimensions, queryCount, topK = 100_000, 32, 100, 10
	rng := rand.New(rand.NewSource(42))
	batch := make([]BatchVector32, vectorCount)
	data := make([][]float64, vectorCount)
	for id := range batch {
		values := make([]float32, dimensions)
		data[id] = make([]float64, dimensions)
		for d := range values {
			values[d] = rng.Float32()
			data[id][d] = float64(values[d])
		}
		batch[id] = BatchVector32{ID: id, Values: values}
	}
	queries := make([][]float64, queryCount)
	truth := make([][]int, queryCount)
	for i := range queries {
		queries[i] = append([]float64(nil), data[i]...)
		truth[i] = exactTopKForTest(data, queries[i], topK)
	}
	base := Options{M: 20, EfConstruction: 128, EfSearch: 256, Seed: 42}
	for _, tc := range []struct {
		name string
		opts Options
	}{
		{"default", base},
		{"entrypoint_pool4", func() Options {
			o := base
			o.HierarchicalEntrypointPoolSize = 4
			return o
		}()},
		{"entrypoint_pool2", func() Options {
			o := base
			o.HierarchicalEntrypointPoolSize = 2
			return o
		}()},
		{"upper_beam2", func() Options {
			o := base
			o.HierarchicalUpperLayerEF = 2
			return o
		}()},
		{"dense", func() Options {
			o := base
			o.HierarchicalLevelMultiplier = 1 / math.Log(16)
			o.HierarchicalMaxLevel = 5
			return o
		}()},
	} {
		b.Run(tc.name, func(b *testing.B) {
			started := time.Now()
			index, err := BuildHierarchicalIndex(tc.opts, batch)
			if err != nil {
				b.Fatal(err)
			}
			buildDuration := time.Since(started)
			matches := 0
			for qi, q := range queries {
				results, err := index.SearchWithDistancesEfInto(q, topK, 256, nil)
				if err != nil {
					b.Fatal(err)
				}
				expected := make(map[int]struct{}, topK)
				for _, id := range truth[qi] {
					expected[id] = struct{}{}
				}
				for _, r := range results {
					if _, ok := expected[r.ID]; ok {
						matches++
					}
				}
			}
			latencies := make([]float64, 0, b.N)
			var total time.Duration
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				start := time.Now()
				if _, err := index.SearchWithDistancesEfInto(queries[i%queryCount], topK, 256, nil); err != nil {
					b.Fatal(err)
				}
				elapsed := time.Since(start)
				total += elapsed
				latencies = append(latencies, float64(elapsed)/float64(time.Millisecond))
			}
			b.StopTimer()
			sort.Float64s(latencies)
			p95 := latencies[max(0, int(math.Ceil(float64(len(latencies))*0.95))-1)]
			memory := index.MemoryStats()
			connectivity := index.Connectivity()
			b.ReportMetric(float64(buildDuration)/float64(time.Millisecond), "build_ms")
			b.ReportMetric(float64(matches)*100/float64(queryCount*topK), "recall_pct")
			b.ReportMetric(p95, "p95_ms")
			b.ReportMetric(float64(b.N)/total.Seconds(), "qps")
			b.ReportMetric(float64(memory.TotalBytes)/(1<<20), "ann_mib")
			b.ReportMetric(float64(connectivity.Unreachable), "unreachable")
		})
	}
}
