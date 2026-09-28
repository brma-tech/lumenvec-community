package core

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"
	"time"

	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"
)

type serviceANNBenchCandidate struct {
	id       int
	distance float64
}

func BenchmarkServiceANNBackends10K(b *testing.B) {
	const (
		vectorCount = 10_000
		dimensions  = 128
		queryCount  = 100
		topK        = 10
	)
	data, queries := serviceANNBenchData(vectorCount, queryCount, dimensions, 42)
	truth := serviceANNBenchTruth(data, queries, topK)
	cases := []struct {
		name    string
		backend string
		ef      int
	}{
		{name: "compatibility_nsw_ef768", backend: "hnsw", ef: 768},
		{name: "hierarchical_hnsw_ef256", backend: "hierarchical-hnsw", ef: 256},
	}
	for _, benchmarkCase := range cases {
		b.Run(benchmarkCase.name, func(b *testing.B) {
			service := NewServiceWithDeps(ServiceOptions{
				MaxVectorDim: dimensions,
				MaxK:         topK,
				SearchMode:   "ann",
				ANNBackend:   benchmarkCase.backend,
				ANNOptions: ann.Options{
					M: 32, EfConstruction: 128, EfSearch: benchmarkCase.ef, Seed: 42,
				},
				SnapshotEvery: 1 << 30,
			}, ServiceDeps{Persistence: &noopPersistence{}})
			b.Cleanup(func() { _ = service.Close() })
			started := time.Now()
			for offset := 0; offset < len(data); offset += 1000 {
				end := min(offset+1000, len(data))
				batch := make([]index.Vector, end-offset)
				for position, values := range data[offset:end] {
					batch[position] = index.Vector{ID: fmt.Sprintf("v-%d", offset+position), Values: values}
				}
				if err := service.AddVectors(batch); err != nil {
					b.Fatal(err)
				}
			}
			buildDuration := time.Since(started)
			recall := serviceANNBenchRecall(b, service, queries, truth, topK)
			for _, query := range queries {
				if _, err := service.Search(query, topK); err != nil {
					b.Fatal(err)
				}
			}

			latencies := make([]float64, 0, b.N)
			var searchDuration time.Duration
			b.ReportAllocs()
			b.ResetTimer()
			for operation := 0; operation < b.N; operation++ {
				started = time.Now()
				if _, err := service.Search(queries[operation%len(queries)], topK); err != nil {
					b.Fatal(err)
				}
				elapsed := time.Since(started)
				searchDuration += elapsed
				latencies = append(latencies, float64(elapsed)/float64(time.Millisecond))
			}
			b.StopTimer()
			sort.Float64s(latencies)
			p95 := latencies[max(0, int(math.Ceil(float64(len(latencies))*0.95))-1)]
			b.ReportMetric(recall*100, "recall_pct")
			b.ReportMetric(p95, "p95_ms")
			b.ReportMetric(float64(buildDuration)/float64(time.Millisecond), "build_ms")
			if searchDuration > 0 {
				b.ReportMetric(float64(b.N)/searchDuration.Seconds(), "qps")
			}
			if memoryIndex, ok := service.currentANNIndex().(interface{ MemoryStats() ann.MemoryStats }); ok {
				b.ReportMetric(float64(memoryIndex.MemoryStats().TotalBytes)/(1<<20), "ann_mib")
			}
		})
	}
}

func serviceANNBenchData(vectorCount, queryCount, dimensions int, seed int64) ([][]float64, [][]float64) {
	rng := rand.New(rand.NewSource(seed)) // #nosec G404 -- deterministic benchmark data
	makeRows := func(count int) [][]float64 {
		rows := make([][]float64, count)
		for row := range rows {
			rows[row] = make([]float64, dimensions)
			for dimension := range rows[row] {
				rows[row][dimension] = rng.Float64()
			}
		}
		return rows
	}
	return makeRows(vectorCount), makeRows(queryCount)
}

func serviceANNBenchTruth(data, queries [][]float64, k int) [][]int {
	truth := make([][]int, len(queries))
	for queryIndex, query := range queries {
		candidates := make([]serviceANNBenchCandidate, len(data))
		for id, values := range data {
			var distance float64
			for dimension, value := range values {
				delta := value - query[dimension]
				distance += delta * delta
			}
			candidates[id] = serviceANNBenchCandidate{id: id, distance: distance}
		}
		sort.Slice(candidates, func(left, right int) bool { return candidates[left].distance < candidates[right].distance })
		truth[queryIndex] = make([]int, k)
		for rank := range truth[queryIndex] {
			truth[queryIndex][rank] = candidates[rank].id
		}
	}
	return truth
}

func serviceANNBenchRecall(b *testing.B, service *Service, queries [][]float64, truth [][]int, k int) float64 {
	b.Helper()
	matches := 0
	for queryIndex, query := range queries {
		results, err := service.Search(query, k)
		if err != nil {
			b.Fatal(err)
		}
		expected := make(map[string]struct{}, k)
		for _, id := range truth[queryIndex] {
			expected[fmt.Sprintf("v-%d", id)] = struct{}{}
		}
		for _, result := range results {
			if _, ok := expected[result.ID]; ok {
				matches++
			}
		}
	}
	return float64(matches) / float64(len(queries)*k)
}
