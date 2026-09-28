//go:build ann100k

package core

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"runtime/trace"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"
)

// TestServiceANN100KGate is opt-in because graph construction is intentionally
// excluded from the default test suite. It measures one physical ANN index;
// shard fanout is a separate experiment.
func TestServiceANN100KGate(t *testing.T) {
	const (
		vectorCount = 100_000
		warmupCount = 100
		queryCount  = 500
		dimensions  = 128
		topK        = 10
		concurrency = 4
	)
	m := ann100kPositiveEnv(t, "LUMENVEC_ANN_M", 32)
	efConstruction := ann100kPositiveEnv(t, "LUMENVEC_ANN_EF_CONSTRUCTION", 128)
	buildScratchMiB := ann100kPositiveEnv(t, "LUMENVEC_ANN_BUILD_SCRATCH_MIB", 128)
	searchRuns := ann100kPositiveEnv(t, "LUMENVEC_ANN_SEARCH_RUNS", 3)
	efSearchValues := ann100kPositiveListEnv(t, "LUMENVEC_ANN_EF_SEARCH_VALUES", []int{256, 512, 640, 672, 688, 704, 720, 736, 752, 768, 896, 1000, 1024, 1536, 2048})
	// Match benchmarks/runner exactly: vectors use seed 42, while the 100
	// warmups and 500 measured queries come from an independent seed 43.
	// Keeping this workload identical makes the pgvector, Weaviate and
	// in-process LumenVec recall/QPS results directly comparable.
	data, _ := serviceANNBenchData(vectorCount, 0, dimensions, 42)
	allQueries, _ := serviceANNBenchData(warmupCount+queryCount, 0, dimensions, 43)
	warmups, queries := allQueries[:warmupCount], allQueries[warmupCount:]
	truth := serviceANNBenchTruth(data, queries, topK)
	service := NewServiceWithDeps(ServiceOptions{
		MaxVectorDim: dimensions,
		MaxK:         topK,
		SearchMode:   "ann",
		ANNBackend:   "hierarchical-hnsw",
		ANNOptions: ann.Options{
			M: m, EfConstruction: efConstruction, EfSearch: 256, Seed: 42, DiversifiedExistingPruning: true,
			HierarchicalBuildScratchBudgetBytes: uint64(buildScratchMiB) << 20,
		},
		SnapshotEvery: 1 << 30,
	}, ServiceDeps{Persistence: &noopPersistence{}})
	t.Cleanup(func() { _ = service.Close() })

	started := time.Now()
	for offset := 0; offset < len(data); offset += 1000 {
		end := min(offset+1000, len(data))
		batch := make([]index.Vector, end-offset)
		for position, values := range data[offset:end] {
			batch[position] = index.Vector{ID: fmt.Sprintf("vec-%09d", offset+position+1), Values: values}
		}
		// This gate intentionally mirrors pgvector's bulk-load methodology:
		// persist the canonical rows first, then build one private ANN
		// generation. Public AddVectors keeps its immediately-searchable
		// incremental semantics and is measured by the mixed-workload suite.
		for _, vector := range batch {
			if err := service.index.AddVector(vector); err != nil {
				t.Fatal(err)
			}
		}
		if err := service.upsertVectors(batch); err != nil {
			t.Fatal(err)
		}
	}
	ingestDuration := time.Since(started)
	rebuildStarted := time.Now()
	service.rebuildANNLocked()
	rebuildDuration := time.Since(rebuildStarted)

	memory := service.currentANNIndex().(interface{ MemoryStats() ann.MemoryStats }).MemoryStats()
	connectivity := service.currentANNIndex().(interface{ Connectivity() ann.ConnectivityStats }).Connectivity()
	hierarchical := service.currentANNIndex().(*ann.HierarchicalIndex)
	buildStats := hierarchical.BuildStats()
	traceCleanup := ann100kMaybeTrace(t)
	defer traceCleanup()
	passedRecall := false
	for _, efSearch := range efSearchValues {
		hierarchical.SetEfSearch(efSearch)
		for _, query := range warmups {
			if _, err := service.Search(query, topK); err != nil {
				t.Fatal(err)
			}
		}
		var results [][]SearchResult
		qpsSamples := make([]float64, 0, searchRuns)
		p95Samples := make([]float64, 0, searchRuns)
		for run := 0; run < searchRuns; run++ {
			currentResults, qps, p95 := serviceANNMeasureConcurrent(t, service, queries, topK, concurrency)
			if run == 0 {
				results = currentResults
			}
			qpsSamples = append(qpsSamples, qps)
			p95Samples = append(p95Samples, p95)
		}
		sort.Float64s(qpsSamples)
		sort.Float64s(p95Samples)
		qps, p95 := qpsSamples[len(qpsSamples)/2], p95Samples[len(p95Samples)/2]
		var memStats runtime.MemStats
		runtime.ReadMemStats(&memStats)
		matches := 0
		for queryIndex, queryResults := range results {
			expected := make(map[string]struct{}, topK)
			for _, id := range truth[queryIndex] {
				expected[fmt.Sprintf("vec-%09d", id+1)] = struct{}{}
			}
			for _, result := range queryResults {
				if _, ok := expected[result.ID]; ok {
					matches++
				}
			}
		}
		recall := float64(matches) / float64(queryCount*topK)
		t.Logf("100k one-index gate: m=%d ef_construction=%d ef=%d ingest=%.3fs rebuild=%.3fs qps_median=%.2f qps_range=[%.2f,%.2f] p95_median=%.3fms p95_range=[%.3f,%.3f] recall@10=%.4f ann_mib=%.2f build_scratch_bound_mib=%.2f build_scratch_budget_mib=%.2f reachable=%d unreachable=%d weak_components=%d edges=%d reverse_edges=%d avg_degree=%.2f gomaxprocs=%d numcpu=%d goroutines=%d numgc=%d heap_alloc_mib=%.2f gc_pause_ms_total=%.3f", m, efConstruction, efSearch, ingestDuration.Seconds(), rebuildDuration.Seconds(), qps, qpsSamples[0], qpsSamples[len(qpsSamples)-1], p95, p95Samples[0], p95Samples[len(p95Samples)-1], recall, float64(memory.TotalBytes)/(1<<20), float64(buildStats.NeighborCacheUpperBoundBytes)/(1<<20), float64(buildStats.ScratchBudgetBytes)/(1<<20), connectivity.Reachable, connectivity.Unreachable, connectivity.WeakComponents, connectivity.Edges, connectivity.ReverseEdges, connectivity.AverageDegreeZero, runtime.GOMAXPROCS(0), runtime.NumCPU(), runtime.NumGoroutine(), memStats.NumGC, float64(memStats.HeapAlloc)/(1<<20), float64(memStats.PauseTotalNs)/float64(time.Millisecond))
		if recall >= 0.98 {
			passedRecall = true
		}
	}
	if !passedRecall {
		t.Fatal("no efSearch value reached Recall@10 >= 0.98")
	}
}

func ann100kMaybeTrace(t *testing.T) func() {
	t.Helper()
	path := strings.TrimSpace(os.Getenv("LUMENVEC_ANN_TRACE_PATH"))
	if path == "" {
		return func() {}
	}
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("create runtime trace %q: %v", path, err)
	}
	if err := trace.Start(file); err != nil {
		_ = file.Close()
		t.Fatalf("start runtime trace: %v", err)
	}
	return func() {
		trace.Stop()
		if err := file.Close(); err != nil {
			t.Errorf("close runtime trace: %v", err)
		}
	}
}

func ann100kPositiveListEnv(t *testing.T, key string, fallback []int) []int {
	t.Helper()
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	parts := strings.Split(raw, ",")
	values := make([]int, 0, len(parts))
	for _, part := range parts {
		value, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || value <= 0 {
			t.Fatalf("%s must be a comma-separated list of positive integers, got %q", key, raw)
		}
		values = append(values, value)
	}
	return values
}

func ann100kPositiveEnv(t *testing.T, key string, fallback int) int {
	t.Helper()
	raw := os.Getenv(key)
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value <= 0 {
		t.Fatalf("%s must be a positive integer, got %q", key, raw)
	}
	return value
}

func serviceANNMeasureConcurrent(t *testing.T, service *Service, queries [][]float64, k, concurrency int) ([][]SearchResult, float64, float64) {
	t.Helper()
	workerCount := min(concurrency, len(queries))
	if workerCount <= 0 {
		return nil, 0, 0
	}
	// Partition deterministically instead of dispatching each query through
	// jobs/out channels. The old harness included queueing and channel
	// scheduling in wall-clock throughput and made the runtime trace mostly a
	// profile of the measurement machinery rather than the ANN search.
	results := make([][]SearchResult, len(queries))
	latencies := make([]float64, len(queries))
	var workers sync.WaitGroup
	var errorMu sync.Mutex
	var firstError error
	for worker := 0; worker < workerCount; worker++ {
		workers.Add(1)
		go func(worker int) {
			defer workers.Done()
			for queryIndex := worker; queryIndex < len(queries); queryIndex += workerCount {
				started := time.Now()
				currentResults, err := service.Search(queries[queryIndex], k)
				if err != nil {
					errorMu.Lock()
					if firstError == nil {
						firstError = err
					}
					errorMu.Unlock()
					continue
				}
				results[queryIndex] = currentResults
				latencies[queryIndex] = float64(time.Since(started)) / float64(time.Millisecond)
			}
		}(worker)
	}

	started := time.Now()
	workers.Wait()
	if firstError != nil {
		t.Fatal(firstError)
	}
	duration := time.Since(started)
	sort.Float64s(latencies)
	p95 := latencies[max(0, int(math.Ceil(float64(len(latencies))*0.95))-1)]
	return results, float64(len(queries)) / duration.Seconds(), p95
}
