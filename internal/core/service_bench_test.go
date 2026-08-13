package core

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"testing"

	"lumenvec/internal/index"
)

func BenchmarkServiceAddVector(b *testing.B) {
	for _, batchSize := range []int{1, 32} {
		b.Run(fmt.Sprintf("batch_%d", batchSize), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				svc := benchmarkService(b, "exact")
				if batchSize == 1 {
					if err := svc.AddVector("vec-0", benchmarkVector(256, 1)); err != nil {
						b.Fatal(err)
					}
					continue
				}

				vectors := make([]index.Vector, 0, batchSize)
				for j := 0; j < batchSize; j++ {
					vectors = append(vectors, index.Vector{
						ID:     fmt.Sprintf("vec-%d", j),
						Values: benchmarkVector(256, float64(j)),
					})
				}
				if err := svc.AddVectors(vectors); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkServiceMixedIngestSearch measures query progress while batches are
// being appended, the critical storage acceptance criterion for Fase 4.
func BenchmarkServiceMixedIngestSearch(b *testing.B) {
	svc := benchmarkService(b, "ann")
	seedCount := 256
	if raw := os.Getenv("LUMENVEC_MIXED_SEED"); raw != "" { if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 { seedCount = parsed } }
	seed := make([]index.Vector, seedCount)
	for i := range seed {
		seed[i] = index.Vector{ID: fmt.Sprintf("seed-%d", i), Values: benchmarkVector(64, float64(i%17))}
	}
	if err := svc.AddVectors(seed); err != nil {
		b.Fatal(err)
	}
	query := benchmarkVector(64, 3)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var ingestErr error
		var wg sync.WaitGroup
		wg.Add(1)
		go func(round int) {
			defer wg.Done()
			batch := []index.Vector{{ID: fmt.Sprintf("mixed-%d", round), Values: benchmarkVector(64, float64(round%17))}}
			ingestErr = svc.AddVectors(batch)
		}(i)
		if _, err := svc.SearchBatch([]BatchSearchQuery{{ID: "q", Values: query, K: 10}}); err != nil {
			b.Fatal(err)
		}
		wg.Wait()
		if ingestErr != nil {
			b.Fatal(ingestErr)
		}
	}
}

func BenchmarkServiceAddVectorsIngest(b *testing.B) {
	for _, mode := range []string{"exact", "ann"} {
		for _, batchSize := range []int{100, 500, 1000} {
			b.Run(fmt.Sprintf("%s_batch_%d", mode, batchSize), func(b *testing.B) {
				svc := benchmarkService(b, mode)
				batches := make([][]index.Vector, b.N)
				for i := 0; i < b.N; i++ {
					vectors := make([]index.Vector, 0, batchSize)
					base := i * batchSize
					for j := 0; j < batchSize; j++ {
						n := base + j
						vectors = append(vectors, index.Vector{
							ID:     fmt.Sprintf("vec-%d", n),
							Values: benchmarkVector(256, float64(n%31)),
						})
					}
					batches[i] = vectors
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if err := svc.AddVectors(batches[i]); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func BenchmarkServiceAddVectorSyncEvery(b *testing.B) {
	for _, syncEvery := range []int{1, 64} {
		b.Run(fmt.Sprintf("sync_every_%d", syncEvery), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				svc := benchmarkServiceWithOptions(b, ServiceOptions{
					MaxVectorDim:  1024,
					MaxK:          64,
					SnapshotEvery: 1 << 30,
					SearchMode:    "exact",
					VectorStore:   "disk",
					SyncEvery:     syncEvery,
				})
				for j := 0; j < 64; j++ {
					if err := svc.AddVector(fmt.Sprintf("vec-%d", j), benchmarkVector(256, float64(j))); err != nil {
						b.Fatal(err)
					}
				}
				if err := svc.Close(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkServiceSearch(b *testing.B) {
	for _, mode := range []string{"exact", "ann"} {
		b.Run(mode, func(b *testing.B) {
			svc := benchmarkService(b, mode)
			vectors := make([]index.Vector, 0, 512)
			for i := 0; i < 512; i++ {
				vectors = append(vectors, index.Vector{
					ID:     fmt.Sprintf("vec-%d", i),
					Values: benchmarkVector(256, float64(i%13)),
				})
			}
			if err := svc.AddVectors(vectors); err != nil {
				b.Fatal(err)
			}

			query := benchmarkVector(256, 3)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := svc.Search(query, 10); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkServiceGetVector(b *testing.B) {
	svc := benchmarkService(b, "exact")
	if err := svc.AddVector("vec-0", benchmarkVector(256, 1)); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := svc.GetVector("vec-0"); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkServiceGetVectorByStore(b *testing.B) {
	for _, tc := range []struct {
		name  string
		store string
		cache CacheOptions
	}{
		{name: "memory", store: "memory"},
		{name: "disk", store: "disk"},
		{
			name:  "disk_cache",
			store: "disk",
			cache: CacheOptions{Enabled: true, MaxBytes: 8 << 20, MaxItems: 1024},
		},
	} {
		b.Run(tc.name, func(b *testing.B) {
			svc := benchmarkServiceWithStore(b, "exact", tc.store, tc.cache)
			if err := svc.AddVector("vec-0", benchmarkVector(256, 1)); err != nil {
				b.Fatal(err)
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := svc.GetVector("vec-0"); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkServiceListVectorsPage(b *testing.B) {
	for _, tc := range []struct {
		name  string
		store string
	}{
		{name: "memory", store: "memory"},
		{name: "disk", store: "disk"},
	} {
		b.Run(tc.name, func(b *testing.B) {
			svc := benchmarkServiceWithStore(b, "exact", tc.store, CacheOptions{})
			vectors := make([]index.Vector, 0, 8192)
			for i := 0; i < 8192; i++ {
				vectors = append(vectors, index.Vector{
					ID:     fmt.Sprintf("vec-%08d", i),
					Values: benchmarkVector(8, float64(i%13)),
				})
			}
			if err := svc.AddVectors(vectors); err != nil {
				b.Fatal(err)
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				page := svc.ListVectorsPage(ListVectorsOptions{
					AfterID: "vec-00004000",
					Limit:   100,
					IDsOnly: true,
				})
				if len(page.Vectors) != 100 {
					b.Fatalf("got %d vectors, want 100", len(page.Vectors))
				}
			}
		})
	}
}

func BenchmarkServiceSearchByStore(b *testing.B) {
	for _, tc := range []struct {
		name  string
		mode  string
		store string
		cache CacheOptions
	}{
		{name: "exact_memory", mode: "exact", store: "memory"},
		{name: "exact_disk", mode: "exact", store: "disk"},
		{name: "ann_disk", mode: "ann", store: "disk"},
		{
			name:  "ann_disk_cache",
			mode:  "ann",
			store: "disk",
			cache: CacheOptions{Enabled: true, MaxBytes: 8 << 20, MaxItems: 1024},
		},
	} {
		b.Run(tc.name, func(b *testing.B) {
			svc := benchmarkServiceWithStore(b, tc.mode, tc.store, tc.cache)
			vectors := make([]index.Vector, 0, 512)
			for i := 0; i < 512; i++ {
				vectors = append(vectors, index.Vector{
					ID:     fmt.Sprintf("vec-%d", i),
					Values: benchmarkVector(256, float64(i%13)),
				})
			}
			if err := svc.AddVectors(vectors); err != nil {
				b.Fatal(err)
			}

			query := benchmarkVector(256, 3)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := svc.Search(query, 10); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkServiceSearchBatch(b *testing.B) {
	svc := benchmarkService(b, "exact")
	vectors := make([]index.Vector, 0, 512)
	for i := 0; i < 512; i++ {
		vectors = append(vectors, index.Vector{
			ID:     fmt.Sprintf("vec-%d", i),
			Values: benchmarkVector(256, float64(i%17)),
		})
	}
	if err := svc.AddVectors(vectors); err != nil {
		b.Fatal(err)
	}

	singleQuery := benchmarkVector(256, 5)
	batchQueries := make([]BatchSearchQuery, 0, 16)
	for i := 0; i < 16; i++ {
		batchQueries = append(batchQueries, BatchSearchQuery{
			ID:     fmt.Sprintf("q-%d", i),
			Values: benchmarkVector(256, float64(i)),
			K:      10,
		})
	}

	b.Run("single_x16", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			for j := 0; j < 16; j++ {
				if _, err := svc.Search(singleQuery, 10); err != nil {
					b.Fatal(err)
				}
			}
		}
	})

	b.Run("batch_16", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := svc.SearchBatch(batchQueries); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkServiceSearchBatchANN(b *testing.B) {
	svc := benchmarkService(b, "ann")
	vectors := make([]index.Vector, 0, 2048)
	for i := 0; i < 2048; i++ {
		vectors = append(vectors, index.Vector{
			ID:     fmt.Sprintf("vec-%d", i),
			Values: benchmarkVector(256, float64(i%31)),
		})
	}
	if err := svc.AddVectors(vectors); err != nil {
		b.Fatal(err)
	}

	singleQuery := benchmarkVector(256, 5)
	batchQueries := make([]BatchSearchQuery, 0, 16)
	for i := 0; i < 16; i++ {
		batchQueries = append(batchQueries, BatchSearchQuery{
			ID:     fmt.Sprintf("q-%d", i),
			Values: benchmarkVector(256, float64(i%13)),
			K:      10,
		})
	}

	b.Run("single_x16", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			for j := 0; j < 16; j++ {
				if _, err := svc.Search(singleQuery, 10); err != nil {
					b.Fatal(err)
				}
			}
		}
	})

	b.Run("batch_16", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := svc.SearchBatch(batchQueries); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkServiceSearchBatchScales(b *testing.B) {
	for _, queryCount := range []int{4, 16, 64} {
		b.Run(fmt.Sprintf("queries_%d", queryCount), func(b *testing.B) {
			svc := benchmarkService(b, "exact")
			vectors := make([]index.Vector, 0, 1024)
			for i := 0; i < 1024; i++ {
				vectors = append(vectors, index.Vector{
					ID:     fmt.Sprintf("vec-%d", i),
					Values: benchmarkVector(256, float64(i%29)),
				})
			}
			if err := svc.AddVectors(vectors); err != nil {
				b.Fatal(err)
			}

			queries := make([]BatchSearchQuery, 0, queryCount)
			for i := 0; i < queryCount; i++ {
				queries = append(queries, BatchSearchQuery{
					ID:     fmt.Sprintf("q-%d", i),
					Values: benchmarkVector(256, float64(i%11)),
					K:      10,
				})
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := svc.SearchBatch(queries); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func benchmarkService(tb testing.TB, mode string) *Service {
	return benchmarkServiceWithStore(tb, mode, "memory", CacheOptions{})
}

func benchmarkServiceWithStore(tb testing.TB, mode, store string, cache CacheOptions) *Service {
	tb.Helper()
	svc := benchmarkServiceWithOptions(tb, ServiceOptions{
		MaxVectorDim:  1024,
		MaxK:          64,
		SnapshotEvery: 1 << 30,
		SearchMode:    mode,
		VectorStore:   store,
		Cache:         cache,
	})
	return svc
}

func benchmarkServiceWithOptions(tb testing.TB, opts ServiceOptions) *Service {
	tb.Helper()
	base := tb.TempDir()
	if opts.MaxVectorDim <= 0 {
		opts.MaxVectorDim = 1024
	}
	if opts.MaxK <= 0 {
		opts.MaxK = 64
	}
	if opts.SnapshotPath == "" {
		opts.SnapshotPath = filepath.Join(base, "snapshot.json")
	}
	if opts.WALPath == "" {
		opts.WALPath = filepath.Join(base, "wal.log")
	}
	if opts.VectorPath == "" {
		opts.VectorPath = filepath.Join(base, "vectors")
	}
	svc := NewService(opts)
	tb.Cleanup(func() { _ = svc.Close() })
	return svc
}

func benchmarkVector(dim int, seed float64) []float64 {
	values := make([]float64, dim)
	for i := range values {
		values[i] = seed + float64(i%7)*0.125
	}
	return values
}
