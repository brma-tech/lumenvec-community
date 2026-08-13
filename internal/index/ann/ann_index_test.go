package ann

import (
	"fmt"
	"slices"
	"testing"

	"lumenvec/internal/vector"
)

func TestAnnIndexBasicSearch(t *testing.T) {
	idx := NewAnnIndexWithOptions(Options{
		M:              8,
		EfConstruction: 32,
		EfSearch:       32,
		Seed:           7,
	})

	_ = idx.AddVector(1, []float64{0, 0})
	_ = idx.AddVector(2, []float64{1, 1})
	_ = idx.AddVector(3, []float64{10, 10})

	got, err := idx.Search([]float64{0.1, 0.1}, 2)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 neighbors, got %d", len(got))
	}
	if got[0] != 1 && got[0] != 2 {
		t.Fatalf("unexpected nearest id: %d", got[0])
	}

	withDistances, err := idx.SearchWithDistances([]float64{0.1, 0.1}, 2)
	if err != nil {
		t.Fatalf("search with distances failed: %v", err)
	}
	if len(withDistances) != 2 {
		t.Fatalf("expected 2 neighbors with distances, got %d", len(withDistances))
	}
	if withDistances[0].Distance > withDistances[1].Distance {
		t.Fatalf("expected sorted distances, got %+v", withDistances)
	}
	if withDistances[0].ID != got[0] {
		t.Fatalf("SearchWithDistances first id = %d, want %d", withDistances[0].ID, got[0])
	}
}

func TestAnnIndexCompactAdjacencyPreservesSearchAndSnapshot(t *testing.T) {
	idx := NewAnnIndexWithOptions(Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 11})
	for i := 0; i < 64; i++ {
		if err := idx.AddVector(i, []float64{float64(i), float64(i % 7)}); err != nil {
			t.Fatal(err)
		}
	}
	before, err := idx.Search([]float64{31, 3}, 5)
	if err != nil {
		t.Fatal(err)
	}
	idx.CompactAdjacency()
	after, err := idx.Search([]float64{31, 3}, 5)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(before, after) {
		t.Fatalf("compact search changed results: before=%v after=%v", before, after)
	}
	data, err := idx.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	reopened, err := UnmarshalBinary(data)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := reopened.Search([]float64{31, 3}, 5)
	if err != nil || !slices.Equal(before, restored) {
		t.Fatalf("snapshot search changed results: got=%v err=%v want=%v", restored, err, before)
	}
}

func TestAnnIndexMemoryStatsTracksStructuralBytes(t *testing.T) {
	idx := NewAnnIndexWithOptions(Options{M: 4})
	for i := 0; i < 4; i++ {
		if err := idx.AddVector(i, []float64{float64(i), 1}); err != nil {
			t.Fatal(err)
		}
	}
	stats := idx.MemoryStats()
	if stats.VectorBytes != 4*2*4 || stats.NodeBytes == 0 || stats.MapBytes == 0 ||
		stats.TotalBytes != stats.VectorBytes+stats.NodeBytes+stats.AdjacencyBytes+stats.MapBytes {
		t.Fatalf("unexpected memory stats: %+v", stats)
	}
}

func TestAnnIndexDeleteVectorUsesTombstone(t *testing.T) {
	idx := NewAnnIndexWithOptions(Options{
		M:              8,
		EfConstruction: 32,
		EfSearch:       32,
		Seed:           7,
	})

	_ = idx.AddVector(1, []float64{0, 0})
	_ = idx.AddVector(2, []float64{0.1, 0.1})
	_ = idx.AddVector(3, []float64{10, 10})

	idx.DeleteVector(1)
	stats := idx.Stats()
	if stats.Nodes != 3 || stats.Deleted != 1 {
		t.Fatalf("stats after delete = %+v, want 3 nodes and 1 deleted", stats)
	}

	got, err := idx.Search([]float64{0, 0}, 2)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if slices.Contains(got, 1) {
		t.Fatalf("deleted id returned in results: %v", got)
	}

	_ = idx.AddVector(1, []float64{0, 0})
	stats = idx.Stats()
	if stats.Nodes != 3 || stats.Deleted != 0 {
		t.Fatalf("stats after re-add = %+v, want 3 nodes and 0 deleted", stats)
	}

	got, err = idx.Search([]float64{0, 0}, 1)
	if err != nil {
		t.Fatalf("search failed after re-add: %v", err)
	}
	if len(got) != 1 || got[0] != 1 {
		t.Fatalf("re-added id not returned first: %v", got)
	}
}

func TestAnnIndexUsesCompactSlotsWithSparseExternalIDs(t *testing.T) {
	idx := NewAnnIndexWithOptions(Options{
		M:              8,
		EfConstruction: 32,
		EfSearch:       32,
		Seed:           7,
	})

	_ = idx.AddVector(1001, []float64{0, 0})
	_ = idx.AddVector(42, []float64{1, 1})
	_ = idx.AddVector(900000, []float64{10, 10})

	if got, want := len(idx.nodes), 3; got != want {
		t.Fatalf("compact node count = %d, want %d", got, want)
	}
	if idx.nodes[idx.idToSlot[1001]].id != 1001 {
		t.Fatal("expected sparse external id to resolve through slot map")
	}

	got, err := idx.Search([]float64{0, 0}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 || got[0] != 1001 {
		t.Fatalf("Search returned public IDs %v, want 1001 first", got)
	}

	withDistances, err := idx.SearchWithDistances([]float64{10, 10}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(withDistances) != 1 || withDistances[0].ID != 900000 {
		t.Fatalf("SearchWithDistances returned %+v, want external id 900000", withDistances)
	}
}

func TestAnnIndexSearchWithDistancesIntoReusesBuffer(t *testing.T) {
	idx := NewAnnIndexWithOptions(Options{
		M:              8,
		EfConstruction: 32,
		EfSearch:       32,
		Seed:           7,
	})

	_ = idx.AddVector(1, []float64{0, 0})
	_ = idx.AddVector(2, []float64{1, 1})
	_ = idx.AddVector(3, []float64{10, 10})

	buf := make([]Result, 0, 4)
	got, err := idx.SearchWithDistancesInto([]float64{0, 0}, 2, buf)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || cap(got) != cap(buf) {
		t.Fatalf("SearchWithDistancesInto len/cap = %d/%d, want len 2 cap %d", len(got), cap(got), cap(buf))
	}
	if got[0].ID != 1 {
		t.Fatalf("SearchWithDistancesInto first = %+v, want id 1", got[0])
	}
}

func TestAnnIndexNativeDistanceMetrics(t *testing.T) {
	for _, metric := range []string{MetricCosine, MetricInnerProduct} {
		t.Run(metric, func(t *testing.T) {
			idx := NewAnnIndexWithOptions(Options{M: 8, EfConstruction: 32, EfSearch: 32, Metric: metric})
			if err := idx.AddVector(1, []float64{100, 1}); err != nil {
				t.Fatal(err)
			}
			if err := idx.AddVector(2, []float64{1, 1}); err != nil {
				t.Fatal(err)
			}
			hits, err := idx.SearchWithDistancesInto([]float64{1, 0}, 1, nil)
			if err != nil || len(hits) != 1 || hits[0].ID != 1 {
				t.Fatalf("metric=%s hits=%+v err=%v", metric, hits, err)
			}
			if metric == MetricInnerProduct && hits[0].Distance != -100 {
				t.Fatalf("inner-product distance=%v", hits[0].Distance)
			}
		})
	}
}

func TestQuantizedAnnIndexNativeCosineMetric(t *testing.T) {
	idx := NewAnnIndexWithOptions(Options{M: 8, EfConstruction: 32, EfSearch: 32, Metric: MetricCosine})
	for id, values := range [][]float64{{100, 1}, {1, 1}, {0, 1}} {
		if err := idx.AddVector(id+1, values); err != nil {
			t.Fatal(err)
		}
	}
	if err := idx.Quantize(); err != nil {
		t.Fatal(err)
	}
	hits, err := idx.SearchWithDistancesInto([]float64{1, 0}, 1, nil)
	if err != nil || len(hits) != 1 || hits[0].ID != 1 {
		t.Fatalf("hits=%+v err=%v", hits, err)
	}
}

func TestAnnIndexDimensionValidation(t *testing.T) {
	idx := NewAnnIndex()
	if err := idx.AddVector(1, []float64{1, 2, 3}); err != nil {
		t.Fatalf("unexpected add error: %v", err)
	}
	if err := idx.AddVector(2, []float64{1, 2}); err == nil {
		t.Fatal("expected dimension error")
	}
	if _, err := idx.Search([]float64{1, 2}, 1); err == nil {
		t.Fatal("expected dimension error")
	}
}

func TestAnnIndexConcurrentAccess(t *testing.T) {
	idx := NewAnnIndex()
	done := make(chan struct{})

	for i := 0; i < 50; i++ {
		go func(n int) {
			_ = idx.AddVector(n, []float64{float64(n), float64(n + 1)})
			_, _ = idx.Search([]float64{1, 2}, 1)
			done <- struct{}{}
		}(i + 1)
	}

	for i := 0; i < 50; i++ {
		<-done
	}
}

func BenchmarkAnnSearch(b *testing.B) {
	idx := NewAnnIndexWithOptions(Options{
		M:              16,
		EfConstruction: 64,
		EfSearch:       64,
		Seed:           9,
	})

	for i := 0; i < 2000; i++ {
		v := []float64{float64(i % 97), float64((i * 3) % 89), float64((i * 7) % 83)}
		_ = idx.AddVector(i, v)
	}

	query := []float64{12.4, 18.2, 7.1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := idx.Search(query, 10)
		if err != nil {
			b.Fatalf("search failed: %v", err)
		}
	}
}

func BenchmarkAnnSearchTuning(b *testing.B) {
	cases := []struct {
		name string
		opts Options
	}{
		{
			name: "m8_ef32_32",
			opts: Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 9},
		},
		{
			name: "m16_ef64_64",
			opts: Options{M: 16, EfConstruction: 64, EfSearch: 64, Seed: 9},
		},
		{
			name: "m24_ef96_48",
			opts: Options{M: 24, EfConstruction: 96, EfSearch: 48, Seed: 9},
		},
		{
			name: "m24_ef96_96",
			opts: Options{M: 24, EfConstruction: 96, EfSearch: 96, Seed: 9},
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			idx := buildBenchmarkAnnIndex(tc.opts)
			query := []float64{12.4, 18.2, 7.1}
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_, err := idx.Search(query, 10)
				if err != nil {
					b.Fatalf("search failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkAnnSearchEmbeddingDimensions(b *testing.B) {
	for _, dim := range []int{384, 768, 1536} {
		b.Run(fmt.Sprintf("dim_%d", dim), func(b *testing.B) {
			idx := NewAnnIndexWithOptions(Options{
				M:              16,
				EfConstruction: 64,
				EfSearch:       64,
				Seed:           9,
			})

			for i := 0; i < 512; i++ {
				if err := idx.AddVector(i, benchmarkEmbeddingVector(dim, i)); err != nil {
					b.Fatal(err)
				}
			}

			query := benchmarkEmbeddingVector(dim, 3)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_, err := idx.Search(query, 10)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestAnnSearchTuningRecall(t *testing.T) {
	dataset := benchmarkAnnDataset(512)
	queries := [][]float64{
		{1.2, 5.4, 7.8},
		{9.1, 3.3, 4.4},
		{12.4, 18.2, 7.1},
		{22.0, 11.5, 0.9},
		{31.7, 6.8, 15.3},
	}
	expected := make([][]int, 0, len(queries))
	for _, query := range queries {
		expected = append(expected, bruteForceTopK(dataset, query, 10))
	}

	cases := []struct {
		name      string
		opts      Options
		minRecall float64
	}{
		{
			name:      "m8_ef32_32",
			opts:      Options{M: 8, EfConstruction: 32, EfSearch: 32, Seed: 9},
			minRecall: 0.50,
		},
		{
			name:      "m16_ef64_64",
			opts:      Options{M: 16, EfConstruction: 64, EfSearch: 64, Seed: 9},
			minRecall: 0.80,
		},
		{
			name:      "m24_ef96_96",
			opts:      Options{M: 24, EfConstruction: 96, EfSearch: 96, Seed: 9},
			minRecall: 0.85,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			idx := NewAnnIndexWithOptions(tc.opts)
			for _, item := range dataset {
				if err := idx.AddVector(item.id, item.vector); err != nil {
					t.Fatalf("add failed: %v", err)
				}
			}

			var matched int
			var total int
			for i, query := range queries {
				got, err := idx.Search(query, 10)
				if err != nil {
					t.Fatalf("search failed: %v", err)
				}
				total += len(expected[i])
				for _, id := range got {
					if slices.Contains(expected[i], id) {
						matched++
					}
				}
			}

			recall := float64(matched) / float64(total)
			t.Logf("Recall@10=%.4f", recall)
			if recall < tc.minRecall {
				t.Fatalf("recall %.2f below threshold %.2f", recall, tc.minRecall)
			}
		})
	}
}

func TestSearchWithDistancesEfIntoNeverUsesBudgetBelowK(t *testing.T) {
	idx := NewAnnIndexWithOptions(Options{
		M:              4,
		EfConstruction: 16,
		EfSearch:       4,
		Seed:           9,
	})
	for i := 0; i < 32; i++ {
		if err := idx.AddVector(i, []float64{float64(i), 0}); err != nil {
			t.Fatal(err)
		}
	}

	results, err := idx.SearchWithDistancesEfInto([]float64{15.5, 0}, 10, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 10 {
		t.Fatalf("results = %d, want 10", len(results))
	}
}

type benchmarkAnnItem struct {
	id     int
	vector []float64
}

func buildBenchmarkAnnIndex(opts Options) *AnnIndex {
	idx := NewAnnIndexWithOptions(opts)
	for _, item := range benchmarkAnnDataset(2000) {
		_ = idx.AddVector(item.id, item.vector)
	}
	return idx
}

func benchmarkAnnDataset(n int) []benchmarkAnnItem {
	out := make([]benchmarkAnnItem, 0, n)
	for i := 0; i < n; i++ {
		v := []float64{float64(i % 97), float64((i * 3) % 89), float64((i * 7) % 83)}
		out = append(out, benchmarkAnnItem{id: i, vector: v})
	}
	return out
}

func benchmarkEmbeddingVector(dim int, seed int) []float64 {
	values := make([]float64, dim)
	state := uint64(seed + 1)
	for i := range values {
		state = state*6364136223846793005 + 1442695040888963407
		values[i] = float64(int(state>>48)%2000-1000) / 1000
	}
	return values
}

func bruteForceTopK(dataset []benchmarkAnnItem, query []float64, k int) []int {
	type scored struct {
		id       int
		distance float64
	}
	scoredItems := make([]scored, 0, len(dataset))
	for _, item := range dataset {
		scoredItems = append(scoredItems, scored{
			id:       item.id,
			distance: vector.SquaredEuclideanDistance(query, item.vector),
		})
	}
	slices.SortFunc(scoredItems, func(a, b scored) int {
		switch {
		case a.distance < b.distance:
			return -1
		case a.distance > b.distance:
			return 1
		default:
			return 0
		}
	})
	if k > len(scoredItems) {
		k = len(scoredItems)
	}
	out := make([]int, 0, k)
	for i := 0; i < k; i++ {
		out = append(out, scoredItems[i].id)
	}
	return out
}

func ExampleAnnIndex_searchTuning() {
	fmt.Println("ann tuning defaults: m=16 ef_construction=64 ef_search=64")
	// Output: ann tuning defaults: m=16 ef_construction=64 ef_search=64
}
