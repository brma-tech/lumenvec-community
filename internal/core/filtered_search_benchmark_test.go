package core

import (
	"fmt"
	"lumenvec/internal/index"
	"path/filepath"
	"testing"
)

func BenchmarkFilteredSearch(b *testing.B) {
	base := b.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 8, MaxK: 10, SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal.log"), VectorPath: filepath.Join(base, "vectors")})
	for i := 0; i < 1000; i++ {
		if err := s.AddVector(fmt.Sprintf("group-%d", i), []float64{float64(i % 8), 1, 2, 3}); err != nil {
			b.Fatal(err)
		}
	}
	query := []float64{1, 1, 2, 3}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.SearchFiltered(query, 10, func(v index.Vector) bool { return v.ID >= "group-0900" }); err != nil {
			b.Fatal(err)
		}
	}
	_ = s.Close()
}

func BenchmarkFilteredSearchMetrics(b *testing.B) {
	base := b.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 8, MaxK: 10, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log"), VectorPath: filepath.Join(base, "v")})
	for i := 0; i < 1000; i++ {
		_ = s.AddVector(fmt.Sprintf("v-%d", i), []float64{1, 0, 0, 0})
	}
	filter := func(v index.Vector) bool { return true }
	b.ResetTimer()
	for _, metric := range []DistanceMetric{MetricL2, MetricCosine, MetricInnerProduct} {
		b.Run(string(metric), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = s.SearchFilteredMetric([]float64{1, 0, 0, 0}, 10, filter, metric)
			}
		})
	}
	_ = s.Close()
}

func BenchmarkStructuredTextSearch(b *testing.B) {
	base := b.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 8, MaxK: 10, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log"), VectorPath: filepath.Join(base, "v")})
	for i := 0; i < 1000; i++ {
		_ = s.AddVector(fmt.Sprintf("article-%04d", i), []float64{1, 0, 0, 0})
	}
	defer s.Close()
	filter := StructuredFilter{TextQuery: "article-0999"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.SearchStructured([]float64{1, 0, 0, 0}, 10, filter, MetricCosine); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkStructuredMetadataSearch(b *testing.B) {
	base := b.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 8, MaxK: 10, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log"), VectorPath: filepath.Join(base, "v")})
	for i := 0; i < 1000; i++ {
		if err := s.AddVectorWithMetadata(fmt.Sprintf("v-%04d", i), []float64{1, 0, 0, 0}, map[string]string{"tenant": fmt.Sprintf("t-%d", i%10)}); err != nil {
			b.Fatal(err)
		}
	}
	defer s.Close()
	filter := StructuredFilter{Metadata: map[string]string{"tenant": "t-7"}}
	// Build the lazy text/metadata structures outside the timed region. The
	// gate measures steady-state query cost, not one-time index initialization.
	if _, err := s.SearchStructured([]float64{1, 0, 0, 0}, 10, filter, MetricCosine); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.SearchStructured([]float64{1, 0, 0, 0}, 10, filter, MetricCosine); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFilteredSearchSelectivities(b *testing.B) {
	base := b.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 8, MaxK: 10, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log"), VectorPath: filepath.Join(base, "v")})
	for i := 0; i < 1000; i++ {
		if err := s.AddVector(fmt.Sprintf("v-%04d", i), []float64{1, 0, 0, 0}); err != nil {
			b.Fatal(err)
		}
	}
	defer s.Close()
	for _, tc := range []struct {
		name string
		keep int
	}{{"05", 50}, {"20", 200}, {"50", 500}, {"100", 1000}} {
		b.Run(tc.name, func(b *testing.B) {
			filter := func(v index.Vector) bool { var n int; _, _ = fmt.Sscanf(v.ID, "v-%04d", &n); return n < tc.keep }
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := s.SearchFiltered([]float64{1, 0, 0, 0}, 10, filter); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkSearchMultiVector(b *testing.B) {
	base := b.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 8, MaxK: 10, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log"), VectorPath: filepath.Join(base, "v")})
	for i := 0; i < 1000; i++ {
		if err := s.AddVector(fmt.Sprintf("v-%04d", i), []float64{float64(i % 8), 1, 0, 0}); err != nil {
			b.Fatal(err)
		}
	}
	defer s.Close()
	queries := [][]float64{{1, 1, 0, 0}, {2, 1, 0, 0}, {3, 1, 0, 0}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.SearchMultiVector(queries, 10, nil, MetricL2); err != nil {
			b.Fatal(err)
		}
	}
}
