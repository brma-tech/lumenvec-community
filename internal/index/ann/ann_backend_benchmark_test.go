package ann

import (
	"math/rand"
	"testing"
)

type backendBenchmarkData struct {
	vectors [][]float64
	queries [][]float64
	hnsw    *AnnIndex
	ivf     *IVFANNIndex
	ivfFine *IVFANNIndex
}

func newBackendBenchmarkData(b *testing.B) *backendBenchmarkData {
	b.Helper()
	rng := rand.New(rand.NewSource(42))
	d := &backendBenchmarkData{vectors: make([][]float64, 100000), queries: make([][]float64, 100)}
	for n := range d.vectors {
		d.vectors[n] = make([]float64, 128)
		for i := range d.vectors[n] {
			d.vectors[n][i] = rng.Float64()
		}
	}
	for n := range d.queries {
		d.queries[n] = make([]float64, 128)
		for dim := range d.queries[n] {
			d.queries[n][dim] = rng.Float64()
		}
	}
	d.hnsw = NewAnnIndexWithOptions(Options{M: 16, EfConstruction: 64, EfSearch: 64})
	for n, vector := range d.vectors {
		if err := d.hnsw.AddVector(n, vector); err != nil {
			b.Fatal(err)
		}
	}
	centroids, err := TrainIVFCentroids(d.vectors, 64)
	if err != nil {
		b.Fatal(err)
	}
	d.ivf, err = NewIVFANNIndex(centroids, 4)
	if err != nil {
		b.Fatal(err)
	}
	for n, vector := range d.vectors {
		if err := d.ivf.AddVector(n, vector); err != nil {
			b.Fatal(err)
		}
	}
	centroids, err = TrainIVFCentroidsKMeans(d.vectors, 256, 6, 4096)
	if err != nil {
		b.Fatal(err)
	}
	d.ivfFine, err = NewIVFANNIndex(centroids, 1)
	if err != nil {
		b.Fatal(err)
	}
	for n, vector := range d.vectors {
		if err := d.ivfFine.AddVector(n, vector); err != nil {
			b.Fatal(err)
		}
	}
	return d
}

func BenchmarkANNBackends100k(b *testing.B) {
	d := newBackendBenchmarkData(b)
	for _, tc := range []struct {
		name   string
		search func([]float64) ([]Result, error)
	}{
		{name: "hnsw", search: func(q []float64) ([]Result, error) { return d.hnsw.SearchWithDistancesInto(q, 10, nil) }},
		{name: "ivf", search: func(q []float64) ([]Result, error) { return d.ivf.SearchWithDistancesInto(q, 10, nil) }},
		{name: "ivf-256-nprobe1", search: func(q []float64) ([]Result, error) { return d.ivfFine.SearchWithDistancesInto(q, 10, nil) }},
		{name: "ivf-256-adaptive", search: func(q []float64) ([]Result, error) {
			return d.ivfFine.SearchAdaptiveWithDistancesInto(q, 10, 2000, 64, nil)
		}},
	} {
		b.Run(tc.name, func(b *testing.B) {
			for _, query := range d.queries {
				if _, err := tc.search(query); err != nil {
					b.Fatal(err)
				}
			}
			b.ResetTimer()
			b.ReportAllocs()
			for n := 0; n < b.N; n++ {
				if _, err := tc.search(d.queries[n%len(d.queries)]); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
