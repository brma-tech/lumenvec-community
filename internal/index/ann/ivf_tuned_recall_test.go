package ann

import (
	"math/rand"
	"sort"
	"testing"
)

func TestIVF256NProbe1RecallMeasurement(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	vectors := make([][]float64, 10000)
	for n := range vectors {
		vectors[n] = make([]float64, 32)
		for i := range vectors[n] {
			vectors[n][i] = rng.Float64()
		}
	}
	centroids, err := TrainIVFCentroidsKMeans(vectors, 256, 8, 4096)
	if err != nil {
		t.Fatal(err)
	}
	idx, err := NewIVFIndex(centroids, 128)
	if err != nil {
		t.Fatal(err)
	}
	for n, vector := range vectors {
		if err := idx.Add(n, vector); err != nil {
			t.Fatal(err)
		}
	}
	queries := vectors[:100]
	matched := 0
	for _, query := range queries {
		hits, err := idx.Search(query, 10)
		if err != nil {
			t.Fatal(err)
		}
		exact := make([]Result, len(vectors))
		for n, vector := range vectors {
			exact[n] = Result{ID: n, Distance: l2(query, vector)}
		}
		sort.Slice(exact, func(a, b int) bool { return exact[a].Distance < exact[b].Distance })
		want := make(map[int]struct{}, 10)
		for _, hit := range exact[:10] {
			want[hit.ID] = struct{}{}
		}
		for _, hit := range hits {
			if _, ok := want[hit.ID]; ok {
				matched++
			}
		}
	}
	recall := float64(matched) / float64(len(queries)*10)
	t.Logf("IVF-256/nprobe128 Recall@10=%.4f", recall)
}
