package ann

import (
	"math/rand"
	"sort"
	"testing"
)

func TestIVFAdaptiveRecall10kMeasurement(t *testing.T) {
	rng := rand.New(rand.NewSource(20260713))
	const n, dim, queries = 10000, 16, 40
	vectors := make([][]float64, n)
	for i := range vectors {
		vectors[i] = make([]float64, dim)
		for d := range vectors[i] {
			vectors[i][d] = rng.Float64()
		}
	}
	centroids, err := TrainIVFCentroidsKMeans(vectors, 64, 6, 4096)
	if err != nil {
		t.Fatal(err)
	}
	fixed, err := NewIVFIndex(centroids, 1)
	if err != nil {
		t.Fatal(err)
	}
	adaptive, err := NewIVFIndex(centroids, 1)
	if err != nil {
		t.Fatal(err)
	}
	for id, vector := range vectors {
		if err := fixed.Add(id, vector); err != nil {
			t.Fatal(err)
		}
		if err := adaptive.Add(id, vector); err != nil {
			t.Fatal(err)
		}
	}
	var fixedHits, total int
	targets := []int{40, 200, 500, 1000, 2000, 4000}
	adaptiveHits := make([]int, len(targets))
	for q := 0; q < queries; q++ {
		query := vectors[q*(n/queries)]
		exact := make([]Result, n)
		for id, vector := range vectors {
			exact[id] = Result{ID: id, Distance: l2(query, vector)}
		}
		sort.Slice(exact, func(a, b int) bool { return exact[a].Distance < exact[b].Distance })
		want := make(map[int]struct{}, 10)
		for _, hit := range exact[:10] {
			want[hit.ID] = struct{}{}
		}
		fixedResults, err := fixed.Search(query, 10)
		if err != nil {
			t.Fatal(err)
		}
		for _, hit := range fixedResults {
			if _, ok := want[hit.ID]; ok {
				fixedHits++
			}
		}
		for targetIndex, target := range targets {
			adaptiveResults, err := adaptive.SearchAdaptive(query, 10, target, 64)
			if err != nil {
				t.Fatal(err)
			}
			for _, hit := range adaptiveResults {
				if _, ok := want[hit.ID]; ok {
					adaptiveHits[targetIndex]++
				}
			}
		}
		total += 10
	}
	t.Logf("10k Recall@10 fixed_nprobe1=%.4f", float64(fixedHits)/float64(total))
	for n, target := range targets {
		t.Logf("10k Recall@10 adaptive_candidates_%d=%.4f", target, float64(adaptiveHits[n])/float64(total))
	}
}
