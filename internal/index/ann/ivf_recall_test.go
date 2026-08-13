package ann

import (
	"sort"
	"testing"
)

func TestIVFRecallAgainstExactCandidates(t *testing.T) {
	centroids := [][]float64{{0, 0}, {10, 10}}
	idx, err := NewIVFIndex(centroids, 2)
	if err != nil {
		t.Fatal(err)
	}
	for n := 0; n < 100; n++ {
		base := float64(n%2) * 10
		if err := idx.Add(n, []float64{base + float64(n%5)*.01, base + float64(n%7)*.01}); err != nil {
			t.Fatal(err)
		}
	}
	query := []float64{10.01, 10.02}
	hits, err := idx.Search(query, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 10 {
		t.Fatalf("hits=%d", len(hits))
	}
	// With nprobe=2 every point is a candidate, so IVF recall against exact
	// ranking must be one for the requested top-k.
	exact := make([]Result, 0, 100)
	for n := 0; n < 100; n++ {
		base := float64(n%2) * 10
		vector := []float64{base + float64(n%5)*.01, base + float64(n%7)*.01}
		exact = append(exact, Result{ID: n, Distance: l2(query, vector)})
	}
	sort.Slice(exact, func(a, b int) bool {
		if exact[a].Distance == exact[b].Distance {
			return exact[a].ID < exact[b].ID
		}
		return exact[a].Distance < exact[b].Distance
	})
	want := make(map[int]struct{})
	for _, hit := range exact[:10] {
		want[hit.ID] = struct{}{}
	}
	matches := 0
	for _, hit := range hits {
		if _, ok := want[hit.ID]; ok {
			matches++
		}
	}
	if matches != 10 {
		t.Fatalf("recall=%d/10", matches)
	}
}
