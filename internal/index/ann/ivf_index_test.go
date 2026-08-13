package ann

import "testing"

func TestIVFIndexRoutesAndSearchesCandidates(t *testing.T) {
	idx, err := NewIVFIndex([][]float64{{0, 0}, {10, 10}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Add(1, []float64{0, 1}); err != nil {
		t.Fatal(err)
	}
	if err := idx.Add(2, []float64{10, 9}); err != nil {
		t.Fatal(err)
	}
	hits, err := idx.Search([]float64{9, 10}, 1)
	if err != nil || len(hits) != 1 || hits[0].ID != 2 {
		t.Fatalf("hits=%+v err=%v", hits, err)
	}
}

func TestIVFIndexSearchAdaptiveExpandsCandidatePool(t *testing.T) {
	idx, err := NewIVFIndex([][]float64{{0}, {100}, {200}, {300}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	for n, value := range []float64{0, 1, 100, 101, 200, 201, 300, 301} {
		if err := idx.Add(n, []float64{value}); err != nil {
			t.Fatal(err)
		}
	}
	hits, err := idx.SearchAdaptive([]float64{150}, 2, 4, 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 2 || hits[0].ID != 3 || hits[1].ID != 2 {
		t.Fatalf("unexpected adaptive results: %+v", hits)
	}
}

func TestIVFIndexValidation(t *testing.T) {
	if _, err := NewIVFIndex(nil, 1); err == nil {
		t.Fatal("expected centroid error")
	}
	idx, _ := NewIVFIndex([][]float64{{0}}, 1)
	if err := idx.Add(1, []float64{1, 2}); err == nil {
		t.Fatal("expected add mismatch")
	}
}

func TestIVFIndexUpsertMovesEntryWithoutDuplicate(t *testing.T) {
	idx, err := NewIVFIndex([][]float64{{0}, {100}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Add(7, []float64{1}); err != nil {
		t.Fatal(err)
	}
	if err := idx.Add(7, []float64{99}); err != nil {
		t.Fatal(err)
	}
	if got := len(idx.lists[0]) + len(idx.lists[1]); got != 1 {
		t.Fatalf("upsert created %d entries", got)
	}
	hits, err := idx.Search([]float64{100}, 1)
	if err != nil || len(hits) != 1 || hits[0].ID != 7 {
		t.Fatalf("hits=%+v err=%v", hits, err)
	}
}

func TestIVFClosestCentroidsReturnsExactPrefix(t *testing.T) {
	idx, err := NewIVFIndex([][]float64{{0}, {10}, {20}, {30}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	got := idx.closestCentroids([]float64{17}, 2)
	want := []int{2, 1}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("closest centroids=%v, want %v", got, want)
	}
}

func TestTrainIVFCentroidsIsDeterministic(t *testing.T) {
	centroids, err := TrainIVFCentroids([][]float64{{0}, {1}, {2}, {3}}, 2)
	if err != nil || len(centroids) != 2 || centroids[0][0] != 0 || centroids[1][0] != 3 {
		t.Fatalf("centroids=%v err=%v", centroids, err)
	}
}
