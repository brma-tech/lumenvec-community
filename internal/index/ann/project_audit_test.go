//go:build projectaudit

package ann

import (
	"sync"
	"testing"
)

func TestProjectAuditDeletedCandidates(t *testing.T) {
	batch := make([]BatchVector32, 64)
	for i := range batch {
		batch[i] = BatchVector32{ID: i, Values: []float32{float32(i), 0}}
	}
	h, err := BuildHierarchicalIndex(Options{M: 16, EfConstruction: 64, EfSearch: 10, Seed: 42}, batch)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 16; i++ {
		h.DeleteVector(i)
	}
	r, err := h.Search([]float64{0, 0}, 10)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("requested=10 live=48 returned=%d results=%v", len(r), r)
	if len(r) != 10 {
		t.Fatalf("deleted candidates consume the search budget: got %d of 10 with 48 live nodes", len(r))
	}
}

func TestProjectAuditInvalidBudget(t *testing.T) {
	h := NewHierarchicalIndex(Options{})
	_, _ = h.SearchWithDistancesEfInto([]float64{0}, -1, -1, nil)
	q, s, ef := h.SearchBudgetState()
	if q != 0 || s != 0 || ef != 0 {
		t.Fatalf("invalid request mutated counters: queries=%d segments=%d ef=%d", q, s, ef)
	}
}

func TestProjectAuditEfSearchRace(t *testing.T) {
	h, err := BuildHierarchicalIndex(Options{M: 8, EfConstruction: 32, EfSearch: 32}, []BatchVector32{{ID: 0, Values: []float32{1, 0}}})
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 10000; i++ {
			h.SetEfSearch(32 + i%2)
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 10000; i++ {
			_, _ = h.Search([]float64{1, 0}, 1)
		}
	}()
	wg.Wait()
}
