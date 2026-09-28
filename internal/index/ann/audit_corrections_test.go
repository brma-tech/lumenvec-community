package ann

import "testing"

func TestHierarchicalIncrementalMemoryCounters(t *testing.T) {
	for _, diverse := range []bool{false, true} {
		batch := make([]BatchVector32, 128)
		for i := range batch {
			batch[i] = BatchVector32{ID: i, Values: []float32{float32(i), float32(i % 7)}}
		}
		h, err := BuildHierarchicalIndex(Options{M: 8, EfConstruction: 32, EfSearch: 32, DiversifiedExistingPruning: diverse}, batch)
		if err != nil {
			t.Fatal(err)
		}
		check := func() {
			var layers, edges uint64
			for _, node := range h.nodes {
				layers += uint64(len(node.links))*24 + uint64(len(node.farthest))*8
				for _, links := range node.links {
					edges += uint64(len(links)) * 4
				}
			}
			m := h.MemoryStats()
			if m.NodeBytes != layers || m.AdjacencyBytes != edges {
				t.Fatalf("counter mismatch: %+v want layers=%d edges=%d", m, layers, edges)
			}
		}
		check()
		if err := h.AddVector32(200, []float32{1, 2}); err != nil {
			t.Fatal(err)
		}
		h.DeleteVector(0)
		check()
	}
}
