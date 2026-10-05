package ann

import (
	"math/rand"
	"testing"
)

func TestHierarchicalGraphLinkInvariants(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	batch := make([]BatchVector32, 10000)
	for id := range batch {
		values := make([]float32, 128)
		for d := range values {
			values[d] = float32(rng.Float64())
		}
		batch[id] = BatchVector32{ID: id, Values: values}
	}
	h, err := BuildHierarchicalIndex(Options{M: 20, EfConstruction: 128, EfSearch: 256, Seed: 42}, batch)
	if err != nil {
		t.Fatal(err)
	}
	incoming := make([]int, len(h.nodes))
	for owner, node := range h.nodes {
		for layer, links := range node.links {
			seen := make(map[int]bool)
			if len(links) > h.layerLimit(layer) {
				t.Fatalf("node %d layer %d exceeds degree", owner, layer)
			}
			for _, link := range links {
				target := int(link)
				if target == owner || target < 0 || target >= len(h.nodes) {
					t.Fatalf("invalid edge %d -> %d", owner, target)
				}
				if seen[target] {
					t.Fatalf("duplicate edge node=%d layer=%d target=%d", owner, layer, target)
				}
				seen[target] = true
				if layer == 0 {
					incoming[target]++
				}
				if len(h.nodes[target].links) <= layer {
					t.Fatalf("target %d absent from layer %d", target, layer)
				}
			}
		}
	}
	zeroIncoming := 0
	for _, count := range incoming {
		if count == 0 {
			zeroIncoming++
		}
	}
	t.Logf("Connectivity (diagnostic, not forced repair): %+v; layer-zero nodes without incoming edges=%d", h.Connectivity(), zeroIncoming)
}
