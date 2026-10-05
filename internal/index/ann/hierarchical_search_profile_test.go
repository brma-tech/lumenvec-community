//go:build annprofile

package ann

import (
	"math/rand"
	"os"
	"runtime/pprof"
	"testing"
	"time"
)

// TestHierarchicalSearchCPUProfile profiles only the hot search loop. Build,
// warmup, result validation and file I/O are deliberately outside the CPU
// profile so pprof percentages describe query work rather than setup.
func TestHierarchicalSearchCPUProfile(t *testing.T) {
	const vectorCount, dimensions, queryCount = 10_000, 128, 100
	profilePath := os.Getenv("LUMENVEC_ANN_CPU_PROFILE_PATH")
	if profilePath == "" {
		t.Skip("set LUMENVEC_ANN_CPU_PROFILE_PATH to enable the opt-in profile")
	}
	rng := rand.New(rand.NewSource(42))
	batch := make([]BatchVector32, vectorCount)
	for id := range batch {
		values := make([]float32, dimensions)
		for dimension := range values {
			values[dimension] = rng.Float32()
		}
		batch[id] = BatchVector32{ID: id, Values: values}
	}
	queries := make([][]float64, queryCount)
	for queryIndex := range queries {
		queries[queryIndex] = make([]float64, dimensions)
		for dimension := range queries[queryIndex] {
			queries[queryIndex][dimension] = rng.Float64()
		}
	}
	index, err := BuildHierarchicalIndex(Options{
		M: 30, EfConstruction: 224, EfSearch: 672, Seed: 42,
		DiversifiedExistingPruning:          true,
		HierarchicalBuildScratchBudgetBytes: 128 << 20,
	}, batch)
	if err != nil {
		t.Fatal(err)
	}
	for _, query := range queries {
		if _, err := index.SearchWithDistancesEfInto(query, 10, 672, nil); err != nil {
			t.Fatal(err)
		}
	}
	file, err := os.Create(profilePath)
	if err != nil {
		t.Fatal(err)
	}
	if err := pprof.StartCPUProfile(file); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	started := time.Now()
	deadline := started.Add(15 * time.Second)
	searches := 0
	for time.Now().Before(deadline) {
		query := queries[searches%len(queries)]
		if _, err := index.SearchWithDistancesEfInto(query, 10, 672, nil); err != nil {
			pprof.StopCPUProfile()
			_ = file.Close()
			t.Fatal(err)
		}
		searches++
	}
	pprof.StopCPUProfile()
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	t.Logf("profiled %d searches in %s at efSearch=672", searches, time.Since(started))
}
