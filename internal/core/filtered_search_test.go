package core

import (
	"errors"
	"fmt"
	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"
	"time"
)

func TestFilteredSearchRecallAt10(t *testing.T) {
	base := t.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 20, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log"), VectorPath: filepath.Join(base, "v")})
	defer s.Close()
	for i := 0; i < 40; i++ {
		if err := s.AddVector(fmt.Sprintf("v-%02d", i), []float64{float64(i), 0}); err != nil {
			t.Fatal(err)
		}
	}
	filter := func(v index.Vector) bool {
		var n int
		_, _ = fmt.Sscanf(v.ID, "v-%d", &n)
		return n%2 == 0
	}
	results, err := s.SearchFilteredMetric([]float64{0, 0}, 10, filter, MetricL2)
	if err != nil {
		t.Fatal(err)
	}
	// The exact filtered nearest-neighbor set is v-00 through v-18 (even IDs).
	if len(results) != 10 {
		t.Fatalf("results=%d", len(results))
	}
	hits := 0
	for i, result := range results {
		if result.ID == fmt.Sprintf("v-%02d", i*2) {
			hits++
		}
	}
	if recall := float64(hits) / 10; recall < 0.95 {
		t.Fatalf("Recall@10=%.2f results=%+v", recall, results)
	}
}

func TestStructuredFilterTextQueryMatchesMetadata(t *testing.T) {
	base := t.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 5, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log")})
	defer s.Close()
	if err := s.AddVectorWithMetadata("doc-1", []float64{1, 0}, map[string]string{"title": "Distributed Raft guide"}); err != nil {
		t.Fatal(err)
	}
	if err := s.AddVectorWithMetadata("doc-2", []float64{0, 1}, map[string]string{"title": "Storage notes"}); err != nil {
		t.Fatal(err)
	}
	filter := StructuredFilter{TextQuery: "raft"}
	results, err := s.SearchFilteredMetric([]float64{1, 0}, 5, func(v index.Vector) bool { return filter.Match(v) }, MetricL2)
	if err != nil || len(results) != 1 || results[0].ID != "doc-1" {
		t.Fatalf("results=%+v err=%v", results, err)
	}
}

// TestFilteredSearchRecallAcrossSelectivities prevents the published recall
// claim from depending on one unusually dense filter. It exercises sparse,
// medium and dense candidate sets and records a latency sample for each.
func TestFilteredSearchRecallAcrossSelectivities(t *testing.T) {
	base := t.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 20, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log"), VectorPath: filepath.Join(base, "v")})
	defer s.Close()
	for i := 0; i < 200; i++ {
		if err := s.AddVector(fmt.Sprintf("v-%03d", i), []float64{float64(i), 0}); err != nil {
			t.Fatal(err)
		}
	}
	for _, modulus := range []int{20, 5, 2, 1} { // 5%, 20%, 50%, 100%
		modulus := modulus
		filter := func(v index.Vector) bool {
			var n int
			_, _ = fmt.Sscanf(v.ID, "v-%d", &n)
			return n%modulus == 0
		}
		got, err := s.SearchFilteredMetric([]float64{0, 0}, 10, filter, MetricL2)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 10 {
			t.Fatalf("modulus %d returned %d results", modulus, len(got))
		}
		for i, result := range got {
			want := fmt.Sprintf("v-%03d", i*modulus)
			if result.ID != want {
				t.Fatalf("modulus %d result[%d]=%s want %s", modulus, i, result.ID, want)
			}
		}
	}
}

func TestFilteredSearchP95SLOAcrossSelectivities(t *testing.T) {
	base := t.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 8, MaxK: 10, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log"), VectorPath: filepath.Join(base, "v")})
	defer s.Close()
	for i := 0; i < 1000; i++ {
		if err := s.AddVector(fmt.Sprintf("v-%04d", i), []float64{float64(i % 8), 1, 2, 3, 4, 5, 6, 7}); err != nil {
			t.Fatal(err)
		}
	}
	for _, modulus := range []int{20, 5, 2, 1} {
		filter := func(v index.Vector) bool {
			var n int
			_, _ = fmt.Sscanf(v.ID, "v-%d", &n)
			return n%modulus == 0
		}
		durations := make([]time.Duration, 0, 40)
		for i := 0; i < cap(durations); i++ {
			started := time.Now()
			if _, err := s.SearchFilteredMetric([]float64{0, 1, 2, 3, 4, 5, 6, 7}, 10, filter, MetricCosine); err != nil {
				t.Fatal(err)
			}
			durations = append(durations, time.Since(started))
		}
		sort.Slice(durations, func(i, j int) bool { return durations[i] < durations[j] })
		p95 := durations[(len(durations)*95+99)/100-1]
		t.Logf("selectivity=%d%% p95=%s samples=%d", 100/modulus, p95, len(durations))
		if p95 > 250*time.Millisecond {
			t.Fatalf("modulus %d p95=%s exceeds 250ms SLO", modulus, p95)
		}
	}
}

func TestSearchFilteredMetrics(t *testing.T) {
	base := t.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 4, MaxK: 10, SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal.log"), VectorPath: filepath.Join(base, "vectors")})
	defer s.Close()
	_ = s.AddVector("x", []float64{1, 0})
	_ = s.AddVector("y", []float64{0, 1})
	filter := func(v index.Vector) bool { return true }
	for _, metric := range []DistanceMetric{MetricL2, MetricCosine, MetricInnerProduct} {
		results, err := s.SearchFilteredMetric([]float64{1, 0}, 1, filter, metric)
		if err != nil || len(results) != 1 || results[0].ID != "x" {
			t.Fatalf("metric %s results=%+v err=%v", metric, results, err)
		}
	}
	if _, err := s.SearchFilteredMetric([]float64{1, 0}, 1, filter, DistanceMetric("manhattan")); !errors.Is(err, ErrInvalidMetric) {
		t.Fatalf("invalid metric error=%v", err)
	}
}

func TestSearchMetricWithoutFilterDoesNotFallBackToL2(t *testing.T) {
	base := t.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 4, MaxK: 10, SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal.log"), VectorPath: filepath.Join(base, "vectors")})
	defer s.Close()
	if err := s.AddVector("cosine-and-dot", []float64{100, 1}); err != nil {
		t.Fatal(err)
	}
	if err := s.AddVector("l2", []float64{1, 1}); err != nil {
		t.Fatal(err)
	}
	for _, metric := range []DistanceMetric{MetricCosine, MetricInnerProduct} {
		results, err := s.SearchFilteredMetric([]float64{1, 0}, 1, nil, metric)
		if err != nil || len(results) != 1 || results[0].ID != "cosine-and-dot" {
			t.Fatalf("metric %s results=%+v err=%v", metric, results, err)
		}
	}
	if stats := s.Stats(); stats.ANNMetricIndexes != 0 {
		t.Fatalf("exact mode unexpectedly built metric ANN indexes: %+v", stats)
	}
}

func TestMetricANNTracksAddsDeletesAndStats(t *testing.T) {
	base := t.TempDir()
	s := NewService(ServiceOptions{
		MaxVectorDim: 8, MaxK: 100, SearchMode: "ann",
		SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal.log"),
		VectorPath: filepath.Join(base, "vectors"), ANNSegmentMaxNodes: 2,
		ANNOptions: ann.Options{M: 8, EfConstruction: 32, EfSearch: 64},
	})
	defer s.Close()
	for id, values := range map[string][]float64{
		"first": {100, 1}, "l2": {1, 1}, "other": {0, 1},
	} {
		if err := s.AddVector(id, values); err != nil {
			t.Fatal(err)
		}
	}
	for _, metric := range []DistanceMetric{MetricCosine, MetricInnerProduct} {
		if results, err := s.SearchFilteredMetric([]float64{1, 0}, 1, nil, metric); err != nil ||
			len(results) != 1 || results[0].ID != "first" {
			t.Fatalf("initial metric=%s results=%+v err=%v", metric, results, err)
		}
	}
	if stats := s.Stats(); stats.ANNMetricIndexes != 2 || stats.ANNMetricNodes != 6 ||
		stats.ANNMetricStructuralBytes == 0 || stats.ANNStructuralBytes < stats.ANNMetricStructuralBytes {
		t.Fatalf("metric ANN stats=%+v", stats)
	}
	if err := s.AddVector("new-winner", []float64{1000, 0}); err != nil {
		t.Fatal(err)
	}
	for _, metric := range []DistanceMetric{MetricCosine, MetricInnerProduct} {
		results, err := s.SearchFilteredMetric([]float64{1, 0}, 1, nil, metric)
		if err != nil || len(results) != 1 || results[0].ID != "new-winner" {
			t.Fatalf("after add metric=%s results=%+v err=%v", metric, results, err)
		}
	}
	if err := s.DeleteVector("new-winner"); err != nil {
		t.Fatal(err)
	}
	for _, metric := range []DistanceMetric{MetricCosine, MetricInnerProduct} {
		results, err := s.SearchFilteredMetric([]float64{1, 0}, 1, nil, metric)
		if err != nil || len(results) != 1 || results[0].ID != "first" {
			t.Fatalf("after delete metric=%s results=%+v err=%v", metric, results, err)
		}
	}
	if err := s.AddVectors32([]vectorBatch32{{ID: "compact-winner", Values: []float32{2000, 0}}}); err != nil {
		t.Fatal(err)
	}
	for _, metric := range []DistanceMetric{MetricCosine, MetricInnerProduct} {
		results, err := s.SearchFilteredMetric([]float64{1, 0}, 1, nil, metric)
		if err != nil || len(results) != 1 || results[0].ID != "compact-winner" {
			t.Fatalf("after float32 add metric=%s results=%+v err=%v", metric, results, err)
		}
	}
}

func TestMetricANNRecallAndP95(t *testing.T) {
	const (
		vectorCount = 10000
		dimension   = 32
		queryCount  = 40
		topK        = 10
	)
	base := t.TempDir()
	s := NewService(ServiceOptions{
		MaxVectorDim: 64, MaxK: 500, SearchMode: "ann",
		SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal.log"),
		VectorStore: "segment", VectorPath: filepath.Join(base, "vectors"),
		ANNSegmentMaxNodes: 1000,
		ANNOptions:         ann.Options{M: 16, EfConstruction: 128, EfSearch: 512, DiversifiedPruning: true},
		ANNAdaptive:        true,
		ANNMinCandidates:   200,
	})
	defer s.Close()
	random := rand.New(rand.NewSource(84))
	vectors := make([]index.Vector, vectorCount)
	for id := range vectors {
		values := make([]float64, dimension)
		scale := 0.5 + float64(id%17)/8
		for d := range values {
			values[d] = random.NormFloat64() * scale
		}
		vectors[id] = index.Vector{ID: fmt.Sprintf("metric-%05d", id), Values: values}
	}
	if err := s.AddVectors(vectors); err != nil {
		t.Fatal(err)
	}

	for _, metric := range []DistanceMetric{MetricL2, MetricCosine, MetricInnerProduct} {
		t.Run(string(metric), func(t *testing.T) {
			// Build the lazy generation outside the latency sample.
			if _, err := s.SearchFilteredMetric(vectors[0].Values, topK, nil, metric); err != nil {
				t.Fatal(err)
			}
			recallSum := 0.0
			latencies := make([]time.Duration, 0, queryCount)
			for queryIndex := 0; queryIndex < queryCount; queryIndex++ {
				query := append([]float64(nil), vectors[(queryIndex*71)%vectorCount].Values...)
				for d := range query {
					query[d] += random.NormFloat64() * 0.001
				}
				scorer := newMultiVectorQuery(query, metric)
				truth := make([]SearchResult, vectorCount)
				for position, vector := range vectors {
					truth[position] = SearchResult{ID: vector.ID, Distance: scorer.distance(vector.Values)}
				}
				sort.Slice(truth, func(left, right int) bool {
					if truth[left].Distance == truth[right].Distance {
						return truth[left].ID < truth[right].ID
					}
					return truth[left].Distance < truth[right].Distance
				})
				const searchesPerSample = 5
				started := time.Now()
				var results []SearchResult
				for repeat := 0; repeat < searchesPerSample; repeat++ {
					var err error
					results, err = s.SearchFilteredMetric(query, topK, nil, metric)
					if err != nil {
						t.Fatal(err)
					}
				}
				latencies = append(latencies, time.Since(started)/searchesPerSample)
				expected := make(map[string]struct{}, topK)
				for _, result := range truth[:topK] {
					expected[result.ID] = struct{}{}
				}
				matched := 0
				for _, result := range results {
					if _, ok := expected[result.ID]; ok {
						matched++
					}
				}
				recallSum += float64(matched) / topK
			}
			sort.Slice(latencies, func(left, right int) bool { return latencies[left] < latencies[right] })
			recall := recallSum / queryCount
			p95 := latencies[(len(latencies)*95+99)/100-1]
			if recall < 0.95 {
				t.Fatalf("Recall@10=%.4f, want >=0.95", recall)
			}
			if p95 > 250*time.Millisecond {
				t.Fatalf("p95=%s, want <=250ms", p95)
			}
			t.Logf("Recall@10=%.4f p95=%s", recall, p95)
		})
	}
}

func TestMetricANNRebuildsAfterRestart(t *testing.T) {
	base := t.TempDir()
	opts := ServiceOptions{
		MaxVectorDim: 8, MaxK: 100, SearchMode: "ann",
		SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal.log"),
		ANNOptions: ann.Options{M: 8, EfConstruction: 32, EfSearch: 64},
	}
	first := NewService(opts)
	if err := first.AddVectors([]index.Vector{
		{ID: "winner", Values: []float64{100, 1}},
		{ID: "l2", Values: []float64{1, 1}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := first.SearchFilteredMetric([]float64{1, 0}, 1, nil, MetricCosine); err != nil {
		t.Fatal(err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}

	reopened := NewService(opts)
	defer reopened.Close()
	if stats := reopened.Stats(); stats.ANNMetricIndexes != 0 {
		t.Fatalf("derived metric indexes must start lazy after restart: %+v", stats)
	}
	results, err := reopened.SearchFilteredMetric([]float64{1, 0}, 1, nil, MetricCosine)
	if err != nil || len(results) != 1 || results[0].ID != "winner" {
		t.Fatalf("results=%+v err=%v", results, err)
	}
	if stats := reopened.Stats(); stats.ANNMetricIndexes != 1 || stats.ANNMetricNodes != 2 {
		t.Fatalf("rebuilt metric ANN stats=%+v", stats)
	}
}

func TestMetricANNWarmupBuildsBeforeQuery(t *testing.T) {
	base := t.TempDir()
	s := NewService(ServiceOptions{
		MaxVectorDim: 8, MaxK: 100, SearchMode: "ann",
		SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal.log"),
		ANNOptions: ann.Options{M: 8, EfConstruction: 32, EfSearch: 64},
	})
	defer s.Close()
	if err := s.AddVectors([]index.Vector{
		{ID: "one", Values: []float64{1, 0}},
		{ID: "two", Values: []float64{0, 1}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := s.WarmMetricANN([]DistanceMetric{MetricL2, MetricCosine, MetricInnerProduct}); err != nil {
		t.Fatal(err)
	}
	if stats := s.Stats(); stats.ANNMetricIndexes != 2 || stats.ANNMetricNodes != 4 {
		t.Fatalf("warmup stats=%+v", stats)
	}
	if err := s.WarmMetricANN([]DistanceMetric{"manhattan"}); !errors.Is(err, ErrInvalidMetric) {
		t.Fatalf("invalid warmup metric error=%v", err)
	}
}

func TestMetricANNConcurrentSearchAndIngest(t *testing.T) {
	base := t.TempDir()
	s := NewService(ServiceOptions{
		MaxVectorDim: 8, MaxK: 100, SearchMode: "ann",
		SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal.log"),
		ANNOptions: ann.Options{M: 8, EfConstruction: 32, EfSearch: 64},
	})
	defer s.Close()
	if err := s.AddVector("seed", []float64{1, 0, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.SearchFilteredMetric([]float64{1, 0, 0, 0}, 1, nil, MetricCosine); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for worker := 0; worker < 3; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for iteration := 0; iteration < 50; iteration++ {
				if _, err := s.SearchFilteredMetric([]float64{1, 0, 0, 0}, 5, nil, MetricCosine); err != nil {
					t.Errorf("search: %v", err)
					return
				}
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for iteration := 0; iteration < 50; iteration++ {
			if err := s.AddVector(fmt.Sprintf("concurrent-%03d", iteration), []float64{float64(iteration + 2), 1, 0, 0}); err != nil {
				t.Errorf("add: %v", err)
				return
			}
		}
	}()
	wg.Wait()
	if stats := s.Stats(); stats.ANNMetricNodes != 51 {
		t.Fatalf("metric nodes=%d want=51", stats.ANNMetricNodes)
	}
}

func TestSearchMultiVectorAndReranking(t *testing.T) {
	base := t.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 10, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log"), VectorPath: filepath.Join(base, "v")})
	defer s.Close()
	for id, values := range map[string][]float64{"near-a": {1, 0}, "near-b": {0, 1}, "far": {9, 9}} {
		if err := s.AddVector(id, values); err != nil {
			t.Fatal(err)
		}
	}
	hits, err := s.SearchMultiVector([][]float64{{1, 0}, {0, 1}}, 2, nil, MetricL2)
	if err != nil || len(hits) != 2 || hits[0].ID == "far" {
		t.Fatalf("hits=%+v err=%v", hits, err)
	}
	reranked, err := s.SearchFilteredReranked([]float64{1, 0}, 1, 3, nil, MetricL2, func(hit SearchResult) float64 {
		if hit.ID == "far" {
			return -1
		}
		return hit.Distance
	})
	if err != nil || len(reranked) != 1 || reranked[0].ID != "far" {
		t.Fatalf("reranked=%+v err=%v", reranked, err)
	}
}

func TestStructuredFilter(t *testing.T) {
	f := StructuredFilter{IDs: map[string]struct{}{"Article-1": {}}, TextQuery: "article"}
	if !f.Match(index.Vector{ID: "Article-1"}) {
		t.Fatal("expected structured filter match")
	}
	if f.Match(index.Vector{ID: "image-1"}) {
		t.Fatal("unexpected structured filter match")
	}
}

func TestSearchFilteredMetadata(t *testing.T) {
	base := t.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 5, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log"), VectorPath: filepath.Join(base, "v")})
	defer s.Close()
	if err := s.AddVectorWithMetadata("a", []float64{1, 0}, map[string]string{"tenant": "one"}); err != nil {
		t.Fatal(err)
	}
	if err := s.AddVectorWithMetadata("b", []float64{1, 0}, map[string]string{"tenant": "two"}); err != nil {
		t.Fatal(err)
	}
	results, err := s.SearchFiltered([]float64{1, 0}, 5, StructuredFilter{Metadata: map[string]string{"tenant": "one"}}.Match)
	if err != nil || len(results) != 1 || results[0].ID != "a" {
		t.Fatalf("metadata results=%+v err=%v", results, err)
	}
}

func TestSearchStructuredMetadataCandidateIntersection(t *testing.T) {
	base := t.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 5, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log"), VectorPath: filepath.Join(base, "v")})
	defer s.Close()
	_ = s.AddVectorWithMetadata("match", []float64{1, 0}, map[string]string{"tenant": "one", "tier": "hot"})
	_ = s.AddVectorWithMetadata("wrong-tier", []float64{1, 0}, map[string]string{"tenant": "one", "tier": "cold"})
	_ = s.AddVectorWithMetadata("wrong-tenant", []float64{1, 0}, map[string]string{"tenant": "two", "tier": "hot"})
	results, err := s.SearchStructured([]float64{1, 0}, 5, StructuredFilter{Metadata: map[string]string{"tenant": "one", "tier": "hot"}}, MetricL2)
	if err != nil || len(results) != 1 || results[0].ID != "match" {
		t.Fatalf("results=%+v err=%v", results, err)
	}
}

func TestSearchStructuredIntersectsTextIDsAndMetadataWithoutMutatingFilter(t *testing.T) {
	base := t.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 5, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log"), VectorPath: filepath.Join(base, "v")})
	defer s.Close()
	_ = s.AddVectorWithMetadata("raft-hot", []float64{1, 0}, map[string]string{"tier": "hot"})
	_ = s.AddVectorWithMetadata("raft-cold", []float64{1, 0}, map[string]string{"tier": "cold"})
	_ = s.AddVectorWithMetadata("storage-hot", []float64{1, 0}, map[string]string{"tier": "hot"})
	ids := map[string]struct{}{"raft-hot": {}, "raft-cold": {}, "storage-hot": {}}
	results, err := s.SearchStructured([]float64{1, 0}, 5, StructuredFilter{
		IDs:       ids,
		TextQuery: "raft",
		Metadata:  map[string]string{"tier": "hot"},
	}, MetricL2)
	if err != nil || len(results) != 1 || results[0].ID != "raft-hot" {
		t.Fatalf("results=%+v err=%v", results, err)
	}
	if len(ids) != 3 {
		t.Fatalf("caller ID filter was mutated: %v", ids)
	}
}

func TestMetadataSidecarPersists(t *testing.T) {
	base := t.TempDir()
	opts := ServiceOptions{MaxVectorDim: 2, MaxK: 5, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log"), VectorPath: filepath.Join(base, "v")}
	s := NewService(opts)
	_ = s.AddVectorWithMetadata("persisted", []float64{1, 0}, map[string]string{"tenant": "one"})
	_ = s.Close()
	reopened := NewService(opts)
	defer reopened.Close()
	results, err := reopened.SearchFiltered([]float64{1, 0}, 5, StructuredFilter{Metadata: map[string]string{"tenant": "one"}}.Match)
	if err != nil || len(results) != 1 || results[0].ID != "persisted" {
		t.Fatalf("restored metadata results=%+v err=%v", results, err)
	}
	textResults, err := reopened.SearchStructured([]float64{1, 0}, 5, StructuredFilter{TextQuery: "tenant"}, MetricL2)
	if err != nil || len(textResults) != 1 || textResults[0].ID != "persisted" {
		t.Fatalf("restored full-text results=%+v err=%v", textResults, err)
	}
}

func TestMetadataWALReplayRestoresWithoutSidecar(t *testing.T) {
	base := t.TempDir()
	snapshot := filepath.Join(base, "s.json")
	wal := filepath.Join(base, "w.log")
	s := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 5, SnapshotPath: snapshot, WALPath: wal})
	if err := s.AddVectorWithMetadata("wal-meta", []float64{1, 0}, map[string]string{"tenant": "replayed"}); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	_ = os.Remove(snapshot + ".metadata.json")
	reopened := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 5, SnapshotPath: snapshot, WALPath: wal})
	defer reopened.Close()
	results, err := reopened.SearchFiltered([]float64{1, 0}, 5, StructuredFilter{Metadata: map[string]string{"tenant": "replayed"}}.Match)
	if err != nil || len(results) != 1 || results[0].ID != "wal-meta" {
		t.Fatalf("replayed metadata=%+v err=%v", results, err)
	}
}

func TestDeleteRemovesMetadata(t *testing.T) {
	base := t.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 5, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log"), VectorPath: filepath.Join(base, "v")})
	defer s.Close()
	_ = s.AddVectorWithMetadata("gone", []float64{1, 0}, map[string]string{"tenant": "one"})
	if err := s.DeleteVector("gone"); err != nil {
		t.Fatal(err)
	}
	results, err := s.SearchFiltered([]float64{1, 0}, 5, StructuredFilter{Metadata: map[string]string{"tenant": "one"}}.Match)
	if err != nil || len(results) != 0 {
		t.Fatalf("stale metadata results=%+v err=%v", results, err)
	}
}

func TestSearchStructuredExactMetadataTracksUpdateAndDelete(t *testing.T) {
	base := t.TempDir()
	s := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 5, SnapshotPath: filepath.Join(base, "s.json"), WALPath: filepath.Join(base, "w.log"), VectorPath: filepath.Join(base, "v")})
	defer s.Close()
	if err := s.AddVectorWithMetadata("moving", []float64{1, 0}, map[string]string{"tenant": "old"}); err != nil {
		t.Fatal(err)
	}
	if err := s.DeleteVector("moving"); err != nil {
		t.Fatal(err)
	}
	if err := s.AddVectorWithMetadata("moving", []float64{1, 0}, map[string]string{"tenant": "new"}); err != nil {
		t.Fatal(err)
	}
	oldResults, err := s.SearchStructured([]float64{1, 0}, 5, StructuredFilter{Metadata: map[string]string{"tenant": "old"}}, MetricL2)
	if err != nil || len(oldResults) != 0 {
		t.Fatalf("stale update results=%+v err=%v", oldResults, err)
	}
	newResults, err := s.SearchStructured([]float64{1, 0}, 5, StructuredFilter{Metadata: map[string]string{"tenant": "new"}}, MetricL2)
	if err != nil || len(newResults) != 1 || newResults[0].ID != "moving" {
		t.Fatalf("updated results=%+v err=%v", newResults, err)
	}
	if err := s.DeleteVector("moving"); err != nil {
		t.Fatal(err)
	}
	deletedResults, err := s.SearchStructured([]float64{1, 0}, 5, StructuredFilter{Metadata: map[string]string{"tenant": "new"}}, MetricL2)
	if err != nil || len(deletedResults) != 0 {
		t.Fatalf("deleted results=%+v err=%v", deletedResults, err)
	}
}

func TestCloneMetadataKeepsEmptyMetadataNil(t *testing.T) {
	if got := cloneMetadata(nil); got != nil {
		t.Fatalf("expected nil metadata, got %#v", got)
	}
	if got := cloneMetadata(map[string]string{}); got != nil {
		t.Fatalf("expected nil empty metadata, got %#v", got)
	}
}
