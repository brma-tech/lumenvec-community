package ann

import (
	"context"
	"fmt"
	"math"
	"sort"
)

type ivfEntry struct {
	id     int
	vector []float64
}

// IVFIndex is a deterministic inverted-file candidate index. Centroids are
// trained from a supplied sample; each list is reranked exactly during search.
type IVFIndex struct {
	centroids [][]float64
	lists     [][]ivfEntry
	// positions keeps upserts constant-time instead of scanning the target
	// list. The value is the current list and offset for the vector ID.
	positions map[int]ivfPosition
	nprobe    int
}

type ivfPosition struct {
	list  int
	index int
}

// TrainIVFCentroids selects k deterministic, evenly spaced samples. It is a
// lightweight bootstrap; production deployments can replace it with k-means
// without changing the IVFIndex contract.
func TrainIVFCentroids(vectors [][]float64, k int) ([][]float64, error) {
	if len(vectors) == 0 || k <= 0 {
		return nil, fmt.Errorf("vectors and positive k are required")
	}
	if k > len(vectors) {
		k = len(vectors)
	}
	dim := len(vectors[0])
	if dim == 0 {
		return nil, fmt.Errorf("vector dimension is required")
	}
	centroids := make([][]float64, k)
	for n := 0; n < k; n++ {
		index := n * (len(vectors) - 1) / maxInt(k-1, 1)
		if len(vectors[index]) != dim {
			return nil, fmt.Errorf("vector dimension mismatch")
		}
		centroids[n] = append([]float64(nil), vectors[index]...)
	}
	return centroids, nil
}

// TrainIVFCentroidsKMeans trains deterministic coarse centroids over a bounded
// sample. It keeps startup work bounded for very large collections while
// avoiding the severe recall loss of evenly spaced bootstrap points.
func TrainIVFCentroidsKMeans(vectors [][]float64, k, iterations, sampleLimit int) ([][]float64, error) {
	if len(vectors) == 0 || k <= 0 || iterations <= 0 {
		return nil, fmt.Errorf("vectors, k and iterations are required")
	}
	if sampleLimit <= 0 || sampleLimit > len(vectors) {
		sampleLimit = len(vectors)
	}
	dim := len(vectors[0])
	if dim == 0 {
		return nil, fmt.Errorf("vector dimension is required")
	}
	if k > sampleLimit {
		k = sampleLimit
	}
	samples := make([][]float64, sampleLimit)
	for n := range samples {
		idx := n * (len(vectors) - 1) / maxInt(sampleLimit-1, 1)
		if len(vectors[idx]) != dim {
			return nil, fmt.Errorf("vector dimension mismatch")
		}
		samples[n] = vectors[idx]
	}
	centroids := make([][]float64, k)
	centroids[0] = append([]float64(nil), samples[0]...)
	nearest := make([]float64, len(samples))
	for n := range nearest {
		nearest[n] = l2(samples[n], centroids[0])
	}
	for c := 1; c < k; c++ {
		best := 0
		for n := 1; n < len(samples); n++ {
			if nearest[n] > nearest[best] {
				best = n
			}
		}
		centroids[c] = append([]float64(nil), samples[best]...)
		for n := range samples {
			if d := l2(samples[n], centroids[c]); d < nearest[n] {
				nearest[n] = d
			}
		}
	}
	for iter := 0; iter < iterations; iter++ {
		sums := make([][]float64, k)
		counts := make([]int, k)
		for _, sample := range samples {
			best, distance := 0, math.Inf(1)
			for c, centroid := range centroids {
				if d := l2(sample, centroid); d < distance {
					best, distance = c, d
				}
			}
			counts[best]++
			if sums[best] == nil {
				sums[best] = make([]float64, dim)
			}
			for i, value := range sample {
				sums[best][i] += value
			}
		}
		for c := range centroids {
			if counts[c] == 0 {
				continue
			}
			for i := range centroids[c] {
				centroids[c][i] = sums[c][i] / float64(counts[c])
			}
		}
	}
	return centroids, nil
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func NewIVFIndex(centroids [][]float64, nprobe int) (*IVFIndex, error) {
	if len(centroids) == 0 || len(centroids[0]) == 0 {
		return nil, fmt.Errorf("centroids are required")
	}
	if nprobe <= 0 {
		nprobe = 1
	}
	if nprobe > len(centroids) {
		nprobe = len(centroids)
	}
	dim := len(centroids[0])
	copied := make([][]float64, len(centroids))
	for i, c := range centroids {
		if len(c) != dim {
			return nil, fmt.Errorf("centroid dimension mismatch")
		}
		copied[i] = append([]float64(nil), c...)
	}
	return &IVFIndex{centroids: copied, lists: make([][]ivfEntry, len(copied)), positions: make(map[int]ivfPosition), nprobe: nprobe}, nil
}

func (i *IVFIndex) Add(id int, vector []float64) error {
	if i == nil || len(vector) != len(i.centroids[0]) {
		return fmt.Errorf("vector dimension mismatch")
	}
	list := i.closestCentroid(vector)
	entry := ivfEntry{id: id, vector: append([]float64(nil), vector...)}
	if previous, ok := i.positions[id]; ok {
		if previous.list == list {
			i.lists[list][previous.index] = entry
			return nil
		}
		// The closest centroid can change after an update. Remove the old
		// entry and repair offsets for the compacted tail.
		oldList := i.lists[previous.list]
		last := len(oldList) - 1
		oldList[previous.index] = oldList[last]
		i.positions[oldList[previous.index].id] = ivfPosition{list: previous.list, index: previous.index}
		i.lists[previous.list] = oldList[:last]
	}
	i.lists[list] = append(i.lists[list], entry)
	i.positions[id] = ivfPosition{list: list, index: len(i.lists[list]) - 1}
	return nil
}

func (i *IVFIndex) Search(query []float64, k int) ([]Result, error) {
	return i.searchWithProbes(query, k, i.nprobe)
}

func (i *IVFIndex) SearchContext(ctx context.Context, query []float64, k int) ([]Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	hits, _, err := i.searchWithProbesCount(query, k, i.nprobe, ctx)
	return hits, err
}

// SearchAdaptive progressively probes additional IVF lists when the initial
// candidate set is too small. It is opt-in so existing latency characteristics
// of Search remain unchanged. minCandidates is the desired pre-rerank pool;
// maxProbe is capped to the number of trained centroids.
func (i *IVFIndex) SearchAdaptive(query []float64, k, minCandidates, maxProbe int) ([]Result, error) {
	return i.SearchAdaptiveContext(context.Background(), query, k, minCandidates, maxProbe)
}

func (i *IVFIndex) SearchAdaptiveContext(ctx context.Context, query []float64, k, minCandidates, maxProbe int) ([]Result, error) {
	if i == nil {
		return nil, fmt.Errorf("index is nil")
	}
	if minCandidates <= 0 {
		minCandidates = k
	}
	if maxProbe <= 0 || maxProbe > len(i.centroids) {
		maxProbe = len(i.centroids)
	}
	probes := i.nprobe
	if probes > maxProbe {
		probes = maxProbe
	}
	for {
		hits, candidateCount, err := i.searchWithProbesCount(query, k, probes, ctx)
		if err != nil {
			return nil, err
		}
		// Expand until the candidate pool target is met. This avoids accepting
		// a locally complete top-k from a single list when neighboring lists may
		// contain closer vectors.
		if probes >= maxProbe || (candidateCount >= minCandidates && len(hits) >= k) {
			return hits, nil
		}
		next := probes * 2
		if next <= probes {
			next = probes + 1
		}
		if next > maxProbe {
			next = maxProbe
		}
		probes = next
	}
}

func (i *IVFIndex) searchWithProbes(query []float64, k, probes int) ([]Result, error) {
	hits, _, err := i.searchWithProbesCount(query, k, probes)
	return hits, err
}

func (i *IVFIndex) searchWithProbesCount(query []float64, k, probes int, contexts ...context.Context) ([]Result, int, error) {
	ctx := searchContext(contexts)
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	if i == nil || len(query) != len(i.centroids[0]) {
		return nil, 0, fmt.Errorf("query dimension mismatch")
	}
	if k <= 0 {
		return nil, 0, ErrInvalidK
	}
	order := i.closestCentroids(query, probes, ctx)
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	hits := make([]Result, 0)
	seen := make(map[int]struct{})
	if probes <= 0 {
		probes = 1
	}
	if probes > len(order) {
		probes = len(order)
	}
	for _, list := range order[:probes] {
		for _, entry := range i.lists[list] {
			if ctx.Err() != nil {
				return nil, 0, ctx.Err()
			}
			if _, ok := seen[entry.id]; ok {
				continue
			}
			seen[entry.id] = struct{}{}
			hits = append(hits, Result{ID: entry.id, Distance: l2(query, entry.vector)})
		}
	}
	sort.Slice(hits, func(a, b int) bool {
		if hits[a].Distance == hits[b].Distance {
			return hits[a].ID < hits[b].ID
		}
		return hits[a].Distance < hits[b].Distance
	})
	if len(hits) > k {
		hits = hits[:k]
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	return hits, len(seen), nil
}

// closestCentroids selects only the requested prefix instead of sorting every
// centroid. For the normal small-P regime this changes query-side centroid
// selection from O(C log C) to O(C·P), while preserving exact probe ordering.
func (i *IVFIndex) closestCentroids(query []float64, probes int, contexts ...context.Context) []int {
	ctx := searchContext(contexts)
	if probes <= 0 || probes > len(i.centroids) {
		probes = len(i.centroids)
	}
	selected := make([]int, 0, probes)
	distances := make([]float64, 0, probes)
	for centroidID, centroid := range i.centroids {
		if ctx.Err() != nil {
			return nil
		}
		distance := l2(query, centroid)
		position := len(selected)
		for n := range distances {
			if distance < distances[n] {
				position = n
				break
			}
		}
		if position == len(selected) && len(selected) == probes {
			continue
		}
		if len(selected) < probes {
			selected = append(selected, centroidID)
			distances = append(distances, distance)
		} else {
			selected = append(selected, 0)
			distances = append(distances, 0)
		}
		copy(selected[position+1:], selected[position:len(selected)-1])
		copy(distances[position+1:], distances[position:len(distances)-1])
		selected[position] = centroidID
		distances[position] = distance
		if len(selected) > probes {
			selected = selected[:probes]
			distances = distances[:probes]
		}
	}
	return selected
}

func (i *IVFIndex) closestCentroid(vector []float64) int {
	best := 0
	distance := l2(vector, i.centroids[0])
	for n := 1; n < len(i.centroids); n++ {
		if d := l2(vector, i.centroids[n]); d < distance {
			best, distance = n, d
		}
	}
	return best
}
func l2(a, b []float64) float64 {
	var d float64
	for n := range a {
		delta := a[n] - b[n]
		d += delta * delta
	}
	return d
}
