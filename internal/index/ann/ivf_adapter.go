package ann

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// IVFANNIndex adapts IVFIndex to the ANNIndex contract used by core.Service.
// Deletes are logical so immutable IVF lists remain compactable later.
type IVFANNIndex struct {
	mu      sync.RWMutex
	index   *IVFIndex
	deleted map[int]struct{}
	known   map[int]struct{}
	nodes   int
}

type IVFVector struct {
	ID     int
	Values []float64
}

func NewIVFANNIndex(centroids [][]float64, nprobe int) (*IVFANNIndex, error) {
	index, err := NewIVFIndex(centroids, nprobe)
	if err != nil {
		return nil, err
	}
	return &IVFANNIndex{index: index, deleted: make(map[int]struct{}), known: make(map[int]struct{})}, nil
}

// NewIVFANNIndexFromSamples trains deterministic centroids and returns a
// ready-to-inject adapter. Samples are only used for coarse partitioning;
// callers can then add the complete dataset incrementally.
func NewIVFANNIndexFromSamples(samples [][]float64, centroidCount, nprobe int) (*IVFANNIndex, error) {
	centroids, err := TrainIVFCentroids(samples, centroidCount)
	if err != nil {
		return nil, err
	}
	return NewIVFANNIndex(centroids, nprobe)
}

// Retrain rebuilds the coarse partition and swaps it only after all entries
// have been accepted, making distribution changes safe for callers.
func (i *IVFANNIndex) Retrain(entries []IVFVector, centroidCount, nprobe int) error {
	if i == nil {
		return fmt.Errorf("index is nil")
	}
	samples := make([][]float64, len(entries))
	for n, entry := range entries {
		samples[n] = entry.Values
	}
	next, err := NewIVFANNIndexFromSamples(samples, centroidCount, nprobe)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if err := next.AddVector(entry.ID, entry.Values); err != nil {
			return err
		}
	}
	i.mu.Lock()
	i.index, i.deleted, i.known, i.nodes = next.index, next.deleted, next.known, next.nodes
	i.mu.Unlock()
	return nil
}

// RunRetrainLoop periodically obtains a consistent entry snapshot and
// rebuilds the partitions until cancellation. A non-positive interval is
// rejected to prevent a hot loop from competing with queries.
func (i *IVFANNIndex) RunRetrainLoop(ctx context.Context, interval time.Duration, source func() []IVFVector, centroidCount, nprobe int) error {
	if ctx == nil || source == nil {
		return fmt.Errorf("context and source are required")
	}
	if interval <= 0 {
		return fmt.Errorf("retrain interval must be positive")
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	retrain := func() error { return i.Retrain(source(), centroidCount, nprobe) }
	if err := retrain(); err != nil {
		return err
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := retrain(); err != nil {
				return err
			}
		}
	}
}

// RunDriftAwareLoop samples entries periodically and retrains only when the
// configured mean centroid-distance threshold is exceeded.
func (i *IVFANNIndex) RunDriftAwareLoop(ctx context.Context, interval time.Duration, source func() []IVFVector, threshold float64, centroidCount, nprobe int) error {
	if ctx == nil || source == nil {
		return fmt.Errorf("context and source are required")
	}
	if interval <= 0 || threshold < 0 {
		return fmt.Errorf("positive interval and non-negative threshold are required")
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	check := func() error {
		entries := source()
		needs, err := i.NeedsRetrain(entries, threshold)
		if err != nil {
			return err
		}
		if needs {
			return i.Retrain(entries, centroidCount, nprobe)
		}
		return nil
	}
	if err := check(); err != nil {
		return err
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := check(); err != nil {
				return err
			}
		}
	}
}

// DriftScore returns the mean squared distance from entries to their nearest
// current centroid. NeedsRetrain compares this score to an operator threshold.
func (i *IVFANNIndex) DriftScore(entries []IVFVector) (float64, error) {
	if i == nil || len(entries) == 0 {
		return 0, fmt.Errorf("index and entries are required")
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	var total float64
	for _, entry := range entries {
		if len(entry.Values) != len(i.index.centroids[0]) {
			return 0, fmt.Errorf("entry dimension mismatch")
		}
		total += l2(entry.Values, i.index.centroids[i.index.closestCentroid(entry.Values)])
	}
	return total / float64(len(entries)), nil
}

func (i *IVFANNIndex) NeedsRetrain(entries []IVFVector, threshold float64) (bool, error) {
	if threshold < 0 {
		return false, fmt.Errorf("threshold must be non-negative")
	}
	score, err := i.DriftScore(entries)
	return score > threshold, err
}
func (i *IVFANNIndex) AddVector(id int, vector []float64) error {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i == nil {
		return fmt.Errorf("index is nil")
	}
	if err := i.index.Add(id, vector); err != nil {
		return err
	}
	delete(i.deleted, id)
	if _, exists := i.known[id]; !exists {
		i.known[id] = struct{}{}
		i.nodes++
	}
	return nil
}
func (i *IVFANNIndex) SearchWithDistancesInto(query []float64, k int, dst []Result) ([]Result, error) {
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i == nil {
		return nil, fmt.Errorf("index is nil")
	}
	hits, err := i.index.Search(query, k+len(i.deleted))
	if err != nil {
		return nil, err
	}
	out := dst[:0]
	for _, hit := range hits {
		if _, ok := i.deleted[hit.ID]; ok {
			continue
		}
		out = append(out, hit)
		if len(out) == k {
			break
		}
	}
	return out, nil
}

// SearchAdaptiveWithDistancesInto exposes the adaptive IVF probe policy while
// retaining the ANNIndex-compatible SearchWithDistancesInto method unchanged.
func (i *IVFANNIndex) SearchAdaptiveWithDistancesInto(query []float64, k, minCandidates, maxProbe int, dst []Result) ([]Result, error) {
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i == nil {
		return nil, fmt.Errorf("index is nil")
	}
	hits, err := i.index.SearchAdaptive(query, k+len(i.deleted), minCandidates, maxProbe)
	if err != nil {
		return nil, err
	}
	out := dst[:0]
	for _, hit := range hits {
		if _, ok := i.deleted[hit.ID]; ok {
			continue
		}
		out = append(out, hit)
		if len(out) == k {
			break
		}
	}
	return out, nil
}
func (i *IVFANNIndex) DeleteVector(id int) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i != nil {
		i.deleted[id] = struct{}{}
	}
}
func (i *IVFANNIndex) Stats() Stats {
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i == nil {
		return Stats{}
	}
	return Stats{Nodes: i.nodes, Deleted: len(i.deleted)}
}
