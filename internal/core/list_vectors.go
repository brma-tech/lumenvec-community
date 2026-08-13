package core

import (
	"container/heap"
	"sort"

	"lumenvec/internal/index"
)

type ListVectorsOptions struct {
	AfterID string
	Limit   int
	IDsOnly bool
}

type ListVectorsPage struct {
	Vectors    []index.Vector
	NextCursor string
}

type vectorIDRanger interface {
	RangeVectorIDs(fn func(string) bool)
}

type vectorIDPager interface {
	PageVectorIDs(afterID string, limit int) []string
}

func (s *Service) ListVectorsPage(opts ListVectorsOptions) ListVectorsPage {
	s.ensureRuntimeDeps()
	if opts.Limit <= 0 {
		return ListVectorsPage{}
	}

	pageIDs := s.selectVectorPageIDs(opts.AfterID, opts.Limit+1)
	hasMore := len(pageIDs) > opts.Limit
	if hasMore {
		pageIDs = pageIDs[:opts.Limit]
	}

	vectors := make([]index.Vector, 0, len(pageIDs))
	if opts.IDsOnly {
		for _, id := range pageIDs {
			vectors = append(vectors, index.Vector{ID: id})
		}
	} else {
		for _, id := range pageIDs {
			vec, err := s.vectorStore.GetVector(id)
			if err != nil {
				continue
			}
			vectors = append(vectors, vec)
		}
	}

	nextCursor := ""
	if hasMore && len(pageIDs) > 0 {
		nextCursor = pageIDs[len(pageIDs)-1]
	}
	return ListVectorsPage{Vectors: vectors, NextCursor: nextCursor}
}

func (s *Service) selectVectorPageIDs(afterID string, limit int) []string {
	if limit <= 0 {
		return nil
	}
	if pager, ok := s.vectorStore.(vectorIDPager); ok {
		return pager.PageVectorIDs(afterID, limit)
	}

	return selectPageIDsFromRange(afterID, limit, s.rangeVectorIDs)
}

func selectPageIDsFromRange(afterID string, limit int, rangeIDs func(func(string) bool)) []string {
	if limit <= 0 {
		return nil
	}
	candidates := &stringMaxHeap{}
	heap.Init(candidates)
	rangeIDs(func(id string) bool {
		if afterID != "" && id <= afterID {
			return true
		}
		if candidates.Len() < limit {
			heap.Push(candidates, id)
			return true
		}
		if id < (*candidates)[0] {
			(*candidates)[0] = id
			heap.Fix(candidates, 0)
		}
		return true
	})

	out := make([]string, candidates.Len())
	copy(out, *candidates)
	sort.Strings(out)
	return out
}

func (s *Service) rangeVectorIDs(fn func(string) bool) {
	if ranger, ok := s.vectorStore.(vectorIDRanger); ok {
		ranger.RangeVectorIDs(fn)
		return
	}
	for _, vec := range s.vectorStore.ListVectors() {
		if !fn(vec.ID) {
			return
		}
	}
}

type stringMaxHeap []string

func (h stringMaxHeap) Len() int           { return len(h) }
func (h stringMaxHeap) Less(i, j int) bool { return h[i] > h[j] }
func (h stringMaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *stringMaxHeap) Push(x any) {
	*h = append(*h, x.(string))
}

func (h *stringMaxHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}
