package core

import (
	"math"
	"sort"
	"strings"

	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"
	"lumenvec/internal/vector"
)

type DistanceMetric string

const (
	MetricL2           DistanceMetric = "l2"
	MetricCosine       DistanceMetric = "cosine"
	MetricInnerProduct DistanceMetric = "inner_product"
)

// VectorFilter is evaluated before distance ranking. It is intentionally
// transport-neutral so API layers can add structured predicates without
// changing the existing Search contract.
type VectorFilter func(index.Vector) bool

// Reranker returns a comparable score; lower scores rank first, matching the
// distance semantics used by SearchResult.
type Reranker func(SearchResult) float64

type StructuredFilter struct {
	IDs       map[string]struct{}
	TextQuery string
	Metadata  map[string]string
}

func (f StructuredFilter) Match(v index.Vector) bool {
	if len(f.IDs) > 0 {
		if _, ok := f.IDs[v.ID]; !ok {
			return false
		}
	}
	for key, expected := range f.Metadata {
		if v.Metadata == nil || v.Metadata[key] != expected {
			return false
		}
	}
	if f.TextQuery == "" {
		return true
	}
	query := strings.ToLower(strings.TrimSpace(f.TextQuery))
	if strings.Contains(strings.ToLower(v.ID), query) {
		return true
	}
	for key, value := range v.Metadata {
		if strings.Contains(strings.ToLower(key), query) || strings.Contains(strings.ToLower(value), query) {
			return true
		}
	}
	return false
}

func (s *Service) SearchFiltered(values []float64, k int, filter VectorFilter) ([]SearchResult, error) {
	return s.SearchFilteredMetric(values, k, filter, MetricL2)
}

func (s *Service) SearchFilteredMetric(values []float64, k int, filter VectorFilter, metric DistanceMetric) ([]SearchResult, error) {
	return s.searchFilteredMetric(values, k, filter, metric, nil)
}

// SearchStructured uses the inverted text index to narrow candidates before
// evaluating the structured predicate and vector distance.
func (s *Service) SearchStructured(values []float64, k int, filter StructuredFilter, metric DistanceMetric) ([]SearchResult, error) {
	if err := validateDistanceMetric(metric); err != nil {
		return nil, err
	}
	// Rebuild once after WAL/sidecar replay; subsequent writes update it
	// incrementally in AddVectors/AddVectorWithMetadata.
	textQuery := strings.ToLower(strings.TrimSpace(filter.TextQuery))
	if textQuery != "" {
		s.textIndexOnce.Do(func() {
			indexVector := func(id string) {
				s.withMetadataReadOnly(id, func(metadata map[string]string) bool {
					var local [9]string
					capacity := 1 + len(metadata)*2
					values := local[:1]
					values[0] = id
					if capacity > len(local) {
						values = make([]string, 1, capacity)
						values[0] = id
					}
					for key, value := range metadata {
						values = append(values, key, value)
					}
					s.textIndex.add(id, values...)
					s.textIndex.addMetadata(id, metadata)
					return true
				})
			}
			if ids, ok := s.vectorStore.(rangeVectorIDReader); ok {
				ids.RangeVectorIDs(func(id string) bool { indexVector(id); return true })
			} else {
				for _, vec := range s.ListVectors() {
					indexVector(vec.ID)
				}
			}
			s.textIndexReady.Store(true)
		})
	}
	matcher := func(v index.Vector) bool {
		if len(filter.IDs) > 0 {
			if _, ok := filter.IDs[v.ID]; !ok {
				return false
			}
		}
		for key, expected := range filter.Metadata {
			if v.Metadata == nil || v.Metadata[key] != expected {
				return false
			}
		}
		if textQuery == "" {
			return true
		}
		if strings.Contains(strings.ToLower(v.ID), textQuery) {
			return true
		}
		for key, value := range v.Metadata {
			if strings.Contains(strings.ToLower(key), textQuery) || strings.Contains(strings.ToLower(value), textQuery) {
				return true
			}
		}
		return false
	}
	if textQuery == "" && len(filter.IDs) == 0 && len(filter.Metadata) == 1 {
		if selected, ok := s.vectorStore.(selectedVectorIDs32Reader); ok {
			for key, expected := range filter.Metadata {
				ids := s.textIndex.searchMetadataIDs(key, expected)
				return s.searchStructuredDirectIDs(values, k, metric, ids, selected)
			}
		}
	}
	var candidateIDs map[string]struct{}
	constrained := false
	intersect := func(ids map[string]struct{}) {
		if !constrained {
			candidateIDs = ids
			constrained = true
			return
		}
		for id := range candidateIDs {
			if _, ok := ids[id]; !ok {
				delete(candidateIDs, id)
			}
		}
	}
	if textQuery != "" {
		intersect(s.textIndex.search(filter.TextQuery))
	}
	if len(filter.IDs) > 0 {
		// An explicit ID set is already an exact candidate index; avoid scanning
		// the complete vector store even when no text query is present. Clone
		// it because intersections must not mutate the caller's filter.
		ids := make(map[string]struct{}, len(filter.IDs))
		for id := range filter.IDs {
			ids[id] = struct{}{}
		}
		intersect(ids)
	}
	if len(filter.Metadata) > 0 {
		// Exact key/value postings avoid a full collection scan for structured
		// predicates. Intersect the smallest posting sets first.
		terms := make([]map[string]struct{}, 0, len(filter.Metadata))
		for key, expected := range filter.Metadata {
			terms = append(terms, s.textIndex.searchMetadata(key, expected))
		}
		sort.Slice(terms, func(i, j int) bool { return len(terms[i]) < len(terms[j]) })
		for _, ids := range terms {
			intersect(ids)
		}
	}
	if !constrained && len(filter.Metadata) > 0 {
		if reader, ok := s.vectorStore.(readOnlyVector32Reader); ok {
			if ids, ok := s.vectorStore.(rangeVectorIDReader); ok {
				candidateIDs = make(map[string]struct{})
				ids.RangeVectorIDs(func(id string) bool {
					candidateIDs[id] = struct{}{}
					return true
				})
				return s.searchStructuredDirect(values, k, filter, matcher, metric, candidateIDs, reader)
			}
		}
	}
	if candidateIDs != nil {
		if reader, ok := s.vectorStore.(readOnlyVector32Reader); ok {
			// Explicit IDs and exact metadata postings are authoritative and
			// already implement the complete predicate when no text clause is
			// present. Avoid taking metadataMu and repeating the same map
			// lookups once per candidate in this common selective path.
			if textQuery == "" {
				matcher = nil
			}
			return s.searchStructuredDirect(values, k, filter, matcher, metric, candidateIDs, reader)
		}
	}
	return s.searchFilteredMetric(values, k, matcher, metric, candidateIDs)
}

func (s *Service) searchStructuredDirectIDs(values []float64, k int, metric DistanceMetric, candidates []string, reader selectedVectorIDs32Reader) ([]SearchResult, error) {
	if err := s.validateSearchRequest(values, k); err != nil {
		return nil, err
	}
	query := newMultiVectorQuery(values, metric)
	acc := newTopKAccumulator(k)
	reader.RangeSelectedVectorIDs32(candidates, func(id string, vectorValues []float32) bool {
		acc.Add(SearchResult{ID: id, Distance: query.distance32(vectorValues)})
		return true
	})
	return acc.ResultsRaw(), nil
}

func (s *Service) searchStructuredDirect(values []float64, k int, filter StructuredFilter, matcher VectorFilter, metric DistanceMetric, candidates map[string]struct{}, reader readOnlyVector32Reader) ([]SearchResult, error) {
	if err := s.validateSearchRequest(values, k); err != nil {
		return nil, err
	}
	query := newMultiVectorQuery(values, metric)
	acc := newTopKAccumulator(k)
	visit := func(id string, vectorValues []float32) bool {
		if matcher != nil {
			matched := s.withMetadataReadOnly(id, func(metadata map[string]string) bool {
				return matcher(index.Vector{ID: id, Metadata: metadata})
			})
			if !matched {
				return true
			}
		}
		acc.Add(SearchResult{ID: id, Distance: query.distance32(vectorValues)})
		return true
	}
	if selected, ok := reader.(selectedVector32Reader); ok {
		selected.RangeSelectedVectors32(candidates, visit)
		return acc.ResultsRaw(), nil
	}
	for id := range candidates {
		vectorValues, err := reader.GetVectorReadOnly32(id)
		if err != nil {
			continue
		}
		if !visit(id, vectorValues) {
			break
		}
	}
	return acc.ResultsRaw(), nil
}

func (s *Service) searchFilteredMetric(values []float64, k int, filter VectorFilter, metric DistanceMetric, candidateIDs map[string]struct{}) ([]SearchResult, error) {
	if err := s.validateSearchRequest(values, k); err != nil {
		return nil, err
	}
	if err := validateDistanceMetric(metric); err != nil {
		return nil, err
	}
	if filter == nil && metric == MetricL2 {
		return s.Search(values, k)
	}
	if filter == nil && s.searchMode == "ann" && (metric == MetricCosine || metric == MetricInnerProduct) {
		if results, err := s.searchMetricANN(values, k, metric); err == nil {
			return results, nil
		}
		// The canonical store remains the correctness fallback when an
		// auxiliary ANN generation cannot be built or queried.
	}
	if filter == nil {
		filter = func(index.Vector) bool { return true }
	}
	// Stream the store's native float32 representation. ListVectors materializes
	// and converts the complete collection before filtering, which makes a
	// selective query pay large avoidable allocation and disk-read costs.
	if reader, ok := s.vectorStore.(rangeVector32Reader); ok {
		query := newMultiVectorQuery(values, metric)
		acc := newTopKAccumulator(k)
		var filterValues []float64
		reader.RangeVectors32(func(id string, vectorValues []float32) bool {
			if candidateIDs != nil {
				if _, exists := candidateIDs[id]; !exists {
					return true
				}
			}
			if cap(filterValues) < len(vectorValues) {
				filterValues = make([]float64, len(vectorValues))
			} else {
				filterValues = filterValues[:len(vectorValues)]
			}
			for i, value := range vectorValues {
				filterValues[i] = float64(value)
			}
			matched := s.withMetadataReadOnly(id, func(metadata map[string]string) bool {
				return filter(index.Vector{ID: id, Values: filterValues, Metadata: metadata})
			})
			if matched {
				acc.Add(SearchResult{ID: id, Distance: query.distance32(vectorValues)})
			}
			return true
		})
		return acc.ResultsRaw(), nil
	}
	acc := newTopKAccumulator(k)
	q := make([]float32, len(values))
	for i := range values {
		q[i] = float32(values[i])
	}
	vectors := s.ListVectors()
	var scratch []float32
	for _, vec := range vectors {
		if candidateIDs != nil {
			if _, ok := candidateIDs[vec.ID]; !ok {
				continue
			}
		}
		vec.Metadata = s.metadataFor(vec.ID)
		if !filter(vec) {
			continue
		}
		if cap(scratch) < len(vec.Values) {
			scratch = make([]float32, len(vec.Values))
		} else {
			scratch = scratch[:len(vec.Values)]
		}
		v := scratch
		for i := range vec.Values {
			v[i] = float32(vec.Values[i])
		}
		distance := vector.SquaredEuclideanDistance32SameLen(q, v)
		if metric == MetricCosine || metric == MetricInnerProduct {
			dot, nq, nv := 0.0, 0.0, 0.0
			for i := range q {
				dot += float64(q[i]) * float64(v[i])
				nq += float64(q[i]) * float64(q[i])
				nv += float64(v[i]) * float64(v[i])
			}
			if metric == MetricCosine {
				if nq == 0 || nv == 0 {
					distance = 1
				} else {
					distance = 1 - dot/(math.Sqrt(nq)*math.Sqrt(nv))
				}
			} else {
				distance = -dot
			}
		}
		acc.Add(SearchResult{ID: vec.ID, Distance: distance})
	}
	return acc.ResultsRaw(), nil
}

func (s *Service) searchMetricANN(values []float64, k int, metric DistanceMetric) ([]SearchResult, error) {
	candidateK := max(k*20, 100)
	if s.annAdaptive && s.annMinCandidates > candidateK {
		candidateK = s.annMinCandidates
	}
	if maxCandidates := s.maxK * 4; maxCandidates > 0 && candidateK > maxCandidates {
		candidateK = maxCandidates
	}
	var candidates []ann.Result
	for {
		metricIndex, err := s.metricANNIndex(metric)
		if err != nil {
			return nil, err
		}
		s.metricANNMu.RLock()
		if s.metricANN[metric] != metricIndex {
			s.metricANNMu.RUnlock()
			continue
		}
		candidates, err = metricIndex.SearchWithDistancesInto(values, candidateK, nil)
		s.metricANNMu.RUnlock()
		if err != nil {
			return nil, err
		}
		break
	}
	query := newMultiVectorQuery(values, metric)
	acc := newTopKAccumulator(k)
	candidateIDs := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		id, ok, err := s.lookupID(candidate.ID)
		if err != nil || !ok {
			continue
		}
		candidateIDs = append(candidateIDs, id)
	}
	if reader, ok := s.vectorStore.(selectedVectorIDs32Reader); ok {
		reader.RangeSelectedVectorIDs32(candidateIDs, func(id string, vectorValues []float32) bool {
			acc.Add(SearchResult{ID: id, Distance: query.distance32(vectorValues)})
			return true
		})
		return acc.ResultsRaw(), nil
	}
	for _, id := range candidateIDs {
		vector, err := s.vectorStore.GetVector(id)
		if err != nil {
			continue
		}
		acc.Add(SearchResult{ID: id, Distance: query.distance(vector.Values)})
	}
	return acc.ResultsRaw(), nil
}

// SearchMultiVector aggregates multiple query vectors by the best (minimum)
// distance per document. It is deterministic and works with every metric.
func (s *Service) SearchMultiVector(queries [][]float64, k int, filter VectorFilter, metric DistanceMetric) ([]SearchResult, error) {
	if len(queries) == 0 {
		return nil, ErrInvalidValues
	}
	if k <= 0 {
		return nil, ErrInvalidK
	}
	if err := validateDistanceMetric(metric); err != nil {
		return nil, err
	}
	for _, query := range queries {
		if err := s.validateSearchRequest(query, k); err != nil {
			return nil, err
		}
	}
	plans := make([]multiVectorQuery, len(queries))
	for i, query := range queries {
		plans[i] = newMultiVectorQuery(query, metric)
	}
	if filter == nil {
		if reader, ok := s.vectorStore.(rangeVector32Reader); ok {
			acc := newTopKAccumulator(k)
			reader.RangeVectors32(func(id string, values []float32) bool {
				bestDistance := math.Inf(1)
				for _, query := range plans {
					distance := query.distance32(values)
					if distance < bestDistance {
						bestDistance = distance
					}
				}
				acc.Add(SearchResult{ID: id, Distance: bestDistance})
				return true
			})
			return acc.ResultsRaw(), nil
		}
	}
	vectors := s.ListVectors()
	best := make(map[string]SearchResult, len(vectors))
	for _, vec := range vectors {
		vec.Metadata = s.metadataFor(vec.ID)
		if filter != nil && !filter(vec) {
			continue
		}
		for _, query := range plans {
			distance := query.distance(vec.Values)
			old, ok := best[vec.ID]
			if !ok || distance < old.Distance {
				best[vec.ID] = SearchResult{ID: vec.ID, Distance: distance}
			}
		}
	}
	acc := newTopKAccumulator(k)
	for _, hit := range best {
		acc.Add(hit)
	}
	return acc.ResultsRaw(), nil
}

func validateDistanceMetric(metric DistanceMetric) error {
	switch metric {
	case MetricL2, MetricCosine, MetricInnerProduct:
		return nil
	default:
		return ErrInvalidMetric
	}
}

type multiVectorQuery struct {
	values []float64
	norm   float64
	metric DistanceMetric
}

func newMultiVectorQuery(values []float64, metric DistanceMetric) multiVectorQuery {
	q := multiVectorQuery{values: values, metric: metric}
	if metric == MetricCosine {
		for _, value := range values {
			q.norm += value * value
		}
		q.norm = math.Sqrt(q.norm)
	}
	return q
}

func (q multiVectorQuery) distance(values []float64) float64 {
	if q.metric == MetricL2 {
		distance := 0.0
		for i, value := range q.values {
			d := value - values[i]
			distance += d * d
		}
		return distance
	}
	dot, norm := 0.0, 0.0
	for i, value := range q.values {
		dot += value * values[i]
		if q.metric == MetricCosine {
			norm += values[i] * values[i]
		}
	}
	if q.metric == MetricInnerProduct {
		return -dot
	}
	if q.norm == 0 || norm == 0 {
		return 1
	}
	return 1 - dot/(q.norm*math.Sqrt(norm))
}

func (q multiVectorQuery) distance32(values []float32) float64 {
	if q.metric == MetricL2 {
		distance := 0.0
		for i, value := range q.values {
			d := value - float64(values[i])
			distance += d * d
		}
		return distance
	}
	dot, norm := 0.0, 0.0
	for i, value := range q.values {
		v := float64(values[i])
		dot += value * v
		if q.metric == MetricCosine {
			norm += v * v
		}
	}
	if q.metric == MetricInnerProduct {
		return -dot
	}
	if q.norm == 0 || norm == 0 {
		return 1
	}
	return 1 - dot/(q.norm*math.Sqrt(norm))
}

// SearchFilteredReranked fetches a candidate set and applies a deterministic
// reranker, useful for hybrid lexical/semantic ranking.
func (s *Service) SearchFilteredReranked(values []float64, k, candidateK int, filter VectorFilter, metric DistanceMetric, reranker Reranker) ([]SearchResult, error) {
	if candidateK < k {
		candidateK = k
	}
	hits, err := s.SearchFilteredMetric(values, candidateK, filter, metric)
	if err != nil || reranker == nil {
		if err != nil {
			return nil, err
		}
		if len(hits) > k {
			hits = hits[:k]
		}
		return hits, nil
	}
	for i := range hits {
		hits[i].Distance = reranker(hits[i])
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Distance == hits[j].Distance {
			return hits[i].ID < hits[j].ID
		}
		return hits[i].Distance < hits[j].Distance
	})
	if len(hits) > k {
		hits = hits[:k]
	}
	return hits, nil
}
