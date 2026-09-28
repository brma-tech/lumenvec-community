package core

import "context"

func SearchStructuredWithContext(ctx context.Context, service interface {
	SearchStructured([]float64, int, StructuredFilter, DistanceMetric) ([]SearchResult, error)
}, values []float64, k int, filter StructuredFilter, metric DistanceMetric) ([]SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if contextual, ok := service.(interface {
		SearchStructuredContext(context.Context, []float64, int, StructuredFilter, DistanceMetric) ([]SearchResult, error)
	}); ok {
		return contextual.SearchStructuredContext(ctx, values, k, filter, metric)
	}
	results, err := service.SearchStructured(values, k, filter, metric)
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	return results, err
}

func SearchFilteredMetricWithContext(ctx context.Context, service interface {
	SearchFilteredMetric([]float64, int, VectorFilter, DistanceMetric) ([]SearchResult, error)
}, values []float64, k int, filter VectorFilter, metric DistanceMetric) ([]SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if contextual, ok := service.(interface {
		SearchFilteredMetricContext(context.Context, []float64, int, VectorFilter, DistanceMetric) ([]SearchResult, error)
	}); ok {
		return contextual.SearchFilteredMetricContext(ctx, values, k, filter, metric)
	}
	results, err := service.SearchFilteredMetric(values, k, filter, metric)
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	return results, err
}

func SearchBatchWithContext(ctx context.Context, service interface {
	SearchBatch([]BatchSearchQuery) ([]BatchSearchResult, error)
}, queries []BatchSearchQuery) ([]BatchSearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if contextual, ok := service.(interface {
		SearchBatchContext(context.Context, []BatchSearchQuery) ([]BatchSearchResult, error)
	}); ok {
		return contextual.SearchBatchContext(ctx, queries)
	}
	results, err := service.SearchBatch(queries)
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	return results, err
}

// SearchWithContext preserves compatibility with external adapters. Built-in
// contextual backends cancel computation; legacy adapters can only be checked
// before and after their call and must opt into SearchContext for early abort.
func SearchWithContext(ctx context.Context, service interface {
	Search([]float64, int) ([]SearchResult, error)
}, values []float64, k int) ([]SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if contextual, ok := service.(interface {
		SearchContext(context.Context, []float64, int) ([]SearchResult, error)
	}); ok {
		return contextual.SearchContext(ctx, values, k)
	}
	results, err := service.Search(values, k)
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	return results, err
}
