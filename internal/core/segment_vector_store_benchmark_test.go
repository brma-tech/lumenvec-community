package core

import (
	"fmt"
	"testing"

	"lumenvec/internal/index"
)

// BenchmarkSegmentVectorStoreBatchCommit measures the durable batch path used
// by segmented ingestion. It intentionally keeps fsync enabled so the result
// reflects the commit point rather than an in-memory write loop.
func BenchmarkSegmentVectorStoreBatchCommit(b *testing.B) {
	vectors := make([]index.Vector, 256)
	for i := range vectors {
		vectors[i] = index.Vector{ID: fmt.Sprintf("v-%d", i), Values: []float64{float64(i), 1, 2, 3}}
	}
	store := newSegmentVectorStore(b.TempDir(), DefaultStorageSecurityOptions())
	b.Cleanup(func() { _ = store.Close() })
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		batch := make([]index.Vector, len(vectors))
		copy(batch, vectors)
		batch[0].ID = fmt.Sprintf("v-%d-%d", i, 0)
		if err := store.UpsertVectors(batch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSegmentVectorStoreRangeVectors32(b *testing.B) {
	store := newSegmentVectorStore(b.TempDir(), DefaultStorageSecurityOptions())
	b.Cleanup(func() { _ = store.Close() })
	vectors := make([]index.Vector, 1000)
	for i := range vectors {
		vectors[i] = index.Vector{ID: fmt.Sprintf("stream-%04d", i), Values: []float64{float64(i), 1, 2, 3}}
	}
	if err := store.UpsertVectors(vectors); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var total float32
		store.RangeVectors32(func(_ string, values []float32) bool {
			if len(values) > 0 {
				total += values[0]
			}
			return true
		})
		if total < 0 {
			b.Fatal("unexpected negative total")
		}
	}
}
