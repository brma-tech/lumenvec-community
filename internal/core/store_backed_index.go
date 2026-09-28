package core

import "lumenvec/internal/index"

// storeBackedIndex preserves the legacy VectorIndex contract without keeping
// a second in-memory vector payload. It is used by persistent segmented
// services, where the VectorStore is the canonical source of record.
type storeBackedIndex struct {
	store VectorStore
}

func (i *storeBackedIndex) AddVector(index.Vector) error { return nil }

func (i *storeBackedIndex) AddVector32(string, []float32) error { return nil }

func (i *storeBackedIndex) AddVectors32([]index.Vector32) error { return nil }

func (i *storeBackedIndex) SearchVector(id string) (index.Vector, error) {
	return i.store.GetVector(id)
}

func (i *storeBackedIndex) CopyVector32(id string, dst []float32) ([]float32, error) {
	if reader, ok := i.store.(readOnlyVector32Reader); ok {
		values, err := reader.GetVectorReadOnly32(id)
		if err != nil {
			return nil, err
		}
		return append(dst[:0], values...), nil
	}
	v, err := i.store.GetVector(id)
	if err != nil {
		return nil, err
	}
	dst = dst[:0]
	for _, value := range v.Values {
		dst = append(dst, float32(value))
	}
	return dst, nil
}

func (i *storeBackedIndex) RangeVectors32(fn func(string, []float32) bool) {
	if reader, ok := i.store.(rangeVector32Reader); ok {
		reader.RangeVectors32(fn)
		return
	}
	var scratch []float32
	i.RangeVectors(func(v index.Vector) bool {
		scratch = scratch[:0]
		for _, value := range v.Values {
			scratch = append(scratch, float32(value))
		}
		return fn(v.ID, scratch)
	})
}

func (i *storeBackedIndex) DeleteVector(string) error { return nil }

func (i *storeBackedIndex) ListVectors() []index.Vector {
	return i.store.ListVectors()
}

func (i *storeBackedIndex) RangeVectors(fn func(index.Vector) bool) {
	if reader, ok := i.store.(rangeVector32Reader); ok {
		reader.RangeVectors32(func(id string, values []float32) bool {
			return fn(index.Vector{ID: id, Values: float32ToFloat64(values)})
		})
		return
	}
	for _, vec := range i.store.ListVectors() {
		if !fn(vec) {
			return
		}
	}
}

func float32ToFloat64(values []float32) []float64 {
	out := make([]float64, len(values))
	for n, value := range values {
		out[n] = float64(value)
	}
	return out
}
