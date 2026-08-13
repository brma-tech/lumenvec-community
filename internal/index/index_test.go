package index

import (
	"errors"
	"testing"
)

func TestIndexCRUD(t *testing.T) {
	idx := NewIndex()
	if _, err := idx.SearchVector("missing"); !errors.Is(err, ErrVectorNotFound) {
		t.Fatalf("expected ErrVectorNotFound for missing search, got %v", err)
	}
	if err := idx.AddVector(Vector{ID: "a", Values: []float64{1, 2}}); err != nil {
		t.Fatalf("AddVector() error = %v", err)
	}
	if err := idx.AddVector(Vector{ID: "a", Values: []float64{1, 2}}); !errors.Is(err, ErrVectorExists) {
		t.Fatalf("expected ErrVectorExists, got %v", err)
	}

	vec, err := idx.SearchVector("a")
	if err != nil {
		t.Fatalf("SearchVector() error = %v", err)
	}
	vec.Values[0] = 99
	vec2, _ := idx.SearchVector("a")
	if vec2.Values[0] == 99 {
		t.Fatal("expected defensive copy")
	}

	if err := idx.DeleteVector("a"); err != nil {
		t.Fatalf("DeleteVector() error = %v", err)
	}
	if err := idx.DeleteVector("a"); !errors.Is(err, ErrVectorNotFound) {
		t.Fatalf("expected ErrVectorNotFound, got %v", err)
	}
}

func TestIndexListAndRange(t *testing.T) {
	idx := NewIndex()
	_ = idx.AddVector(Vector{ID: "a", Values: []float64{1, 2}})
	_ = idx.AddVector(Vector{ID: "b", Values: []float64{3, 4}})

	list := idx.ListVectors()
	if len(list) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(list))
	}
	list[0].Values[0] = 99
	vec, _ := idx.SearchVector(list[0].ID)
	if vec.Values[0] == 99 {
		t.Fatal("expected cloned values from ListVectors")
	}

	var count int
	idx.RangeVectors(func(v Vector) bool {
		count++
		return count < 2
	})
	if count != 2 {
		t.Fatalf("expected 2 visits, got %d", count)
	}

	var seen32 int
	idx.RangeVectors32(func(id string, values []float32) bool {
		seen32++
		if id == "" || len(values) != 2 {
			t.Fatalf("unexpected RangeVectors32 item id=%q values=%v", id, values)
		}
		return seen32 < 1
	})
	if seen32 != 1 {
		t.Fatalf("expected early stop after 1 RangeVectors32 visit, got %d", seen32)
	}
}

func TestIndexAddVector32CopiesInput(t *testing.T) {
	idx := NewIndex()
	values := []float32{1, 2}
	if err := idx.AddVector32("native", values); err != nil {
		t.Fatalf("AddVector32() error = %v", err)
	}
	values[0] = 99
	got, err := idx.SearchVector("native")
	if err != nil {
		t.Fatalf("SearchVector() error = %v", err)
	}
	if got.Values[0] != 1 {
		t.Fatalf("AddVector32 retained caller buffer: %v", got.Values)
	}
	if err := idx.AddVector32("native", values); !errors.Is(err, ErrVectorExists) {
		t.Fatalf("duplicate AddVector32() error = %v", err)
	}
}

func TestIndexAddVectors32Batch(t *testing.T) {
	idx := NewIndex()
	values := []float32{1, 2}
	if err := idx.AddVectors32([]Vector32{{ID: "a", Values: values}, {ID: "b", Values: []float32{3, 4}}}); err != nil {
		t.Fatalf("AddVectors32() error = %v", err)
	}
	values[0] = 99
	got, err := idx.SearchVector("a")
	if err != nil || got.Values[0] != 1 {
		t.Fatalf("batch did not copy values: got=%v err=%v", got.Values, err)
	}
	if err := idx.AddVectors32([]Vector32{{ID: "a", Values: []float32{5}}}); !errors.Is(err, ErrVectorExists) {
		t.Fatalf("duplicate batch error = %v", err)
	}
}
