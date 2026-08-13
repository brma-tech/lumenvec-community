package vector

import (
	"math"
	"testing"
)

func TestEuclideanDistance(t *testing.T) {
	got := EuclideanDistance([]float64{1, 2}, []float64{1, 5})
	if got != 3 {
		t.Fatalf("EuclideanDistance() = %v", got)
	}
}

func TestSquaredEuclideanDistance(t *testing.T) {
	got := SquaredEuclideanDistance([]float64{1, 2}, []float64{1, 5})
	if got != 9 {
		t.Fatalf("SquaredEuclideanDistance() = %v", got)
	}
}

func TestToFloat32AndSquaredEuclideanDistance32(t *testing.T) {
	values := ToFloat32([]float64{1.5, 2.5})
	if len(values) != 2 || values[0] != 1.5 || values[1] != 2.5 {
		t.Fatalf("ToFloat32() = %v", values)
	}
	back := ToFloat64(values)
	if len(back) != 2 || back[0] != 1.5 || back[1] != 2.5 {
		t.Fatalf("ToFloat64() = %v", back)
	}
	got := SquaredEuclideanDistance32(values, []float32{1.5, 5.5})
	if got != 9 {
		t.Fatalf("SquaredEuclideanDistance32() = %v", got)
	}
	got = SquaredEuclideanDistance64To32([]float64{1.5, 2.5}, []float32{1.5, 5.5})
	if got != 9 {
		t.Fatalf("SquaredEuclideanDistance64To32() = %v", got)
	}
	if !math.IsNaN(SquaredEuclideanDistance32([]float32{1}, []float32{1, 2})) {
		t.Fatal("expected NaN")
	}
	if !math.IsNaN(SquaredEuclideanDistance64To32([]float64{1}, []float32{1, 2})) {
		t.Fatal("expected NaN")
	}
	got = SquaredEuclideanDistance32SameLen([]float32{1.5, 2.5, 3.5, 4.5, 5.5}, []float32{1.5, 5.5, 3.5, 1.5, 1.5})
	if got != 34 {
		t.Fatalf("SquaredEuclideanDistance32SameLen() = %v", got)
	}
	got0, got1, got2, got3 := SquaredEuclideanDistance32x4SameLen(
		[]float32{1.5, 2.5, 3.5, 4.5, 5.5},
		[]float32{1.5, 5.5, 3.5, 1.5, 1.5},
		[]float32{0, 0, 0, 0, 0},
		[]float32{2, 2, 2, 2, 2},
		[]float32{1.5, 5.5, 3.5, 1.5, 1.5},
	)
	if got0 != 34 || got1 != 0 || got2 != 49.25 || got3 != 15.25 {
		t.Fatalf("SquaredEuclideanDistance32x4SameLen() = %v %v %v %v", got0, got1, got2, got3)
	}
}

func TestEuclideanDistanceDimensionMismatch(t *testing.T) {
	if !math.IsNaN(EuclideanDistance([]float64{1}, []float64{1, 2})) {
		t.Fatal("expected NaN")
	}
	if !math.IsNaN(SquaredEuclideanDistance([]float64{1}, []float64{1, 2})) {
		t.Fatal("expected NaN")
	}
}

func TestCosineDistance(t *testing.T) {
	got := CosineDistance([]float64{1, 0}, []float64{1, 0})
	if got != 0 {
		t.Fatalf("CosineDistance() = %v", got)
	}
}

func TestCosineDistanceZeroVectorAndMismatch(t *testing.T) {
	if got := CosineDistance([]float64{0, 0}, []float64{1, 0}); got != 1 {
		t.Fatalf("CosineDistance() zero = %v", got)
	}
	if !math.IsNaN(CosineDistance([]float64{1}, []float64{1, 2})) {
		t.Fatal("expected NaN")
	}
}

func TestVectorOperations(t *testing.T) {
	v1 := NewVector([]float64{3, 4})
	v2 := NewVector([]float64{1, 2})

	added := v1.Add(v2)
	if added == nil || added.Values[0] != 4 || added.Values[1] != 6 {
		t.Fatal("unexpected Add result")
	}

	sub := v1.Subtract(v2)
	if sub == nil || sub.Values[0] != 2 || sub.Values[1] != 2 {
		t.Fatal("unexpected Subtract result")
	}

	norm := v1.Normalize()
	if norm == nil || math.Abs(norm.Length()-1) > 1e-9 {
		t.Fatal("expected normalized vector")
	}

	if v1.Length() != 5 {
		t.Fatalf("Length() = %v", v1.Length())
	}
}

func TestVectorOperationsMismatchAndZeroNormalize(t *testing.T) {
	if NewVector([]float64{1}).Add(NewVector([]float64{1, 2})) != nil {
		t.Fatal("expected nil for mismatched Add")
	}
	if NewVector([]float64{1}).Subtract(NewVector([]float64{1, 2})) != nil {
		t.Fatal("expected nil for mismatched Subtract")
	}
	if NewVector([]float64{0, 0}).Normalize() != nil {
		t.Fatal("expected nil for zero Normalize")
	}
}
