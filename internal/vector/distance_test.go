package vector

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func BenchmarkSquaredEuclideanDistance32Kernel(b *testing.B) {
	for _, dimension := range []int{128, 256} {
		b.Run(fmt.Sprintf("dim%d", dimension), func(b *testing.B) {
			rng := rand.New(rand.NewSource(int64(dimension)))
			a := make([]float32, dimension)
			c := make([]float32, dimension)
			for i := range a {
				a[i], c[i] = rng.Float32(), rng.Float32()
			}
			for _, kernel := range []struct {
				name string
				fn   func([]float32, []float32) float64
			}{
				{name: "platform", fn: squaredEuclideanDistance32Platform},
				{name: "generic", fn: squaredEuclideanDistance32FastGeneric},
			} {
				b.Run(kernel.name, func(b *testing.B) {
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						_ = kernel.fn(a, c)
					}
				})
			}
		})
	}
}

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

func TestSquaredEuclideanDistance32x4FastMatchesANNAccumulation(t *testing.T) {
	vectors := [][]float32{
		{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7},
		{7.7, 6.6, 5.5, 4.4, 3.3, 2.2, 1.1},
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7},
		{9.1, 8.2, 7.3, 6.4, 5.5, 4.6, 3.7},
	}
	query := []float32{0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3}
	got0, got1, got2, got3 := SquaredEuclideanDistance32x4FastSameLen(vectors[0], vectors[1], vectors[2], vectors[3], query)
	got := []float64{got0, got1, got2, got3}
	for vectorIndex, vector := range vectors {
		var lane0, lane1, lane2, lane3 float32
		dimension := 0
		for ; dimension+3 < len(query); dimension += 4 {
			d0 := vector[dimension] - query[dimension]
			d1 := vector[dimension+1] - query[dimension+1]
			d2 := vector[dimension+2] - query[dimension+2]
			d3 := vector[dimension+3] - query[dimension+3]
			lane0 += d0 * d0
			lane1 += d1 * d1
			lane2 += d2 * d2
			lane3 += d3 * d3
		}
		for ; dimension < len(query); dimension++ {
			delta := vector[dimension] - query[dimension]
			lane0 += delta * delta
		}
		want := float64((lane0 + lane1) + (lane2 + lane3))
		if got[vectorIndex] != want {
			t.Fatalf("distance %d = %v, want bit-consistent %v", vectorIndex, got[vectorIndex], want)
		}
	}
}

func TestSquaredEuclideanDistance32x4FastMatchesGenericAcrossLengths(t *testing.T) {
	rng := rand.New(rand.NewSource(73))
	for length := 1; length <= 257; length++ {
		vectors := [5][]float32{}
		for vectorIndex := range vectors {
			vectors[vectorIndex] = make([]float32, length)
			for dimension := range vectors[vectorIndex] {
				vectors[vectorIndex][dimension] = float32(rng.Float64()*20 - 10)
			}
		}
		got0, got1, got2, got3 := SquaredEuclideanDistance32x4FastSameLen(vectors[0], vectors[1], vectors[2], vectors[3], vectors[4])
		singleGot := SquaredEuclideanDistance32FastSameLen(vectors[0], vectors[4])
		arena := make([]float32, 0, length*4)
		for vectorIndex := 0; vectorIndex < 4; vectorIndex++ {
			arena = append(arena, vectors[vectorIndex]...)
		}
		arena0, arena1, arena2, arena3 := SquaredEuclideanDistance32x4ArenaFastSameLen(arena, 0, length, length*2, length*3, length, vectors[4])
		want0, want1, want2, want3 := squaredEuclideanDistance32x4Generic(vectors[0], vectors[1], vectors[2], vectors[3], vectors[4])
		got := [4]float64{got0, got1, got2, got3}
		arenaGot := [4]float64{arena0, arena1, arena2, arena3}
		want := [4]float64{want0, want1, want2, want3}
		if got != want {
			t.Fatalf("length %d: SSE=%v, generic=%v", length, got, want)
		}
		if singleGot != want0 {
			t.Fatalf("length %d: single SSE=%v, generic=%v", length, singleGot, want0)
		}
		if arenaGot != want {
			t.Fatalf("length %d: arena SSE=%v, generic=%v", length, arenaGot, want)
		}
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
