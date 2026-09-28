package vector

import (
	"math"
)

// EuclideanDistance calculates the Euclidean distance between two vectors.
func EuclideanDistance(a, b []float64) float64 {
	squared := SquaredEuclideanDistance(a, b)
	if squared != squared {
		return math.NaN()
	}
	return math.Sqrt(squared)
}

// SquaredEuclideanDistance calculates the squared Euclidean distance.
// It preserves nearest-neighbor ordering while avoiding sqrt in hot ranking paths.
func SquaredEuclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.NaN() // Return NaN if vectors are of different dimensions
	}
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// ToFloat32 converts a float64 vector into the internal float32 representation.
func ToFloat32(values []float64) []float32 {
	out := make([]float32, len(values))
	for i, value := range values {
		out[i] = float32(value)
	}
	return out
}

// ToFloat64 converts an internal float32 vector back to the public float64 representation.
func ToFloat64(values []float32) []float64 {
	out := make([]float64, len(values))
	for i, value := range values {
		out[i] = float64(value)
	}
	return out
}

// SquaredEuclideanDistance32 calculates squared Euclidean distance for float32 vectors.
func SquaredEuclideanDistance32(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.NaN()
	}
	return SquaredEuclideanDistance32SameLen(a, b)
}

// SquaredEuclideanDistance32SameLen calculates squared Euclidean distance for float32 vectors.
// Callers must ensure both slices have the same length.
func SquaredEuclideanDistance32SameLen(a, b []float32) float64 {
	sum := float64(0)
	i := 0
	for limit := len(a) - len(a)%4; i < limit; i += 4 {
		diff0 := float64(a[i] - b[i])
		diff1 := float64(a[i+1] - b[i+1])
		diff2 := float64(a[i+2] - b[i+2])
		diff3 := float64(a[i+3] - b[i+3])
		sum += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
	}
	for ; i < len(a); i++ {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}
	return sum
}

// SquaredEuclideanDistance32FastSameLen uses the ANN-compatible four-lane
// float32 accumulation order. Callers must ensure equal lengths.
func SquaredEuclideanDistance32FastSameLen(a, b []float32) float64 {
	return squaredEuclideanDistance32Platform(a, b)
}

func squaredEuclideanDistance32FastGeneric(a, b []float32) float64 {
	var distance0, distance1, distance2, distance3 float32
	dimension := 0
	limit := len(a) - len(a)%4
	for ; dimension < limit; dimension += 4 {
		delta0 := a[dimension] - b[dimension]
		delta1 := a[dimension+1] - b[dimension+1]
		delta2 := a[dimension+2] - b[dimension+2]
		delta3 := a[dimension+3] - b[dimension+3]
		distance0 += delta0 * delta0
		distance1 += delta1 * delta1
		distance2 += delta2 * delta2
		distance3 += delta3 * delta3
	}
	for ; dimension < len(a); dimension++ {
		delta := a[dimension] - b[dimension]
		distance0 += delta * delta
	}
	return float64((distance0 + distance1) + (distance2 + distance3))
}

// SquaredEuclideanDistance32x4SameLen calculates squared Euclidean distance from four
// float32 queries to one float32 vector. Callers must ensure all slices have the same length.
func SquaredEuclideanDistance32x4SameLen(a0, a1, a2, a3, b []float32) (float64, float64, float64, float64) {
	sum0, sum1, sum2, sum3 := float64(0), float64(0), float64(0), float64(0)
	i := 0
	for limit := len(b) - len(b)%4; i < limit; i += 4 {
		diff00 := float64(a0[i] - b[i])
		diff01 := float64(a0[i+1] - b[i+1])
		diff02 := float64(a0[i+2] - b[i+2])
		diff03 := float64(a0[i+3] - b[i+3])
		sum0 += diff00*diff00 + diff01*diff01 + diff02*diff02 + diff03*diff03

		diff10 := float64(a1[i] - b[i])
		diff11 := float64(a1[i+1] - b[i+1])
		diff12 := float64(a1[i+2] - b[i+2])
		diff13 := float64(a1[i+3] - b[i+3])
		sum1 += diff10*diff10 + diff11*diff11 + diff12*diff12 + diff13*diff13

		diff20 := float64(a2[i] - b[i])
		diff21 := float64(a2[i+1] - b[i+1])
		diff22 := float64(a2[i+2] - b[i+2])
		diff23 := float64(a2[i+3] - b[i+3])
		sum2 += diff20*diff20 + diff21*diff21 + diff22*diff22 + diff23*diff23

		diff30 := float64(a3[i] - b[i])
		diff31 := float64(a3[i+1] - b[i+1])
		diff32 := float64(a3[i+2] - b[i+2])
		diff33 := float64(a3[i+3] - b[i+3])
		sum3 += diff30*diff30 + diff31*diff31 + diff32*diff32 + diff33*diff33
	}
	for ; i < len(b); i++ {
		diff0 := float64(a0[i] - b[i])
		diff1 := float64(a1[i] - b[i])
		diff2 := float64(a2[i] - b[i])
		diff3 := float64(a3[i] - b[i])
		sum0 += diff0 * diff0
		sum1 += diff1 * diff1
		sum2 += diff2 * diff2
		sum3 += diff3 * diff3
	}
	return sum0, sum1, sum2, sum3
}

// SquaredEuclideanDistance32x4FastSameLen evaluates four vectors against one
// query while preserving the four-lane float32 accumulation order used by the
// ANN L2 kernel. This matters during graph construction: changing rounding can
// change edge tie-breaking and therefore the complete HNSW topology.
func SquaredEuclideanDistance32x4FastSameLen(a0, a1, a2, a3, b []float32) (float64, float64, float64, float64) {
	return squaredEuclideanDistance32x4Platform(a0, a1, a2, a3, b)
}

// SquaredEuclideanDistance32x4ArenaFastSameLen evaluates four vectors stored
// in one contiguous arena. Offsets and dim are trusted by ANN callers that
// own the arena; avoiding four temporary slice headers is material in HNSW's
// inner edge-expansion loop.
func SquaredEuclideanDistance32x4ArenaFastSameLen(arena []float32, offset0, offset1, offset2, offset3, dim int, b []float32) (float64, float64, float64, float64) {
	return squaredEuclideanDistance32x4ArenaPlatform(arena, offset0, offset1, offset2, offset3, dim, b)
}

func squaredEuclideanDistance32x4Generic(a0, a1, a2, a3, b []float32) (float64, float64, float64, float64) {
	var sum00, sum01, sum02, sum03 float32
	var sum10, sum11, sum12, sum13 float32
	var sum20, sum21, sum22, sum23 float32
	var sum30, sum31, sum32, sum33 float32
	dimension := 0
	limit := len(b) - len(b)%4
	for ; dimension < limit; dimension += 4 {
		d00 := a0[dimension] - b[dimension]
		d01 := a0[dimension+1] - b[dimension+1]
		d02 := a0[dimension+2] - b[dimension+2]
		d03 := a0[dimension+3] - b[dimension+3]
		sum00 += d00 * d00
		sum01 += d01 * d01
		sum02 += d02 * d02
		sum03 += d03 * d03

		d10 := a1[dimension] - b[dimension]
		d11 := a1[dimension+1] - b[dimension+1]
		d12 := a1[dimension+2] - b[dimension+2]
		d13 := a1[dimension+3] - b[dimension+3]
		sum10 += d10 * d10
		sum11 += d11 * d11
		sum12 += d12 * d12
		sum13 += d13 * d13

		d20 := a2[dimension] - b[dimension]
		d21 := a2[dimension+1] - b[dimension+1]
		d22 := a2[dimension+2] - b[dimension+2]
		d23 := a2[dimension+3] - b[dimension+3]
		sum20 += d20 * d20
		sum21 += d21 * d21
		sum22 += d22 * d22
		sum23 += d23 * d23

		d30 := a3[dimension] - b[dimension]
		d31 := a3[dimension+1] - b[dimension+1]
		d32 := a3[dimension+2] - b[dimension+2]
		d33 := a3[dimension+3] - b[dimension+3]
		sum30 += d30 * d30
		sum31 += d31 * d31
		sum32 += d32 * d32
		sum33 += d33 * d33
	}
	for ; dimension < len(b); dimension++ {
		d0 := a0[dimension] - b[dimension]
		d1 := a1[dimension] - b[dimension]
		d2 := a2[dimension] - b[dimension]
		d3 := a3[dimension] - b[dimension]
		sum00 += d0 * d0
		sum10 += d1 * d1
		sum20 += d2 * d2
		sum30 += d3 * d3
	}
	return float64((sum00 + sum01) + (sum02 + sum03)),
		float64((sum10 + sum11) + (sum12 + sum13)),
		float64((sum20 + sum21) + (sum22 + sum23)),
		float64((sum30 + sum31) + (sum32 + sum33))
}

// SquaredEuclideanDistance64To32 compares an external float64 query with an internal float32 vector.
func SquaredEuclideanDistance64To32(a []float64, b []float32) float64 {
	if len(a) != len(b) {
		return math.NaN()
	}
	sum := float64(0)
	for i := range a {
		diff := a[i] - float64(b[i])
		sum += diff * diff
	}
	return sum
}

// CosineDistance calculates the cosine distance between two vectors.
func CosineDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.NaN() // Return NaN if vectors are of different dimensions
	}
	dotProduct := 0.0
	magnitudeA := 0.0
	magnitudeB := 0.0
	for i := range a {
		dotProduct += a[i] * b[i]
		magnitudeA += a[i] * a[i]
		magnitudeB += b[i] * b[i]
	}
	if magnitudeA == 0 || magnitudeB == 0 {
		return 1.0 // Return distance of 1 if either vector is zero
	}
	return 1 - (dotProduct / (math.Sqrt(magnitudeA) * math.Sqrt(magnitudeB)))
}
