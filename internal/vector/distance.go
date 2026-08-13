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
