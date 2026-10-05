//go:build !amd64

package vector

func squaredEuclideanDistance32Platform(a, b []float32) float64 {
	return squaredEuclideanDistance32FastGeneric(a, b)
}

func squaredEuclideanDistance32x4Platform(a0, a1, a2, a3, b []float32) (float64, float64, float64, float64) {
	return squaredEuclideanDistance32x4Generic(a0, a1, a2, a3, b)
}

func squaredEuclideanDistance32x4ArenaPlatform(arena []float32, offset0, offset1, offset2, offset3, dim int, b []float32) (float64, float64, float64, float64) {
	return squaredEuclideanDistance32x4Generic(
		arena[offset0:offset0+dim], arena[offset1:offset1+dim],
		arena[offset2:offset2+dim], arena[offset3:offset3+dim], b,
	)
}
