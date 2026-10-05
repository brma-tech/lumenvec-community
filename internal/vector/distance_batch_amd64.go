//go:build amd64

package vector

//go:noescape
func squaredEuclideanDistance32SSE(a, b []float32) (sum float32)

//go:noescape
func squaredEuclideanDistance32x4SSE(a0, a1, a2, a3, b []float32) (sum0, sum1, sum2, sum3 float32)

//go:noescape
func squaredEuclideanDistance32x4ArenaSSE(arena []float32, offset0, offset1, offset2, offset3, dim int, b []float32) (sum0, sum1, sum2, sum3 float32)

func squaredEuclideanDistance32Platform(a, b []float32) float64 {
	return float64(squaredEuclideanDistance32SSE(a, b))
}

func squaredEuclideanDistance32x4Platform(a0, a1, a2, a3, b []float32) (float64, float64, float64, float64) {
	sum0, sum1, sum2, sum3 := squaredEuclideanDistance32x4SSE(a0, a1, a2, a3, b)
	return float64(sum0), float64(sum1), float64(sum2), float64(sum3)
}

func squaredEuclideanDistance32x4ArenaPlatform(arena []float32, offset0, offset1, offset2, offset3, dim int, b []float32) (float64, float64, float64, float64) {
	sum0, sum1, sum2, sum3 := squaredEuclideanDistance32x4ArenaSSE(arena, offset0, offset1, offset2, offset3, dim, b)
	return float64(sum0), float64(sum1), float64(sum2), float64(sum3)
}
