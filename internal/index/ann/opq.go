package ann

import (
	"fmt"
	"sort"
)

// OPQ applies a deterministic orthogonal permutation before Product
// Quantization. The permutation groups high-variance dimensions consistently;
// because it is orthogonal, it preserves L2 distances before quantization.
type OPQ struct {
	Permutation []int
	Quantizer   *ProductQuantizer
}

func TrainOPQ(vectors [][]float64, subspaces int) (*OPQ, error) {
	if len(vectors) == 0 || len(vectors[0]) == 0 {
		return nil, fmt.Errorf("training vectors are required")
	}
	dim := len(vectors[0])
	means := make([]float64, dim)
	for _, vector := range vectors {
		if len(vector) != dim {
			return nil, fmt.Errorf("training dimension mismatch")
		}
		for n, value := range vector {
			means[n] += value
		}
	}
	for n := range means {
		means[n] /= float64(len(vectors))
	}
	variance := make([]float64, dim)
	for _, vector := range vectors {
		for n, value := range vector {
			d := value - means[n]
			variance[n] += d * d
		}
	}
	order := make([]int, dim)
	for n := range order {
		order[n] = n
	}
	sort.Slice(order, func(a, b int) bool {
		if variance[order[a]] == variance[order[b]] {
			return order[a] < order[b]
		}
		return variance[order[a]] > variance[order[b]]
	})
	permuted := permuteVectors(vectors, order)
	quantizer, err := TrainProductQuantizer(permuted, subspaces)
	if err != nil {
		return nil, err
	}
	return &OPQ{Permutation: order, Quantizer: quantizer}, nil
}

func (o *OPQ) Encode(vector []float64) ([]byte, error) {
	if o == nil || len(vector) != len(o.Permutation) {
		return nil, fmt.Errorf("vector dimension mismatch")
	}
	return o.Quantizer.Encode(permute(vector, o.Permutation))
}
func (o *OPQ) Decode(encoded []byte) ([]float64, error) {
	permuted, err := o.Quantizer.Decode(encoded)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(permuted))
	for n, original := range o.Permutation {
		out[original] = permuted[n]
	}
	return out, nil
}
func permuteVectors(vectors [][]float64, permutation []int) [][]float64 {
	out := make([][]float64, len(vectors))
	for n, vector := range vectors {
		out[n] = permute(vector, permutation)
	}
	return out
}
func permute(vector []float64, permutation []int) []float64 {
	out := make([]float64, len(vector))
	for n, index := range permutation {
		out[n] = vector[index]
	}
	return out
}
