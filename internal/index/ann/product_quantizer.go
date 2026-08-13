package ann

import "fmt"

// ProductQuantizer splits a vector into equal subspaces and quantizes each
// independently. It is a compact PQ representation for candidate retrieval.
type ProductQuantizer struct {
	Subspaces []ScalarQuantizer
	SubDim    int
}

func TrainProductQuantizer(vectors [][]float64, subspaces int) (*ProductQuantizer, error) {
	if len(vectors) == 0 || subspaces <= 0 || len(vectors[0]) == 0 || len(vectors[0])%subspaces != 0 {
		return nil, fmt.Errorf("invalid product quantizer dimensions")
	}
	dim := len(vectors[0])
	pq := &ProductQuantizer{SubDim: dim / subspaces, Subspaces: make([]ScalarQuantizer, subspaces)}
	for _, vector := range vectors {
		if len(vector) != dim {
			return nil, fmt.Errorf("training dimension mismatch")
		}
	}
	for n := 0; n < subspaces; n++ {
		samples := make([][]float64, len(vectors))
		for row, vector := range vectors {
			samples[row] = vector[n*pq.SubDim : (n+1)*pq.SubDim]
		}
		q, err := TrainScalarQuantizer(samples)
		if err != nil {
			return nil, err
		}
		pq.Subspaces[n] = *q
	}
	return pq, nil
}

func (p *ProductQuantizer) Encode(vector []float64) ([]byte, error) {
	if p == nil || p.SubDim <= 0 || len(vector) != p.SubDim*len(p.Subspaces) {
		return nil, fmt.Errorf("vector dimension mismatch")
	}
	out := make([]byte, 0, len(vector))
	for n, q := range p.Subspaces {
		part, err := q.Encode(vector[n*p.SubDim : (n+1)*p.SubDim])
		if err != nil {
			return nil, err
		}
		out = append(out, part...)
	}
	return out, nil
}

func (p *ProductQuantizer) Decode(encoded []byte) ([]float64, error) {
	if p == nil || len(encoded) != p.SubDim*len(p.Subspaces) {
		return nil, fmt.Errorf("encoded dimension mismatch")
	}
	out := make([]float64, 0, len(encoded))
	for n, q := range p.Subspaces {
		part, err := q.Decode(encoded[n*p.SubDim : (n+1)*p.SubDim])
		if err != nil {
			return nil, err
		}
		out = append(out, part...)
	}
	return out, nil
}
