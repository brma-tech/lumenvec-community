package ann

import "fmt"

// ScalarQuantizer stores per-dimension min/max ranges and encodes vectors as
// one byte per dimension. It is deterministic and intended for candidate
// generation; exact reranking remains available from the canonical vectors.
type ScalarQuantizer struct {
	Min []float32
	Max []float32
}

func TrainScalarQuantizer(vectors [][]float64) (*ScalarQuantizer, error) {
	if len(vectors) == 0 || len(vectors[0]) == 0 {
		return nil, fmt.Errorf("training vectors are required")
	}
	dim := len(vectors[0])
	q := &ScalarQuantizer{Min: make([]float32, dim), Max: make([]float32, dim)}
	for i := range q.Min {
		q.Min[i], q.Max[i] = float32(vectors[0][i]), float32(vectors[0][i])
	}
	for _, vector := range vectors {
		if len(vector) != dim {
			return nil, fmt.Errorf("training dimension mismatch")
		}
		for i, value := range vector {
			v := float32(value)
			if v < q.Min[i] {
				q.Min[i] = v
			}
			if v > q.Max[i] {
				q.Max[i] = v
			}
		}
	}
	return q, nil
}

func (q *ScalarQuantizer) Encode(vector []float64) ([]byte, error) {
	if q == nil || len(vector) != len(q.Min) {
		return nil, fmt.Errorf("vector dimension mismatch")
	}
	out := make([]byte, len(vector))
	for i, value := range vector {
		span := q.Max[i] - q.Min[i]
		if span <= 0 {
			out[i] = 0
			continue
		}
		x := (float32(value) - q.Min[i]) / span
		if x < 0 {
			x = 0
		}
		if x > 1 {
			x = 1
		}
		out[i] = byte(x*255 + 0.5)
	}
	return out, nil
}

func (q *ScalarQuantizer) Decode(encoded []byte) ([]float64, error) {
	if q == nil || len(encoded) != len(q.Min) {
		return nil, fmt.Errorf("encoded dimension mismatch")
	}
	out := make([]float64, len(encoded))
	for i, value := range encoded {
		out[i] = float64(q.Min[i] + (q.Max[i]-q.Min[i])*float32(value)/255)
	}
	return out, nil
}

func (q *ScalarQuantizer) ApproximateL2(query []float64, encoded []byte) (float64, error) {
	decoded, err := q.Decode(encoded)
	if err != nil || len(query) != len(decoded) {
		if err != nil {
			return 0, err
		}
		return 0, fmt.Errorf("query dimension mismatch")
	}
	var distance float64
	for i := range query {
		d := query[i] - decoded[i]
		distance += d * d
	}
	return distance, nil
}
