package ann

import "testing"

func TestProductQuantizerRoundTrip(t *testing.T) {
	pq, err := TrainProductQuantizer([][]float64{{0, 10, 0, 10}, {10, 20, 10, 20}}, 2)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := pq.Encode([]float64{5, 15, 5, 15})
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := pq.Decode(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded) != 4 || decoded[0] < 4.9 || decoded[2] < 4.9 {
		t.Fatalf("decoded=%v", decoded)
	}
}

func TestProductQuantizerRejectsUnevenDimensions(t *testing.T) {
	if _, err := TrainProductQuantizer([][]float64{{1, 2, 3}}, 2); err == nil {
		t.Fatal("expected uneven dimension error")
	}
}
