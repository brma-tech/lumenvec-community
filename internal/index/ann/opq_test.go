package ann

import "testing"

func TestOPQEncodeDecodeRestoresDimensionOrder(t *testing.T) {
	opq, err := TrainOPQ([][]float64{{0, 0, 0, 0}, {10, 1, 10, 1}}, 2)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := opq.Encode([]float64{5, .5, 5, .5})
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := opq.Decode(encoded)
	if err != nil || len(decoded) != 4 {
		t.Fatalf("decoded=%v err=%v", decoded, err)
	}
	if decoded[0] < 4.9 || decoded[2] < 4.9 {
		t.Fatalf("decoded=%v", decoded)
	}
}
