package ann

import "testing"

func TestScalarQuantizerRoundTripAndApproximateDistance(t *testing.T) {
	q, err := TrainScalarQuantizer([][]float64{{0, 10}, {10, 20}})
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := q.Encode([]float64{5, 15})
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := q.Decode(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if decoded[0] < 4.9 || decoded[0] > 5.1 || decoded[1] < 14.9 || decoded[1] > 15.1 {
		t.Fatalf("decoded=%v", decoded)
	}
	distance, err := q.ApproximateL2([]float64{5, 15}, encoded)
	if err != nil || distance > 0.01 {
		t.Fatalf("distance=%v err=%v", distance, err)
	}
}

func TestScalarQuantizerRejectsDimensionMismatch(t *testing.T) {
	q, _ := TrainScalarQuantizer([][]float64{{1, 2}})
	if _, err := q.Encode([]float64{1}); err == nil {
		t.Fatal("expected encode mismatch")
	}
	if _, err := TrainScalarQuantizer([][]float64{{1}, {1, 2}}); err == nil {
		t.Fatal("expected training mismatch")
	}
}
