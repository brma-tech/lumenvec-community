package vector

import (
	"math"
)

// Vector represents a mathematical vector.
type Vector struct {
	Values []float64
}

// NewVector creates a new vector with the given values.
func NewVector(values []float64) *Vector {
	return &Vector{Values: values}
}

// Add adds two vectors and returns the result.
func (v *Vector) Add(other *Vector) *Vector {
	if len(v.Values) != len(other.Values) {
		return nil // or handle error
	}
	result := make([]float64, len(v.Values))
	for i := range v.Values {
		result[i] = v.Values[i] + other.Values[i]
	}
	return NewVector(result)
}

// Subtract subtracts another vector from the current vector and returns the result.
func (v *Vector) Subtract(other *Vector) *Vector {
	if len(v.Values) != len(other.Values) {
		return nil // or handle error
	}
	result := make([]float64, len(v.Values))
	for i := range v.Values {
		result[i] = v.Values[i] - other.Values[i]
	}
	return NewVector(result)
}

// Normalize returns a normalized version of the vector.
func (v *Vector) Normalize() *Vector {
	length := v.Length()
	if length == 0 {
		return nil // or handle error
	}
	result := make([]float64, len(v.Values))
	for i := range v.Values {
		result[i] = v.Values[i] / length
	}
	return NewVector(result)
}

// Length returns the length (magnitude) of the vector.
func (v *Vector) Length() float64 {
	sum := 0.0
	for _, value := range v.Values {
		sum += value * value
	}
	return math.Sqrt(sum)
}