package handlers

import "math"

// SoftmaxRow normalizes a slice of values into a probability distribution in-place.
func SoftmaxRow(row []float64) {
	maxVal := row[0]
	for _, v := range row {
		if v > maxVal {
			maxVal = v
		}
	}
	sum := 0.0
	for i := range row {
		row[i] = math.Exp(row[i] - maxVal)
		sum += row[i]
	}
	for i := range row {
		row[i] /= sum
	}
}
