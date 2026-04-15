package main

import (
	"math"
	"testing"
)

func assertCloseAbsRel(t *testing.T, got, want, absTol, relTol float64, label string) {
	t.Helper()
	diff := math.Abs(got - want)
	if diff <= absTol {
		return
	}
	scale := math.Max(1.0, math.Max(math.Abs(got), math.Abs(want)))
	if diff/scale <= relTol {
		return
	}
	t.Fatalf("%s: got %.18e want %.18e (|diff|=%.3e)", label, got, want, diff)
}

func assertCloseSliceAbs(t *testing.T, got, want []float64, tol float64, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length mismatch: got %d values, expected %d", label, len(got), len(want))
	}
	for i := range want {
		diff := math.Abs(got[i] - want[i])
		if diff > tol {
			t.Fatalf("%s mismatch at index %d: got %.18e, expected %.18e (|diff|=%.3e)", label, i, got[i], want[i], diff)
		}
	}
}