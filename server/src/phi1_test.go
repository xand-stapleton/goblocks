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

func TestPhi1PrecomputedMatchesAnalyticAtCrossingSymmetricPointD3(t *testing.T) {
	rg := NewRecursiveG(10, 10, 0, 10, 3)

	// Keep nMax consistent with other tests (avoids derivative-order edge cases).
	rdPre := NewRecursiveDerivatives(*rg, 8, true, false, false)
	rdAna := NewRecursiveDerivatives(*rg, 8, false, false, false)

	r := 3 - 2*math.Sqrt(2)
	nu := 0.5

	// The precomputed table currently provides i=0..27.
	for i := 0; i <= 27; i++ {
		gotPre := rdPre.phi1(i, 0, r, nu)
		gotAna := rdAna.phi1(i, 0, r, nu)

		// These are pure float64 computations; tolerances are set tightly but not unrealistically.
		assertCloseAbsRel(t, gotPre, gotAna, 1e-13, 1e-13, "phi1 precomputed vs analytic (i="+itoa(i)+")")
	}
}

func TestPhi1PrecomputedPanicsOutsideCrossingSymmetricPoint(t *testing.T) {
	rg := NewRecursiveG(10, 10, 0, 10, 3)
	rdPre := NewRecursiveDerivatives(*rg, 8, true, false, false)
	rdAna := NewRecursiveDerivatives(*rg, 8, false, false, false)

	// Away from the crossing-symmetric point the precomputed table is not valid,
	// so rdPre.phi1 should transparently fall back to the analytic implementation.
	r := (3 - 2*math.Sqrt(2)) + 1e-3
	nu := 0.5

	for i := 0; i <= 10; i++ {
		gotPre := rdPre.phi1(i, 0, r, nu)
		gotAna := rdAna.phi1(i, 0, r, nu)
		assertCloseAbsRel(t, gotPre, gotAna, 1e-13, 1e-13, "phi1 precomputed fallback vs analytic (i="+itoa(i)+")")
	}
}

func itoa(x int) string {
	// tiny helper to avoid fmt import in test hot path
	if x == 0 {
		return "0"
	}
	neg := false
	if x < 0 {
		neg = true
		x = -x
	}
	buf := make([]byte, 0, 12)
	for x > 0 {
		buf = append(buf, byte('0'+x%10))
		x /= 10
	}
	// reverse
	for i, j := 0, len(buf)-1; i < j; i, j = i+1, j-1 {
		buf[i], buf[j] = buf[j], buf[i]
	}
	if neg {
		buf = append([]byte{'-'}, buf...)
	}
	return string(buf)
}
