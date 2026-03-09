package main

import (
	"math"
	"testing"
)

// TestNumericVsSymbolicDerivatives verifies that numeric and symbolic implementations
// of r(z,zbar) and eta(z,zbar) derivatives match at the crossing symmetric point.
func TestNumericVsSymbolicDerivatives(t *testing.T) {
	numericEval := NewRetaEvaluator(NumericMode)
	symbolicEval := NewRetaEvaluator(SymbolicMode)

	const z = 0.5
	const zbar = 0.5
	const tolerance = 1e-5

	testCases := []struct {
		m, n int
		name string
	}{
		{0, 0, "zeroth order"},
		{1, 0, "first order r only"},
		{0, 1, "first order eta only"},
		{1, 1, "first order both"},
		{2, 0, "second order r only"},
		{0, 2, "second order eta only"},
		{2, 2, "second order both"},
		{3, 3, "third order both"},
		{4, 2, "mixed order 4,2"},
		{2, 4, "mixed order 2,4"},
		{5, 5, "fifth order both"},
		{7, 8, "high mixed order"},
	}

	for _, tc := range testCases {
		t.Run("r_derivative_"+tc.name, func(t *testing.T) {
			numericR, err := numericEval.EvalR(tc.m, tc.n, z, zbar)
			if err != nil {
				t.Fatalf("Numeric EvalR(%d, %d) failed: %v", tc.m, tc.n, err)
			}
			symbolicR, _ := symbolicEval.EvalR(tc.m, tc.n, z, zbar)

			diff := math.Abs(numericR - symbolicR)
			relativeError := 0.0
			if math.Abs(numericR) > 1e-4 {
				relativeError = diff / math.Abs(numericR)
			}

			if diff > tolerance && relativeError > tolerance {
				t.Errorf("r derivative mismatch at (m=%d, n=%d):\n  Numeric:  %.15e\n  Symbolic: %.15e\n  Diff:     %.15e\n  RelErr:   %.15e",
					tc.m, tc.n, numericR, symbolicR, diff, relativeError)
			}
		})

		t.Run("eta_derivative_"+tc.name, func(t *testing.T) {
			numericEta, err := numericEval.EvalEta(tc.m, tc.n, z, zbar)
			if err != nil {
				t.Fatalf("Numeric EvalEta(%d, %d) failed: %v", tc.m, tc.n, err)
			}
			symbolicEta, _ := symbolicEval.EvalEta(tc.m, tc.n, z, zbar)

			diff := math.Abs(numericEta - symbolicEta)
			relativeError := 0.0
			if math.Abs(numericEta) > 1e-4 {
				relativeError = diff / math.Abs(numericEta)
			}

			if diff > tolerance && relativeError > tolerance {
				t.Errorf("eta derivative mismatch at (m=%d, n=%d):\n  Numeric:  %.15e\n  Symbolic: %.15e\n  Diff:     %.15e\n  RelErr:   %.15e",
					tc.m, tc.n, numericEta, symbolicEta, diff, relativeError)
			}
		})
	}
}

// TestSymbolicDerivativesAtVariousPoints tests symbolic derivatives at points
// other than the crossing symmetric point to ensure formulas work generally.
func TestSymbolicDerivativesAtVariousPoints(t *testing.T) {
	symbolicEval := NewRetaEvaluator(SymbolicMode)

	points := []struct {
		z, zbar float64
		name    string
	}{
		{0.5, 0.5, "crossing symmetric"},
		{0.3, 0.3, "symmetric 0.3"},
		{0.7, 0.7, "symmetric 0.7"},
		{0.4, 0.6, "asymmetric 0.4, 0.6"},
		{0.2, 0.8, "asymmetric 0.2, 0.8"},
		{0.1, 0.9, "asymmetric 0.1, 0.9"},
	}

	for _, pt := range points {
		t.Run("r_at_"+pt.name, func(t *testing.T) {
			for m := 0; m <= 3; m++ {
				for n := 0; n <= 3; n++ {
					r, err := symbolicEval.EvalR(m, n, pt.z, pt.zbar)
					if err != nil {
						t.Errorf("Symbolic EvalR(%d, %d) at (%.2f, %.2f) failed: %v",
							m, n, pt.z, pt.zbar, err)
					}
					if math.IsNaN(r) || math.IsInf(r, 0) {
						t.Errorf("Symbolic EvalR(%d, %d) at (%.2f, %.2f) gave non-finite result: %v",
							m, n, pt.z, pt.zbar, r)
					}
				}
			}
		})

		t.Run("eta_at_"+pt.name, func(t *testing.T) {
			for m := 0; m <= 3; m++ {
				for n := 0; n <= 3; n++ {
					eta, err := symbolicEval.EvalEta(m, n, pt.z, pt.zbar)
					if err != nil {
						t.Errorf("Symbolic EvalEta(%d, %d) at (%.2f, %.2f) failed: %v",
							m, n, pt.z, pt.zbar, err)
					}
					if math.IsNaN(eta) || math.IsInf(eta, 0) {
						t.Errorf("Symbolic EvalEta(%d, %d) at (%.2f, %.2f) gave non-finite result: %v",
							m, n, pt.z, pt.zbar, eta)
					}
				}
			}
		})
	}
}

// TestSymbolicCachingAtCrossingPoint verifies that the precomputed tables
// at the crossing symmetric point are being used correctly.
func TestSymbolicCachingAtCrossingPoint(t *testing.T) {
	symbolicEval := NewREtaDerivativesSymbolic(true, 10)

	const z = 0.5
	const zbar = 0.5

	if len(symbolicEval.rTableCrossingPoint) == 0 {
		t.Error("rTableCrossingPoint is empty after initialization with precompute=true")
	}
	if len(symbolicEval.etaTableCrossingPoint) == 0 {
		t.Error("etaTableCrossingPoint is empty after initialization with precompute=true")
	}

	testOrders := [][2]int{
		{0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 2}, {5, 5}, {10, 10},
	}

	for _, order := range testOrders {
		m, n := order[0], order[1]

		cachedR, okR := symbolicEval.rTableCrossingPoint[[2]int{m, n}]
		cachedEta, okEta := symbolicEval.etaTableCrossingPoint[[2]int{m, n}]
		computedR, errR := symbolicEval.EvalR(m, n, z, zbar)
		computedEta, errEta := symbolicEval.EvalEta(m, n, z, zbar)

		if okR {
			if errR != nil {
				t.Errorf("EvalR(%d, %d) failed: %v", m, n, errR)
			}
			if math.Abs(cachedR-computedR) > 1e-14 {
				t.Errorf("Cached r value differs from computed for (%d, %d): cached=%v, computed=%v",
					m, n, cachedR, computedR)
			}
		}

		if okEta {
			if errEta != nil {
				t.Errorf("EvalEta(%d, %d) failed: %v", m, n, errEta)
			}
			if math.Abs(cachedEta-computedEta) > 1e-14 {
				t.Errorf("Cached eta value differs from computed for (%d, %d): cached=%v, computed=%v",
					m, n, cachedEta, computedEta)
			}
		}
	}
}

// TestZerothOrderDerivatives checks that 0th order derivatives match expected values.
func TestZerothOrderDerivatives(t *testing.T) {
	const z = 0.5
	const zbar = 0.5

	expectedR := 3.0 - 2.0*math.Sqrt(2.0)
	expectedEta := 1.0

	numericEval := NewRetaEvaluator(NumericMode)
	symbolicEval := NewRetaEvaluator(SymbolicMode)

	evaluators := map[string]RetaEvaluator{
		"Numeric":  numericEval,
		"Symbolic": symbolicEval,
	}

	for name, eval := range evaluators {
		t.Run("r_"+name, func(t *testing.T) {
			r, err := eval.EvalR(0, 0, z, zbar)
			if err != nil {
				t.Fatalf("EvalR(0, 0) failed: %v", err)
			}
			if math.Abs(r-expectedR) > 1e-10 {
				t.Errorf("%s r(0,0) mismatch: expected %.15f, got %.15f",
					name, expectedR, r)
			}
		})

		t.Run("eta_"+name, func(t *testing.T) {
			eta, err := eval.EvalEta(0, 0, z, zbar)
			if err != nil {
				t.Fatalf("EvalEta(0, 0) failed: %v", err)
			}
			if math.Abs(eta-expectedEta) > 1e-10 {
				t.Errorf("%s eta(0,0) mismatch: expected %.15f, got %.15f",
					name, expectedEta, eta)
			}
		})
	}
}

// TestSymmetryProperties verifies that r(z,zbar) = r(zbar,z) and similar symmetries.
func TestSymmetryProperties(t *testing.T) {
	symbolicEval := NewRetaEvaluator(SymbolicMode)

	const z = 0.4
	const zbar = 0.6
	const tolerance = 1e-12

	testOrders := [][2]int{
		{0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 0}, {0, 2}, {2, 2}, {3, 3},
	}

	for _, order := range testOrders {
		m, n := order[0], order[1]

		r1, _ := symbolicEval.EvalR(m, n, z, zbar)
		r2, _ := symbolicEval.EvalR(n, m, zbar, z)
		eta1, _ := symbolicEval.EvalEta(m, n, z, zbar)
		eta2, _ := symbolicEval.EvalEta(n, m, zbar, z)

		if math.Abs(r1-r2) > tolerance {
			t.Errorf("r symmetry broken for (%d,%d): r(%.2f,%.2f)=%v, r(%.2f,%.2f)=%v",
				m, n, z, zbar, r1, zbar, z, r2)
		}

		if math.Abs(eta1-eta2) > tolerance {
			t.Errorf("eta symmetry broken for (%d,%d): eta(%.2f,%.2f)=%v, eta(%.2f,%.2f)=%v",
				m, n, z, zbar, eta1, zbar, z, eta2)
		}
	}
}

// BenchmarkNumericDerivatives benchmarks the numeric implementation.
func BenchmarkNumericDerivatives(b *testing.B) {
	numericEval := NewRetaEvaluator(NumericMode)
	const z = 0.5
	const zbar = 0.5

	b.Run("r_derivative_5_5", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = numericEval.EvalR(5, 5, z, zbar)
		}
	})

	b.Run("eta_derivative_5_5", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = numericEval.EvalEta(5, 5, z, zbar)
		}
	})
}

// BenchmarkSymbolicDerivatives benchmarks the symbolic implementation.
func BenchmarkSymbolicDerivatives(b *testing.B) {
	symbolicEval := NewRetaEvaluator(SymbolicMode)
	const z = 0.5
	const zbar = 0.5

	b.Run("r_derivative_5_5", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = symbolicEval.EvalR(5, 5, z, zbar)
		}
	})

	b.Run("eta_derivative_5_5", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = symbolicEval.EvalEta(5, 5, z, zbar)
		}
	})
}

// BenchmarkSymbolicWithoutPrecompute benchmarks symbolic without precomputation.
func BenchmarkSymbolicWithoutPrecompute(b *testing.B) {
	eval := NewREtaDerivativesSymbolic(false, 10)
	const z = 0.5
	const zbar = 0.5

	b.Run("r_derivative_5_5_no_cache", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = eval.EvalR(5, 5, z, zbar)
		}
	})

	b.Run("eta_derivative_5_5_no_cache", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = eval.EvalEta(5, 5, z, zbar)
		}
	})
}
