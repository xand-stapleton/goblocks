package main

import "testing"

func BenchmarkRecurseOnlyDefaults(b *testing.B) {
	b.ReportAllocs()
	// Match the typical parameter sizes used in tests / CLI.
	rg := NewRecursiveG(10, 10, 0, 10, 3)
	delta12, delta34 := 1.6, 1.2
	z := complex(0.2, 0.1)
	r, eta := rg.zToREta(z)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rg.Recurse(delta12, delta34, r, eta, 100, 1e-4)
	}
}

func BenchmarkRecurseAndEvaluateFUsingZDefaults(b *testing.B) {
	b.ReportAllocs()
	rg := NewRecursiveG(10, 10, 0, 10, 3)
	delta12, delta34, deltaAve23 := 1.6, 1.2, 3.1
	z := complex(0.2, 0.1)

	deltas := []float64{5.1, 6.2, 7.3, 8.4, 9.5, 10.6}
	ells := []int{0, 1, 2, 3, 4, 5}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = rg.RecurseAndEvaluateFUsingZ(BlockPlus, delta12, delta34, deltaAve23, z, deltas, ells, 100, 1e-4)
	}
}
