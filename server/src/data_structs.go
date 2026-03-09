package main

import (
	"sync"
)

// ----------- data structures -----------

type PolesData struct {
	N     int
	Delta float64
	Ell   int
	C     float64
	Idx   int
}

type ConvergedBlockData struct {
	R, Eta     float64
	Delta12    float64
	Delta34    float64
	PolesData  map[int][]PolesData // ell -> list
	HFinal     []float64
	IndexToKey []PoleKey // index -> (delta, ell)
	Converged  bool
}

// Unified cache key
type gCacheKey struct {
	Delta12, Delta34, Delta float64
	Z                       complex128 // for z-based calls
	R, Eta                  float64    // for r/eta-based calls
	Ell                     int
	MaxIter                 int
	Tol                     float64
}

type PoleKey struct {
	Delta float64
	Ell   int
}

// Extend RecursiveG with a cache for g-values
type RecursiveG struct {
	K1Max, K2Max int
	EllMin       int
	EllMax       int
	Nu           float64

	unique_poles_map map[PoleKey]int
	idxToKey         []PoleKey

	ConvergedCache map[REtaKey]*ConvergedBlockData

	// Cache for RecurseAndEvaluateGUsingZ
	gCache sync.Map // map[gCacheKey]float64
}

type REtaKey struct {
	R, Eta float64
}

func NewRecursiveG(k1max, k2max, ellmin, ellmax, d int) *RecursiveG {
	return &RecursiveG{
		K1Max:            k1max,
		K2Max:            k2max,
		EllMin:           ellmin,
		EllMax:           ellmax,
		Nu:               float64(d-2) / 2.0,
		unique_poles_map: make(map[PoleKey]int),
	}
}
