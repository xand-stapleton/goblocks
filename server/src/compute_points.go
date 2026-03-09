package main

import (
	"fmt"
	"log"
	"math"

	"github.com/scientificgo/special"
)

// ----------- core API (port of Python) -----------

func (rg *RecursiveG) Recurse(delta12, delta34, r, eta float64, maxIter int, tol float64) {
	// Reset unique poles
	rg.unique_poles_map = make(map[PoleKey]int)
	rg.idxToKey = nil

	polesData := rg.getAllPolesData(delta12, delta34)
	numKeys := len(rg.idxToKey)
	if numKeys == 0 {
		panic("Recurse: idxToKey is empty")
	}

	hTilde := make([]float64, numKeys)
	h := make([]float64, numKeys)
	hNew := make([]float64, numKeys)

	// Precompute hTilde
	for idx, key := range rg.idxToKey {
		hTilde[idx] = rg.htilde(delta12, delta34, key.Ell, r, eta)
	}
	copy(h, hTilde)

	// Precompute r^N for all poles
	powerCache := make(map[int]float64)
	for _, poles := range polesData {
		for _, p := range poles {
			if _, exists := powerCache[p.N]; !exists {
				powerCache[p.N] = math.Pow(r, float64(p.N))
			}
		}
	}

	// Fixed-point iteration
	var converged bool
	var maxDiff float64
	var fracDiff float64
	for iter := 0; iter < maxIter; iter++ {
		maxDiff = 0.0
		fracDiff = 0.0
		converged = false

		for idx, key := range rg.idxToKey {
			sum := 0.0
			for _, p := range polesData[key.Ell] {
				sum += p.C * powerCache[p.N] * h[p.Idx] / (key.Delta - p.Delta)
			}
			hNew[idx] = hTilde[idx] + sum

			// Track max difference for convergence
			d := hNew[idx] - h[idx]
			if h[idx] != 0.0 {
				fracDiff = math.Abs(d / h[idx])
			} else {
				fracDiff = math.NaN()
			}
			if d < 0 {
				d = -d
			}
			if d > maxDiff {
				maxDiff = d
			}
		}

		if fracDiff < tol {
			converged = true
			break
		}

		// Swap slices without copying
		h, hNew = hNew, h
	}

	if !converged {
		log.Printf("Failed to converge: maxdiff = %v; fracdiff = %v\n", maxDiff, fracDiff)
	}

	if rg.ConvergedCache == nil {
		rg.ConvergedCache = make(map[REtaKey]*ConvergedBlockData)
	}
	rg.ConvergedCache[REtaKey{r, eta}] = &ConvergedBlockData{
		R:          r,
		Eta:        eta,
		Delta12:    delta12,
		Delta34:    delta34,
		PolesData:  polesData,
		HFinal:     append([]float64(nil), h...),           // copy for safety
		IndexToKey: append([]PoleKey(nil), rg.idxToKey...), // copy for safety
		Converged:  converged,
	}
}

func (rg *RecursiveG) EvaluateG(delta12, delta34, delta float64, ell int, r, eta float64) (float64, error) {
	if rg.ConvergedCache == nil {
		return 0, fmt.Errorf("no recursion data available; call Recurse first")
	}

	cd, ok := rg.ConvergedCache[REtaKey{r, eta}]
	if !ok {
		return 0, fmt.Errorf("no cached recursion for r=%v, eta=%v", r, eta)
	}

	// Bail if that recursion failed to converge
	if !cd.Converged {
		log.Printf("Recursion did not converge for r=%v, eta=%v", r, eta)
		return 0, fmt.Errorf("recursion did not converge for r=%v, eta=%v", r, eta)
	}

	if delta12 != cd.Delta12 || delta34 != cd.Delta34 {
		return 0, fmt.Errorf("delta values do not match the ones used in recursion")
	}

	poles, ok := cd.PolesData[ell]
	if !ok {
		return 0, fmt.Errorf("no poles data available for spin %d; call Recurse first", ell)
	}

	g := rg.htilde(delta12, delta34, ell, r, eta)

	for _, p := range poles {
		if delta == p.Delta { // protect against division-by-zero
			continue
		}
		g += p.C * math.Pow(r, float64(p.N)) * cd.HFinal[p.Idx] / (delta - p.Delta)
	}

	g *= math.Pow(r, delta)
	return g, nil
}

// Recurse once per r, eta and evaluate all (delta, ell) pairs for G
func (rg *RecursiveG) RecurseAndEvaluateGUsingZ(delta12, delta34 float64, z complex128, deltas []float64, ells []int, maxIter int, tol float64) ([]float64, error) {
	if len(deltas) != len(ells) {
		return nil, fmt.Errorf("deltas and ells must have the same length")
	}

	r, eta := rg.zToREta(z)

	// Recurse only once per z
	rg.Recurse(delta12, delta34, r, eta, maxIter, tol)

	results := make([]float64, len(deltas))
	for i := range deltas {
		val, err := rg.EvaluateG(delta12, delta34, deltas[i], ells[i], r, eta)
		if err != nil {
			results[i] = math.NaN()
		} else {
			results[i] = val
		}
	}

	return results, nil
}

// Recurse once per z and evaluate all (delta, ell) pairs for F blocks
func (rg *RecursiveG) RecurseAndEvaluateFUsingZ(block BlockType, delta12, delta34, deltaAve23 float64, z complex128, deltas []float64, ells []int, maxIter int, tol float64) ([]float64, error) {
	if len(deltas) != len(ells) {
		return nil, fmt.Errorf("deltas and ells must have the same length")
	}

	results := make([]float64, len(deltas))

	// Run the recursion steps
	r, eta := rg.zToREta(z)
	rg.Recurse(delta12, delta34, r, eta, maxIter, tol)

	z1 := 1 - z
	var r1, eta1 float64
	if z1 != z {
		r1, eta1 = rg.zToREta(z1)
		rg.Recurse(delta12, delta34, r1, eta1, maxIter, tol)
	}

	for i := range deltas {
		delta := deltas[i]
		ell := ells[i]

		// g(u,v)
		gUV, err := rg.EvaluateG(delta12, delta34, delta, ell, r, eta)
		if err != nil {
			results[i] = math.NaN()
			continue
		}

		// g(v,u)
		var gVU float64
		if z1 != z {
			gVU, err = rg.EvaluateG(delta12, delta34, delta, ell, r1, eta1)
			if err != nil {
				results[i] = math.NaN()
				continue
			}
		} else {
			gVU = gUV
		}

		if block == BlockMinus {
			gVU *= -1
		}

		u, v := rg.zToUV(z)
		results[i] = math.Pow(v, deltaAve23)*gUV + math.Pow(u, deltaAve23)*gVU
	}

	return results, nil
}

// htilde (Eq. 4.6)
func (rg *RecursiveG) htilde(delta12, delta34 float64, ell int, r, eta float64) float64 {
	coeff := factorial(ell)
	denom := rf(2*rg.Nu, ell)
	gegen := special.GegenbauerC(ell, rg.Nu, eta)

	prefactor := coeff / denom * math.Pow(-1, float64(ell)) * gegen
	f1 := math.Pow(1-r*r, rg.Nu)
	f2 := math.Pow(1+r*r+2*r*eta, 0.5*(1+delta12-delta34))
	f3 := math.Pow(1+r*r-2*r*eta, 0.5*(1-delta12+delta34))

	return prefactor / (f1 * f2 * f3)
}
