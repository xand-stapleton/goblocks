package main

import (
	"fmt"
	"log"
	"math"

	"github.com/scientificgo/special"
)

// ----------- core API (port of Python) -----------

// RecurseLegacy runs the original fixed-point iteration algorithm for h.
//
// NOTE: This is retained for backwards-compatibility and benchmarking.
func (rg *RecursiveG) RecurseLegacy(delta12, delta34, r, eta float64, maxIter int, tol float64) {
	// Reset unique poles
	rg.unique_poles_map = make(map[PoleKey]int)
	rg.idxToKey = nil

	polesData := rg.getAllPolesData(delta12, delta34)
	numKeys := len(rg.idxToKey)
	if numKeys == 0 {
		panic("RecurseLegacy: idxToKey is empty")
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

// Recurse runs an order-by-order (in r) recursion for h.
//
// At recursion order p, it uses the Taylor expansion of htilde up to order p
// (computed via expansion_helpers.go) and only includes pole terms with N <= p
// in the r^N * c * h_child expansion.
//
// maxIter is interpreted as the maximum Taylor order. If tol > 0, the recursion
// tolerances are only checked after at least 2*k2max iterations and the function
// stops early once the newest Taylor term is smaller than tol (relative to the
// current partial sum) for all cached keys.
func (rg *RecursiveG) Recurse(delta12, delta34, r, eta float64, maxIter int, tol float64) {
	if maxIter < 0 {
		panic("Recurse: maxIter must be >= 0")
	}

	// Reset unique poles
	rg.unique_poles_map = make(map[PoleKey]int)
	rg.idxToKey = nil

	// This call also populates rg.idxToKey via getOrAddPoleIdx.
	polesData := rg.getAllPolesData(delta12, delta34)
	numKeys := len(rg.idxToKey)
	if numKeys == 0 {
		panic("Recurse: idxToKey is empty")
	}

	// Precompute Taylor coefficients for htilde per ell.
	coeffsByEll := make(map[int][]float64)
	for _, key := range rg.idxToKey {
		if _, ok := coeffsByEll[key.Ell]; ok {
			continue
		}
		coeffs, err := rg.HTildeTaylorCoefficientsAtEta(delta12, delta34, key.Ell, eta, maxIter)
		if err != nil {
			panic(fmt.Sprintf("Recurse: failed to compute htilde Taylor coefficients (ell=%d, order=%d): %v", key.Ell, maxIter, err))
		}
		coeffsByEll[key.Ell] = coeffs
	}

	// r powers for evaluation of the partial sums.
	rPows := make([]float64, maxIter+1)
	rPows[0] = 1.0
	for n := 1; n <= maxIter; n++ {
		rPows[n] = rPows[n-1] * r
	}

	// hCoeffs[idx][n] stores the Taylor coefficient of h for idxToKey[idx] at order n.
	hCoeffs := make([][]float64, numKeys)
	for i := range hCoeffs {
		hCoeffs[i] = make([]float64, maxIter+1)
	}

	// Evaluate partial sums at the requested r as we build coefficients.
	hPartial := make([]float64, numKeys)

	// Ensure tolerance is not checked until at least 2*k2max iterations
	minIterForTolerance := 2 * rg.K2Max

	converged := false
	var maxFrac float64
	for n := 0; n <= maxIter; n++ {
		maxFrac = 0.0
		for idx, key := range rg.idxToKey {
			val := coeffsByEll[key.Ell][n]
			for _, p := range polesData[key.Ell] {
				if p.N > n {
					continue
				}
				denom := key.Delta - p.Delta
				if denom == 0.0 {
					continue
				}
				val += p.C * hCoeffs[p.Idx][n-p.N] / denom
			}
			hCoeffs[idx][n] = val

			term := val * rPows[n]
			hPartial[idx] += term

			if tol > 0 && n >= minIterForTolerance {
				den := math.Abs(hPartial[idx])
				if den < 1.0 {
					den = 1.0
				}
				frac := math.Abs(term) / den
				if frac > maxFrac {
					maxFrac = frac
				}
			}
		}

		if tol > 0 && n >= minIterForTolerance && maxFrac < tol {
			converged = true
			break
		}
	}

	if tol <= 0 {
		converged = true
	}
	if !converged {
		log.Printf("Recurse series did not reach tolerance: maxFrac=%v (tol=%v)\n", maxFrac, tol)
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
		HFinal:     append([]float64(nil), hPartial...),    // copy for safety
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
