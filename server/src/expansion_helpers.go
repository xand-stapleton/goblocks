package main

import (
	"fmt"
	"math"

	"github.com/scientificgo/special"
)

// HTildeTaylorConfig controls derivative-based extraction of the Taylor
// coefficients of htilde(r, eta_fixed) around r = R0.
type HTildeTaylorConfig struct {
	Delta12 float64
	Delta34 float64
	Ell     int
	Eta     float64
	Order   int
	R0      float64
}

// HTildeTaylorCoefficients computes coefficients c_n in
// htilde(r, eta) = sum_{n=0}^{Order} c_n (r - R0)^n
// using analytic derivatives wrt (r, eta):
// c_n = (1/n!) * d^n/dr^n htilde(r, eta)|_{r=R0, eta=fixed}.
func (rd *RecursiveDerivatives) HTildeTaylorCoefficients(cfg HTildeTaylorConfig) ([]float64, error) {
	if cfg.Order < 0 {
		return nil, fmt.Errorf("order must be >= 0")
	}
	if cfg.Ell < 0 {
		return nil, fmt.Errorf("ell must be >= 0")
	}

	alpha := (1 + cfg.Delta12 - cfg.Delta34) / 2
	beta := (1 - cfg.Delta12 + cfg.Delta34) / 2

	coeffs := make([]float64, cfg.Order+1)
	for n := 0; n <= cfg.Order; n++ {
		deriv := rd.derivativeHTilde(
			n,
			0,
			cfg.Ell,
			cfg.R0,
			cfg.Eta,
			rd.recursiveG.Nu,
			alpha,
			beta,
		)
		coeffs[n] = deriv / rd.factorial(n)
	}
	return coeffs, nil
}

// HTildeTaylorCoefficientsAtEtaDeriv is a convenience wrapper on RecursiveG
// that builds a temporary derivative engine and computes coefficients
// around r = 0 at fixed eta.
//
// NOTE: This depends on the derivative engine (and its current limitations).
func (rg *RecursiveG) HTildeTaylorCoefficientsAtEtaDeriv(delta12, delta34 float64, ell int, eta float64, order int) ([]float64, error) {
	if order < 0 {
		return nil, fmt.Errorf("order must be >= 0")
	}
	rd := NewRecursiveDerivatives(*rg, 4, false, false, false)
	return rd.HTildeTaylorCoefficients(HTildeTaylorConfig{
		Delta12: delta12,
		Delta34: delta34,
		Ell:     ell,
		Eta:     eta,
		Order:   order,
		R0:      0.0,
	})
}

func polyAddScaledTrunc(dst []float64, scale float64, src []float64, order int) {
	n := len(src)
	if n > order+1 {
		n = order + 1
	}
	for i := 0; i < n; i++ {
		dst[i] += scale * src[i]
	}
}

func polyMulTrunc(a, b []float64, order int) []float64 {
	out := make([]float64, order+1)
	maxI := len(a)
	if maxI > order+1 {
		maxI = order + 1
	}
	maxJ := len(b)
	if maxJ > order+1 {
		maxJ = order + 1
	}
	for i := 0; i < maxI; i++ {
		ai := a[i]
		if ai == 0 {
			continue
		}
		for j := 0; j < maxJ && i+j <= order; j++ {
			out[i+j] += ai * b[j]
		}
	}
	return out
}

// HTildeTaylorCoefficientsAtEta computes coefficients c_n in
// htilde(r, eta) = sum_{n=0}^{order} c_n r^n
// by expanding the closed-form expression of htilde around r=0.
//
// This avoids the derivative engine and is stable for moderate orders.
func (rg *RecursiveG) HTildeTaylorCoefficientsAtEta(delta12, delta34 float64, ell int, eta float64, order int) ([]float64, error) {
	if order < 0 {
		return nil, fmt.Errorf("order must be >= 0")
	}
	if ell < 0 {
		return nil, fmt.Errorf("ell must be >= 0")
	}

	alpha := (1 + delta12 - delta34) / 2
	beta := (1 - delta12 + delta34) / 2

	// A(r) = (1 - r^2)^(-nu) = sum_{m>=0} (nu)_m/m! * r^{2m}
	A := make([]float64, order+1)
	for m := 0; 2*m <= order; m++ {
		A[2*m] = rf(rg.Nu, m) / factorial(m)
	}

	buildB := func(exp, u1, u2 float64) []float64 {
		B := make([]float64, order+1)
		u := make([]float64, 3)
		u[0] = 0
		u[1] = u1
		u[2] = u2
		uPow := []float64{1.0}
		for m := 0; m <= order; m++ {
			// (1+u)^(-exp) = sum_m (-1)^m (exp)_m / m! * u^m
			coef := rf(exp, m) / factorial(m)
			if m%2 == 1 {
				coef = -coef
			}
			polyAddScaledTrunc(B, coef, uPow, order)
			uPow = polyMulTrunc(uPow, u, order)
		}
		return B
	}

	Bplus := buildB(alpha, 2*eta, 1.0)
	Bminus := buildB(beta, -2*eta, 1.0)

	series := polyMulTrunc(polyMulTrunc(A, Bplus, order), Bminus, order)

	// r-independent prefactor
	coeff := factorial(ell)
	denom := rf(2*rg.Nu, ell)
	gegen := special.GegenbauerC(ell, rg.Nu, eta)
	prefactor := coeff / denom * math.Pow(-1, float64(ell)) * gegen
	for i := range series {
		series[i] *= prefactor
	}

	return series, nil
}

// EvalTaylorSeries evaluates sum_n c_n (r-r0)^n.
func EvalTaylorSeries(coeffs []float64, r float64, r0 float64) float64 {
	out := 0.0
	x := r - r0
	for i := len(coeffs) - 1; i >= 0; i-- {
		out = out*x + coeffs[i]
	}
	return out
}
