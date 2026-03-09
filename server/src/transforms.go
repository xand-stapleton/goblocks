package main

import (
	// "math"
	"math/cmplx"
)

// z -> (r, eta) (Eq. 4.3)
func (rg *RecursiveG) zToREta(z complex128) (float64, float64) {
	// r e^{iθ} = z / (1 + sqrt(1 - z))^2
	den := 1 + cmplx.Sqrt(1-z)
	rExp := z / (den * den)

	r := cmplx.Abs(rExp)
	var eta float64
	if r != 0 {
		eta = real(rExp) / r
	} else {
		eta = 0
	}
	return r, eta
}

// z -> (u, v) with z* = conj(z), Euclidean signature
func (rg *RecursiveG) zToUV(z complex128) (float64, float64) {
	zStar := cmplx.Conj(z)
	u := z * zStar
	v := (1 - z) * (1 - zStar)
	return real(u), real(v)
}

// func rEtaToZ(r, eta float64) complex128 {
// 	sign := 1.0
// 	// sin(theta) with chosen sign
// 	s := sign * math.Sqrt(math.Max(0, 1-eta*eta))

// 	// rho = r (eta + i s)
// 	rho := complex(r*eta, r*float64(s))

// 	// z = 4 rho / (1 + rho)^2
// 	num := 4 * rho
// 	den := (1 + rho) * (1 + rho)

// 	return num / den
// }
