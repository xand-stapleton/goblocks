package main

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

// ----------- maths helpers -----------

// Factorial function using a loop
func factorial(n int) float64 {
	if n < 0 {
		return math.NaN() // Return NaN for negative inputs
	}
	result := 1.0
	for i := 1; i <= n; i++ {
		result *= float64(i)
	}
	return result
}

// comb returns n choose k as float64.
// Returns NaN for invalid inputs (negative, k>n).
func comb(n, k int) float64 {
	// invalid inputs
	if n < 0 || k < 0 || k > n {
		return math.NaN()
	}
	// symmetry: C(n,k) == C(n, n-k)
	if k > n-k {
		k = n - k
	}
	// compute result by multiplying numerator terms and dividing by denominator terms
	// to reduce overflow compared to computing full factorials.
	if k == 0 {
		return 1.0
	}
	num := 1.0
	den := 1.0
	for i := 1; i <= k; i++ {
		num *= float64(n - (k - i))
		den *= float64(i)
	}
	return num / den
}

// Rising factorial (x)_n = x * (x + 1) * ... * (x + n - 1)
func rf(x float64, n int) float64 {
	if n < 0 {
		return math.NaN() // Return NaN for negative inputs
	}
	result := 1.0
	for i := 0; i < n; i++ {
		result *= (x + float64(i))
	}
	return result
}

// Falling factorial = x * (x - 1) * ... * (x - n + 1)
func ff(x float64, n int) float64 {
	if n < 0 {
		return math.NaN() // Return NaN for negative n
	}
	result := 1.0
	for i := 0; i < n; i++ {
		result *= (x - float64(i))
	}
	return result
}

func powInt(a float64, n int) float64 {
	// integer power with sign support
	res := 1.0
	switch {
	case n < 0:
		for i := 0; i < -n; i++ {
			res *= a
		}
		return 1 / res
	case n == 0:
		return 1
	default:
		for i := 0; i < n; i++ {
			res *= a
		}
		return res
	}
}

func negOnePow(k int) float64 {
	if k%2 == 0 || k == 0.0 {
		return 1.0
	}
	return -1.0
}

func defaultIfZeroInt(x, def int) int {
	if x == 0 {
		return def
	}
	return x
}

func defaultIfZeroFloat(x, def float64) float64 {
	if x == 0 {
		return def
	}
	return x
}

func floatEquals(a, b float64) bool {
	return math.Abs(a-b) < 1e-12
}

// GegenbauerC returns the nth Gegenbauer polynomial with paramater a at x.
//
// See http://mathworld.wolfram.com/GegenbauerPolynomial.html for more information.
func GegenbauerC(n int, a, x float64) float64 {
	switch {
	case math.IsNaN(a) || math.IsNaN(x):
		return math.NaN()
	case n < 0:
		return 0.0
	case n == 0:
		return 1.0
	case n == 1:
		return 2 * a * x
	}

	tmp := 1.0
	res := 2 * a * x
	for k := 1; k < n; k++ {
		p := 2 * (float64(k) + a) * x
		q := float64(k-1) + 2*a
		res, tmp = (p*res-q*tmp)/float64(k+1), res
	}
	return res
}

// Helper: min of two ints
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func flatten3D(arr [][][]float64) []float64 {
	var flat []float64
	for _, matrix := range arr {
		for _, row := range matrix {
			for _, val := range row {
				flat = append(flat, val)
			}
		}
	}
	return flat
}

// ApplyMatrixToColumnInPlace multiplies the idx-th column of dst by M,
// writing the result back into that column.
// Equivalent to: dst[:, idx] = M @ dst[:, idx]
func ApplyMatrixToColumnInPlace(M, dst *mat.Dense, idx int) {
	rm, cm := M.Dims()
	r, c := dst.Dims()
	if rm != r || rm != cm {
		panic("ApplyMatrixToColumnInPlace: dimension mismatch")
	}
	if idx < 0 || idx >= c {
		panic("ApplyMatrixToColumnInPlace: column index out of range")
	}

	mdata := M.RawMatrix().Data
	d := dst.RawMatrix()
	ddata := d.Data
	stride := d.Stride

	// Temporary accumulator
	tmp := make([]float64, r)

	// Compute M * dst[:, idx]
	for i := 0; i < r; i++ {
		sum := 0.0
		for j := 0; j < r; j++ {
			sum += mdata[i*cm+j] * ddata[j*stride+idx]
		}
		tmp[i] = sum
	}

	// Write back
	for i := 0; i < r; i++ {
		ddata[i*stride+idx] = tmp[i]
	}
}

// ApplyMatrixToRowInPlace multiplies the idx-th row of dst by M,
// writing the result back into that row.
// Equivalent to: dst[idx, :] = dst[idx, :] @ M
func ApplyMatrixToRowInPlace(M, dst *mat.Dense, idx int) {
	rm, cm := M.Dims()
	r, c := dst.Dims()
	if cm != c || rm != cm {
		panic("ApplyMatrixToRowInPlace: dimension mismatch")
	}
	if idx < 0 || idx >= r {
		panic("ApplyMatrixToRowInPlace: row index out of range")
	}

	mdata := M.RawMatrix().Data
	d := dst.RawMatrix()
	ddata := d.Data
	stride := d.Stride

	tmp := make([]float64, c)

	// Compute dst[idx, :] * M
	for j := 0; j < c; j++ {
		sum := 0.0
		for k := 0; k < c; k++ {
			sum += ddata[idx*stride+k] * mdata[k*cm+j]
		}
		tmp[j] = sum
	}

	// Write back
	for j := 0; j < c; j++ {
		ddata[idx*stride+j] = tmp[j]
	}
}

// MatVecColumn multiplies M by the idx-th column of dg
// and returns the result as a new *mat.Dense (a column vector).
// Equivalent to: term = M @ dg[:, idx]
func MatVecColumn(M1, M2 *mat.Dense, idx int) *mat.Dense {
	rm, cm := M1.Dims()
	r, c := M2.Dims()
	if rm != r || cm != r {
		panic("MatVecColumn: dimension mismatch")
	}
	if idx < 0 || idx >= c {
		panic("MatVecColumn: column index out of range")
	}

	mdata := M1.RawMatrix().Data
	d := M2.RawMatrix()
	ddata := d.Data
	stride := d.Stride

	// Allocate result as a column vector (r×1)
	out := mat.NewDense(r, 1, nil)

	for i := 0; i < r; i++ {
		sum := 0.0
		for j := 0; j < r; j++ {
			sum += mdata[i*cm+j] * ddata[j*stride+idx]
		}
		out.Set(i, 0, sum)
	}

	return out
}

// ApplyAddToColumnInPlace adds the vector 'term' to the idx-th column of dst.
// Equivalent to: dst[:, idx] += term
func ApplyAddToColumnInPlace(dst *mat.Dense, idx int, term *mat.Dense) {
	r, c := dst.Dims()
	tr, tc := term.Dims()
	if tc != 1 {
		panic("ApplyAddToColumnInPlace: term must be a column vector (n×1)")
	}
	if tr != r {
		panic("ApplyAddToColumnInPlace: dimension mismatch")
	}
	if idx < 0 || idx >= c {
		panic("ApplyAddToColumnInPlace: column index out of range")
	}

	d := dst.RawMatrix()
	ddata := d.Data
	stride := d.Stride
	tdata := term.RawMatrix().Data

	for i := 0; i < r; i++ {
		ddata[i*stride+idx] += tdata[i]
	}
}

// GeneratePartitions computes all (k_tuple, l_tuple) pairs satisfying the constraints.
func GeneratePartitions(abPairs []Pair, targetP, targetQ, m, n int) []Result {
	N := len(abPairs)
	memo := make(map[Key][]Result)

	var rec func(idx, sumP, sumQ, sumA, sumB int) []Result
	rec = func(idx, sumP, sumQ, sumA, sumB int) []Result {
		key := Key{idx, sumP, sumQ, sumA, sumB}
		if val, ok := memo[key]; ok {
			return val
		}

		// Base case
		if idx == N {
			if sumP == targetP && sumQ == targetQ && sumA == m && sumB == n {
				return []Result{{K: []int{}, L: []int{}}}
			}
			return nil
		}

		a := abPairs[idx].A
		b := abPairs[idx].B

		remP := targetP - sumP
		remQ := targetQ - sumQ
		if remP < 0 || remQ < 0 {
			return nil
		}

		maxByA := remP + remQ
		if a > 0 {
			tmp := (m - sumA) / a
			if tmp < maxByA {
				maxByA = tmp
			}
		}
		maxByB := remP + remQ
		if b > 0 {
			tmp := (n - sumB) / b
			if tmp < maxByB {
				maxByB = tmp
			}
		}

		maxTotal := maxByA
		if maxByB < maxTotal {
			maxTotal = maxByB
		}
		if remP+remQ < maxTotal {
			maxTotal = remP + remQ
		}

		results := []Result{}

		for total := 0; total <= maxTotal; total++ {
			kmin := 0
			if total-remQ > kmin {
				kmin = total - remQ
			}
			kmax := total
			if remP < kmax {
				kmax = remP
			}

			for k := kmin; k <= kmax; k++ {
				ell := total - k
				newSumA := sumA + a*total
				newSumB := sumB + b*total
				if newSumA > m || newSumB > n {
					continue
				}

				tails := rec(idx+1, sumP+k, sumQ+ell, newSumA, newSumB)
				if tails == nil {
					continue
				}

				for _, tail := range tails {
					kTuple := append([]int{k}, tail.K...)
					lTuple := append([]int{ell}, tail.L...)
					results = append(results, Result{K: kTuple, L: lTuple})
				}
			}
		}

		memo[key] = results
		return results
	}

	return rec(0, 0, 0, 0, 0)
}

func sumInt(xs []int) int {
	s := 0
	for _, v := range xs {
		s += v
	}
	return s
}
