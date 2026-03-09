package main

import (
	"fmt"
	"math"
	"slices"
	"sync"
)

// -----------------------------
// Unified evaluator selector
// -----------------------------

type EvaluatorMode string

const (
	NumericMode  EvaluatorMode = "numeric"
	SymbolicMode EvaluatorMode = "symbolic"
)

// NewRetaEvaluator returns either a numeric or symbolic evaluator.
// For SymbolicMode, uses a default maxOrder if nMax < 0.
func NewRetaEvaluator(mode EvaluatorMode) RetaEvaluator {
	return NewRetaEvaluatorWithMaxOrder(mode, -1)
}

// NewRetaEvaluatorWithMaxOrder returns an evaluator with specified max derivative order.
// For SymbolicMode, precomputes derivatives up to (nMax-1) choose 2 at crossing symmetric point.
// If nMax < 0, uses default of 16.
func NewRetaEvaluatorWithMaxOrder(mode EvaluatorMode, nMax int) RetaEvaluator {
	switch mode {
	case NumericMode:
		rnum := NewRDerivativesNumeric()
		etnum := NewEtaDerivativesNumeric()
		return NewNumericRetaEvaluator(rnum, etnum)

	case SymbolicMode:
		// Create symbolic evaluator with precomputation at crossing symmetric point
		// Uses corrected analytic formulas from Appendix C
		maxOrder := 16 // default
		if nMax > 2 {
			maxOrder = (nMax - 1) * (nMax - 2) / 2
		}
		return NewREtaDerivativesSymbolic(true, maxOrder)

	default:
		panic(fmt.Sprintf("unknown evaluator mode: %v", mode))
	}
}

// ComputeDerivativeOrdersZZbar computes (m, n) derivative orders in (z, zbar) space up to C(nMax-1, 2).
func ComputeDerivativeOrdersZZbar(nMax int) [][2]int {
	nMaxDeriv := 2*nMax - 1
	var orders [][2]int

	for m := 0; m <= nMaxDeriv; m++ {
		for n := 0; n <= min(m, nMaxDeriv-m); n++ {
			orders = append(orders, [2]int{m, n})
		}
	}
	return orders
}

// ComputeDerivativeOrdersREta computes the list of (p, q) derivative orders in (r, eta) space
// corresponding to the given (z, zbar) derivative orders.
func ComputeDerivativeOrdersREta(derivativeOrdersZZbar [][2]int) [][2]int {
	derivativeOrdersMap := make(map[[2]int]struct{})

	for _, mn := range derivativeOrdersZZbar {
		m, n := mn[0], mn[1]
		for i := 0; i <= m; i++ {
			for j := 0; j <= n; j++ {
				sum := m + n - i - j
				for p := 0; p <= sum; p++ {
					for q := 0; q <= sum; q++ {
						if p+q <= sum {
							derivativeOrdersMap[[2]int{p, q}] = struct{}{}
						}
					}
				}
			}
		}
	}

	// Convert map keys to slice (unique derivative orders)
	var derivativeOrders [][2]int
	for key := range derivativeOrdersMap {
		derivativeOrders = append(derivativeOrders, key)
	}

	// Lexicographic sort (first by p, then q)
	slices.SortFunc(derivativeOrders, func(a, b [2]int) int {
		if a[0] == b[0] {
			return a[1] - b[1]
		}
		return a[0] - b[0]
	})

	return derivativeOrders
}

// small key types for maps
type tripleKey struct{ A, B, K int }

type ABPair struct {
	A, B int
}

type Partition struct {
	K []int
	L []int
}

// -----------------------------
// Numeric derivative tables
// -----------------------------

// RDerivativesNumeric holds precomputed derivatives for r(z,zbar)
type RDerivativesNumeric struct {
	derivatives map[pairKey]float64
}

func NewRDerivativesNumeric() *RDerivativesNumeric {
	d := map[pairKey]float64{
		{0, 0}: 0.1715728753,
		{0, 1}: 0.2426406871,
		{0, 2}: 0.1005050634,
		{0, 3}: 1.1543289326,
		{0, 4}: 1.8608229691,
		{0, 5}: 42.8993462598,
		{0, 6}: 124.5747209591,
		{0, 7}: 4345.2039257042,
		{0, 8}: 18327.3952888535,
		{0, 9}: 857510.6385313133,
		{1, 1}: 0.3431457505,
		{1, 2}: 0.1421356237,
		{1, 3}: 1.6324676319,
		{1, 4}: 2.6316010801,
		{1, 5}: 60.6688372976,
		{1, 6}: 176.1752599092,
		{1, 7}: 6145.0463230076,
		{1, 8}: 25918.8509804694,
		{1, 9}: 1212703.1748901960,
		{2, 2}: 0.0588745030,
		{2, 3}: 0.6761902332,
		{2, 4}: 1.0900448581,
		{2, 5}: 25.1298552221,
		{2, 6}: 72.9741820090,
		{2, 7}: 2545.3615284007,
		{2, 8}: 10735.9395972373,
		{2, 9}: 502318.1021724199,
		{3, 3}: 7.7662350914,
		{3, 4}: 12.5194719061,
		{3, 5}: 288.6234581193,
		{3, 6}: 838.1290134284,
		{3, 7}: 29234.1933528438,
		{3, 8}: 123305.2870911192,
		{3, 9}: 5769264.74282372,
		{4, 4}: 20.1818738376,
		{4, 5}: 465.2721985427,
		{4, 6}: 1351.0964468420,
		{4, 7}: 47126.6524991831,
		{4, 8}: 198772.9008270586,
		{4, 9}: 9300278.3223660290,
		{5, 5}: 10726.3686453867,
		{5, 6}: 31148.1292234618,
		{5, 7}: 1086456.1633229163,
		{5, 8}: 4582503.3553225324,
		{5, 9}: 214408284.2309231758,
		{6, 6}: 90450.5509926548,
		{6, 7}: 3154942.5615885705,
		{6, 8}: 13307057.7180386782,
		{6, 9}: 622616764.7831015587,
		{7, 7}: 110045350.2815852761,
		{7, 8}: 464154195.9917950630,
		{7, 9}: 21717060969.4115142822,
		{8, 8}: 1957730309.6224517822,
		{8, 9}: 91599190193.3742675781,
		{9, 9}: 4285785232396.5854492188,
	}
	return &RDerivativesNumeric{derivatives: d}
}

// Eval returns d^m dbar^n r at the symmetric point (symmetric lookup)
func (r *RDerivativesNumeric) Eval(m, n int) (float64, error) {
	k := pairKey{m, n}
	if v, ok := r.derivatives[k]; ok {
		return v, nil
	}
	k2 := pairKey{n, m}
	if v, ok := r.derivatives[k2]; ok {
		return v, nil
	}
	return 0, fmt.Errorf("r derivative (%d,%d) not found", m, n)
}

// EtaDerivativesNumeric holds precomputed derivatives for eta(z,zbar)
type EtaDerivativesNumeric struct {
	derivatives map[pairKey]float64
}

func NewEtaDerivativesNumeric() *EtaDerivativesNumeric {
	d := map[pairKey]float64{
		{0, 0}: 1.0,
		{0, 1}: 0.0,
		{0, 2}: 2.0,
		{0, 3}: -6.0,
		{0, 4}: 66.0,
		{0, 5}: -450.0,
		{0, 6}: 6390.0,
		{0, 7}: -68670.0,
		{0, 8}: 1231650.0,
		{0, 9}: -18115650.0,
		{1, 1}: -2.0,
		{1, 2}: 2.0,
		{1, 3}: -18.0,
		{1, 4}: 78.0,
		{1, 5}: -990.0,
		{1, 6}: 8010.0,
		{1, 7}: -132930.0,
		{1, 8}: 1590750.0,
		{1, 9}: -32687550.0,
		{2, 2}: 2.0,
		{2, 3}: 6.0,
		{2, 4}: 54.0,
		{2, 5}: 90.0,
		{2, 6}: 4770.0,
		{2, 7}: -4410.0,
		{2, 8}: 872550.0,
		{2, 9}: -3543750.0,
		{3, 3}: -126.0,
		{3, 4}: 306.0,
		{3, 5}: -6210.0,
		{3, 6}: 33750.0,
		{3, 7}: -784350.0,
		{3, 8}: 6926850.0,
		{3, 9}: -185494000.0,
		{4, 4}: 1314.0,
		{4, 5}: 8910.0,
		{4, 6}: 109350.0,
		{4, 7}: 652050.0,
		{4, 8}: 19249650.0,
		{4, 9}: 79181550.0,
		{5, 5}: -287550.0,
		{5, 6}: 1089450.0,
		{5, 7}: -34898850.0,
		{5, 8}: 233178750.0,
		{5, 9}: -8028294755.6,
		{6, 6}: 8752050.0,
		{6, 7}: 93583349.95,
		{6, 8}: 1499289754.5,
		{6, 9}: 15154633907.0,
		{7, 7}: -4119623546.23,
		{7, 8}: 21151793137.0,
		{7, 9}: -928576320832.0,
		{8, 8}: 251718941164.0,
	}
	return &EtaDerivativesNumeric{derivatives: d}
}

// Eval returns d^m dbar^n eta at the symmetric point (symmetric lookup)
func (e *EtaDerivativesNumeric) Eval(m, n int) (float64, error) {
	k := pairKey{m, n}
	if v, ok := e.derivatives[k]; ok {
		return v, nil
	}
	k2 := pairKey{n, m}
	if v, ok := e.derivatives[k2]; ok {
		return v, nil
	}
	return 0, fmt.Errorf("eta derivative (%d,%d) not found", m, n)
}

// -----------------------------
// Evaluator interface and numeric adapter
// -----------------------------

// RetaEvaluator is an interface that can evaluate r and eta derivatives at any (z,zbar).
// The Python code used different backends (numeric, symbolic, jax); here we provide a numeric adapter.
type RetaEvaluator interface {
	EvalR(m, n int, z, zbar float64) (float64, error)
	EvalEta(m, n int, z, zbar float64) (float64, error)
}

// NumericRetaEvaluator adapts the two numeric tables to the interface.
// It ignores z,zbar (assumes evaluation at crossing symmetric point).
type NumericRetaEvaluator struct {
	R *RDerivativesNumeric
	E *EtaDerivativesNumeric
}

func NewNumericRetaEvaluator(r *RDerivativesNumeric, e *EtaDerivativesNumeric) *NumericRetaEvaluator {
	return &NumericRetaEvaluator{R: r, E: e}
}

func (nre *NumericRetaEvaluator) EvalR(m, n int, z, zbar float64) (float64, error) {
	// the numeric tables are for the crossing symmetric point (0.5,0.5).
	// We ignore z,zbar and return table values. If desired, you can check z,zbar.
	return nre.R.Eval(m, n)
}

func (nre *NumericRetaEvaluator) EvalEta(m, n int, z, zbar float64) (float64, error) {
	return nre.E.Eval(m, n)
}

// -----------------------------
// Cached wrapper (equivalent to Python @cache methods)
// -----------------------------

type RetaCache struct {
	evaluator RetaEvaluator
	zStar     float64
	zbarStar  float64

	// concurrent caches
	rDerivCache        sync.Map // key: pairKey, value: float64
	etaDerivCache      sync.Map // key: pairKey, value: float64
	rDerivToPowCache   sync.Map // key: tripleKey, value: float64
	etaDerivToPowCache sync.Map // key: tripleKey, value: float64

}

func NewRetaCache(eval RetaEvaluator, zStar, zbarStar float64) *RetaCache {
	return &RetaCache{
		evaluator: eval,
		zStar:     zStar,
		zbarStar:  zbarStar,
	}
}

// RDerivCache is equivalent to Python's @cache def r_deriv_cache(self, a,b)
func (rc *RetaCache) RDerivCache(a, b int) (float64, error) {
	k := pairKey{a, b}

	if v, ok := rc.rDerivCache.Load(k); ok {
		return v.(float64), nil
	}

	v, err := rc.evaluator.EvalR(a, b, rc.zStar, rc.zbarStar)
	if err != nil {
		return 0, err
	}

	// Store result in sync.Map
	rc.rDerivCache.Store(k, v)
	return v, nil
}

// EtaDerivCache is equivalent to Python's @cache def eta_deriv_cache(self, a,b)
// EtaDerivCache uses sync.Map
func (rc *RetaCache) EtaDerivCache(a, b int) (float64, error) {
	k := pairKey{a, b}

	if v, ok := rc.etaDerivCache.Load(k); ok {
		return v.(float64), nil
	}

	v, err := rc.evaluator.EvalEta(a, b, rc.zStar, rc.zbarStar)
	if err != nil {
		return 0, err
	}

	rc.etaDerivCache.Store(k, v)
	return v, nil
}

// RDerivToPowCache caches r_deriv_cache(a,b) ** k
func (rc *RetaCache) RDerivToPowCache(a, b, kpow int) (float64, error) {
	k := tripleKey{a, b, kpow}

	if v, ok := rc.rDerivToPowCache.Load(k); ok {
		return v.(float64), nil
	}

	base, err := rc.RDerivCache(a, b)
	if err != nil {
		return 0, err
	}

	val := math.Pow(base, float64(kpow))
	rc.rDerivToPowCache.Store(k, val)
	return val, nil
}

// EtaDerivToPowCache caches eta_deriv_cache(a,b) ** k
// EtaDerivToPowCache uses sync.Map
func (rc *RetaCache) EtaDerivToPowCache(a, b, kpow int) (float64, error) {
	k := tripleKey{a, b, kpow}

	if v, ok := rc.etaDerivToPowCache.Load(k); ok {
		return v.(float64), nil
	}

	base, err := rc.EtaDerivCache(a, b)
	if err != nil {
		return 0, err
	}

	val := math.Pow(base, float64(kpow))
	rc.etaDerivToPowCache.Store(k, val)
	return val, nil
}

// rEtaDerivativesSymbolic computes derivatives of r(z, zbar) and eta(z, zbar) symbolically.
// Uses the corrected analytic formulas from Appendix C of the paper.
// Formulas match equations for:
//
//	∂_m ∂̄_n r(z,z̄)  [final formula near end of appendix]
//	∂_m ∂̄_n η(z,z̄)  [derived from φ_1 derivatives]
type REtaDerivativesSymbolic struct {
	// Runtime caches for arbitrary z, zbar
	rCache   sync.Map // key: pairKey, value: float64
	etaCache sync.Map // key: pairKey, value: float64
	phiCache sync.Map // key: phiKey, value: float64
	fmzCache sync.Map // key: fzKey, value: float64

	// Precomputed tables at crossing symmetric point z = zbar = 0.5
	rTableCrossingPoint   map[[2]int]float64
	etaTableCrossingPoint map[[2]int]float64
}

// pairKey, tripleKey, etc.
type pairKey struct {
	M, N int
}
type phiKey struct {
	M, N      int
	Z, Zbar   float64
	Direction string
}
type fzKey struct {
	M int
	Z float64
}

// NewREtaDerivativesSymbolic creates a new symbolic evaluator with caches.
// If precomputeCrossingPoint is true, precomputes values at z = zbar = 0.5 for efficiency.
// maxOrder specifies the maximum derivative order to precompute (computes up to m+n <= maxOrder).
func NewREtaDerivativesSymbolic(precomputeCrossingPoint bool, maxOrder int) *REtaDerivativesSymbolic {
	rd := &REtaDerivativesSymbolic{
		rTableCrossingPoint:   make(map[[2]int]float64),
		etaTableCrossingPoint: make(map[[2]int]float64),
	}

	if precomputeCrossingPoint {
		rd.populateCrossingPointTables(maxOrder)
	}

	return rd
}

// populateCrossingPointTables precomputes derivatives at crossing symmetric point z = zbar = 0.5.
// Uses the analytic formulas from Appendix C for maximum accuracy.
// maxOrder specifies the maximum sum of derivative orders (m+n) to precompute.
func (rd *REtaDerivativesSymbolic) populateCrossingPointTables(maxOrder int) {
	const zStar = 0.5
	const zbarStar = 0.5

	for m := 0; m <= maxOrder; m++ {
		for n := 0; n <= maxOrder; n++ {
			if m+n > maxOrder {
				continue
			}

			// Compute using symbolic formulas at crossing symmetric point
			rVal, err := rd.evalRUncached(m, n, zStar, zbarStar)
			if err == nil {
				rd.rTableCrossingPoint[[2]int{m, n}] = rVal
			}

			etaVal, err := rd.evalEtaUncached(m, n, zStar, zbarStar)
			if err == nil {
				rd.etaTableCrossingPoint[[2]int{m, n}] = etaVal
			}
		}
	}
}

// PairSplits yields all (m1, m2, n1, n2) with m1+m2=m and n1+n2=n
func (rd *REtaDerivativesSymbolic) PairSplits(m, n int) [][4]int {
	var splits [][4]int
	for m1 := 0; m1 <= m; m1++ {
		m2 := m - m1
		for n1 := 0; n1 <= n; n1++ {
			n2 := n - n1
			splits = append(splits, [4]int{m1, m2, n1, n2})
		}
	}
	return splits
}

// ---------------------- eval_r ----------------------
// Implements: ∂_m ∂̄_n r(z,z̄) = Σ_{m1+m2=m, n1+n2=n} [m!n!/(m1!m2!n1!n2!)] (1/2)_{(m1)} (1/2)_{(n1)} z^(1/2-m1) z̄^(1/2-n1) ψ̃_{m2}(z) ψ̃_{n2}(z̄)

func (rd *REtaDerivativesSymbolic) EvalR(m, n int, z, zbar float64) (float64, error) {
	// Check precomputed table for crossing symmetric point
	const eps = 1e-10
	if math.Abs(z-0.5) < eps && math.Abs(zbar-0.5) < eps {
		if val, ok := rd.rTableCrossingPoint[[2]int{m, n}]; ok {
			return val, nil
		}
	}

	// Check runtime cache
	k := pairKey{m, n}
	if v, ok := rd.rCache.Load(k); ok {
		return v.(float64), nil
	}

	return rd.evalRUncached(m, n, z, zbar)
}

func (rd *REtaDerivativesSymbolic) evalRUncached(m, n int, z, zbar float64) (float64, error) {
	k := pairKey{m, n}

	sum := 0.0
	splits := rd.PairSplits(m, n)

	for _, s := range splits {
		m1, m2, n1, n2 := s[0], s[1], s[2], s[3]

		term := ff(0.5, m1) * ff(0.5, n1)
		term *= math.Pow(z, 0.5-float64(m1)) * math.Pow(zbar, 0.5-float64(n1))

		fmz1 := rd.FmZ(m2, z)
		fmz2 := rd.FmZ(n2, zbar)

		term *= fmz1 * fmz2
		term /= factorial(m1) * factorial(m2) * factorial(n1) * factorial(n2)
		sum += term
	}

	sum *= factorial(m) * factorial(n)

	// Store result in sync.Map
	rd.rCache.Store(k, sum)

	return sum, nil
}

// ---------------------- eval_eta ----------------------
// Implements: ∂_m ∂̄_n η(z,z̄) = 1/2 (∂_m ∂̄_n φ_1(z,z̄) + ∂_m ∂̄_n φ_1(z̄,z))

func (rd *REtaDerivativesSymbolic) EvalEta(m, n int, z, zbar float64) (float64, error) {
	// Check precomputed table for crossing symmetric point
	const eps = 1e-10
	if math.Abs(z-0.5) < eps && math.Abs(zbar-0.5) < eps {
		if val, ok := rd.etaTableCrossingPoint[[2]int{m, n}]; ok {
			return val, nil
		}
	}

	// Check runtime cache
	k := pairKey{m, n}
	if v, ok := rd.etaCache.Load(k); ok {
		return v.(float64), nil
	}

	return rd.evalEtaUncached(m, n, z, zbar)
}

func (rd *REtaDerivativesSymbolic) evalEtaUncached(m, n int, z, zbar float64) (float64, error) {
	k := pairKey{m, n}

	// Compute left and right phi derivatives
	left, err := rd.Phi3Derivative(m, n, z, zbar, "left")
	if err != nil {
		return 0, err
	}
	right, err := rd.Phi3Derivative(m, n, z, zbar, "right")
	if err != nil {
		return 0, err
	}

	val := 0.5 * (left + right)

	// Store result in sync.Map
	rd.etaCache.Store(k, val)

	return val, nil
}

// ---------------------- phi_3_derivative ----------------------
// Computes ∂_m ∂̄_n φ_1(z,z̄) where φ_1 = φ_3(z,z̄)/φ_3(z̄,z)
// Direction "left": φ_1(z,z̄), Direction "right": φ_1(z̄,z)
// Formula: Σ_{i,j} C(m,i)C(n,j) (1/2)_{(i)} z^(1/2-i) [δ_{j,0} + (-1)^j(1/2)_{(j)}(1-z̄)^(1/2-j)] ψ_{m-i,n-j}(z̄,z)

func (rd *REtaDerivativesSymbolic) Phi3Derivative(m, n int, z, zbar float64, direction string) (float64, error) {
	k := phiKey{M: m, N: n, Z: z, Zbar: zbar, Direction: direction}

	// Check cache first
	if v, ok := rd.phiCache.Load(k); ok {
		return v.(float64), nil
	}

	sum := 0.0

	if direction == "left" {
		for i := 0; i <= m; i++ {
			fac_i := comb(m, i) * ff(0.5, i) * math.Pow(z, 0.5-float64(i))
			fmz := rd.FmZ(m-i, z)
			fac_i *= fmz

			for j := 0; j <= n; j++ {
				deltaJ0 := 0.0
				if j == 0 {
					deltaJ0 = 1.0
				}
				term := fac_i * comb(n, j) * ff(-0.5, n-j)
				term *= math.Pow(zbar, -0.5-float64(n)+float64(j))
				term *= deltaJ0 + math.Pow(-1, float64(j))*ff(0.5, j)*math.Pow(1-zbar, 0.5-float64(j))
				sum += term
			}
		}
		rd.phiCache.Store(k, sum)
		return sum, nil
	}

	for i := 0; i <= m; i++ {
		deltaI0 := 0.0
		if i == 0 {
			deltaI0 = 1.0
		}
		fac_i := comb(m, i) * ff(-0.5, m-i) * math.Pow(z, -0.5-float64(m)+float64(i))
		fac_i *= deltaI0 + math.Pow(-1, float64(i))*ff(0.5, i)*math.Pow(1-z, 0.5-float64(i))
		for j := 0; j <= n; j++ {
			term := fac_i * comb(n, j) * ff(0.5, j)
			fmz := rd.FmZ(n-j, zbar)
			term *= math.Pow(zbar, 0.5-float64(j)) * fmz
			sum += term
		}
	}

	rd.phiCache.Store(k, sum)
	return sum, nil
}

// ---------------------- f_m_z ----------------------
// FmZ computes ψ̃_m(z) = ∂_m^z [(1 + √(1-z))^{-1}] using Faà di Bruno formula
// This is ψ̃ in the notation of the appendix
func (rd *REtaDerivativesSymbolic) FmZ(m int, z float64) float64 {
	totalSum := 0.0

	for k := 0; k <= m; k++ {
		// Step 1: Build (i, 0) pairs for partitions
		abPairs := make([]Pair, 0)
		for i := 1; i <= m-k+1; i++ {
			abPairs = append(abPairs, Pair{A: i, B: 0})
		}

		// Step 2: Generate partitions
		partitions := GeneratePartitions(abPairs, k, 0, m, 0)
		bellPoly := 0.0

		// Step 3: Compute Bell polynomial
		for _, part := range partitions {
			kTuple := part.K // corresponds to j_i in Python
			prod := 1.0

			for i, j_i := range kTuple {
				if j_i == 0 {
					continue
				}
				order := i + 1 // because Python's enumerate starts at 1

				// fac_i = [ff(0.5, i) * (-1)^i * (1 - z)^(0.5 - i)]^j_i / [j_i! * (i!)^j_i]
				ffVal := ff(0.5, order)
				sign := math.Pow(-1, float64(order))
				powTerm := math.Pow(1-z, 0.5-float64(order))

				num := math.Pow(ffVal*sign*powTerm, float64(j_i))
				den := factorial(j_i) * math.Pow(factorial(order), float64(j_i))

				prod *= num / den
			}

			bellPoly += prod
		}

		// Step 4: Combine terms
		ffTerm := ff(-1, k)
		rootTerm := math.Pow(1+math.Sqrt(1-z), -1-float64(k))
		totalSum += ffTerm * rootTerm * bellPoly
	}

	totalSum *= factorial(m)
	return totalSum
}
