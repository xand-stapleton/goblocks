package main

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
)

// NOTE: This file is broadly a line-by-line Go translation of the Python
// RecursiveDerivatives class.

// FType represents function types for conformal block computation
type FType int

const (
	F_PLUS  FType = iota // 0
	F_MINUS              // 1
)

// DerivKey represents derivative order keys
type DerivKey struct {
	M, N int
}

// PolesDataMatrix holds pole information for a given spin
type PolesDataMatrix struct {
	Delta float64
	C     float64
	Idx   int
	Mat   *mat.Dense
}

// Pair represents a tuple (a_i, b_i)
type Pair struct {
	A, B int
}

// Key is used for memoization in recursive computations
type Key struct {
	Idx, SumP, SumQ, SumA, SumB int
}

// Result represents a pair of integer slices (k_tuple, l_tuple)
type Result struct {
	K []int
	L []int
}

// phi123Key represents cache key for phi123 derivative computations
type phi123Key struct {
	I, J        int
	R, Eta, Nu  float64
	Alpha, Beta float64
}

// ConvergedBlockDerivativeData holds final results after recursion
type ConvergedBlockDerivativeData struct {
	Delta12   float64
	Delta34   float64
	PolesData map[int][]PolesData // keyed by spin (ell)
	DhTilde   *mat.Dense
	DgFinal   *mat.Dense
	Converged bool // whether the iteration converged within max iterations
}

// Pole represents a pole in the conformal block computation
type Pole struct {
	Delta float64
	C     float64
	Idx   int
	N     float64
}

// RecursiveDerivatives implements the recursive conformal block derivatives algorithm
type RecursiveDerivatives struct {
	// Core dependencies
	recursiveG             RecursiveG
	usePrecomputedPhi1     bool
	phi1NumericCache       *Phi1Numeric
	usePhi1DerivsDiskCache bool
	useGPU                 bool

	// Crossing symmetric point parameters
	rStar    float64
	etaStar  float64
	zStar    float64
	zBarStar float64

	// Derivative order configurations
	derivativeOrdersZZBar [][2]int
	derivativeOrdersREta  [][2]int
	numDerivs             int

	// Caches
	functionCache               *FunctionCache
	phi1DerivsDiskCache         *Phi1DerivsDiskCache
	cacheRMatrix                sync.Map
	derivativeOrdersZZbarFPlus  [][2]int
	derivativeOrdersZZbarFMinus [][2]int
	cachePhi123                 map[phi123Key]float64
	cacheMu                     sync.Mutex

	// Final results and evaluation cache
	convergedData            *ConvergedBlockDerivativeData
	derivativeEvaluatorCache *RetaCache
}

// NewRecursiveDerivatives creates a new RecursiveDerivatives instance
func NewRecursiveDerivatives(recG RecursiveG, nMax int, usePrecomputedPhi1, useNumericDerivs, useGPU bool) *RecursiveDerivatives {
	// recG should be constructed by the caller (the Python code constructed RecursiveG inside).
	// In Go we accept it as a dependency to avoid circular construction here.
	zZbDerivOrders := ComputeDerivativeOrdersZZbar(nMax)

	// Configure derivative evaluator
	var derivMode EvaluatorMode
	if useNumericDerivs {
		derivMode = NumericMode
	} else {
		// Use symbolic mode with precomputation at crossing symmetric point
		// This uses the corrected analytic formulas from Appendix C
		derivMode = SymbolicMode
	}
	evaluator := NewRetaEvaluatorWithMaxOrder(derivMode, nMax)

	// Create RecursiveDerivatives instance
	rd := &RecursiveDerivatives{
		// Core configuration
		recursiveG:             recG,
		usePrecomputedPhi1:     usePrecomputedPhi1,
		phi1NumericCache:       NewPhi1Numeric(),
		usePhi1DerivsDiskCache: true,
		useGPU:                 useGPU,

		// Crossing symmetric point coordinates
		rStar:    3 - 2*math.Sqrt(2),
		etaStar:  1.0,
		zStar:    0.5,
		zBarStar: 0.5,

		// Derivative orders and caches
		derivativeOrdersZZBar:       zZbDerivOrders,
		derivativeOrdersREta:        ComputeDerivativeOrdersREta(zZbDerivOrders),
		derivativeOrdersZZbarFPlus:  make([][2]int, 0, len(zZbDerivOrders)),
		derivativeOrdersZZbarFMinus: make([][2]int, 0, len(zZbDerivOrders)),
		derivativeEvaluatorCache:    NewRetaCache(evaluator, 0.5, 0.5),
		functionCache:               &FunctionCache{},
		cachePhi123:                 make(map[phi123Key]float64),
	}

	// Separate derivative orders by parity
	for _, p := range rd.derivativeOrdersZZBar {
		m, n := p[0], p[1]
		if (m+n)%2 == 0 {
			rd.derivativeOrdersZZbarFPlus = append(rd.derivativeOrdersZZbarFPlus, [2]int{m, n})
		} else {
			rd.derivativeOrdersZZbarFMinus = append(rd.derivativeOrdersZZbarFMinus, [2]int{m, n})
		}
	}

	// Validate derivative order counts
	if len(rd.derivativeOrdersZZbarFPlus) != len(rd.derivativeOrdersZZbarFMinus) {
		panic("number of F_plus and F_minus derivatives must be equal")
	}

	rd.numDerivs = len(rd.derivativeOrdersREta)
	return rd
}

// Recurse runs the derivative recursion using the order-by-order
// r-series approach. This replaces the iterative fixed-point
// method and converges for all parameter values.
//
// The algorithm:
//  1. Compute htilde r-coefficients for each spin and each eta-derivative order
//  2. Run the r-coefficient recursion order-by-order (same as multi-point)
//  3. Convert r-coefficients to (r,eta) derivatives at (rStar, etaStar)
//  4. Store in DgFinal (same format as iterative Recurse)
func (rd *RecursiveDerivatives) Recurse(delta12, delta34 float64, rOrder int) error {
	// --- 1. Setup ---
	rd.recursiveG.unique_poles_map = make(map[PoleKey]int)
	rd.recursiveG.idxToKey = nil

	polesData := rd.recursiveG.getAllPolesData(delta12, delta34)
	numPoles := len(rd.recursiveG.idxToKey)
	if numPoles == 0 {
		return fmt.Errorf("Recurse: no pole keys generated")
	}

	// Determine the maximum eta-derivative order needed
	maxEtaDeriv := 0
	for _, pair := range rd.derivativeOrdersREta {
		if pair[1] > maxEtaDeriv {
			maxEtaDeriv = pair[1]
		}
	}
	numEtaDerivs := maxEtaDeriv + 1

	// --- 2. Compute htilde r-coefficients for each spin and eta-derivative ---
	// htildeCoeffs[ell][etaDeriv][rOrder] = d^q/deta^q (r-coeff of htilde at order p) | eta=etaStar
	htildeCoeffsByEll := make(map[int][][]float64)
	for _, key := range rd.recursiveG.idxToKey {
		if _, ok := htildeCoeffsByEll[key.Ell]; ok {
			continue
		}
		coeffs, err := computeHTildeRCoeffsWithEtaDerivs(
			rd, &rd.recursiveG, delta12, delta34, key.Ell, rd.etaStar, rOrder, maxEtaDeriv)
		if err != nil {
			return fmt.Errorf("Recurse: htilde coeffs for ell=%d: %w", key.Ell, err)
		}
		htildeCoeffsByEll[key.Ell] = coeffs
	}

	// --- 3. r-coefficient recursion (order by order) ---
	// hCoeffs[colIdx][etaDeriv][rOrder]
	hCoeffs := make([][][]float64, numPoles)
	for i := range hCoeffs {
		hCoeffs[i] = make([][]float64, numEtaDerivs)
		for q := range hCoeffs[i] {
			hCoeffs[i][q] = make([]float64, rOrder+1)
		}
	}

	for p := 0; p <= rOrder; p++ {
		for idx, key := range rd.recursiveG.idxToKey {
			for q := 0; q < numEtaDerivs; q++ {
				val := htildeCoeffsByEll[key.Ell][q][p]

				for _, pole := range polesData[key.Ell] {
					if pole.N > p {
						continue
					}
					denom := key.Delta - pole.Delta
					if denom == 0.0 {
						continue
					}
					val += pole.C * hCoeffs[pole.Idx][q][p-pole.N] / denom
				}
				hCoeffs[idx][q][p] = val
			}
		}
	}

	// --- 4. Convert r-coefficients to derivatives at rStar ---
	// For each column j, compute:
	//   h_deriv[m][q] = sum_{p>=m} p!/(p-m)! * rStar^{p-m} * hCoeffs[j][q][p]
	// Then apply R(delta_j) to get DgFinal[:, j]

	// First compute h-derivatives (without r^delta factor)
	// hDerivs[colIdx] is a dense matrix: rows = derivativeOrdersREta indices, 1 column
	// We need to map (rDerivOrder, etaDerivOrder) -> derivativeOrdersREta index
	retaIndex := make(map[[2]int]int)
	for i, pair := range rd.derivativeOrdersREta {
		retaIndex[pair] = i
	}

	// Precompute rStar powers
	rStarPows := make([]float64, rOrder+1)
	rStarPows[0] = 1.0
	for i := 1; i <= rOrder; i++ {
		rStarPows[i] = rStarPows[i-1] * rd.rStar
	}

	// Precompute factorials
	factorials := make([]float64, rOrder+2)
	factorials[0] = 1.0
	for i := 1; i <= rOrder+1; i++ {
		factorials[i] = factorials[i-1] * float64(i)
	}

	// Build DhTilde and DgFinal
	// DgFinal[:, j] = derivatives of r^{delta_j} * h(r, eta) at (rStar, etaStar)
	// DhTilde[:, j] = derivatives of htilde(r, eta) at (rStar, etaStar) (no R matrix)
	//
	// Direct computation (no R matrix needed):
	// d^m/dr^m [r^delta * h(r)] = sum_p a_p * ff(delta+p, m) * rStar^{delta+p-m}
	// where a_p are the r-coefficients of h.

	dhTilde := mat.NewDense(rd.numDerivs, numPoles, nil)
	dgFinal := mat.NewDense(rd.numDerivs, numPoles, nil)

	type derivColumnResult struct {
		j      int
		dhVals []float64
		dgVals []float64
	}

	jobs := make(chan int, numPoles)
	results := make(chan derivColumnResult, numPoles)
	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers < 1 {
		numWorkers = 1
	}
	if numWorkers > numPoles {
		numWorkers = numPoles
	}

	var convWG sync.WaitGroup
	worker := func() {
		defer convWG.Done()
		for j := range jobs {
			key := rd.recursiveG.idxToKey[j]
			delta := key.Delta
			htCoeffs := htildeCoeffsByEll[key.Ell]

			dhVals := make([]float64, rd.numDerivs)
			dgVals := make([]float64, rd.numDerivs)

			for _, pair := range rd.derivativeOrdersREta {
				rDeriv := pair[0]   // m
				etaDeriv := pair[1] // q (eta deriv is just the q-th set of coefficients)
				idx := retaIndex[pair]
				if etaDeriv >= numEtaDerivs {
					continue
				}

				// dhTilde: derivatives of htilde at rStar (no r^delta factor)
				htVal := 0.0
				for p := rDeriv; p <= rOrder; p++ {
					htVal += factorials[p] / factorials[p-rDeriv] * rStarPows[p-rDeriv] * htCoeffs[etaDeriv][p]
				}
				dhVals[idx] = htVal

				// dgFinal: derivatives of r^delta * h(r) at rStar
				// d^m/dr^m [sum_p a_p r^{delta+p}] = sum_p a_p * ff(delta+p, m) * rStar^{delta+p-m}
				dgVal := 0.0
				for p := 0; p <= rOrder; p++ {
					dgVal += hCoeffs[j][etaDeriv][p] * ff(delta+float64(p), rDeriv) *
						math.Pow(rd.rStar, delta+float64(p)-float64(rDeriv))
				}
				dgVals[idx] = dgVal
			}

			results <- derivColumnResult{j: j, dhVals: dhVals, dgVals: dgVals}
		}
	}

	convWG.Add(numWorkers)
	for w := 0; w < numWorkers; w++ {
		go worker()
	}
	for j := 0; j < numPoles; j++ {
		jobs <- j
	}
	close(jobs)

	go func() {
		convWG.Wait()
		close(results)
	}()

	for col := range results {
		dhTilde.SetCol(col.j, col.dhVals)
		dgFinal.SetCol(col.j, col.dgVals)
	}

	// --- 5. Store results ---
	rd.convergedData = &ConvergedBlockDerivativeData{
		Delta12:   delta12,
		Delta34:   delta34,
		PolesData: polesData,
		DhTilde:   dhTilde,
		DgFinal:   dgFinal,
		Converged: true,
	}

	return nil
}

// cpuIterativeUpdate performs iterative update using CPU parallelization
func (rd *RecursiveDerivatives) cpuIterativeUpdate(
	dg, dgTilde *mat.Dense,
	RMap map[[2]int]*mat.Dense,
	polesData map[int][]PolesData,
	maxIterations int,
	tol float64,
) (*mat.Dense, bool) {
	m, n := dg.Dims()
	dgNew := mat.NewDense(m, n, nil)
	dgCopy := mat.DenseCopyOf(dg)
	converged := false

	// Number of workers = #CPUs
	numWorkers := runtime.GOMAXPROCS(0)

	type job struct {
		j   int
		key PoleKey
	}

	jobs := make(chan job, n)

	// Workers computations for columns j
	worker := func() {
		// Thread-local buffers for efficient computation
		colUpdate := mat.NewVecDense(m, nil)
		tempVec := mat.NewVecDense(m, nil)
		var term mat.VecDense

		for task := range jobs {
			j, key := task.j, task.key

			// Reset thread-local accumulator
			for ii := 0; ii < m; ii++ {
				colUpdate.SetVec(ii, 0)
			}

			// Compute update for this column j
			for _, pole := range polesData[key.Ell] {
				R := RMap[[2]int{j, pole.Idx}]

				// tempVec = dgCopy[:, pole.Idx]
				for ii := 0; ii < m; ii++ {
					tempVec.SetVec(ii, dgCopy.At(ii, pole.Idx))
				}

				term.MulVec(R, tempVec)

				scale := pole.C / (key.Delta - pole.Delta)
				blas64.Scal(scale, term.RawVector())

				// colUpdate += term
				for ii := 0; ii < m; ii++ {
					colUpdate.SetVec(ii, colUpdate.AtVec(ii)+term.AtVec(ii))
				}
			}

			// Write into dgNew column
			for ii := 0; ii < m; ii++ {
				dgNew.Set(ii, j, dgNew.At(ii, j)+colUpdate.AtVec(ii))
			}
		}
	}

	for iter := 0; iter < maxIterations; iter++ {
		dgNew.Copy(dgTilde)

		// Spawn worker pool
		var wg sync.WaitGroup
		wg.Add(numWorkers)
		for w := 0; w < numWorkers; w++ {
			go func() {
				defer wg.Done()
				worker()
			}()
		}

		// Push column jobs
		for j, key := range rd.recursiveG.idxToKey {
			jobs <- job{j, key}
		}
		close(jobs)
		wg.Wait()

		// -----------------------------------------
		// Parallel convergence check
		// -----------------------------------------
		maxDiff := 0.0
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				d := math.Abs(dgNew.At(i, j)-dgCopy.At(i, j)) / dgCopy.At(i, j)
				if d > maxDiff {
					maxDiff = d
				}
			}
		}

		if maxDiff < tol {
			converged = true
			break
		}

		// Prepare for next iteration
		dgCopy, dgNew = dgNew, dgCopy

		// Re-open jobs channel for next iteration
		jobs = make(chan job, n)
	}

	return dgCopy, converged
}

// Evaluate evaluates the recursion result for a specific spin and scaling dimension.
//
// Returns: map keyed by [2]int{m,n} to the conformal block derivative value.
func (rd *RecursiveDerivatives) Evaluate(delta float64, ell int) (map[[2]int]float64, error) {
	if rd.convergedData == nil {
		return nil, fmt.Errorf("recursion has not been run yet; call `Recurse` first")
	}

	// Ensure we have poles data for this spin
	polesList, ok := rd.convergedData.PolesData[ell]
	if !ok || len(polesList) == 0 {
		return nil, fmt.Errorf("no poles data available for spin %d. Run `Recurse` first", ell)
	}

	// Find index in idxToKey corresponding to this spin (dgTilde column index)
	dhTildeIndex := -1
	for idx, key := range rd.recursiveG.idxToKey {
		if key.Ell == ell {
			dhTildeIndex = idx
			break
		}
	}
	if dhTildeIndex == -1 {
		return nil, fmt.Errorf("no poles data available for spin %d. Run `Recurse` first", ell)
	}

	// Start with dh = dh_tilde[:, dhTildeIndex] as a *mat.VecDense
	if rd.convergedData.DhTilde == nil {
		return nil, fmt.Errorf("converged data has empty DhTilde")
	}

	// Multiply my r^delta
	rMatrix := rd.buildRMatrix(delta)
	dg := MatVecColumn(rMatrix, rd.convergedData.DhTilde, dhTildeIndex)

	for _, pole := range rd.convergedData.PolesData[ell] {
		rMatrixShifted := rd.buildRMatrix(delta - pole.Delta)
		term := MatVecColumn(rMatrixShifted, rd.convergedData.DgFinal, pole.Idx)
		term.Scale(pole.C/(delta-pole.Delta), term)
		dg.Add(dg, term)
	}

	// Build a map from (m, n) -> dg[idx]
	dgMap := make(map[[2]int]float64)
	for idx, pair := range rd.derivativeOrdersREta {
		// TODO: Change pair[0] and pair[1] to deriv order struct
		// pair is m,n
		dgMap[[2]int{pair[0], pair[1]}] = dg.At(idx, 0)
	}
	return dgMap, nil
}

// buildRMatrix constructs the R matrix for the given r power (corresponds to r_delta_mat in Python)
func (rd *RecursiveDerivatives) buildRMatrix(rPower float64) *mat.Dense {
	// Check cache first
	if cached, ok := rd.cacheRMatrix.Load(rPower); ok {
		return cached.(*mat.Dense)
	}

	// Construct the R matrix
	m := mat.NewDense(rd.numDerivs, rd.numDerivs, nil)
	for i, mn := range rd.derivativeOrdersREta {
		mOut := mn[0]
		nOut := mn[1]
		for j, ns := range rd.derivativeOrdersREta {
			s := ns[0]
			nIn := ns[1]

			if nOut == nIn && mOut >= s {
				entry := rd.comb(mOut, s) * rd.ff(rPower, mOut-s) * rd.Pow(rd.rStar, rPower-float64(mOut)+float64(s))
				m.Set(i, j, entry)
			}
		}
	}

	// Store in cache (thread-safe)
	rd.cacheRMatrix.Store(rPower, m)
	return m
}

// derivativeHTilde corresponds to derivative_h_tilde
func (rd *RecursiveDerivatives) derivativeHTilde(m, n, ell int, r, eta, nu, alpha, beta float64) float64 {
	total := 0.0
	for i := range n + 1 {

		term := rd.comb(n, i) * rd.Pow(2.0, float64(n-i)) * rd.rf(nu, n-i)
		term *= rd.derivativePhi123(m, i, r, eta, nu, alpha, beta)
		term *= rd.GegenbauerC(ell-n+i, nu+float64(n)-float64(i), eta)

		total += term
	}

	total *= rd.Pow(-1.0, float64(ell)) * rd.factorial(ell) / rd.rf(2*nu, ell)
	return total
}

// derivativePhi123 computes phi123 derivatives with caching
func (rd *RecursiveDerivatives) derivativePhi123(
	i, j int,
	r, eta, nu, alpha, beta float64,
) float64 {
	key := phi123Key{i, j, r, eta, nu, alpha, beta}

	// Check cache
	rd.cacheMu.Lock()
	if rd.cachePhi123 == nil {
		rd.cachePhi123 = make(map[phi123Key]float64)
	}
	if val, ok := rd.cachePhi123[key]; ok {
		rd.cacheMu.Unlock()
		return val
	}
	rd.cacheMu.Unlock()

	// COMPUTE
	result := 0.0
	for i1 := 0; i1 <= i; i1++ {
		for i2 := 0; i2 <= i-i1; i2++ {
			i3 := i - i1 - i2

			for j2 := 0; j2 <= j; j2++ {
				j3 := j - j2

				coefI := rd.factorial(i) / (rd.factorial(i1) * rd.factorial(i2) * rd.factorial(i3))
				coefJ := rd.factorial(j) / (rd.factorial(j2) * rd.factorial(j3))

				phi1 := rd.phi1(i1, 0, r, nu)

				term := coefI * coefJ
				term *= phi1
				term *= rd.phi2(i2, j2, r, eta, alpha)
				term *= rd.phi3(i3, j3, r, eta, beta)

				result += term
			}
		}
	}

	// Update cache
	rd.cacheMu.Lock()
	if rd.cachePhi123 == nil {
		rd.cachePhi123 = make(map[phi123Key]float64)
	}
	rd.cachePhi123[key] = result
	rd.cacheMu.Unlock()

	return result
}

// ClearPhi123Cache clears the phi123 derivative cache
func (rd *RecursiveDerivatives) ClearPhi123Cache() {
	rd.cacheMu.Lock()
	rd.cachePhi123 = make(map[phi123Key]float64)
	rd.cacheMu.Unlock()
}

func (rd *RecursiveDerivatives) phi2(i, j int, r, eta, alpha float64) float64 {
	return rd.derivativeFwrtrEta(i, j, r, eta, alpha, F_PLUS)
}

func (rd *RecursiveDerivatives) phi3(i, j int, r, eta, beta float64) float64 {
	return rd.derivativeFwrtrEta(i, j, r, eta, beta, F_MINUS)
}

func (rd *RecursiveDerivatives) derivativeFwrtrEta(i, j int, r, eta, exponent float64, fType FType) float64 {
	result := 0.0
	for k := range i + 1 {
		// For k > j, ff(j,k)=0 so the contribution is identically zero.
		// Skipping avoids 0*Inf -> NaN when r=0 and j-k is negative.
		if k > j {
			continue
		}
		term := rd.comb(i, k) * rd.ff(float64(j), k) * rd.Pow(r, float64(j-k))
		term *= rd.derivativeFwrtr(r, eta, float64(i-k), -exponent-float64(j), fType)
		result += term
	}

	result *= rd.Pow(2., float64(j)) * rd.rf(exponent, j)
	if fType == F_PLUS {
		// multiply by (-1)^j
		if j%2 == 1 {
			result = -result
		}
	}
	return result
}

func (rd *RecursiveDerivatives) derivativeFwrtr(r, eta, nVal, mVal float64, fType FType) float64 {
	fac := 1.0
	if fType == F_MINUS {
		fac = -1.0
	}
	f0 := 1 + r*r + fac*2*r*eta
	f1 := 2*r + fac*2*eta

	nInt := int(nVal)

	result := 0.0
	start := (nInt + 1) / 2
	for k := start; k <= nInt; k++ {
		j1 := 2*k - nInt
		j2 := nInt - k
		bell := rd.factorial(nInt) / (rd.factorial(j1) * rd.factorial(j2))
		term := rd.ff(mVal, k) * rd.Pow(f0, mVal-float64(k)) * bell * rd.Pow(f1, float64(j1))
		result += term
	}
	return result
}

// ComputeGDerivativeswrtZZbar converts dg derivatives from (r,eta) to (z,zbar) coordinates
func (rd *RecursiveDerivatives) ComputeGDerivativeswrtZZbar(
	dg map[[2]int]float64,
) map[[2]int]float64 {
	dgZZbar := make(map[[2]int]float64)

	for _, ord := range rd.derivativeOrdersZZBar {
		m, n := ord[0], ord[1]
		dgZZbar[[2]int{m, n}] = rd.partialZDerivative(dg, m, n)
	}

	return dgZZbar
}

// RecurseAndEvaluateDG is a convenience wrapper that runs Recurse
// and evaluates dg for the given operators.
func (rd *RecursiveDerivatives) RecurseAndEvaluateDG(
	delta12, delta34 float64,
	deltas []float64, spins []int,
	rOrder int,
) ([]map[[2]int]float64, error) {
	if err := rd.Recurse(delta12, delta34, rOrder); err != nil {
		return nil, err
	}
	if len(spins) != len(deltas) {
		return nil, fmt.Errorf("mismatch between number of spins and deltas")
	}
	dg := make([]map[[2]int]float64, len(deltas))
	for i := range deltas {
		evaluatedBlock, err := rd.Evaluate(deltas[i], spins[i])
		if err != nil {
			return nil, err
		}
		dg[i] = evaluatedBlock
	}
	return dg, nil
}

// RecurseAndEvaluateDF runs Strategy A recursion and converts dg -> dF.
func (rd *RecursiveDerivatives) RecurseAndEvaluateDF(
	blockTypes []BlockType,
	delta12, delta34, deltaAve23 float64,
	deltas []float64, spins []int,
	rOrder int,
	normalise bool,
) ([][][]float64, error) {
	dg, err := rd.RecurseAndEvaluateDG(delta12, delta34, deltas, spins, rOrder)
	if err != nil {
		return nil, err
	}
	result, err := rd.convertDGtoDFDerivatives(blockTypes, delta12, delta34, deltaAve23, deltas, spins, dg, normalise)
	return result, err
}

// convertDGtoDFDerivatives converts dg maps to dF arrays (shared by both strategies).
func (rd *RecursiveDerivatives) convertDGtoDFDerivatives(
	blockTypes []BlockType,
	delta12, delta34, deltaAve23 float64,
	deltas []float64, spins []int,
	dg []map[[2]int]float64,
	normalise bool,
) ([][][]float64, error) {
	FDerivs := make([][][]float64, len(blockTypes))
	for i := range FDerivs {
		FDerivs[i] = make([][]float64, len(spins))
		for j := range FDerivs[i] {
			FDerivs[i][j] = rd.ComputeFDerivativeswrtZZbar(blockTypes[i], dg[j], deltaAve23, normalise)
		}
	}
	return FDerivs, nil
}

func (rd *RecursiveDerivatives) ComputeFDerivativeswrtZZbar(
	blockType BlockType,
	dg map[[2]int]float64,
	deltaAve23 float64,
	normalise bool,
) []float64 {
	// Choose derivative orders
	var derivativeOrders [][2]int
	if blockType == BlockPlus {
		derivativeOrders = rd.derivativeOrdersZZbarFPlus
	} else {
		derivativeOrders = rd.derivativeOrdersZZbarFMinus
	}

	FDerivs := make([]float64, 0, len(derivativeOrders))

	// Cache for computed partial derivatives
	dgZZbar := make(map[DerivKey]float64)

	for _, ord := range derivativeOrders {
		m, n := ord[0], ord[1]
		sum := 0.0

		for i := 0; i <= m; i++ {
			for j := 0; j <= n; j++ {
				key := DerivKey{M: m - i, N: n - j}

				// Compute partial derivative if not cached
				if _, ok := dgZZbar[key]; !ok {
					dgZZbar[key] = rd.partialZDerivative(dg, key.M, key.N)
				}

				term := 2.0 * rd.Pow(-1, float64(i+j)) * rd.Pow(2, float64(i+j)-2*deltaAve23)
				term *= rd.rf(1+deltaAve23-float64(i), i) * rd.rf(1+deltaAve23-float64(j), j)
				term *= float64(rd.comb(m, i) * rd.comb(n, j))
				term *= dgZZbar[key]

				sum += term
			}
		}

		if normalise {
			sum /= rd.Pow(2, float64(m+n)) * float64(rd.factorial(m)*rd.factorial(n))
		}

		FDerivs = append(FDerivs, sum)
	}

	return FDerivs
}

// partialZDerivative computes ∂_z^m ∂_zbar^n g(r(z,z̄), η(z,z̄))
func (rd *RecursiveDerivatives) partialZDerivative(dgDrEta map[[2]int]float64, m, n int) float64 {
	var pqPairs [][2]int
	for p := 0; p <= m+n; p++ {
		for q := 0; q <= m+n; q++ {
			if p+q <= m+n {
				pqPairs = append(pqPairs, [2]int{p, q})
			}
		}
	}

	result := 0.0
	for _, pq := range pqPairs {
		p, q := pq[0], pq[1]
		result += rd.PartitionFactor(p, q, m, n) * dgDrEta[[2]int{p, q}]
	}

	result *= rd.factorial(m) * rd.factorial(n)
	return result
}

// computeHTildeRCoeffsWithEtaDerivs computes the r-series coefficients of
// d^q/d(eta)^q htilde(r, eta)|_{eta=etaStar} analytically.
//
// Uses the Leibniz rule on htilde = prefactor(eta) * phi1(r) * phi2(r,eta) * phi3(r,eta):
//
//	d^q_eta htilde = phi1(r) * sum_{j1+j2+j3=q} (q!/(j1!j2!j3!))
//	  * d^{j1}_eta [prefactor]
//	  * d^{j2}_eta [phi2]
//	  * d^{j3}_eta [phi3]
//
// The eta-derivatives of phi2 and phi3 are analytic:
//
//	d^j_eta (1+r^2+2r*eta)^{-alpha} = (-1)^j rf(alpha,j) (2r)^j (1+r^2+2r*eta)^{-alpha-j}
//	d^j_eta (1+r^2-2r*eta)^{-beta}  = rf(beta,j) (2r)^j (1+r^2-2r*eta)^{-beta-j}
//
// Returns coeffs[q][p] for q = 0..maxEtaDeriv, p = 0..rOrder.
func computeHTildeRCoeffsWithEtaDerivs(
	rd *RecursiveDerivatives,
	rg *RecursiveG,
	delta12, delta34 float64,
	ell int,
	etaStar float64,
	rOrder, maxEtaDeriv int,
) ([][]float64, error) {
	nu := rg.Nu
	alpha := (1 + delta12 - delta34) / 2
	beta := (1 - delta12 + delta34) / 2

	// r-series of phi1 = (1-r^2)^{-nu}: only even powers
	phi1 := make([]float64, rOrder+1)
	for m := 0; 2*m <= rOrder; m++ {
		phi1[2*m] = rf(nu, m) / factorial(m)
	}

	// Precompute Gegenbauer eta-derivatives at etaStar:
	// d^j_eta C_ell^nu(eta) = 2^j * rf(nu, j) * C_{ell-j}^{nu+j}(eta)  [j <= ell]
	gegenDerivs := make([]float64, maxEtaDeriv+1)
	for j := 0; j <= maxEtaDeriv && j <= ell; j++ {
		gegenDerivs[j] = math.Pow(2, float64(j)) * rf(nu, j) *
			rd.GegenbauerC(ell-j, nu+float64(j), etaStar)
	}

	// Overall prefactor (without Gegenbauer): (-1)^ell * ell! / (2*nu)_ell
	basePrefactor := math.Pow(-1, float64(ell)) * factorial(ell) / rf(2*nu, ell)

	// Helper: build r-series for (1+r^2+sign*2r*eta)^{-exp} truncated to rOrder
	buildBseries := func(exp, sign float64) []float64 {
		B := make([]float64, rOrder+1)
		u := []float64{0, sign * 2 * etaStar, 1.0}
		uPow := []float64{1.0}
		for m := 0; m <= rOrder; m++ {
			coef := rf(exp, m) / factorial(m)
			if m%2 == 1 {
				coef = -coef
			}
			polyAddScaledTrunc(B, coef, uPow, rOrder)
			uPow = polyMulTrunc(uPow, u, rOrder)
		}
		return B
	}

	// Precompute all needed B-series and shifted versions to avoid redundant work.
	// phi2 part for eta-deriv j2: (-1)^j2 rf(alpha,j2) (2r)^j2 B(alpha+j2, +1)
	// phi3 part for eta-deriv j3: rf(beta,j3) (2r)^j3 B(beta+j3, -1)

	// Cache: phi1 * shifted_phi2[j2] for each j2
	phi1xPhi2 := make([][]float64, maxEtaDeriv+1)
	phi2Scales := make([]float64, maxEtaDeriv+1)
	for j2 := 0; j2 <= maxEtaDeriv; j2++ {
		phi2Scales[j2] = math.Pow(-1, float64(j2)) * rf(alpha, j2) * math.Pow(2, float64(j2))
		bSeries := buildBseries(alpha+float64(j2), 1.0)
		// Shift by j2
		shifted := make([]float64, rOrder+1)
		for p := j2; p <= rOrder; p++ {
			shifted[p] = bSeries[p-j2]
		}
		phi1xPhi2[j2] = polyMulTrunc(phi1, shifted, rOrder)
	}

	// Cache: shifted phi3[j3] series
	phi3Cache := make([][]float64, maxEtaDeriv+1)
	phi3Scales := make([]float64, maxEtaDeriv+1)
	for j3 := 0; j3 <= maxEtaDeriv; j3++ {
		phi3Scales[j3] = rf(beta, j3) * math.Pow(2, float64(j3))
		bSeries := buildBseries(beta+float64(j3), -1.0)
		shifted := make([]float64, rOrder+1)
		for p := j3; p <= rOrder; p++ {
			shifted[p] = bSeries[p-j3]
		}
		phi3Cache[j3] = shifted
	}

	result := make([][]float64, maxEtaDeriv+1)
	for q := 0; q <= maxEtaDeriv; q++ {
		result[q] = make([]float64, rOrder+1)
	}

	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup

	for q := 0; q <= maxEtaDeriv; q++ {
		q := q
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()

			row := make([]float64, rOrder+1)
			for j1 := 0; j1 <= q && j1 <= ell; j1++ {
				gegen := gegenDerivs[j1]
				if gegen == 0 {
					continue
				}
				for j2 := 0; j2 <= q-j1; j2++ {
					j3 := q - j1 - j2

					multinomial := factorial(q) / (factorial(j1) * factorial(j2) * factorial(j3))

					// Convolve precomputed phi1*phi2[j2] with phi3[j3]
					conv := polyMulTrunc(phi1xPhi2[j2], phi3Cache[j3], rOrder)

					scale := basePrefactor * multinomial * gegen * phi2Scales[j2] * phi3Scales[j3]
					for p := 0; p <= rOrder; p++ {
						row[p] += scale * conv[p]
					}
				}
			}

			result[q] = row
		}()
	}

	wg.Wait()

	return result, nil
}
