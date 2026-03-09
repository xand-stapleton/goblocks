package main

import (
	"fmt"
	iterativegpu "goblocks/gpu_iterative"
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
	recursiveG         RecursiveG
	usePrecomputedPhi1 bool
	phi1NumericCache   *Phi1Numeric
	useGPU             bool

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
		recursiveG:         recG,
		usePrecomputedPhi1: usePrecomputedPhi1,
		phi1NumericCache:   NewPhi1Numeric(),
		useGPU:             useGPU,

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

// Recurse runs the recursive conformal block derivatives algorithm
func (rd *RecursiveDerivatives) Recurse(delta12, delta34 float64, maxIterations int, tol float64) error {
	// Clear caches
	rd.ClearPhi123Cache()

	// --- 1. Initialization / setup ---
	alpha := (1 + delta12 - delta34) / 2
	beta := (1 - delta12 + delta34) / 2

	// Reset pole data structures
	rd.recursiveG.unique_poles_map = make(map[PoleKey]int)
	rd.recursiveG.idxToKey = nil

	// --- 2. Compute polesData ---
	polesData := rd.recursiveG.getAllPolesData(delta12, delta34)
	numPoles := len(rd.recursiveG.idxToKey)

	// --- 3. Precompute dhTilde (parallelized) ---
	dhTilde := mat.NewDense(rd.numDerivs, numPoles, nil)
	mNEllCache := sync.Map{} // thread-safe cache for computed values
	var wg sync.WaitGroup

	for i, orders := range rd.derivativeOrdersREta {
		m, n := orders[0], orders[1]
		wg.Add(1)
		go func(i, m, n int) {
			defer wg.Done()
			for j, key := range rd.recursiveG.idxToKey {
				cacheKey := [3]int{m, n, key.Ell}
				if valIface, ok := mNEllCache.Load(cacheKey); ok {
					dhTilde.Set(i, j, valIface.(float64))
				} else {
					val := rd.derivativeHTilde(m, n, key.Ell, rd.rStar, rd.etaStar, rd.recursiveG.Nu, alpha, beta)
					mNEllCache.Store(cacheKey, val)
					dhTilde.Set(i, j, val)
				}
			}
		}(i, m, n)
	}

	wg.Wait()

	// --- 4. Copy dgTilde ---
	dgTilde := mat.DenseCopyOf(dhTilde)

	// --- 5. Apply R matrices sequentially ---
	for j, key := range rd.recursiveG.idxToKey {
		col := mat.NewVecDense(rd.numDerivs, mat.Col(nil, j, dgTilde))
		var result mat.VecDense
		result.MulVec(rd.buildRMatrix(key.Delta), col)
		dgTilde.SetCol(j, result.RawVector().Data)
	}

	// --- 6. Precompute RMap ---
	RMap := make(map[[2]int]*mat.Dense)
	var rMapMutex sync.Mutex
	var rMapWG sync.WaitGroup

	for j, key := range rd.recursiveG.idxToKey {
		for _, pole := range polesData[key.Ell] {
			rMapWG.Add(1)
			go func(j int, pole PolesData, key PoleKey) {
				defer rMapWG.Done()
				matrix := rd.buildRMatrix(key.Delta - pole.Delta)
				rMapMutex.Lock()
				RMap[[2]int{j, pole.Idx}] = matrix
				rMapMutex.Unlock()
			}(j, pole, key)
		}
	}

	rMapWG.Wait()

	// --- 7. CPU / GPU iterative update ---
	dg := mat.DenseCopyOf(dgTilde)
	var converged bool
	if rd.useGPU {
		dg, converged = rd.gpuIterativeUpdate(dg, dgTilde, RMap, polesData, maxIterations, tol)
	} else {
		dg, converged = rd.cpuIterativeUpdate(dg, dgTilde, RMap, polesData, maxIterations, tol)
	}

	// --- 8. Store results ---
	rd.convergedData = &ConvergedBlockDerivativeData{
		Delta12:   delta12,
		Delta34:   delta34,
		PolesData: polesData,
		DhTilde:   dhTilde,
		DgFinal:   dg,
		Converged: converged,
	}

	return nil
}

// gpuIterativeUpdate prepares GPU data and runs the iterative update on GPU.
// Falls back to CPU implementation if GPU computation fails.
func (rd *RecursiveDerivatives) gpuIterativeUpdate(
	dg, dgTilde *mat.Dense,
	RMap map[[2]int]*mat.Dense,
	polesData map[int][]PolesData,
	maxIterations int,
	tol float64,
) (*mat.Dense, bool) {

	idxToKey := rd.recursiveG.idxToKey
	nPolesCols := len(idxToKey)
	m := rd.numDerivs

	// ---------- 1. Keys ----------
	keys := make([]iterativegpu.KeyDataGo, nPolesCols)
	for j := 0; j < nPolesCols; j++ {
		key := idxToKey[j]
		keys[j] = iterativegpu.KeyDataGo{Ell: key.Ell, Delta: key.Delta}
	}

	// ---------- 2. Count total poles ----------
	totalPoles := 0
	for j := 0; j < nPolesCols; j++ {
		totalPoles += len(polesData[idxToKey[j].Ell])
	}

	// ---------- 3. Build polesFlat + polesOffset ----------
	polesFlat := make([]iterativegpu.PolesDataGo, totalPoles)
	polesOffset := make([]int, nPolesCols+1)
	{
		count := 0
		for j := 0; j < nPolesCols; j++ {
			key := idxToKey[j]
			list := polesData[key.Ell]
			polesOffset[j] = count
			for k := 0; k < len(list); k++ {
				p := list[k]
				polesFlat[count] = iterativegpu.PolesDataGo{
					N:     p.N,
					Delta: p.Delta,
					Ell:   p.Ell,
					C:     p.C,
					Idx:   p.Idx,
				}
				count++
			}
		}
		polesOffset[nPolesCols] = count
	}

	// ---------- 4. Build Rlist in parallel ----------
	Rlist := make([][]float64, totalPoles)
	var wg sync.WaitGroup
	wg.Add(nPolesCols)

	for j := 0; j < nPolesCols; j++ {
		j := j // capture variable
		go func() {
			defer wg.Done()
			key := idxToKey[j]
			list := polesData[key.Ell]
			start := polesOffset[j]

			for k := 0; k < len(list); k++ {
				p := list[k]
				Rmat, ok := RMap[[2]int{j, p.Idx}]
				if !ok || Rmat == nil {
					panic(fmt.Errorf("missing R matrix for column %d pole idx %d", j, p.Idx))
				}

				data := make([]float64, m*m)
				for col := 0; col < m; col++ {
					off := col * m
					for row := 0; row < m; row++ {
						data[off+row] = Rmat.At(row, col)
					}
				}
				Rlist[start+k] = data
			}
		}()
	}

	// Wait for R matrix computation with error handling
	func() {
		defer func() {
			if r := recover(); r != nil {
				fmt.Printf("Error in parallel R matrix computation, falling back to CPU: %v\n", r)
				var converged bool
				dg, converged = rd.cpuIterativeUpdate(dg, dgTilde, RMap, polesData, maxIterations, tol)
				_ = converged // will be returned by outer function
			}
		}()
		wg.Wait()
	}()

	// ---------- 5. GPU call ----------
	out, converged, err := iterativegpu.GPUIterativeUpdate(
		dg, dgTilde, keys, polesFlat, polesOffset, Rlist, maxIterations, tol,
	)

	if err != nil {
		fmt.Printf("GPU computation failed, falling back to CPU: %v\n", err)
		return rd.cpuIterativeUpdate(dg, dgTilde, RMap, polesData, maxIterations, tol)
	}

	return out, converged
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
	if rd.usePrecomputedPhi1 && !floatEquals(r, 3-2*math.Sqrt(2)) && !floatEquals(eta, 1.0) {
		panic("When using precomputed phi1 derivatives, r and eta must be the crossing symmetric point.")
	}
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

				phi1, _ := rd.phi1(i1, 0.0, r, nu)

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

func (rd *RecursiveDerivatives) phi1(i, j int, r, nu float64) (float64, error) {
	if rd.usePrecomputedPhi1 {
		// Error handling is performed downstream
		return rd.phi1NumericCache.Eval(i, j, r, nu)
	}
	// Calculate analytically...
	return 0, fmt.Errorf("analytic phi1 is not yet implemented")
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

// RecurseAndEvaluateDG runs the recursion and evaluates dg in one step.
func (rd *RecursiveDerivatives) RecurseAndEvaluateDG(
	delta12 float64,
	delta34 float64,
	deltas []float64, spins []int,
	maxIterations int,
	tol float64,
) ([]map[[2]int]float64, error) {
	if err := rd.Recurse(delta12, delta34, maxIterations, tol); err != nil {
		return nil, err
	}
	if len(spins) != len(deltas) {
		return nil, fmt.Errorf("mismatch between number of spins and deltas")
	}

	dg := make([]map[[2]int]float64, len(deltas))

	for i := range deltas {
		delta := deltas[i]
		spin := spins[i]
		evaluatedBlock, err := rd.Evaluate(delta, spin)
		dg[i] = evaluatedBlock

		if err != nil {
			return nil, err
		}
	}
	return dg, nil
}

// RecurseAndEvaluateDF runs recursion and converts dg -> dF (the F block derivatives).
// nMax is the "n_max" parameter from the Python.
func (rd *RecursiveDerivatives) RecurseAndEvaluateDF(
	blockTypes []BlockType,
	delta12, delta34, deltaAve23 float64,
	deltas []float64,
	spins []int,
	maxIterations int,
	tol float64,
	normalise bool,
) ([][][]float64, error) {
	// First compute dg (derivatives of g wrt r,eta)
	dg, err := rd.RecurseAndEvaluateDG(delta12, delta34, deltas, spins, maxIterations, tol)
	if err != nil {
		return nil, err
	}
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
