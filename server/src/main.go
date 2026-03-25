package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
	"sync"
)

/*
This file implements a Go port of the Python class "RecursiveG" for computing
conformal blocks recursively following 1406.4858 (Eq. 4.5, 4.6, etc.).
It also provides a simple stdin/stdout "server" that accepts newline-delimited JSON
requests and returns binary little-endian float64 outputs (length-prefixed with uint32).
*/

type BlockType int

const (
	BlockPlus BlockType = iota
	BlockMinus
)

func parseBlockType(s string) (BlockType, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "+", "plus", "block_plus", "minussignfalse", "pos":
		return BlockPlus, nil
	case "-", "minus", "block_minus", "minussigntrue", "neg":
		return BlockMinus, nil
	default:
		return BlockPlus, fmt.Errorf("unknown block type %q (use '+' or '-')", s)
	}
}

type Request struct {
	// Shared params
	Command    string    `json:"command"`
	K1Max      int       `json:"k1_max"`
	K2Max      int       `json:"k2_max"`
	EllMin     int       `json:"ell_min"`
	EllMax     int       `json:"ell_max"`
	D          int       `json:"d"`
	Delta12    float64   `json:"delta_12"`
	Delta34    float64   `json:"delta_34"`
	DeltaAve23 float64   `json:"delta_ave_23"`
	Deltas     []float64 `json:"deltas"`
	Ells       []int     `json:"ells"`
	BlockTypes []string  `json:"block_types"`
	MaxIter    int       `json:"max_iterations"`
	Tol        float64   `json:"tol"`

	// Deriv block params
	R                  float64 `json:"r"`
	Eta                float64 `json:"eta"`
	Nmax               int     `json:"nmax"`
	Normalise          bool    `json:"normalise"`
	CacheDir           string  `json:"cache_dir"`
	UsePhi1Cache       *bool   `json:"use_phi1_cache"`
	UsePrecomputedPhi1 bool    `json:"use_precomputed_phi1"`
	UseNumericDerivs   bool    `json:"use_numeric_derivs"`
	UseGPU             bool    `json:"use_gpu"`

	// Points params
	ZsStr []string `json:"zsstr"`
	Zs    []complex128

	// Handle
	Handle int64 `json:"handle"`

	// Output options
	ReturnConvergenceMask bool `json:"return_convergence_mask"`
}

func internalRunRequest(req Request) ([]float64, error) {
	usePhi1Cache := true
	if req.UsePhi1Cache != nil {
		usePhi1Cache = *req.UsePhi1Cache
	}

	// Attempt to load a persistent RD instance (optional)
	var persistentRD *RecursiveDerivatives
	if req.Handle != 0 {
		obj := getHandle(req.Handle)
		if obj == nil {
			return nil, fmt.Errorf("invalid handle %d", req.Handle)
		}
		persistentRD = obj.(*RecursiveDerivatives)
		if persistentRD.usePrecomputedPhi1 != req.UsePrecomputedPhi1 ||
			persistentRD.usePhi1DerivsDiskCache != usePhi1Cache ||
			persistentRD.recursiveG.K1Max != req.K1Max ||
			persistentRD.recursiveG.K2Max != req.K2Max ||
			persistentRD.recursiveG.EllMax != req.EllMax ||
			persistentRD.recursiveG.Nu != (float64(req.D)-2)/2 {
			return nil, fmt.Errorf("parameters do not match the persistent handle; recreate the object")
		}
	}

	maxIter := defaultIfZeroInt(req.MaxIter, 100)
	tol := defaultIfZeroFloat(req.Tol, 1e-6)

	// Validate that maxIter >= 2 * k2max
	minIter := 2 * req.K2Max
	if maxIter < minIter {
		return nil, fmt.Errorf("max_iterations (%d) must be at least 2 * k2max (%d)", maxIter, minIter)
	}

	switch strings.ToLower(strings.TrimSpace(req.Command)) {

	case "recurse_and_evaluate_g":
		if len(req.Deltas) != len(req.Ells) {
			return nil, fmt.Errorf("deltas and ells must have the same length")
		}

		numOperators := len(req.Deltas)
		numZ := len(req.Zs)
		results := make([][]float64, numOperators)
		for i := range results {
			results[i] = make([]float64, numZ)
		}

		// Track convergence per z-point if requested
		var convergenceMask []float64
		if req.ReturnConvergenceMask {
			convergenceMask = make([]float64, numZ)
		}

		var wg sync.WaitGroup
		wg.Add(numZ)

		for zi, z := range req.Zs {
			go func(zi int, z complex128) {
				defer wg.Done()
				rg := NewRecursiveG(req.K1Max, req.K2Max, req.EllMin, req.EllMax, req.D)

				values, err := rg.RecurseAndEvaluateGUsingZ(
					req.Delta12, req.Delta34, z,
					req.Deltas, req.Ells,
					maxIter, tol,
				)
				if err != nil {
					for ci := 0; ci < numOperators; ci++ {
						results[ci][zi] = math.NaN()
					}
					if req.ReturnConvergenceMask {
						convergenceMask[zi] = 0.0
					}
					return
				}

				for ci, val := range values {
					results[ci][zi] = val
				}

				// Check convergence status from the cache
				if req.ReturnConvergenceMask {
					r, eta := rg.zToREta(z)
					if cd, ok := rg.ConvergedCache[REtaKey{r, eta}]; ok && cd.Converged {
						convergenceMask[zi] = 1.0
					} else {
						convergenceMask[zi] = 0.0
					}
				}
			}(zi, z)
		}

		wg.Wait()

		// flatten
		flat := make([]float64, 0, numOperators*numZ)
		for ci := 0; ci < numOperators; ci++ {
			flat = append(flat, results[ci]...)
		}

		// Append convergence mask if requested
		if req.ReturnConvergenceMask {
			flat = append(flat, convergenceMask...)
		}

		return flat, nil

	case "recurse_and_evaluate_f":
		if len(req.Deltas) != len(req.Ells) {
			return nil, fmt.Errorf("deltas and ells must have the same length")
		}

		numOperators := len(req.Deltas)
		numZ := len(req.Zs)
		numBlocks := len(req.BlockTypes)

		results := make([][][]float64, numBlocks)
		for bi := range results {
			results[bi] = make([][]float64, numOperators)
			for ci := range results[bi] {
				results[bi][ci] = make([]float64, numZ)
			}
		}

		// Track convergence per z-point if requested
		// We only need one mask per z since the same z is used for all block types
		var convergenceMask []float64
		var convergenceMutex sync.Mutex
		if req.ReturnConvergenceMask {
			convergenceMask = make([]float64, numZ)
		}

		for bi, btStr := range req.BlockTypes {
			bt, err := parseBlockType(btStr)
			if err != nil {
				return nil, err
			}

			var wg sync.WaitGroup
			wg.Add(numZ)

			for zi, z := range req.Zs {
				go func(bi, zi int, z complex128, bt BlockType) {
					defer wg.Done()
					rg := NewRecursiveG(req.K1Max, req.K2Max, req.EllMin, req.EllMax, req.D)

					values, err := rg.RecurseAndEvaluateFUsingZ(
						bt,
						req.Delta12,
						req.Delta34,
						req.DeltaAve23,
						z,
						req.Deltas,
						req.Ells,
						maxIter,
						tol,
					)
					if err != nil {
						for ci := 0; ci < numOperators; ci++ {
							results[bi][ci][zi] = math.NaN()
						}
						if req.ReturnConvergenceMask {
							convergenceMutex.Lock()
							convergenceMask[zi] = 0.0
							convergenceMutex.Unlock()
						}
						return
					}

					for ci, val := range values {
						results[bi][ci][zi] = val
					}

					// Check convergence status from the cache (only for first block type)
					if req.ReturnConvergenceMask && bi == 0 {
						r, eta := rg.zToREta(z)
						converged := false
						if cd, ok := rg.ConvergedCache[REtaKey{r, eta}]; ok && cd.Converged {
							converged = true
						}
						convergenceMutex.Lock()
						if converged {
							convergenceMask[zi] = 1.0
						} else {
							convergenceMask[zi] = 0.0
						}
						convergenceMutex.Unlock()
					}
				}(bi, zi, z, bt)
			}

			wg.Wait()
		}

		flat := flatten3D(results)

		// Append convergence mask if requested
		if req.ReturnConvergenceMask {
			flat = append(flat, convergenceMask...)
		}

		return flat, nil

	case "recurse_and_evaluate_dg":
		if persistentRD == nil {
			// Backward-compatible: create a temporary local RD
			cacheDir := strings.TrimSpace(req.CacheDir)
			if cacheDir == "" {
				cacheDir = "cache"
			}

			rg := NewRecursiveG(req.K1Max, req.K2Max, req.EllMin, req.EllMax, req.D)
			rd := NewRecursiveDerivatives(
				*rg,
				req.Nmax, req.UsePrecomputedPhi1,
				req.UseNumericDerivs, req.UseGPU,
			)
			rd.usePhi1DerivsDiskCache = usePhi1Cache
			if err := rd.BuildLoadCache(cacheDir); err != nil {
				return nil, err
			}
			persistentRD = rd
		}

		if len(req.Deltas) != len(req.Ells) {
			return nil, fmt.Errorf("deltas and ells must have the same length")
		}

		dg, err := persistentRD.RecurseAndEvaluateDG(
			req.Delta12,
			req.Delta34,
			req.Deltas,
			req.Ells,
			req.MaxIter,
		)
		if err != nil {
			return nil, err
		}

		// Convert from (r,eta) derivatives to (z,zbar) derivatives
		for i := range dg {
			dg[i] = persistentRD.ComputeGDerivativeswrtZZbar(dg[i])
		}

		// Flatten []map[[2]int]float64 into operator-major, derivative-order (m,n) order
		// Now using z,zbar derivative orders instead of r,eta
		numOperators := len(dg)
		numDerivs := len(persistentRD.derivativeOrdersZZBar)
		results := make([][]float64, numOperators)
		for i := range results {
			results[i] = make([]float64, numDerivs)
		}

		for i := 0; i < numOperators; i++ {
			m := dg[i]
			for j, ord := range persistentRD.derivativeOrdersZZBar {
				results[i][j] = m[[2]int{ord[0], ord[1]}]
			}
		}

		flat := make([]float64, 0, numOperators*numDerivs)
		for ci := 0; ci < numOperators; ci++ {
			flat = append(flat, results[ci]...)
		}

		// Optionally append convergence mask
		if req.ReturnConvergenceMask {
			if persistentRD.convergedData != nil && persistentRD.convergedData.Converged {
				flat = append(flat, 1.0)
			} else {
				flat = append(flat, 0.0)
			}
		}

		return flat, nil

	case "recurse_and_evaluate_df":
		// Strategy A derivative blocks (order-by-order r-series)
		if persistentRD == nil {
			rg := NewRecursiveG(req.K1Max, req.K2Max, req.EllMin, req.EllMax, req.D)
			rd := NewRecursiveDerivatives(
				*rg,
				req.Nmax, req.UsePrecomputedPhi1,
				req.UseNumericDerivs, req.UseGPU,
			)
			rd.BuildLoadCache(req.CacheDir)
			persistentRD = rd
		}

		if len(req.Deltas) != len(req.Ells) {
			return nil, fmt.Errorf("deltas and ells must have the same length")
		}

		blockTypes := make([]BlockType, 0, len(req.BlockTypes))
		for _, btStr := range req.BlockTypes {
			bt, err := parseBlockType(btStr)
			if err != nil {
				return nil, err
			}
			blockTypes = append(blockTypes, bt)
		}

		rOrder := defaultIfZeroInt(maxIter, 30)

		result, err := persistentRD.RecurseAndEvaluateDF(
			blockTypes,
			req.Delta12,
			req.Delta34,
			req.DeltaAve23,
			req.Deltas,
			req.Ells,
			rOrder,
			req.Normalise,
		)
		if err != nil {
			return nil, err
		}

		return flatten3D(result), nil

	case "build_cache":
		cacheDir := strings.TrimSpace(req.CacheDir)
		if cacheDir == "" {
			cacheDir = "cache"
		}

		if persistentRD == nil {
			rg := NewRecursiveG(req.K1Max, req.K2Max, req.EllMin, req.EllMax, req.D)
			rd := NewRecursiveDerivatives(
				*rg,
				defaultIfZeroInt(req.Nmax, 7),
				req.UsePrecomputedPhi1,
				req.UseNumericDerivs,
				req.UseGPU,
			)
			persistentRD = rd
		}
		persistentRD.usePhi1DerivsDiskCache = usePhi1Cache

		if err := persistentRD.BuildLoadCache(cacheDir); err != nil {
			return nil, err
		}
		// Seed phi1 derivative cache at rStar so phi1derivs_*.bin is created.
		if usePhi1Cache {
			_, _ = persistentRD.ensurePhi1DerivsCached(persistentRD.rStar, persistentRD.recursiveG.Nu)
		}

		// Return a single success flag for machine-readable callers.
		return []float64{1.0}, nil

	default:
		return nil, fmt.Errorf("unknown command %q", req.Command)
	}
}

// ----------- Batch concurrency -----------

type BatchRequest struct {
	Requests []Request `json:"requests"`
}

func runBatch(reqs []Request) [][]float64 {
	var wg sync.WaitGroup
	results := make([][]float64, len(reqs))
	errors := make([]error, len(reqs))

	wg.Add(len(reqs))
	for i, r := range reqs {
		go func(i int, r Request) {
			defer wg.Done()
			val, err := internalRunRequest(r)
			results[i] = val
			errors[i] = err
		}(i, r)
	}
	wg.Wait()

	// replace failed with NaN
	for i, err := range errors {
		if err != nil {
			results[i] = []float64{math.NaN()}
		}
	}
	return results
}

// ----------- main entrypoint -----------

func main() {
	// rg := NewRecursiveG(10, 10, 0, 6, 3)
	// rd := NewRecursiveDerivatives(*rg, 8, true, false, false)
	// PrintRecursiveDerivatives(rd)
	// rd.BuildLoadCache("cache")

	// // Test blocks
	// blockArr := []BlockType{BlockPlus, BlockMinus}
	// deltaArr := []float64{5.1}
	// spinArr := []int{3}

	// derivBlocks, err := rd.RecurseAndEvaluateDF(blockArr, 1.6, 1.2, 3.1, deltaArr, spinArr, 100, 1e-4, true)
	// fmt.Println(derivBlocks)
	// }
	output := flag.String("output", "csv", "output format: 'binary' or 'csv'")

	// CLI flags
	command := flag.String("command", "recurse_and_evaluate_df", "command to run (cli mode), e.g. recurse_and_evaluate_df or build_cache")
	k1max := flag.Int("k1max", 10, "k1max (cli mode)")
	k2max := flag.Int("k2max", 10, "k2max (cli mode)")
	ellmin := flag.Int("ellmin", 0, "ellmin (cli mode)")
	ellmax := flag.Int("ellmax", 10, "ellmax (cli mode)")
	d := flag.Int("d", 3, "dimension (cli mode)")
	delta12 := flag.Float64("delta12", 1.6, "delta12 (cli mode)")
	delta34 := flag.Float64("delta34", 1.2, "delta34 (cli mode)")
	maxIter := flag.Int("maxiter", 100, "max iterations (cli mode)")
	tol := flag.Float64("tol", 1e-4, "tolerance (cli mode)")
	deltaAve23 := flag.Float64("deltaave23", 3.1, "delta average 23 (cli mode)")
	nMax := flag.Int("nmax", 7, "Maximum derivative order (cli mode)")
	r := flag.Float64("r", 3-2*math.Sqrt(2), "r (radial coords -- cli mode)")
	eta := flag.Float64("eta", 1.0, "eta (radial coords -- cli mode)")
	normalise := flag.Bool("normalise", true, "Normalise the derivative block (cli mode)")
	usePrecomputedPhi1 := flag.Bool("use_precomputed_phi_1", true, "Use the precomputed phi 1 (cli mode)")
	useNumericDerivs := flag.Bool("use_numeric_derivs", true, "Use the numeric derivatives where available (cli mode)")
	cacheDir := flag.String("cache_dir", "cache", "Cache directory (cli mode)")
	usePhi1Cache := flag.Bool("use_phi1_cache", true, "Enable disk caching of phi1 derivatives (cli mode)")

	// list-style CLI flags
	deltas := flag.String("deltas", "5.1", "comma-separated list of delta values (cli mode)")
	ells := flag.String("ells", "3", "comma-separated list of ell values (cli mode)")
	blocks := flag.String("blocks", "+,-", "comma-separated list of block types '+' or '-' (cli mode)")

	// parallel sweep flags (for recurse and recurse_and_evaluate_g)
	zs := flag.String("zs", "0.5+0i", "comma-separated list of z values (cli mode)")

	flag.Parse()

	// CLI already uses parseComplexList(*zs), so nothing changes
	zVals := parseComplexList(*zs)
	deltaVals := parseFloatList(*deltas)
	ellVals := parseIntList(*ells)
	blockVals := parseStringList(*blocks)

	var reqs []Request
	usePhi1CacheVal := *usePhi1Cache
	reqs = append(reqs, Request{
		Command:    *command,
		K1Max:      *k1max,
		K2Max:      *k2max,
		EllMin:     *ellmin,
		EllMax:     *ellmax,
		D:          *d,
		Delta12:    *delta12,
		Delta34:    *delta34,
		MaxIter:    *maxIter,
		Tol:        *tol,
		DeltaAve23: *deltaAve23,
		Deltas:     deltaVals,
		Ells:       ellVals,
		BlockTypes: blockVals,
		Zs:         zVals,

		// Deriv block params
		R:                  *r,
		Eta:                *eta,
		Nmax:               *nMax,
		Normalise:          *normalise,
		CacheDir:           *cacheDir,
		UsePhi1Cache:       &usePhi1CacheVal,
		UsePrecomputedPhi1: *usePrecomputedPhi1,
		UseNumericDerivs:   *useNumericDerivs,
	})

	allResults := runBatch(reqs)
	for _, res := range allResults {
		if *output == "csv" {
			if err := writeCSV(os.Stdout, res); err != nil {
				fmt.Fprintln(os.Stderr, "write error:", err)
				os.Exit(1)
			}
		} else {
			if err := writeBinary(os.Stdout, res); err != nil {
				fmt.Fprintln(os.Stderr, "write error:", err)
				os.Exit(1)
			}
		}
	}
}
