package main

import (
	"crypto/sha256"
	"encoding/gob"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

type FunctionCache struct {
	rfCache              sync.Map // key: [2]float64 -> float64
	ffCache              sync.Map // key: [2]float64 > float64
	binomialCache        sync.Map // key: [2]int -> float64
	factorialCache       sync.Map // key: int -> float64
	gegenbauerCache      sync.Map // key: [3]float64 -> float64
	partitionFactorCache sync.Map // key: [4]int -> float64
	powCache             sync.Map // key: [2]float64 -> float64
	cacheMu              sync.RWMutex
}

// RDConfig -- we will want to serialise our caches, so we need a config which we can hash to ensure that the cache makes sense when loaded (i.e. allow for different args).
type RDConfig struct {
	RStar    float64
	EtaStar  float64
	ZStar    float64
	ZBarStar float64

	DerivativeOrdersZZBar [][2]int
	DerivativeOrdersREta  [][2]int
	NumDerivs             int
}

// NewRDConfig -- creates a new recursive derivative config for serialisation
func (rd *RecursiveDerivatives) NewRDConfig() RDConfig {
	return RDConfig{
		RStar:                 rd.rStar,
		EtaStar:               rd.etaStar,
		ZStar:                 rd.zStar,
		ZBarStar:              rd.zBarStar,
		DerivativeOrdersZZBar: rd.derivativeOrdersZZBar,
		DerivativeOrdersREta:  rd.derivativeOrdersREta,
		NumDerivs:             rd.numDerivs,
	}
}

func (c RDConfig) Hash() string {
	b, _ := json.Marshal(c)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// SerialisableFunctionCache -- since our function cache contains map (which cannot be serialised), we need a serialisable function cache
type SerialisableFunctionCache struct {
	RF              map[[2]float64]float64
	FF              map[[2]float64]float64
	Binomial        map[[2]int]float64
	Factorial       map[int]float64
	Gegenbauer      map[[3]float64]float64
	PartitionFactor map[[4]int]float64
}

func (fc *FunctionCache) ToSerialisable() *SerialisableFunctionCache {
	s := &SerialisableFunctionCache{
		RF:              make(map[[2]float64]float64),
		FF:              make(map[[2]float64]float64),
		Binomial:        make(map[[2]int]float64),
		Factorial:       make(map[int]float64),
		Gegenbauer:      make(map[[3]float64]float64),
		PartitionFactor: make(map[[4]int]float64),
	}

	// rfCache
	fc.rfCache.Range(func(k, v any) bool {
		s.RF[k.([2]float64)] = v.(float64)
		return true
	})

	// ffCache
	fc.ffCache.Range(func(k, v any) bool {
		s.FF[k.([2]float64)] = v.(float64)
		return true
	})

	// binomialCache
	fc.binomialCache.Range(func(k, v any) bool {
		s.Binomial[k.([2]int)] = v.(float64)
		return true
	})

	// factorialCache
	fc.factorialCache.Range(func(k, v any) bool {
		s.Factorial[k.(int)] = v.(float64)
		return true
	})

	// gegenbauerCache
	fc.gegenbauerCache.Range(func(k, v any) bool {
		s.Gegenbauer[k.([3]float64)] = v.(float64)
		return true
	})

	// partitionFactorCache
	fc.partitionFactorCache.Range(func(k, v any) bool {
		s.PartitionFactor[k.([4]int)] = v.(float64)
		return true
	})

	return s
}

func (fc *FunctionCache) LoadSerialisable(s *SerialisableFunctionCache) {
	// rfCache
	for k, v := range s.RF {
		fc.rfCache.Store(k, v)
	}

	// ffCache
	for k, v := range s.FF {
		fc.ffCache.Store(k, v)
	}

	// binomialCache
	for k, v := range s.Binomial {
		fc.binomialCache.Store(k, v)
	}

	// factorialCache
	for k, v := range s.Factorial {
		fc.factorialCache.Store(k, v)
	}

	// gegenbauerCache
	for k, v := range s.Gegenbauer {
		fc.gegenbauerCache.Store(k, v)
	}

	// partitionFactorCache
	for k, v := range s.PartitionFactor {
		fc.partitionFactorCache.Store(k, v)
	}
}

func SaveCache(path string, s *SerialisableFunctionCache) error {

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
	return enc.Encode(s)
}

func LoadCache(path string) (*SerialisableFunctionCache, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	dec := gob.NewDecoder(f)
	var s SerialisableFunctionCache
	if err := dec.Decode(&s); err != nil {
		return nil, err
	}
	return &s, nil
}

func (rd *RecursiveDerivatives) BuildLoadCache(basePath string) error {
	// Create a deterministic cache filename based on the current config
	cfg := rd.NewRDConfig()
	if err := os.MkdirAll(basePath, 0o755); err != nil {
		return err
	}

	// Try to load an existing cache
	hash := cfg.Hash()
	path := filepath.Join(basePath, "functioncache_"+hash+".bin")
	serial, err := LoadCache(path)
	if err == nil {
		// Cache file exists – deserialize it into the in‑memory cache
		rd.functionCache.LoadSerialisable(serial)
		return nil
	}

	// No cache found (or loading failed); build a fresh one
	if err := rd.PrepopulateCache(); err != nil {
		return fmt.Errorf("prepopulating cache: %w", err)
	}

	// // Serialize the newly built cache and write it to disk for future runs
	if err := rd.SaveCache(path); err != nil {
		return fmt.Errorf("saving new cache: %w", err)
	}
	return nil
}

func (rd *RecursiveDerivatives) SaveCache(path string) error {
	s := rd.functionCache.ToSerialisable()
	return SaveCache(path, s)
}

// ---------------------------------------------------------------------------------
// Cachable functions
// ---------------------------------------------------------------------------------
func (rd *RecursiveDerivatives) rf(i float64, j int) float64 {
	key := [2]float64{i, float64(j)}
	if val, ok := rd.functionCache.rfCache.Load(key); ok {
		return val.(float64)
	}
	val := rf(i, j)
	rd.functionCache.rfCache.Store(key, val)
	return val
}

func (rd *RecursiveDerivatives) ff(i float64, j int) float64 {
	key := [2]float64{i, float64(j)}
	if val, ok := rd.functionCache.ffCache.Load(key); ok {
		return val.(float64)
	}
	val := ff(i, j)
	rd.functionCache.ffCache.Store(key, val)
	return val
}

func (rd *RecursiveDerivatives) comb(n, i int) float64 {
	key := [2]int{n, i}
	if val, ok := rd.functionCache.binomialCache.Load(key); ok {
		return val.(float64)
	}
	val := comb(n, i)
	rd.functionCache.binomialCache.Store(key, val)
	return val
}

func (rd *RecursiveDerivatives) Pow(x, y float64) float64 {
	key := [2]float64{float64(x), float64(y)}
	if val, ok := rd.functionCache.powCache.Load(key); ok {
		return val.(float64)
	}
	val := math.Pow(x, y)
	rd.functionCache.powCache.Store(key, val)
	return val
}

func (rd *RecursiveDerivatives) factorial(l int) float64 {
	if val, ok := rd.functionCache.factorialCache.Load(l); ok {
		return val.(float64)
	}
	val := factorial(l)
	rd.functionCache.factorialCache.Store(l, val)
	return val
}

func (rd *RecursiveDerivatives) GegenbauerC(i int, j, k float64) float64 {
	key := [3]float64{float64(i), float64(j), float64(k)}
	if val, ok := rd.functionCache.gegenbauerCache.Load(key); ok {
		return val.(float64)
	}
	val := GegenbauerC(i, j, k)
	rd.functionCache.gegenbauerCache.Store(key, val)
	return val
}

func (rd *RecursiveDerivatives) PartitionFactor(p, q, m, n int) float64 {
	key := [4]int{p, q, m, n}

	// --- 1. Try reading from cache ---
	if val, ok := rd.functionCache.partitionFactorCache.Load(key); ok {
		return val.(float64)
	}

	// --- 2. Compute value if not cached ---
	abPairs := make([]Pair, 0)
	for a := 0; a <= m+n; a++ {
		for b := 0; b <= m+n; b++ {
			if sum := a + b; sum >= 1 && sum <= m+n {
				abPairs = append(abPairs, Pair{A: a, B: b})
			}
		}
	}

	partitions := GeneratePartitions(abPairs, p, q, m, n)
	totalPQ := 0.0

	for _, part := range partitions {
		kTuple, lTuple := part.K, part.L

		sumK, sumL := sumInt(kTuple), sumInt(lTuple)
		if sumK != p || sumL != q {
			panic(fmt.Sprintf("partition p,q constraint failed: p=%d q=%d sumK=%d sumL=%d", p, q, sumK, sumL))
		}

		sumA, sumB := 0, 0
		for i, pair := range abPairs {
			k, ell := kTuple[i], lTuple[i]
			sumA += pair.A * (k + ell)
			sumB += pair.B * (k + ell)
		}
		if sumA != m || sumB != n {
			panic(fmt.Sprintf("partition derivative constraint failed: m=%d n=%d sumA=%d sumB=%d", m, n, sumA, sumB))
		}

		coeffComb := 1.0
		coeffDeriv := 1.0

		for i, pair := range abPairs {
			a, b := pair.A, pair.B
			k := kTuple[i]
			ell := lTuple[i]
			kPlusL := k + ell

			coeffComb /= math.Pow(rd.factorial(a), float64(kPlusL))
			coeffComb /= math.Pow(rd.factorial(b), float64(kPlusL))
			coeffComb /= rd.factorial(k)
			coeffComb /= rd.factorial(ell)

			rPowerCache, _ := rd.derivativeEvaluatorCache.RDerivToPowCache(a, b, k)
			etaPowerCache, _ := rd.derivativeEvaluatorCache.EtaDerivToPowCache(a, b, ell)
			coeffDeriv *= rPowerCache
			coeffDeriv *= etaPowerCache
		}

		totalPQ += coeffComb * coeffDeriv
	}

	// --- 3. Store result in cache ---
	rd.functionCache.partitionFactorCache.Store(key, totalPQ)

	return totalPQ
}

// --- Parallel prepopulation ---
func (rd *RecursiveDerivatives) PrepopulateCache() error {
	type Args struct{ P, Q, M, N int }
	unique := make(map[Args]struct{})

	// Generate unique combinations
	for _, mn := range rd.derivativeOrdersZZBar {
		m, n := mn[0], mn[1]
		for i := 0; i <= m; i++ {
			for j := 0; j <= n; j++ {
				sum := m + n - i - j
				for p := 0; p <= sum; p++ {
					for q := 0; q <= sum-p; q++ {
						key := Args{P: p, Q: q, M: m - i, N: n - j}
						unique[key] = struct{}{}
					}
				}
			}
		}
	}

	argsList := make([]Args, 0, len(unique))
	for k := range unique {
		argsList = append(argsList, k)
	}
	sort.Slice(argsList, func(i, j int) bool {
		a, b := argsList[i], argsList[j]
		if a.M != b.M {
			return a.M < b.M
		}
		if a.N != b.N {
			return a.N < b.N
		}
		if a.P != b.P {
			return a.P < b.P
		}
		return a.Q < b.Q
	})

	total := len(argsList)
	fmt.Printf("Prepopulating cache with %d entries...\n", total)

	// Parallel worker pool
	workers := runtime.NumCPU()
	tasks := make(chan Args, workers*2)
	var wg sync.WaitGroup
	var progress uint64

	doneProgress := make(chan struct{})
	go func() {
		ticker := time.NewTicker(200 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				current := atomic.LoadUint64(&progress)
				fmt.Printf("\rProgress: %d/%d", current, total)
			case <-doneProgress:
				fmt.Printf("\rProgress: %d/%d\n", atomic.LoadUint64(&progress), total)
				return
			}
		}
	}()

	worker := func() {
		defer wg.Done()
		for a := range tasks {
			_ = rd.PartitionFactor(a.P, a.Q, a.M, a.N)
			atomic.AddUint64(&progress, 1)
		}
	}

	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go worker()
	}

	for _, a := range argsList {
		tasks <- a
	}
	close(tasks)

	wg.Wait()
	close(doneProgress)
	fmt.Println("Done prepopulating cache.")
	return nil
}
