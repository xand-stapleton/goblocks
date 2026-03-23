package main

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func assertCloseAbsRel(t *testing.T, got, want, absTol, relTol float64, label string) {
	t.Helper()
	diff := math.Abs(got - want)
	if diff <= absTol {
		return
	}
	scale := math.Max(1.0, math.Max(math.Abs(got), math.Abs(want)))
	if diff/scale <= relTol {
		return
	}
	t.Fatalf("%s: got %.18e want %.18e (|diff|=%.3e)", label, got, want, diff)
}

func TestPhi1PrecomputedMatchesAnalyticAtCrossingSymmetricPointD3(t *testing.T) {
	rg := NewRecursiveG(10, 10, 0, 10, 3)

	// Keep nMax consistent with other tests (avoids derivative-order edge cases).
	rdPre := NewRecursiveDerivatives(*rg, 8, true, false, false)
	rdAna := NewRecursiveDerivatives(*rg, 8, false, false, false)

	r := 3 - 2*math.Sqrt(2)
	nu := 0.5

	// The precomputed table currently provides i=0..27.
	for i := 0; i <= 27; i++ {
		gotPre := rdPre.phi1(i, 0, r, nu)
		gotAna := rdAna.phi1(i, 0, r, nu)

		// These are pure float64 computations; tolerances are set tightly but not unrealistically.
		assertCloseAbsRel(t, gotPre, gotAna, 1e-13, 1e-13, "phi1 precomputed vs analytic (i="+itoa(i)+")")
	}
}

func TestPhi1PrecomputedPanicsOutsideCrossingSymmetricPoint(t *testing.T) {
	rg := NewRecursiveG(10, 10, 0, 10, 3)
	rdPre := NewRecursiveDerivatives(*rg, 8, true, false, false)
	rdAna := NewRecursiveDerivatives(*rg, 8, false, false, false)

	// Away from the crossing-symmetric point the precomputed table is not valid,
	// so rdPre.phi1 should transparently fall back to the analytic implementation.
	r := (3 - 2*math.Sqrt(2)) + 1e-3
	nu := 0.5

	for i := 0; i <= 10; i++ {
		gotPre := rdPre.phi1(i, 0, r, nu)
		gotAna := rdAna.phi1(i, 0, r, nu)
		assertCloseAbsRel(t, gotPre, gotAna, 1e-13, 1e-13, "phi1 precomputed fallback vs analytic (i="+itoa(i)+")")
	}
}

func TestPhi1DerivsDiskCacheRoundTrip(t *testing.T) {
	rg := NewRecursiveG(10, 10, 0, 10, 3)
	rd1 := NewRecursiveDerivatives(*rg, 4, false, false, false)
	cacheDir := t.TempDir()

	// Avoid the expensive PrepopulateCache() path by pre-creating an empty function cache file.
	// This keeps the test focused on phi1's dedicated cache file.
	hash := rd1.NewRDConfig().Hash()
	functionPath := filepath.Join(cacheDir, "functioncache_"+hash+".bin")
	if err := SaveCache(functionPath, &SerialisableFunctionCache{}); err != nil {
		t.Fatalf("failed to write dummy function cache: %v", err)
	}

	if err := rd1.BuildLoadCache(cacheDir); err != nil {
		t.Fatalf("BuildLoadCache: %v", err)
	}

	r := rd1.rStar
	nu := rd1.recursiveG.Nu
	got1 := rd1.phi1(10, 0, r, nu)

	phiCfg := Phi1DerivsConfig{Nu: nu, MaxOrder: rd1.maxPhi1DerivOrder()}
	phiPath := filepath.Join(cacheDir, "phi1derivs_"+phiCfg.Hash()+".bin")
	if _, err := os.Stat(phiPath); err != nil {
		t.Fatalf("expected phi1 cache file to exist at %s: %v", phiPath, err)
	}

	loaded, err := loadPhi1DerivsCache(phiPath)
	if err != nil {
		t.Fatalf("loadPhi1DerivsCache: %v", err)
	}
	if _, ok := loaded.ByR[phi1RKey(r)]; !ok {
		t.Fatalf("expected cached derivatives for r=%g", r)
	}

	rd2 := NewRecursiveDerivatives(*rg, 4, false, false, false)
	if err := rd2.BuildLoadCache(cacheDir); err != nil {
		t.Fatalf("BuildLoadCache (2): %v", err)
	}
	got2 := rd2.phi1(10, 0, r, nu)

	assertCloseAbsRel(t, got2, got1, 0, 1e-15, "phi1 disk-cache round trip")
}

func itoa(x int) string {
	// tiny helper to avoid fmt import in test hot path
	if x == 0 {
		return "0"
	}
	neg := false
	if x < 0 {
		neg = true
		x = -x
	}
	buf := make([]byte, 0, 12)
	for x > 0 {
		buf = append(buf, byte('0'+x%10))
		x /= 10
	}
	// reverse
	for i, j := 0, len(buf)-1; i < j; i, j = i+1, j-1 {
		buf[i], buf[j] = buf[j], buf[i]
	}
	if neg {
		buf = append([]byte{'-'}, buf...)
	}
	return string(buf)
}
