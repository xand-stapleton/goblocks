package main

import (
	"os"
	"strings"
	"testing"
)

func TestBuildCacheCommandCreatesCacheFile(t *testing.T) {
	tmpDir := t.TempDir()

	result, err := internalRunRequest(Request{
		Command:            "build_cache",
		K1Max:              10,
		K2Max:              10,
		EllMin:             0,
		EllMax:             10,
		D:                  3,
		Nmax:               4,
		CacheDir:           tmpDir,
		UsePrecomputedPhi1: true,
		UseNumericDerivs:   false,
		UseGPU:             false,
	})
	if err != nil {
		t.Fatalf("build_cache command failed: %v", err)
	}

	if len(result) != 1 || result[0] != 1.0 {
		t.Fatalf("unexpected build_cache result: %#v", result)
	}

	entries, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatalf("failed to read cache dir: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("build_cache did not create any files")
	}

	foundCacheFile := false
	for _, entry := range entries {
		name := entry.Name()
		if strings.HasPrefix(name, "functioncache_") && strings.HasSuffix(name, ".bin") {
			foundCacheFile = true
			break
		}
	}
	if !foundCacheFile {
		t.Fatalf("build_cache did not create a functioncache_*.bin file in %s", tmpDir)
	}
}
