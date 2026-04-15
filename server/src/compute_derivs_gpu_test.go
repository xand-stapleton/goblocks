//go:build gpu

package main

import (
	"testing"
)

func buildDerivParityInputs(
	t *testing.T,
	rd *RecursiveDerivatives,
	delta12, delta34 float64,
	rOrder int,
) (map[int][][]float64, map[int][]PolesData, int) {
	t.Helper()

	rd.recursiveG.unique_poles_map = make(map[PoleKey]int)
	rd.recursiveG.idxToKey = nil
	polesData := rd.recursiveG.getAllPolesData(delta12, delta34)
	if len(rd.recursiveG.idxToKey) == 0 {
		t.Fatal("no pole keys generated")
	}

	maxEtaDeriv := 0
	for _, pair := range rd.derivativeOrdersREta {
		if pair[1] > maxEtaDeriv {
			maxEtaDeriv = pair[1]
		}
	}
	numEtaDerivs := maxEtaDeriv + 1

	htildeCoeffsByEll := make(map[int][][]float64)
	for _, key := range rd.recursiveG.idxToKey {
		if _, ok := htildeCoeffsByEll[key.Ell]; ok {
			continue
		}
		coeffs, err := computeHTildeRCoeffsWithEtaDerivs(
			rd,
			&rd.recursiveG,
			delta12,
			delta34,
			key.Ell,
			rd.etaStar,
			rOrder,
			maxEtaDeriv,
		)
		if err != nil {
			t.Fatalf("computeHTildeRCoeffsWithEtaDerivs failed for ell=%d: %v", key.Ell, err)
		}
		htildeCoeffsByEll[key.Ell] = coeffs
	}

	return htildeCoeffsByEll, polesData, numEtaDerivs
}

func TestHCoeffsGPUEqualsCPU_Nmax8(t *testing.T) {
	rg := NewRecursiveG(10, 10, 0, 10, 3)
	rd := NewRecursiveDerivatives(*rg, 8, true, false, true)

	delta12 := 1.6
	delta34 := 1.2
	rOrder := 60

	htildeCoeffsByEll, polesData, numEtaDerivs := buildDerivParityInputs(t, rd, delta12, delta34, rOrder)

	cpu := rd.computeHCoeffsCPU(htildeCoeffsByEll, polesData, numEtaDerivs, rOrder)
	gpu, err := rd.computeHCoeffsGPU(htildeCoeffsByEll, polesData, numEtaDerivs, rOrder)
	if err != nil {
		t.Fatalf("computeHCoeffsGPU failed: %v", err)
	}

	if len(cpu) != len(gpu) {
		t.Fatalf("column count mismatch: cpu=%d gpu=%d", len(cpu), len(gpu))
	}
	for j := range cpu {
		if len(cpu[j]) != len(gpu[j]) {
			t.Fatalf("eta-deriv count mismatch at col %d: cpu=%d gpu=%d", j, len(cpu[j]), len(gpu[j]))
		}
		for q := range cpu[j] {
			if len(cpu[j][q]) != len(gpu[j][q]) {
				t.Fatalf("r-order length mismatch at col %d q %d: cpu=%d gpu=%d", j, q, len(cpu[j][q]), len(gpu[j][q]))
			}
			for p := range cpu[j][q] {
				assertCloseAbsRel(t, gpu[j][q][p], cpu[j][q][p], 1e-12, 1e-10,
					"h-coeff mismatch")
			}
		}
	}
}

func TestRecurseAndEvaluateDF_GPUEqualsCPU_Nmax8(t *testing.T) {
	params := struct {
		delta12    float64
		delta34    float64
		deltaAve23 float64
		deltas     []float64
		spins      []int
		rOrder     int
		normalise  bool
	}{
		delta12:    1.6,
		delta34:    1.2,
		deltaAve23: 3.1,
		deltas:     []float64{5.1, 6.2},
		spins:      []int{3, 4},
		rOrder:     100,
		normalise:  true,
	}

	blockTypes := []BlockType{BlockPlus, BlockMinus}

	rgCPU := NewRecursiveG(10, 10, 0, 10, 3)
	rdCPU := NewRecursiveDerivatives(*rgCPU, 8, true, false, false)
	cpu3D, err := rdCPU.RecurseAndEvaluateDF(
		blockTypes,
		params.delta12,
		params.delta34,
		params.deltaAve23,
		params.deltas,
		params.spins,
		params.rOrder,
		params.normalise,
	)
	if err != nil {
		t.Fatalf("CPU RecurseAndEvaluateDF failed: %v", err)
	}
	cpu := flatten3D(cpu3D)

	rgGPU := NewRecursiveG(10, 10, 0, 10, 3)
	rdGPU := NewRecursiveDerivatives(*rgGPU, 8, true, false, true)
	gpu3D, err := rdGPU.RecurseAndEvaluateDF(
		blockTypes,
		params.delta12,
		params.delta34,
		params.deltaAve23,
		params.deltas,
		params.spins,
		params.rOrder,
		params.normalise,
	)
	if err != nil {
		t.Fatalf("GPU RecurseAndEvaluateDF failed: %v", err)
	}
	gpu := flatten3D(gpu3D)

	if len(cpu) != len(gpu) {
		t.Fatalf("output length mismatch: cpu=%d gpu=%d", len(cpu), len(gpu))
	}

	for i := range cpu {
		assertCloseAbsRel(t, gpu[i], cpu[i], 1e-12, 1e-10, "df derivative mismatch")
	}
}
