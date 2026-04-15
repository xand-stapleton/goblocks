// gpu_stub.go
//go:build !gpu
// +build !gpu

package iterativegpu

import (
	"errors"
	"unsafe"
)

// Go-side helper types (mirrors gpu build file API).
type PolesDataGo struct {
	N     int
	Delta float64
	Ell   int
	C     float64
	Idx   int
}

type KeyDataGo struct {
	Ell   int
	Delta float64
}

func GPURecurseHCoeffs(
	keys []KeyDataGo,
	poles []PolesDataGo,
	polesOffset []int,
	htildeCoeffs []float64,
	numEtaDerivs int,
	rOrder int,
) ([]float64, error) {
	return nil, errors.New("GPU support not compiled; rebuild with -tags gpu to enable")
}

func BuildRMatrixOnGPU(numDerivs int, rStar float64, derivativeOrdersREta [][2]int, rPower float64) unsafe.Pointer {
	return nil
}

func FreeRMatrix(rPower float64) {}

func FreeAllRMatrices() {}
