// gpu_stub.go
//go:build !gpu
// +build !gpu

package iterativegpu

import (
	"errors"
)

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
