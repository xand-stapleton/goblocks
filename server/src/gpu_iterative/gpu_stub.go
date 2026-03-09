// gpu_stub.go
//go:build !gpu
// +build !gpu

package iterativegpu

import (
	"errors"

	"gonum.org/v1/gonum/mat"
)

// GPUIterativeUpdate is a stub used when GPU support is not compiled.
// It returns an error to indicate that GPU functionality is unavailable.
func GPUIterativeUpdate(
	dg, dgTilde *mat.Dense,
	keys []KeyDataGo,
	poles []PolesDataGo,
	polesOffset []int,
	Rlist [][]float64,
	maxIter int,
	tol float64,
) (*mat.Dense, bool, error) {
	return nil, false, errors.New("GPU support not compiled; rebuild with -tags gpu to enable")
}
