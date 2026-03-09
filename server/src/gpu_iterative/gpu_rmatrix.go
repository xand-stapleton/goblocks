//go:build gpu

package iterativegpu

/*
#cgo LDFLAGS: -L${SRCDIR}/../lib -lrmatrix
#include <stdlib.h>

// Must match the C/CUDA header exactly
double* buildRMatrixGPUDev(int* orders, double rPower, double rStar, int num);
void freeRMatrixGPU(double rPower);
void freeAllRMatricesGPU();
*/
import "C"
import (
	"unsafe"
)

// BuildRMatrixOnGPU returns a device pointer to the R matrix (GPU memory)
func BuildRMatrixOnGPU(numDerivs int, rStar float64, derivativeOrdersREta [][2]int, rPower float64) unsafe.Pointer {
	orders := make([]C.int, 2*numDerivs)
	for i, mn := range derivativeOrdersREta {
		orders[2*i] = C.int(mn[0])
		orders[2*i+1] = C.int(mn[1])
	}
	return unsafe.Pointer(C.buildRMatrixGPUDev(
		(*C.int)(unsafe.Pointer(&orders[0])),
		C.double(rPower),
		C.double(rStar),
		C.int(numDerivs),
	))
}

// FreeRMatrix frees a single matrix on the GPU
func FreeRMatrix(rPower float64) {
	C.freeRMatrixGPU(C.double(rPower))
}

// FreeAllRMatrices frees all cached matrices on the GPU
func FreeAllRMatrices() {
	C.freeAllRMatricesGPU()
}
