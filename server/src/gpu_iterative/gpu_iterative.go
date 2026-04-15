//go:build gpu

package iterativegpu

/*
#cgo LDFLAGS: -L${SRCDIR}/../../lib -lgpuiter -lrmatrix -lcublas -lcudart
#include <stdlib.h>
#include <string.h>

// Must match the C/CUDA header exactly
typedef struct {
    int N;
    double Delta;
    int Ell;
    double C;
    int Idx;
} PolesData;

typedef struct {
    int Ell;
    double Delta;
} KeyData;

int recurse_hcoeffs_gpu(
	int n,
	int numEtaDerivs,
	int rOrder,
	const KeyData* h_keys,
	const PolesData* h_poles,
	const int* polesOffset,
	const double* h_htildeCoeffs,
	double* h_out_hcoeffs
);

double* buildRMatrixGPUDev(int* orders, double rPower, double rStar, int num);
void freeRMatrixGPU(double rPower);
void freeAllRMatricesGPU();
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// Go-side helper types (N and Ell retained for C/CUDA binary compatibility).
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

// helper to malloc C memory for n doubles and copy goSlice into it using memcpy
func mallocAndCopyDoubles(goSlice []float64) *C.double {
	n := len(goSlice)
	if n == 0 {
		return nil
	}
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(C.double(0)))
	ptr := C.malloc(size)
	if ptr == nil {
		return nil
	}
	// copy bytes
	src := unsafe.Pointer(&goSlice[0])
	C.memcpy(ptr, src, size)
	return (*C.double)(ptr)
}

// helper to malloc an array of C PolesData and copy from Go
func mallocAndCopyPoles(poles []PolesDataGo) *C.PolesData {
	n := len(poles)
	if n == 0 {
		return nil
	}
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(C.PolesData{}))
	ptr := C.malloc(size)
	if ptr == nil {
		return nil
	}
	cArray := (*[1 << 30]C.PolesData)(ptr)[:n:n]
	for i := 0; i < n; i++ {
		cArray[i].N = C.int(poles[i].N)
		cArray[i].Delta = C.double(poles[i].Delta)
		cArray[i].Ell = C.int(poles[i].Ell)
		cArray[i].C = C.double(poles[i].C)
		cArray[i].Idx = C.int(poles[i].Idx)
	}
	return (*C.PolesData)(ptr)
}

// helper to malloc and copy KeyData
func mallocAndCopyKeys(keys []KeyDataGo) *C.KeyData {
	n := len(keys)
	if n == 0 {
		return nil
	}
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(C.KeyData{}))
	ptr := C.malloc(size)
	if ptr == nil {
		return nil
	}
	cArray := (*[1 << 30]C.KeyData)(ptr)[:n:n]
	for i := 0; i < n; i++ {
		cArray[i].Ell = C.int(keys[i].Ell)
		cArray[i].Delta = C.double(keys[i].Delta)
	}
	return (*C.KeyData)(ptr)
}

// helper to malloc and copy ints (polesOffset)
func mallocAndCopyInts(goInts []int) *C.int {
	n := len(goInts)
	if n == 0 {
		return nil
	}
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(C.int(0)))
	ptr := C.malloc(size)
	if ptr == nil {
		return nil
	}
	cArr := (*[1 << 30]C.int)(ptr)[:n:n]
	for i := 0; i < n; i++ {
		cArr[i] = C.int(goInts[i])
	}
	return (*C.int)(ptr)
}

func GPURecurseHCoeffs(
	keys []KeyDataGo,
	poles []PolesDataGo,
	polesOffset []int,
	htildeCoeffs []float64,
	numEtaDerivs int,
	rOrder int,
) ([]float64, error) {
	n := len(keys)
	if n == 0 {
		return nil, errors.New("keys cannot be empty")
	}
	if numEtaDerivs <= 0 {
		return nil, errors.New("numEtaDerivs must be > 0")
	}
	if rOrder < 0 {
		return nil, errors.New("rOrder must be >= 0")
	}
	if len(polesOffset) != n+1 {
		return nil, fmt.Errorf("polesOffset length %d must be n+1=%d", len(polesOffset), n+1)
	}
	totalPoles := polesOffset[n]
	if totalPoles != len(poles) {
		return nil, fmt.Errorf("poles length %d != polesOffset[n] %d", len(poles), totalPoles)
	}
	expectedCoeffs := n * numEtaDerivs * (rOrder + 1)
	if len(htildeCoeffs) != expectedCoeffs {
		return nil, fmt.Errorf("htildeCoeffs length %d != expected %d", len(htildeCoeffs), expectedCoeffs)
	}

	cKeys := mallocAndCopyKeys(keys)
	if cKeys == nil {
		return nil, errors.New("failed to allocate keys")
	}
	defer C.free(unsafe.Pointer(cKeys))

	cPoles := mallocAndCopyPoles(poles)
	if cPoles == nil && len(poles) > 0 {
		return nil, errors.New("failed to allocate poles")
	}
	if cPoles != nil {
		defer C.free(unsafe.Pointer(cPoles))
	}

	cPolesOffset := mallocAndCopyInts(polesOffset)
	if cPolesOffset == nil {
		return nil, errors.New("failed to allocate polesOffset")
	}
	defer C.free(unsafe.Pointer(cPolesOffset))

	cHTilde := mallocAndCopyDoubles(htildeCoeffs)
	if cHTilde == nil {
		return nil, errors.New("failed to allocate htildeCoeffs")
	}
	defer C.free(unsafe.Pointer(cHTilde))

	outLen := expectedCoeffs
	cOut := C.malloc(C.size_t(outLen) * C.size_t(unsafe.Sizeof(C.double(0))))
	if cOut == nil {
		return nil, errors.New("failed to allocate output buffer")
	}
	defer C.free(cOut)

	status := C.recurse_hcoeffs_gpu(
		C.int(n),
		C.int(numEtaDerivs),
		C.int(rOrder),
		cKeys,
		cPoles,
		cPolesOffset,
		cHTilde,
		(*C.double)(cOut),
	)
	if status != 0 {
		return nil, fmt.Errorf("recurse_hcoeffs_gpu returned error %d", int(status))
	}

	resultData := unsafe.Slice((*C.double)(cOut), outLen)
	out := make([]float64, outLen)
	for i := range outLen {
		out[i] = float64(resultData[i])
	}
	return out, nil
}

// BuildRMatrixOnGPU returns a device pointer to the R matrix (GPU memory).
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

// FreeRMatrix frees a single matrix on the GPU.
func FreeRMatrix(rPower float64) {
	C.freeRMatrixGPU(C.double(rPower))
}

// FreeAllRMatrices frees all cached matrices on the GPU.
func FreeAllRMatrices() {
	C.freeAllRMatricesGPU()
}
