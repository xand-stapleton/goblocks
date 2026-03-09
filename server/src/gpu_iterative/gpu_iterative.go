//go:build gpu

package iterativegpu

/*
#cgo LDFLAGS: -L${SRCDIR}/../lib -lgpuiter -lcublas -lcudart
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

// Function from iterative_update.cu
int iterative_update_gpu(
    int m, int n,
    const double* h_dg,
    const double* h_dgTilde,
    int maxIterations,
    double tol,
    const KeyData* h_keys,
    const PolesData* h_poles,
    const int* polesOffset,
    const double** h_R_ptrs,
    double* h_out_dg,
    int* converged
);
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"gonum.org/v1/gonum/mat"
)

// convertToColumnMajor returns a newly allocated []float64 in column-major order suitable for C consumption.
func convertToColumnMajor(r *mat.Dense) []float64 {
	m, n := r.Dims()
	data := make([]float64, m*n)
	// Gonum stores in row-major; copy into column-major
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			data[j*m+i] = r.At(i, j)
		}
	}
	return data
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

// freeRPtrs frees the pointer array and each R matrix allocated by mallocAndCopyRPtrs
func freeRPtrs(cptr **C.double, total int) {
	if cptr == nil {
		return
	}
	ptrs := (*[1 << 30]*C.double)(unsafe.Pointer(cptr))[:total:total]
	for i := 0; i < total; i++ {
		if ptrs[i] != nil {
			C.free(unsafe.Pointer(ptrs[i]))
		}
	}
	C.free(unsafe.Pointer(cptr))
}

func mallocAndCopyRPtrsParallel(Rlist [][]float64, m int) (**C.double, error) {
	total := len(Rlist)
	if total == 0 {
		return nil, nil
	}

	c_Rptrs := C.malloc(C.size_t(total) * C.size_t(unsafe.Sizeof(uintptr(0))))
	if c_Rptrs == nil {
		return nil, fmt.Errorf("malloc failed for R ptrs")
	}
	ptrs := (*[1 << 30]*C.double)(c_Rptrs)[:total:total]

	numWorkers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	errCh := make(chan error, 1) // only need to report first error

	chunkSize := (total + numWorkers - 1) / numWorkers // divide evenly
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > total {
			end = total
		}
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				r := Rlist[i]
				if len(r) != m*m {
					select {
					case errCh <- fmt.Errorf("Rlist[%d] length mismatch: got %d, want %d", i, len(r), m*m):
					default:
					}
					return
				}
				ptr := mallocAndCopyDoubles(r)
				if ptr == nil {
					select {
					case errCh <- fmt.Errorf("malloc failed for R[%d]", i):
					default:
					}
					return
				}
				ptrs[i] = ptr
			}
		}(start, end)
	}

	wg.Wait()
	close(errCh)

	if err, ok := <-errCh; ok {
		// cleanup
		for _, p := range ptrs {
			if p != nil {
				C.free(unsafe.Pointer(p))
			}
		}
		C.free(c_Rptrs)
		return nil, err
	}

	return (**C.double)(c_Rptrs), nil
}

func GPUIterativeUpdate(
	dg, dgTilde *mat.Dense,
	keys []KeyDataGo,
	poles []PolesDataGo,
	polesOffset []int,
	Rlist [][]float64,
	maxIter int,
	tol float64,
) (*mat.Dense, bool, error) {
	startTotal := time.Now()

	// --- 1. Input validation ---
	start := time.Now()
	if dg == nil || dgTilde == nil {
		return nil, false, errors.New("dg and dgTilde cannot be nil")
	}
	m, n := dg.Dims()
	mt, nt := dgTilde.Dims()
	if m != mt || n != nt {
		return nil, false, errors.New("dg and dgTilde dimensions must match")
	}
	if len(keys) != n {
		return nil, false, fmt.Errorf("keys length %d must equal number of columns %d", len(keys), n)
	}
	if len(polesOffset) != n+1 {
		return nil, false, fmt.Errorf("polesOffset length must be n+1")
	}
	totalPoles := polesOffset[n]
	if len(poles) != totalPoles {
		return nil, false, fmt.Errorf("poles length %d != expected %d (polesOffset[n])", len(poles), totalPoles)
	}
	if len(Rlist) != totalPoles {
		return nil, false, fmt.Errorf("Rlist length %d != totalPoles %d", len(Rlist), totalPoles)
	}
	fmt.Println("Validation:", time.Since(start))

	// --- 2. Column-major conversion ---
	start = time.Now()
	h_dg := convertToColumnMajor(dg)
	h_dgTilde := convertToColumnMajor(dgTilde)
	fmt.Println("Column-major conversion:", time.Since(start))

	// --- 3. Allocate GPU data ---
	start = time.Now()
	c_dg := mallocAndCopyDoubles(h_dg)
	if c_dg == nil {
		return nil, false, errors.New("failed to allocate C memory for dg")
	}
	defer C.free(unsafe.Pointer(c_dg))

	c_dgTilde := mallocAndCopyDoubles(h_dgTilde)
	if c_dgTilde == nil {
		return nil, false, errors.New("failed to allocate C memory for dgTilde")
	}
	defer C.free(unsafe.Pointer(c_dgTilde))

	c_keys := mallocAndCopyKeys(keys)
	if c_keys == nil {
		return nil, false, errors.New("failed to allocate keys")
	}
	defer C.free(unsafe.Pointer(c_keys))

	c_poles := mallocAndCopyPoles(poles)
	if c_poles == nil && len(poles) > 0 {
		return nil, false, errors.New("failed to allocate poles")
	}
	if c_poles != nil {
		defer C.free(unsafe.Pointer(c_poles))
	}

	c_polesOffset := mallocAndCopyInts(polesOffset)
	if c_polesOffset == nil {
		return nil, false, errors.New("failed to allocate polesOffset")
	}
	defer C.free(unsafe.Pointer(c_polesOffset))
	fmt.Println("GPU data allocation:", time.Since(start))

	// --- 4. Parallel allocation of R matrices ---
	start = time.Now()
	c_Rptrs, err := mallocAndCopyRPtrsParallel(Rlist, m)
	if err != nil {
		return nil, false, err
	}
	if c_Rptrs != nil {
		defer freeRPtrs(c_Rptrs, totalPoles)
	}
	fmt.Println("Parallel R allocation:", time.Since(start))

	// --- 5. Allocate output buffer ---
	start = time.Now()
	outCount := m * n
	c_out := C.malloc(C.size_t(outCount) * C.size_t(unsafe.Sizeof(C.double(0))))
	if c_out == nil {
		return nil, false, errors.New("failed to allocate output buffer")
	}
	defer C.free(c_out)
	fmt.Println("Output buffer allocation:", time.Since(start))

	// --- 6. GPU kernel execution ---
	start = time.Now()
	var c_converged C.int
	status := C.iterative_update_gpu(
		C.int(m),
		C.int(n),
		(*C.double)(c_dg),
		(*C.double)(c_dgTilde),
		C.int(maxIter),
		C.double(tol),
		c_keys,
		c_poles,
		c_polesOffset,
		c_Rptrs,
		(*C.double)(c_out),
		&c_converged,
	)
	if status != 0 {
		return nil, false, fmt.Errorf("iterative_update_gpu returned error %d", int(status))
	}
	converged := c_converged != 0
	fmt.Println("GPU kernel execution:", time.Since(start))

	// --- 7. Convert output to row-major ---
	start = time.Now()
	resultData := unsafe.Slice((*C.double)(c_out), outCount)
	outMat := mat.NewDense(m, n, nil)
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			outMat.Set(i, j, float64(resultData[j*m+i]))
		}
	}
	fmt.Println("Output conversion:", time.Since(start))

	fmt.Println("Total GPUIterativeUpdate time:", time.Since(startTotal))
	return outMat, converged, nil
}
