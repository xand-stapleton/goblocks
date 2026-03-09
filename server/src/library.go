package main

/*
#cgo CFLAGS: -g -Wall
#include <stdlib.h>
*/
import "C"

import (
	"encoding/json"
	"sync"
	"unsafe"
)

// Global handle table
var handleMu sync.Mutex
var nextHandle int64 = 1
var handles = map[int64]interface{}{}

type rdProperties struct {
	K1Max              int    `json:"k1_max"`
	K2Max              int    `json:"k2_max"`
	EllMin             int    `json:"ell_min"`
	EllMax             int    `json:"ell_max"`
	D                  int    `json:"d"`
	Nmax               int    `json:"nmax"`
	UsePrecomputedPhi1 bool   `json:"use_precomputed_phi1"`
	UseNumericDerivs   bool   `json:"use_numeric_derivs"`
	UseGPU             bool   `json:"use_gpu"`
	CacheDir           string `json:"cache_dir"`
}

//export RunRequest
func RunRequest(jsonStr *C.char, outLen *C.longlong) *C.double {
	reqStr := C.GoString(jsonStr)

	var req Request
	if err := json.Unmarshal([]byte(reqStr), &req); err != nil {
		*outLen = 0
		return nil
	}

	if req.ZsStr != nil {
		req.Zs = parseComplexList(joinStrings(req.ZsStr))
	}

	// if req.Ells != nil {
	// 	ellVals := parseIntList(req.Ells)
	// }

	// if req.Deltas != nil {
	// 	deltaVals := parseFloatList(req.Deltas)
	// }

	res, err := internalRunRequest(req)
	if err != nil {
		*outLen = 0
		return nil
	}

	// suppose internalRunRequest returns []float64 (flat)
	*outLen = C.longlong(len(res))

	// allocate C memory for result
	size := len(res) * int(unsafe.Sizeof(res[0]))
	ptr := C.malloc(C.size_t(size))
	if ptr == nil {
		*outLen = 0
		return nil
	}

	// copy Go slice -> C memory
	dst := (*[1 << 30]float64)(ptr)[:len(res):len(res)]
	copy(dst, res)

	return (*C.double)(ptr)
}

//export FreeResult
func FreeResult(ptr *C.double) {
	C.free(unsafe.Pointer(ptr))
}

// helper
func joinStrings(s []string) string {
	joined := ""
	for i, v := range s {
		if i > 0 {
			joined += ","
		}
		joined += v
	}
	return joined
}

func newHandle(obj interface{}) int64 {
	handleMu.Lock()
	h := nextHandle
	nextHandle++
	handles[h] = obj
	handleMu.Unlock()
	return h
}

func getHandle(h int64) interface{} {
	handleMu.Lock()
	obj := handles[h]
	handleMu.Unlock()
	return obj
}

func deleteHandle(h int64) {
	handleMu.Lock()
	delete(handles, h)
	handleMu.Unlock()
}

// TODO: Want to create a subrequest because request contains a bunch of redundant fields
//
//export CreateRD
func CreateRD(jsonStr *C.char) C.longlong {
	rdPropStr := C.GoString(jsonStr)

	var rdProps rdProperties
	if err := json.Unmarshal([]byte(rdPropStr), &rdProps); err != nil {
		return 0
	}

	rg := NewRecursiveG(rdProps.K1Max, rdProps.K2Max, rdProps.EllMin, rdProps.EllMax, rdProps.D)
	rd := NewRecursiveDerivatives(*rg, rdProps.Nmax, rdProps.UsePrecomputedPhi1, rdProps.UseNumericDerivs, rdProps.UseGPU)
	rd.BuildLoadCache(rdProps.CacheDir)
	h := newHandle(rd)
	return C.longlong(h)
}

//export FreeRD
func FreeRD(h C.longlong) {
	deleteHandle(int64(h))
}
