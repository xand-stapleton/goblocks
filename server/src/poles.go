package main

import (
	"math"
)

func (rg *RecursiveG) getAllPolesData(delta12, delta34 float64) map[int][]PolesData {
	// polesTotal is keyed by spin!
	polesTotal := make(map[int][]PolesData)
	for ell := rg.EllMin; ell <= rg.EllMax; ell++ {
		list := []PolesData{}
		list = append(list, rg.getDelta1Poles(ell, delta12, delta34)...)
		list = append(list, rg.getDelta2Poles(ell, delta12, delta34)...)
		list = append(list, rg.getDelta3Poles(ell, delta12, delta34)...)
		if len(list) > 0 {
			polesTotal[ell] = list
		}
	}
	return polesTotal
}

// Type I (row 1 of Table 1)
func (rg *RecursiveG) getDelta1Poles(ell int, delta12, delta34 float64) []PolesData {
	c1 := func(ell, k int) float64 {
		denom := math.Pow(factorial(k), 2) * rf(float64(ell)+rg.Nu, k)

		num := rf(float64(ell)+2*rg.Nu, k)
		num *= rf((1-float64(k)+delta12)/2.0, k)
		num *= rf((1-float64(k)+delta34)/2.0, k)
		num *= -powInt(4.0, k) * float64(k) * negOnePow(k)

		return num / denom
	}

	poles := []PolesData{}
	for k := 1; k <= rg.K1Max; k++ {
		ellVal := ell + k
		if ellVal > rg.EllMax || ellVal < rg.EllMin {
			continue
		}
		n := k
		delta := 1 - float64(ell) - float64(k)
		idx := rg.getOrAddPoleIdx(delta+float64(n), ellVal)
		poles = append(poles, PolesData{
			N:     n,
			Delta: delta,
			Ell:   ellVal,
			C:     c1(ell, k),
			Idx:   idx,
		})
	}
	return poles
}

// Type II (row 2 of Table 1)
func (rg *RecursiveG) getDelta2Poles(ell int, delta12, delta34 float64) []PolesData {
	c2 := func(ell, k int) float64 {
		denom := math.Pow(factorial(k), 2) *
			rf(float64(ell)+rg.Nu-float64(k), 2*k) *
			rf(float64(ell)+rg.Nu+1-float64(k), 2*k)

		num := rf(rg.Nu-float64(k), 2*k)
		num *= rf((1-float64(k)+float64(ell)-delta12+rg.Nu)/2.0, k)
		num *= rf((1-float64(k)+float64(ell)+delta12+rg.Nu)/2.0, k)
		num *= rf((1-float64(k)+float64(ell)-delta34+rg.Nu)/2.0, k)
		num *= rf((1-float64(k)+float64(ell)+delta34+rg.Nu)/2.0, k)
		num *= -powInt(4.0, 2*k) * float64(k) * negOnePow(k)

		return num / denom
	}

	poles := []PolesData{}
	for k := 1; k <= rg.K2Max; k++ {
		ellVal := ell
		if ellVal > rg.EllMax || ellVal < rg.EllMin {
			continue
		}
		n := 2 * k
		delta := 1 + rg.Nu - float64(k)
		idx := rg.getOrAddPoleIdx(delta+float64(n), ellVal)
		poles = append(poles, PolesData{
			N:     n,
			Delta: delta,
			Ell:   ellVal,
			C:     c2(ell, k),
			Idx:   idx,
		})
	}
	return poles
}

// Type III (row 3 of Table 1)
func (rg *RecursiveG) getDelta3Poles(ell int, delta12, delta34 float64) []PolesData {
	c3 := func(ell, k int) float64 {
		denom := math.Pow(factorial(k), 2) * rf(float64(ell)+rg.Nu+1-float64(k), k)

		num := rf(float64(ell)+1-float64(k), k)
		num *= rf((1-float64(k)+delta12)/2.0, k)
		num *= rf((1-float64(k)+delta34)/2.0, k)
		num *= -powInt(4.0, k) * float64(k) * negOnePow(k)

		return num / denom
	}

	poles := []PolesData{}
	for k := 1; k <= ell; k++ {
		ellVal := ell - k
		if ellVal > rg.EllMax || ellVal < rg.EllMin {
			continue
		}
		n := k
		delta := 1 + float64(ell) + 2*rg.Nu - float64(k)
		idx := rg.getOrAddPoleIdx(delta+float64(n), ellVal)
		poles = append(poles, PolesData{
			N:     n,
			Delta: delta,
			Ell:   ellVal,
			C:     c3(ell, k),
			Idx:   idx,
		})
	}
	return poles
}

// add or get index for pole key (deltaKey, ell)
// NOTE: In Python, they keyed by (delta+n, ell). We do the same to mirror indexing.
func (rg *RecursiveG) getOrAddPoleIdx(deltaKey float64, ell int) int {
	key := PoleKey{Delta: deltaKey, Ell: ell}
	if idx, ok := rg.unique_poles_map[key]; ok {
		return idx
	}
	idx := len(rg.unique_poles_map)
	rg.unique_poles_map[key] = idx
	rg.idxToKey = append(rg.idxToKey, key)
	return idx
}
