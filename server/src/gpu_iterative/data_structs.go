package iterativegpu

// TODO: Unify these with the ones defined in the parent package
// Go-side helper types (only include the fields we need; N and Ell included for binary compatibility)
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
