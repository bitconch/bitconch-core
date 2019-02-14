package bus

// #include <stdlib.h>
// #include "rustelo.h"
import "C"

//CallGenesis start the initialization process of bitconch chain
func CallGenesis(
	tokens string,
	ledger string,
) {

	//convert string to CString pointer
	CTokens := C.CString(tokens)
	CLedger := C.CString(ledger)

	C.genesis_main_entry(
		CTokens,
		CLedger,
	)
}
