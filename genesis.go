package bus

// #include <stdlib.h>
// #include "rustelo.h"
import "C"

//CallFullnode create an instance of a fullnode for bitconch blockchain
func CallFullnode(
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
