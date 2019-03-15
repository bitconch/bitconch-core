package bus

// #include <stdlib.h>
// #include "rustelo.h"
import "C"

//CallFullnode create an instance of a fullnode for bitconch blockchain
func CallFullnode(
	identity string,
	network string,
	ledger string) {

	//convert string to CString pointer
	CIdentityFile := C.CString(identity)
	CNetworkEntryPoint := C.CString(network)
	CLedgerLocation := C.CString(ledger)

	C.fullnode_main_entry(
		CIdentityFile,
		CNetworkEntryPoint,
		CLedgerLocation,
	)
}
