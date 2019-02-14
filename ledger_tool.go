package bus

// #include <stdlib.h>
// #include "rustelo.h"
import "C"

//CallLedgerTool
func CallLedgerTool(
	outfile string,
) {

	//convert string to CString pointer
	COutFile := C.CString(outfile)

	C.keygen_main_entry(
		COutFile,
	)
}
