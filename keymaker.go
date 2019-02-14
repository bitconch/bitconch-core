package bus

// #include <stdlib.h>
// #include "rustelo.h"
import "C"

//CallKeymaker create a stack of keypairs into a file
func CallKeymaker(outfile string) {

	//convert string to CString pointer
	COutFile := C.CString(outfile)

	C.keygen_main_entry(COutFile)
}
