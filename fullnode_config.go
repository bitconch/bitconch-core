package bus

// #include <stdlib.h>
// #include "rustelo.h"
import "C"

//CallFullnodeCongfig create the configuration for a full node
func CallFullnodeConfig(
	localmode string,
	keypair string,
	publicmode string,
	bindportnum string,
	outfilepath string,
) {

	//convert string to CString pointer
	CLocalMode := C.CString(localmode)
	CKeypairFile := C.CString(keypair)
	CPublicMode := C.CString(publicmode)
	CBindPortNum := C.CString(bindportnum)
	COutFilePath := C.CString(outfilepath)

	C.fullnode_config_main_entry(
		CLocalMode,
		CKeypairFile,
		CPublicMode,
		CBindPortNum,
		COutFilePath,
	)
}
