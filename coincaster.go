package bus

// #include <stdlib.h>
// #include "rustelo.h"
import "C"

//CallCoincaster will fly and do some airdrops
func CallCoincaster(network string,
	keypair string,
	slicetime string,
	reqcapnum string) {

	//convert string to CString
	CNetworkEntryPoint := C.CString(network)
	CKeypairFile := C.CString(keypair)
	CSliceTime := C.CString(slicetime)
	CReqCapNum := C.CString(reqcapnum)

	C.coincaster_main_entry(CNetworkEntryPoint,
		CKeypairFile,
		CSliceTime,
		CReqCapNum)
}
