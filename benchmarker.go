package bus

// #include <stdlib.h>
// #include "rustelo.h"
import "C"

/* to do: pass a struct to rust ffi
type BenchmarkerArgument struct {
	NetWorkEntryPoint string
	IdentityFile      string
	NodeThreshold     string
	RejectExtraNode   bool
	ThreadsNum        string
	DurationTime      string
	ConvergeOnly      bool
	SustainedMode     bool
	TransactionCount  string
}
*/

func CallBenchmarker(network string,
	identity string,
	num string,
	reject string,
	threads string,
	duration string,
	converge string,
	sustained string,
	txcount string) {

	//convert string to CString, CString return a pointer
	CPtr1NetworkEntryPoint := C.CString(network)
	CPtr2IdentityFile := C.CString(identity)
	CPtr3NodeThreshold := C.CString(num)
	CPtr4RejectExtraNode := C.CString(reject)
	CPtr5ThreadsNum := C.CString(threads)
	CPtr6DurationTime := C.CString(duration)
	CPtr7ConvergeOnly := C.CString(converge)
	CPtr8SustainedMode := C.CString(sustained)
	CPtr9TransactionCount := C.CString(txcount)

	//call the main entry of benchmarker utility
	C.benchmarker_main_entry(CPtr1NetworkEntryPoint,
		CPtr2IdentityFile,
		CPtr3NodeThreshold,
		CPtr4RejectExtraNode,
		CPtr5ThreadsNum,
		CPtr6DurationTime,
		CPtr7ConvergeOnly,
		CPtr8SustainedMode,
		CPtr9TransactionCount)
}
