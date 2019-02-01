package bus

// #include <stdlib.h>
// #include "rustelo.h"
import "C"
import "unsafe"
import "errors"

func CallBenchmarker(network string, 
					identity string,
					num string,
					reject string,
					threads string,
					duration string,
					converge string,
					sustained string,
					txcount string){
	//convert string to CString
	CNetworkEntryPoint :=  C.CString(network)
	CIdentityFile :=  C.CString(identity)
	CNodeThreshold :=  C.CString(num)
	CRejectExtraNode :=  C.CString(reject)
	CThreadsNum :=  C.CString(threads)
	CDurationTime :=  C.CString(duration)
	CConvergeOnly :=  C.CString(converge)
	CSustainedMode :=  C.CString(sustained)					
	CTransactionCount :=  C.CString(txcount)
	
	

	C.benchmarker_main_entry(CNetworkEntryPoint,
							CIdentityFile,
							CNodeThreshold,
							CRejectExtraNode,
							CThreadsNum,
							CDurationTime,
							CConvergeOnly,
							CSustainedMode,
							CTransactionCount)
}
