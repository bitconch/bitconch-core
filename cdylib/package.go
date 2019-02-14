package cdylib

/*
#cgo CFLAGS: -I ./include
#cgo LDFLAGS: -lhellolib
#cgo linux LDFLAGS: -L lib/x86_64-unknown-linux-gnu
#cgo windows LDFLAGS: -L lib/x86_64-pc-windows-msvc -l ws2_32 -l iphlpapi -l dbghelp -l userenv
#include <stdlib.h>
#include "hello.h"
*/
import "C"

type Argument struct {
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

func CallRustcodeHello(network *string,
						identity *string, 
						nodethreshold *string,
					) {

	//NetWorkEntryPoint := "128.0.0.1:8755"
	//IdentityFile := "use:r/bi.n/loc_al/mint.json"
	//C.hello(C.CString(NetWorkEntryPoint),
	//		C.CString(IdentityFile),
	//		)
	//C.hello_dummy()
	//C.rustcode_clap_cli(C.CString("some arguments"),)
	//C.rustcode_clap_cli(C.CString(""),)

	CNetwork := C.CString(*network)
	CIdentity := C.CString(*identity)
	CNodethreshold := C.CString(*nodethreshold)
	C.rustcode_clap_cli(CNetwork,CIdentity,CNodethreshold)
 
}

/*
//define a struct 
type S010Arg01 struct {
	inner *C.RusteloArguments
}

//create a new S010Arg01 struct
func P010NewArg01() (S010Arg01,error){
	var inner *C.S010Arg01
	res := C.rustelo_s010arg01_new(&inner)
	if res != 0 {
		return S010Arg01{}, rusteloLastError()
	}

	return S010Arg01{inner}, nil
}

func CallRustcodeHelloStruct(cliargument Argument) {

	//create a new struct from rust, return a point cliArgument

	//pass the point of the argument to rust code to handle &cliArgument
	

	C.rustcode_clap_cli_struct(cliArgument.inner)

}
*/