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



func CallRustcodeHello(network string,
						identity string, 
						nodethreshold string,
					) {

	NetWorkEntryPoint := "128.0.0.1:8755"
	IdentityFile := "use:r/bi.n/loc_al/mint.json"
	C.hello(C.CString(NetWorkEntryPoint),
			C.CString(IdentityFile),
			)
	//C.hello_dummy()
	//C.rustcode_clap_cli(C.CString("some arguments"),)
	//C.rustcode_clap_cli(C.CString(""),)

	CNetwork := C.CString(network)
	CIdentity := C.CString(identity)
	CNodethreshold := C.CString(nodethreshold)
	C.rustcode_clap_cli(CNetwork,CIdentity,CNodethreshold)
}