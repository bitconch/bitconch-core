package main

/*
#cgo CFLAGS: -I ./include
#cgo LDFLAGS: -lhellolib
#cgo linux LDFLAGS: -L lib
#cgo windows LDFLAGS: -L lib/x86_64-pc-windows-msvc -l ws2_32 -l iphlpapi -l dbghelp -l userenv
#include <stdlib.h>
#include "hello.h"
*/
import "C"

func main() {
	C.hello(C.CString("Hello Dummy"))
	//C.hello_dummy()
	C.rustcode_clap_cli()
}