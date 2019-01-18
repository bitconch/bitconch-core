package main

/*
#cgo CFLAGS: -I ./include
#cgo LDFLAGS: -lhello
#cgo LDFLAGS: -L lib/x86_64-unknown-linux-gnu
#include <stdlib.h>
#include "hello.h"
*/
import "C"

func main() {
	C.hello(C.CString("Hello Dummy"))
	//C.hello_dummy()
}
