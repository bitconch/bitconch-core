package bus


// #cgo CFLAGS: -I ./include
// #cgo LDFLAGS: -l rustelo
// #cgo darwin LDFLAGS: -L libs/x86_64-apple-darwin -framework Security -l c++
// #cgo linux LDFLAGS: -L libs/x86_64-unknown-linux-musl
// #cgo windows LDFLAGS: -L libs/x86_64-pc-windows-gnu -l ws2_32 -l iphlpapi -l dbghelp -l userenv
// #include <stdlib.h>
// #include "rustelo.h"
import "C"
import "unsafe"
import "errors"

//handleLastError returns an error from the rust side.
func handleLastError() error {

	err_str_bytes  := C.hedera_last_error()
	
	defer C.free(unsafe.Pointer(err_str_byes))
	err_str_string := C.GoString(err_str_byes)

	return errors.New(err_str_string)
}

