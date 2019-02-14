package bus


// #cgo CFLAGS: -I ./include
// #cgo LDFLAGS: -l rustelo
// #cgo linux LDFLAGS: -L libs/x86_64-unknown-linux-gnu
// #include <stdlib.h>
// #include "rustelo.h"
import "C"
import "unsafe"
import "errors"

//handleLastError returns an error from the rust side.
func handleLastError() error {

	err_str_bytes  := C.rustelo_handle_error()
	
	defer C.free(unsafe.Pointer(err_str_bytes))
	err_str_string := C.GoString(err_str_bytes)

	return errors.New(err_str_string)
}


