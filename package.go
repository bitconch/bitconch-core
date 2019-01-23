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


func CalllBenchmarker(){
	C.benchmarker_main_entry()
}

func CallCoinCaster(){
	C.coincaster_main_entry()

}

func CallFullnode(){
	C.fullnode_main_entry()
}

func CallFullnodeConfig(){
	C.fullnode_config_main_entry()

}
/*
func CallFullnodeCuda(){
	
}
*/

func CallGenesis(){
	C.genesis_main_entry()
}

func CallKeymaker(){
	C.keygen_main_entry()
}

func CallLedgerTool(){
	C.ledgertool_main_entry()
}

func CallWallet(){
	C.wallet_main_entry()
}
