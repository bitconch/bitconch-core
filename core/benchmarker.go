package core
// #include <stdlib.h>
// #include "rustelo.h"
import "C"
import "unsafe"
import "errors"

func Benchmarker(){
	C.benchmarker_main_entry()
}
