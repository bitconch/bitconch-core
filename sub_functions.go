package rustelo 

// #include <stdlib.h>
// #include "rustelo.h"
import "C"
import "unsafe"

// rustelo.h is a must to include. 

//AsimpleStruct is a struct defined in 
type ASimpleStruct struct {
	inner C.HederaSecretKey
}
