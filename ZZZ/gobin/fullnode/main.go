// Call the main execute from vendor/rust_src_easy/main_execute.rs
package main

import (
	"fmt"
	"github.com/caesarchad/rustelo"
)


func main() {

	//execute the main_entry func in rustelo
	rustelo.FullNodeMainEntry()
	
}