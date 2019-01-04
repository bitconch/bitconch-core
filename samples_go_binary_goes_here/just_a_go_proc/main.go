package main

import (
	"fmt"
	"os"
	"time"

	//the go bases package
	"github.com/caesarchad/rustelo"
)

func main() {
	//
	// Do some staff
	//

	// Call a function which is defined in the imported package, namely "rustelo"
	operatorSecret, err := rustelo.ASimpleGoFunction(os.Getenv("USERNAME"))
	if err != nil {
		panic(err)
	}

	// Call yet another function in the package
	secret, _ := rustelo.AnotherSimpleFunction()
	public := secret.Public()

	fmt.Printf("secret = %v\n", secret)
	fmt.Printf("public = %v\n", public)

	
}
