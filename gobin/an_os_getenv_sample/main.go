package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Printf("User %s's Gopath is %s.\n Goroot is %s.\n", os.Getenv("USERNAME"),os.Getenv("GOPATH"), os.Getenv("GOROOT"))

}
