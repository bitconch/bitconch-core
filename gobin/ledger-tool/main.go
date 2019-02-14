package main

import (
	"fmt"
	"os"

	"github.com/bitconch/bus/gobin/utils"
	"gopkg.in/urfave/cli.v1"
)

var (
	// LedgerLocation is the path for ledger file
	LedgerLocation string
	// HeadNum

	gitCommit = ""
	// add new App with description
	app = utils.NewApp(gitCommit, "Bitconch chain genesis tool  CLI")
)

// Global flags to be used in the cli
/*
var (
	outfileFlag = cli.StringFlag{
		Name:        "outfile,o",
		Usage:       "Path to generated file",
		Destination: &OutFile,
	}
)
*/

//init define subcommand and flags linked to cli
func init() {

	// define the sub commands
	app.Commands = []cli.Command{
		printCommand,
		jsonCommand,
		verifyCommand,
	}

	// define the global flags
	app.Flags = []cli.Flag{
		//outfileFlag,
	}
}

func main() {
	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
