package main

import (
	"fmt"
	"os"

	"github.com/bitconch/bus"
	"github.com/bitconch/bus/gobin/utils"
	"gopkg.in/urfave/cli.v1"
)

var (
	// OutFile is the location for genereated keypair file
	OutFile string

	gitCommit = ""
	// add new App with description
	app = utils.NewApp(gitCommit, "Bitconch chain genesis tool  CLI")
)

// Flags to be used in the cli
var (
	outfileFlag = cli.StringFlag{
		Name:        "outfile,o",
		Usage:       "Path to generated file",
		Destination: &OutFile,
	}
)

//init define subcommand and flags linked to cli
func init() {
	// keymakerCli is the action function
	app.Action = keymakerCli

	// define the sub commands
	app.Commands = []cli.Command{
		//commandGenerate,
	}

	// define the flags
	app.Flags = []cli.Flag{
		outfileFlag,
	}
}

func main() {

	//bus.CallKeymaker()
	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

}

func keymakerCli(ctx *cli.Context) error {
	if args := ctx.Args(); len(args) > 0 {
		return fmt.Errorf("invalid command: %q", args[0])
	}

	// handle the arguments
	fmt.Println("Do some stuff")
	// start the full node instance
	bus.CallKeymaker(OutFile)

	return nil
}
