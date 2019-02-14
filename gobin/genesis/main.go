package main

import (
	"fmt"
	"os"

	"github.com/bitconch/bus"
	"github.com/bitconch/bus/gobin/utils"
	"gopkg.in/urfave/cli.v1"
)

var (
	// TokenAmt is the amount of token issued at the initialization process
	TokenAmt string
	// LedgerLocation is the PATH for ledger file which stores all the transactions
	LedgerLocation string

	gitCommit = ""
	// add new App with description
	app = utils.NewApp(gitCommit, "Bitconch chain genesis tool  CLI")
)

// Flags to be used in the cli
var (
	tokenFlag = cli.StringFlag{
		Name:        "tokens,t",
		Usage:       "Number of tokens with which to initialize mint",
		Destination: &TokenAmt,
	}

	ledgerFlag = cli.StringFlag{
		Name:        "ledger,l",
		Usage:       "Use directory as persistent ledger location",
		Destination: &LedgerLocation,
	}
)

//init define subcommand and flags linked to cli
func init() {
	// genesisCli is the action function
	app.Action = genesisCli

	// define the sub commands
	app.Commands = []cli.Command{
		//commandGenerate,
	}

	// define the flags
	app.Flags = []cli.Flag{
		tokenFlag,
		ledgerFlag,
	}
}

func main() {

	//bus.CallGenesis()
	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

}

func genesisCli(ctx *cli.Context) error {
	if args := ctx.Args(); len(args) > 0 {
		return fmt.Errorf("invalid command: %q", args[0])
	}

	// handle the arguments
	fmt.Println("Do some stuff")
	// start the full node instance
	bus.CallGenesis(
		TokenAmt,
		LedgerLocation,
	)

	return nil
}

/*
func main() {

	bus.CallGenesis(
		&TokenAmt,
		&LedgerLocation,
	)

}
*/
