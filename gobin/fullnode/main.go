// Call the main execute from vendor/rust_src_easy/main_execute.rs
package main

import (
	"fmt"
	"os"

	"github.com/bitconch/bus"
	"github.com/bitconch/bus/gobin/utils"
	"gopkg.in/urfave/cli.v1"

	"github.com/pkg/profile"
)

var (
	// IdentityFile stores users' keypair
	IdentityFile string
	// NetworkEntryPoint represent the network entry point
	NetworkEntryPoint string
	// LedgerLocation is the location of the ledger file
	LedgerLocation string

	gitCommit = ""
	// add new App with description
	app = utils.NewApp(gitCommit, "Bitconch chain fullnode CLI")
)

// Flags to be used in the cli
var (
	identityFlag = cli.StringFlag{
		Name:        "identity,i",
		Usage:       "ID file location",
		Destination: &IdentityFile,
	}

	networkFlag = cli.StringFlag{
		Name:        "network,n",
		Usage:       "Connect to the network entry point `HOST:PORT` ; defaults to 127.0.0.1:8001 ",
		Destination: &NetworkEntryPoint,
	}

	ledgerFlag = cli.StringFlag{
		Name:        "ledger,l",
		Usage:       "Ledger file location",
		Destination: &LedgerLocation,
	}
)

//init define subcommand and flags linked to cli
func init() {
	// clapcli is the action function
	app.Action = fullnodeCli

	// define the sub commands
	app.Commands = []cli.Command{
		//commandGenerate,
	}

	// define the flags
	app.Flags = []cli.Flag{
		identityFlag,
		networkFlag,
		ledgerFlag,
	}
}

func main() {
	defer profile.Start().Stop()
	//bus.CallFullnode()
	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

}

func fullnodeCli(ctx *cli.Context) error {
	if args := ctx.Args(); len(args) > 0 {
		return fmt.Errorf("invalid command: %q", args[0])
	}

	// handle the arguments
	fmt.Println("Do some stuff")
	// start the full node instance
	bus.CallFullnode(
		IdentityFile,
		NetworkEntryPoint,
		LedgerLocation,
	)
	return nil
}
