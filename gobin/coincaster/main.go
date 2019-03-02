package main

import (
	"fmt"
	"os"

	"github.com/bitconch/bus"
	"github.com/bitconch/bus/gobin/utils"
	"gopkg.in/urfave/cli.v1"

	"github.com/pkg/profile"
)

const (
	defaultKeyfileName = "keyfile.json"
)

var (
	// NetworkEntryPoint represent the network entry point
	NetworkEntryPoint string
	// KeypairFile stores the coincaster's keypair
	KeypairFile string
	// SliceTime is casting time for coincast spell
	SliceTime string
	// CapTIme is the casting requests' upper limit
	ReqCapNum string
	// Git SHA1 commit hash of the release (set via linker flags)
	gitCommit = ""
	// add new App with description
	app = utils.NewApp(gitCommit, "Bitconch chain benchmark CLI")
)

// Flags to be used in the cli
var (
	networkFlag = cli.StringFlag{
		Name:        "network,n",
		Usage:       "Connect to the network entry point `HOST:PORT` ; defaults to 127.0.0.1:8001 ",
		Destination: &NetworkEntryPoint,
	}

	keypairFlag = cli.StringFlag{
		Name:        "keypair,k",
		Usage:       "Stores the coincaster's keypair in `PATH`",
		Destination: &KeypairFile,
	}

	sliceTimeFlag = cli.StringFlag{
		Name:        "slice",
		Usage:       "Time interval during which the `SECS`",
		Destination: &SliceTime,
	}

	reqCapNumFlag = cli.StringFlag{
		Name:        "cap",
		Usage:       "Request cap number per time slice",
		Destination: &ReqCapNum,
	}
)

//init define subcommand and flags linked to cli
func init() {
	// clapcli is the action function
	app.Action = coinCasterCli

	// define the sub commands
	app.Commands = []cli.Command{
		//commandGenerate,
	}

	// define the flags
	app.Flags = []cli.Flag{
		networkFlag,
		keypairFlag,
		sliceTimeFlag,
		reqCapNumFlag,
	}
}

func main() {
	defer profile.Start().Stop()
	
	//bus.CallCoinCaster()
	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

}

// cli is the main entry point into the system if no special subcommand is ran.
func coinCasterCli(ctx *cli.Context) error {
	if args := ctx.Args(); len(args) > 0 {
		return fmt.Errorf("invalid command: %q", args[0])
	}

	// handle the arguments

	// evoke the benchmarker, passing the parameters
	// bus.CalllBenchmarker()
	fmt.Println("Do some stuff")
	bus.CallCoincaster(
		NetworkEntryPoint,
		KeypairFile,
		SliceTime,
		ReqCapNum,
	)
	return nil

}
