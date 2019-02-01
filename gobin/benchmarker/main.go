package main

import (
	"fmt"
	"os"

	"github.com/bitconch/bus/gobin/utils"
	"gopkg.in/urfave/cli.v1"
)

const (
	defaultKeyfileName = "keyfile.json"
)

var (
	// NetWorkEntryPoint represent the network entry point
	NetworkEntryPoint string
	// IdentityFile stores the keypair of users
	IdentityFile     string
	NodeThreshold    string
	RejectExtraNode  bool
	ThreadsNum       string
	DurationTime     string
	ConvergeOnly     bool
	SustainedMode    bool
	TransactionCount string
	// Git SHA1 commit hash of the release (set via linker flags)
	gitCommit = ""
	// add new App with description
	app = utils.NewApp(gitCommit, "Bitconch chain benchmark CLI")
)

// Flags to be used in the cli
var (
	networkFlag = cli.StringFlag{
		Name:        "network,n",
		Value:       "127.0.0.1:8001",
		Usage:       "Connect to the network entry point `HOST:PORT` ; defaults to 127.0.0.1:8001 ",
		Destination: &NetworkEntryPoint,
	}

	identityFlag = cli.StringFlag{
		Name:        "identity,i",
		Usage:       "Specify the user identity (keypair), which is stored in a json file",
		Destination: &IdentityFile,
	}

	nodeThresholdFlag = cli.StringFlag{
		Name:        "num-nodes,N",
		Usage:       "Minimum number of nodes in the network",
		Destination: &NodeThreshold,
	}

	rejectExtraNodeFlag = cli.BoolFlag{
		Name:        "reject-extra-node",
		Usage:       "Requires exact `num-nodes` of nodes to run, for dev only",
		Destination: &RejectExtraNode,
	}

	threadsFlag = cli.StringFlag{
		Name:        "threads,t",
		Usage:       "Number of threads during benchmarking",
		Destination: &ThreadsNum,
	}

	durationFlag = cli.StringFlag{
		Name:        "duration,t",
		Usage:       "Duration time for benchmarking, default is forever",
		Destination: &DurationTime,
	}

	convergeOnlyFlag = cli.BoolFlag{
		Name:        "converge-only",
		Usage:       "Exit immediately after converging",
		Destination: &ConvergeOnly,
	}

	sustainedFlag = cli.BoolFlag{
		Name:        "sustained",
		Usage:       "Use sustained performance mode vs. peak mode. This overlaps the tx generation with transfers.",
		Destination: &SustainedMode,
	}

	txCountFlag = cli.StringFlag{
		Name:        "tx_count",
		Usage:       "Number of transactions to send per batch",
		Destination: &TransactionCount,
	}
)

//init define subcommand and flags linked to cli
func init() {
	// clapcli is the action function
	app.Action = benchMarkerCli

	// define the sub commands
	app.Commands = []cli.Command{
		commandGenerate,
	}

	// define the flags
	app.Flags = []cli.Flag{
		networkFlag,
		identityFlag,
		nodeThresholdFlag,
		rejectExtraNodeFlag,
		threadsFlag,
		durationFlag,
		convergeOnlyFlag,
		sustainedFlag,
		txCountFlag,
	}
}

func main() {
	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// cli is the main entry point into the system if no special subcommand is ran.
func benchMarkerCli(ctx *cli.Context) error {
	if args := ctx.Args(); len(args) > 0 {
		return fmt.Errorf("invalid command: %q", args[0])
	}

	// handle the arguments

	if KeyFilePath == "" {
		fmt.Println("Default path:", username)
	} else {
		fmt.Println("New path:", KeyFilePath, " for：", username, " ")
	}

	// evoke the benchmarker, passing the parameters
	// bus.CalllBenchmarker()
	fmt.Println("Do some stuff")

	return nil

}
