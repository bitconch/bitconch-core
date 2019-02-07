package main

import (
	"fmt"
	"os"

	"github.com/bitconch/bus/cdylib"
	"github.com/bitconch/bus/gobin/utils"
	"gopkg.in/urfave/cli.v1"
)

const (
	defaultKeyfileName = "keyfile.json"
)

type Argument struct {
}

var (
	// NetWorkEntryPoint represent the network entry point
	NetWorkEntryPoint string
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
		Usage:       "Connect to the network entry point `HOST:PORT` ; defaults to 127.0.0.1:8001 ",
		Destination: &NetWorkEntryPoint,
	}

	identityFlag = cli.StringFlag{
		Name:        "identity,i",
		Usage:       "Specify the user identity file location `PATH`",
		Destination: &IdentityFile,
	}

	nodeThresholdFlag = cli.StringFlag{
		Name:        "num-nodes,N",
		Usage:       "Specify the minimum number of nodes `NUM` for the network to work properly",
		Destination: &NodeThreshold,
	}

	rejectExtraNodeFlag = cli.BoolFlag{
		Name:        "reject-extra-node",
		Usage:       "Requires exact number nodes to run as specified by num-ndes, for dev only",
		Destination: &RejectExtraNode,
	}

	threadsFlag = cli.StringFlag{
		Name:        "threads,t",
		Usage:       "Specify the number of threads `NUM` during benchmarking",
		Destination: &ThreadsNum,
	}

	durationFlag = cli.StringFlag{
		Name:        "duration",
		Usage:       "Duration time `SEC` for benchmarking, default is forever",
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
		Usage:       "Number of transactions `NUM` to send per batch",
		Destination: &TransactionCount,
	}
)

//init define subcommand and flags linked to cli
func init() {

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

	app.Before = func(c *cli.Context) error {
		fmt.Fprintf(c.App.Writer, "HEEEERE GOES\n")
		return nil
	}
	app.After = func(c *cli.Context) error {
		fmt.Fprintf(c.App.Writer, "Phew!\n")
		return nil
	}
	app.CommandNotFound = commandNotFind

	app.OnUsageError = func(c *cli.Context, err error, isSubcommand bool) error {
		if isSubcommand {
			return err
		}

		fmt.Fprintf(c.App.Writer, "WRONG: %#v\n", err)
		return nil
	}
	// clapcli is the action function
	app.Action = benchMarkerCli

}

func main() {

	/*
		sort.Sort(cli.FlagsByName(app.Flags))
		sort.Sort(cli.CommandsByName(app.Commands))
	*/

	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// command not find action
func commandNotFind(c *cli.Context, command string) {
	//fmt.Fprintf(c.App.Writer, "Thar be no %q here.\n", command)
	fmt.Println("Invalid command: %q here.\n", command)
}

// cli is the main entry point into the system if no special subcommand is ran.
func benchMarkerCli(ctx *cli.Context) error {

	/*
		if args := ctx.Args(); len(args) > 0 {
			return fmt.Errorf("invalid command: %q", args[0])
		}
	*/
	fmt.Println("--- print flag names ---")
	fmt.Println(ctx.FlagNames())
	fmt.Println("--- print global flag names ---")
	fmt.Println(ctx.GlobalFlagNames())
	fmt.Println("--- print number of arguments ---")
	fmt.Println(ctx.NArg())
	fmt.Println("--- print arguments ---")
	fmt.Println(ctx.Args())
	fmt.Println("--- print the 0th argument ---")
	fmt.Println(ctx.Args().Get(0))
	fmt.Println("--- print number of flags ---")
	fmt.Println(ctx.NumFlags())
	fmt.Println("--- print number of subcomand ---")
	fmt.Println(ctx.NArg())
	fmt.Println("--- print the parent context ---")
	fmt.Println(ctx.Parent())

	cdylib.CallRustcodeHello(NetWorkEntryPoint, IdentityFile, NodeThreshold)

	return nil

}
