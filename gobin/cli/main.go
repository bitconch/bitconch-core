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
	// KeyFilePath store the output file path from the argument
	KeyFilePath string
	// Git SHA1 commit hash of the release (set via linker flags)
	gitCommit = ""

	app = utils.NewApp(gitCommit, "the go-ethereum command line interface")
)

// Flags
// Commonly used command line flags.
var (
	OutfileFlag = cli.StringFlag{
		Name:        "outfile,o",
		Value:       "USERNAME/DEFAULT_PATH/DEFAULT_SUBPATH",
		Usage:       "path to generate a key file",
		Destination: &KeyFilePath,
	}

	passphraseFlag = cli.StringFlag{
		Name:  "passwordfile",
		Usage: "the file that contains the passphrase for the keyfile",
	}
	jsonFlag = cli.BoolFlag{
		Name:  "json",
		Usage: "output JSON instead of human-readable format",
	}
)

func init() {
	// clapcli is the action function
	app.Action = clapcli

	// define the sub commands
	app.Commands = []cli.Command{
		commandGenerate,
	}

	// define the flags
	app.Flags = []cli.Flag{
		OutfileFlag,
		passphraseFlag,
		jsonFlag,
	}
}

func main() {
	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// cli is the main entry point into the system if no special subcommand is ran.
func clapcli(ctx *cli.Context) error {
	if args := ctx.Args(); len(args) > 0 {
		return fmt.Errorf("invalid command: %q", args[0])
	}
	username := "Dummy"
	if ctx.NArg() > 0 {
		username = ctx.Args()[0]
	}
	if KeyFilePath == "" {
		fmt.Println("Default path:", username)
	} else {
		fmt.Println("New path:", KeyFilePath, " for：", username, " ")
	}
	return nil
}
