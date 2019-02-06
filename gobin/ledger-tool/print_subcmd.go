package main

import (
	"gopkg.in/urfave/cli.v1"
)

var (
	printCommand = cli.Command{
		Action:    printLedger,
		Name:      "print",
		Usage:     "Print the ledger into console output",
		ArgsUsage: "<filename> (<filename 2> ... <filename N>) ",
		Flags:     []cli.Flag{
			/*
				utils.DataDirFlag,
				utils.CacheFlag,
				utils.SyncModeFlag,
				utils.GCModeFlag,
				utils.CacheDatabaseFlag,
				utils.CacheGCFlag,
			*/
		},
		Category: "LEDGERTOOL COMMANDS",
		Description: `
The json command print out the ledger file into console.

`,
	}
)

func printLedger(ctx *cli.Context) error {
	return nil
}
