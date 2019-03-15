package main

import (
	"gopkg.in/urfave/cli.v1"
)

var (
	verifyCommand = cli.Command{
		Action:    verifyLedger,
		Name:      "verify",
		Usage:     "Verify the ledger's timestamp",
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
The verify command will try to verify the ledger's timestamp.

`,
	}
)

func verifyLedger(ctx *cli.Context) error {
	return nil
}
