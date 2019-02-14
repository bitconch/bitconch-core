package main

import (
	"gopkg.in/urfave/cli.v1"
)

var (
	jsonCommand = cli.Command{
		Action:    jsonLedger,
		Name:      "json",
		Usage:     "Print the ledger in json format",
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
The json command print out the ledger file, which contains all the transactions that ever
been confirmed, into a json file.

`,
	}
)

func jsonLedger(ctx *cli.Context) error {
	return nil
}
