#!/usr/bin/env bash

here=$(dirname "$0")
# shellcheck source=multinode-demo/common.sh
source "$here"/common.sh

set -e
"$here"/clear-config.sh

# Create genesis ledger
$morgan_keygen -o "$MORGAN_CONFIG_DIR"/mint-keypair.json
$morgan_keygen -o "$MORGAN_CONFIG_DIR"/bootstrap-leader-keypair.json
$morgan_keygen -o "$MORGAN_CONFIG_DIR"/bootstrap-leader-vote-keypair.json
$morgan_keygen -o "$MORGAN_CONFIG_DIR"/bootstrap-leader-stake-keypair.json
$morgan_keygen -o "$MORGAN_CONFIG_DIR"/bootstrap-leader-storage-keypair.json

args=("$@")
default_arg --bootstrap-leader-keypair "$MORGAN_CONFIG_DIR"/bootstrap-leader-keypair.json
default_arg --bootstrap-vote-keypair "$MORGAN_CONFIG_DIR"/bootstrap-leader-vote-keypair.json
default_arg --bootstrap-stake-keypair "$MORGAN_CONFIG_DIR"/bootstrap-leader-stake-keypair.json
default_arg --bootstrap-storage-keypair "$MORGAN_CONFIG_DIR"/bootstrap-leader-storage-keypair.json
default_arg --ledger "$MORGAN_RSYNC_CONFIG_DIR"/ledger
default_arg --mint "$MORGAN_CONFIG_DIR"/mint-keypair.json
default_arg --difs 100000000000000
default_arg --hashes-per-tick sleep

$morgan_genesis "${args[@]}"

test -d "$MORGAN_RSYNC_CONFIG_DIR"/ledger
cp -a "$MORGAN_RSYNC_CONFIG_DIR"/ledger "$MORGAN_CONFIG_DIR"/bootstrap-leader-ledger
