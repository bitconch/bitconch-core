#!/usr/bin/env bash
#
# Starts an instance of solana-drone
#
set -ex

here=$(dirname "$0")


# Set the PATH 
PATH=/usr/bin/bitconch/bin:$PATH

# shellcheck source=multinode-demo/common.sh
source "$here"/common.sh

[[ -f "$SOROS_CONFIG_DIR"/mint-id.json ]] || {
  echo "$SOROS_CONFIG_DIR/mint-id.json not found, create it by running:"
  echo
  echo "  ${here}/setup.sh"
  exit 1
}



trap 'kill "$pid" && wait "$pid"' INT TERM ERR
soros-drone \
  --keypair "$SOROS_CONFIG_DIR"/mint-id.json \
  "$@" \
  > >($drone_logger) 2>&1 &
pid=$!
wait "$pid"

