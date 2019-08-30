#!/usr/bin/env bash
#
# Starts an instance of morgan-drone
#
here=$(dirname "$0")

# shellcheck source=multinode-demo/common.sh
source "$here"/common.sh

[[ -f "$MORGAN_CONFIG_DIR"/mint-keypair.json ]] || {
  echo "$MORGAN_CONFIG_DIR/mint-keypair.json not found, create it by running:"
  echo
  echo "  ${here}/setup.sh"
  exit 1
}

set -x
# shellcheck disable=SC2086 # Don't want to double quote $morgan_drone
exec $morgan_drone --keypair "$MORGAN_CONFIG_DIR"/mint-keypair.json "$@"
