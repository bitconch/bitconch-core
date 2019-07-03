#!/usr/bin/env bash
#
# Clear the current cluster configuration
#

# Set the PATH 
PATH=/usr/bin/bitconch/bin:$PATH

here=$(dirname "$0")
# shellcheck source=multinode-demo/common.sh
source "$here"/common.sh

set -e

for i in "$SOROS_RSYNC_CONFIG_DIR" "$SOROS_CONFIG_DIR"; do
  echo "Cleaning $i"
  rm -rvf "$i"
  mkdir -p "$i"
done


