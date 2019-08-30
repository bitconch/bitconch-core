#!/usr/bin/env bash
#
# Clear the current cluster configuration
#

here=$(dirname "$0")
# shellcheck source=multinode-demo/common.sh
source "$here"/common.sh

set -e

for i in "$MORGAN_RSYNC_CONFIG_DIR" "$MORGAN_CONFIG_DIR"; do
  echo "Cleaning $i"
  rm -rvf "$i"
  mkdir -p "$i"
done

