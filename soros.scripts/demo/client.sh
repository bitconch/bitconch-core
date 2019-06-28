#!/usr/bin/env bash
set -e

# Set the PATH 
PATH=/usr/bin/bitconch/bin:$PATH

here=$(dirname "$0")
# shellcheck source=demo/common.sh
source "$here"/common.sh

usage() {
  if [[ -n $1 ]]; then
    echo "$*"
    echo
  fi
  echo "usage: $0 [extra args]"
  echo
  echo " Run bench-tps "
  echo
  echo "   extra args: additional arguments are pass along to buffett-bench-tps"
  echo
  exit 1
}

if [[ -z $1 ]]; then # default behavior
  soros-bench-tps \
    --network 127.0.0.1:10001 \
    --drone 127.0.0.1:11100 \
    --duration 90 \
    --tx_count 50000 \

else
  soros-bench-tps "$@"
fi

