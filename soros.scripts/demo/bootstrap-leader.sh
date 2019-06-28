#!/usr/bin/env bash
#
# Start the bootstrap leader node
# All binaries will be installed on /usr/bin/bitconch/bin and /usr/bin/bitconch/bin/deps

# Set the PATH 
PATH=/usr/bin/bitconch/bin:$PATH

here=$(dirname "$0")
# shellcheck source=multinode-demo/common.sh
source "$here"/common.sh

# shellcheck source=scripts/oom-score-adj.sh
source "$here"/../scripts/oom-score-adj.sh

if [[ $1 = -h ]]; then
  fullnode_usage "$@"
fi

# shellcheck source=multinode-demo/extra-fullnode-args.sh
source "$here"/extra-fullnode-args.sh


[[ -f "$SOROS_CONFIG_DIR"/bootstrap-leader-id.json ]] || {
  echo "$SOROS_CONFIG_DIR/bootstrap-leader-id.json not found, create it by running:"
  echo
  echo "  ${here}/setup.sh"
  exit 1
}

if [[ -n "$SOROS_CUDA" ]]; then
  program="soros-fullnode-cuda"
else
  program="soros-fullnode"
fi

tune_system

soros-legder-tool --ledger "$SOROS_CONFIG_DIR"/bootstrap-leader-ledger verify

bootstrap_leader_id_path="$SOROS_CONFIG_DIR"/bootstrap-leader-id.json
bootstrap_leader_vote_id_path="$SOROS_CONFIG_DIR"/bootstrap-leader-vote-id.json
bootstrap_leader_vote_id=$($soros_keygen pubkey "$bootstrap_leader_vote_id_path")

trap 'kill "$pid" && wait "$pid"' INT TERM ERR

default_fullnode_arg --identity "$bootstrap_leader_id_path"
default_fullnode_arg --voting-keypair "$bootstrap_leader_vote_id_path"
default_fullnode_arg --vote-account  "$bootstrap_leader_vote_id"
default_fullnode_arg --ledger "$SOROS_CONFIG_DIR"/bootstrap-leader-ledger
default_fullnode_arg --accounts "$SOROS_CONFIG_DIR"/bootstrap-leader-accounts
default_fullnode_arg --rpc-port 10099
default_fullnode_arg --rpc-drone-address 127.0.0.1:11100
default_fullnode_arg --gossip-port 10001
default_fullnode_arg --blockstream /tmp/bitconch-blockstream.sock # Default to location used by the block explorer
echo "$PS4 $program ${extra_fullnode_args[*]}"
$program "${extra_fullnode_args[@]}" > >($bootstrap_leader_logger) 2>&1 &
pid=$!
oom_score_adj "$pid" 1000

wait "$pid"
