#!/usr/bin/env bash
#
# Start a fullnode
#
here=$(dirname "$0")
# shellcheck source=multinode-demo/common.sh
source "$here"/common.sh

# shellcheck source=scripts/oom-score-adj.sh
source "$here"/../scripts/oom-score-adj.sh

fullnode_usage() {
  if [[ -n $1 ]]; then
    echo "$*"
    echo
  fi
  cat <<EOF

Fullnode Usage:
usage: $0 [--blockstream PATH] [--init-complete-file FILE] [--label LABEL] [--stake DIFS] [--no-voting] [--rpc-port port] [rsync network path to bootstrap leader configuration] [cluster entry point]

Start a validator or a replicator

  --blockstream PATH        - open blockstream at this unix domain socket location
  --init-complete-file FILE - create this file, if it doesn't already exist, once node initialization is complete
  --label LABEL             - Append the given label to the configuration files, useful when running
                              multiple fullnodes in the same workspace
  --stake DIFS          - Number of difs to stake
  --no-voting               - start node without vote signer
  --rpc-port port           - custom RPC port for this node
  --no-restart              - do not restart the node if it exits

EOF
  exit 1
}

find_entrypoint() {
  declare entrypoint entrypoint_address
  declare shift=0

  if [[ -z $1 ]]; then
    entrypoint="$MORGAN_ROOT"         # Default to local tree for rsync
    entrypoint_address=127.0.0.1:10001 # Default to local entrypoint
  elif [[ -z $2 ]]; then
    entrypoint=$1
    entrypoint_address=$entrypoint:10001
    shift=1
  else
    entrypoint=$1
    entrypoint_address=$2
    shift=2
  fi

  echo "$entrypoint" "$entrypoint_address" "$shift"
}

rsync_url() { # adds the 'rsync://` prefix to URLs that need it
  declare url="$1"

  if [[ $url =~ ^.*:.*$ ]]; then
    # assume remote-shell transport when colon is present, use $url unmodified
    echo "$url"
    return 0
  fi

  if [[ -d $url ]]; then
    # assume local directory if $url is a valid directory, use $url unmodified
    echo "$url"
    return 0
  fi

  # Default to rsync:// URL
  echo "rsync://$url"
}

setup_validator_accounts() {
  declare entrypoint_ip=$1
  declare node_keypair_path=$2
  declare vote_keypair_path=$3
  declare stake_keypair_path=$4
  declare storage_keypair_path=$5
  declare stake=$6

  declare node_pubkey
  node_pubkey=$($morgan_keygen pubkey "$node_keypair_path")

  declare vote_pubkey
  vote_pubkey=$($morgan_keygen pubkey "$vote_keypair_path")

  declare stake_pubkey
  stake_pubkey=$($morgan_keygen pubkey "$stake_keypair_path")

  declare storage_pubkey
  storage_pubkey=$($morgan_keygen pubkey "$storage_keypair_path")

  if [[ -f $configured_flag ]]; then
    echo "Vote and stake accounts have already been configured"
  else
    # Fund the node with enough tokens to fund its Vote, Staking, and Storage accounts
    $morgan_wallet --keypair "$node_keypair_path" --url "http://$entrypoint_ip:10099" airdrop $((stake*2+2)) || return $?

    # Fund the vote account from the node, with the node as the node_pubkey
    $morgan_wallet --keypair "$node_keypair_path" --url "http://$entrypoint_ip:10099" \
      create-vote-account "$vote_pubkey" "$node_pubkey" "$stake" || return $?

    # Fund the stake account from the node, with the node as the node_pubkey
    $morgan_wallet --keypair "$node_keypair_path" --url "http://$entrypoint_ip:10099" \
      create-stake-account "$stake_pubkey" "$stake" || return $?

    # Delegate the stake.  The transaction fee is paid by the node but the
    #  transaction must be signed by the stake_keypair
    $morgan_wallet --keypair "$node_keypair_path" --url "http://$entrypoint_ip:10099" \
      delegate-stake "$stake_keypair_path" "$vote_pubkey" || return $?

    # Setup validator storage account
    $morgan_wallet --keypair "$node_keypair_path" --url "http://$entrypoint_ip:10099" \
      create-validator-storage-account "$storage_pubkey" || return $?

    touch "$configured_flag"
  fi

  $morgan_wallet --keypair "$node_keypair_path" --url "http://$entrypoint_ip:10099" \
    show-vote-account "$vote_pubkey"
  $morgan_wallet --keypair "$node_keypair_path" --url "http://$entrypoint_ip:10099" \
    show-stake-account "$stake_pubkey"
  $morgan_wallet --keypair "$node_keypair_path" --url "http://$entrypoint_ip:10099" \
    show-storage-account "$storage_pubkey"

  return 0
}

setup_replicator_account() {
  declare entrypoint_ip=$1
  declare node_keypair_path=$2
  declare storage_keypair_path=$3
  declare stake=$4

  declare storage_pubkey
  storage_pubkey=$($morgan_keygen pubkey "$storage_keypair_path")

  if [[ -f $configured_flag ]]; then
    echo "Replicator account has already been configured"
  else
    $morgan_wallet --keypair "$node_keypair_path" --url "http://$entrypoint_ip:10099" airdrop "$stake" || return $?

    # Setup replicator storage account
    $morgan_wallet --keypair "$node_keypair_path" --url "http://$entrypoint_ip:10099" \
      create-replicator-storage-account "$storage_pubkey" || return $?

    touch "$configured_flag"
  fi

  $morgan_wallet --keypair "$node_keypair_path" --url "http://$entrypoint_ip:10099" \
    show-storage-account "$storage_pubkey"

  return 0
}

ledger_not_setup() {
  echo "Error: $*"
  echo
  echo "Please run: ${here}/setup.sh"
  exit 1
}

args=()
node_type=validator
stake=42 # number of difs to assign as stake
poll_for_new_genesis_block=0
label=
identity_keypair_path=
no_restart=0

positional_args=()
while [[ -n $1 ]]; do
  if [[ ${1:0:1} = - ]]; then
    if [[ $1 = --label ]]; then
      label="-$2"
      shift 2
    elif [[ $1 = --no-restart ]]; then
      no_restart=1
      shift
    elif [[ $1 = --bootstrap-leader ]]; then
      node_type=bootstrap_leader
      shift
    elif [[ $1 = --replicator ]]; then
      node_type=replicator
      shift
    elif [[ $1 = --validator ]]; then
      node_type=validator
      shift
    elif [[ $1 = --poll-for-new-genesis-block ]]; then
      poll_for_new_genesis_block=1
      shift
    elif [[ $1 = --blockstream ]]; then
      stake=0
      args+=("$1" "$2")
      shift 2
    elif [[ $1 = --identity ]]; then
      identity_keypair_path=$2
      args+=("$1" "$2")
      shift 2
    elif [[ $1 = --enable-rpc-exit ]]; then
      args+=("$1")
      shift
    elif [[ $1 = --init-complete-file ]]; then
      args+=("$1" "$2")
      shift 2
    elif [[ $1 = --stake ]]; then
      stake="$2"
      shift 2
    elif [[ $1 = --no-voting ]]; then
      args+=("$1")
      shift
    elif [[ $1 = --no-sigverify ]]; then
      args+=("$1")
      shift
    elif [[ $1 = --rpc-port ]]; then
      args+=("$1" "$2")
      shift 2
    elif [[ $1 = --dynamic-port-range ]]; then
      args+=("$1" "$2")
      shift 2
    elif [[ $1 = --gossip-port ]]; then
      args+=("$1" "$2")
      shift 2
    elif [[ $1 = -h ]]; then
      fullnode_usage "$@"
    else
      echo "Unknown argument: $1"
      exit 1
    fi
  else
    positional_args+=("$1")
    shift
  fi
done


if [[ $node_type = replicator ]]; then
  if [[ ${#positional_args[@]} -gt 2 ]]; then
    fullnode_usage "$@"
  fi

  read -r entrypoint entrypoint_address shift < <(find_entrypoint "${positional_args[@]}")
  shift "$shift"

  : "${identity_keypair_path:=$MORGAN_CONFIG_DIR/replicator-keypair$label.json}"
  storage_keypair_path="$MORGAN_CONFIG_DIR"/replicator-storage-keypair$label.json
  ledger_config_dir=$MORGAN_CONFIG_DIR/replicator-ledger$label
  configured_flag=$MORGAN_CONFIG_DIR/replicator$label.configured

  mkdir -p "$MORGAN_CONFIG_DIR"
  [[ -r "$identity_keypair_path" ]] || $morgan_keygen -o "$identity_keypair_path"
  [[ -r "$storage_keypair_path" ]] || $morgan_keygen -o "$storage_keypair_path"

  identity_pubkey=$($morgan_keygen pubkey "$identity_keypair_path")
  storage_pubkey=$($morgan_keygen pubkey "$storage_keypair_path")

  cat <<EOF
======================[ $node_type configuration ]======================
replicator pubkey: $identity_pubkey
storage pubkey: $storage_pubkey
ledger: $ledger_config_dir
======================================================================
EOF
  program=$morgan_replicator
  default_arg --entrypoint "$entrypoint_address"
  default_arg --identity "$identity_keypair_path"
  default_arg --storage-keypair "$storage_keypair_path"
  default_arg --ledger "$ledger_config_dir"

elif [[ $node_type = bootstrap_leader ]]; then
  if [[ ${#positional_args[@]} -ne 0 ]]; then
    fullnode_usage "Unknown argument: ${positional_args[0]}"
  fi

  [[ -f "$MORGAN_CONFIG_DIR"/bootstrap-leader-keypair.json ]] ||
    ledger_not_setup "$MORGAN_CONFIG_DIR/bootstrap-leader-keypair.json not found"

  #$morgan_ledger_tool --ledger "$MORGAN_CONFIG_DIR"/bootstrap-leader-ledger verify

  : "${identity_keypair_path:=$MORGAN_CONFIG_DIR/bootstrap-leader-keypair.json}"
  vote_keypair_path="$MORGAN_CONFIG_DIR"/bootstrap-leader-vote-keypair.json
  ledger_config_dir="$MORGAN_CONFIG_DIR"/bootstrap-leader-ledger
  accounts_config_dir="$MORGAN_CONFIG_DIR"/bootstrap-leader-accounts
  storage_keypair_path=$MORGAN_CONFIG_DIR/bootstrap-leader-storage-keypair.json
  configured_flag=$MORGAN_CONFIG_DIR/bootstrap-leader.configured

  default_arg --rpc-port 10099
  default_arg --rpc-drone-address 127.0.0.1:11100
  default_arg --gossip-port 10001

elif [[ $node_type = validator ]]; then
  if [[ ${#positional_args[@]} -gt 2 ]]; then
    fullnode_usage "$@"
  fi

  read -r entrypoint entrypoint_address shift < <(find_entrypoint "${positional_args[@]}")
  shift "$shift"

  : "${identity_keypair_path:=$MORGAN_CONFIG_DIR/validator-keypair$label.json}"
  vote_keypair_path=$MORGAN_CONFIG_DIR/validator-vote-keypair$label.json
  stake_keypair_path=$MORGAN_CONFIG_DIR/validator-stake-keypair$label.json
  storage_keypair_path=$MORGAN_CONFIG_DIR/validator-storage-keypair$label.json
  ledger_config_dir=$MORGAN_CONFIG_DIR/validator-ledger$label
  accounts_config_dir=$MORGAN_CONFIG_DIR/validator-accounts$label
  configured_flag=$MORGAN_CONFIG_DIR/validator$label.configured

  mkdir -p "$MORGAN_CONFIG_DIR"
  [[ -r "$identity_keypair_path" ]] || $morgan_keygen -o "$identity_keypair_path"
  [[ -r "$vote_keypair_path" ]] || $morgan_keygen -o "$vote_keypair_path"
  [[ -r "$stake_keypair_path" ]] || $morgan_keygen -o "$stake_keypair_path"
  [[ -r "$storage_keypair_path" ]] || $morgan_keygen -o "$storage_keypair_path"

  default_arg --entrypoint "$entrypoint_address"
  default_arg --rpc-drone-address "${entrypoint_address%:*}:11100"

  rsync_entrypoint_url=$(rsync_url "$entrypoint")
else
  echo "Error: Unknown node_type: $node_type"
  exit 1
fi


if [[ $node_type != replicator ]]; then
  identity_pubkey=$($morgan_keygen pubkey "$identity_keypair_path")
  vote_pubkey=$($morgan_keygen pubkey "$vote_keypair_path")
  storage_pubkey=$($morgan_keygen pubkey "$storage_keypair_path")

  cat <<EOF
======================[ $node_type configuration ]======================
identity pubkey: $identity_pubkey
vote pubkey: $vote_pubkey
storage pubkey: $storage_pubkey
ledger: $ledger_config_dir
accounts: $accounts_config_dir
========================================================================
EOF

  default_arg --identity "$identity_keypair_path"
  default_arg --voting-keypair "$vote_keypair_path"
  default_arg --vote-account "$vote_pubkey"
  default_arg --storage-keypair "$storage_keypair_path"
  default_arg --ledger "$ledger_config_dir"
  default_arg --accounts "$accounts_config_dir"

  if [[ -n $MORGAN_CUDA ]]; then
    program=$morgan_validator_cuda
  else
    program=$morgan_validator
  fi
fi

if [[ -z $CI ]]; then # Skip in CI
  # shellcheck source=scripts/tune-system.sh
  source "$here"/../scripts/tune-system.sh
fi

new_gensis_block() {
  ! diff -q "$MORGAN_RSYNC_CONFIG_DIR"/ledger/genesis.json "$ledger_config_dir"/genesis.json >/dev/null 2>&1
}

set -e
PS4="$(basename "$0"): "
while true; do
  if [[ ! -d "$MORGAN_RSYNC_CONFIG_DIR"/ledger ]]; then
    if [[ $node_type = bootstrap_leader ]]; then
      ledger_not_setup "$MORGAN_RSYNC_CONFIG_DIR/ledger does not exist"
    fi
    $rsync -vPr "${rsync_entrypoint_url:?}"/config/ledger "$MORGAN_RSYNC_CONFIG_DIR"
  fi

  if new_gensis_block; then
    # If the genesis block has changed remove the now stale ledger and vote
    # keypair for the node and start all over again
    (
      set -x
      rm -rf "$ledger_config_dir" "$accounts_config_dir" "$configured_flag"
    )
  fi

  if [[ ! -d "$ledger_config_dir" ]]; then
    cp -a "$MORGAN_RSYNC_CONFIG_DIR"/ledger/ "$ledger_config_dir"
    #$morgan_ledger_tool --ledger "$ledger_config_dir" verify
  fi

  trap '[[ -n $pid ]] && kill "$pid" >/dev/null 2>&1 && wait "$pid"' INT TERM ERR

  if ((stake)); then
    if [[ $node_type = validator ]]; then
      setup_validator_accounts "${entrypoint_address%:*}" \
        "$identity_keypair_path" \
        "$vote_keypair_path" \
        "$stake_keypair_path" \
        "$storage_keypair_path" \
        "$stake"
    elif [[ $node_type = replicator ]]; then
      setup_replicator_account "${entrypoint_address%:*}" \
        "$identity_keypair_path" \
        "$storage_keypair_path" \
        "$stake"
    fi
  fi

  echo "$PS4$program ${args[*]}"
  $program "${args[@]}" &
  pid=$!
  oom_score_adj "$pid" 1000

  if ((no_restart)); then
    wait "$pid"
    exit $?
  fi

  if [[ $node_type = bootstrap_leader ]]; then
    wait "$pid" || true
    echo "############## $node_type exited, restarting ##############"
    sleep 1
  else
    secs_to_next_genesis_poll=1
    while true; do
      if ! kill -0 "$pid"; then
        wait "$pid" || true
        echo "############## $node_type exited, restarting ##############"
        break
      fi

      sleep 1

      ((poll_for_new_genesis_block)) || continue
      ((secs_to_next_genesis_poll--)) && continue
      (
        set -x
        $rsync -r "${rsync_entrypoint_url:?}"/config/ledger "$MORGAN_RSYNC_CONFIG_DIR"
      ) || true
      new_gensis_block && break

      secs_to_next_genesis_poll=60
    done

    echo "############## New genesis detected, restarting $node_type ##############"
    kill "$pid" || true
    wait "$pid" || true
    # give the cluster time to come back up
    (
      set -x
      sleep 60
    )
  fi
done
