# |source| this file
#
# Common utilities shared by other scripts in this directory
#
# The following directive disable complaints about unused variables in this
# file:
# shellcheck disable=2034
#

SOROS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. || exit 1; pwd)"

rsync=rsync
bootstrap_leader_logger="tee bootstrap-leader.log"
fullnode_logger="tee fullnode.log"
drone_logger="tee drone.log"

if [[ $(uname) != Linux ]]; then
  # Protect against unsupported configurations to prevent non-obvious errors
  # later. Arguably these should be fatal errors but for now prefer tolerance.
  if [[ -n $SOROS_CUDA ]]; then
    echo "Warning: CUDA is not supported on $(uname)"
    SOROS_CUDA=
  fi
fi

if [[ -n $USE_INSTALL || ! -f "$SOROS_ROOT"/Cargo.toml ]]; then
  soros_program() {
    declare program="$1"
    printf "soros-%s" "$program"
  }
else
  soros_program() {
    declare program="$1"
    declare features="--features="
    if [[ "$program" =~ ^(.*)-cuda$ ]]; then
      program=${BASH_REMATCH[1]}
      features+="cuda,"
    fi

    if [[ -r "$SOROS_ROOT/$program"/Cargo.toml ]]; then
      maybe_package="--package soros-$program"
    fi
    if [[ -n $NDEBUG ]]; then
      maybe_release=--release
    fi
    declare manifest_path="--manifest-path=$SOROS_ROOT/$program/Cargo.toml"
    printf "cargo run $manifest_path $maybe_release $maybe_package --bin soros-%s %s -- " "$program" "$features"
  }
  # shellcheck disable=2154 # 'here' is referenced but not assigned
  LD_LIBRARY_PATH=$(cd "$SOROS_ROOT/target/perf-libs" && pwd):$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH
fi

soros_bench_tps=$(soros_program bench-tps)
soros_drone=$(soros_program drone)
soros_fullnode=$(soros_program fullnode)
soros_fullnode_cuda=$(soros_program fullnode-cuda)
soros_genesis=$(soros_program genesis)
soros_gossip=$(soros_program gossip)
soros_keygen=$(soros_program keygen)
soros_ledger_tool=$(soros_program ledger-tool)
soros_wallet=$(soros_program wallet)

export RUST_LOG=${RUST_LOG:-soros=info} # if RUST_LOG is unset, default to info
export RUST_BACKTRACE=1

# shellcheck source=scripts/configure-metrics.sh
source "$SOROS_ROOT"/scripts/configure-metrics.sh

tune_system() {
  # Skip in CI
  [[ -z $CI ]] || return 0

  # shellcheck source=scripts/ulimit-n.sh
  source "$SOROS_ROOT"/scripts/ulimit-n.sh

  # Reference: https://medium.com/@CameronSparr/increase-os-udp-buffers-to-improve-performance-51d167bb1360
  if [[ $(uname) = Linux ]]; then
    (
      set -x +e
      # test the existence of the sysctls before trying to set them
      # go ahead and return true and don't exit if these calls fail
      sysctl net.core.rmem_max 2>/dev/null 1>/dev/null &&
          sudo sysctl -w net.core.rmem_max=161061273 1>/dev/null 2>/dev/null

      sysctl net.core.rmem_default 2>/dev/null 1>/dev/null &&
          sudo sysctl -w net.core.rmem_default=161061273 1>/dev/null 2>/dev/null

      sysctl net.core.wmem_max 2>/dev/null 1>/dev/null &&
          sudo sysctl -w net.core.wmem_max=161061273 1>/dev/null 2>/dev/null

      sysctl net.core.wmem_default 2>/dev/null 1>/dev/null &&
          sudo sysctl -w net.core.wmem_default=161061273 1>/dev/null 2>/dev/null
    ) || true
  fi

  if [[ $(uname) = Darwin ]]; then
    (
      if [[ $(sysctl net.inet.udp.maxdgram | cut -d\  -f2) != 65535 ]]; then
        echo "Adjusting maxdgram to allow for large UDP packets, see BLOB_SIZE in src/packet.rs:"
        set -x
        sudo sysctl net.inet.udp.maxdgram=65535
      fi
    )

  fi
}

fullnode_usage() {
  if [[ -n $1 ]]; then
    echo "$*"
    echo
  fi
  cat <<EOF
usage: $0 [--blockstream PATH] [--init-complete-file FILE] [--label LABEL] [--stake DIF] [--no-voting] [--rpc-port port] [rsync network path to bootstrap leader configuration] [network entry point]

Start a full node

  --blockstream PATH        - open blockstream at this unix domain socket location
  --init-complete-file FILE - create this file, if it doesn't already exist, once node initialization is complete
  --label LABEL             - Append the given label to the fullnode configuration files, useful when running
                              multiple fullnodes from the same filesystem location
  --stake DIF          - Number of dif to stake
  --no-voting               - start node without vote signer
  --rpc-port port           - custom RPC port for this node

EOF
  exit 1
}

# The directory on the cluster entrypoint that is rsynced by other full nodes
SOROS_RSYNC_CONFIG_DIR=$SOROS_ROOT/config

# Configuration that remains local
SOROS_CONFIG_DIR=$SOROS_ROOT/config-local
