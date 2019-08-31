# |source| this file
#
# Common utilities shared by other scripts in this directory
#
# The following directive disable complaints about unused variables in this
# file:
# shellcheck disable=2034
#

MORGAN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. || exit 1; pwd)"

rsync=rsync

if [[ $(uname) != Linux ]]; then
  # Protect against unsupported configurations to prevent non-obvious errors
  # later. Arguably these should be fatal errors but for now prefer tolerance.
  if [[ -n $MORGAN_CUDA ]]; then
    echo "Warning: CUDA is not supported on $(uname)"
    MORGAN_CUDA=
  fi
fi

if [[ -f "$MORGAN_ROOT"/target/perf-libs/env.sh ]]; then
  # shellcheck source=/dev/null
  source "$MORGAN_ROOT"/target/perf-libs/env.sh
fi

if [[ -n $USE_INSTALL || ! -f "$MORGAN_ROOT"/Cargo.toml ]]; then
  morgan_program() {
    declare program="$1"
    printf "morgan-%s" "$program"
  }
else
  morgan_program() {
    declare program="$1"
    declare features="--features="
    if [[ "$program" =~ ^(.*)-cuda$ ]]; then
      program=${BASH_REMATCH[1]}
      features+="cuda,"
    fi
    if [[ $program = replicator ]]; then
      features+="chacha,"
    fi

    if [[ -r "$MORGAN_ROOT/$program"/Cargo.toml ]]; then
      maybe_package="--package morgan-$program"
    fi
    if [[ -n $NDEBUG ]]; then
      maybe_release=--release
    fi
    declare manifest_path="--manifest-path=$MORGAN_ROOT/$program/Cargo.toml"
    printf "cargo run $manifest_path $maybe_release $maybe_package --bin morgan-%s %s -- " "$program" "$features"
  }
fi

morgan_bench_tps=$(morgan_program bench-tps)
morgan_drone=$(morgan_program drone)
morgan_validator=$(morgan_program validator)
morgan_validator_cuda=$(morgan_program validator-cuda)
morgan_genesis=$(morgan_program genesis)
morgan_gossip=$(morgan_program gossip)
morgan_keygen=$(morgan_program keygen)
morgan_ledger_tool=$(morgan_program ledger-tool)
morgan_wallet=$(morgan_program wallet)
morgan_replicator=$(morgan_program replicator)

export RUST_LOG=${RUST_LOG:-morgan=info} # if RUST_LOG is unset, default to info
export RUST_BACKTRACE=1

# shellcheck source=scripts/configure-metrics.sh
source "$MORGAN_ROOT"/scripts/configure-metrics.sh

# The directory on the cluster entrypoint that is rsynced by other full nodes
MORGAN_RSYNC_CONFIG_DIR=$MORGAN_ROOT/config

# Configuration that remains local
MORGAN_CONFIG_DIR=$MORGAN_ROOT/config-local

default_arg() {
  declare name=$1
  declare value=$2

  for arg in "${args[@]}"; do
    if [[ $arg = "$name" ]]; then
      return
    fi
  done

  if [[ -n $value ]]; then
    args+=("$name" "$value")
  else
    args+=("$name")
  fi
}
