#!/bin/bash
#
# Creates a fullnode configuration
#

here=$(dirname "$0")
# shellcheck source=multinode-demo/common.sh
source "$here"/common.sh

usage () {
  exitcode=0
  if [[ -n "$1" ]]; then
    exitcode=1
    echo "Error: $*"
  fi
  cat <<EOF
usage: $0 [-n num_tokens] [-l] [-p] [-t node_type]

Creates a fullnode configuration

 -n num_tokens  - Number of tokens to create
 -l             - Detect network address from local machine configuration, which
                  may be a private IP address unaccessible on the Intenet (default)
 -p             - Detect public address using public Internet servers
 -t node_type   - Create configuration files only for this kind of node.  Valid
                  options are validator or leader.  Creates configuration files
                  for both by default

EOF
  exit $exitcode
}

#go build the binaries
 
eval $go_build_buffett_bench_tps
eval $go_build_buffett_wallet
eval $go_build_buffett_drone
eval $go_build_buffett_fullnode
eval $go_build_buffett_fullnode_config
eval $go_build_buffett_fullnode_cuda
eval $go_build_buffett_genesis
eval $go_build_buffett_keygen
echo $go_build_buffett_ledger_tool
eval $go_build_buffett_ledger_tool

echo "Building completed, should check the build result."

ip_address_arg=-l
num_tokens=1000000000
node_type_leader=true
node_type_validator=true
node_type_client=true
while getopts "h?n:lpt:" opt; do
  case $opt in
  h|\?)
    usage
    exit 0
    ;;
  l)
    ip_address_arg=-l
    ;;
  p)
    ip_address_arg=-p
    ;;
  n)
    num_tokens="$OPTARG"
    ;;
  t)
    node_type="$OPTARG"
    case $OPTARG in
    leader)
      node_type_leader=true
      node_type_validator=false
      node_type_client=false
      ;;
    validator)
      node_type_leader=false
      node_type_validator=true
      node_type_client=false
      ;;
    client)
      node_type_leader=false
      node_type_validator=false
      node_type_client=true
      ;;
    *)
      usage "Error: unknown node type: $node_type"
      ;;
    esac
    ;;
  *)
    usage "Error: unhandled option: $opt"
    ;;
  esac
done


set -e

for i in "$BUFFETT_CONFIG_DIR" "$BUFFETT_CONFIG_VALIDATOR_DIR" "$BUFFETT_CONFIG_PRIVATE_DIR"; do
  echo "Cleaning $i"
  rm -rvf "$i"
  mkdir -p "$i"
done

if $node_type_client; then
  echo "================================================="
  echo "= Evoke keymaker to create some stuff on client ="
  echo "================================================="
  client_id_path="$BUFFETT_CONFIG_PRIVATE_DIR"/client-id.json
  $buffett_keygen -o "$client_id_path"
  ls -lhR "$BUFFETT_CONFIG_PRIVATE_DIR"/
fi

if $node_type_leader; then
  leader_address_args=("$ip_address_arg")
  leader_id_path="$BUFFETT_CONFIG_PRIVATE_DIR"/leader-id.json
  mint_path="$BUFFETT_CONFIG_PRIVATE_DIR"/mint.json
  echo "================================================="
  echo "= On Leader Node                                ="  
  echo "================================================="
  echo " Utility    : keymaker "
  echo " Out File   : $leader_id_path "
  echo "================================================="
  $buffett_keygen -o "$leader_id_path"

  echo " Utility    : keymaker "
  echo " Out File   : $mint_path with $num_tokens tokens"
  echo "================================================="
  $buffett_keygen -o "$mint_path"

  echo " Utility    : genesis tool "
  echo " Out File   : $BUFFETT_CONFIG_DIR/ledger"
  echo "================================================="
  $buffett_genesis --tokens="$num_tokens" --ledger "$BUFFETT_CONFIG_DIR"/ledger < "$mint_path"

  echo " Utility    : fullnode configuration tool "
  echo " Out File   : $BUFFETT_CONFIG_DIR/leader.json"
  echo "================================================="
  $buffett_fullnode_config --keypair="$leader_id_path" "${leader_address_args[@]}" -o "$BUFFETT_CONFIG_DIR"/leader.json

  ls -lhR "$BUFFETT_CONFIG_DIR"/
  ls -lhR "$BUFFETT_CONFIG_PRIVATE_DIR"/
fi


if $node_type_validator; then
  echo "================================================="
  echo "= On Voter Node                                 ="  
  echo "================================================="
  echo " Utility    : keymaker "
  echo " Out File   : $validator_id_path "
  echo "================================================="
  validator_address_args=("$ip_address_arg" -b 9000)
  validator_id_path="$BUFFETT_CONFIG_PRIVATE_DIR"/validator-id.json
  $buffett_keygen -o "$validator_id_path"

  echo " Utility    : fullnode config "
  echo " In File    : $BUFFETT_CONFIG_VALIDATOR_DIR/validator.json "
  echo " Out File   : $validator_id_path"
  echo "================================================="
  $buffett_fullnode_config --keypair="$validator_id_path" "${validator_address_args[@]}" -o "$BUFFETT_CONFIG_VALIDATOR_DIR"/validator.json

  ls -lhR "$BUFFETT_CONFIG_VALIDATOR_DIR"/
fi
