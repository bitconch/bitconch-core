#!/usr/bin/env bash
#
# Wallet sanity test
#
set -e

cd "$(dirname "$0")"/..

# shellcheck source=multinode-demo/common.sh
source multinode-demo/common.sh

if [[ -z $1 ]]; then # no network argument, use localhost by default
  entrypoint=(--url http://127.0.0.1:10099)
else
  entrypoint=("$@")
fi

# Tokens transferred to this address are lost forever...
garbage_address=vS3ngn1TfQmpsW1Z4NkLuqNAQFF3dYQw8UZ6TCx9bmq

check_balance_output() {
  declare expected_output="$1"
  exec 42>&1
  attempts=3
  while [[ $attempts -gt 0 ]]; do
    output=$($morgan_wallet "${entrypoint[@]}" balance | tee >(cat - >&42))
    if [[ "$output" =~ $expected_output ]]; then
      break
    else
      sleep 1
      (( attempts=attempts-1 ))
      if [[ $attempts -eq 0 ]]; then
        echo "Balance is incorrect.  Expected: $expected_output"
        exit 1
      fi
    fi
  done
}

pay_and_confirm() {
  exec 42>&1
  signature=$($morgan_wallet "${entrypoint[@]}" pay "$@" | tee >(cat - >&42))
  $morgan_wallet "${entrypoint[@]}" confirm "$signature"
}

$morgan_keygen

node_readiness=false
timeout=60
while [[ $timeout -gt 0 ]]; do
  output=$($morgan_wallet "${entrypoint[@]}" get-transaction-count)
  if [[ -n $output ]]; then
    node_readiness=true
    break
  fi
  sleep 2
  (( timeout=timeout-2 ))
done
if ! "$node_readiness"; then
  echo "Timed out waiting for cluster to start"
  exit 1
fi

$morgan_wallet "${entrypoint[@]}" address
check_balance_output "0 difs"
$morgan_wallet "${entrypoint[@]}" airdrop 60
check_balance_output "60 difs"
$morgan_wallet "${entrypoint[@]}" airdrop 40
check_balance_output "100 difs"
pay_and_confirm $garbage_address 99
check_balance_output "1 dif"

echo PASS
exit 0
