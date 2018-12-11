#!/bin/bash -e

pwd=$PWD

echo --- Build
cd "$(dirname "$0")/signing"
(
  set -x
  mkdir -p "$pwd"/temp
  openssl genrsa -out "$pwd"/temp/priv_key.pem -3 3072
  openssl rsa -in "$pwd"/temp/priv_key.pem -pubout -out "$pwd"/temp/pub_key.pem
  make LIBS_PATH="$pwd"/libs OUT="$pwd"/dist PRIV_KEY="$pwd"/temp/priv_key.pem PUB_KEY="$pwd"/temp/pub_key.pem
)

echo --- Build Enclave Test
cd "../test"
(
  set -x
  make LIBS_PATH="$pwd"/libs OUT="$pwd"/dist
)
