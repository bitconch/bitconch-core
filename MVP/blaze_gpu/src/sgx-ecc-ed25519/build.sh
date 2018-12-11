#!/bin/bash -e

pwd=$PWD
cd "$(dirname "$0")"

echo --- Build
(
  set -x
  make OUT="$pwd"/libs
)

