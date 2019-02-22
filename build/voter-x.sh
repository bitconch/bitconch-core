#!/bin/bash
#
# Start a dynamically-configured voter node
#

here=$(dirname "$0")

exec "$here"/validator.sh -x "$@"
