#!/usr/bin/env bash
set -euo pipefail
export TUNING=full
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" "$@"
