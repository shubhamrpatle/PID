#!/usr/bin/env bash
set -euo pipefail
export TUNING=lora
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" "$@"
