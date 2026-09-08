#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"
exec "$PYTHON_BIN" "$ROOT/tools/download_refwave_assets.py" --work-root "$WORK_ROOT" "$@"
