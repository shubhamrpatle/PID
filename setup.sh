#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"
if [[ "${1:-}" == --help ]]; then
  printf '%s\n' 'Usage: WORK_ROOT=/absolute/storage/path bash setup.sh' \
    'Optional: PYTHON_BOOTSTRAP=python3.11 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121'
  exit 0
fi
if [[ -z "${PYTHON_BOOTSTRAP:-}" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BOOTSTRAP=python3.10
  else
    PYTHON_BOOTSTRAP=python3
  fi
fi
"$PYTHON_BOOTSTRAP" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,12) else "Use Python 3.10 or 3.11, e.g. PYTHON_BOOTSTRAP=python3.10 bash setup.sh")'
if [[ "$WORK_ROOT" != /* ]]; then
  printf '%s\n' 'WORK_ROOT must be an absolute path.' >&2
  exit 2
fi
"$PYTHON_BOOTSTRAP" -m venv "$ROOT/.venv"
PYTHON_BIN="$ROOT/.venv/bin/python"
"$PYTHON_BIN" -m pip install --upgrade pip==25.2 wheel==0.45.1 setuptools==80.9.0
"$PYTHON_BIN" -m pip install --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}" \
  torch==2.5.1 torchvision==0.20.1
"$PYTHON_BIN" -m pip install -r "$ROOT/requirements.txt"
"$PYTHON_BIN" -m pip install --no-deps -e "$ROOT"
mkdir -p "$WORK_ROOT"
printf '%s\n' "$WORK_ROOT" > "$ROOT/.work-root"
printf '%s\n' 'Setup complete. Next: bash download.sh' "Assets: $WORK_ROOT"
