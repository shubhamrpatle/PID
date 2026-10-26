#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CONTROLS_DIR="${CONTROLS_DIR:-$(cd .. && pwd)/Final-Controls-Eval}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  elif [[ -x ../.venv/bin/python ]]; then
    PYTHON_BIN="$(cd .. && pwd)/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python || command -v python3 || true)"
  fi
fi
if [[ -z "$PYTHON_BIN" ]] || ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf '%s\n' 'Set PYTHON_BIN to the existing LA_DET/.venv/bin/python.' >&2
  exit 2
fi
export PYTHON_BIN
if (( $# == 0 )); then
  set -- 0 1 2 3
fi
exec "$PYTHON_BIN" -u dynamic_queue.py launch --max-workers "${MAX_WORKERS:-4}" --gpus "$@"
