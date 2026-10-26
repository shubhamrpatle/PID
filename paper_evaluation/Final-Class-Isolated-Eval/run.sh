#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CONTROLS_DIR="${CONTROLS_DIR:-$(cd .. && pwd)/Final-Controls-Eval}"
export PID_DIR="${PID_DIR:-$(cd .. && pwd)/Final-RefWave-Eval}"
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
python_path="$(command -v "${PYTHON_BIN:-}" || true)"
if [[ -z "$python_path" || ! -x "$python_path" || -d "$python_path" ]]; then
  printf '%s\n' "Python executable not found: ${PYTHON_BIN:-none}" \
    'Activate your evaluation environment or set PYTHON_BIN=/absolute/environment/bin/python.' >&2
  exit 1
fi
export PYTHON_BIN="$python_path"
if (( $# == 0 )); then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=, read -ra allocated <<< "$CUDA_VISIBLE_DEVICES"
    set -- "${allocated[@]}"
  else
    set -- 0 1 2 3 4 5 6 7
  fi
fi
exec "$PYTHON_BIN" -u evaluate.py launch --max-workers "${MAX_WORKERS:-4}" --gpus "$@"
