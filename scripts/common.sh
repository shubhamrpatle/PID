#!/usr/bin/env bash
# Shared path resolution only: no GPU reservation, audits or background jobs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${WORK_ROOT:-}" ]]; then
  if [[ -f "$ROOT/.work-root" ]]; then
    IFS= read -r WORK_ROOT < "$ROOT/.work-root"
  else
    WORK_ROOT="$(dirname "$ROOT")/refwave-assets"
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
export WORK_ROOT
# Do not inherit another checkout's PYTHONPATH or user-installed Python modules.
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/src"
