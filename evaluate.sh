#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"
if (( $# != 2 )); then
  printf '%s\n' 'Usage: bash evaluate.sh [1-8 GPUs] /path/to/best.pt' \
    'Use baseline instead of a checkpoint to evaluate released direct-box LA.' >&2
  exit 2
fi
GPUS="$1"
case "$GPUS" in [1-8]) ;; *) printf '%s\n' 'GPU count must be 1-8.' >&2; exit 2;; esac
if (( ${SLURM_JOB_NUM_NODES:-1} != 1 )); then
  printf '%s\n' 'Request one compute node for this launcher.' >&2; exit 2
fi
checkpoint=()
if [[ "$2" == baseline ]]; then
  METHOD=standard
  output_default="$WORK_ROOT/evaluation/baseline"
else
  checkpoint=(--checkpoint "$2")
  output_default="$(dirname "$2")/evaluation"
fi
ROUND2_EXECUTION="${ROUND2_EXECUTION:-fused_one_pass}"
if [[ "${METHOD:-refwave}" == refwave ]]; then
  output_default="${output_default}_${ROUND2_EXECUTION}"
fi
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DATA_ROOT="${DATA_ROOT:-$WORK_ROOT/dataset}"
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$GPUS" \
  -m refwave.evaluate --model "${MODEL_DIR:-$WORK_ROOT/LocateAnything-3B}" \
  --data-root "$DATA_ROOT" \
  --records "${EVAL_RECORDS:-$DATA_ROOT/refwave_v1/validation.jsonl}" \
  --output "${EVAL_OUTPUT:-$output_default}" --method "${METHOD:-refwave}" \
  --class-wave "${CLASS_WAVE:-18}" --box-wave "${BOX_WAVE:-200}" \
  --round2-execution "$ROUND2_EXECUTION" \
  --max-sequence "${MAX_SEQUENCE:-32768}" \
  --image-token-limit "${IMAGE_TOKEN_LIMIT:-6000}" --max-new-tokens "${MAX_NEW_TOKENS:-16384}" \
  --limit "${EVAL_LIMIT:-0}" --seed "${SEED:-20260908}" "${checkpoint[@]}"
