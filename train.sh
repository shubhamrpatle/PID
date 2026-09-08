#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"
GPUS="${1:-8}"
STAGES="${2:-both}"
if [[ "$GPUS" == --help ]]; then
  printf '%s\n' 'Usage: bash train.sh [1-8 GPUs] [both|stage1|stage2]' \
    'Fresh runs create a unique folder automatically.' \
    'Resume or run Stage 2: RUN_ROOT=/absolute/existing/run/path' \
    'Optional: METHOD=standard BATCH_SIZE=4'
  exit 0
fi
case "$GPUS" in [1-8]) ;; *) printf '%s\n' 'GPU count must be 1-8.' >&2; exit 2;; esac
case "$STAGES" in both|stage1|stage2) ;; *) printf '%s\n' 'Stage must be both, stage1 or stage2.' >&2; exit 2;; esac
METHOD="${METHOD:-refwave}"
TUNING="${TUNING:-lora}"
case "$TUNING" in lora|full) ;; *) printf '%s\n' 'TUNING must be lora or full.' >&2; exit 2;; esac
case "$METHOD" in refwave|standard) ;; *) printf '%s\n' 'METHOD must be refwave or standard.' >&2; exit 2;; esac
if (( ${SLURM_JOB_NUM_NODES:-1} != 1 )); then
  printf '%s\n' 'This launcher supports one compute node. Request --nodes=1.' >&2
  exit 2
fi
MODEL_DIR="${MODEL_DIR:-$WORK_ROOT/LocateAnything-3B}"
DATA_ROOT="${DATA_ROOT:-$WORK_ROOT/dataset}"
LORA_RANK="${LORA_RANK:-32}"
if [[ -z "${RUN_ROOT:-}" ]]; then
  if [[ "$STAGES" == stage2 ]]; then
    printf '%s\n' 'Set RUN_ROOT to the existing run containing completed stage1/ before starting Stage 2.' >&2
    exit 2
  fi
  mkdir -p "$WORK_ROOT/runs"
  RUN_ROOT="$(mktemp -d "$WORK_ROOT/runs/${METHOD}_${TUNING}_${GPUS}gpu_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
else
  mkdir -p "$RUN_ROOT"
fi
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
if [[ "$TUNING" == full ]]; then
  BATCH_SIZE="${BATCH_SIZE:-1}"
  GRAD_ACCUM="${GRAD_ACCUM:-4}"
  STAGE1_LR="${STAGE1_LR:-2e-6}"
  STAGE2_LR="${STAGE2_LR:-1e-6}"
fi
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Respect CUDA_VISIBLE_DEVICES from the allocation; never choose or clear GPUs.
run_stage() {
  local stage="$1" lr="$2" head_lr="$3"
  local parent=()
  if [[ "$stage" == stage2 ]]; then
    if [[ ! -f "$RUN_ROOT/stage1/done.json" ]]; then
      printf '%s\n' "Stage 1 is incomplete in $RUN_ROOT/stage1. Keep RUN_ROOT set to this run and finish Stage 1 first." >&2
      return 1
    fi
    parent=(--init-checkpoint "$RUN_ROOT/stage1/last.pt")
  fi
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$GPUS" \
    -m refwave.train --model "$MODEL_DIR" --data-root "$DATA_ROOT" \
    --train-data "$DATA_ROOT/refwave_v1/$stage.jsonl" \
    --validation-data "$DATA_ROOT/refwave_v1/validation.jsonl" \
    --stage "$stage" --method "$METHOD" --tuning "$TUNING" --output "$RUN_ROOT/$stage" \
    --lora-rank "$LORA_RANK" --batch-size "${BATCH_SIZE:-4}" \
    --accumulation "${GRAD_ACCUM:-1}" --batch-tokens "${BATCH_TOKENS:-8192}" \
    --max-sequence "${MAX_SEQUENCE:-32768}" --image-token-limit "${IMAGE_TOKEN_LIMIT:-6000}" \
    --workers "${WORKERS:-2}" --learning-rate "$lr" --head-lr "$head_lr" \
    --vision-lr "${VISION_LR:-1e-6}" --projector-lr "${PROJECTOR_LR:-1e-5}" \
    --eval-steps "${EVAL_STEPS:-300}" --save-steps "${SAVE_STEPS:-300}" \
    --seed "${SEED:-20260908}" --wandb-project "${WANDB_PROJECT:-refwave-fullscale}" \
    --wandb-mode "${WANDB_MODE:-disabled}" "${parent[@]}"
}
run_pipeline() {
  printf '\n=== %s ===\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'RUN_ROOT=%s\nCONSOLE_LOG=%s/console.log\n' "$RUN_ROOT" "$RUN_ROOT"
  printf 'Resume with the same settings: RUN_ROOT=%q METHOD=%q bash %q %q %q\n' \
    "$RUN_ROOT" "$METHOD" "$ROOT/train_${TUNING}.sh" "$GPUS" "$STAGES"
  if [[ "$STAGES" != stage2 ]]; then
    run_stage stage1 "${STAGE1_LR:-2e-5}" "${STAGE1_HEAD_LR:-1e-6}"
  fi
  if [[ "$STAGES" != stage1 ]]; then
    run_stage stage2 "${STAGE2_LR:-1e-5}" "${STAGE2_HEAD_LR:-5e-7}"
  fi
}
# Preserve failures through tee; keep logging during a normal Ctrl+C shutdown.
run_pipeline 2>&1 | tee -ia "$RUN_ROOT/console.log"
