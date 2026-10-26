#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"
GPUS="${1:-8}"
STAGES="${2:-both}"
RUN_TAG="${3:-}"
if [[ "$GPUS" == --help ]]; then
  printf '%s\n' 'Usage: bash train.sh [1-8 GPUs] [both|stage1|stage2] [run-label]' \
    'Fresh runs create a unique folder automatically.' \
    'Optional label: ft_2 (1-64 letters/digits/._-, starting with a letter or digit)' \
    'Resume or run Stage 2: RUN_ROOT=/absolute/existing/run/path' \
    'Epochs: STAGE1_EPOCHS=1 STAGE2_EPOCHS=1 (set independently)' \
    'Extra Stage-2 training in a new run: CONTINUE_FROM=/old/run/stage2/best.pt' \
    'Methods: METHOD=refwave (default), standard, or class_isolated' \
    'Optional: BATCH_SIZE=4 RECORDS_DIR=/data/refwave_v2_cod'
  exit 0
fi
if (( $# > 3 )); then
  printf '%s\n' 'Expected GPUs, stages and at most one optional run label.' >&2
  exit 2
fi
if [[ -n "$RUN_TAG" && ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  printf '%s\n' 'Run label must be 1-64 letters/digits/._-, starting with a letter or digit.' >&2
  exit 2
fi
case "$GPUS" in [1-8]) ;; *) printf '%s\n' 'GPU count must be 1-8.' >&2; exit 2;; esac
case "$STAGES" in both|stage1|stage2) ;; *) printf '%s\n' 'Stage must be both, stage1 or stage2.' >&2; exit 2;; esac
METHOD="${METHOD:-refwave}"
TUNING="${TUNING:-lora}"
case "$TUNING" in lora|full) ;; *) printf '%s\n' 'TUNING must be lora or full.' >&2; exit 2;; esac
case "$METHOD" in refwave|standard|class_isolated) ;; *) printf '%s\n' 'METHOD must be refwave, standard or class_isolated.' >&2; exit 2;; esac
if (( ${SLURM_JOB_NUM_NODES:-1} != 1 )); then
  printf '%s\n' 'This launcher supports one compute node. Request --nodes=1.' >&2
  exit 2
fi
MODEL_DIR="${MODEL_DIR:-$WORK_ROOT/LocateAnything-3B}"
DATA_ROOT="${DATA_ROOT:-$WORK_ROOT/dataset}"
RECORDS_DIR="${RECORDS_DIR:-$DATA_ROOT/refwave_v1}"
LORA_RANK="${LORA_RANK:-32}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-1}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-1}"
for epochs in "$STAGE1_EPOCHS" "$STAGE2_EPOCHS"; do
  if [[ ! "$epochs" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\n' 'STAGE1_EPOCHS and STAGE2_EPOCHS must be positive integers.' >&2
    exit 2
  fi
done
if [[ -n "${CONTINUE_FROM:-}" && "$STAGES" != stage2 ]]; then
  printf '%s\n' 'CONTINUE_FROM requires stage2; it must not rerun Stage 1.' >&2
  exit 2
fi
if [[ -z "${RUN_ROOT:-}" ]]; then
  if [[ "$STAGES" == stage2 && -z "${CONTINUE_FROM:-}" ]]; then
    printf '%s\n' 'Set RUN_ROOT to the existing run containing completed stage1/ before starting Stage 2.' >&2
    exit 2
  fi
  mkdir -p "$WORK_ROOT/runs"
  run_label="${METHOD}_${TUNING}_${GPUS}gpu_s1e${STAGE1_EPOCHS}_s2e${STAGE2_EPOCHS}_${STAGES}"
  if [[ -n "${CONTINUE_FROM:-}" ]]; then
    run_label="${METHOD}_${TUNING}_${GPUS}gpu_stage2_extrae${STAGE2_EPOCHS}"
  fi
  if [[ -n "$RUN_TAG" ]]; then
    run_label="${run_label}_${RUN_TAG}"
  fi
  RUN_ROOT="$(mktemp -d "$WORK_ROOT/runs/${run_label}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
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
  local stage="$1" lr="$2" head_lr="$3" epochs="$4"
  local parent=()
  if [[ "$stage" == stage2 ]]; then
    if [[ -n "${CONTINUE_FROM:-}" ]]; then
      parent=(--continue-checkpoint "$CONTINUE_FROM")
    elif [[ ! -f "$RUN_ROOT/stage2/last.pt" ]]; then
      if [[ ! -f "$RUN_ROOT/stage1/done.json" ]]; then
        printf '%s\n' "Stage 1 is incomplete in $RUN_ROOT/stage1. Keep RUN_ROOT set to this run and finish Stage 1 first." >&2
        return 1
      fi
      parent=(--init-checkpoint "$RUN_ROOT/stage1/best.pt")
    fi
  fi
  printf 'Stage=%s epochs=%s parent=%s\n' "$stage" "$epochs" "${parent[*]:-base or existing last.pt}"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$GPUS" \
    -m refwave.train --model "$MODEL_DIR" --data-root "$DATA_ROOT" \
    --train-data "$RECORDS_DIR/$stage.jsonl" \
    --validation-data "$RECORDS_DIR/validation.jsonl" \
    --stage "$stage" --epochs "$epochs" --method "$METHOD" --tuning "$TUNING" --output "$RUN_ROOT/$stage" \
    --lora-rank "$LORA_RANK" --batch-size "${BATCH_SIZE:-8}" \
    --accumulation "${GRAD_ACCUM:-1}" --batch-tokens "${BATCH_TOKENS:-16384}" \
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
  if [[ -n "$RUN_TAG" ]]; then
    printf 'Requested run label=%s (applies to new folders only)\n' "$RUN_TAG"
  fi
  printf 'DATA_ROOT=%s\nRECORDS_DIR=%s\n' "$DATA_ROOT" "$RECORDS_DIR"
  printf 'STAGE1_EPOCHS=%s STAGE2_EPOCHS=%s\n' "$STAGE1_EPOCHS" "$STAGE2_EPOCHS"
  printf 'Resume with the same settings: RUN_ROOT=%q RECORDS_DIR=%q METHOD=%q STAGE1_EPOCHS=%q STAGE2_EPOCHS=%q bash %q %q %q\n' \
    "$RUN_ROOT" "$RECORDS_DIR" "$METHOD" "$STAGE1_EPOCHS" "$STAGE2_EPOCHS" "$ROOT/train_${TUNING}.sh" "$GPUS" "$STAGES"
  if [[ "$STAGES" != stage2 ]]; then
    run_stage stage1 "${STAGE1_LR:-2e-5}" "${STAGE1_HEAD_LR:-1e-6}" "$STAGE1_EPOCHS"
  fi
  if [[ "$STAGES" != stage1 ]]; then
    run_stage stage2 "${STAGE2_LR:-1e-5}" "${STAGE2_HEAD_LR:-5e-7}" "$STAGE2_EPOCHS"
  fi
}
# Preserve failures through tee; keep logging during a normal Ctrl+C shutdown.
run_pipeline 2>&1 | tee -ia "$RUN_ROOT/console.log"
