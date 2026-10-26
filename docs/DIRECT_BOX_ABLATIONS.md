# Class-isolated direct boxes versus RefWave

Use the existing dataset, environment and training launchers. No data conversion,
new image download or startup verifier is required. This adds only
`METHOD=class_isolated`; existing `standard` and `refwave` behavior is unchanged.

| Method | Detection supervision | Grounding supervision |
|---|---|---|
| standard | Original multi-category direct answer | Original direct answer |
| class_isolated | One direct answer per queried class | Original direct answer |
| refwave | Class points plus addressed boxes | Points plus addressed boxes |

The isolated control partitions the existing grouped direct answer, preserving
every box, reference tag, class order and existing None answer. It introduces no
new negatives and never uses sampled Round-2 boxes as its detection targets.
Each source episode is still visited once per epoch. The image is encoded once
for its training episode and class branches cannot attend to each other's answers.
The loss is the mean of the per-class token-mean direct losses for detection;
grounding retains its original direct loss. There is no point-generation loss.
More branches can mean more computation despite equal source-episode exposure.

This control tests whether RefWave helps beyond class-separated direct decoding.
It is not a pure compute-matched or supervised-token-matched ablation: direct
outputs retain native reference tags, and RefWave has two supervised routes.
Do not claim a benefit until the matched checkpoints have been evaluated.

All full modes train MoonViT, projector, Qwen and head. Keep the same batch,
accumulation, learning rates, epochs and seed across controls. Each method selects
its own Stage-1 best validation-loss checkpoint as the Stage-2 parent; Stage 2
uses a fresh optimizer. Validation-loss values across different objectives are
not directly comparable. Compare generated results on the same evaluation set.

LoRA modes instead train Qwen LoRA plus the complete LM head, with the same
rank (default 32); vision and projector are frozen. Compare LoRA to LoRA or
full to full, not one tuning scope to another. Both stages select by the minimum
`detection_grounding_macro_loss` on the same validation records, at the same
evaluation interval (default 300 updates) and epoch boundaries. Stage 2 starts
from its own method's Stage-1 `best.pt`, not RefWave's checkpoint for both methods.

## Matched commands

Run from `RefWave/` in the activated environment, one job per GPU allocation.
These commands use the same default `refwave_v1` records (or the same exported
`RECORDS_DIR` if configured), one epoch per stage and the same training defaults.
Use identical epoch, batch, accumulation, seed and learning-rate overrides for
the two methods. Clear `RUN_ROOT` and `CONTINUE_FROM` for fresh runs.

```bash
# LoRA control:
RUN_ROOT= CONTINUE_FROM= METHOD=class_isolated WANDB_MODE=online bash train_lora.sh 8 both
# Matched RefWave LoRA, in its own allocation:
RUN_ROOT= CONTINUE_FROM= METHOD=refwave WANDB_MODE=online bash train_lora.sh 8 both
```

For full-parameter training use `train_full.sh` for **both** commands. Replace
8 with the allocated GPU count for both jobs. Record hashes, global batch,
trainable scope and validation hashes are saved in each stage's `config.json`;
compare these along with the complete recipe before reporting results.

## Training

Run from `RefWave/`, in the activated environment. Keep your existing WORK_ROOT.
For the COD update, select the already downloaded record directory:

```bash
export DATA_ROOT="$WORK_ROOT/dataset"
export RECORDS_DIR="$DATA_ROOT/refwave_v2_cod"
unset RUN_ROOT CONTINUE_FROM
export STAGE1_EPOCHS=1 STAGE2_EPOCHS=1

# Run one experiment per allocation, not both concurrently on the same GPUs.
METHOD=standard WANDB_MODE=online bash train_full.sh 8 both
# Alternatively:
METHOD=class_isolated WANDB_MODE=online bash train_full.sh 8 both
```

Use 1-8 allocated GPUs. Defaults are unchanged. Fresh runs have separate
`standard_full_...` and `class_isolated_full_...` folders. Do not resume one
method from the other method's checkpoint. RefWave remains available through
`METHOD=refwave bash train_full.sh 8 both`.

## Evaluation

Set CHECKPOINT to the selected method's `stage2/best.pt`. EVAL_RECORDS must be
the same explicit-query evaluation file for all methods. The existing default
is validation, not automatically a full benchmark. All benchmark categories,
including absent categories, must be in the evaluation queries; do not construct
the query list from GT-present categories.

```bash
METHOD=standard EVAL_OUTPUT="$WORK_ROOT/evaluation/standard_full_direct" \
  bash evaluate.sh 8 "$CHECKPOINT"

# For the class-isolated checkpoint: one class at a time or shared-prefix waves.
METHOD=class_isolated CLASS_WAVE=1 \
  EVAL_OUTPUT="$WORK_ROOT/evaluation/isolated_full_seq" \
  bash evaluate.sh 8 "$CHECKPOINT"
METHOD=class_isolated CLASS_WAVE=18 \
  EVAL_OUTPUT="$WORK_ROOT/evaluation/isolated_full_wave18" \
  bash evaluate.sh 8 "$CHECKPOINT"
```

The isolated method has no second inference round. It uses the existing hybrid
PBD decoder for multiple boxes per class, not the one-box refinement decoder.
Outputs retain raw answers, latency and peak allocated memory; no NMS or
containment filter is added. MAX_NEW_TOKENS is per class branch for the isolated
control, versus per complete query for standard. Report this budget difference;
equal numeric limits alone do not establish compute-matched inference. Wave
equivalence and actual GPU timing still require checkpoint evaluation.
