# Training

The method identifier `refwave` in commands and file names is PID; see [Naming](REPRODUCIBILITY.md#naming). Set up the environment first ([Installation](../INSTALL.md)).

## 2. Download the Model and Data

```bash
export WORK_ROOT=/scratch/$USER/refwave-assets
bash download.sh
```

Choose a writable persistent path on your cluster. With manual venv/Conda setup,
export the **same** `WORK_ROOT` in every new shell and submission environment.
Default when unset: the `refwave-assets/` sibling of this repository.

The downloader bootstraps released [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B),
the existing image archives, and historical `refwave_v1/` records at pinned
revisions. **For new training, follow it with the corrected-record download below.**
The downloader remains pinned for old-run reproducibility; it does not automatically
select the correction. Existing users with extracted images and model weights can
skip `bash download.sh` and download only these records.

The corrected release is
[`refwave_v6_xview_corrected_20260914/`](https://huggingface.co/datasets/shubhampatle/Pixel-PIVR-Magnified-v2/tree/591d53da15234597941394f2848740151209434b/refwave_v6_xview_corrected_20260914).
It contains **116,386 Stage-1, 39,851 Stage-2 and 740 validation episodes**.
Both RefWave rounds and standard direct-box targets are stored in each episode.
It repairs xView source category alignment and regenerates affected negatives;
ambiguous grounding and confirmed benchmark-overlap episodes are quarantined.
It does not include the rolled-back COD or grounding-balanced extensions.
See [the correction guide](XVIEW_CORRECTED_V6.md) for counts and limitations.
The historical dataset name does not imply magnified-crop training.

Run on an internet-enabled node before requesting offline compute nodes.
Public downloads need no Hugging Face login. Images are unchanged; this command
downloads only records, manifests and documentation into your existing dataset root:

```bash
export DATA_ROOT="$WORK_ROOT/dataset"
hf download shubhampatle/Pixel-PIVR-Magnified-v2 \
  --repo-type dataset \
  --revision 591d53da15234597941394f2848740151209434b \
  --include "refwave_v6_xview_corrected_20260914/*" --local-dir "$DATA_ROOT"
export RECORDS_DIR="$DATA_ROOT/refwave_v6_xview_corrected_20260914"
export EVAL_RECORDS="$RECORDS_DIR/validation.jsonl"
```

Keep `WORK_ROOT`, `DATA_ROOT` and `RECORDS_DIR` exported in each training shell or
submission script. For validation inference, also keep `EVAL_RECORDS` exported;
`evaluate.sh` otherwise retains its historical v1 validation default.
`DATA_ROOT` contains `images/` and the versioned records directory; do not set it
to the records subfolder itself. Split membership is defined by the JSONL files,
not by all files in the reused image directories.
Set `MODEL_DIR` to existing base LA weights if they are outside the default
`$WORK_ROOT/LocateAnything-3B/`. This records-only command does not fetch weights
or extract images. On a new machine, run `bash download.sh` first.
No dataset or package audit is required by the training launcher.
**Start a fresh run for corrected records. Do not switch records while resuming
an old checkpoint.** Old weights still reflect old annotations and exposure.

### Historical LAE-COD Extension

The earlier `refwave_v2_cod/` extension was rolled back from the current dataset
branch and is not part of v6. [Its old guide](LAE_COD_UPDATE.md) is retained
for historical context, not as the current download recipe. The launcher's
unset `RECORDS_DIR` default remains `refwave_v1` for compatibility with old runs;
use the explicit export above for new corrected-data training.

Reserve roughly 250 GiB for LoRA or 450 GiB for full fine-tuning, including
archives, extracted images, environment, model and checkpoints. Storage estimates
are not GPU-memory estimates. Full fine-tuning also needs substantially more
host RAM for consolidated checkpoints; 256 GiB is a prudent starting request.

```text
RefWave/
  requirements.txt
  train_lora.sh              rank-32 Qwen LoRA + complete LM head
  train_full.sh              full MoonViT + projector + Qwen + head
  evaluate.sh                generated HBB evaluation, 1-8 GPU shards
  export.sh                  optional portable model export
  cluster.slurm              one-node SLURM example
  src/refwave/               self-contained implementation
refwave-assets/
  LocateAnything-3B/
  dataset/
    refwave_v6_xview_corrected_20260914/{stage1,stage2,validation}.jsonl
    refwave_v1/...             # preserved historical records
    images/{train,validation}/...
    archives/...
  runs/
    refwave_lora_8gpu_s1e2_s2e2_both_<UTC-timestamp>_<unique>/
      console.log
      stage1/
      stage2/
    refwave_full_8gpu_s1e2_s2e2_both_<UTC-timestamp>_<unique>/...
    standard_lora_4gpu_s1e2_s2e2_both_<UTC-timestamp>_<unique>/...
```

## 3. Train on an Allocated GPU Node

Choose one command. `8` can be any allocated GPU count from **1 to 8**.
These scripts support **one compute node**, not multi-node training.
They respect `CUDA_VISIBLE_DEVICES`; they never reserve GPUs or terminate jobs.

```bash
unset RUN_ROOT  # Fresh run; do not reuse an earlier exported output path.
bash train_lora.sh 8 both
```

Or full-parameter fine-tuning:

```bash
bash train_full.sh 8 both
```

Or the standard direct-box control on the same source episodes:

```bash
METHOD=standard bash train_lora.sh 8 both
# Or: METHOD=standard bash train_full.sh 8 both
```

Do not start all commands simultaneously on the same allocated GPUs.
Set the epoch count independently for each stage (both default to `1`):

```bash
STAGE1_EPOCHS=2 STAGE2_EPOCHS=3 WANDB_MODE=online bash train_lora.sh 8 both
```

This runs two passes over Stage-1 data, then initializes Stage 2 from
`stage1/best.pt` and runs three passes over Stage-2 data. The best checkpoint is
selected across all validation evaluations in the stage, not just the final
epoch. Stage 2 starts only after Stage 1 completes all requested epochs, with a
fresh optimizer and scheduler. The same epoch settings work with `train_full.sh`
and `METHOD=standard`. You can also change the defaults near the top of `train.sh`.

With `RUN_ROOT` unset, each invocation creates a new directory under
`$WORK_ROOT/runs/`, named by method, tuning mode, GPU count, stage epoch counts,
requested stages, UTC timestamp and a unique suffix. For example,
`refwave_full_4gpu_s1e2_s2e2_both_<timestamp>_<unique>` means RefWave full training
on four GPUs, with two epochs in each stage. `s1e` and `s2e` are configured epoch
counts; the next field (`both`, `stage1` or `stage2`) identifies which stages this
invocation runs. Extra Stage-2 training instead uses names such as
`refwave_lora_4gpu_stage2_extrae1_<timestamp>_<unique>`.
Repeated runs with identical settings also get separate folders. An explicit
`RUN_ROOT` is never renamed, so existing checkpoints and W&B identities remain usable.
The launcher prints `RUN_ROOT` and the console-log path before training starts.
Both stages of a `both` invocation stay inside that one run directory.

Optionally add your own label as the third argument to either training wrapper:

```bash
RUN_ROOT= CONTINUE_FROM= METHOD=standard \
STAGE1_EPOCHS=1 STAGE2_EPOCHS=1 WANDB_MODE=online \
bash train_full.sh 8 both ft_2
```

This creates `standard_full_8gpu_s1e1_s2e1_both_ft_2_<timestamp>_<unique>`.
W&B stage names and the group inherit this folder name. Labels may contain
1-64 letters, digits, dots, underscores or hyphens and must start with a letter
or digit. Reusing a label still creates a unique fresh folder. Labels only name
new folders; resuming with an explicit `RUN_ROOT` keeps the existing identity.

Terminal output is automatically saved to `<run>/console.log` while still being
displayed. There is no need to pipe the command through `tee` yourself. Each stage
also saves `training_curve.jsonl`, `validation_curve.jsonl`, `config.json` and its
best/last checkpoints. Resuming appends console output instead of replacing it.

To resume, explicitly use the run path printed by the earlier launch:

```bash
RUN_ROOT="/absolute/path/to/existing/run" bash train_lora.sh 8 both
```

Use `train_full.sh` for a full-parameter run and retain `METHOD=standard` for the
direct-box control. Use `stage1` or `stage2` instead of `both` to run one stage.
Starting `stage2` alone requires an explicit `RUN_ROOT` containing completed
Stage-1 **best** weights of the same method/tuning mode; its optimizer starts fresh.
An already-started Stage 2 resumes its own `stage2/last.pt`, preserving its original
parent even if the old launcher originally used Stage-1 last weights.
Keep epoch counts, world size, source records, seed, batch and optimization settings unchanged
for exact resume. An explicit `RUN_ROOT` opts into reuse of that directory;
unset it again before starting another fresh experiment.

If an older cluster copy fails on a fresh directory with `Existing run settings
differ` on one rank just after another rank prints the configuration, update
`src/refwave/train.py`. The earlier comparison could use a stale missing-config
snapshot while another rank published `config.json`. The current code rereads
the atomically published configuration before comparing it. Genuine mismatches
still stop training and now report the changed fields and saved/requested values;
do not delete a real run's configuration to bypass them. This is a startup fix,
not a change to data, losses, precision, optimizer or batch sizes.

### More Training After a Completed Stage 2

Do not increase epochs inside a completed run or delete its completion marker.
Use its best weights to initialize a **new Stage-2 continuation run**:

```bash
unset RUN_ROOT
CONTINUE_FROM="/absolute/path/to/previous/run/stage2/best.pt" \
STAGE2_EPOCHS=1 STAGE2_LR=5e-6 STAGE2_HEAD_LR=2.5e-7 \
BATCH_SIZE=8 GRAD_ACCUM=1 BATCH_TOKENS=16384 \
WANDB_MODE=online bash train_lora.sh 4 stage2
```

This example matches the previous four-GPU LoRA batch settings and uses lower
trial learning rates. Set `STAGE2_EPOCHS` to the number of **additional passes in
this new run**, not the combined epoch count across old and new runs. It skips
Stage 1 entirely, loads the Stage-2 model weights, resets optimizer/scheduler and
record progress, and saves to a new folder and W&B run. The old checkpoints are
untouched. Keep `done.json` beside the source checkpoint when transferring an
earlier best checkpoint from a completed stage. For Standard LoRA, also set
`METHOD=standard` and use that method's own Stage-2 checkpoint.

The new run has its own best/last checkpoint selection; compare its validation
performance with the preserved parent before choosing a final model. To resume
an interrupted continuation, use its printed `RUN_ROOT`, the same epoch/batch/LR
settings and `stage2`; `CONTINUE_FROM` can be omitted because `last.pt` preserves
the original initialization identity.

| Setting | LoRA | Full fine-tuning |
|---|---|---|
| Trainable | Rank-32 Qwen adapters + complete head | All model parameters |
| Physical episodes/GPU | 4 | 1 |
| Gradient accumulation | 1 | 4 |
| Global episodes/update on 8 GPUs | 32 | 32 |
| Stage-1 Qwen / head LR | 2e-5 / 1e-6 | 2e-6 / 1e-6 |
| Stage-2 Qwen / head LR | 1e-5 / 5e-7 | 1e-6 / 5e-7 |
| Vision / projector LR | Frozen | 1e-6 / 1e-5, both stages |
| Optimizer | AdamW | AdamW, PyTorch ZeRO-1 state sharding on >1 GPU |

Both use BF16 forwards, FP32 trainable parameters/optimizer state, gradient
checkpointing, 6,000 pre-merge image patches and a 32,768-token wave limit.
The normal 2x2 visual merge remains enabled; 6,000 is **not** the Qwen visual-token
count. Wave grouping targets 8,192 padded tokens; a larger indivisible wave runs
alone. A wave contains at most 200 branches, not at most 200 targets per image.
There is no silent target truncation and no 180-box cap.

`STAGE1_EPOCHS`, `STAGE2_EPOCHS`, `BATCH_SIZE`, `GRAD_ACCUM`, `BATCH_TOKENS`,
`WORKERS`, learning rates and `RUN_ROOT`
are explicit environment overrides. For a new memory-constrained LoRA run:

```bash
RUN_ROOT= BATCH_SIZE=1 GRAD_ACCUM=4 bash train_lora.sh 8 both
```

Full training keeps model parameters and optimizer states on GPU, without CPU
optimizer offload. No automatic OOM retry changes the recipe, drops targets or
restarts training with different settings.

### GPU-Only Vision Attention

Full training uses PyTorch's built-in fused SDPA for MoonViT. Packed images are
processed separately with four-dimensional attention inputs, preserving image
isolation without constructing the original quadratic mask. Qwen's branch and
PBD attention masks are unchanged. LoRA and standalone evaluation retain their
existing attention path.

This changes only the vision attention implementation: FP32 trainable weights,
GPU AdamW/ZeRO-1 states, BF16 forwards, existing gradient checkpointing, data,
losses, batch sizes and token limits are preserved. It needs no external
`flash-attn` or DeepSpeed package, no new dependencies and no launcher prechecks.
CUDA uses PyTorch's flash or memory-efficient backend; unsupported kernels fail
rather than silently returning to the high-memory math implementation. CPU
developer tests use the regular SDPA path.

From `RefWave/`, with your environment activated and `WORK_ROOT` set:

```bash
RUN_ROOT= WANDB_MODE=online bash train_full.sh 4 both
```

Replace `4` with your allocated GPU count (1-8). Start a new run directory: the
full-run configuration now records
`trainable.vision_attention = refwave_fused_per_image_sdpa`, so old full-run
configurations are not accepted for exact resume with this code. Checkpoint
tensor and optimizer formats are unchanged. New runs can resume as documented
above, using their original `RUN_ROOT`. LoRA run configurations are unchanged.

The previous GPU-only implementation failed during MoonViT backward on a
4 x 80-GB allocation at batch one. CPU output/gradient equivalence checks do not
establish that the updated full workload fits or is faster on that cluster;
CUDA peak memory and seconds per step still need measurement. Floating-point
rounding can differ between attention kernels.

### Data and Objective

| Split | Detection | Grounding | Episodes | HBB target occurrences |
|---|---:|---:|---:|---:|
| Stage 1 coarse | 62,633 | 53,881 | 116,514 | 238,637 |
| Stage 2 dense + replay | 23,438 | 16,480 | 39,918 | 1,619,799 |
| Validation | 397 | 350 | 747 | 84,384 |

One scheduled pass per stage by default, configurable independently. Every epoch
reshuffles the records deterministically and retains its final partial batch.
On 8 GPUs at global batch 32: **3,642 Stage-1 updates per epoch**, then
**1,248 Stage-2 updates per epoch**. Total updates are the respective per-epoch
count multiplied by `STAGE1_EPOCHS` or `STAGE2_EPOCHS`. The training curve and W&B
also log fractional epoch progress. `config.json` records the unique scheduled
record count; `records_seen` includes repeated visits across epochs.
Stage 2 includes 10,952 replay episodes. Episodes are queries, not unique images;
replay intentionally repeats some Stage-1 data. No standalone pointing records.

RefWave loss per episode is `0.5 * R1 token mean + 0.5 * R2 token mean`;
updates average episodes across all GPUs. Standard uses direct answers only.
Equal source exposure does not imply equal decoder compute. GT points are used
for supervised training; evaluation uses **predicted** points only.

Round-2 example (global normalized coordinates, no repeated class output):

```text
Human: <image>
Locate the single ship containing point <box><200><300></box> in horizontal box format. Return None if absent.
GPT: <box><150><240><250><360></box>
Absent: <box>None</box>
```

Validation loss and saving run every 300 optimizer updates and at each epoch end.
`best.pt` minimizes task-balanced detection/grounding validation loss, **not F1**.
`last.pt` resumes training. The trainer keeps those two checkpoints per stage.
An ordinary interrupt requests an aligned checkpoint after the current update;
SIGKILL/power loss can lose work since the previous save. Full fine-tuning retains
visual gradients from all waves before one vision backward, with checkpoint
recomputation; it is not the frozen-vision LoRA code with a changed flag alone.

Resume repairs only an unfinished final entry in a training log or prediction
shard, with a warning; corruption in a completed entry is an error. If an
interrupt saved the final update before validation, re-running the same command
finishes validation and best-checkpoint selection before marking the stage done.
Loss-curve learning rates are the rates used for that update, not the next update.

### W&B (Optional)

From the `RefWave/` repository, after activating your Conda or venv environment:

```bash
source scripts/common.sh
"$PYTHON_BIN" -m pip install -r requirements-tracking.txt
"$PYTHON_BIN" -m wandb login

# A fresh experiment with a new output folder and new W&B runs.
unset RUN_ROOT
WANDB_MODE=online bash train_lora.sh 8 both
```

Setup and training use the same Python selected by `scripts/common.sh`, including
an active Conda environment. No `.venv/` is required for Conda users.
Replace `8` with your allocated GPU count, or use `train_full.sh` for full tuning.

Each fresh experiment has its own W&B group, named after the unique `RUN_ROOT`
folder. Stage 1 and Stage 2 appear as **two separate runs within that group**, for
example `refwave_lora_8gpu_s1e2_s2e2_both_<timestamp>_<unique>_stage1` and `..._stage2`.
Only GPU rank 0 logs, so using eight GPUs does not create eight W&B runs per stage.
To resume, set the original `RUN_ROOT` and keep the same `WANDB_PROJECT` and W&B
account/team; the stage's run ID is reused. Do **not** unset `RUN_ROOT` when resuming.

No W&B login is required by default. Training and validation losses are always
written to `training_curve.jsonl` and `validation_curve.jsonl` in each stage.
With tracking enabled, W&B also receives `train/*` and `validation/*` loss metrics.
Training metrics are committed after every optimizer update, with charts using
`optimizer_step` as their X-axis. Validation metrics appear only after validation
finishes (every 300 updates by default, plus stage end). Terminal loss lines print
at step 1 and every 10 steps by default; silence between them is not a W&B failure.
Its local files are under `$RUN_ROOT/stage1/wandb/` and `$RUN_ROOT/stage2/wandb/`.
An offline compute node can use `WANDB_MODE=offline`; those runs are stored locally
until synced from an internet-connected machine.

`wandb login` authenticates your account but does not enable tracking in the
trainer. The startup log reports `W&B tracking mode: online`, `offline`, or
`disabled`. Set `WANDB_MODE=online` on the **same command line** as the launcher,
or use `export WANDB_MODE=online` before launching. A non-exported assignment on
a separate line is not passed to the launcher in a fresh shell. Online mode also
needs network access from the compute node; login-node connectivity is not enough.

## 5. SLURM

Install/download once on an internet-enabled node. Submit from this repository:

```bash
export WORK_ROOT=/scratch/$USER/refwave-assets
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT cluster.slurm
```

Change account/partition/GPU resource syntax to your site's rules. The supplied
example requests one node, 8 GPUs, 16 CPU cores, 128 GiB RAM and 48 hours; its time
limit is **not** a training ETA. For full fine-tuning, request more host RAM:

```bash
sbatch --mem=256G --export=ALL,TUNING=full cluster.slurm
```

For 2 GPUs: `sbatch --gres=gpu:2 --export=ALL,GPUS=2 cluster.slurm` plus your
account/partition arguments. For Conda, export `PYTHON_BIN` before submission.
An offline worker must see the downloaded assets and the same installed
environment. No tmux is needed. The job writes `refwave-JOBID.log`.
Training additionally writes `console.log` in its unique run directory, whose
path is printed in the SLURM log. Unset `RUN_ROOT` before a fresh submission;
export the existing run path before submitting a resume or Stage-2-only job.

## Full-Parameter Ablation Controls

For standard multi-category direct boxes and class-isolated direct boxes on the
same records, see [Direct-box ablations](DIRECT_BOX_ABLATIONS.md).
Select `METHOD=standard` or `METHOD=class_isolated` with `train_lora.sh` or
`train_full.sh`. For a matched comparison against RefWave, use the same wrapper,
record files, epochs, batch settings and validation-selection policy for both.
The existing RefWave default is unchanged; no new dataset download is required.
