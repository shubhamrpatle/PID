# PID: Parallel Instance Decoding

Training and evaluation code for **Parallel Instance Decoding (PID)**, a unified
generative localization framework for language-guided localization in aerial
imagery. PID separates instance identification from spatial localization: given a
textual reference, it first generates a variable-length set of point addresses,
one per matching instance, then predicts the bounding boxes for all identified
instances in a single parallel decoding pass.

## Naming

The internal method identifier in this codebase is `refwave`, the project's
working name. It is kept deliberately: `method: refwave` is recorded in the saved
`config.json` of every trained checkpoint, and the Hugging Face dataset release
used for training is published under a `refwave_` directory name. Renaming it
would break loading and provenance for those existing artifacts. Read `refwave`
as PID throughout.

| Paper name | `METHOD=` value |
|---|---|
| PID (ours) | `refwave` |
| LA-3B (FT), joint direct-box control | `standard` |
| LA-3B (FT, class-iso.), class-isolated control | `class_isolated` |

## Correspondence with the paper

The paper identifies its data by SHA-256 prefix in Table 7. The training release
downloaded in step 2 and the frozen evaluation manifests reproduce every prefix:

| File | Records | SHA-256 prefix |
|---|---:|---|
| `stage1.jsonl` | 116,386 | `2fd6c5e8df29` |
| `stage2.jsonl` | 39,851 | `c5d0e3156f2c` |
| `validation.jsonl` | 740 | `00e14c17087b` |
| DIOR manifest | 586 | `44b493d6e497` |
| DOTAv2 manifest | 874 | `ada8bbdc3ff1` |
| DIOR-RSVG manifest | 7,500 | `11443dbd68a1` |
| VRSBench-VG manifest | 16,159 | `c96aeb3d8aad` |

The reported PID model was trained on eight GPUs, two epochs per stage, seed
`20260908`, with the corrected release from step 2. `train.sh` defaults to the
older `refwave_v1` records, so set `RECORDS_DIR` explicitly:

```bash
export RECORDS_DIR="$DATA_ROOT/refwave_v6_xview_corrected_20260914"
METHOD=refwave STAGE1_EPOCHS=2 STAGE2_EPOCHS=2 bash train_full.sh 8 both
```

The two direct-box controls use the same command with `METHOD=standard` or
`METHOD=class_isolated`. The saved configuration of the reported PID run records
`method: refwave`, 2 and 2 epochs, world size 8, seed 20260908, and the three
training hashes above.

`src/refwave/` is byte-identical to the 22 source files whose SHA-256 hashes were
frozen into the plans of the evaluation runs that produced the paper's detection,
efficiency and ablation results.

## Evaluation

[`paper_evaluation/`](paper_evaluation/README.md) holds the code that scored the
paper's results under the frozen protocol: detection, grounding and pointing, the
paired sequential/packed efficiency comparison, the class-isolated control, and the
address ablation. Every source file there is byte-identical to the version that ran.
`evaluate.sh` in this directory generates predictions from a trained checkpoint;
`paper_evaluation/` is what produced the reported numbers.

## Repository notes

A standalone training repository for HBB detection and phrase grounding.
It does **not** import or require the old Pixel-PIVR checkout, Eagle trainer,
DeepSpeed, FlashAttention builds, crop experiments, audit reports or verification
scripts. Required LA token-label code is included with attribution.

One source query is an episode: class/phrase-to-points, then point-to-one-box.
The image is global-only; no magnified local crops are used in this release.

To back up completed cluster runs, see [Upload Runs](docs/UPLOAD_RUNS.md).
The standalone `upload_cluster_runs.sh` uploads retained checkpoints and results
to a public Hugging Face repository without changing the training environment.

## License

Apache-2.0, see `LICENSE`. Third-party attribution is in `THIRD_PARTY.md`;
`src/refwave/native_mtp.py` derives from NVlabs/Eagle under Apache-2.0. Model
weights and datasets retain their own separate licenses and use conditions.

## 1. Get the Code and Create an Environment

Use Linux x86-64 with glibc 2.28 or newer, Python 3.10 or 3.11, and a
CUDA-12.1-compatible NVIDIA driver.
Clone this standalone repository, not the old Pixel-PIVR repository:

```bash
git clone https://github.com/shubhamrpatle/PID.git
cd PID
```

Run the following commands from the cloned `PID/` directory. The code and
Python environment stay here; downloaded images, model weights and training
outputs go in the separate `WORK_ROOT` directory selected in step 2. If the
repository is private, your GitHub account must have access before cloning.

Choose **one** environment method. No test or data-audit step is required.

### Venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
export PYTHON_BIN="$(command -v python)"
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Alternatively, `WORK_ROOT=/scratch/$USER/refwave-assets bash setup.sh` performs
these installations and remembers the storage path in the ignored `.work-root`.
Set `PYTHON_BOOTSTRAP=python3.11` if that is the cluster's available interpreter.
After using `setup.sh`, activate `.venv` and set `PYTHON_BIN` as above too.
Do not reuse the old Pixel-PIVR virtual environment.

### Conda

```bash
conda create -n refwave python=3.10 -y
conda activate refwave
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
export PYTHON_BIN="$(command -v python)"
```

Activate your environment in each new shell or submission environment. Launchers
use an explicit `PYTHON_BIN` first, then an active venv or Conda environment, then
`RefWave/.venv/bin/python` if present, and finally Python on `PATH`. The explicit
export above also works with older checkouts. Conda users do not need a `.venv/`.
Dependencies are pinned to the LA model's APIs. Do not replace PyTorch or
Transformers with arbitrary newer versions in this environment.

The upstream Decord 0.6.0 wheel has an old Python tag in its metadata, so a
manually run `pip check` can report `decord 0.6.0 is not supported on this platform`
([upstream issue](https://github.com/dmlc/decord/issues/366)). Decord imports on the
tested Python 3.10 environment; these image-only launchers do not use video
decoding and do not run that check as a prerequisite.

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
See [the correction guide](docs/XVIEW_CORRECTED_V6.md) for counts and limitations.
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
branch and is not part of v6. [Its old guide](docs/LAE_COD_UPDATE.md) is retained
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

## 4. Evaluate Generated Predictions

Evaluation GPU count is independent of training GPU count. Default: all **747
validation queries**, without GT points/GT boxes passed to the model, with no NMS
or max-detections cap. This is validation evaluation, **not a full benchmark**.

```bash
RUN_ROOT="/absolute/path/to/the/refwave/run"  # Printed when training starts.
bash evaluate.sh 4 "$RUN_ROOT/stage2/best.pt"

METHOD=standard bash evaluate.sh 4 "/absolute/path/to/the/standard/run/stage2/best.pt"
bash evaluate.sh 4 baseline
```

Use the corresponding `refwave_full_...` run for a full fine-tuned checkpoint. The loader restores
the full trained state, including the output head; it does not use a LoRA-only
loader for a full checkpoint.

```bash
bash evaluate.sh 4 "/absolute/path/to/the/full/run/stage2/best.pt"
```

For independent sequential decoding rather than class/address waves:

```bash
CLASS_WAVE=1 BOX_WAVE=1 EVAL_OUTPUT="$RUN_ROOT/evaluation_sequential" \
  bash evaluate.sh 4 "$RUN_ROOT/stage2/best.pt"
```

RefWave defaults to class waves of 18 and box waves of 200. It reuses projected
global vision across rounds and shares Qwen prefixes within each round; it does
**not** claim cross-round Qwen KV reuse. Each Round-2 address generates one PBD6
box/None. Direct-box control uses the native LA generator. Existing decoding
rules are retained; these should not be presented as identical output grammars.

Round 2 now defaults to `ROUND2_EXECUTION=fused_one_pass`: each wave combines
its address prompts and six-position geometry inputs into **one Qwen forward**.
`BOX_WAVE=1` also uses one forward per address. Prefix preparation and Round 1
are separate work; this does not mean one forward for the entire image.
No retraining is needed. Training, prompts, point filtering and scoring are unchanged.
BF16 fusion can slightly change coordinates or `None` decisions, so historical
two-pass results must not be relabeled as one-pass results.

For the explicit two-pass reference only:

```bash
ROUND2_EXECUTION=two_pass bash evaluate.sh 4 "$RUN_ROOT/stage2/best.pt"
```

RefWave outputs default to `evaluation_fused_one_pass/` alongside the checkpoint
(`evaluation_two_pass/` for the reference). `EVAL_OUTPUT` overrides this directory.
Standard/baseline controls retain their existing output directory and decoding.
Outputs: `predictions.rank*.jsonl`, `metrics.json`, `config.json` and
`done.json` alongside the checkpoint. Each shard is resumable with identical
settings. Detection uses class-aware greedy one-to-one IoU@0.5 P/R/F1. Grounding
additionally reports first-box Acc@0.5, Acc@0.7 and mean IoU on single-GT queries.
Latency is per query excluding model loading and scoring; raw responses are kept.
First-query warmup is included; these are recorded timings, not a controlled
latency benchmark. No AP/mAP or statistically reliable gain is claimed.

`EVAL_LIMIT=100` selects the first 100 records; it is **not** the historical
DOTAv2 Balanced-100 set. To use another frozen evaluation set, supply
`EVAL_RECORDS=/path/to/episodes.jsonl` and its `DATA_ROOT`, retaining the same
episode schema. Detection benchmarks must list all protocol-requested classes,
including absent ones; never build the query list from GT-present labels.

Optional portable model export:

```bash
bash export.sh "$RUN_ROOT/stage2/best.pt" "$WORK_ROOT/exports/refwave_stage2"
```

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

## Verification Scope

Developer tests are kept in `tests/`, never run automatically by the user-facing
scripts. The release preserves the tested RefWave label/loss construction and
includes CPU tests of masks, gradient accumulation, exact coverage and export.
Full fine-tuning, fresh A100 execution and generated-score equivalence are not
guaranteed solely by CPU tests. See `VERIFICATION.md` for checks actually run.
The tests are for maintainers; they are not part of environment setup, data
download or the training/evaluation workflow.
# Full-parameter ablation controls

For standard multi-category direct boxes and class-isolated direct boxes on the
same records, see [Direct-box ablations](docs/DIRECT_BOX_ABLATIONS.md).
Select `METHOD=standard` or `METHOD=class_isolated` with `train_lora.sh` or
`train_full.sh`. For a matched comparison against RefWave, use the same wrapper,
record files, epochs, batch settings and validation-selection policy for both.
The existing RefWave default is unchanged; no new dataset download is required.
