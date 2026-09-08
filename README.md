# RefWave

A standalone training repository for HBB detection and phrase grounding.
It does **not** import or require the old Pixel-PIVR checkout, Eagle trainer,
DeepSpeed, FlashAttention builds, crop experiments, audit reports or verification
scripts. Required LA token-label code is included with attribution.

One source query is an episode: class/phrase-to-points, then point-to-one-box.
The image is global-only; no magnified local crops are used in this release.

## 1. Get the Code and Create an Environment

Use Linux x86-64 with glibc 2.28 or newer, Python 3.10 or 3.11, and a
CUDA-12.1-compatible NVIDIA driver.
Clone this standalone repository, not the old Pixel-PIVR repository:

```bash
git clone https://github.com/shubhamrpatle/RefWave.git
cd RefWave
```

Run the following commands from the cloned `RefWave/` directory. The code and
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

For Conda, set `PYTHON_BIN` in each new shell or submission environment after
activation. The shell launchers otherwise use `RefWave/.venv/bin/python`.
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

The downloader fetches released [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)
and the `refwave_v1/` records in
[Pixel-PIVR-Magnified-v2](https://huggingface.co/datasets/shubhampatle/Pixel-PIVR-Magnified-v2),
at pinned revisions.
The historical dataset name does not mean this is a magnified-crop experiment.
Existing train/validation image archives are reused, and downloads are resumable.
No Hugging Face login is needed for public downloads. Run this step on an
internet-enabled machine before requesting offline compute nodes.

Existing assets can be reused with `MODEL_DIR` and `DATA_ROOT`; the latter must
contain `refwave_v1/` and `images/`. No dataset audit is run by the launcher.
You do not need to download, rename or move the individual files manually.
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
    refwave_v1/{stage1,stage2,validation}.jsonl
    images/{train,validation}/...
    archives/...
  runs/
    refwave_lora/{stage1,stage2}/
    refwave_full/{stage1,stage2}/
    standard_lora/{stage1,stage2}/
```

## 3. Train on an Allocated GPU Node

Choose one command. `8` can be any allocated GPU count from **1 to 8**.
These scripts support **one compute node**, not multi-node training.
They respect `CUDA_VISIBLE_DEVICES`; they never reserve GPUs or terminate jobs.

```bash
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
Use `stage1` or `stage2` instead of `both` to run a single stage.
Stage 2 requires completed Stage-1 **last** weights of the same method/tuning
mode and starts a fresh optimizer. Re-run the identical command to resume.
Keep world size, source records, seed, batch and optimization settings unchanged
for exact resume. New settings require a new `RUN_ROOT`.

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

`BATCH_SIZE`, `GRAD_ACCUM`, `BATCH_TOKENS`, `WORKERS`, learning rates and `RUN_ROOT`
are explicit environment overrides. For a new memory-constrained LoRA run:

```bash
BATCH_SIZE=1 GRAD_ACCUM=4 RUN_ROOT="$WORK_ROOT/runs/refwave_lora_b1" bash train_lora.sh 8 both
```

An 80-GB A100 run has **not** been benchmarked here. Full fine-tuning on a single
small GPU may not fit even at batch one. No automatic OOM retry changes the
recipe, drops targets or restarts training with different settings.

### Data and Objective

| Split | Detection | Grounding | Episodes | HBB target occurrences |
|---|---:|---:|---:|---:|
| Stage 1 coarse | 62,633 | 53,881 | 116,514 | 238,637 |
| Stage 2 dense + replay | 23,438 | 16,480 | 39,918 | 1,619,799 |
| Validation | 397 | 350 | 747 | 84,384 |

One scheduled pass per stage, with the final partial batch retained. On 8 GPUs
at global batch 32: **3,642 Stage-1 updates**, then **1,248 Stage-2 updates**.
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

Validation loss and saving run every 300 optimizer updates and at stage end.
`best.pt` minimizes task-balanced detection/grounding validation loss, **not F1**.
`last.pt` resumes training. The trainer keeps those two checkpoints per stage.
An ordinary interrupt requests an aligned checkpoint after the current update;
SIGKILL/power loss can lose work since the previous save. Full fine-tuning retains
visual gradients from all waves before one vision backward, with checkpoint
recomputation; it is not the frozen-vision LoRA code with a changed flag alone.

### W&B (Optional)

```bash
"${PYTHON_BIN:-.venv/bin/python}" -m pip install -r requirements-tracking.txt
"${PYTHON_BIN:-.venv/bin/python}" -m wandb login
WANDB_MODE=online bash train_lora.sh 8 both
```

No W&B login is required by default. Training and validation losses are always
written to `training_curve.jsonl` and `validation_curve.jsonl` in each stage.

## 4. Evaluate Generated Predictions

Evaluation GPU count is independent of training GPU count. Default: all **747
validation queries**, without GT points/GT boxes passed to the model, with no NMS
or max-detections cap. This is validation evaluation, **not a full benchmark**.

```bash
bash evaluate.sh 4 "$WORK_ROOT/runs/refwave_lora/stage2/best.pt"
METHOD=standard bash evaluate.sh 4 "$WORK_ROOT/runs/standard_lora/stage2/best.pt"
bash evaluate.sh 4 baseline
```

Use the `refwave_full` path for a full fine-tuned checkpoint. The loader restores
the full trained state, including the output head; it does not use a LoRA-only
loader for a full checkpoint.

```bash
bash evaluate.sh 4 "$WORK_ROOT/runs/refwave_full/stage2/best.pt"
```

For independent sequential decoding rather than class/address waves:

```bash
CLASS_WAVE=1 BOX_WAVE=1 EVAL_OUTPUT="$WORK_ROOT/evaluation/refwave_sequential" \
  bash evaluate.sh 4 "$WORK_ROOT/runs/refwave_lora/stage2/best.pt"
```

RefWave defaults to class waves of 18 and box waves of 200. It reuses projected
global vision across rounds and shares Qwen prefixes within each round; it does
**not** claim cross-round Qwen KV reuse. Each Round-2 address generates one PBD6
box/None. Direct-box control uses the native LA generator. Existing decoding
rules are retained; these should not be presented as identical output grammars.

Outputs: `evaluation/predictions.rank*.jsonl`, `metrics.json`, `config.json` and
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
bash export.sh "$WORK_ROOT/runs/refwave_lora/stage2/best.pt" "$WORK_ROOT/exports/refwave_stage2"
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

## Verification Scope

Developer tests are kept in `tests/`, never run automatically by the user-facing
scripts. The release preserves the tested RefWave label/loss construction and
includes CPU tests of masks, gradient accumulation, exact coverage and export.
Full fine-tuning, fresh A100 execution and generated-score equivalence are not
guaranteed solely by CPU tests. See `VERIFICATION.md` for checks actually run.
The tests are for maintainers; they are not part of environment setup, data
download or the training/evaluation workflow.
