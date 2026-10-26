# Verification Scope

Checked on 2026-09-08. These are developer checks, not steps that a cluster user
must run. No user-facing launcher invokes pytest, an audit, a manifest scan or a
training smoke test.

## Completed

- Initial review: **92 tests passed, 2 warnings, 137.66 seconds**, in a clean virtual
  environment, with public-model decoding, export, upstream source and
  representative-record checks enabled. CUDA was hidden for every test; no
  existing GPU workload was interrupted.
- Final review: **104 tests passed, 3 warnings, 129.99 seconds**, using the same
  environment and pinned public-model code. Two warnings come from upstream LA/
  Transformers; the third is the expected unfinished-log recovery warning in its
  regression test. See the review findings below for the added regression scope.
- Built the final package wheel with no extra dependencies and imported training,
  evaluation and export from a separate `/tmp` installation in isolated Python.
  Imports of legacy Pixel-PIVR, Eagle, DeepSpeed and Liger were blocked. The first
  probe incorrectly treated Transformers' optional-package availability lookup
  as an import; the corrected probe reports those packages unavailable while
  still rejecting real imports, and passed. The wheel is a maintainer check,
  not a replacement for the README's Git clone/setup workflow.
- Python 3.10.12, torch 2.5.1+cu121, torchvision 0.20.1+cu121,
  transformers 4.57.1 and peft 0.12.0. All pinned direct dependencies match the
  installed versions in the test environment. Installation followed the README:
  CUDA-12.1 PyTorch wheels, `requirements.txt`, then `pip install --no-deps -e .`.
  Pytest was installed separately for these developer tests only.
- `pip check` reported the upstream Decord 0.6.0 wheel's platform-metadata warning
  and no dependency-version conflicts. Its wheel file declares a CPython-3.6 tag
  despite its Python-3 wheel filename. Decord imports and all image-only checks
  passed on Python 3.10. This warning is documented in README.md and is not a
  launcher prerequisite; package metadata was not modified to conceal it.
- Shell syntax passed for every `.sh` file. Launcher tests cover 1, 2, 4 and 8
  processes, LoRA/full modes, paths with spaces, independent evaluation GPU count,
  `DATA_ROOT` overrides and stopping the pipeline after failed Stage 1.
- Fresh subprocess imports of training, evaluation and export succeed while
  legacy Pixel-PIVR, Eagle, DeepSpeed and Liger imports are explicitly blocked.
- The vendored native MTP target-label method matches the AST of Eagle commit
  `8442db3b79f7fd2357e468e6eecdd9b6a82049ff`. Nine message-format cases also match
  that upstream image/text message construction.
- Twenty-eight representative compilations across both stages and both methods
  match previously measured image-token, wave, sequence and branch counts. These
  include the 1,942-target dense record. This is not a fresh audit of every record.
- Exact distributed episode coverage, final partial-batch normalization, padded
  wave loss/gradients and route balancing pass unit tests.
- Actual LA Qwen code, with a small CPU configuration, passes shared-prefix
  class-wave and geometry-wave equivalence checks. This verifies masks/cache
  behavior, not a GPU-generated benchmark-score equivalence result.
- Full-model CPU wrapper backward reaches vision, projector, Qwen and output
  head through the per-episode visual-gradient bridge. Two CPU distributed ranks
  pass native ZeRO optimizer consolidation and checkpoint restoration tests.
- The actual released 3B model passes LoRA plus modified LM-head checkpoint
  export/reload: the head change survives and adapters are merged in the export.
- Fresh model Python/configuration/tokenizer files were downloaded from LA
  revision `c32291ca5e996f5a7a485845b4f57a233936bba0`. Both reused local weight
  shards match the SHA256 values published by Hugging Face. These checks do not
  rely on the locally patched research model code.
- The real 3B model completes native direct decoding, a two-class point wave
  and two-address geometry decoding on a synthetic image, with one shared vision
  encoding. This CPU-only API smoke test uses FP32 because the local AVX2 CPU
  lacks native BF16 arithmetic. The GPU recipe remains BF16. Synthetic addresses
  exercise execution only; they provide no evidence of localization accuracy.
- The decoding smoke test caught and fixed result serialization: box-only mode
  now returns only `<box>...</box>` or `<box>None</box>`. The input class stays in
  the separate `label` field instead of being prepended to the saved answer.
  Four regression cases cover present/absent targets and both prefix modes.
- Evaluation query extraction does not depend on stored GT coordinates or
  answers. Duplicate predictions are counted as false positives in scorer tests;
  pooled metrics are recomputed from counts.
- Hugging Face's pinned data revision is reachable and lists the three record
  files, 17 training-image archives and one validation-image archive. Its manifest
  reports 116,514 Stage-1, 39,918 Stage-2 and 747 validation episodes. The pinned
  base-model revision is also reachable. Archives were not downloaded again.

## Not Verified Here

- Installation on the destination cluster, or creation of a Conda environment.
  A fresh Python-3.10 venv installation was completed on the local Linux host.
- A100 GPU execution, NCCL behavior on that cluster, peak GPU memory, throughput
  or an end-to-end full-parameter training run. CPU full-gradient tests cannot
  establish that an arbitrary device allocation fits.
- Newly generated validation or benchmark scores from this standalone package.
  Its default generated evaluation is the 747-query validation set, not the
  historical Balanced-100 set or a full official benchmark.
- Multi-node training. The supported scope is one allocated node with 1-8 GPUs.
- Cross-round Qwen KV reuse. Projected vision is reused across rounds; Qwen
  prefixes are shared within each round and rebuilt between rounds.

## GPU-Only Full Training Restored (2026-09-08)

The cluster log reported CUDA OOM inside MoonViT SDPA during the first vision
backward. Rank 3 reported 78.07 GiB in use on a 79.25-GiB GPU, with a failed
1.79-GiB allocation. The run used batch one, accumulation four, 6,000 image
patches and a 32,768-token wave limit. W&B initialization succeeded.

At the user's request, the subsequent CPU-offload memory fix was reverted before
cluster deployment. Full training again uses GPU FP32 model parameters, BF16
forwards, native GPU AdamW (ZeRO-1 optimizer-state sharding on multiple GPUs), and
the original whole-vision checkpointing. The CPU-master optimizer, per-block
vision patch and optimizer-sidecar format were removed. This restores the earlier
memory requirements; it is not a fix for the recorded OOM.

Conda interpreter selection, unique run folders, console logs and W&B fixes remain
in place. Dataset files, loss definitions, token limits and active jobs were not
changed. No additional launcher verification or dependencies were introduced.

After the revert, the local suite reported **117 passed, 5 skipped and 1 expected
log-recovery warning in 14.96 seconds**, with CUDA hidden and the actual LA Qwen
source tests enabled. The skipped tests require optional assets/runtime settings.
`git diff --check` passed. No GPU training or cluster fit test was performed.

## Fused MoonViT Attention Only (2026-09-08)

After restoring GPU-only training, the user approved an attention-only memory
optimization. Full training now selects `refwave_fused_per_image_sdpa` on its
MoonViT block instances. It presents contiguous four-dimensional Q/K/V tensors
to native PyTorch SDPA, handles each packed image separately and allows only
flash/memory-efficient CUDA kernels. It does not modify the module's original
SDPA entry, Qwen masks, LoRA path, model precision, optimizer, objective or existing
checkpoint recomputation. There is no CPU optimizer offload or external kernel
dependency. The implementation is recorded in each new full run's configuration;
the previous full-run configuration requires a fresh output directory.

Developer tests compare masked-reference outputs and Q/K/V gradients, verify
packed-image isolation, and exercise the actual small LA MoonViT encoder with
and without whole-encoder checkpointing. Training-loop tests check full-only
activation, FP32 AdamW state and resumed completion. Optional CUDA tests cover
BF16 fused forward/backward but are not invoked by any launcher.

All local GPUs had allocated workloads, so CUDA was hidden during verification.
Cluster CUDA execution, full-size peak memory, training speed and final model
accuracy have not been measured for this patch. No running job was modified.

The final CPU suite reported **127 passed, 7 skipped and 1 expected log-recovery
warning in 17.41 seconds**. Five skips require optional assets/runtime settings;
two require a free CUDA GPU. The small actual MoonViT/Qwen source tests ran.
`git diff --check` passed. Runtime transfer requires both `src/refwave/train.py`
and the new `src/refwave/vision_attention.py`; launchers and requirements did not
change. The source was not pushed or transferred by the assistant.

## Final Code Review Findings

The review began from local commit `9a9d237`. The following fixes are local
changes for the user to publish, not changes to any existing training run:

- **Multi-rank stop alignment:** the old signal handler could change the local
  exit decision during validation/checkpointing, after ranks had synchronized.
  The handler now only records a request. Stop decisions are synchronized before
  and after validation. A two-process CPU distributed test verifies that a
  request on one rank reaches both ranks. This does not test NCCL on A100.
- **Final-update completion:** reproduced a final-step interrupt followed by a
  resume that wrote `done.json` without performing final validation or creating
  `best.pt`. Resume now completes pending validation without another optimizer
  update. It also recovers a missing/stale best pointer when the last checkpoint
  records the best loss, and removes validation entries that were never committed
  to a checkpoint.
- **Torn log recovery:** reproduced a JSON parsing error when an interrupted
  final append existed despite a valid checkpoint. Training and evaluation now
  recover only an unfinished final entry, with a warning, and reject corruption
  in completed/interior entries. An already complete final JSON entry lacking a
  newline is retained and separated from the next append. Dataset files are not
  repaired or modified by this logic.
- **RNG isolation:** branch initialization previously called `torch.manual_seed`,
  which also reseeds CUDA generators before their state was captured. Branches
  now initialize private CPU/device generators. The direct-generation control
  uses the same scoped RNG helper. DataLoader startup also uses its own generator
  rather than advancing the training RNG on restart or validation.
- **Learning-rate plots:** the old row logged the scheduler's next rate after
  an optimizer update. Rows now log the rates actually used by that update.
  The learning-rate schedule itself and the loss objective were not changed.

Tests exercise the real training loop with a tiny CPU stand-in, actual checkpoint
files, interrupted/resumed execution, final validation, best-pointer recovery,
log repair and learning-rate logging. Model/gradient/decoding checks above remain
separate from those control-flow tests. No dataset, architecture, optimizer
hyperparameter or target format was changed by these fixes. No test or audit was
added to setup/download/train/evaluation launchers.

The assistant has not pushed these review changes to GitHub. Original datasets,
checkpoints, running jobs and the Pixel-PIVR checkout were left unchanged.

## Multi-Rank Startup Config Race (2026-09-10)

The supplied cluster log stopped on rank 1 at the settings comparison before
the first optimizer update. Rank 0 had already printed the new configuration.
The two-epoch implementation cached an absent config as `{}` before hashing
inputs, then checked file existence again later. If rank 0 published config.json
between those operations, another rank compared its stale empty snapshot with
the real settings and incorrectly rejected the fresh run.

A deterministic regression reproduces that exact exception by publishing a
matching config during input hashing. The fix reads the persisted config at the
time of comparison. Genuine differences still fail and now name the changed
fields. Empty existing configs are not silently treated as fresh runs. No
architecture, objective, hyperparameters, epoch schedule, GPU allocation,
dependencies or launcher verification steps changed.

Developer tests cover the publication interleaving for RefWave LoRA, RefWave
full fine-tuning and Standard LoRA at one and two epochs, plus mismatch rejection
without overwriting config/checkpoint files. The focused suite passed 91 tests
with 4 optional skips. The full CPU suite passed **198 tests, 13 skipped, one
expected log-recovery warning, in 10.66 seconds**. The first full-suite invocation
omitted PYTHONPATH for the distributed test subprocesses; rerunning with the
same `src` path exported by the launcher resolved that test environment issue.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" \
  python -m pytest -q
```

These tests use tiny CPU training stand-ins and CPU distributed optimizer tests;
they do not establish A100/NCCL execution or full-model memory fit. The remote
cluster run was not restarted here. Only `src/refwave/train.py` needs transferring
for this runtime fix; tests and this note are optional maintainer files.

## Four-Process Training Recheck (2026-09-11)

The complete CPU suite now passes **215 tests, 13 skipped**, with one expected
log-recovery warning, in 25.03 seconds. Bash syntax checks passed for the training,
evaluation, export, shared environment and SLURM launchers. Python compilation and
active-Conda interpreter selection passed too.

New maintainer tests execute the actual training loop on four Gloo processes with
tiny substitute models and records, for RefWave LoRA, Standard LoRA and RefWave
full mode. They deliberately delay rank 1 until rank 0 publishes config.json,
request a coordinated stop, resume, complete two epochs with an uneven tail,
initialize Stage 2 from Stage-1 best weights, and continue from Stage-2 best weights.
All ranks' combined records are counted to verify exact per-epoch coverage.
Full mode uses the real sharded AdamW optimizer, consolidation and restoration.
The test harness uses a separate rendezvous for each stage, matching production's
separate torchrun invocations. Reusing a Gloo rendezvous across stages failed in
the initial test harness; no production change was needed for that test issue.

No additional production blocker was found during this recheck. Tests are not
invoked by any user launcher. The production updates remain `train.sh` (descriptive
run names) and `src/refwave/train.py` (config-race fix and mismatch diagnostics).
This does not establish actual A100/NCCL compatibility, full-model peak memory,
cluster filesystem capacity, or successful online W&B connectivity. No remote
training, dataset modification, package installation or GPU job was launched.

## Public Release Alignment (2026-10-26)

The package source was aligned to the exact code that produced the paper's
results. Each evaluation run on the A100 server froze the SHA-256 of every source
file it executed into its `plan.json`. Three independent runs recorded the same 22
files for `src/refwave/`: the final rotary-stabilized PID evaluation, the paired
sequential/packed efficiency run, and the class-isolated evaluation.
`src/refwave/` now matches all 22 hashes, with no extra and no missing file.

Five files had changed after those evaluations to add an optional causal extent
control: `data.py`, `episode_model.py`, `evaluate.py`, `model.py` and `train.py`.
They were restored to the evaluated versions. The causal control itself was never
trained, produced no checkpoint, and appears in no reported result, so its modules,
launcher, tests and documentation were removed from the release. That work is
preserved on the local branch `backup/pre-release-causal-20261026`.

Training provenance was checked against saved run configurations. The PID,
`standard` and `class_isolated` runs each record two epochs per stage, world size
eight, seed 20260908, and training/validation hashes whose prefixes match the
paper's Table 7. The frozen DIOR, DOTAv2, DIOR-RSVG and VRSBench-VG manifests also
match Table 7. The run directory naming of all three runs is produced only by the
current `train.sh`; the `refwave_causal_extent` option added to it later was
removed. No evaluation hash covers shell scripts, so this naming evidence is the
basis for the launcher version, not a byte-level record.

The complete CPU suite passes **260 tests, 16 skipped**, including the four-process
distributed trainer for RefWave LoRA and full, Standard LoRA, and class-isolated
LoRA and full. Run with the package importable, as `setup.sh` arranges through an
editable install:

```bash
PYTHONPATH=src python -m pytest -q
```

No GPU job, training run or dataset change was performed for this alignment.
