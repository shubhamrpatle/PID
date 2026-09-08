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
