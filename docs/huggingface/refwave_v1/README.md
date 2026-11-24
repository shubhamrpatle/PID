# RefWave Full-Scale HBB Records

These records use the existing images in `shubhampatle/Pixel-PIVR-Magnified-v2`.
They implement **global-only RefWave**, not pixel-crop magnification. The dataset's
historical name is retained to reuse its image archives.

## Counts

| Split | Detection episodes | Grounding episodes | Total episodes | HBB target occurrences | Unique images |
|---|---:|---:|---:|---:|---:|
| Stage 1 coarse | 62,633 | 53,881 | 116,514 | 238,637 | 88,803 |
| Stage 2 dense + retained replay | 23,438 | 16,480 | 39,918 | 1,619,799 | 36,497 |
| Validation | 397 | 350 | 747 | 84,384 | 747 |

Training has 156,432 scheduled episodes across both stages and 106,354 unique
image hashes in their union. Stages intentionally share some images/queries.
An episode is a source query, not a unique image or an individual box.
Stage 2 retains 10,952 replay episodes (27.44% after filtering); it is not a
freshly sampled 20% replay mixture. Maximum GT count is 19 in Stage 1 and
1,942 in Stage 2. There is no 180-box cap, no 400-target pilot cap and no
cropping/re-encoding of uploaded images in this conversion.

Detection-only GT distribution:

| GT boxes/query | Stage 1 | Stage 2 | Validation |
|---|---:|---:|---:|
| 0 | 6,271 | 0 | 50 |
| 1-19 | 56,362 | 5,887 | 61 |
| 20-99 | 0 | 13,696 | 39 |
| 100-499 | 0 | 3,460 | 204 |
| 500-999 | 0 | 334 | 43 |
| 1,000+ | 0 | 61 | 0 |

## Exclusions and Scope

| Reason | Stage 1 | Stage 2 | Validation |
|---|---:|---:|---:|
| Standalone pointing records | 7,493 | 10,580 | 250 |
| Known benchmark image hashes | 8,580 | 2,198 | 0 |
| Same-reference quantized point collisions | 69 | 204 | 3 |

Collision exclusions remove **whole source queries**, not selected boxes from an
otherwise positive query. They account for 514, 22,386 and 1,414 target occurrences,
respectively. Those targets are not claimed as trained. Overlapping boxes with
distinct point addresses are allowed. `excluded.jsonl` lists collision exclusions.

The train/validation and train/known-benchmark image-hash intersections are zero.
This statement concerns this fine-tuning recipe and the supplied HBB benchmark
holdout ledger, not LocateAnything's unknown pretraining exposure, unlisted
benchmarks, or geographic/parent-scene independence.

## One Episode, Two Routes

A JSONL line contains:
- `image`: a relative path such as `images/train/fd/<sha256>.jpg`;
- `direct.conversations`: unchanged Type-2 direct answer, used only by the control;
- `round1.branches`: every queried class/phrase, including absent references;
- `round2.branches`: one positive HBB target for every retained GT occurrence,
  plus verified negative point-address branches;
- `target_groups`: complete normalized HBB targets and point centers;
- source IDs, content hash, task, source dataset, stage and replay provenance.

Example Round 1 for a detection branch:

```text
Human: <image>
Point to all instances that match the following categories: ship.

GPT: <box><200><300></box><box><600><700></box>
```

For grounding, the input is `<image>\nPoint to: {phrase}.`
The stored Round-1 conversation includes a reference tag for provenance;
the **cluster loader removes it from the supervised output** after validating
it against the structured points. Class/phrase identity is already in the input.
An absent reference returns `<box>None</box>`.

Example Round 2:

```text
Human: <image>
Locate the single ship containing point <box><200><300></box> in horizontal box format. Return None if absent.

GPT: <box><150><240><250><360></box>
```

All coordinates are integer-normalized to [0,1000], in the global image frame.
Round 1 uses GT centers. Round-2 positives use deterministic interior addresses:
requested center or offsets at 0.25, 0.50 or 0.75 of the box half-size, avoiding
same-reference address/center collisions. An interior point need not be the center.
This is supervised address conditioning, not on-policy generated-point training.
At inference, GT is unavailable; Round 2 must use predicted points.

Round 1 retains every queried absent reference. Round-2 negatives are selected
globally per split/task to approximately 10% of refinement branches, while every
all-negative episode retains supervision. A negative address lies outside every
annotated target of its queried reference; wrong-class points undergo that same
check. No negative category is inferred merely from an unrelated source vocabulary.

| Split | R1 branches | R2 positive branches | R2 negative branches |
|---|---:|---:|---:|
| Stage 1 | 467,249 | 238,637 | 26,515 |
| Stage 2 | 139,572 | 1,619,799 | 179,978 |
| Validation | 5,494 | 84,384 | 9,376 |

RefWave loss is the mean over source episodes of
`0.5 * R1 supervised-token mean + 0.5 * R2 supervised-token mean`.
The direct answer does not contribute. The standard control uses only the direct
answer on the same episodes. R2 has exactly six supervised MTP positions per
box/None branch after native LA tokenization.

## Training and Provenance

Use the standalone [RefWave repository](https://github.com/shubhamrpatle/RefWave),
not the older Pixel-PIVR cluster launcher. Its README gives venv and Conda
installation commands with pinned dependencies.

After installing that environment, run from the cloned `RefWave/` directory:

```bash
export WORK_ROOT=/scratch/$USER/refwave-assets
bash download.sh
bash train_full.sh 8 both
```

The downloader places records in `$WORK_ROOT/dataset/refwave_v1/`, extracts
existing images into `$WORK_ROOT/dataset/images/`, and places base LA weights in
`$WORK_ROOT/LocateAnything-3B/`. Keep the same `WORK_ROOT` in later shells/jobs.
For previously extracted images, set `DATA_ROOT` to their parent directory and
download only the added annotations:

```bash
export DATA_ROOT=/path/to/existing/dataset
hf download shubhampatle/Pixel-PIVR-Magnified-v2 \
  --repo-type dataset --include "refwave_v1/*" --local-dir "$DATA_ROOT"
```

This records-only update does not download model weights or image archives.
`DATA_ROOT` must contain both `images/` and `refwave_v1/`; a new machine should
use `download.sh` instead. Image filenames in the episodes are relative to
`DATA_ROOT`, not to `refwave_v1/`.

Full fine-tuning defaults to one episode/GPU with accumulation four, giving the
following scheduled update counts for one pass (1-8 GPUs on one node):

| GPUs | Global episode batch | Stage 1 updates | Stage 2 updates |
|---|---:|---:|---:|
| 1 | 4 | 29,129 | 9,980 |
| 2 | 8 | 14,565 | 4,990 |
| 4 | 16 | 7,283 | 2,495 |
| 8 | 32 | 3,642 | 1,248 |

Stage 2 starts from completed Stage-1 last weights with a fresh optimizer.
Validation loss and checkpoints are recorded every 300 updates and at the end.

The last partial batch is retained. A dense source episode may require multiple
forward/backward waves per update. All GT is retained in the published episodes;
an indivisible over-budget sequence fails explicitly at runtime.

Source schedules:
- Stage 1: Type-2 v6 `coverage/stage1_coarse_seed152.jsonl`;
- Stage 2: Type-2 v8 balanced `coverage/stage2_type2_balanced_seed353.jsonl`;
- Validation: `locany_rs_unified_validation1000_v1/all_tasks_validation1000.jsonl`.

`manifest.json` records exact source/file hashes, source-dataset breakdowns and
filter counts. `verification.json` records the exhaustive structural check of
all published episodes. Image archives are unchanged; no new image uploads are
required. The original datasets' licensing and use conditions still apply.
