---
pretty_name: Pixel-PIVR and RefWave Remote-Sensing HBB
license: other
task_categories:
- object-detection
- visual-question-answering
language:
- en
tags:
- remote-sensing
- visual-grounding
- pointing
- locate-anything
- pixel-pivr
- refwave
---

# Remote-Sensing HBB Data: RefWave and Pixel-PIVR

## Current RefWave Training

**Use [`refwave_v1/`](https://huggingface.co/datasets/shubhampatle/Pixel-PIVR-Magnified-v2/tree/main/refwave_v1)
with the standalone [RefWave code](https://github.com/shubhamrpatle/RefWave).**
The repository name is historical: these new records implement **global-only
RefWave**, not magnified pixel re-entry. The original image archives are reused;
there is no need to download the older expanded Pixel-PIVR annotation files.

| File | Detection episodes | Grounding episodes | Total episodes | HBB target occurrences |
|---|---:|---:|---:|---:|
| `refwave_v1/stage1.jsonl` | 62,633 | 53,881 | 116,514 | 238,637 |
| `refwave_v1/stage2.jsonl` | 23,438 | 16,480 | 39,918 | 1,619,799 |
| `refwave_v1/validation.jsonl` | 397 | 350 | 747 | 84,384 |

Each episode stores both class/phrase-to-points and point-conditioned HBB
supervision. Stage 1 is coarse; Stage 2 contains dense examples and 10,952 retained
replay episodes. Standalone pointing and OBB records are not included.
Validation is for checkpoint selection, not a full benchmark result.

After cloning RefWave and installing its environment following its README,
run from the `RefWave/` directory:

```bash
export WORK_ROOT=/scratch/$USER/refwave-assets
bash download.sh
bash train_full.sh 8 both
```

Use only GPUs allocated to your job. The launcher accepts 1-8 GPUs on one node.
The downloader fetches pinned model weights, RefWave records, and existing
train/validation image archives, then extracts the images. No manual file moving
or old Pixel-PIVR setup is needed.

```text
$WORK_ROOT/
  LocateAnything-3B/
  dataset/
    refwave_v1/stage1.jsonl
    refwave_v1/stage2.jsonl
    refwave_v1/validation.jsonl
    refwave_v1/manifest.json
    images/train/...
    images/validation/...
    archives/...
  runs/...
```

If you already have the original images extracted under a dataset directory,
only the small relative-to-images annotation update is needed (about 1.29 GB):

```bash
export DATA_ROOT=/path/to/existing/dataset
hf download shubhampatle/Pixel-PIVR-Magnified-v2 \
  --repo-type dataset --include "refwave_v1/*" --local-dir "$DATA_ROOT"
```

`DATA_ROOT` must contain both `images/` and `refwave_v1/`. Retain the same
`DATA_ROOT` when launching training. This records-only command does not download
the base model or extract missing images; use `download.sh` for a new machine.
See the [RefWave dataset guide](refwave_v1/README.md) for templates, exclusions,
target counts, and file hashes in its manifest.

## Legacy Pixel-PIVR 144-to-384 Dataset

The remaining sections describe the earlier magnified Pixel-PIVR release, not
the standalone RefWave setup above. Old annotations and images are preserved.

This is the versioned `pixel-pivr-hf-hbb-magnified-v2` package. It is the **real pixel re-encoding**
variant, not the older cached pre-projector ROI variant.

Round 1 encodes the global image and predicts target-complete point addresses.
Round 2 first encodes the same global image plus a deterministic 144 x 144
source-pixel crop resized to 384 x 384 with Lanczos, then predicts exactly one box
or `<box>None</box>`. There are 2,171,892 local-gate rows. A local `None`,
invalid box, or crop-edge box triggers one of 320,576 paired global
fallback rows. These include both large positive targets and negative addresses,
so inference never selects a route using ground truth. No target or box is clipped.

LocateAnything's unchanged image processor subsequently aligns 384 x 384 to
392 x 392 (28 x 28 MoonViT patches, or 14 x 14 = 196 projected visual tokens).

| Split | Records | Unique images | Image bytes |
|---|---:|---:|---:|
| Stage 1 coarse | 460,263 | shared train pool | - |
| Stage 2 dense + replay | 2,100,576 | shared train pool | - |
| Validation | 105,767 | 1,000 | 0.54 GiB |
| Checkpoint-selection monitor | 2,141 | same validation pool | - |
| Test | benchmark protocols | 16,712 | 7.22 GiB |

The train, validation, and test image SHA-256 sets have zero overlap. Stage 1 and
Stage 2 may intentionally reuse training images. Use the recipes under `recipes/`;
do not construct splits from directory names.

Stage 2 contains 1,960,076 dense optimizer records and
140,500 replay optimizer records
(6.69% replay after point-to-box expansion).
Before expansion, its Round-1 source-query schedule contains
39,477 dense and
11,204 replay records
(22.11% replay). These percentages
describe different units and must not be interchanged.

## Integrity

The original `SHA256SUMS` describes the legacy release's annotations, recipes,
manifests, documentation snapshot, and images; it does not describe this updated
card or the added `refwave_v1/` release. RefWave annotation SHA-256 values are in
`refwave_v1/manifest.json`. `IMAGE_INVENTORY.jsonl` provides a machine-readable image inventory.
The training repository preflight additionally verifies this schema, crop contract,
record counts, GPU topology, sequence budget, and the pinned Eagle compatibility
patch before training can start.

## Download

```bash
hf download shubhampatle/Pixel-PIVR-Magnified-v2   --repo-type dataset --local-dir Pixel-PIVR-Magnified-v2
```

The full operator guide is in the Pixel-PIVR code repository under
`docs/A100_8GPU_FULL_SCALE_RUNBOOK.md`.

## License note

This is a derived research compilation. Every source dataset retains its own terms.
Review the source licenses before redistribution or commercial use. No model weights
are included.


## Archive transport

Images are transported in deterministic uncompressed tar shards so the Hub
repository remains well below its recommended 100,000-file ceiling. The JSONL
paths still point to `images/...`. Run the code repository bootstrap command;
it validates every archive, extracts images atomically, and verifies every image
SHA-256 before training is unlocked. Do not train directly from the tar files.
