# Add LAE-COD to Stage 1

This optional release uses `refwave_v2_cod/` in the existing
`shubhampatle/Pixel-PIVR-Magnified-v2` dataset repository. The default downloader
remains pinned to the original `refwave_v1/` release. Publish the new release
before running the incremental download below.

Only `train.sh` changes on the cluster: `RECORDS_DIR` selects the annotation
directory. The Python trainer, environment, loss, image processor, and model
architecture do not change. No dataset verifier runs before training.

| Split | Detection | Grounding | Total episodes | Annotated HBB targets |
|---|---:|---:|---:|---:|
| Original Stage 1 | 62,633 | 53,881 | 116,514 | 238,637 |
| Added COD | 23,454 | 0 | 23,454 | 71,079 |
| New Stage 1 | 86,087 | 53,881 | 139,968 | 309,716 |
| Stage 2, unchanged | 23,438 | 16,480 | 39,918 | 1,619,799 |
| Validation, unchanged | 397 | 350 | 747 | 84,384 |

Added episodes by source: AID 2,568; EMS 569; NWPU-RESISC45 20,284; SLM 33.
Of 36,910 source image records, 13,456 were excluded: 13,164 ambiguous
same-reference discovery centers, 283 records at the 20-target density boundary,
6 duplicate images, and 3 same-reference quantized-center collisions. Exclusions
remove whole records. Stage-1 detection Round-2 negatives remain 20,775, now
7.45% of detection refinement branches after adding positive COD targets.

The incremental image archive is 2,043,822,080 bytes (1.90 GiB); its extracted
images occupy another 2,025,797,292 bytes. Allow about 6 GiB of additional space
for the archive, extracted images, annotation release, and download overhead.

## Dataset Policy

- Original Stage-1 lines remain unchanged; accepted COD episodes are appended.
- Stage 2 and validation remain byte-for-byte identical to `refwave_v1`.
- New COD records are detection episodes, not fabricated phrase-grounding data.
- Each accepted image keeps every supplied annotation. Images with 20 or more
  annotations stay outside Stage 1; no boxes are trimmed to meet this threshold.
- Duplicate and held-out images are excluded. Geometry/address ambiguities
  quarantine the whole image record, with reasons in `cod_excluded.jsonl`.
- COD labels are automatic and non-exhaustive. New COD records therefore contain
  positive supervision only. Unlisted classes and background are not treated as
  verified negatives. All existing RefWave negative examples remain unchanged,
  with `<box>None</box>` and the same tokenization. The combined negative fraction
  is reported in the manifest; it is not artificially held at 10%.
- Label normalization changes only case and whitespace, not semantic meaning.
  Plurals, synonyms, attributes, and broad region labels are not silently merged.
- Point discovery uses the same GT-center policy. Both the discovery center and
  the Round-2 address must identify one annotated box of that reference. An
  ambiguous offset falls back to its unambiguous center; inherently ambiguous
  source records are excluded.

The unchanged episode schema is `caw-pivr-unified-hbb-episode-v2`. Each episode
has `direct`, `round1`, `round2`, and `target_groups`. RefWave trains only the two
rounds; the matched standard control uses `direct` on the same records.
Round-1 stored reference tags are removed by the existing geometry-only loader.
Round 2 retains the existing prompt and emits one box, without a reference tag:

```text
Human: <image>
Locate the single ship containing point <box><200><300></box> in horizontal box format. Return None if absent.
GPT: <box><150><240><250><360></box>
```

Coordinates are normalized to [0,1000] in the original image. No image cropping,
resizing, pixel re-encoding, or magnified re-entry is added by this release.

## Existing Cluster: Download Only the Update

First replace the cluster's `RefWave/train.sh` with the updated file, or update
your checkout after publishing it. Keep all existing assets.

Run from `RefWave/` in the existing activated environment:

```bash
export WORK_ROOT=$WORK_ROOT
export DATA_ROOT="$WORK_ROOT/dataset"

hf download shubhampatle/Pixel-PIVR-Magnified-v2 \
  --repo-type dataset \
  --include "refwave_v2_cod/*" "archives/images-train-laecod-v2-*.tar" \
  --local-dir "$DATA_ROOT"

for archive in "$DATA_ROOT"/archives/images-train-laecod-v2-*.tar; do
  tar -xf "$archive" -C "$DATA_ROOT"
done

export RECORDS_DIR="$DATA_ROOT/refwave_v2_cod"
```

For a frozen experiment, add `--revision <upload-commit-SHA>` to `hf download`.
The new archive names are distinct from the original image archives. Existing
images do not need to be transferred again. The release includes unchanged
Stage-2/validation files so both stages resolve within one annotation directory.

Expected paths:

```text
refwave-assets/
  LocateAnything-3B/                 existing model, unchanged
  dataset/
    refwave_v1/                     original release, retained
    refwave_v2_cod/
      stage1.jsonl                  original + accepted COD episodes
      stage2.jsonl                  unchanged
      validation.jsonl              unchanged
      manifest.json
      cod_additions.jsonl
      cod_excluded.jsonl
      image_inventory_additions.jsonl
    images/train/...                existing plus new hashed image files
    images/validation/...           unchanged
    archives/images-train-laecod-v2-*.tar
```

On a new machine, follow the main README and run `bash download.sh` first to
obtain the base model, original records, and original images. Then apply this
incremental download. Do not download only the update onto an empty machine.

## Start a Fresh Run

Use an allocated GPU node; choose 1-8 GPUs on that node. These commands do not
reserve GPUs or stop other jobs. Use one of the following commands, not both
simultaneously on the same GPU allocation:

```bash
unset RUN_ROOT CONTINUE_FROM
export RECORDS_DIR="$DATA_ROOT/refwave_v2_cod"
export STAGE1_EPOCHS=1 STAGE2_EPOCHS=1

# RefWave
WANDB_MODE=online bash train_full.sh 8 both

# Alternatively: matched standard direct-box control
METHOD=standard WANDB_MODE=online bash train_full.sh 8 both
```

The launcher derives coverage from the new file length and selected epoch count;
there is no fixed old Stage-1 step cap. Stage 2 starts from Stage-1 `best.pt`,
selected by validation loss, with a fresh optimizer. It uses the unchanged dense
and replay records. Do not replace an active run's records or resume a v1 run
with the expanded Stage-1 file. Keep `RECORDS_DIR` exported when resuming a new
v2 run; the console log prints it in the resume command.

Validation and benchmark evaluation sets are unchanged. Existing evaluation
commands remain valid. To point explicitly at the copied validation file, use
`EVAL_RECORDS="$RECORDS_DIR/validation.jsonl"` with `evaluate.sh`.

## Source and Limits

Source: [official LAE-DINO/LAE-1M repository](https://github.com/jaychempan/LAE-DINO).
LAE-COD contains automatically labeled coarse regions. Structural consistency
and hash separation do not prove semantic annotation correctness, exhaustive
instance coverage, or geographic independence. Existing dataset licenses and
source attribution requirements continue to apply. OPT-RSVG is not added.

The release manifest contains exact before/after counts, per-source exclusions,
annotation hashes, new archive hashes, and the measured negative proportions.
`verification.json` and `loader_check.json` are offline preparation results,
not cluster training prerequisites.
