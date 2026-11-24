# Corrected RefWave v6 data

Use this version for fresh runs, with the existing RefWave trainer and image store.
No new training-time audit, dependency or model change is required.

## Exact Release

- Dataset: `shubhampatle/Pixel-PIVR-Magnified-v2`
- Immutable revision: `591d53da15234597941394f2848740151209434b`
- Records directory: `refwave_v6_xview_corrected_20260914`
- Parent: `refwave_v1` at `8d0bd1cdc2aa835c09fb2ecb3ced93f9af991cf6`
- No rolled-back COD or grounding-balanced additions are included.

```bash
export WORK_ROOT=/your/persistent/path/refwave-assets
export DATA_ROOT="$WORK_ROOT/dataset"
hf download shubhampatle/Pixel-PIVR-Magnified-v2 \
  --repo-type dataset --revision 591d53da15234597941394f2848740151209434b \
  --include "refwave_v6_xview_corrected_20260914/*" --local-dir "$DATA_ROOT"
export RECORDS_DIR="$DATA_ROOT/refwave_v6_xview_corrected_20260914"
export EVAL_RECORDS="$RECORDS_DIR/validation.jsonl"
```

Keep these exports in the cluster submission script. Existing images must remain
under `$DATA_ROOT/images/`. On a new machine, bootstrap model and images with
`bash download.sh` first, then download the corrected records above. Downloading
the correction does not replace or delete old files.

From the `RefWave/` code checkout, with the configured environment active:

```bash
unset RUN_ROOT CONTINUE_FROM
WANDB_MODE=online bash train_full.sh 8 both xview_corrected
```

Use `METHOD=standard` for the matched direct-box control. Choose one job for the allocated GPUs, not all simultaneously.
Use 1-8 allocated GPUs. Both training methods use the same corrected records.
Validation inference needs the explicit `EVAL_RECORDS` setting above; benchmark
evaluation should explicitly select the intended untouched benchmark instead.

## What Changed

| Split | Before episodes | After episodes | Before HBB targets | After HBB targets |
|---|---:|---:|---:|---:|
| Stage 1 | 116,514 | 116,386 | 238,637 | 238,475 |
| Stage 2 | 39,918 | 39,851 | 1,619,799 | 1,619,629 |
| Validation | 747 | 740 | 84,384 | 84,377 |

The xView category table used IDs 0-59 while annotations used 1-60. Only that
source's category table was aligned to 1-60, verified against the source taxonomy.
Other datasets and annotation coordinates were not shifted. Corrected labels
were used to regenerate direct boxes, derived points, both RefWave rounds and
negative queries/addresses. There are 12,900 regenerated episodes; 144,077 retained
episodes are byte-identical. Dense target lists remain uncapped (maximum 1,942).

There are 202 quarantined episodes: 72 ambiguous/contradicted xView grounding
episodes and 130 exact benchmark-overlap episodes. The latter comprise 123
training episodes on 36 images and 7 validation episodes on 7 images. Image
files stay in the old store, but are not referenced by the new training/validation
records. No replacement phrases or samples were invented.

All 65 historical unscoped pointing cases were reviewed: their original point
sets are preserved, and none feed this release. Region-scoped point lists were
not treated as errors merely for differing from full-image targets.

## Evidence and Limits

The release directory contains per-dataset/task/stage counts, SHA256 hashes,
every record correction/quarantine, benchmark matches, and offline verification.
See `CORRECTION_REPORT.md`, `verification.json`, `changes.jsonl`,
`counts_by_dataset_task_stage.csv`, and `VISUAL_REVIEW.md`.

Source consistency is not exhaustive semantic certification. In particular,
no independent original xView GeoJSON per-object join was available; negatives
are verified against available annotations, which may be incomplete. Exact-byte
and decoded-RGB split checks do not establish geographic/near-duplicate separation.
The remaining validation grounding is DIOR-RSVG only: the five VRSBench queries
were benchmark overlaps and were removed, not replaced to preserve balance.

Start from the intended base model in a fresh run. Corrected files cannot undo
old checkpoint exposure to incorrect labels or the newly identified benchmark
overlaps; historical image-unseen claims require reassessment.
