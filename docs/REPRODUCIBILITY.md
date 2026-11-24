# Reproducibility

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
downloaded in step 2 of [Training](TRAINING.md#2-download-the-model-and-data) and the frozen evaluation manifests reproduce every prefix:

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

## Repository notes

A standalone training repository for HBB detection and phrase grounding.
It does **not** import or require the old Pixel-PIVR checkout, Eagle trainer,
DeepSpeed, FlashAttention builds, crop experiments, audit reports or verification
scripts. Required LA token-label code is included with attribution.

One source query is an episode: class/phrase-to-points, then point-to-one-box.
The image is global-only; no magnified local crops are used in this release.

To back up completed cluster runs, see [Upload Runs](UPLOAD_RUNS.md).
The standalone `upload_cluster_runs.sh` uploads retained checkpoints and results
to a public Hugging Face repository without changing the training environment.

## Verification Scope

Developer tests are kept in `tests/`, never run automatically by the user-facing
scripts. The release preserves the tested RefWave label/loss construction and
includes CPU tests of masks, gradient accumulation, exact coverage and export.
Full fine-tuning, fresh A100 execution and generated-score equivalence are not
guaranteed solely by CPU tests. See [`VERIFICATION.md`](../VERIFICATION.md) for checks actually run.
The tests are for maintainers; they are not part of environment setup, data
download or the training/evaluation workflow.
