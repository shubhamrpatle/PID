# Paper evaluation

The evaluation code that produced the paper's detection, grounding, pointing,
efficiency and ablation results. Every source file here is byte-identical to the
version that ran; see [Provenance](#provenance) for how that was checked.

Training lives at the repository root. This directory scores trained checkpoints
under the frozen protocol described in the paper's Appendix C.

## Packages

Directory names are kept exactly as they ran, because the packages locate one
another through these sibling paths.

| Package | What it evaluates |
|---|---|
| `LA-Rex-Protocol-Eval-v3/` | The frozen annotated-category protocol, LA-native parser and class-aware scorer, and the comparison models: Qwen3-VL-4B/8B, Rex-Omni-3B, LocateAnything-3B, the finetuned joint direct control, Grounding DINO and LAE-DINO. Specification in `PROTOCOL.md`. |
| `PID-Address-RoPE-Eval/` | Final PID detection with the FP32 rotary-cache correction, and the predicted-versus-annotation-derived address comparison. |
| `PID-GT-Efficiency-Eval/` | Paired sequential versus packed extent decoding on the same checkpoint, image and GPU. |
| `Final-Class-Isolated-Eval/` | The class-isolated direct-box control. |
| `Final-RefWave-Eval/` | The PID evaluation runtime used by the class-isolated package, and `runtime_hashes.json`. |
| `Final-Controls-Eval/workspace/` | The shared runtime these packages import. `RefWave/src` links to the repository's `src/`. |
| `model_patches/LocateAnything-3B/` | Three patches to the released LocateAnything-3B model code. |

## Setup

Run everything from this directory, on Linux with an NVIDIA GPU.

### 1. Environment

The reported measurements used one NVIDIA A100 PCIe 40 GB per worker with:

| Package | Version |
|---|---|
| Python | 3.10 |
| torch | 2.11.0+cu128 |
| transformers | 4.57.1 |
| torchvision | 0.26.0+cu128 |
| tokenizers | 0.22.0 |
| accelerate | 1.5.2 |
| numpy | 1.26.4 |
| Pillow | 12.2.0 |

Timing on other GPUs or software is not comparable to the paper's tables. LAE-DINO
needs its own environment; see `LA-Rex-Protocol-Eval-v3/setup_laedino.sh` and
`requirements-laedino-constraints.txt`.

### 2. Model runtime

The evaluations ran on `nvidia/LocateAnything-3B` with three source files patched:
`generate_utils.py`, `modeling_locateanything.py` and `processing_locateanything.py`.
The patches add point decoding and PBD block scoring to the model, scoring helpers,
and crop support in the processor. NVIDIA's files are not redistributed here; the
script below fetches the released model and applies the patches:

```bash
python prepare_model_runtime.py --source "$WORK_ROOT/LocateAnything-3B"
```

`--source` reuses the copy that the repository's `download.sh` already fetched.
Omit it to download. The script pins revision
`c32291ca5e996f5a7a485845b4f57a233936bba0`, the same revision the training
downloader uses. It links the weight shards rather than copying them, and it stops
unless all 21 model runtime files match `Final-RefWave-Eval/runtime_hashes.json`.

### 3. Released scorer

`LA-Rex-Protocol-Eval-v3/prepare.py` downloads two files from NVlabs/Eagle at commit
`783f656d127ee498137b5ff52603ce36c292d317`: the released LocateAnything scorer
`other_metric.py` and `inference_grounding_ddp.py`. It verifies each by SHA-256 and
refuses a mismatch. They are not redistributed here because their file headers
reserve distribution rights to NVIDIA.

### 4. Frozen evaluation manifests

The evaluation reads frozen manifests for DIOR, DOTAv2, DIOR-RSVG and VRSBench-VG
from `Final-Controls-Eval/original_manifests/`. They are not included. Each is
derived from its source benchmark, which carries its own licence and terms of use.

A manifest stores absolute image paths, so its file hash depends on where the images
live. The paper's Table 7 lists the hashes of the manifests used for the reported
results, with image paths on the evaluation server:

| Manifest | Records | SHA-256 prefix in Table 7 |
|---|---:|---|
| DIOR | 586 | `44b493d6e497` |
| DOTAv2 | 874 | `ada8bbdc3ff1` |
| DIOR-RSVG | 7,500 | `11443dbd68a1` |
| VRSBench-VG | 16,159 | `c96aeb3d8aad` |

Relocating image paths changes these file hashes without changing images or
annotations. The packages check annotation content separately, through the
semantic hash computed by `common.semantic_rows`, which excludes `image_path`.

### 5. Run

Each package has a `run.sh` that schedules fixed shards over idle GPUs, resumes
completed work, and refuses to mix results from a changed plan. Run
`bash <package>/run.sh --help` for its arguments.

## Provenance

Every evaluation run froze the SHA-256 of each source file it executed into its
`plan.json`. Those records were compared against this directory:

| Check | Result |
|---|---|
| Package source files recorded by the four run plans | 28 match, 0 differ |
| Runtime files enforced by `runtime_hashes.json` | 158 match, 0 differ |
| Model runtime rebuilt from upstream plus patches | 21 of 21 exact |

The runtime includes research modules from the project's development, such as the
`zero_shot_analysis/*_pbd.py` family. They are present because the evaluation's own
integrity check requires every file in `runtime_hashes.json`. They are not all
executed on the PID path.

Launchers, tests and `PID-GT-Efficiency-Eval/summarize_saved.py` are not recorded by
the run plans, which hash executed sources only. `summarize_saved.py` recovered the
efficiency report from saved predictions without new inference.

## Tests

Developer tests sit beside each package. With the scorer downloaded by step 3 and
the model runtime from step 2, these pass without any benchmark data:

| Package | Result |
|---|---|
| `LA-Rex-Protocol-Eval-v3` | 50 passed |
| `PID-GT-Efficiency-Eval` | 25 passed |

The remaining packages also check against artifacts that are not shipped: the frozen
manifests from step 4, the class-isolated run's saved configurations, and archived
evaluation outputs. Those tests fail with `FileNotFoundError` until the artifacts are
supplied. The class-isolated run's configurations and training curves are published
in the Hugging Face dataset `CaseStudyRef/RefWave-Cluster-Runs`.
