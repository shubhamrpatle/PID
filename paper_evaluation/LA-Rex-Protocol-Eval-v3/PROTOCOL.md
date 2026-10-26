# Frozen evaluation contract

Identifier: `rs_gt_categories_la_native_v3`.

## Evidence and adaptation

This is a remote-sensing adaptation, not a reproduction of COCO, VisDrone,
Dense200 or the Molmo benchmark. The sources are:

- [Rex-Omni paper, Sections 5.1, 5.3 and 5.6](https://arxiv.org/html/2510.12798v1).
- [LocateAnything paper, Section 4](https://arxiv.org/html/2605.27365v1).
- [Pinned LA dense/pointing scorer](https://github.com/NVlabs/Eagle/blob/783f656d127ee498137b5ff52603ce36c292d317/Embodied/evaluation/metrics/other_metric.py).
- [Official Rex implementation](https://github.com/IDEA-Research/Rex-Omni/tree/6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e).
- [LAE-DINO code and inference configuration](https://github.com/jaychempan/LAE-DINO/tree/6b1519626e39d1f39f8ed1f38761c20f7e0e8c35).
- [Grounding DINO-Swin-T HF checkpoint](https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/a2bb814dd30d776dcf7e30523b00659f4f141c71).
- [Grounding DINO native phrase-to-class API](https://github.com/IDEA-Research/GroundingDINO/blob/856dde20aee659246248e20734ef9ba5214f5e44/groundingdino/util/inference.py), `Model.phrases2classes`.

Rex's general detection setup conditions open-set models on annotated categories
and distinguishes single-category from joint VLM querying. Its COCO experiment
sweeps detector confidence; its dense table instead reports Grounding DINO at
0.25. LA reports the same dense operating point. This package adopts the latter,
not an undisclosed test-selected threshold. Qwen3 and the RS benchmarks are
adaptations beyond Rex's original model/benchmark set.

The released LA dense scorer averages image precision and recall before forming
F1. Its box matcher flattens categories, whereas its pointing matcher checks
category identity. These implementation details must not be conflated with the
paper's general per-category wording. This package retains the released dense
compatibility result and also applies that same box matcher separately by class
for its primary class-aware result. This is an explicit correction/adaptation,
not a claim that the released dense matcher was class-aware.

The user-selected main parser for LA-family outputs is the
[pinned LA inference parser](https://github.com/NVlabs/Eagle/blob/783f656d127ee498137b5ff52603ce36c292d317/Embodied/evaluation/inference_grounding_ddp.py)
with SHA-256 `0a08b029a2633b6293eb0428a806c08041a352e497a917fea84d32d6472c40ac`.
Its four parsing/conversion functions are executed unchanged, without loading
the inference program or its model. They sort reversed box corners and clamp
coordinates to width/height minus one, including accepted numeric values above
the nominal 0-1000 range. Strict no-repair parsing is a separately reported
sensitivity on identical raw responses. This policy was selected explicitly,
not by choosing whichever test score is higher. Other models keep their native
coordinate formats; LA parsing is not applied to Rex or Qwen token syntax.
The primary class-aware matcher remains an explicit RS adaptation. Therefore
native LA parsing still does not imply reproduction of every paper setting.

## Data and annotation boundary

| Dataset | Images | GT HBBs | Vocabulary | GT-present category queries |
|---|---:|---:|---:|---:|
| DIOR | 586 | 3,379 | 20 | 861 |
| DOTAv2 | 874 | 29,329 | 18 | 1,618 |

Records, GT order, query aliases and image content come from the completed final
controls manifests. Only image paths may be relocated. SHA-256 checks compare
every image with the original transfer inventory. No image, box, class or hard
case is removed. DOTAv2 contains 875 recorded outside-image GT boxes, retained
unchanged; DIOR has none. Source agreement does not establish independent semantic
completeness of the annotations or eliminate possible pretraining exposure.

At the inference boundary a record becomes only `{uid, suite, class_specs}`.
The requested classes are those present in the annotations, ordered by the
frozen benchmark vocabulary. No GT multiplicity, extent, center, region or answer
is supplied. Images are not cropped to annotated objects. The scorer alone
receives GT geometry. This is nevertheless privileged category information, so
report it as **GT-present-category localization**, not unrestricted detection.
Earlier full-vocabulary results answer a different question and remain untouched.

## Checkpoints and prompts

All HF/code revisions are pinned in `common.py` and recorded in `models.json`.
The Standard final checkpoint is Stage-2 best step 1200, SHA-256
`b0c4c248b16de59f0339c2d4658e36de86cdbcd8c81c4b408bd2f9d199fadb83`.
No selection uses the evaluation images.

| Model | Detection request | Pointing request |
|---|---|---|
| Base LA, final Standard | Joint categories, native `</c>` separator | One native point query per category |
| Rex-Omni | Joint categories through official detection wrapper | Official pointing wrapper per category |
| Qwen3-VL-4B/8B | One category, normalized `bbox_2d` JSON | One category, normalized `point_2d` JSON |
| Grounding DINO / LAE-DINO | Joint period-separated category text | Not applicable |
| Optional PID | Concurrent category discovery, then packed-all extents | Concurrent native discovery |
| Optional class-isolated | Concurrent direct category queries | Native point query per category |

Native model syntax is retained; literal prompts are saved for every call.
The Qwen prompt explicitly requests all matching instances, 0-1000 coordinates,
and an empty list for absence. There is no GT object-count hint. Seeds depend
only on UID, task and category, not shard or GPU availability. LA/Qwen sampling
uses temperature 0.7, top-p 0.9 and repetition penalty 1.1; Rex uses temperature
0 with the official wrapper. These are fixed model-specific settings, not equal
random trajectories across different models.

The same 8192 output-token budget applies to VLM generation calls. This is per
category for single-category querying and per joint answer otherwise. Saturation
is logged. This is a practical native-interface comparison, not matched FLOPs or
equal total per-image token allocation. LA uses its 6000 image-token limit; Qwen
uses its native processor with max_pixels=1,176,000. Detectors use their native
800/1333 resizing. These choices are frozen, not changed for hard cases.

## Geometry, classes and postprocessing

- LA consumes native reference/box blocks through the released parser; a
  2-coordinate block is a point, a 4-coordinate block a box. No point-to-center
  or box-to-point conversion. Decimal tokens, accepted numeric range, corner
  ordering and boundary clipping follow the pinned implementation. Base LA and
  Standard must emit their own reference labels; missing labels are not guessed.
  For the optional PID/isolated routes, an explicit input-address/query identity
  supplies an omitted label, without overwriting a wrong emitted label.
- Rex uses its own native extraction and original-image coordinate conversion.
- Qwen JSON coordinates are mapped from 0-1000 to original-image pixels.
- A Qwen singleton query supplies the class only when the JSON omits it. An
  explicit wrong `label` or `category` remains an unmatched prediction, just as
  for native LA/Rex class labels.
- Class aliases come exclusively from the frozen class specifications. Unknown
  emitted classes remain unmatched predictions, not reassigned to a valid class.
- Grounding DINO uses its official `phrases2classes` rule: the first requested
  category name contained in the decoded phrase determines the label. A composite
  phrase therefore follows query order, while an incomplete name that contains no
  full requested category remains unmapped. No GT box or overlap is used to assign
  a label. The original decoded phrase is retained for inspection.
- LA-native corner repair is applied before geometry validation in the main
  results. Recognized outputs that remain degenerate or have the wrong geometry
  type count as FP with an `invalid_geometry` marker. Syntax rejected by the
  native parser is not turned into fabricated candidates. For Rex/Qwen, complete
  extracted invalid candidates remain FP under the existing native-format
  adapters. These model-specific parsing contracts are not identical grammars.
  Detector coordinates are clipped to the image as in native postprocessing;
  collapsed boxes still count as FP if retained by score/rank. GT is unchanged.
- `strict_sensitivity` reparses the same LA-family responses without corner
  repair; complete invalid candidates remain FP. It uses the exact same scorer,
  thresholds and records as the main result and never reruns generation. For
  non-LA models the main and strict tables use the same unchanged predictions.
  It retains the preceding parser's singleton-query label fallback and integer
  token grammar. Thus it is a parser-policy sensitivity, not a single-factor
  corner-sorting ablation.
- No added NMS, IoU merging, deduplication or evaluator maxDets=100 is used.

Grounding DINO uses confidence >0.25 and text threshold 0.25, retaining its native
query count. Low-score candidates remain stored so the selected threshold is
auditable. LAE-DINO stores its entire native query-by-category score pool from the
same forward and reports: native top-300 at >0.25 (matched dense operating point),
all pairs at >0.25 (uncapped sensitivity), and native top-300 at >0.30 (author
operating-point sensitivity). The native rank is recorded before invalid geometry
filtering, so an invalid top-300 entry cannot promote a lower-ranked output.
These variants are labelled; the best test score is not silently chosen.
Detector timing includes collecting the stored candidate pool; it is not a
separate optimized latency measurement of only the top-300 postprocessing path.

## Metrics

For each IoU threshold, match each GT to the best unused eligible prediction in
GT order, using the released LA function. Primary eligibility requires the same
class. Retain duplicates as additional predictions. With image-level counts:

`P_i = TP_i / (TP_i + FP_i)`, `R_i = TP_i / (TP_i + FN_i)`.

Zero predictions on these nonempty-GT images give P=R=0. Average P and R over
images and compute their harmonic mean. Report F1@0.50, F1@0.75, and mF1, the
mean of F1 at all ten thresholds 0.50:0.05:0.95. Also retain pooled-count (micro)
P/R/F1 and exact released class-flattened compatibility scores. Do not combine
image-macro and earlier micro F1 in one column.

Pointing uses the released LA class-aware HBB-containment matcher: a GT can match
one unused point inside its box, not only its center, and vice versa. GT order
and native point output order are preserved. Report the same image-macro P/R/F1
with micro scores as secondary data. This is an RS HBB-derived pointing task,
not Molmo's official mask-based evaluation.

## Timing and execution

Each image is evaluated end to end after one untimed warmup per worker. CUDA
synchronization brackets preprocessing, all requested category calls and parsing.
Disk image loading/hash checks, model loading, scoring, queue wait and JSON
serialization are excluded. Aggregate seconds by complete image, never divide
a multi-category image by its number of category queries. Report mean/p50/p95,
valid-geometry outputs/s, matched outputs/s at IoU 0.50 (or pointing containment),
and peak allocated GPU memory. LA token counts are retokenized responses and
marked as such, not claimed as exact underlying decoder iterations.
Strict sensitivity is computed offline and excluded from inference timing.
Main timings include native parsing. Complete attempted-output counts/rates,
including invalid geometry remaining after native parsing, are retained
separately. Invalid repetitive geometry cannot inflate the displayed usable
box/point throughput. Native PBD can finish an atomic block slightly beyond the
nominal token budget; `budget_saturated` records this rather than trimming outputs.

GPUs are homogeneous within a run. Worker environments and source/checkpoint/
manifest hashes are frozen. Scheduling on whichever permitted GPU is idle does
not change the assigned images, seeds or scoring. Shared-server PCIe/I/O/thermal
contention may still influence time; isolated profiling remains a separate test.
All VLMs and HF Grounding DINO must retain the same recorded environment within
a run; LAE-DINO uses its separately recorded MMDetection environment. This does
not make the two detector/VLM software stacks identical.

## Implementation checks

`test_protocol.py` exercises leakage boundaries, normalization, wrong-class false
positives, duplicate handling, point containment, macro/micro differences, native
caps, immutable completion hashes, complete shard merging and GPU selection.
`evidence/other_metric.py` is the unmodified hash-pinned upstream file. Tests and
one-time audits are separate from normal training startup. Actual local GPU test
coverage and unresolved limits are documented in LOCAL_CHECKS.md.
