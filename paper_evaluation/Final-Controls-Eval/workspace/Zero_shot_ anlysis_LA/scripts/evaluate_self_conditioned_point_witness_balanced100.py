#!/usr/bin/env python3
"""Evaluate Self-Conditioned Point-Witness Refinement on DOTAv2 Balanced-100.

The deployable protocol never consults ground truth while producing boxes:

1. one full-vocabulary point-scaffold query is issued for each image;
2. every generated labeled point is written back as a spatial witness;
3. the same adapter refines that witness into exactly one labeled HBB;
4. boxes that do not contain their own witness are rejected;
5. exact point duplicates and class-wise box duplicates are merged.

Both raw and fixed IoU-0.5 NMS metrics are retained.  The script is resumable
at image and witness-task granularity and uses stable per-call RNG seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluate_locateanything_lae_style import (  # noqa: E402
    benchmark_class_specs,
    build_alias_lookup,
    canonicalize_label,
    clip_hbox_to_image,
    evaluate_lae_style,
    load_lae_records,
    nms_predictions,
    read_json,
    resolve_image_path,
    selected_benchmarks,
)
from zero_shot_analysis.geometry import hbox_area, hbox_center, point_inside_hbox  # noqa: E402
from zero_shot_analysis.locate_anything import (  # noqa: E402
    LocateAnythingWorker,
    parse_locateanything_labeled_boxes,
    parse_locateanything_labeled_points,
)


NORMALIZED_HBB = re.compile(
    r"<box><(\d{1,4})><(\d{1,4})><(\d{1,4})><(\d{1,4})></box>"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def stable_seed(seed: int, *parts: Any) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


def set_generation_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(path: Path, benchmark: str, limit: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Diagnostic manifests may be benchmark-specific and therefore omit the
    # redundant benchmark column. Only filter when that column is present.
    if rows and "benchmark" in rows[0]:
        rows = [row for row in rows if row.get("benchmark") == benchmark]
    return rows if limit is None else rows[:limit]


def select_records(
    benchmark_name: str,
    benchmark: Mapping[str, Any],
    class_specs: Sequence[Mapping[str, str]],
    manifest_rows: Sequence[Mapping[str, str]],
    label_field: str,
) -> list[dict[str, Any]]:
    records = load_lae_records(benchmark_name, benchmark, class_specs, label_field)
    by_id = {str(record["image_id"]): record for record in records}
    selected = []
    for manifest_row in manifest_rows:
        image_id = str(manifest_row["image_id"])
        if image_id not in by_id:
            raise RuntimeError(f"Balanced manifest image is absent from annotations: {image_id}")
        selected.append(by_id[image_id])
    return selected


def point_prompt(class_specs: Sequence[Mapping[str, str]]) -> str:
    categories = "</c>".join(str(spec["prompt"]) for spec in class_specs)
    return f"Point to all instances that match the following categories: {categories}."


def point_to_norm(point: Sequence[float], width: int, height: int) -> tuple[int, int]:
    x = int(round(max(0.0, min(float(width), float(point[0]))) / max(1.0, float(width)) * 1000))
    y = int(round(max(0.0, min(float(height), float(point[1]))) / max(1.0, float(height)) * 1000))
    return max(0, min(1000, x)), max(0, min(1000, y))


def witness_prompt(label: str, point: Sequence[float], width: int, height: int) -> str:
    x, y = point_to_norm(point, width, height)
    point_token = f"<box><{x}><{y}></box>"
    return (
        f"The point {point_token} is a spatial witness. "
        f"Locate the single {label} instance containing this point. "
        f"If no {label} contains the point, return None."
    )


def point_address_prompt(
    label: str, point: Sequence[float], width: int, height: int
) -> str:
    """Use the exact prompt learned by the single-image point-address route."""
    x, y = point_to_norm(point, width, height)
    return (
        f"Locate a single {label} instance addressed by point "
        f"<box><{x}><{y}></box>. If no {label} contains the point, return None."
    )


def point_address_proposal_prompt(
    label: str,
    point: Sequence[float],
    width: int,
    height: int,
    proposal_norm: Sequence[int],
) -> str:
    """Match proposal-conditioned training without exposing ground truth."""

    x, y = point_to_norm(point, width, height)
    x1, y1, x2, y2 = map(int, proposal_norm)
    return (
        f"Locate a single {label} instance addressed by point "
        f"<box><{x}><{y}></box>. Refine provisional box "
        f"<box><{x1}><{y1}><{x2}><{y2}></box>. "
        f"If no {label} contains the point, return None."
    )


def frozen_proposal(
    row: Mapping[str, Any] | None,
    point: Sequence[float],
    width: int,
    height: int,
    fallback_side: int,
) -> tuple[list[int], str]:
    """Build a point-containing proposal from a frozen non-oracle model output."""

    point_x, point_y = point_to_norm(point, width, height)
    parsed = [] if row is None else NORMALIZED_HBB.findall(str(row.get("output", "")))
    if parsed:
        values = list(map(int, parsed[0]))
        left, right = sorted((max(0, min(1000, values[0])), max(0, min(1000, values[2]))))
        top, bottom = sorted((max(0, min(1000, values[1])), max(0, min(1000, values[3]))))
        source = "frozen_first_pass_output"
    else:
        side_x = max(4, round(min(fallback_side, width) / max(1, width) * 1000))
        side_y = max(4, round(min(fallback_side, height) / max(1, height) * 1000))
        left, right = point_x - side_x // 2, point_x + side_x // 2
        top, bottom = point_y - side_y // 2, point_y + side_y // 2
        source = "deterministic_point_centered_fallback"
    # This uses only the predicted address, never GT, and guarantees that the
    # proposal remains a valid candidate region for that address.
    left, right = max(0, min(left, point_x - 2)), min(1000, max(right, point_x + 2))
    top, bottom = max(0, min(top, point_y - 2)), min(1000, max(bottom, point_y + 2))
    if right <= left or bottom <= top:
        raise RuntimeError("Frozen proposal is degenerate")
    return [left, top, right, bottom], source


def pivr_witness_prompt(label: str, point: Sequence[float], width: int, height: int) -> str:
    """Match the synchronized global/local PIVR-LoRA training prompt."""
    x, y = point_to_norm(point, width, height)
    point_token = f"<box><{x}><{y}></box>"
    return (
        "Image 1 is the global scene. Image 2 is the point-indexed local view for address A1. "
        f"In Image 2, point {point_token} addresses one {label}. "
        f"Return exactly one box for that {label} in Image 2 coordinates. "
        f"If no {label} contains the address point, return None."
    )


def direct_detection_prompt(label: str) -> str:
    """Use LA's ordinary class-detection template for the crop-only control."""
    return f"Locate all the instances that matches the following description: {label}."


def point_centered_crop(
    point: Sequence[float], width: int, height: int, side: int
) -> tuple[int, int, int, int]:
    """Return a fixed-size square around a predicted point without using GT."""
    side = max(1, min(int(side), width, height))
    x1 = int(round(float(point[0]) - side / 2.0))
    y1 = int(round(float(point[1]) - side / 2.0))
    x1 = max(0, min(width - side, x1))
    y1 = max(0, min(height - side, y1))
    return x1, y1, x1 + side, y1 + side


def canonical_points(
    answer: str,
    alias_lookup: Mapping[str, str],
    class_set: set[str],
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    unknown: Counter[str] = Counter()
    points: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for source_index, raw in enumerate(parse_locateanything_labeled_points(answer, width, height)):
        label = canonicalize_label(raw.get("label"), alias_lookup, class_set)
        if label is None:
            unknown[str(raw.get("label") or "<missing>")] += 1
            continue
        point = [float(raw["point"][0]), float(raw["point"][1])]
        nx, ny = point_to_norm(point, width, height)
        key = (label, nx, ny)
        if key in seen:
            continue
        seen.add(key)
        points.append(
            {
                "label": label,
                "raw_label": raw.get("label"),
                "point": point,
                "point_norm": [nx, ny],
                "source": raw.get("source"),
                "source_index": source_index,
                "point_id": f"{label}:{nx}:{ny}",
            }
        )
    return points, unknown


def choose_witness_box(
    answer: str,
    expected_label: str,
    point: Sequence[float],
    alias_lookup: Mapping[str, str],
    class_set: set[str],
    width: int,
    height: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    candidates = []
    rejected = Counter()
    for order, raw in enumerate(parse_locateanything_labeled_boxes(answer, width, height)):
        label = canonicalize_label(raw.get("label"), alias_lookup, class_set)
        if label != expected_label:
            rejected["wrong_or_unknown_label"] += 1
            continue
        box = clip_hbox_to_image(raw.get("hbox") or [], width, height)
        if box is None:
            rejected["invalid_geometry"] += 1
            continue
        if not point_inside_hbox(point, box):
            rejected["does_not_contain_witness"] += 1
            continue
        cx, cy = hbox_center(box)
        candidates.append(
            (
                (cx - float(point[0])) ** 2 + (cy - float(point[1])) ** 2,
                hbox_area(box),
                order,
                {
                    "label": expected_label,
                    "raw_label": raw.get("label"),
                    "hbox": box,
                    "score": 1.0,
                    "source": "self_conditioned_point_witness_refinement",
                },
            )
        )
    if not candidates:
        return None, {"candidate_count": 0, "rejected": dict(rejected)}
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3], {"candidate_count": len(candidates), "rejected": dict(rejected)}


def point_match(record: Mapping[str, Any], points: Sequence[Mapping[str, Any]]) -> tuple[int, int, int]:
    gt_by_class: dict[str, list[Sequence[float]]] = defaultdict(list)
    for gt in record.get("gt", []):
        gt_by_class[str(gt["label"])].append(gt["hbox"])
    matched: set[tuple[str, int]] = set()
    tp = fp = 0
    for point_row in points:
        label = str(point_row["label"])
        point = point_row["point"]
        candidates = []
        for index, box in enumerate(gt_by_class.get(label, [])):
            if (label, index) in matched or not point_inside_hbox(point, box):
                continue
            cx, cy = hbox_center(box)
            candidates.append(((cx - point[0]) ** 2 + (cy - point[1]) ** 2, index))
        if candidates:
            matched.add((label, min(candidates)[1]))
            tp += 1
        else:
            fp += 1
    return tp, fp, sum(len(items) for items in gt_by_class.values()) - tp


def ratio(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config/lae_benchmarks_balanced100.json")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT / "dataset/eval_balanced100/DOTAv2/manifest/dotav2_balanced_100_manifest.csv",
    )
    parser.add_argument("--benchmark", default="DOTAv2")
    parser.add_argument("--model", type=Path, default=PROJECT / "model/LocateAnything-3B/checkpoint")
    parser.add_argument(
        "--adapter",
        type=Path,
        help=(
            "Optional localization LoRA checkpoint. Omit it to evaluate the "
            "released Base LA checkpoint without an adapter."
        ),
    )
    parser.add_argument(
        "--adapter-kind",
        choices=("standard_lora", "ravel"),
        default="standard_lora",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--class-source", default="lae-metainfo", choices=("lae-metainfo", "annotations"))
    parser.add_argument("--gt-label-field", default="raw_class_name")
    parser.add_argument("--generation-mode", default="hybrid", choices=("slow", "fast", "hybrid"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--image-token-limit", type=int, default=1024)
    parser.add_argument("--point-max-new-tokens", type=int, default=2048)
    parser.add_argument("--refine-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--point-predictions",
        type=Path,
        help="Reuse a completed point-scaffold JSONL instead of generating points again.",
    )
    parser.add_argument(
        "--refine-crop-size",
        type=int,
        default=0,
        help="If positive, refine each predicted point in a non-oracle point-centered square crop.",
    )
    parser.add_argument(
        "--refine-prompt-mode",
        choices=("witness", "point-address", "point-address-proposal", "direct"),
        default="witness",
        help=(
            "Use the learned point-witness prompt, the exact single-image "
            "point-address prompt, or LA's ordinary class-detection prompt. "
            "In every mode only a box containing the predicted point is retained."
        ),
    )
    parser.add_argument(
        "--proposal-predictions",
        type=Path,
        help=(
            "Frozen refinement JSONL used only to provide a shared non-oracle "
            "provisional HBB for point-address-proposal mode."
        ),
    )
    parser.add_argument("--proposal-fallback-side", type=int, default=384)
    parser.add_argument(
        "--refine-context-mode",
        choices=("crop-only", "global-plus-crop"),
        default="crop-only",
        help=(
            "Use the legacy one-view crop refiner or synchronized global/local PIVR re-entry. "
            "The latter requires --refine-crop-size > 0."
        ),
    )
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--refinement-task-shard-index", type=int, default=0)
    parser.add_argument("--refinement-task-shard-count", type=int, default=1)
    parser.add_argument(
        "--refinement-output-name",
        default="witness_refinement_predictions.jsonl",
        help="Per-shard refinement JSONL name inside --output.",
    )
    parser.add_argument(
        "--skip-finalize",
        action="store_true",
        help="Write this refinement shard without assembling corpus metrics.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.refinement_task_shard_count < 1:
        parser.error("--refinement-task-shard-count must be positive")
    if not 0 <= args.refinement_task_shard_index < args.refinement_task_shard_count:
        parser.error("--refinement-task-shard-index must be in [0, shard-count)")
    if Path(args.refinement_output_name).name != args.refinement_output_name:
        parser.error("--refinement-output-name must be a file name, not a path")
    if args.refinement_task_shard_count > 1 and not args.skip_finalize:
        parser.error("sharded refinement workers must use --skip-finalize")
    if args.refine_context_mode == "global-plus-crop" and args.refine_crop_size <= 0:
        parser.error("--refine-context-mode global-plus-crop requires --refine-crop-size > 0")
    if args.refine_prompt_mode == "point-address-proposal" and args.proposal_predictions is None:
        parser.error("point-address-proposal mode requires --proposal-predictions")

    args.adapter = None if args.adapter is None else args.adapter.resolve()
    args.output = args.output.resolve()
    if args.adapter is not None and not args.adapter.is_file() and not args.adapter.is_symlink():
        raise FileNotFoundError(args.adapter)
    # Keep released-Base and single-adapter controls isolated from adapters that
    # may have been exported by an earlier experiment in the same shell.
    for name in list(os.environ):
        if name.startswith("LOCANY_") and name.endswith("_ADAPTER"):
            os.environ.pop(name, None)
    if args.adapter is not None:
        adapter_environment = {
            "standard_lora": "LOCANY_POINTRELAY_ADAPTER",
            "ravel": "LOCANY_RAVEL_ADAPTER",
        }[args.adapter_kind]
        os.environ[adapter_environment] = str(args.adapter)
    method_kind = args.adapter_kind if args.adapter is not None else "released_base_la"

    config = read_json(args.config)
    benchmark = dict(selected_benchmarks(config, [args.benchmark]))[args.benchmark]
    class_specs = benchmark_class_specs(
        args.benchmark, benchmark, args.class_source, args.gt_label_field
    )
    alias_lookup = build_alias_lookup(class_specs)
    class_set = {str(spec["canonical"]) for spec in class_specs}
    records = select_records(
        args.benchmark,
        benchmark,
        class_specs,
        load_manifest(args.manifest, args.benchmark, args.limit),
        args.gt_label_field,
    )
    records_by_id = {str(record["image_id"]): record for record in records}
    image_root = PROJECT / benchmark["image_root"]
    prompt = point_prompt(class_specs)
    args.output.mkdir(parents=True, exist_ok=True)
    point_path = (
        args.point_predictions.resolve()
        if args.point_predictions is not None
        else args.output / "point_scaffold_predictions.jsonl"
    )
    canonical_refinement_path = args.output / "witness_refinement_predictions.jsonl"
    refinement_path = args.output / args.refinement_output_name
    print(f"Images: {len(records)} | GT: {sum(len(record.get('gt', [])) for record in records)}")
    print(f"Adapter: {args.adapter if args.adapter is not None else 'none (released Base LA)'}")
    print(f"Point prompt: {prompt}")
    print(f"Output: {args.output}")
    if args.dry_run:
        return

    from PIL import Image
    import torch

    worker = LocateAnythingWorker.from_args(str(args.model), args.device, args.dtype)
    if args.adapter is not None:
        active_adapter = (
            worker.pointrelay_pilot
            if args.adapter_kind == "standard_lora"
            else worker.ravel_pilot
        )
        if active_adapter is None:
            raise RuntimeError(f"{args.adapter_kind} adapter was not loaded")
    elif worker.pointrelay_pilot is not None or worker.ravel_pilot is not None:
        raise RuntimeError("Released Base LA evaluation unexpectedly loaded an adapter")
    image_processor = getattr(worker.processor, "image_processor", None)
    if image_processor is None or not hasattr(image_processor, "in_token_limit"):
        raise RuntimeError("LocateAnything image processor has no in_token_limit")
    image_processor.in_token_limit = int(args.image_token_limit)
    if int(image_processor.in_token_limit) != args.image_token_limit:
        raise RuntimeError("Image-token limit override failed")

    existing_points = (
        {str(row["image_id"]): row for row in read_jsonl(point_path)}
        if args.point_predictions is not None or args.resume
        else {}
    )
    if args.point_predictions is not None:
        missing_point_images = set(records_by_id) - set(existing_points)
        if missing_point_images:
            raise RuntimeError(
                f"External point predictions are missing {len(missing_point_images)} images"
            )
    for index, record in enumerate(records, 1):
        image_id = str(record["image_id"])
        if image_id in existing_points:
            continue
        image_path = resolve_image_path(record, image_root)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        set_generation_seed(stable_seed(args.seed, image_id, "point-scaffold"))
        started = time.perf_counter()
        output = worker.predict(
            image,
            prompt,
            generation_mode=args.generation_mode,
            max_new_tokens=args.point_max_new_tokens,
            temperature=args.temperature,
        )
        points, unknown = canonical_points(
            str(output.get("answer", "")), alias_lookup, class_set, width, height
        )
        tp, fp, fn = point_match(record, points)
        append_jsonl(
            point_path,
            [
                {
                    "dataset": args.benchmark,
                    "image_id": image_id,
                    "file_name": record["file_name"],
                    "image_path": str(image_path),
                    "query": prompt,
                    "output": str(output.get("answer", "")),
                    "points": points,
                    "unknown_labels": dict(unknown),
                    "gt_count": len(record.get("gt", [])),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "meta": {"width": width, "height": height, "image_token_limit": args.image_token_limit},
                }
            ],
        )
        existing_points[image_id] = read_jsonl(point_path)[-1]
        if index % args.log_every == 0 or index == len(records):
            print(f"Point scaffold {index}/{len(records)}: points={len(points)}", flush=True)

    point_rows = {str(row["image_id"]): row for row in read_jsonl(point_path)}
    proposal_rows = (
        {
            (str(row["image_id"]), str(row["point_id"])): row
            for row in read_jsonl(args.proposal_predictions.resolve())
        }
        if args.proposal_predictions is not None
        else {}
    )
    completed_refinements: set[tuple[str, str]] = set()
    if args.resume:
        completed_refinements = {
            (str(row["image_id"]), str(row["point_id"]))
            for path in {canonical_refinement_path, refinement_path}
            for row in read_jsonl(path)
        }
    total_tasks = sum(len(point_rows.get(str(record["image_id"]), {}).get("points", [])) for record in records)
    shard_total_tasks = sum(
        task_index % args.refinement_task_shard_count
        == args.refinement_task_shard_index
        for task_index in range(total_tasks)
    )
    completed_count = 0
    task_index = -1
    for record in records:
        image_id = str(record["image_id"])
        point_row = point_rows[image_id]
        image_path = Path(point_row["image_path"])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        for point in point_row.get("points", []):
            task_index += 1
            if (
                task_index % args.refinement_task_shard_count
                != args.refinement_task_shard_index
            ):
                continue
            point_id = str(point["point_id"])
            key = (image_id, point_id)
            completed_count += 1
            if key in completed_refinements:
                continue
            inference_image = image
            inference_point = point["point"]
            crop_box = None
            proposal_norm = None
            proposal_source = None
            if args.refine_crop_size > 0:
                crop_box = point_centered_crop(
                    point["point"], width, height, args.refine_crop_size
                )
                x1, y1, x2, y2 = crop_box
                inference_image = image.crop(crop_box)
                inference_point = [
                    float(point["point"][0]) - x1,
                    float(point["point"][1]) - y1,
                ]
            if args.refine_prompt_mode == "witness":
                prompt_builder = (
                    pivr_witness_prompt
                    if args.refine_context_mode == "global-plus-crop"
                    else witness_prompt
                )
                query = prompt_builder(
                    str(point["label"]), inference_point,
                    inference_image.width, inference_image.height,
                )
            elif args.refine_prompt_mode == "point-address":
                if args.refine_context_mode != "crop-only":
                    raise RuntimeError(
                        "point-address mode requires one full-image input"
                    )
                query = point_address_prompt(
                    str(point["label"]),
                    inference_point,
                    inference_image.width,
                    inference_image.height,
                )
            elif args.refine_prompt_mode == "point-address-proposal":
                if args.refine_context_mode != "crop-only" or crop_box is not None:
                    raise RuntimeError(
                        "point-address-proposal mode requires one uncropped full image"
                    )
                proposal_norm, proposal_source = frozen_proposal(
                    proposal_rows.get(key),
                    inference_point,
                    inference_image.width,
                    inference_image.height,
                    args.proposal_fallback_side,
                )
                query = point_address_proposal_prompt(
                    str(point["label"]),
                    inference_point,
                    inference_image.width,
                    inference_image.height,
                    proposal_norm,
                )
            else:
                query = direct_detection_prompt(str(point["label"]))
            inference_media = (
                [image, inference_image]
                if args.refine_context_mode == "global-plus-crop"
                else inference_image
            )
            set_generation_seed(stable_seed(args.seed, image_id, point_id, "witness-refinement"))
            started = time.perf_counter()
            output = worker.predict(
                inference_media,
                query,
                generation_mode=args.generation_mode,
                max_new_tokens=args.refine_max_new_tokens,
                temperature=args.temperature,
            )
            answer = str(output.get("answer", ""))
            prediction, diagnostics = choose_witness_box(
                answer,
                str(point["label"]),
                inference_point,
                alias_lookup,
                class_set,
                inference_image.width,
                inference_image.height,
            )
            if prediction is not None:
                if crop_box is not None:
                    x1, y1, _x2, _y2 = crop_box
                    prediction["hbox"] = [
                        prediction["hbox"][0] + x1,
                        prediction["hbox"][1] + y1,
                        prediction["hbox"][2] + x1,
                        prediction["hbox"][3] + y1,
                    ]
                    prediction["source"] = (
                        (
                            "point_indexed_global_local_visual_reentry"
                            if args.refine_context_mode == "global-plus-crop"
                            else "point_centered_visual_requery"
                        )
                        if args.refine_prompt_mode == "witness"
                        else "point_centered_direct_detection_control"
                    )
                prediction["point_id"] = point_id
                prediction["witness_point"] = point["point"]
            append_jsonl(
                refinement_path,
                [
                    {
                        "dataset": args.benchmark,
                        "image_id": image_id,
                        "file_name": record["file_name"],
                        "image_path": str(image_path),
                        "point_id": point_id,
                        "target_class": point["label"],
                        "point": point["point"],
                        "refinement_crop": list(crop_box) if crop_box is not None else None,
                        "refinement_context_mode": args.refine_context_mode,
                        "provisional_box_norm": proposal_norm,
                        "provisional_box_source": proposal_source,
                        "proposal_predictions": None
                        if args.proposal_predictions is None
                        else str(args.proposal_predictions.resolve()),
                        "adapter_kind": method_kind,
                        "query": query,
                        "output": answer,
                        "pilot_diagnostics": output.get("pilot_diagnostics", {}),
                        "prediction": prediction,
                        "diagnostics": diagnostics,
                        "latency_ms": (time.perf_counter() - started) * 1000,
                    }
                ],
            )
            if completed_count % args.log_every == 0 or completed_count == shard_total_tasks:
                print(
                    f"Witness refinement shard "
                    f"{args.refinement_task_shard_index}/"
                    f"{args.refinement_task_shard_count} "
                    f"{completed_count}/{shard_total_tasks}: "
                    f"kept={int(prediction is not None)}",
                    flush=True,
                )

    if args.skip_finalize:
        print(
            f"Refinement shard {args.refinement_task_shard_index}/"
            f"{args.refinement_task_shard_count} complete: {refinement_path}",
            flush=True,
        )
        return

    refinement_rows = read_jsonl(canonical_refinement_path)
    predictions_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in refinement_rows:
        if str(row["image_id"]) not in records_by_id or row.get("prediction") is None:
            continue
        predictions_by_image[str(row["image_id"])].append(dict(row["prediction"]))

    raw_rows = []
    nms_rows = []
    for record in records:
        image_id = str(record["image_id"])
        raw = predictions_by_image.get(image_id, [])
        base = {
            "dataset": args.benchmark,
            "image_id": image_id,
            "file_name": record["file_name"],
            "prompt_mode": "self_conditioned_point_witness",
        }
        raw_rows.append({**base, "predictions": raw})
        nms_rows.append({**base, "predictions": nms_predictions(raw, args.nms_iou)})
    nms_slug = str(args.nms_iou).replace(".", "p")
    write_jsonl(args.output / "predictions_raw.jsonl", raw_rows)
    write_jsonl(args.output / f"predictions_nms{nms_slug}.jsonl", nms_rows)

    raw_metrics = evaluate_lae_style(
        records, raw_rows, class_specs, args.benchmark, args.class_source,
        args.gt_label_field, 0, prompt_mode="self_conditioned_point_witness_raw",
    )
    nms_metrics = evaluate_lae_style(
        records, nms_rows, class_specs, args.benchmark, args.class_source,
        args.gt_label_field, 0, prompt_mode="self_conditioned_point_witness_nms",
    )
    point_totals = Counter()
    for image_id in records_by_id:
        row = point_rows[image_id]
        for key in ("tp", "fp", "fn"):
            point_totals[key] += int(row.get(key, 0))
    point_summary = {
        **dict(point_totals),
        **ratio(point_totals["tp"], point_totals["fp"], point_totals["fn"]),
        "predicted_points": point_totals["tp"] + point_totals["fp"],
    }
    summary = {
        "schema_version": "self-conditioned-point-witness-balanced100-v1",
        "adapter": None if args.adapter is None else str(args.adapter),
        "method_kind": method_kind,
        "images": len(records),
        "gt_instances": sum(len(record.get("gt", [])) for record in records),
        "inference_seed": args.seed,
        "image_token_limit": args.image_token_limit,
        "refine_crop_size": args.refine_crop_size,
        "refine_prompt_mode": args.refine_prompt_mode,
        "refine_context_mode": args.refine_context_mode,
        "point_scaffold": point_summary,
        "refinement_tasks": total_tasks,
        "successful_refinements": sum(row.get("prediction") is not None for row in refinement_rows),
        "raw": raw_metrics,
        "nms_iou_0_5": nms_metrics,
    }
    write_json(args.output / "metrics.json", summary)
    accuracy = nms_metrics["accuracy"]["f1_iou_0.50"]
    print(
        json.dumps(
            {
                "point_scaffold": point_summary,
                "raw": raw_metrics["accuracy"]["f1_iou_0.50"],
                "nms_iou_0_5": accuracy,
                "predicted_boxes_raw": raw_metrics["predicted_boxes"],
                "predicted_boxes_nms": nms_metrics["predicted_boxes"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
