"""Evaluate LocateAnything-3B with LAE benchmark detection protocols.

This script is intentionally separate from evaluate_locateanything_lae_benchmark.py.
The older script evaluates a diagnostic prompt suite with negatives, relations,
and selected classes. This one supports two LAE-compatible comparison shapes:

* multi_label: one full-vocabulary prompt per image using categories joined by </c>
* single_label: one prompt per image/class, including absent classes
* class-labeled boxes parsed from <ref>...</ref><box>...</box>
* COCO-like AP50, AP75, and mAP_50_95 over horizontal boxes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from zero_shot_analysis.geometry import greedy_match, hbox_iou, normalize_hbox, size_bucket
from zero_shot_analysis.locate_anything import (
    LocateAnythingWorker,
    parse_locateanything_labeled_boxes,
)
from zero_shot_analysis.metrics import IOU_THRESHOLDS, write_metrics
from zero_shot_analysis.schema import normalize_class_name, read_jsonl


DIOR_LAE_CLASSES = (
    "airplane",
    "airport",
    "groundtrackfield",
    "harbor",
    "baseballfield",
    "overpass",
    "basketballcourt",
    "ship",
    "bridge",
    "stadium",
    "storagetank",
    "tenniscourt",
    "expressway service area",
    "trainstation",
    "expressway toll station",
    "vehicle",
    "golffield",
    "windmill",
    "chimney",
    "dam",
)

DOTA_LAE_CLASSES = (
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
    "container crane",
    "airport",
    "helipad",
)

LAE80C_CLASSES = (
    "airplane",
    "airport",
    "groundtrackfield",
    "harbor",
    "baseballfield",
    "overpass",
    "basketballcourt",
    "bridge",
    "stadium",
    "storagetank",
    "tenniscourt",
    "expressway service area",
    "trainstation",
    "expressway toll station",
    "vehicle",
    "golffield",
    "windmill",
    "dam",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
    "container crane",
    "helipad",
    "Bus",
    "Cargo Truck",
    "Dry Cargo Ship",
    "Dump Truck",
    "Engineering Ship",
    "Excavator",
    "Fishing Boat",
    "Intersection",
    "Liquid Cargo Ship",
    "Motorboat",
    "Passenger Ship",
    "Small Car",
    "Tractor",
    "Trailer",
    "Truck Tractor",
    "Tugboat",
    "Van",
    "Warship",
    "working condensing tower",
    "unworking condensing tower",
    "working chimney",
    "unworking chimney",
    "Fixed-wing Aircraft",
    "Small Aircraft",
    "Cargo Plane",
    "Pickup Truck",
    "Utility Truck",
    "Passenger Car",
    "Cargo Car",
    "Flat Car",
    "Locomotive",
    "Sailboat",
    "Barge",
    "Ferry",
    "Yacht",
    "Oil Tanker",
    "Engineering Vehicle",
    "Tower crane",
    "Reach Stacker",
    "Straddle Carrier",
    "Mobile Crane",
    "Haul Truck",
    "Front loader/Bulldozer",
    "Cement Mixer",
    "Ground Grader",
    "Hut/Tent",
    "Shed",
    "Building",
    "Aircraft Hangar",
    "Damaged Building",
    "Facility",
    "Construction Site",
    "Shipping container lot",
    "Shipping Container",
    "Pylon",
    "Tower",
)

LAE_METAINFOS = {
    "DIOR": DIOR_LAE_CLASSES,
    "DOTAv2": DOTA_LAE_CLASSES,
    "LAE-80C": LAE80C_CLASSES,
}

PROMPT_ALIASES = {
    "baseballfield": "baseball field",
    "basketballcourt": "basketball court",
    "golffield": "golf field",
    "groundtrackfield": "ground track field",
    "storagetank": "storage tank",
    "tenniscourt": "tennis court",
    "trainstation": "train station",
}

EXTRA_OUTPUT_ALIASES = {
    "aeroplane": "airplane",
    "fixed wing aircraft": "fixed-wing aircraft",
    "front loader bulldozer": "front loader/bulldozer",
    "shipping container": "shipping container",
}

LOCATEANYTHING_WEIGHT_FILES = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
PROMPT_MODES = ("multi_label", "single_label", "single_label_tiled")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(64).startswith(b"version https://git-lfs.github.com/spec")


def model_path(config: dict[str, Any], use_hf_id: bool, override: str | None = None) -> str:
    if override:
        return override
    env_override = os.environ.get("MODEL_PATH")
    if env_override:
        return env_override
    if use_hf_id:
        return str(config["model"]["hf_model_id"])
    return str(PROJECT_ROOT / config["model"]["local_checkpoint"])


def local_model_ready(config: dict[str, Any]) -> tuple[bool, list[str]]:
    checkpoint = PROJECT_ROOT / config["model"]["local_checkpoint"]
    missing = []
    for name in LOCATEANYTHING_WEIGHT_FILES:
        path = checkpoint / name
        if not path.exists():
            missing.append(str(path))
        elif is_lfs_pointer(path):
            missing.append(f"{path} is only an LFS pointer")
    return not missing, missing


def runtime_ready() -> tuple[bool, list[str]]:
    missing = []
    for module_name in ["torch", "transformers", "PIL", "numpy"]:
        try:
            __import__(module_name)
        except Exception as exc:
            missing.append(f"{module_name}: {exc.__class__.__name__}: {exc}")
    if not missing:
        import torch

        if not torch.cuda.is_available():
            missing.append("CUDA GPU is not available through torch.cuda")
    return not missing, missing


def selected_benchmarks(config: dict[str, Any], requested: list[str] | None) -> list[tuple[str, dict[str, Any]]]:
    benchmarks = config["benchmarks"]
    names = requested or sorted(benchmarks, key=lambda name: benchmarks[name].get("priority") or 999)
    selected = []
    for name in names:
        if name not in benchmarks:
            raise SystemExit(f"Unknown benchmark '{name}'. Available: {', '.join(sorted(benchmarks))}")
        selected.append((name, benchmarks[name]))
    return selected


def display_class_name(class_name: str) -> str:
    normalized = normalize_class_name(class_name)
    return PROMPT_ALIASES.get(normalized, class_name.strip().lower())


def class_specs_from_names(class_names: Sequence[str]) -> list[dict[str, str]]:
    specs = []
    seen = set()
    for raw in class_names:
        canonical = normalize_class_name(raw)
        if canonical in seen:
            continue
        specs.append(
            {
                "canonical": canonical,
                "prompt": display_class_name(raw),
                "lae_name": raw,
            }
        )
        seen.add(canonical)
    return specs


def infer_annotation_classes(annotation_path: Path, label_field: str) -> list[str]:
    classes = []
    seen = set()
    for record in read_jsonl(annotation_path):
        for obj in record.get("objects", []):
            label = obj.get(label_field) or obj.get("raw_class_name") or obj.get("class_name")
            if not label:
                continue
            normalized = normalize_class_name(str(label))
            if normalized not in seen:
                classes.append(str(label))
                seen.add(normalized)
    return classes


def benchmark_class_specs(name: str, benchmark: dict[str, Any], class_source: str, label_field: str) -> list[dict[str, str]]:
    if class_source == "lae-metainfo" and name in LAE_METAINFOS:
        return class_specs_from_names(LAE_METAINFOS[name])
    annotation_path = PROJECT_ROOT / benchmark["annotation_jsonl"]
    return class_specs_from_names(infer_annotation_classes(annotation_path, label_field))


def build_alias_lookup(class_specs: Sequence[Mapping[str, str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    class_set = {spec["canonical"] for spec in class_specs}
    for spec in class_specs:
        canonical = spec["canonical"]
        variants = {
            canonical,
            spec["prompt"],
            spec["lae_name"],
            canonical.replace("/", " "),
            spec["prompt"].replace("/", " "),
            spec["lae_name"].replace("/", " "),
        }
        for variant in variants:
            lookup[normalize_class_name(variant)] = canonical
    for source, target in EXTRA_OUTPUT_ALIASES.items():
        normalized_target = normalize_class_name(target)
        if normalized_target in class_set:
            lookup[normalize_class_name(source)] = normalized_target
    if "plane" in class_set and "airplane" not in class_set:
        lookup["airplane"] = "plane"
    if "airplane" in class_set and "plane" not in class_set:
        lookup["plane"] = "airplane"
    return lookup


def canonicalize_label(label: str | None, alias_lookup: Mapping[str, str], class_set: set[str]) -> str | None:
    if not label:
        return None
    normalized = normalize_class_name(str(label))
    if normalized in alias_lookup:
        return alias_lookup[normalized]
    collapsed = normalized.replace(" ", "")
    for class_name in class_set:
        if class_name.replace(" ", "") == collapsed:
            return class_name
    return None


def detection_prompt(class_specs: Sequence[Mapping[str, str]], template: str | None = None) -> str:
    categories = "</c>".join(spec["prompt"] for spec in class_specs)
    template = template or "Locate all the instances that matches the following description: {classes}."
    return template.replace("{classes}", categories)


def single_label_prompt(class_prompt: str, template: str) -> str:
    return template.replace("{class_name}", class_prompt)


def resolve_image_path(record: Mapping[str, Any], image_root: Path) -> Path:
    candidates = []
    image_path = record.get("image_path") or record.get("file_name")
    if image_path:
        raw_path = Path(str(image_path))
        candidates.append(raw_path)
        if not raw_path.is_absolute():
            candidates.append(image_root / raw_path)
    file_name = record.get("file_name")
    if file_name:
        candidates.append(image_root / str(file_name))
    image_id = str(record.get("image_id", ""))
    if image_id:
        for suffix in IMAGE_EXTENSIONS:
            candidates.append(image_root / f"{image_id}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for {record.get('dataset')}/{record.get('image_id')}; tried {candidates}")


def load_lae_records(
    name: str,
    benchmark: Mapping[str, Any],
    class_specs: Sequence[Mapping[str, str]],
    label_field: str,
) -> list[dict[str, Any]]:
    annotation_path = PROJECT_ROOT / benchmark["annotation_jsonl"]
    alias_lookup = build_alias_lookup(class_specs)
    class_set = {spec["canonical"] for spec in class_specs}
    records = []
    for record in read_jsonl(annotation_path):
        gt = []
        outside_image = 0
        for obj in record.get("objects", []):
            raw_label = obj.get(label_field) or obj.get("raw_class_name") or obj.get("class_name")
            label = canonicalize_label(str(raw_label) if raw_label is not None else None, alias_lookup, class_set)
            if label is None or label not in class_set:
                continue
            hbox = normalize_hbox(obj["hbox"])
            width = float(record.get("width") or 0)
            height = float(record.get("height") or 0)
            if width and height and (hbox[0] < 0 or hbox[1] < 0 or hbox[2] > width or hbox[3] > height):
                outside_image += 1
            gt.append({"label": label, "hbox": hbox, "raw_label": str(raw_label)})
        records.append(
            {
                "dataset": name,
                "image_id": str(record["image_id"]),
                "split": str(record.get("split", benchmark.get("split", ""))),
                "file_name": str(record.get("file_name", record.get("image_path", record["image_id"]))),
                "image_path": str(record.get("image_path", record.get("file_name", record["image_id"]))),
                "width": float(record.get("width") or 0),
                "height": float(record.get("height") or 0),
                "gt": gt,
                "outside_image_gt_boxes": outside_image,
            }
        )
    return records


def apply_limit(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return records
    return records[:limit]


def apply_shard(records: list[dict[str, Any]], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if num_shards <= 1:
        return records
    return records[shard_index::num_shards]


def _image_dimensions(record: Mapping[str, Any], image_root: Path) -> tuple[int, int]:
    width = int(float(record.get("width") or 0))
    height = int(float(record.get("height") or 0))
    if width > 0 and height > 0:
        return width, height
    from PIL import Image

    image_path = resolve_image_path(record, image_root)
    with Image.open(image_path) as image:
        return image.size


def _axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    step = max(1, tile_size - overlap)
    starts = list(range(0, max(1, length - tile_size + 1), step))
    last = max(0, length - tile_size)
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def tile_windows(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    if tile_size <= 0:
        raise ValueError("--tile-size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("--tile-overlap must satisfy 0 <= overlap < tile_size")
    if width <= tile_size and height <= tile_size:
        return [(0, 0, width, height)]
    windows = []
    for y1 in _axis_starts(height, tile_size, overlap):
        for x1 in _axis_starts(width, tile_size, overlap):
            windows.append((x1, y1, min(width, x1 + tile_size), min(height, y1 + tile_size)))
    return windows


def build_prediction_tasks(
    dataset: str,
    benchmark: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    class_specs: Sequence[Mapping[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    image_root = PROJECT_ROOT / benchmark["image_root"]
    prompt_mode = args.prompt_mode
    tasks: list[dict[str, Any]] = []
    if prompt_mode == "multi_label":
        query = detection_prompt(class_specs, args.multi_label_prompt_template)
        for record in records:
            tasks.append(
                {
                    "dataset": dataset,
                    "record": record,
                    "prompt_mode": prompt_mode,
                    "query": query,
                    "target_class": None,
                    "target_prompt": None,
                    "tile_id": "",
                    "tile_window": None,
                    "classes": [spec["canonical"] for spec in class_specs],
                    "prompt_classes": [spec["prompt"] for spec in class_specs],
                }
            )
        return tasks

    if prompt_mode not in {"single_label", "single_label_tiled"}:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")

    for record in records:
        windows: list[tuple[int, int, int, int]]
        if prompt_mode == "single_label_tiled":
            width, height = _image_dimensions(record, image_root)
            windows = tile_windows(width, height, args.tile_size, args.tile_overlap)
        else:
            windows = [(0, 0, 0, 0)]

        for spec in class_specs:
            query = single_label_prompt(spec["prompt"], args.single_label_prompt_template)
            for tile_idx, window in enumerate(windows):
                tile_id = ""
                tile_window = None
                if prompt_mode == "single_label_tiled":
                    x1, y1, x2, y2 = window
                    tile_id = f"{tile_idx:04d}_{x1}_{y1}_{x2}_{y2}"
                    tile_window = [x1, y1, x2, y2]
                tasks.append(
                    {
                        "dataset": dataset,
                        "record": record,
                        "prompt_mode": prompt_mode,
                        "query": query,
                        "target_class": spec["canonical"],
                        "target_prompt": spec["prompt"],
                        "tile_id": tile_id,
                        "tile_window": tile_window,
                        "classes": None,
                        "prompt_classes": None,
                    }
                )
    return tasks


def prediction_run_key(record: Mapping[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    return (
        str(record.get("dataset", "")),
        str(record.get("image_id", "")),
        str(record.get("decode_mode", "")),
        str(record.get("class_source", "")),
        str(record.get("prompt_mode", "multi_label")),
        str(record.get("target_class") or ""),
        str(record.get("tile_id") or ""),
    )


def target_run_key(
    dataset: str,
    task: Mapping[str, Any],
    decode_mode: str,
    class_source: str,
) -> tuple[str, str, str, str, str, str, str]:
    record = task["record"]
    return (
        dataset,
        str(record["image_id"]),
        decode_mode,
        class_source,
        str(task.get("prompt_mode", "multi_label")),
        str(task.get("target_class") or ""),
        str(task.get("tile_id") or ""),
    )


def read_prediction_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def all_requested_predictions_exist(
    dataset: str,
    tasks: Sequence[Mapping[str, Any]],
    prediction_path: Path,
    decode_mode: str,
    class_source: str,
) -> bool:
    if not prediction_path.exists():
        return False
    completed = {prediction_run_key(record) for record in read_prediction_records(prediction_path)}
    requested = {target_run_key(dataset, task, decode_mode, class_source) for task in tasks}
    return requested.issubset(completed)


def clip_hbox_to_image(box: Sequence[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = normalize_hbox(box)
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def nms_predictions(
    predictions: Sequence[Mapping[str, Any]],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    by_class: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for order, pred in enumerate(predictions):
        by_class[str(pred.get("label", ""))].append((order, pred))

    kept: list[tuple[int, dict[str, Any]]] = []
    for class_predictions in by_class.values():
        candidates = sorted(
            class_predictions,
            key=lambda item: (-float(item[1].get("score", 1.0)), item[0]),
        )
        selected: list[tuple[int, Mapping[str, Any]]] = []
        for order, pred in candidates:
            if all(hbox_iou(pred["hbox"], kept_pred["hbox"]) < iou_threshold for _kept_order, kept_pred in selected):
                selected.append((order, pred))
        kept.extend((order, dict(pred)) for order, pred in selected)

    return [pred for _order, pred in sorted(kept, key=lambda item: item[0])]


def aggregate_prediction_records(
    predictions: Sequence[Mapping[str, Any]],
    prompt_mode: str,
    tile_nms_iou: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    pred_index: dict[tuple[str, str], dict[str, Any]] = {}
    for record in predictions:
        key = (str(record.get("dataset", "")), str(record.get("image_id", "")))
        merged = pred_index.setdefault(
            key,
            {
                "dataset": key[0],
                "image_id": key[1],
                "predictions": [],
            },
        )
        for pred in record.get("predictions", []):
            merged["predictions"].append(dict(pred))

    if prompt_mode == "single_label_tiled":
        for merged in pred_index.values():
            merged["predictions"] = nms_predictions(merged["predictions"], tile_nms_iou)
    return pred_index


def trim_predictions(
    predictions: Sequence[Mapping[str, Any]],
    class_name: str,
    max_detections_per_image: int,
) -> list[Mapping[str, Any]]:
    class_predictions = [pred for pred in predictions if pred.get("label") == class_name]
    if max_detections_per_image > 0:
        class_predictions = class_predictions[:max_detections_per_image]
    return class_predictions


def _ap_from_pr(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    if not recalls:
        return 0.0
    ap = 0.0
    for recall_threshold in [idx / 100.0 for idx in range(101)]:
        eligible = [precision for recall, precision in zip(recalls, precisions) if recall >= recall_threshold]
        ap += max(eligible) if eligible else 0.0
    return ap / 101.0


def average_precision_for_class(
    records: Sequence[Mapping[str, Any]],
    pred_index: Mapping[tuple[str, str], Mapping[str, Any]],
    class_name: str,
    iou_threshold: float,
    max_detections_per_image: int,
) -> tuple[float, int]:
    gt_by_image: dict[tuple[str, str], list[list[float]]] = {}
    detections = []
    npos = 0
    order = 0

    for record in records:
        key = (str(record["dataset"]), str(record["image_id"]))
        gt_boxes = [gt["hbox"] for gt in record["gt"] if gt["label"] == class_name]
        gt_by_image[key] = gt_boxes
        npos += len(gt_boxes)
        prediction = pred_index.get(key, {})
        for pred in trim_predictions(prediction.get("predictions", []), class_name, max_detections_per_image):
            detections.append((float(pred.get("score", 1.0)), order, key, pred["hbox"]))
            order += 1

    if npos == 0:
        return 0.0, 0

    detections.sort(key=lambda item: (-item[0], item[1]))
    matched: dict[tuple[str, str], set[int]] = defaultdict(set)
    tps = []
    fps = []
    for _score, _order, key, pred_box in detections:
        best_gt = None
        best_iou = 0.0
        for gt_idx, gt_box in enumerate(gt_by_image.get(key, [])):
            if gt_idx in matched[key]:
                continue
            iou = hbox_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt is not None and best_iou >= iou_threshold:
            matched[key].add(best_gt)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    cum_tp = 0
    cum_fp = 0
    recalls = []
    precisions = []
    for tp, fp in zip(tps, fps):
        cum_tp += tp
        cum_fp += fp
        recalls.append(cum_tp / npos)
        precisions.append(cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) else 0.0)
    return _ap_from_pr(recalls, precisions), npos


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def f1_for_threshold(
    records: Sequence[Mapping[str, Any]],
    pred_index: Mapping[tuple[str, str], Mapping[str, Any]],
    class_names: Sequence[str],
    iou_threshold: float,
    max_detections_per_image: int,
) -> dict[str, Any]:
    tp = fp = fn = 0
    for record in records:
        prediction = pred_index.get((str(record["dataset"]), str(record["image_id"])), {})
        for class_name in class_names:
            gt_boxes = [gt["hbox"] for gt in record["gt"] if gt["label"] == class_name]
            pred_items = trim_predictions(prediction.get("predictions", []), class_name, max_detections_per_image)
            pred_boxes = [pred["hbox"] for pred in pred_items]
            pred_scores = [float(pred.get("score", 1.0)) for pred in pred_items]
            match = greedy_match(gt_boxes, pred_boxes, iou_threshold, pred_scores)
            tp += match.tp
            fp += match.fp
            fn += match.fn
    return {"tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)}


def locateanything_detection_metrics(
    records: Sequence[Mapping[str, Any]],
    pred_index: Mapping[tuple[str, str], Mapping[str, Any]],
    class_names: Sequence[str],
    max_detections_per_image: int,
) -> dict[str, Any]:
    by_threshold = {
        f"{threshold:.2f}": f1_for_threshold(
            records,
            pred_index,
            class_names,
            threshold,
            max_detections_per_image,
        )
        for threshold in IOU_THRESHOLDS
    }
    mean_keys = ["precision", "recall", "f1"]
    mean_metrics = {
        key: mean(item[key] for item in by_threshold.values()) if by_threshold else 0.0
        for key in mean_keys
    }
    return {
        "description": "LocateAnything paper-style detection metrics: confidence-free box matching with F1 at IoU thresholds.",
        "confidence_required": False,
        "thresholds": list(IOU_THRESHOLDS),
        "f1_iou_0.50": by_threshold.get("0.50", {"tp": 0, "fp": 0, "fn": 0, **prf(0, 0, 0)}),
        "f1_iou_0.75": by_threshold.get("0.75", {"tp": 0, "fp": 0, "fn": 0, **prf(0, 0, 0)}),
        "f1_iou_0.95": by_threshold.get("0.95", {"tp": 0, "fp": 0, "fn": 0, **prf(0, 0, 0)}),
        "f1_iou_mean": mean_metrics,
        "by_threshold": by_threshold,
    }


def summarize_speed(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    latencies = [float(record["latency_ms"]) for record in predictions if record.get("latency_ms") is not None]
    total_boxes = sum(len(record.get("predictions", [])) for record in predictions)
    total_latency_s = sum(latencies) / 1000.0 if latencies else 0.0
    gpu_values = [float(record["gpu_memory_mb"]) for record in predictions if record.get("gpu_memory_mb") is not None]
    return {
        "records": len(predictions),
        "boxes": total_boxes,
        "mean_latency_ms": mean(latencies) if latencies else None,
        "boxes_per_second": total_boxes / total_latency_s if total_latency_s > 0 else None,
        "max_gpu_memory_mb": max(gpu_values) if gpu_values else None,
    }


def size_recall(
    records: Sequence[Mapping[str, Any]],
    pred_index: Mapping[tuple[str, str], Mapping[str, Any]],
    class_names: Sequence[str],
    max_detections_per_image: int,
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {
        "small": {"instances": 0, "tp": 0, "missed": 0},
        "medium": {"instances": 0, "tp": 0, "missed": 0},
        "large": {"instances": 0, "tp": 0, "missed": 0},
    }
    for record in records:
        prediction = pred_index.get((str(record["dataset"]), str(record["image_id"])), {})
        for class_name in class_names:
            gt_items = [gt for gt in record["gt"] if gt["label"] == class_name]
            gt_boxes = [gt["hbox"] for gt in gt_items]
            pred_items = trim_predictions(prediction.get("predictions", []), class_name, max_detections_per_image)
            pred_boxes = [pred["hbox"] for pred in pred_items]
            pred_scores = [float(pred.get("score", 1.0)) for pred in pred_items]
            match = greedy_match(gt_boxes, pred_boxes, 0.5, pred_scores)
            matched_gt = {gt_idx for gt_idx, _pred_idx in match.matched_pairs}
            for gt_idx, gt in enumerate(gt_items):
                bucket = size_bucket(gt["hbox"], float(record.get("width") or 0), float(record.get("height") or 0))
                stats[bucket]["instances"] += 1
                if gt_idx in matched_gt:
                    stats[bucket]["tp"] += 1
                else:
                    stats[bucket]["missed"] += 1
    for bucket_stats in stats.values():
        instances = bucket_stats["instances"]
        bucket_stats["recall"] = bucket_stats["tp"] / instances if instances else 0.0
    return stats


def evaluate_lae_style(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    class_specs: Sequence[Mapping[str, str]],
    benchmark_name: str,
    class_source: str,
    label_field: str,
    max_detections_per_image: int,
    prompt_mode: str = "multi_label",
    multi_label_prompt_template: str | None = None,
    single_label_prompt_template: str | None = None,
    tile_size: int = 1024,
    tile_overlap: int = 256,
    tile_nms_iou: float = 0.5,
) -> dict[str, Any]:
    pred_index = aggregate_prediction_records(predictions, prompt_mode, tile_nms_iou)
    post_merge_boxes = sum(len(record.get("predictions", [])) for record in pred_index.values())
    if prompt_mode == "multi_label":
        protocol_name = "LocateAnything-MultiLabel LAE-style horizontal-box detection"
        prompt_note = "One full-vocabulary prompt per image; categories joined with </c>."
    elif prompt_mode == "single_label":
        protocol_name = "LocateAnything-SingleLabel class-wise horizontal-box detection"
        prompt_note = "One prompt per image/class, including absent classes so false positives count."
    elif prompt_mode == "single_label_tiled":
        protocol_name = "LocateAnything-SingleLabel+Tile class-wise horizontal-box detection"
        prompt_note = "One prompt per tile/image/class; tile predictions are mapped to full image and merged with per-class NMS."
    else:
        protocol_name = f"LocateAnything-{prompt_mode} horizontal-box detection"
        prompt_note = "Custom prompt mode."

    if not records:
        primary_name = "AP50" if benchmark_name == "DIOR" else "mAP_50_95"
        return {
            "protocol": {
                "name": protocol_name,
                "prompt": prompt_note,
                "multi_label_prompt_template": multi_label_prompt_template if prompt_mode == "multi_label" else None,
                "single_label_prompt_template": single_label_prompt_template if prompt_mode in {"single_label", "single_label_tiled"} else None,
                "prompt_mode": prompt_mode,
                "class_source": class_source,
                "gt_label_field": label_field,
                "max_detections_per_image": max_detections_per_image,
                "tile_size": tile_size if prompt_mode == "single_label_tiled" else None,
                "tile_overlap": tile_overlap if prompt_mode == "single_label_tiled" else None,
                "tile_nms_iou": tile_nms_iou if prompt_mode == "single_label_tiled" else None,
                "oriented_box_metric": "not_computed",
                "note": "Empty shard or empty evaluation subset.",
            },
            "dataset": benchmark_name,
            "images": 0,
            "predicted_images": len(pred_index),
            "prediction_records": len(predictions),
            "classes": len(class_specs),
            "classes_with_gt": 0,
            "classes_without_gt": [spec["canonical"] for spec in class_specs],
            "gt_instances": 0,
            "predicted_boxes": post_merge_boxes,
            "raw_predicted_boxes": sum(len(record.get("predictions", [])) for record in predictions),
            "lae_table_metric": {
                "name": primary_name,
                "value": 0.0,
                "value_percent": 0.0,
            },
            "accuracy": {
                "ap50": 0.0,
                "ap75": 0.0,
                "map_50_95": 0.0,
                "f1_iou_0.50": {"tp": 0, "fp": 0, "fn": 0, **prf(0, 0, 0)},
                "f1_iou_0.75": {"tp": 0, "fp": 0, "fn": 0, **prf(0, 0, 0)},
                "f1_iou_0.95": {"tp": 0, "fp": 0, "fn": 0, **prf(0, 0, 0)},
            },
            "la_metrics": locateanything_detection_metrics([], {}, [], max_detections_per_image),
            "small_object": {
                "small": {"instances": 0, "tp": 0, "missed": 0, "recall": 0.0},
                "medium": {"instances": 0, "tp": 0, "missed": 0, "recall": 0.0},
                "large": {"instances": 0, "tp": 0, "missed": 0, "recall": 0.0},
            },
            "per_class": {},
            "unknown_predicted_labels": {},
            "outside_image_gt_boxes": 0,
            "speed": summarize_speed(predictions),
        }

    class_names = [spec["canonical"] for spec in class_specs]
    gt_counts = defaultdict(int)
    for record in records:
        for gt in record["gt"]:
            gt_counts[gt["label"]] += 1

    per_class: dict[str, dict[str, Any]] = {}
    ap_by_threshold = {threshold: [] for threshold in IOU_THRESHOLDS}
    for class_name in class_names:
        class_metrics: dict[str, Any] = {"gt_instances": gt_counts[class_name]}
        for threshold in IOU_THRESHOLDS:
            ap, npos = average_precision_for_class(
                records,
                pred_index,
                class_name,
                threshold,
                max_detections_per_image,
            )
            class_metrics[f"ap_{threshold:.2f}"] = ap
            if npos > 0:
                ap_by_threshold[threshold].append(ap)
        per_class[class_name] = class_metrics

    ap50 = mean(ap_by_threshold[0.5]) if ap_by_threshold[0.5] else 0.0
    ap75 = mean(ap_by_threshold[0.75]) if ap_by_threshold[0.75] else 0.0
    threshold_means = [mean(values) for values in ap_by_threshold.values() if values]
    map_50_95 = mean(threshold_means) if threshold_means else 0.0
    f1_metrics = {
        f"f1_iou_{threshold:.2f}": f1_for_threshold(
            records,
            pred_index,
            class_names,
            threshold,
            max_detections_per_image,
        )
        for threshold in [0.5, 0.75, 0.95]
    }
    la_metrics = locateanything_detection_metrics(records, pred_index, class_names, max_detections_per_image)

    unknown_labels = defaultdict(int)
    for record in predictions:
        for label, count in record.get("unknown_predicted_labels", {}).items():
            unknown_labels[label] += int(count)

    primary = ap50 if benchmark_name == "DIOR" else map_50_95
    primary_name = "AP50" if benchmark_name == "DIOR" else "mAP_50_95"
    return {
        "protocol": {
            "name": protocol_name,
            "prompt": prompt_note,
            "multi_label_prompt_template": multi_label_prompt_template if prompt_mode == "multi_label" else None,
            "single_label_prompt_template": single_label_prompt_template if prompt_mode in {"single_label", "single_label_tiled"} else None,
            "prompt_mode": prompt_mode,
            "class_source": class_source,
            "gt_label_field": label_field,
            "max_detections_per_image": max_detections_per_image,
            "tile_size": tile_size if prompt_mode == "single_label_tiled" else None,
            "tile_overlap": tile_overlap if prompt_mode == "single_label_tiled" else None,
            "tile_nms_iou": tile_nms_iou if prompt_mode == "single_label_tiled" else None,
            "oriented_box_metric": "not_computed",
            "note": "LAE-DINO uses MMDetection CocoMetric bbox. This evaluator reports COCO-like hbox AP without category scores because LocateAnything text outputs do not expose confidence scores.",
        },
        "dataset": benchmark_name,
        "images": len(records),
        "predicted_images": len(pred_index),
        "prediction_records": len(predictions),
        "classes": len(class_names),
        "classes_with_gt": sum(1 for class_name in class_names if gt_counts[class_name] > 0),
        "classes_without_gt": [class_name for class_name in class_names if gt_counts[class_name] == 0],
        "gt_instances": sum(gt_counts.values()),
        "predicted_boxes": post_merge_boxes,
        "raw_predicted_boxes": sum(len(record.get("predictions", [])) for record in predictions),
        "lae_table_metric": {
            "name": primary_name,
            "value": primary,
            "value_percent": primary * 100.0,
        },
        "accuracy": {
            "ap50": ap50,
            "ap75": ap75,
            "map_50_95": map_50_95,
            **f1_metrics,
        },
        "la_metrics": la_metrics,
        "small_object": size_recall(records, pred_index, class_names, max_detections_per_image),
        "per_class": per_class,
        "unknown_predicted_labels": dict(sorted(unknown_labels.items())),
        "outside_image_gt_boxes": sum(int(record.get("outside_image_gt_boxes", 0)) for record in records),
        "speed": summarize_speed(predictions),
    }


def run_inference(
    name: str,
    benchmark: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    class_specs: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
    args: argparse.Namespace,
    prediction_path: Path,
) -> None:
    if not tasks:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.touch(exist_ok=True)
        print(f"{name}: no prediction tasks assigned to this worker")
        return

    image_root = PROJECT_ROOT / benchmark["image_root"]
    mode = args.generation_mode or config["model"].get("default_generation_mode", "hybrid")
    alias_lookup = build_alias_lookup(class_specs)
    class_set = {spec["canonical"] for spec in class_specs}
    completed_keys: set[tuple[str, str, str, str, str, str, str]] = set()
    if args.resume and prediction_path.exists():
        completed_keys = {prediction_run_key(record) for record in read_prediction_records(prediction_path)}
        print(f"{name}: resume enabled, found {len(completed_keys)} completed prediction tasks")

    remaining = [
        task
        for task in tasks
        if target_run_key(name, task, mode, args.class_source) not in completed_keys
    ]
    if not remaining:
        print(f"{name}: all {len(tasks)} requested prediction tasks already exist")
        return

    import torch
    from PIL import Image

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    resolved_model_path = model_path(config, args.use_hf_model_id, args.model_path)
    worker = LocateAnythingWorker.from_args(
        model_path=resolved_model_path,
        device=args.device,
        dtype=args.dtype,
    )
    image_processor = getattr(worker.processor, "image_processor", None)
    if image_processor is None or not hasattr(image_processor, "in_token_limit"):
        raise RuntimeError(
            "LocateAnything processor does not expose image_processor.in_token_limit; "
            "cannot verify the effective image-token budget."
        )
    if args.image_token_limit is not None:
        image_processor.in_token_limit = args.image_token_limit
    effective_image_token_limit = int(image_processor.in_token_limit)
    print(f"{name}: effective image-token limit = {effective_image_token_limit}")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    completed = len(tasks) - len(remaining)
    write_mode = "a" if args.resume and prediction_path.exists() else "w"
    with prediction_path.open(write_mode, encoding="utf-8") as handle:
        for task in remaining:
            record = task["record"]
            image_path = resolve_image_path(record, image_root)
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            inference_image = image
            parse_width, parse_height = width, height
            tile_window = task.get("tile_window")
            if tile_window:
                x1, y1, x2, y2 = [int(value) for value in tile_window]
                inference_image = image.crop((x1, y1, x2, y2))
                parse_width, parse_height = inference_image.size
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            output = worker.predict(
                inference_image,
                str(task["query"]),
                generation_mode=mode,
                max_new_tokens=args.max_new_tokens or int(config["model"].get("max_new_tokens", 8192)),
                temperature=args.temperature if args.temperature is not None else float(config["model"].get("temperature", 0.7)),
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            answer = str(output.get("answer", ""))
            raw_boxes = parse_locateanything_labeled_boxes(answer, parse_width, parse_height)
            predictions = []
            unknown_labels = defaultdict(int)
            for pred in raw_boxes:
                label = canonicalize_label(pred.get("label"), alias_lookup, class_set)
                if label is None:
                    unknown_labels[str(pred.get("label") or "<missing>")] += 1
                    if task.get("prompt_mode") in {"single_label", "single_label_tiled"}:
                        label = str(task["target_class"])
                    else:
                        continue
                hbox = pred["hbox"]
                if tile_window:
                    x1, y1, _x2, _y2 = [int(value) for value in tile_window]
                    hbox = [hbox[0] + x1, hbox[1] + y1, hbox[2] + x1, hbox[3] + y1]
                clipped_hbox = clip_hbox_to_image(hbox, width, height)
                if clipped_hbox is None:
                    continue
                predictions.append(
                    {
                        "label": label,
                        "hbox": clipped_hbox,
                        "score": float(pred.get("score", 1.0)),
                        "raw_label": pred.get("label"),
                        "target_class": task.get("target_class"),
                        "tile_id": task.get("tile_id") or None,
                    }
                )
            stats = output.get("stats") or {}
            gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else None
            output_record = {
                "dataset": name,
                "image_id": record["image_id"],
                "file_name": record["file_name"],
                "image_path": str(image_path),
                "query": str(task["query"]),
                "prompt_mode": task["prompt_mode"],
                "target_class": task.get("target_class"),
                "target_prompt": task.get("target_prompt"),
                "tile_id": task.get("tile_id") or None,
                "tile_window": task.get("tile_window"),
                "class_source": args.class_source,
                "classes": task.get("classes"),
                "prompt_classes": task.get("prompt_classes"),
                "model": "LocateAnything-3B",
                "decode_mode": mode,
                "latency_ms": latency_ms,
                "tokens_generated": stats.get("tokens_generated") if isinstance(stats, dict) else None,
                "gpu_memory_mb": gpu_memory_mb,
                "output": answer,
                "predictions": predictions,
                "unknown_predicted_labels": dict(unknown_labels),
                "meta": {
                    "raw_stats": stats if isinstance(stats, dict) else {},
                    "width": width,
                    "height": height,
                    "image_token_limit": effective_image_token_limit,
                    "tile_size": args.tile_size if task.get("prompt_mode") == "single_label_tiled" else None,
                    "tile_overlap": args.tile_overlap if task.get("prompt_mode") == "single_label_tiled" else None,
                },
            }
            handle.write(json.dumps(output_record, sort_keys=True) + "\n")
            handle.flush()
            completed += 1
            if completed % args.log_every == 0:
                print(f"{name}: completed {completed}/{len(tasks)} prediction tasks")


def preflight_benchmark(name: str, benchmark: Mapping[str, Any]) -> list[str]:
    missing = []
    annotation_path = PROJECT_ROOT / benchmark["annotation_jsonl"]
    image_root = PROJECT_ROOT / benchmark["image_root"]
    if not annotation_path.exists():
        missing.append(f"missing annotation JSONL: {annotation_path}")
    if not image_root.exists() or not any(image_root.iterdir()):
        missing.append(f"missing extracted images: {image_root}")
    if name not in LAE_METAINFOS:
        missing.append(f"{name}: no LAE metainfo class list is embedded; use --class-source annotations")
    return missing


def output_paths(
    config: Mapping[str, Any],
    args: argparse.Namespace,
    benchmark_name: str,
) -> tuple[Path, Path]:
    prediction_dir = PROJECT_ROOT / "runs" / "lae_style_predictions"
    metrics_dir = PROJECT_ROOT / "runs" / "lae_style_metrics"
    if args.limit:
        prediction_dir = prediction_dir / f"limit_{args.limit}"
        metrics_dir = metrics_dir / f"limit_{args.limit}"
    if args.num_shards > 1:
        prediction_dir = prediction_dir / f"shards_{args.num_shards}"
        metrics_dir = metrics_dir / f"shards_{args.num_shards}"
    shard_suffix = "" if args.num_shards == 1 else f"_shard{args.shard_index:02d}of{args.num_shards:02d}"
    mode = args.generation_mode or config["model"].get("default_generation_mode", "hybrid")
    run_tag = str(getattr(args, "run_tag", "") or "").strip().replace("/", "_").replace(" ", "_")
    tag_suffix = f"_{run_tag}" if run_tag else ""
    prefix = f"{benchmark_name}_LocateAnything-3B{tag_suffix}_{args.prompt_mode}_{args.class_source}_{mode}{shard_suffix}"
    return prediction_dir / f"{prefix}.jsonl", metrics_dir / f"{prefix}_metrics.json"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run/evaluate LocateAnything using LAE-style detector metrics")
    parser.add_argument("--config", default="config/lae_benchmarks.json")
    parser.add_argument("--benchmark", action="append", help="Benchmark to run. Default: priority order from config.")
    parser.add_argument("--predictions", help="Evaluate an existing LAE-style prediction JSONL instead of running inference")
    parser.add_argument("--run", action="store_true", help="Run real model inference")
    parser.add_argument("--run-tag", default="", help="Optional filename tag to keep runs/checkpoints separate.")
    parser.add_argument("--use-hf-model-id", action="store_true", help="Load nvidia/LocateAnything-3B from HF/cache instead of local checkpoint")
    parser.add_argument("--model-path", help="Explicit model/checkpoint path. Overrides config and MODEL_PATH env var.")
    parser.add_argument("--generation-mode", choices=["fast", "slow", "hybrid"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--image-token-limit",
        type=int,
        help="Override processor.image_processor.in_token_limit for inference only.",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES, default="multi_label")
    parser.add_argument(
        "--multi-label-prompt-template",
        default="Locate all the instances that matches the following description: {classes}.",
        help="Template used by multi-label mode. Must contain {classes}.",
    )
    parser.add_argument(
        "--single-label-prompt-template",
        default="Locate all instances of {class_name} in the image.",
        help="Template used by single-label modes. Must contain {class_name}.",
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=256)
    parser.add_argument("--tile-nms-iou", type=float, default=0.5)
    parser.add_argument("--class-source", choices=["lae-metainfo", "annotations"], default="lae-metainfo")
    parser.add_argument("--gt-label-field", default="raw_class_name", help="GT object label field. Default keeps DOTA small/large vehicle labels.")
    parser.add_argument("--max-detections-per-image", type=int, default=100, help="COCO-style maxDets cap per image per class")
    parser.add_argument("--limit", type=int, help="Optional image limit for smoke tests")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, num_shards)")
    if args.prompt_mode == "multi_label" and "{classes}" not in args.multi_label_prompt_template:
        raise SystemExit("--multi-label-prompt-template must contain {classes}")
    if args.prompt_mode in {"single_label", "single_label_tiled"} and "{class_name}" not in args.single_label_prompt_template:
        raise SystemExit("--single-label-prompt-template must contain {class_name}")
    if args.tile_size <= 0:
        raise SystemExit("--tile-size must be positive")
    if args.image_token_limit is not None and args.image_token_limit <= 0:
        raise SystemExit("--image-token-limit must be positive")
    if args.tile_overlap < 0 or args.tile_overlap >= args.tile_size:
        raise SystemExit("--tile-overlap must satisfy 0 <= overlap < tile_size")

    config = read_json(PROJECT_ROOT / args.config)
    mode = args.generation_mode or config["model"].get("default_generation_mode", "hybrid")

    for name, benchmark in selected_benchmarks(config, args.benchmark):
        class_specs = benchmark_class_specs(name, benchmark, args.class_source, args.gt_label_field)
        all_records = load_lae_records(name, benchmark, class_specs, args.gt_label_field)
        limited_records = apply_limit(all_records, args.limit)
        all_tasks = build_prediction_tasks(name, benchmark, limited_records, class_specs, args)
        tasks = apply_shard(all_tasks, args.num_shards, args.shard_index)
        prediction_path, metrics_path = output_paths(config, args, name)
        if args.predictions:
            prediction_path = Path(args.predictions)

        if args.predictions:
            predictions = read_prediction_records(prediction_path)
            metrics = evaluate_lae_style(
                limited_records,
                predictions,
                class_specs,
                name,
                args.class_source,
                args.gt_label_field,
                args.max_detections_per_image,
                args.prompt_mode,
                multi_label_prompt_template=args.multi_label_prompt_template,
                single_label_prompt_template=args.single_label_prompt_template,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                tile_nms_iou=args.tile_nms_iou,
            )
            metrics["protocol"]["image_token_limit"] = args.image_token_limit
            write_metrics(metrics_path, metrics)
            print(f"{name}: evaluated existing {args.prompt_mode} predictions -> {metrics_path}")
            print(json.dumps(metrics["lae_table_metric"], indent=2, sort_keys=True))
            continue

        missing = preflight_benchmark(name, benchmark)
        if args.run:
            if not tasks:
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                prediction_path.touch(exist_ok=True)
                metrics = evaluate_lae_style(
                    limited_records,
                    [],
                    class_specs,
                    name,
                    args.class_source,
                    args.gt_label_field,
                    args.max_detections_per_image,
                    args.prompt_mode,
                    multi_label_prompt_template=args.multi_label_prompt_template,
                    single_label_prompt_template=args.single_label_prompt_template,
                    tile_size=args.tile_size,
                    tile_overlap=args.tile_overlap,
                    tile_nms_iou=args.tile_nms_iou,
                )
                metrics["protocol"]["image_token_limit"] = args.image_token_limit
                write_metrics(metrics_path, metrics)
                print(f"{name}: no prediction tasks assigned to this shard")
                print(f"{name}: wrote empty {args.prompt_mode} predictions -> {prediction_path}")
                print(f"{name}: wrote empty {args.prompt_mode} metrics -> {metrics_path}")
                print(json.dumps(metrics["lae_table_metric"], indent=2, sort_keys=True))
                continue
            if args.resume and all_requested_predictions_exist(name, tasks, prediction_path, mode, args.class_source):
                print(f"{name}: all requested {args.prompt_mode} predictions already exist; evaluating without model reload")
            else:
                if args.class_source == "annotations":
                    missing = [item for item in missing if "no LAE metainfo" not in item]
                if not args.use_hf_model_id:
                    ready, model_missing = local_model_ready(config)
                    if not ready:
                        missing.extend(model_missing)
                runtime_ok, runtime_missing = runtime_ready()
                if not runtime_ok:
                    missing.extend(runtime_missing)
                if missing:
                    print(f"{name}: cannot run real LocateAnything {args.prompt_mode} inference yet:")
                    for item in missing:
                        print(f"  - {item}")
                    raise SystemExit(2)
                run_inference(name, benchmark, tasks, class_specs, config, args, prediction_path)

            predictions = read_prediction_records(prediction_path)
            metrics = evaluate_lae_style(
                limited_records,
                predictions,
                class_specs,
                name,
                args.class_source,
                args.gt_label_field,
                args.max_detections_per_image,
                args.prompt_mode,
                multi_label_prompt_template=args.multi_label_prompt_template,
                single_label_prompt_template=args.single_label_prompt_template,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                tile_nms_iou=args.tile_nms_iou,
            )
            metrics["protocol"]["image_token_limit"] = args.image_token_limit
            write_metrics(metrics_path, metrics)
            print(f"{name}: wrote {args.prompt_mode} predictions -> {prediction_path}")
            print(f"{name}: wrote {args.prompt_mode} metrics -> {metrics_path}")
            print(json.dumps(metrics["lae_table_metric"], indent=2, sort_keys=True))
        else:
            prompt = detection_prompt(class_specs, args.multi_label_prompt_template) if args.prompt_mode == "multi_label" else args.single_label_prompt_template
            print(f"{name}: {args.prompt_mode} preflight only")
            print(f"  images: {len(limited_records)}")
            print(f"  inference_tasks: {len(all_tasks)}")
            print(f"  assigned_tasks: {len(tasks)}")
            print(f"  class_source: {args.class_source}")
            print(f"  gt_label_field: {args.gt_label_field}")
            print(f"  classes: {len(class_specs)}")
            print(f"  prompt_chars: {len(prompt)}")
            if args.prompt_mode == "single_label_tiled":
                print(f"  tile_size: {args.tile_size}")
                print(f"  tile_overlap: {args.tile_overlap}")
                print(f"  tile_nms_iou: {args.tile_nms_iou}")
            print(f"  would write predictions: {prediction_path}")
            print(f"  would write metrics: {metrics_path}")
            if args.num_shards > 1:
                print(f"  shard: {args.shard_index}/{args.num_shards}")
            if missing and args.class_source != "annotations":
                print("  missing for real run:")
                for item in missing:
                    print(f"    - {item}")
            else:
                print("  benchmark files are ready; pass --run to launch model inference")


if __name__ == "__main__":
    main()
