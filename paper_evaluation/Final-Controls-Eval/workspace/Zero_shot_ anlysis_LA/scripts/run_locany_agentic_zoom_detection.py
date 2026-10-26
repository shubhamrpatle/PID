#!/usr/bin/env python3
"""Agentic multi-scale LocateAnything detection for geospatial images.

This is a diagnostic framework, not a new benchmark protocol by default.  It
queries LocateAnything class-wise over a recursive image pyramid:

1. Query the full image.
2. Split into quadrants and query each quadrant.
3. Recursively zoom into active quadrants.
4. Map tile boxes back to full-image coordinates.
5. Merge duplicates and assign a simple support-based pseudo-confidence.

Use ``--query-class-mode all`` only when you want benchmark-style behavior.
The default ``present`` mode queries GT-present classes only so that dense and
small-object failure cases can be studied cheaply.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_locateanything_lae_style import (  # noqa: E402
    benchmark_class_specs,
    build_alias_lookup,
    canonicalize_label,
    clip_hbox_to_image,
    evaluate_lae_style,
    load_lae_records,
    read_json,
    resolve_image_path,
    selected_benchmarks,
)
from zero_shot_analysis.geometry import greedy_match, hbox_iou  # noqa: E402
from zero_shot_analysis.locate_anything import LocateAnythingWorker, parse_locateanything_labeled_boxes  # noqa: E402


Window = tuple[int, int, int, int]


CLASS_CONTEXTS = {
    "airplane": "aircraft on runways, aprons, or airport parking areas",
    "airport": "runways, aprons, taxiways, terminal infrastructure, and aircraft parking areas",
    "baseballfield": "baseball diamond and outfield sports ground",
    "basketballcourt": "rectangular court markings, often grouped near sports facilities",
    "bridge": "linear structure crossing water, roads, railways, or valleys",
    "chimney": "tall industrial stack casting a narrow shadow",
    "dam": "barrier structure across water or river channels",
    "expressway service area": "service buildings and parking areas beside an expressway",
    "expressway toll station": "toll-gate structure spanning lanes on an expressway",
    "golffield": "large green golf-course area with fairways and holes",
    "groundtrackfield": "oval running track with central sports field",
    "harbor": "coastal docking and port infrastructure, often near ships and water",
    "overpass": "road or rail crossing above another transport corridor",
    "ship": "elongated vessel on water or docked near harbor infrastructure",
    "stadium": "large sports venue, often oval or rectangular with stands",
    "storagetank": "circular or cylindrical industrial tank, often clustered in industrial areas",
    "storage tank": "circular or cylindrical industrial tank, often clustered in industrial areas",
    "tenniscourt": "small rectangular court with clear boundary markings, often in groups",
    "trainstation": "railway platforms, station buildings, and parallel rail lines",
    "vehicle": "small rectangular road or parking objects such as cars, trucks, and buses",
    "windmill": "wind turbine tower and rotor, often in open areas",
    "plane": "aircraft on runways, aprons, or airport parking areas",
    "ship": "elongated vessel on water or docked near harbor infrastructure",
    "small vehicle": "small cars or vans on roads, parking lots, or dense vehicle rows",
    "large vehicle": "larger trucks, buses, trailers, or heavy vehicles",
    "helicopter": "rotor aircraft on helipads, airports, ships, or open paved areas",
    "container crane": "large port crane structure near containers, ships, and dock edges",
    "container": "rectangular shipping containers in port or storage yards",
}

SMALL_OBJECT_CLASSES = {
    "small vehicle",
    "large vehicle",
    "vehicle",
    "storagetank",
    "storage tank",
    "ship",
    "plane",
    "airplane",
    "helicopter",
    "container crane",
}

MEDIUM_OBJECT_CLASSES = {
    "baseballfield",
    "baseball diamond",
    "basketballcourt",
    "basketball court",
    "tenniscourt",
    "tennis court",
    "swimming pool",
    "roundabout",
    "bridge",
    "overpass",
}

# Conservative remote-sensing size priors in full-image area ratio.
# These are not confidence scores. They are only geometric sanity checks for
# obvious impossible boxes from recursive crops.
RS_SIZE_PRIORS = {
    "small vehicle": (0.000003, 0.0030, 0.12, 8.0),
    "large vehicle": (0.000010, 0.0080, 0.10, 10.0),
    "vehicle": (0.000003, 0.0060, 0.10, 10.0),
    "storagetank": (0.000003, 0.0060, 0.35, 3.0),
    "storage tank": (0.000003, 0.0060, 0.35, 3.0),
    "ship": (0.000010, 0.0300, 0.05, 20.0),
    "plane": (0.000020, 0.0300, 0.15, 8.0),
    "airplane": (0.000020, 0.0300, 0.15, 8.0),
    "helicopter": (0.000005, 0.0100, 0.15, 8.0),
    "container crane": (0.000005, 0.0100, 0.05, 20.0),
    "bridge": (0.000020, 0.0800, 0.03, 30.0),
    "overpass": (0.000020, 0.0800, 0.03, 30.0),
    "harbor": (0.000500, 0.3000, 0.10, 10.0),
    "airport": (0.001000, 0.5000, 0.10, 10.0),
}


@dataclass(frozen=True)
class QueryClass:
    canonical: str
    prompt: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def normalize_context_key(text: str) -> str:
    return "".join(str(text).lower().replace("_", " ").split())


def class_context(canonical: str, prompt: str) -> str:
    for key in {canonical, prompt, normalize_context_key(canonical), normalize_context_key(prompt)}:
        if key in CLASS_CONTEXTS:
            return CLASS_CONTEXTS[key]
    return f"visible {prompt} instances in overhead remote-sensing imagery"


def class_scale_group(class_name: str) -> str:
    normalized = str(class_name).lower().strip()
    compact = normalize_context_key(normalized)
    small = {normalize_context_key(item) for item in SMALL_OBJECT_CLASSES}
    medium = {normalize_context_key(item) for item in MEDIUM_OBJECT_CLASSES}
    if compact in small:
        return "small"
    if compact in medium:
        return "medium"
    return "large"


def class_max_depth(args: argparse.Namespace, class_name: str) -> int:
    if args.strategy != "scale_aware":
        return args.max_depth
    group = class_scale_group(class_name)
    return {
        "small": args.small_class_max_depth,
        "medium": args.medium_class_max_depth,
        "large": args.large_class_max_depth,
    }[group]


def class_expand_all_until_depth(args: argparse.Namespace, class_name: str) -> int:
    if args.strategy != "scale_aware":
        return args.expand_all_until_depth
    group = class_scale_group(class_name)
    return {
        "small": args.small_expand_all_until_depth,
        "medium": args.medium_expand_all_until_depth,
        "large": args.large_expand_all_until_depth,
    }[group]


def make_prompt(class_prompt: str, canonical: str, style: str) -> str:
    if style == "name":
        return f"{class_prompt}."
    if style == "short":
        return f"Find {class_prompt}."
    if style == "satellite":
        return f"In this satellite image tile, find {class_prompt}."
    if style == "context":
        return f"In this satellite image tile, find {class_prompt}: {class_context(canonical, class_prompt)}."
    raise ValueError(f"Unknown prompt style: {style}")


def raw_label_is_absence(raw_label: Any) -> bool:
    """Return true for generated labels that mean "no object".

    LocateAnything sometimes emits malformed absence labels such as
    ``none vehicle`` or ``noneball diamond`` in cropped regions.  If we force
    those boxes to the queried class, patch inference creates many false
    positives.  This helper catches the common absence forms without touching
    valid class aliases.
    """

    text = str(raw_label or "").strip().lower()
    compact = normalize_context_key(text)
    if not compact:
        return False
    return (
        compact == "none"
        or compact.startswith("none")
        or compact.startswith("noobject")
        or text.startswith("no ")
        or text.startswith("not ")
        or text.startswith("absent")
    )


def window_area(window: Window) -> int:
    x1, y1, x2, y2 = window
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def box_center(box: Sequence[float]) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def box_aspect_ratio(box: Sequence[float]) -> float:
    width = max(1e-6, float(box[2]) - float(box[0]))
    height = max(1e-6, float(box[3]) - float(box[1]))
    ratio = width / height
    return max(ratio, 1.0 / ratio)


def center_inside(box: Sequence[float], other: Sequence[float]) -> bool:
    cx, cy = box_center(other)
    return float(box[0]) <= cx <= float(box[2]) and float(box[1]) <= cy <= float(box[3])


def split_window(window: Window, grid: int, min_window_size: int) -> list[Window]:
    x1, y1, x2, y2 = window
    width = x2 - x1
    height = y2 - y1
    if width <= min_window_size or height <= min_window_size:
        return []
    children: list[Window] = []
    for row in range(grid):
        cy1 = y1 + round(row * height / grid)
        cy2 = y1 + round((row + 1) * height / grid)
        for col in range(grid):
            cx1 = x1 + round(col * width / grid)
            cx2 = x1 + round((col + 1) * width / grid)
            if cx2 - cx1 >= min_window_size and cy2 - cy1 >= min_window_size:
                children.append((cx1, cy1, cx2, cy2))
    return children


def window_id(depth: int, index: int, window: Window) -> str:
    return f"d{depth:02d}_{index:04d}_{window[0]}_{window[1]}_{window[2]}_{window[3]}"


def choose_records(
    records: Sequence[Mapping[str, Any]],
    manifest_path: Path | None,
    benchmark: str,
    max_samples: int,
    selection: str,
) -> list[Mapping[str, Any]]:
    if manifest_path and manifest_path.exists():
        wanted: list[str] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if str(row.get("benchmark", "")).strip() == benchmark:
                    wanted.append(str(row["image_id"]))
                    if len(wanted) >= max_samples:
                        break
        by_id = {str(record["image_id"]): record for record in records}
        selected = [by_id[image_id] for image_id in wanted if image_id in by_id]
        if selected:
            return selected

    if selection == "dense":
        ordered = sorted(records, key=lambda record: len(record.get("gt", [])), reverse=True)
    else:
        ordered = list(records)
    return list(ordered[:max_samples])


def present_query_classes(
    record: Mapping[str, Any],
    class_specs: Sequence[Mapping[str, str]],
) -> list[QueryClass]:
    prompt_by_class = {str(spec["canonical"]): str(spec["prompt"]) for spec in class_specs}
    order = {str(spec["canonical"]): idx for idx, spec in enumerate(class_specs)}
    classes = sorted({str(gt["label"]) for gt in record.get("gt", [])}, key=lambda name: (order.get(name, 9999), name))
    return [QueryClass(class_name, prompt_by_class[class_name]) for class_name in classes if class_name in prompt_by_class]


def all_query_classes(class_specs: Sequence[Mapping[str, str]]) -> list[QueryClass]:
    return [QueryClass(str(spec["canonical"]), str(spec["prompt"])) for spec in class_specs]


def custom_query_classes(class_specs: Sequence[Mapping[str, str]], names: str) -> list[QueryClass]:
    lookup = {normalize_context_key(str(spec["canonical"])): spec for spec in class_specs}
    lookup.update({normalize_context_key(str(spec["prompt"])): spec for spec in class_specs})
    selected: list[QueryClass] = []
    for name in [part.strip() for part in names.split(",") if part.strip()]:
        spec = lookup.get(normalize_context_key(name))
        if spec is None:
            raise SystemExit(f"Unknown class '{name}' for --query-classes")
        selected.append(QueryClass(str(spec["canonical"]), str(spec["prompt"])))
    return selected


def query_classes_for_record(
    record: Mapping[str, Any],
    class_specs: Sequence[Mapping[str, str]],
    mode: str,
    custom_classes: str,
) -> list[QueryClass]:
    if mode == "present":
        return present_query_classes(record, class_specs)
    if mode == "all":
        return all_query_classes(class_specs)
    if mode == "custom":
        return custom_query_classes(class_specs, custom_classes)
    raise ValueError(f"Unknown query class mode: {mode}")


def parse_and_project_predictions(
    answer: str,
    parse_width: int,
    parse_height: int,
    full_width: int,
    full_height: int,
    window: Window,
    query_class: QueryClass,
    alias_lookup: Mapping[str, str],
    class_set: set[str],
    force_target_label: bool,
    require_target_label_match: bool,
    reject_absence_labels: bool,
    source: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    default_label = None if require_target_label_match else query_class.canonical
    raw_boxes = parse_locateanything_labeled_boxes(answer, parse_width, parse_height, default_label=default_label)
    unknown_labels: dict[str, int] = defaultdict(int)
    predictions: list[dict[str, Any]] = []
    wx1, wy1, _wx2, _wy2 = window
    for pred in raw_boxes:
        raw_label = pred.get("label")
        if reject_absence_labels and raw_label_is_absence(raw_label):
            unknown_labels[f"absence:{raw_label or '<missing>'}"] += 1
            continue
        label = canonicalize_label(raw_label, alias_lookup, class_set)
        if require_target_label_match:
            if label != query_class.canonical:
                unknown_labels[f"non_target:{raw_label or '<missing>'}"] += 1
                continue
        elif force_target_label:
            label = query_class.canonical
        elif label is None:
            unknown_labels[str(raw_label or "<missing>")] += 1
            continue

        hbox = pred["hbox"]
        full_hbox = [hbox[0] + wx1, hbox[1] + wy1, hbox[2] + wx1, hbox[3] + wy1]
        clipped = clip_hbox_to_image(full_hbox, full_width, full_height)
        if clipped is None:
            continue
        predictions.append(
            {
                "label": label,
                "raw_label": raw_label,
                "hbox": clipped,
                "score": float(pred.get("score", 1.0)),
                "target_class": query_class.canonical,
                **source,
            }
        )
    return predictions, dict(unknown_labels)


def preliminary_score(prediction: Mapping[str, Any]) -> float:
    depth = int(prediction.get("depth") or 0)
    tile_area = float(prediction.get("tile_area") or 1.0)
    image_area = float(prediction.get("image_area") or tile_area)
    zoom_factor = max(1.0, image_area / max(tile_area, 1.0))
    return 1.0 + 0.15 * depth + 0.02 * min(16.0, zoom_factor)


def cluster_predictions(
    predictions: Sequence[Mapping[str, Any]],
    merge_iou: float,
    merge_method: str,
) -> list[dict[str, Any]]:
    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pred in predictions:
        by_class[str(pred.get("label", ""))].append(pred)

    final_predictions: list[dict[str, Any]] = []
    for label, class_predictions in by_class.items():
        ordered = sorted(class_predictions, key=lambda pred: (-preliminary_score(pred), box_area(pred["hbox"])))
        clusters: list[list[Mapping[str, Any]]] = []
        for pred in ordered:
            best_idx = None
            best_iou = 0.0
            for idx, cluster in enumerate(clusters):
                representative = cluster[0]
                iou = hbox_iou(pred["hbox"], representative["hbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is not None and best_iou >= merge_iou:
                clusters[best_idx].append(pred)
            else:
                clusters.append([pred])

        for cluster in clusters:
            support_count = len(cluster)
            source_levels = sorted({int(pred.get("depth") or 0) for pred in cluster})
            max_depth = max(source_levels) if source_levels else 0
            if merge_method == "wbf":
                weights = [preliminary_score(pred) for pred in cluster]
                denom = sum(weights) or 1.0
                merged_box = [
                    sum(float(pred["hbox"][coord]) * weight for pred, weight in zip(cluster, weights)) / denom
                    for coord in range(4)
                ]
                representative = max(cluster, key=preliminary_score)
            else:
                representative = max(cluster, key=preliminary_score)
                merged_box = list(representative["hbox"])

            pseudo_score = min(
                0.99,
                0.20
                + 0.14 * min(support_count, 5)
                + 0.08 * max_depth
                + 0.04 * len(source_levels),
            )
            final_predictions.append(
                {
                    "label": label,
                    "raw_label": representative.get("raw_label"),
                    "hbox": merged_box,
                    "score": pseudo_score,
                    "target_class": representative.get("target_class"),
                    "support_count": support_count,
                    "source_levels": source_levels,
                    "source_windows": [pred.get("tile_window") for pred in cluster],
                    "source_tile_ids": [pred.get("tile_id") for pred in cluster],
                    "merge_method": merge_method,
                }
            )
    return sorted(final_predictions, key=lambda pred: (pred["label"], -float(pred["score"]), pred["hbox"]))


def remove_coarse_containers(
    predictions: Sequence[Mapping[str, Any]],
    min_contained: int,
    area_ratio: float,
) -> list[dict[str, Any]]:
    if min_contained <= 0:
        return [dict(pred) for pred in predictions]
    keep: list[dict[str, Any]] = []
    for idx, pred in enumerate(predictions):
        pred_area = box_area(pred["hbox"])
        contained = 0
        for other_idx, other in enumerate(predictions):
            if idx == other_idx or pred.get("label") != other.get("label"):
                continue
            other_area = box_area(other["hbox"])
            if other_area <= 0 or pred_area < area_ratio * other_area:
                continue
            if center_inside(pred["hbox"], other["hbox"]):
                contained += 1
        if contained < min_contained:
            keep.append(dict(pred))
    return keep


def size_prior_for_label(label: str) -> tuple[float, float, float, float] | None:
    normalized = str(label).lower().strip()
    compact = normalize_context_key(normalized)
    for key, prior in RS_SIZE_PRIORS.items():
        if compact == normalize_context_key(key):
            return prior
    return None


def filter_by_size_priors(
    predictions: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
    source: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if source == "none":
        return [dict(pred) for pred in predictions], {}
    image_area = max(1.0, float(width * height))
    kept: list[dict[str, Any]] = []
    removed: dict[str, int] = defaultdict(int)
    for pred in predictions:
        label = str(pred.get("label", ""))
        prior = size_prior_for_label(label)
        if prior is None:
            kept.append(dict(pred))
            continue
        min_area_ratio, max_area_ratio, min_aspect, max_aspect = prior
        area_ratio = box_area(pred["hbox"]) / image_area
        aspect = box_aspect_ratio(pred["hbox"])
        # min_aspect is kept for readability/future symmetry. Because aspect is
        # normalized to >=1, it only filters if a caller sets it above 1.
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio or aspect < min_aspect or aspect > max_aspect:
            removed[label] += 1
            continue
        kept.append(dict(pred))
    return kept, dict(removed)


def filter_final_predictions(
    predictions: Sequence[Mapping[str, Any]],
    min_support_count: int,
    min_pseudo_score: float,
    max_per_class: int,
) -> list[dict[str, Any]]:
    filtered = [
        dict(pred)
        for pred in predictions
        if int(pred.get("support_count") or 1) >= min_support_count
        and float(pred.get("score") or 0.0) >= min_pseudo_score
    ]
    if max_per_class <= 0:
        return filtered

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in filtered:
        by_class[str(pred.get("label", ""))].append(pred)

    capped: list[dict[str, Any]] = []
    for label in sorted(by_class):
        candidates = sorted(
            by_class[label],
            key=lambda pred: (
                -float(pred.get("score") or 0.0),
                -int(pred.get("support_count") or 1),
                pred.get("hbox", [0, 0, 0, 0]),
            ),
        )
        capped.extend(candidates[:max_per_class])
    return capped


def class_gt_center_in_window(record: Mapping[str, Any], class_name: str, window: Window) -> bool:
    x1, y1, x2, y2 = window
    for gt in record.get("gt", []):
        if str(gt.get("label")) != class_name:
            continue
        cx, cy = box_center(gt["hbox"])
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False


def run_window_query(
    worker: LocateAnythingWorker,
    image: Any,
    query_class: QueryClass,
    window: Window,
    depth: int,
    tile_idx: int,
    args: argparse.Namespace,
    alias_lookup: Mapping[str, str],
    class_set: set[str],
) -> dict[str, Any]:
    import torch

    full_width, full_height = image.size
    x1, y1, x2, y2 = window
    crop = image.crop((x1, y1, x2, y2))
    parse_width, parse_height = crop.size
    query = make_prompt(query_class.prompt, query_class.canonical, args.prompt_style)
    tile = window_id(depth, tile_idx, window)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = worker.predict(
        crop,
        query,
        generation_mode=args.generation_mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=not args.greedy,
        repetition_penalty=args.repetition_penalty,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    answer = str(output.get("answer", ""))
    source = {
        "depth": depth,
        "tile_id": tile,
        "tile_window": [x1, y1, x2, y2],
        "tile_area": window_area(window),
        "image_area": full_width * full_height,
        "query": query,
    }
    predictions, unknown = parse_and_project_predictions(
        answer,
        parse_width,
        parse_height,
        full_width,
        full_height,
        window,
        query_class,
        alias_lookup,
        class_set,
        args.force_target_label,
        args.require_target_label_match,
        args.reject_absence_labels,
        source,
    )
    stats = output.get("stats") or {}
    gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else None
    return {
        "target_class": query_class.canonical,
        "target_prompt": query_class.prompt,
        "query": query,
        "depth": depth,
        "tile_id": tile,
        "tile_window": [x1, y1, x2, y2],
        "latency_ms": latency_ms,
        "gpu_memory_mb": gpu_memory_mb,
        "tokens_generated": stats.get("tokens_generated") if isinstance(stats, dict) else None,
        "output": answer,
        "predictions": predictions,
        "unknown_predicted_labels": unknown,
    }


def run_agentic_class(
    worker: LocateAnythingWorker,
    image: Any,
    record: Mapping[str, Any],
    query_class: QueryClass,
    args: argparse.Namespace,
    alias_lookup: Mapping[str, str],
    class_set: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    full_width, full_height = image.size
    frontier: list[Window] = [(0, 0, full_width, full_height)]
    raw_predictions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    unknown_totals: dict[str, int] = defaultdict(int)
    max_depth = class_max_depth(args, query_class.canonical)
    expand_all_until_depth = class_expand_all_until_depth(args, query_class.canonical)

    for depth in range(max_depth + 1):
        next_frontier: list[Window] = []
        seen_children: set[Window] = set()
        for tile_idx, window in enumerate(frontier):
            trace = run_window_query(worker, image, query_class, window, depth, tile_idx, args, alias_lookup, class_set)
            traces.append(trace)
            raw_predictions.extend(trace["predictions"])
            for label, count in trace.get("unknown_predicted_labels", {}).items():
                unknown_totals[label] += int(count)

            if depth >= max_depth:
                continue
            expand = args.strategy in {"exhaustive", "oracle"} or depth < expand_all_until_depth
            if args.strategy in {"agentic", "scale_aware"} and len(trace["predictions"]) >= args.min_expand_boxes:
                if args.max_expand_boxes <= 0 or len(trace["predictions"]) <= args.max_expand_boxes:
                    expand = True
            if not expand:
                continue
            for child in split_window(window, args.grid, args.min_window_size):
                if args.strategy == "oracle" and not class_gt_center_in_window(record, query_class.canonical, child):
                    continue
                if child not in seen_children:
                    seen_children.add(child)
                    next_frontier.append(child)
        frontier = next_frontier
        if not frontier:
            break
    return raw_predictions, traces, dict(unknown_totals)


def summarize_metrics(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    accuracy = metrics.get("accuracy", {})
    f1 = accuracy.get("f1_iou_0.50", {})
    return [
        {
            "dataset": metrics.get("dataset"),
            "images": metrics.get("images"),
            "gt_instances": metrics.get("gt_instances"),
            "predicted_boxes": metrics.get("predicted_boxes"),
            "primary": metrics.get("lae_table_metric", {}).get("value_percent", 0.0),
            "ap50": accuracy.get("ap50", 0.0) * 100.0,
            "ap75": accuracy.get("ap75", 0.0) * 100.0,
            "map50_95": accuracy.get("map_50_95", 0.0) * 100.0,
            "f1@0.5": f1.get("f1", 0.0) * 100.0,
            "precision@0.5": f1.get("precision", 0.0) * 100.0,
            "recall@0.5": f1.get("recall", 0.0) * 100.0,
            "tp@0.5": f1.get("tp", 0),
            "fp@0.5": f1.get("fp", 0),
            "fn@0.5": f1.get("fn", 0),
        }
    ]


def combined_prediction_index(predictions: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    by_image: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in predictions:
        key = (str(record.get("dataset", "")), str(record.get("image_id", "")))
        for pred in record.get("predictions", []):
            by_image[key].append(dict(pred))
    return by_image


def per_class_f1_rows(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    class_specs: Sequence[Mapping[str, str]],
    max_detections_per_image: int,
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    pred_index = combined_prediction_index(predictions)
    rows: list[dict[str, Any]] = []
    for spec in class_specs:
        class_name = str(spec["canonical"])
        gt_count = pred_count = tp = fp = fn = 0
        for record in records:
            key = (str(record["dataset"]), str(record["image_id"]))
            gt_boxes = [gt["hbox"] for gt in record.get("gt", []) if gt.get("label") == class_name]
            pred_items = [pred for pred in pred_index.get(key, []) if pred.get("label") == class_name]
            if max_detections_per_image > 0:
                pred_items = sorted(
                    pred_items,
                    key=lambda pred: (
                        -float(pred.get("score", 1.0)),
                        -int(pred.get("support_count") or 1),
                    ),
                )[:max_detections_per_image]
            pred_boxes = [pred["hbox"] for pred in pred_items]
            pred_scores = [float(pred.get("score", 1.0)) for pred in pred_items]
            match = greedy_match(gt_boxes, pred_boxes, iou_threshold, pred_scores)
            gt_count += len(gt_boxes)
            pred_count += len(pred_boxes)
            tp += match.tp
            fp += match.fp
            fn += match.fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if gt_count or pred_count:
            rows.append(
                {
                    "class": class_name,
                    "gt_count": gt_count,
                    "pred_count": pred_count,
                    "tp@0.5": tp,
                    "fp@0.5": fp,
                    "fn@0.5": fn,
                    "precision@0.5": precision,
                    "recall@0.5": recall,
                    "f1@0.5": f1,
                }
            )
    return sorted(rows, key=lambda row: (-int(row["gt_count"]), str(row["class"])))


def write_summary_md(path: Path, metrics: Mapping[str, Any], args: argparse.Namespace) -> None:
    row = summarize_metrics(metrics)[0]
    lines = [
        "# LocateAnything Agentic Zoom Detection",
        "",
        "This run uses recursive class-wise zooming. Unless `--query-class-mode all` was used, this is a diagnostic run and not a fair detector benchmark.",
        "",
        f"- Benchmark: `{args.benchmark}`",
        f"- Query class mode: `{args.query_class_mode}`",
        f"- Strategy: `{args.strategy}`",
        f"- Max depth: `{args.max_depth}`",
        f"- Expand-all-until-depth: `{args.expand_all_until_depth}`",
        f"- Scale-aware depths: small `{args.small_class_max_depth}` / medium `{args.medium_class_max_depth}` / large `{args.large_class_max_depth}`",
        f"- Scale-aware exhaustive depths: small `{args.small_expand_all_until_depth}` / medium `{args.medium_expand_all_until_depth}` / large `{args.large_expand_all_until_depth}`",
        f"- Prompt style: `{args.prompt_style}`",
        f"- Force target label: `{args.force_target_label}`",
        f"- Require target-label match: `{args.require_target_label_match}`",
        f"- Reject absence labels: `{args.reject_absence_labels}`",
        f"- Merge IoU / method: `{args.merge_iou}` / `{args.merge_method}`",
        f"- Size prior source: `{args.size_prior_source}`",
        f"- Min support / min pseudo-score: `{args.min_support_count}` / `{args.min_pseudo_score}`",
        f"- Max expand boxes / max final per class: `{args.max_expand_boxes}` / `{args.max_final_per_class}`",
        "",
        "| Dataset | Images | GT | Pred boxes | AP50 | AP75 | mAP50-95 | F1@0.5 | Prec@0.5 | Rec@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {row['dataset']} | {row['images']} | {row['gt_instances']} | {row['predicted_boxes']} | "
            f"{row['ap50']:.2f} | {row['ap75']:.2f} | {row['map50_95']:.2f} | "
            f"{row['f1@0.5']:.2f} | {row['precision@0.5']:.2f} | {row['recall@0.5']:.2f} |"
        ),
        "",
        "Decision rule:",
        "",
        "- Useful: recall and F1 improve while precision remains controlled.",
        "- Risky: recall improves but FP/predicted boxes explode.",
        "- Not useful: small/dense recall does not improve over full-image LA.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_self_test() -> None:
    children = split_window((0, 0, 800, 600), 2, 128)
    assert children == [(0, 0, 400, 300), (400, 0, 800, 300), (0, 300, 400, 600), (400, 300, 800, 600)]
    preds = [
        {"label": "ship", "hbox": [10, 10, 50, 50], "depth": 1, "tile_area": 100, "image_area": 400},
        {"label": "ship", "hbox": [12, 12, 52, 52], "depth": 2, "tile_area": 50, "image_area": 400},
        {"label": "ship", "hbox": [200, 200, 250, 250], "depth": 1, "tile_area": 100, "image_area": 400},
    ]
    merged = cluster_predictions(preds, 0.5, "best")
    assert len(merged) == 2, merged
    assert max(pred["support_count"] for pred in merged) == 2
    assert raw_label_is_absence("none vehicle")
    assert raw_label_is_absence("noneball diamond")
    assert not raw_label_is_absence("small vehicle")
    filtered = remove_coarse_containers(
        [
            {"label": "vehicle", "hbox": [0, 0, 100, 100]},
            {"label": "vehicle", "hbox": [10, 10, 20, 20]},
            {"label": "vehicle", "hbox": [30, 30, 40, 40]},
        ],
        2,
        4.0,
    )
    assert len(filtered) == 2, filtered
    supported = filter_final_predictions(
        [
            {"label": "ship", "hbox": [0, 0, 10, 10], "support_count": 1, "score": 0.9},
            {"label": "ship", "hbox": [20, 20, 30, 30], "support_count": 2, "score": 0.6},
            {"label": "ship", "hbox": [40, 40, 50, 50], "support_count": 3, "score": 0.4},
        ],
        min_support_count=2,
        min_pseudo_score=0.5,
        max_per_class=1,
    )
    assert len(supported) == 1 and supported[0]["hbox"] == [20, 20, 30, 30], supported
    record = {"gt": [{"label": "ship", "hbox": [75, 75, 125, 125]}]}
    assert class_gt_center_in_window(record, "ship", (0, 0, 200, 200))
    assert not class_gt_center_in_window(record, "ship", (200, 200, 400, 400))
    print("Self-test passed.")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run agentic multi-scale LocateAnything detection")
    parser.add_argument("--config", default="config/lae_benchmarks.json")
    parser.add_argument("--benchmark", default="DOTAv2")
    parser.add_argument("--manifest", default=None, help="Optional sample manifest CSV")
    parser.add_argument("--selection", choices=["dense", "first"], default="dense")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-dir", default="runs/agentic_zoom_detection")
    parser.add_argument("--baseline-predictions", default=None, help="Optional existing prediction JSONL to evaluate on the same selected images.")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--generation-mode", choices=["fast", "slow", "hybrid"], default="hybrid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--strategy", choices=["agentic", "scale_aware", "exhaustive", "oracle"], default="agentic")
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--expand-all-until-depth", type=int, default=1)
    parser.add_argument("--small-class-max-depth", type=int, default=2)
    parser.add_argument("--medium-class-max-depth", type=int, default=2)
    parser.add_argument("--large-class-max-depth", type=int, default=1)
    parser.add_argument("--small-expand-all-until-depth", type=int, default=2)
    parser.add_argument("--medium-expand-all-until-depth", type=int, default=1)
    parser.add_argument("--large-expand-all-until-depth", type=int, default=0)
    parser.add_argument("--min-expand-boxes", type=int, default=1)
    parser.add_argument("--max-expand-boxes", type=int, default=0, help="Do not expand a window with more boxes than this; 0 disables the cap")
    parser.add_argument("--min-window-size", type=int, default=128)
    parser.add_argument("--prompt-style", choices=["name", "short", "satellite", "context"], default="context")
    parser.add_argument("--query-class-mode", choices=["present", "all", "custom"], default="present")
    parser.add_argument("--query-classes", default="", help="Comma-separated classes when --query-class-mode custom")
    parser.add_argument("--class-source", choices=["lae-metainfo", "annotations"], default="lae-metainfo")
    parser.add_argument("--gt-label-field", default="raw_class_name")
    parser.add_argument("--force-target-label", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-target-label-match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep class-wise query boxes only when the generated/raw <ref> label canonicalizes to the queried class.",
    )
    parser.add_argument(
        "--reject-absence-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop boxes whose generated label starts with none/no/not/absent.",
    )
    parser.add_argument("--merge-iou", type=float, default=0.55)
    parser.add_argument("--merge-method", choices=["best", "wbf"], default="best")
    parser.add_argument("--container-min-contained", type=int, default=3)
    parser.add_argument("--container-area-ratio", type=float, default=4.0)
    parser.add_argument("--size-prior-source", choices=["none", "rs-defaults"], default="none")
    parser.add_argument("--min-support-count", type=int, default=1)
    parser.add_argument("--min-pseudo-score", type=float, default=0.0)
    parser.add_argument("--max-final-per-class", type=int, default=0, help="Keep only top-K final boxes per class/image; 0 disables")
    parser.add_argument("--max-detections-per-image", type=int, default=100)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.self_test:
        run_self_test()
        return

    config = read_json(PROJECT_ROOT / args.config)
    benchmark_items = dict(selected_benchmarks(config, [args.benchmark]))
    benchmark = benchmark_items[args.benchmark]
    class_specs = benchmark_class_specs(args.benchmark, benchmark, args.class_source, args.gt_label_field)
    alias_lookup = build_alias_lookup(class_specs)
    class_set = {str(spec["canonical"]) for spec in class_specs}
    all_records = load_lae_records(args.benchmark, benchmark, class_specs, args.gt_label_field)
    manifest = PROJECT_ROOT / args.manifest if args.manifest else None
    selected_records = choose_records(all_records, manifest, args.benchmark, args.max_samples, args.selection)
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")
    total_selected_records = len(selected_records)
    if args.num_shards > 1:
        selected_records = selected_records[args.shard_index :: args.num_shards]
    image_root = PROJECT_ROOT / benchmark["image_root"]

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "predictions.jsonl"
    trace_path = output_dir / "query_trace.jsonl"
    metrics_path = output_dir / "metrics.json"
    metrics_csv_path = output_dir / "metrics_summary.csv"
    per_class_path = output_dir / "per_class_metrics.csv"
    summary_path = output_dir / "summary.md"
    comparison_path = output_dir / "comparison.md"
    baseline_metrics_path = output_dir / "baseline_metrics.json"
    plan_path = output_dir / "planned_records.csv"

    plan_rows = []
    min_queries = 0
    for record in selected_records:
        classes = query_classes_for_record(record, class_specs, args.query_class_mode, args.query_classes)
        guaranteed_queries = 0
        scale_groups: dict[str, int] = defaultdict(int)
        for query_class in classes:
            scale_groups[class_scale_group(query_class.canonical)] += 1
            guaranteed_windows = 1
            expand_until = class_expand_all_until_depth(args, query_class.canonical)
            max_depth = class_max_depth(args, query_class.canonical)
            for depth in range(min(expand_until, max_depth)):
                guaranteed_windows += args.grid ** (2 * (depth + 1))
            guaranteed_queries += guaranteed_windows
        min_queries += guaranteed_queries
        plan_rows.append(
            {
                "dataset": record["dataset"],
                "image_id": record["image_id"],
                "file_name": record["file_name"],
                "gt_objects": len(record.get("gt", [])),
                "query_classes": len(classes),
                "small_classes": scale_groups.get("small", 0),
                "medium_classes": scale_groups.get("medium", 0),
                "large_classes": scale_groups.get("large", 0),
                "minimum_queries": guaranteed_queries,
                "query_class_mode": args.query_class_mode,
            }
        )
    write_csv(plan_path, plan_rows)
    if args.num_shards > 1:
        print(
            f"Selected images: {len(selected_records)} "
            f"(shard {args.shard_index}/{args.num_shards} from {total_selected_records})"
        )
    else:
        print(f"Selected images: {len(selected_records)}")
    print(f"Minimum planned queries before active deeper zoom: {min_queries}")
    print(f"Wrote plan: {plan_path}")
    if args.dry_run:
        print("Dry run complete; no model inference launched.")
        return

    completed = {}
    if args.resume and final_path.exists():
        completed = {str(row.get("image_id")): row for row in read_jsonl(final_path)}

    model_path = args.model_path or os.environ.get("MODEL_PATH") or str(PROJECT_ROOT / config["model"]["local_checkpoint"])
    worker = LocateAnythingWorker.from_args(model_path=model_path, device=args.device, dtype=args.dtype)

    final_handle = final_path.open("a" if args.resume and final_path.exists() else "w", encoding="utf-8")
    trace_handle = trace_path.open("a" if args.resume and trace_path.exists() else "w", encoding="utf-8")
    try:
        for image_idx, record in enumerate(selected_records, start=1):
            image_id = str(record["image_id"])
            if image_id in completed:
                print(f"[{image_idx}/{len(selected_records)}] skip completed image_id={image_id}")
                continue
            from PIL import Image

            image_path = resolve_image_path(record, image_root)
            image = Image.open(image_path).convert("RGB")
            query_classes = query_classes_for_record(record, class_specs, args.query_class_mode, args.query_classes)
            all_raw: list[dict[str, Any]] = []
            all_traces: list[dict[str, Any]] = []
            unknown_totals: dict[str, int] = defaultdict(int)
            start = time.perf_counter()
            for class_idx, query_class in enumerate(query_classes, start=1):
                raw, traces, unknown = run_agentic_class(worker, image, record, query_class, args, alias_lookup, class_set)
                all_raw.extend(raw)
                all_traces.extend(traces)
                for label, count in unknown.items():
                    unknown_totals[label] += int(count)
                print(
                    f"[{image_idx}/{len(selected_records)}] image={image_id} "
                    f"class={class_idx}/{len(query_classes)} {query_class.canonical} raw_boxes={len(raw)} queries={len(traces)}"
                )

            merged = cluster_predictions(all_raw, args.merge_iou, args.merge_method)
            cleaned = remove_coarse_containers(merged, args.container_min_contained, args.container_area_ratio)
            cleaned, size_prior_removed = filter_by_size_priors(
                cleaned,
                image.size[0],
                image.size[1],
                args.size_prior_source,
            )
            cleaned = filter_final_predictions(cleaned, args.min_support_count, args.min_pseudo_score, args.max_final_per_class)
            latency_ms = (time.perf_counter() - start) * 1000.0
            final_record = {
                "dataset": args.benchmark,
                "image_id": image_id,
                "file_name": str(record.get("file_name", "")),
                "image_path": str(image_path),
                "model": "LocateAnything-3B",
                "decode_mode": args.generation_mode,
                "prompt_mode": "agentic_zoom",
                "query_class_mode": args.query_class_mode,
                "strategy": args.strategy,
                "max_depth": args.max_depth,
                "prompt_style": args.prompt_style,
                "classes": [query_class.canonical for query_class in query_classes],
                "latency_ms": latency_ms,
                "raw_predicted_boxes": len(all_raw),
                "predictions": cleaned,
                "unknown_predicted_labels": dict(unknown_totals),
                "size_prior_removed": size_prior_removed,
                "meta": {
                    "width": image.size[0],
                    "height": image.size[1],
                    "merge_iou": args.merge_iou,
                    "merge_method": args.merge_method,
                    "container_min_contained": args.container_min_contained,
                    "container_area_ratio": args.container_area_ratio,
                    "min_support_count": args.min_support_count,
                    "min_pseudo_score": args.min_pseudo_score,
                    "max_expand_boxes": args.max_expand_boxes,
                    "max_final_per_class": args.max_final_per_class,
                    "size_prior_source": args.size_prior_source,
                },
            }
            final_handle.write(json.dumps(final_record, sort_keys=True) + "\n")
            final_handle.flush()
            for trace in all_traces:
                trace_record = {
                    "dataset": args.benchmark,
                    "image_id": image_id,
                    "file_name": str(record.get("file_name", "")),
                    **trace,
                }
                trace_handle.write(json.dumps(trace_record, sort_keys=True) + "\n")
            trace_handle.flush()
            print(
                f"[{image_idx}/{len(selected_records)}] image={image_id} "
                f"raw={len(all_raw)} final={len(cleaned)} latency_ms={latency_ms:.1f}"
            )
    finally:
        final_handle.close()
        trace_handle.close()

    predictions = read_jsonl(final_path)
    metrics = evaluate_lae_style(
        selected_records,
        predictions,
        class_specs,
        args.benchmark,
        args.class_source,
        args.gt_label_field,
        args.max_detections_per_image,
        prompt_mode="agentic_zoom",
        multi_label_prompt_template=None,
        single_label_prompt_template=None,
        tile_size=0,
        tile_overlap=0,
        tile_nms_iou=args.merge_iou,
    )
    metrics["protocol"]["name"] = "LocateAnything-AgenticZoom class-wise multi-scale detection"
    metrics["protocol"]["prompt"] = "Recursive class-wise full image, quadrant, and active zoom queries."
    metrics["protocol"]["query_class_mode"] = args.query_class_mode
    metrics["protocol"]["strategy"] = args.strategy
    metrics["protocol"]["max_depth"] = args.max_depth
    metrics["protocol"]["prompt_style"] = args.prompt_style
    metrics["protocol"]["min_support_count"] = args.min_support_count
    metrics["protocol"]["min_pseudo_score"] = args.min_pseudo_score
    metrics["protocol"]["max_expand_boxes"] = args.max_expand_boxes
    metrics["protocol"]["max_final_per_class"] = args.max_final_per_class
    metrics["protocol"]["size_prior_source"] = args.size_prior_source
    if args.strategy == "scale_aware":
        metrics["protocol"]["scale_aware_policy"] = {
            "small": {
                "max_depth": args.small_class_max_depth,
                "expand_all_until_depth": args.small_expand_all_until_depth,
            },
            "medium": {
                "max_depth": args.medium_class_max_depth,
                "expand_all_until_depth": args.medium_expand_all_until_depth,
            },
            "large": {
                "max_depth": args.large_class_max_depth,
                "expand_all_until_depth": args.large_expand_all_until_depth,
            },
        }
    metrics["protocol"]["diagnostic_only"] = args.query_class_mode != "all"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(metrics_csv_path, summarize_metrics(metrics))
    write_csv(per_class_path, per_class_f1_rows(selected_records, predictions, class_specs, args.max_detections_per_image))

    baseline_metrics = None
    if args.baseline_predictions:
        baseline_path = Path(args.baseline_predictions)
        if not baseline_path.is_absolute():
            baseline_path = PROJECT_ROOT / baseline_path
        selected_keys = {(str(record["dataset"]), str(record["image_id"])) for record in selected_records}
        baseline_predictions = [
            record
            for record in read_jsonl(baseline_path)
            if (str(record.get("dataset", args.benchmark)), str(record.get("image_id", ""))) in selected_keys
        ]
        baseline_metrics = evaluate_lae_style(
            selected_records,
            baseline_predictions,
            class_specs,
            args.benchmark,
            args.class_source,
            args.gt_label_field,
            args.max_detections_per_image,
            prompt_mode="multi_label",
            multi_label_prompt_template="existing baseline predictions",
            single_label_prompt_template=None,
            tile_size=0,
            tile_overlap=0,
            tile_nms_iou=args.merge_iou,
        )
        baseline_metrics["protocol"]["name"] = "Baseline predictions evaluated on the same selected images"
        baseline_metrics["protocol"]["source_predictions"] = str(baseline_path)
        baseline_metrics_path.write_text(json.dumps(baseline_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    write_summary_md(summary_path, metrics, args)
    if baseline_metrics is not None:
        baseline_row = summarize_metrics(baseline_metrics)[0]
        zoom_row = summarize_metrics(metrics)[0]
        comparison_lines = [
            "# Agentic Zoom Baseline Comparison",
            "",
            "Both rows are evaluated on the exact same selected images. If `query_class_mode` is not `all`, the agentic row is diagnostic only.",
            "",
            "| Method | Images | GT | Pred boxes | AP50 | mAP50-95 | F1@0.5 | Prec@0.5 | Rec@0.5 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| Baseline | {baseline_row['images']} | {baseline_row['gt_instances']} | {baseline_row['predicted_boxes']} | "
                f"{baseline_row['ap50']:.2f} | {baseline_row['map50_95']:.2f} | {baseline_row['f1@0.5']:.2f} | "
                f"{baseline_row['precision@0.5']:.2f} | {baseline_row['recall@0.5']:.2f} |"
            ),
            (
                f"| Agentic zoom | {zoom_row['images']} | {zoom_row['gt_instances']} | {zoom_row['predicted_boxes']} | "
                f"{zoom_row['ap50']:.2f} | {zoom_row['map50_95']:.2f} | {zoom_row['f1@0.5']:.2f} | "
                f"{zoom_row['precision@0.5']:.2f} | {zoom_row['recall@0.5']:.2f} |"
            ),
        ]
        comparison_path.write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")
    print(f"Wrote final predictions: {final_path}")
    print(f"Wrote query trace: {trace_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote per-class metrics: {per_class_path}")
    print(f"Wrote summary: {summary_path}")
    if baseline_metrics is not None:
        print(f"Wrote baseline metrics: {baseline_metrics_path}")
        print(f"Wrote comparison: {comparison_path}")


if __name__ == "__main__":
    main()
