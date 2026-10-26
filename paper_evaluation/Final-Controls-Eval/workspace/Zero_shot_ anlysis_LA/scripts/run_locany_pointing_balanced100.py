#!/usr/bin/env python3
"""Evaluate LocateAnything pointing on the balanced DOTAv2-100 subset.

This diagnostic compares two output styles on the same images/classes:

* full-image multi-label box detection, evaluated with IoU@0.5
* class-wise pointing, evaluated by whether the point falls inside a GT hbox

Pointing is not the same metric as box detection, but it tells us whether LA
understands approximate object location even when precise box generation fails.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from zero_shot_analysis.geometry import hbox_center, point_inside_hbox  # noqa: E402
from zero_shot_analysis.locate_anything import (  # noqa: E402
    LocateAnythingWorker,
    parse_locateanything_labeled_boxes,
)
from zero_shot_analysis.schema import normalize_class_name  # noqa: E402
from evaluate_locateanything_lae_style import (  # noqa: E402
    benchmark_class_specs,
    build_alias_lookup,
    canonicalize_label,
    clip_hbox_to_image,
    detection_prompt,
    evaluate_lae_style,
    load_lae_records,
    read_json,
    resolve_image_path,
    selected_benchmarks,
)
from run_locany_agentic_zoom_detection import per_class_f1_rows, summarize_metrics, write_csv  # noqa: E402


POINT_OR_BOX_RE = re.compile(
    r"<ref>\s*(.*?)\s*</ref>|<box>\s*<(\d+)>\s*<(\d+)>\s*(?:<(\d+)>\s*<(\d+)>\s*)?</box>",
    re.IGNORECASE | re.DOTALL,
)


def stable_generation_seed(base_seed: int, image_id: str, target_class: str) -> int:
    payload = f"{base_seed}\0{image_id}\0{target_class}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31)


def set_generation_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def load_manifest(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]
    return rows


def selected_records_from_manifest(
    benchmark_name: str,
    benchmark: Mapping[str, Any],
    class_specs: Sequence[Mapping[str, str]],
    manifest_rows: Sequence[Mapping[str, str]],
    label_field: str,
) -> list[dict[str, Any]]:
    all_records = load_lae_records(benchmark_name, benchmark, class_specs, label_field)
    by_id = {str(record["image_id"]): record for record in all_records}
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in manifest_rows:
        image_id = str(row["image_id"])
        if image_id not in by_id:
            missing.append(image_id)
            continue
        records.append(by_id[image_id])
    if missing:
        raise SystemExit(f"Manifest image_ids missing from annotations: {missing[:10]}")
    return records


def point_prompt(class_name: str) -> str:
    return f"Point to: {class_name}."


def parse_locateanything_points(
    answer: str,
    image_width: int,
    image_height: int,
    default_label: str,
) -> list[dict[str, Any]]:
    """Parse LA pointing output.

    Official LA pointing examples use two coordinates inside <box>, for example
    ``<box><312><441></box>``. If the model emits a four-coordinate box instead,
    we use the box center as a fallback point so the run does not silently drop
    valid approximate localization attempts.
    """

    points: list[dict[str, Any]] = []
    current_label: str | None = None
    for match in POINT_OR_BOX_RE.finditer(answer):
        ref_label = match.group(1)
        if ref_label is not None:
            current_label = normalize_class_name(ref_label)
            continue

        x1 = int(match.group(2))
        y1 = int(match.group(3))
        x2_raw = match.group(4)
        y2_raw = match.group(5)
        if x2_raw is None or y2_raw is None:
            x_norm, y_norm = float(x1), float(y1)
            source = "point"
        else:
            x_norm = (float(x1) + float(x2_raw)) / 2.0
            y_norm = (float(y1) + float(y2_raw)) / 2.0
            source = "box_center_fallback"
        x = max(0.0, min(float(image_width), x_norm / 1000.0 * float(image_width)))
        y = max(0.0, min(float(image_height), y_norm / 1000.0 * float(image_height)))
        points.append(
            {
                "point": [x, y],
                "label": default_label,
                "raw_label": current_label,
                "score": 1.0,
                "source": source,
            }
        )
    return points


def match_points_to_gt(gt_boxes: Sequence[Sequence[float]], points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matched_gt: set[int] = set()
    matched_pairs: list[dict[str, Any]] = []
    fp_indices: list[int] = []
    for point_idx, point_record in enumerate(points):
        point = tuple(float(v) for v in point_record["point"])
        candidates = []
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            if point_inside_hbox(point, gt_box):
                cx, cy = hbox_center(gt_box)
                dist = ((point[0] - cx) ** 2 + (point[1] - cy) ** 2) ** 0.5
                candidates.append((dist, gt_idx))
        if candidates:
            _dist, gt_idx = min(candidates)
            matched_gt.add(gt_idx)
            matched_pairs.append({"gt_index": gt_idx, "point_index": point_idx})
        else:
            fp_indices.append(point_idx)
    fn_indices = [idx for idx in range(len(gt_boxes)) if idx not in matched_gt]
    return {
        "tp": len(matched_pairs),
        "fp": len(fp_indices),
        "fn": len(fn_indices),
        "matched_pairs": matched_pairs,
        "fp_point_indices": fp_indices,
        "fn_gt_indices": fn_indices,
    }


def completed_point_keys(path: Path) -> set[tuple[str, str]]:
    return {(str(row["image_id"]), str(row["target_class"])) for row in read_jsonl(path)}


def completed_bbox_keys(path: Path) -> set[str]:
    return {str(row["image_id"]) for row in read_jsonl(path)}


def build_point_tasks(records: Sequence[Mapping[str, Any]], class_specs: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    class_order = [spec["canonical"] for spec in class_specs]
    order_index = {name: idx for idx, name in enumerate(class_order)}
    tasks: list[dict[str, Any]] = []
    for record in records:
        present = sorted({gt["label"] for gt in record.get("gt", [])}, key=lambda name: order_index.get(name, 9999))
        for class_name in present:
            tasks.append({"record": record, "target_class": class_name, "query": point_prompt(class_name)})
    return tasks


def run_pointing(
    worker: LocateAnythingWorker,
    records: Sequence[Mapping[str, Any]],
    class_specs: Sequence[Mapping[str, str]],
    benchmark: Mapping[str, Any],
    args: argparse.Namespace,
    output_path: Path,
) -> list[dict[str, Any]]:
    from PIL import Image
    import torch

    image_root = PROJECT_ROOT / benchmark["image_root"]
    tasks = build_point_tasks(records, class_specs)
    done = completed_point_keys(output_path) if args.resume else set()
    remaining = [task for task in tasks if (str(task["record"]["image_id"]), str(task["target_class"])) not in done]
    print(f"Pointing tasks: {len(tasks)} | completed: {len(done)} | remaining: {len(remaining)}")
    rows_to_append: list[dict[str, Any]] = []
    for idx, task in enumerate(remaining, start=1):
        record = task["record"]
        generation_seed = stable_generation_seed(args.seed, str(record["image_id"]), str(task["target_class"]))
        set_generation_seed(generation_seed)
        image_path = resolve_image_path(record, image_root)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = worker.predict(
            image,
            task["query"],
            generation_mode=args.generation_mode,
            max_new_tokens=args.point_max_new_tokens,
            temperature=args.temperature,
            do_sample=not args.greedy,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        answer = str(output.get("answer", ""))
        points = parse_locateanything_points(answer, width, height, task["target_class"])
        gt_boxes = [gt["hbox"] for gt in record.get("gt", []) if gt["label"] == task["target_class"]]
        match = match_points_to_gt(gt_boxes, points)
        stats = output.get("stats") or {}
        gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else None
        row = {
            "dataset": args.benchmark,
            "image_id": str(record["image_id"]),
            "file_name": record["file_name"],
            "image_path": str(image_path),
            "target_class": task["target_class"],
            "query": task["query"],
            "generation_seed": generation_seed,
            "mode": "pointing",
            "decode_mode": args.generation_mode,
            "output": answer,
            "points": points,
            "gt_count": len(gt_boxes),
            "pred_count": len(points),
            "tp": match["tp"],
            "fp": match["fp"],
            "fn": match["fn"],
            **prf(match["tp"], match["fp"], match["fn"]),
            "match": match,
            "latency_ms": latency_ms,
            "tokens_generated": stats.get("tokens_generated") if isinstance(stats, dict) else None,
            "gpu_memory_mb": gpu_memory_mb,
            "meta": {
                "width": width,
                "height": height,
                "raw_stats": stats if isinstance(stats, dict) else {},
            },
        }
        rows_to_append.append(row)
        if len(rows_to_append) >= args.flush_every:
            append_jsonl(output_path, rows_to_append)
            rows_to_append = []
        if idx % args.log_every == 0 or idx == len(remaining):
            print(
                f"point {idx}/{len(remaining)} image={record['image_id']} class={task['target_class']} "
                f"gt={len(gt_boxes)} pred={len(points)} tp={match['tp']}"
            )
    if rows_to_append:
        append_jsonl(output_path, rows_to_append)
    return read_jsonl(output_path)


def run_bbox(
    worker: LocateAnythingWorker,
    records: Sequence[Mapping[str, Any]],
    class_specs: Sequence[Mapping[str, str]],
    benchmark: Mapping[str, Any],
    args: argparse.Namespace,
    output_path: Path,
) -> list[dict[str, Any]]:
    from PIL import Image
    import torch

    image_root = PROJECT_ROOT / benchmark["image_root"]
    alias_lookup = build_alias_lookup(class_specs)
    class_set = {spec["canonical"] for spec in class_specs}
    query = detection_prompt(class_specs)
    done = completed_bbox_keys(output_path) if args.resume else set()
    remaining = [record for record in records if str(record["image_id"]) not in done]
    print(f"BBox tasks: {len(records)} | completed: {len(done)} | remaining: {len(remaining)}")
    rows_to_append: list[dict[str, Any]] = []
    for idx, record in enumerate(remaining, start=1):
        image_path = resolve_image_path(record, image_root)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = worker.predict(
            image,
            query,
            generation_mode=args.generation_mode,
            max_new_tokens=args.bbox_max_new_tokens,
            temperature=args.temperature,
            do_sample=not args.greedy,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        answer = str(output.get("answer", ""))
        raw_boxes = parse_locateanything_labeled_boxes(answer, width, height)
        predictions: list[dict[str, Any]] = []
        unknown_labels: dict[str, int] = defaultdict(int)
        for pred in raw_boxes:
            label = canonicalize_label(pred.get("label"), alias_lookup, class_set)
            if label is None:
                unknown_labels[str(pred.get("label") or "<missing>")] += 1
                continue
            clipped = clip_hbox_to_image(pred["hbox"], width, height)
            if clipped is None:
                continue
            predictions.append(
                {
                    "label": label,
                    "hbox": clipped,
                    "score": float(pred.get("score", 1.0)),
                    "raw_label": pred.get("label"),
                }
            )
        stats = output.get("stats") or {}
        gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else None
        row = {
            "dataset": args.benchmark,
            "image_id": str(record["image_id"]),
            "file_name": record["file_name"],
            "image_path": str(image_path),
            "query": query,
            "prompt_mode": "multi_label",
            "class_source": args.class_source,
            "classes": [spec["canonical"] for spec in class_specs],
            "mode": "bbox_full_image",
            "decode_mode": args.generation_mode,
            "output": answer,
            "predictions": predictions,
            "unknown_predicted_labels": dict(unknown_labels),
            "latency_ms": latency_ms,
            "tokens_generated": stats.get("tokens_generated") if isinstance(stats, dict) else None,
            "gpu_memory_mb": gpu_memory_mb,
            "meta": {
                "width": width,
                "height": height,
                "raw_stats": stats if isinstance(stats, dict) else {},
            },
        }
        rows_to_append.append(row)
        if len(rows_to_append) >= args.flush_every:
            append_jsonl(output_path, rows_to_append)
            rows_to_append = []
        if idx % args.log_every == 0 or idx == len(remaining):
            print(f"bbox {idx}/{len(remaining)} image={record['image_id']} pred_boxes={len(predictions)}")
    if rows_to_append:
        append_jsonl(output_path, rows_to_append)
    return read_jsonl(output_path)


def evaluate_pointing(records: Sequence[Mapping[str, Any]], point_rows: Sequence[Mapping[str, Any]], class_specs: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    by_class: dict[str, dict[str, int]] = {
        spec["canonical"]: {"gt_count": 0, "pred_count": 0, "tp": 0, "fp": 0, "fn": 0}
        for spec in class_specs
    }
    image_count = len({str(record["image_id"]) for record in records})
    totals = {"gt_count": 0, "pred_count": 0, "tp": 0, "fp": 0, "fn": 0}
    for row in point_rows:
        class_name = str(row["target_class"])
        if class_name not in by_class:
            continue
        item = by_class[class_name]
        for key in ["gt_count", "pred_count", "tp", "fp", "fn"]:
            value = int(row.get(key, 0))
            item[key] += value
            totals[key] += value
    per_class = []
    for class_name, item in by_class.items():
        metrics = prf(item["tp"], item["fp"], item["fn"])
        per_class.append({"class": class_name, **item, **{f"{k}@point": v for k, v in metrics.items()}})
    summary_metrics = prf(totals["tp"], totals["fp"], totals["fn"])
    fallback_count = sum(
        1
        for row in point_rows
        for point in row.get("points", [])
        if point.get("source") == "box_center_fallback"
    )
    return {
        "protocol": {
            "name": "LocateAnything class-wise pointing",
            "prompt": "One prompt per image / GT-present class: Point to: {class_name}.",
            "metric": "A predicted point is TP if it falls inside an unmatched same-class GT horizontal box.",
            "note": "Pointing is evaluated as approximate localization, not IoU box detection.",
        },
        "dataset": "DOTAv2",
        "images": image_count,
        "classes": len(class_specs),
        "point_tasks": len(point_rows),
        "gt_instances": totals["gt_count"],
        "predicted_points": totals["pred_count"],
        "accuracy": {"point_inside_gt": {**totals, **summary_metrics}},
        "fallback_box_centers": fallback_count,
        "per_class": per_class,
    }


def write_point_per_class_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    rows = []
    for row in metrics.get("per_class", []):
        rows.append(
            {
                "class": row["class"],
                "gt_count": row["gt_count"],
                "pred_count": row["pred_count"],
                "tp@point": row["tp"],
                "fp@point": row["fp"],
                "fn@point": row["fn"],
                "precision@point": row["precision@point"],
                "recall@point": row["recall@point"],
                "f1@point": row["f1@point"],
            }
        )
    write_csv(path, rows)


def write_summary(
    path: Path,
    args: argparse.Namespace,
    bbox_metrics: Mapping[str, Any] | None,
    point_metrics: Mapping[str, Any],
) -> None:
    lines = [
        "# LocateAnything Pointing vs Full-Image BBox",
        "",
        f"- Model tag: `{args.run_tag}`",
        f"- Benchmark: `{args.benchmark}`",
        f"- Manifest: `{args.manifest}`",
        f"- Model path: `{args.model_path}`",
        f"- Generation mode: `{args.generation_mode}`",
        "",
        "| Mode | Images | GT | Predictions | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if bbox_metrics:
        bbox = summarize_metrics(bbox_metrics)[0]
        lines.append(
            f"| Full-image bbox IoU@0.5 | {bbox['images']} | {bbox['gt_instances']} | {bbox['predicted_boxes']} | "
            f"{bbox['tp@0.5']} | {bbox['fp@0.5']} | {bbox['fn@0.5']} | "
            f"{bbox['precision@0.5']:.2f} | {bbox['recall@0.5']:.2f} | {bbox['f1@0.5']:.2f} |"
        )
    point = point_metrics["accuracy"]["point_inside_gt"]
    lines.append(
        f"| Point inside GT | {point_metrics['images']} | {point_metrics['gt_instances']} | {point_metrics['predicted_points']} | "
        f"{point['tp']} | {point['fp']} | {point['fn']} | "
        f"{point['precision'] * 100:.2f} | {point['recall'] * 100:.2f} | {point['f1'] * 100:.2f} |"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/lae_benchmarks.json")
    parser.add_argument("--benchmark", default="DOTAv2")
    parser.add_argument("--manifest", default="runs/agentic_zoom_balanced/dotav2_balanced_100_manifest.csv")
    parser.add_argument("--model-path", default="model/LocateAnything-3B/checkpoint")
    parser.add_argument("--run-tag", default="base_la")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--class-source", choices=["lae-metainfo", "annotations"], default="lae-metainfo")
    parser.add_argument("--gt-label-field", default="raw_class_name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--generation-mode", choices=["slow", "fast", "hybrid"], default="hybrid")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--greedy", action="store_true", help="Use do_sample=False")
    parser.add_argument("--seed", type=int, default=0, help="Stable per-image/class generation seed.")
    parser.add_argument("--point-max-new-tokens", type=int, default=512)
    parser.add_argument("--bbox-max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-detections-per-image", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-bbox", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--flush-every", type=int, default=1)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    config = read_json(PROJECT_ROOT / args.config)
    benchmark = dict(selected_benchmarks(config, [args.benchmark]))[args.benchmark]
    class_specs = benchmark_class_specs(args.benchmark, benchmark, args.class_source, args.gt_label_field)
    manifest_path = PROJECT_ROOT / args.manifest
    manifest_rows = load_manifest(manifest_path, args.limit)
    records = selected_records_from_manifest(args.benchmark, benchmark, class_specs, manifest_rows, args.gt_label_field)

    output_dir = PROJECT_ROOT / (args.output_dir or f"runs/pointing_balanced100/{args.run_tag}")
    point_path = output_dir / "pointing_predictions.jsonl"
    bbox_path = output_dir / "bbox_full_image_predictions.jsonl"
    print(f"Selected images: {len(records)}")
    print(f"Pointing tasks: {len(build_point_tasks(records, class_specs))}")
    print(f"Output dir: {output_dir}")
    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(
            output_dir / "planned_pointing_tasks.csv",
            [
                {
                    "image_id": task["record"]["image_id"],
                    "file_name": task["record"]["file_name"],
                    "target_class": task["target_class"],
                    "query": task["query"],
                    "gt_count": sum(1 for gt in task["record"].get("gt", []) if gt["label"] == task["target_class"]),
                }
                for task in build_point_tasks(records, class_specs)
            ],
        )
        print("Dry run complete.")
        return

    model_path = args.model_path
    if not Path(model_path).is_absolute():
        model_path = str(PROJECT_ROOT / model_path)
    args.model_path = model_path
    worker = LocateAnythingWorker.from_args(model_path=model_path, device=args.device, dtype=args.dtype)

    bbox_metrics = None
    if not args.skip_bbox:
        bbox_rows = run_bbox(worker, records, class_specs, benchmark, args, bbox_path)
        bbox_metrics = evaluate_lae_style(
            records,
            bbox_rows,
            class_specs,
            args.benchmark,
            args.class_source,
            args.gt_label_field,
            args.max_detections_per_image,
            prompt_mode="multi_label",
        )
        write_json(output_dir / "bbox_full_image_metrics.json", bbox_metrics)
        write_csv(
            output_dir / "bbox_full_image_per_class_metrics.csv",
            per_class_f1_rows(records, bbox_rows, class_specs, args.max_detections_per_image),
        )

    point_rows = run_pointing(worker, records, class_specs, benchmark, args, point_path)
    point_metrics = evaluate_pointing(records, point_rows, class_specs)
    write_json(output_dir / "pointing_metrics.json", point_metrics)
    write_point_per_class_csv(output_dir / "pointing_per_class_metrics.csv", point_metrics)
    write_summary(output_dir / "summary.md", args, bbox_metrics, point_metrics)
    print(f"Wrote pointing predictions: {point_path}")
    if bbox_metrics:
        print(f"Wrote bbox predictions: {bbox_path}")
    print(f"Wrote summary: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
