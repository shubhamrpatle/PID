#!/usr/bin/env python3
"""Evaluate LocateAnything on remote-sensing visual grounding benchmarks.

Supported benchmarks:
  - DIOR-RSVG / RSVGD: XML object descriptions + official train/val/test ids.
  - VRSBench-VG: VRSBench_EVAL_referring.json.

The evaluator follows referring-expression grounding, not detector AP:
each record is one image + one language expression + one target box.
The fair metric uses the first generated box as the model answer.  We also
report best-box metrics as diagnostics for "the model saw it but ordered it
badly" cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from zero_shot_analysis.geometry import hbox_iou
from zero_shot_analysis.locate_anything import LocateAnythingWorker, parse_locateanything_labeled_boxes
from zero_shot_analysis.schema import normalize_class_name, write_jsonl


DIOR_CLASS_ALIASES = {
    "airplane": "airplane",
    "airport": "airport",
    "baseballfield": "baseball field",
    "basketballcourt": "basketball court",
    "bridge": "bridge",
    "chimney": "chimney",
    "dam": "dam",
    "expresswayservicearea": "expressway service area",
    "expresswaytollstation": "expressway toll station",
    "golffield": "golf field",
    "groundtrackfield": "ground track field",
    "harbor": "harbor",
    "overpass": "overpass",
    "ship": "ship",
    "stadium": "stadium",
    "storagetank": "storage tank",
    "tenniscourt": "tennis court",
    "trainstation": "train station",
    "vehicle": "vehicle",
    "windmill": "windmill",
}

BOX100_RE = re.compile(r"<(\d+(?:\.\d+)?)>")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def canonical_dior_class(name: str) -> str:
    compact = "".join(str(name).strip().lower().replace("_", "").replace("-", "").split())
    return DIOR_CLASS_ALIASES.get(compact, normalize_class_name(str(name)))


def normalize_hbox(box: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def resolve_image_path(file_name: str, roots: Iterable[Path]) -> str:
    for root in roots:
        if not root:
            continue
        candidate = root / file_name
        if candidate.exists():
            return str(candidate)
    return ""


def parse_dior_rsvg(args: argparse.Namespace) -> list[dict[str, Any]]:
    import xml.etree.ElementTree as ET

    root_dir = Path(args.dior_rsvg_root)
    annotations_dir = root_dir / "Annotations"
    split_path = root_dir / f"{args.split}.txt"
    if not annotations_dir.exists():
        raise SystemExit(f"Missing DIOR-RSVG annotation directory: {annotations_dir}")
    if not split_path.exists():
        raise SystemExit(f"Missing DIOR-RSVG split file: {split_path}")

    image_roots = [Path(p) for p in args.image_roots]
    local_jpeg_root = root_dir / "JPEGImages"
    if local_jpeg_root.exists():
        image_roots.insert(0, local_jpeg_root)

    all_records: list[dict[str, Any]] = []
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        file_name = root.findtext("filename") or f"{xml_path.stem}.jpg"
        size = root.find("size")
        width = int(float(size.findtext("width", "800"))) if size is not None else 800
        height = int(float(size.findtext("height", "800"))) if size is not None else 800
        image_path = resolve_image_path(file_name, image_roots)
        for obj_idx, obj in enumerate(root.findall("object")):
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            class_name = canonical_dior_class(obj.findtext("name") or "")
            query = (obj.findtext("description") or class_name).strip()
            hbox = normalize_hbox(
                [
                    float(bnd.findtext("xmin", "0")),
                    float(bnd.findtext("ymin", "0")),
                    float(bnd.findtext("xmax", "0")),
                    float(bnd.findtext("ymax", "0")),
                ]
            )
            sample_id = len(all_records)
            all_records.append(
                {
                    "sample_id": str(sample_id),
                    "benchmark": "DIOR-RSVG",
                    "dataset": "DIOR-RSVG",
                    "split": args.split,
                    "image_id": xml_path.stem,
                    "file_name": file_name,
                    "image_path": image_path,
                    "width": width,
                    "height": height,
                    "query": query,
                    "target_class": class_name,
                    "gt_hbox": hbox,
                    "source_object_index": obj_idx,
                }
            )

    split_ids = [int(line.strip()) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    bad_ids = [idx for idx in split_ids if idx < 0 or idx >= len(all_records)]
    if bad_ids:
        raise SystemExit(f"DIOR-RSVG split contains ids outside 0..{len(all_records)-1}: {bad_ids[:5]}")
    return [all_records[idx] for idx in split_ids]


def parse_vrsbench_grounding(args: argparse.Namespace) -> list[dict[str, Any]]:
    root_dir = Path(args.vrsbench_root)
    json_path = root_dir / "VRSBench_EVAL_referring.json"
    if not json_path.exists():
        raise SystemExit(f"Missing VRSBench referring JSON: {json_path}")

    image_roots = [Path(p) for p in args.image_roots]
    for candidate in [root_dir / "Images_val", root_dir / "images", root_dir]:
        if candidate.exists():
            image_roots.insert(0, candidate)

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for item in raw:
        file_name = str(item["image_id"])
        image_path = resolve_image_path(file_name, image_roots)
        width = height = None
        if image_path:
            with Image.open(image_path) as image:
                width, height = image.size
        else:
            # VRSBench annotations are normalized; most images are 512, but we
            # keep missing-image records explicit so validation can catch them.
            width = height = 512

        coords = [float(v) for v in BOX100_RE.findall(str(item.get("ground_truth", "")))]
        if len(coords) >= 4:
            x1, y1, x2, y2 = coords[:4]
            hbox = [x1 / 100.0 * width, y1 / 100.0 * height, x2 / 100.0 * width, y2 / 100.0 * height]
        elif item.get("obj_coord") and len(item["obj_coord"]) >= 4:
            x1, y1, x2, y2 = [float(v) for v in item["obj_coord"][:4]]
            hbox = [x1 * width, y1 * height, x2 * width, y2 * height]
        else:
            continue
        hbox = normalize_hbox(hbox)
        records.append(
            {
                "sample_id": str(item.get("question_id", len(records))),
                "benchmark": "VRSBench-VG",
                "dataset": "VRSBench-VG",
                "split": "eval",
                "image_id": file_name,
                "file_name": file_name,
                "image_path": image_path,
                "width": width,
                "height": height,
                "query": str(item["question"]).strip(),
                "target_class": normalize_class_name(str(item.get("obj_cls", ""))),
                "gt_hbox": hbox,
                "unique": bool(item.get("unique", False)),
                "size_group": str(item.get("size_group", "")),
                "obj_ids": item.get("obj_ids", []),
            }
        )
    return records


def size_bucket(record: Mapping[str, Any]) -> str:
    x1, y1, x2, y2 = record["gt_hbox"]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    denom = float(record["width"]) * float(record["height"])
    ratio = area / denom if denom > 0 else 0.0
    if ratio < 0.001:
        return "small"
    if ratio < 0.01:
        return "medium"
    return "large"


def prompt_for_record(record: Mapping[str, Any], template: str) -> str:
    return template.format(
        query=record["query"],
        class_name=record.get("target_class", ""),
        dataset=record.get("dataset", ""),
    )


def prediction_key(record: Mapping[str, Any]) -> str:
    return f"{record['dataset']}::{record['sample_id']}::{record['image_id']}::{record['query']}"


def seed_for_record(base_seed: int, record: Mapping[str, Any]) -> int:
    digest = hashlib.sha256(prediction_key(record).encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:4], "big")) % (2**31)


def set_generation_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_inference(records: list[dict[str, Any]], args: argparse.Namespace, prediction_path: Path) -> list[dict[str, Any]]:
    existing = read_jsonl(prediction_path)
    done = {prediction_key(record): record for record in existing}
    todo = [record for record in records if prediction_key(record) not in done]
    print(f"Records: {len(records)} | completed: {len(done)} | remaining: {len(todo)}")
    if args.dry_run or not todo:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.touch(exist_ok=True)
        return existing

    worker = LocateAnythingWorker.from_args(args.model_path, args.device, args.dtype)
    if args.model_max_length > 0:
        worker.processor.tokenizer.model_max_length = args.model_max_length
    if args.image_token_limit > 0:
        worker.processor.image_processor.in_token_limit = args.image_token_limit
    print(
        "Effective grounding limits: "
        f"image_tokens={worker.processor.image_processor.in_token_limit} "
        f"model_length={worker.processor.tokenizer.model_max_length}"
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("a", encoding="utf-8") as handle:
        for idx, record in enumerate(todo, start=1):
            if not record.get("image_path"):
                output = ""
                pred_boxes = []
                latency_ms = None
                stats = {"error": "missing_image"}
            else:
                image = Image.open(record["image_path"]).convert("RGB")
                prompt = prompt_for_record(record, args.prompt_template)
                generation_seed = seed_for_record(args.seed, record)
                set_generation_seed(generation_seed)
                start = time.perf_counter()
                result = worker.predict(
                    image,
                    prompt,
                    generation_mode=args.generation_mode,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=not args.greedy,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    verbose=False,
                )
                latency_ms = (time.perf_counter() - start) * 1000.0
                output = str(result.get("answer", ""))
                pred_boxes = parse_locateanything_labeled_boxes(output, record["width"], record["height"], None)
                stats = result.get("stats") or {}
            pred_record = {
                **record,
                "prompt": prompt_for_record(record, args.prompt_template),
                "model_path": args.model_path,
                "generation_mode": args.generation_mode,
                "generation_seed": seed_for_record(args.seed, record),
                "output": output,
                "predictions": pred_boxes,
                "latency_ms": latency_ms,
                "stats": stats,
            }
            handle.write(json.dumps(pred_record, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"[{idx}/{len(todo)}] {record['dataset']} sample={record['sample_id']} "
                f"class={record.get('target_class')} preds={len(pred_boxes)}"
            )
            existing.append(pred_record)
    return existing


def evaluate_predictions(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    pred_index = {prediction_key(pred): pred for pred in predictions}
    rows = []
    missing_predictions = 0
    for record in records:
        pred = pred_index.get(prediction_key(record))
        if pred is None:
            missing_predictions += 1
            pred_boxes = []
        else:
            pred_boxes = [p["hbox"] for p in pred.get("predictions", [])]
        gt = record["gt_hbox"]
        first_iou = hbox_iou(gt, pred_boxes[0]) if pred_boxes else 0.0
        best_iou = max((hbox_iou(gt, box) for box in pred_boxes), default=0.0)
        rows.append(
            {
                "sample_id": record["sample_id"],
                "dataset": record["dataset"],
                "image_id": record["image_id"],
                "target_class": record.get("target_class", ""),
                "size_bucket": size_bucket(record),
                "pred_count": len(pred_boxes),
                "first_iou": first_iou,
                "best_iou": best_iou,
                "first_acc_0_5": first_iou >= 0.5,
                "first_acc_0_7": first_iou >= 0.7,
                "best_acc_0_5": best_iou >= 0.5,
                "best_acc_0_7": best_iou >= 0.7,
                "missing_image": not bool(record.get("image_path")),
            }
        )

    def aggregate(subrows: list[dict[str, Any]]) -> dict[str, Any]:
        if not subrows:
            return {
                "queries": 0,
                "first_acc_0_5": 0.0,
                "first_acc_0_7": 0.0,
                "first_mean_iou": 0.0,
                "best_acc_0_5": 0.0,
                "best_acc_0_7": 0.0,
                "best_mean_iou": 0.0,
                "avg_pred_count": 0.0,
            }
        return {
            "queries": len(subrows),
            "first_acc_0_5": mean(1.0 if row["first_acc_0_5"] else 0.0 for row in subrows),
            "first_acc_0_7": mean(1.0 if row["first_acc_0_7"] else 0.0 for row in subrows),
            "first_mean_iou": mean(row["first_iou"] for row in subrows),
            "best_acc_0_5": mean(1.0 if row["best_acc_0_5"] else 0.0 for row in subrows),
            "best_acc_0_7": mean(1.0 if row["best_acc_0_7"] else 0.0 for row in subrows),
            "best_mean_iou": mean(row["best_iou"] for row in subrows),
            "avg_pred_count": mean(row["pred_count"] for row in subrows),
        }

    by_class = {name: aggregate([row for row in rows if row["target_class"] == name]) for name in sorted({r["target_class"] for r in rows})}
    by_size = {name: aggregate([row for row in rows if row["size_bucket"] == name]) for name in ["small", "medium", "large"]}
    return {
        "overall": aggregate(rows),
        "missing_predictions": missing_predictions,
        "missing_images": sum(1 for row in rows if row["missing_image"]),
        "classes": len(by_class),
        "by_class": by_class,
        "by_size": by_size,
        "prediction_count_histogram": dict(sorted(Counter(row["pred_count"] for row in rows).items())),
        "rows": rows,
    }


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, benchmark: str, metrics: Mapping[str, Any], reference: Mapping[str, Any] | None = None) -> None:
    overall = metrics["overall"]
    lines = [
        f"# LocateAnything Grounding Evaluation: {benchmark}",
        "",
        "## Overall",
        "",
        "| Queries | First Acc@0.5 | First Acc@0.7 | First mIoU | Best Acc@0.5 | Best Acc@0.7 | Best mIoU | Avg Pred Boxes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {overall['queries']} | {overall['first_acc_0_5']*100:.2f} | {overall['first_acc_0_7']*100:.2f} | "
            f"{overall['first_mean_iou']*100:.2f} | {overall['best_acc_0_5']*100:.2f} | "
            f"{overall['best_acc_0_7']*100:.2f} | {overall['best_mean_iou']*100:.2f} | {overall['avg_pred_count']:.2f} |"
        ),
        "",
        f"Missing images: `{metrics['missing_images']}`",
        f"Missing predictions: `{metrics['missing_predictions']}`",
        "",
    ]
    if reference:
        lines.extend(["## Paper Reference", ""])
        lines.extend(reference.get("markdown_lines", []))
        lines.append("")
    lines.extend(
        [
            "## By Size",
            "",
            "| Size | Queries | First Acc@0.5 | Best Acc@0.5 | First mIoU | Best mIoU |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for size, row in metrics["by_size"].items():
        lines.append(
            f"| {size} | {row['queries']} | {row['first_acc_0_5']*100:.2f} | {row['best_acc_0_5']*100:.2f} | "
            f"{row['first_mean_iou']*100:.2f} | {row['best_mean_iou']*100:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.records_jsonl:
        records = read_jsonl(Path(args.records_jsonl))
        wrong_benchmark = [record for record in records if record.get("benchmark") != args.benchmark]
        if wrong_benchmark:
            raise SystemExit(
                f"--records-jsonl contains {len(wrong_benchmark)} records that do not match "
                f"--benchmark {args.benchmark}"
            )
    elif args.benchmark == "DIOR-RSVG":
        records = parse_dior_rsvg(args)
    elif args.benchmark == "VRSBench-VG":
        records = parse_vrsbench_grounding(args)
    else:
        raise SystemExit(f"Unsupported benchmark: {args.benchmark}")
    if args.limit:
        records = records[: args.limit]
    return records


def shard_records(records: list[dict[str, Any]], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if num_shards <= 1:
        return records
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit(f"--shard-index must be in [0, {num_shards - 1}], got {shard_index}")
    return [record for idx, record in enumerate(records) if idx % num_shards == shard_index]


def shard_suffix(args: argparse.Namespace) -> str:
    if args.num_shards <= 1:
        return ""
    width = max(2, len(str(args.num_shards - 1)))
    return f"_shard{args.shard_index:0{width}d}of{args.num_shards:0{width}d}"


def read_shard_predictions(output_dir: Path, num_shards: int) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    width = max(2, len(str(num_shards - 1)))
    missing = []
    for shard_index in range(num_shards):
        path = output_dir / f"predictions_shard{shard_index:0{width}d}of{num_shards:0{width}d}.jsonl"
        if not path.exists():
            missing.append(str(path))
            continue
        predictions.extend(read_jsonl(path))
    if missing:
        raise SystemExit("Missing shard prediction files:\n" + "\n".join(missing))
    deduped = {prediction_key(prediction): prediction for prediction in predictions}
    return list(deduped.values())


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["DIOR-RSVG", "VRSBench-VG"], required=True)
    parser.add_argument("--split", default="test", help="DIOR-RSVG split: train, val, or test.")
    parser.add_argument("--dior-rsvg-root", default=str(PROJECT_ROOT / "dataset/grounding/DIOR-RSVG"))
    parser.add_argument("--vrsbench-root", default=str(PROJECT_ROOT / "dataset/grounding/VRSBench"))
    parser.add_argument(
        "--image-roots",
        nargs="*",
        default=[
            str(PROJECT_ROOT / "dataset/grounding/DIOR-RSVG/JPEGImages"),
            str(PROJECT_ROOT / "dataset/benchmark/DIOR/images"),
            str(PROJECT_ROOT / "dataset/grounding/VRSBench/Images_val"),
        ],
    )
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "model/LocateAnything-3B/checkpoint"))
    parser.add_argument(
        "--records-jsonl",
        default="",
        help="Optional frozen benchmark subset. Records are evaluated in file order.",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs/grounding_benchmarks"))
    parser.add_argument("--run-tag", default="base_la")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge predictions_shardXXofNN.jsonl files for this run-tag and compute full metrics.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--generation-mode", default="hybrid", choices=["fast", "slow", "hybrid"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--image-token-limit",
        type=int,
        default=0,
        help="Override processor.image_processor.in_token_limit; 0 keeps checkpoint configuration.",
    )
    parser.add_argument(
        "--model-max-length",
        type=int,
        default=0,
        help="Override tokenizer.model_max_length; 0 keeps checkpoint configuration.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; each query receives a stable derived seed.")
    parser.add_argument(
        "--prompt-template",
        default="Locate the object described by this remote sensing expression: {query}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build records and metrics from existing predictions only.")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    all_records = load_records(args)
    records = shard_records(all_records, args.num_shards, args.shard_index)
    missing_images = sum(1 for record in records if not record.get("image_path"))
    if args.num_shards > 1:
        print(
            f"Loaded {len(all_records)} {args.benchmark} records; "
            f"using shard {args.shard_index}/{args.num_shards} with {len(records)} records; "
            f"missing images in shard: {missing_images}"
        )
    else:
        print(f"Loaded {len(records)} {args.benchmark} records; missing images: {missing_images}")

    output_dir = Path(args.output_dir) / args.benchmark / args.run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = shard_suffix(args)
    records_path = output_dir / f"records{suffix}.jsonl"
    predictions_path = output_dir / f"predictions{suffix}.jsonl"
    metrics_path = output_dir / f"metrics{suffix}.json"
    rows_path = output_dir / f"metrics_per_query{suffix}.csv"
    summary_path = output_dir / f"summary{suffix}.md"

    write_jsonl(records_path, records)
    if args.merge_shards:
        records = all_records
        predictions = read_shard_predictions(output_dir, args.num_shards)
        write_jsonl(output_dir / "records.jsonl", records)
        write_jsonl(output_dir / "predictions.jsonl", predictions)
        records_path = output_dir / "records.jsonl"
        predictions_path = output_dir / "predictions.jsonl"
        metrics_path = output_dir / "metrics.json"
        rows_path = output_dir / "metrics_per_query.csv"
        summary_path = output_dir / "summary.md"
    else:
        predictions = run_inference(records, args, predictions_path)
    metrics = evaluate_predictions(records, predictions)
    write_json(metrics_path, {k: v for k, v in metrics.items() if k != "rows"})
    write_csv(rows_path, metrics["rows"])
    write_summary(summary_path, args.benchmark, metrics)

    print(f"Wrote records: {records_path}")
    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote rows: {rows_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(metrics["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
