#!/usr/bin/env python3
"""Run and audit class-addressed shared-prefix point discovery.

The input is a frozen Pixel-PIVR manifest containing the exact images, class
ontology, GT, and historical independent-class point outputs. Inference is
resumable by image and can be deterministically sharded across GPUs. The audit
refuses incomplete or duplicated outputs and reports both point accuracy and
exact equivalence to the independent-call control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

from pixel_pivr.class_wave import (
    ClassAddressedPointWaveDecoder,
    ClassPointBranch,
)
from pixel_pivr.geometry import parse_labeled_points
from pixel_pivr.io import atomic_json, atomic_jsonl, read_jsonl, sha256_file
from pixel_pivr.worker import LocateAnythingPixelPIVRWorker


EOS_TOKENS = ("<|im_end|>", "<|endoftext|>")


def stable_seed(seed: int, *parts: Any) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def strip_eos(value: str) -> str:
    text = str(value).strip()
    changed = True
    while changed:
        changed = False
        for token in EOS_TOKENS:
            if text.endswith(token):
                text = text[: -len(token)].rstrip()
                changed = True
    return text


def aggregate_answers(rows: Sequence[Mapping[str, Any]], answer_format: str = "reference_geometry") -> str:
    if answer_format not in {"reference_geometry", "geometry_only"}:
        raise ValueError(f"Unknown Round-1 answer format: {answer_format}")
    return "".join(
        (f"<ref>{row['label']}</ref>" if answer_format == "geometry_only" else "")
        + strip_eos(str(row["answer"])) for row in rows
    ) + "<|im_end|>"


def parse_class_points(rows: Sequence[Mapping[str, Any]], width: int, height: int, answer_format: str):
    if answer_format == "reference_geometry":
        return parse_labeled_points(aggregate_answers(rows), width, height)
    if answer_format != "geometry_only":
        raise ValueError(f"Unknown Round-1 answer format: {answer_format}")
    points = []
    for row in rows:
        # Class identity is a requested branch input, not a generated target.
        # Use the historical parser's exact-point deduplication and preserve raw
        # responses separately so duplicates/grammar mistakes remain auditable.
        values = parse_labeled_points(aggregate_answers([row], answer_format), width, height)
        for value in values:
            value["label"] = str(row["label"])
        points.extend(values)
    return points


def repair_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    valid_end = 0
    with path.open("rb") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                valid_end = handle.tell()
                continue
            try:
                rows.append(json.loads(line))
                valid_end = handle.tell()
            except json.JSONDecodeError as exc:
                if handle.read().strip():
                    raise RuntimeError(f"Malformed JSONL before final line: {path}") from exc
                with path.open("r+b") as repair:
                    repair.truncate(start if start >= valid_end else valid_end)
                break
    return rows


def append_jsonl_fsync(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(row), sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o664)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def resolve_image(row: Mapping[str, Any], data_root: Path) -> Path:
    for key in ("image_path", "image"):
        value = row.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = data_root / path
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Cannot resolve image for {row.get('sample_key')}")


def command_infer(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    source = read_jsonl(manifest_path)
    tasks = source[args.shard_index :: args.shard_count]
    existing = repair_jsonl(args.output.resolve())
    completed = {str(row["sample_key"]) for row in existing}
    expected = {str(row["sample_key"]) for row in tasks}
    if completed - expected:
        raise RuntimeError("Output contains rows from another shard or manifest")
    if any(row.get("class_wave_protocol", {}).get("round1_answer_format", "reference_geometry") != args.round1_answer_format for row in existing):
        raise RuntimeError("Output belongs to a different Round-1 answer format")
    if any(row.get("class_wave_protocol", {}).get("wave_implementation", "legacy") != args.wave_implementation for row in existing):
        raise RuntimeError("Output belongs to a different wave decoder implementation")

    worker = LocateAnythingPixelPIVRWorker(
        args.model.resolve(),
        args.adapter.resolve() if args.adapter is not None else None,
        device=args.device,
        dtype=args.dtype,
    )
    trained_format = (worker.checkpoint or {}).get("config", {}).get("round1_answer_format")
    if trained_format is not None and trained_format != args.round1_answer_format:
        raise RuntimeError("Round-1 evaluator answer format differs from the checkpoint training contract")
    from pixel_pivr.hybrid_class_wave import CachedHybridClassWaveDecoder

    decoder_type = (CachedHybridClassWaveDecoder if args.wave_implementation == "cached_hybrid"
                    else ClassAddressedPointWaveDecoder)
    decoder = decoder_type(
        worker,
        image_token_limit=args.image_token_limit,
        max_wave_size=args.wave_size,
        bucket_suffix_lengths=args.bucket_suffix_lengths,
    )
    print(
        f"shard={args.shard_index}/{args.shard_count} "
        f"complete={len(completed)}/{len(tasks)}",
        flush=True,
    )
    for index, source_row in enumerate(tasks, 1):
        sample_key = str(source_row["sample_key"])
        if sample_key in completed:
            continue
        image_path = resolve_image(source_row, args.data_root.resolve())
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        classes = [str(value) for value in source_row["classes"]]
        prompts = source_row.get("class_prompts") or {}
        branches = [
            ClassPointBranch(
                label=label,
                question=(
                    "Point to all instances that match the following categories: "
                    f"{prompts.get(label, label)}."
                ),
                seed=stable_seed(
                    args.seed,
                    source_row["image_id"],
                    label,
                    "isolated-point",
                ),
            )
            for label in classes
        ]
        branch_rows, stats = decoder.generate(
            image,
            branches,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            generation_mode=args.generation_mode,
        )
        answer = aggregate_answers(branch_rows, args.round1_answer_format)
        points = parse_class_points(branch_rows, width, height, args.round1_answer_format)
        point_rows = []
        for point in points:
            nx = max(0, min(1000, int(round(point["point"][0] / width * 1000))))
            ny = max(0, min(1000, int(round(point["point"][1] / height * 1000))))
            label = str(point["label"])
            point_rows.append(
                {
                    "label": label,
                    "point": [float(point["point"][0]), float(point["point"][1])],
                    "point_norm": [nx, ny],
                    "source": str(point["source"]),
                    "point_id": f"{label}:{nx}:{ny}",
                }
            )
        row = dict(source_row)
        row.update(
            {
                "point_variant": "class_addressed_shared_prefix_wave",
                "point_generation_calls": 1,
                "point_generation_latency_ms": stats["end_to_end_seconds"] * 1000.0,
                "point_query_output": answer,
                "points": point_rows,
                "class_wave_branches": branch_rows,
                "class_wave_stats": stats,
                "class_wave_protocol": {
                    "wave_implementation": args.wave_implementation,
                    "round1_answer_format": args.round1_answer_format,
                    "point_label_source": "query_branch" if args.round1_answer_format == "geometry_only" else "generated_reference",
                    "model": str(args.model.resolve()),
                    "model_config_sha256": sha256_file(args.model.resolve() / "config.json"),
                    "adapter": (
                        str(args.adapter.resolve())
                        if args.adapter is not None
                        else None
                    ),
                    "adapter_sha256": (
                        sha256_file(args.adapter.resolve())
                        if args.adapter is not None
                        else None
                    ),
                    "source_manifest": str(manifest_path),
                    "source_manifest_sha256": sha256_file(manifest_path),
                    "image_token_limit": args.image_token_limit,
                    "wave_size": args.wave_size,
                    "bucket_suffix_lengths": args.bucket_suffix_lengths,
                    "generation_mode": args.generation_mode,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "max_new_tokens": args.max_new_tokens,
                    "seed": args.seed,
                },
            }
        )
        append_jsonl_fsync(args.output.resolve(), row)
        completed.add(sample_key)
        if index % args.log_every == 0 or completed == expected:
            print(
                f"shard={args.shard_index} {len(completed)}/{len(tasks)} "
                f"points={len(point_rows)} fallbacks={stats['fallbacks']} "
                f"seconds={stats['end_to_end_seconds']:.3f}",
                flush=True,
            )
    if completed != expected:
        raise RuntimeError(f"Incomplete shard: {len(completed)}/{len(expected)}")
    atomic_json(
        args.output.resolve().with_suffix(args.output.suffix + ".done.json"),
        {
            "status": "complete",
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "records": len(completed),
            "completed_unix": time.time(),
        },
    )


def command_merge(args: argparse.Namespace) -> None:
    source = read_jsonl(args.manifest.resolve())
    expected_order = [str(row["sample_key"]) for row in source]
    rows = [row for path in args.shards for row in repair_jsonl(path.resolve())]
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["sample_key"])
        if key in by_key:
            raise RuntimeError(f"Duplicate output row: {key}")
        by_key[key] = row
    missing = set(expected_order) - set(by_key)
    extra = set(by_key) - set(expected_order)
    if missing or extra:
        raise RuntimeError(f"Incomplete merge: missing={len(missing)} extra={len(extra)}")
    atomic_jsonl(args.output.resolve(), (by_key[key] for key in expected_order))
    print(f"merged {len(expected_order)} records -> {args.output.resolve()}")


def point_inside(point: Sequence[float], box: Sequence[float]) -> bool:
    x1, y1, x2, y2 = map(float, box)
    x, y = map(float, point)
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)


def match_points(
    gt: Sequence[Mapping[str, Any]], points: Sequence[Mapping[str, Any]]
) -> tuple[int, int, int]:
    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in gt:
        by_class[str(item["label"])].append(item)
    matched: set[tuple[str, int]] = set()
    tp = 0
    for point in points:
        label = str(point["label"])
        candidates = []
        for index, item in enumerate(by_class.get(label, [])):
            key = (label, index)
            if key in matched or not point_inside(point["point"], item["hbox"]):
                continue
            x1, y1, x2, y2 = map(float, item["hbox"])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            distance = (float(point["point"][0]) - cx) ** 2 + (
                float(point["point"][1]) - cy
            ) ** 2
            candidates.append((distance, index))
        if candidates:
            chosen = min(candidates)[1]
            matched.add((label, chosen))
            tp += 1
    return tp, len(points) - tp, len(gt) - tp


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def canonical_point_set(points: Sequence[Mapping[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    """Compare semantic point content while ignoring diagnostic-only fields/order."""

    values = []
    for point in points:
        if point.get("point_norm") is not None:
            x, y = point["point_norm"]
            coordinate = (int(x), int(y))
        else:
            x, y = point["point"]
            coordinate = (round(float(x), 6), round(float(y), 6))
        values.append((str(point["label"]), *coordinate))
    return tuple(sorted(values))


def paired_bootstrap_f1_delta(
    rows: Sequence[tuple[tuple[int, int, int], tuple[int, int, int]]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    generator = random.Random(seed)
    deltas = []
    for _ in range(samples):
        chosen = [generator.randrange(len(rows)) for _ in rows]
        totals = []
        for arm in (0, 1):
            tp = sum(rows[index][arm][0] for index in chosen)
            fp = sum(rows[index][arm][1] for index in chosen)
            fn = sum(rows[index][arm][2] for index in chosen)
            totals.append(float(prf(tp, fp, fn)["f1"]))
        deltas.append(totals[1] - totals[0])
    return {
        "metric": "class-aware point-containment F1",
        "unit": "image",
        "samples": int(samples),
        "seed": int(seed),
        "delta_class_wave_minus_independent": statistics.fmean(deltas),
        "ci_95": [percentile(deltas, 0.025), percentile(deltas, 0.975)],
        "probability_delta_gt_zero": sum(value > 0 for value in deltas) / len(deltas),
    }


def branch_contract_valid(row: Mapping[str, Any]) -> bool:
    branches = row.get("class_wave_branches") or []
    classes = [str(value) for value in row.get("classes") or []]
    if len(branches) != len(classes):
        return False
    for expected, branch in zip(classes, branches):
        answer = str(branch.get("answer", ""))
        refs = re.findall(r"<ref>(.*?)</ref>", answer)
        geometry_only = row.get("class_wave_protocol", {}).get("round1_answer_format") == "geometry_only"
        if str(branch.get("label")) != expected or refs != ([] if geometry_only else [expected]):
            return False
        if not any(answer.rstrip().endswith(token) for token in EOS_TOKENS):
            return False
        if "<box>" not in answer or "</box>" not in answer:
            return False
    return True


def command_audit(args: argparse.Namespace) -> None:
    original_rows = read_jsonl(args.manifest.resolve())
    output_rows = read_jsonl(args.output.resolve())
    original = {str(row["sample_key"]): row for row in original_rows}
    output = {str(row["sample_key"]): row for row in output_rows}
    if len(original) != len(original_rows) or len(output) != len(output_rows):
        raise RuntimeError("Duplicate sample keys in audit inputs")
    if set(original) != set(output):
        raise RuntimeError("Audit output does not match the frozen manifest")
    exact_answers = 0
    exact_points = 0
    valid_branch_contracts = 0
    old_tp = old_fp = old_fn = 0
    new_tp = new_fp = new_fn = 0
    latencies = []
    old_latencies = []
    peak_memory = []
    requested_wave_sizes: set[int] = set()
    projected_global_tokens: list[int] = []
    global_moonvit_encodes = 0
    global_prefix_forwards = 0
    fallbacks = 0
    paired_rows = []
    for key, row in output.items():
        baseline = original[key]
        exact_answers += row["point_query_output"] == baseline["point_query_output"]
        exact_points += canonical_point_set(row["points"]) == canonical_point_set(
            baseline["points"]
        )
        valid_branch_contracts += branch_contract_valid(row)
        old_counts = match_points(baseline["gt"], baseline["points"])
        new_counts = match_points(row["gt"], row["points"])
        old_tp += old_counts[0]
        old_fp += old_counts[1]
        old_fn += old_counts[2]
        new_tp += new_counts[0]
        new_fp += new_counts[1]
        new_fn += new_counts[2]
        paired_rows.append((old_counts, new_counts))
        latencies.append(float(row["point_generation_latency_ms"]) / 1000.0)
        old_latencies.append(float(baseline["point_generation_latency_ms"]) / 1000.0)
        stats = row["class_wave_stats"]
        fallbacks += int(stats["fallbacks"])
        peak_memory.append(float(stats.get("peak_gpu_mb", 0.0)))
        requested_wave_sizes.add(int(stats["requested_wave_size"]))
        projected_global_tokens.append(int(stats["projected_global_tokens"]))
        global_moonvit_encodes += int(stats["global_moonvit_encodes"])
        global_prefix_forwards += int(stats["qwen_global_prefix_forwards"])
    old_metrics = prf(old_tp, old_fp, old_fn)
    new_metrics = prf(new_tp, new_fp, new_fn)
    old_mean = statistics.fmean(old_latencies)
    new_mean = statistics.fmean(latencies)
    report = {
        "schema": "class-addressed-wave-point-audit-v3",
        "status": "complete",
        "records": len(output),
        "gt": new_tp + new_fn,
        "independent_class_calls": {
            **old_metrics,
            "points": old_tp + old_fp,
        },
        "shared_prefix_class_wave": {
            **new_metrics,
            "points": new_tp + new_fp,
        },
        "paired_bootstrap": paired_bootstrap_f1_delta(
            paired_rows,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ),
        "equivalence": {
            "raw_aggregate_answer_rows": exact_answers,
            "canonical_point_set_rows": exact_points,
            "total_rows": len(output),
            "interpretation": (
                "Sampled BF16 wave execution is not required to be byte-identical: "
                "batched numerical drift can change categorical samples. Accuracy "
                "and latency are therefore reported as observed paired outcomes."
            ),
        },
        "class_branch_contract_valid_rows": valid_branch_contracts,
        "fallback_branches": fallbacks,
        "resource_contract": {
            "requested_wave_sizes": sorted(requested_wave_sizes),
            "global_moonvit_encodes": global_moonvit_encodes,
            "qwen_global_prefix_forwards": global_prefix_forwards,
            "projected_global_tokens_min": min(projected_global_tokens),
            "projected_global_tokens_max": max(projected_global_tokens),
            "peak_gpu_memory_mb_max": max(peak_memory),
            "peak_gpu_memory_mb_mean": statistics.fmean(peak_memory),
        },
        "latency_seconds": {
            "independent_calls_mean": old_mean,
            "class_wave_mean": new_mean,
            "class_wave_p50": percentile(latencies, 0.5),
            "class_wave_p95": percentile(latencies, 0.95),
            "speedup": old_mean / new_mean if new_mean else 0.0,
        },
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.manifest.resolve()),
        "output_manifest": str(args.output.resolve()),
        "output_manifest_sha256": sha256_file(args.output.resolve()),
    }
    atomic_json(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer = subparsers.add_parser("infer")
    infer.add_argument("--manifest", type=Path, required=True)
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument(
        "--adapter",
        type=Path,
        help=(
            "Optional LoRA adapter. Omit it when --model is already a complete "
            "fully fine-tuned LocateAnything checkpoint."
        ),
    )
    infer.add_argument("--data-root", type=Path, default=Path("/"))
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument("--dtype", default="bfloat16")
    infer.add_argument("--shard-index", type=int, default=0)
    infer.add_argument("--shard-count", type=int, default=1)
    infer.add_argument("--wave-size", type=int, default=32)
    infer.add_argument("--wave-implementation", choices=("legacy", "cached_hybrid"), default="legacy",
                       help="Opt-in aligned MTP/AR cached decoder; cannot resume into legacy outputs")
    infer.add_argument(
        "--bucket-suffix-lengths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Group branches by tokenized suffix length before wave decoding. "
            "Use --no-bucket-suffix-lengths for one heterogeneous 18-class wave."
        ),
    )
    infer.add_argument("--image-token-limit", type=int, default=6000)
    infer.add_argument("--round1-answer-format", choices=("reference_geometry", "geometry_only"), default="reference_geometry")
    infer.add_argument("--generation-mode", choices=("fast", "hybrid"), default="hybrid")
    infer.add_argument("--temperature", type=float, default=0.7)
    infer.add_argument("--top-p", type=float, default=0.9)
    infer.add_argument("--repetition-penalty", type=float, default=1.1)
    infer.add_argument("--max-new-tokens", type=int, default=2048)
    infer.add_argument("--seed", type=int, default=0)
    infer.add_argument("--log-every", type=int, default=5)
    infer.set_defaults(func=command_infer)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--shards", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(func=command_merge)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--report", type=Path, required=True)
    audit.add_argument("--bootstrap-samples", type=int, default=10000)
    audit.add_argument("--bootstrap-seed", type=int, default=20260906)
    audit.set_defaults(func=command_audit)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "shard_index"):
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise ValueError("Invalid shard index/count")
    args.func(args)


if __name__ == "__main__":
    main()
