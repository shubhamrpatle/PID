#!/usr/bin/env python3
"""Compare cached sequential and wave PBD6 pixel-PIVR on Balanced-100."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
for value in (str(PROJECT), str(WORKSPACE), str(PROJECT / "scripts")):
    if value not in sys.path:
        sys.path.insert(0, value)

import evaluate_locateanything_lae_style as evaluator  # noqa: E402
from evaluate_self_conditioned_point_witness_balanced100 import (  # noqa: E402
    choose_witness_box,
    load_manifest,
    point_centered_crop,
    point_to_norm,
    select_records,
)
from summarize_causal_spatial_set_order_lora_4k import (  # noqa: E402
    paired_bootstrap,
    paired_randomization_test,
    per_image_counts,
)
from summarize_self_conditioned_point_witness_lora_4k import compact_metrics  # noqa: E402
from pixel_pivr.worker import LocateAnythingPixelPIVRWorker  # noqa: E402
from zero_shot_analysis.pixel_pivr_wave import (  # noqa: E402
    AddressedCrop,
    PixelPIVRWaveDecoder,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_matched_prediction_contract(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    image_token_limit: int,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != len(records):
        raise RuntimeError("Matched LoRA prediction count changed")
    expected_image_ids = [str(row["image_id"]) for row in records]
    matched_image_ids = [str(row.get("image_id")) for row in rows]
    if len(set(matched_image_ids)) != len(matched_image_ids):
        raise RuntimeError("Matched LoRA predictions contain duplicate image IDs")
    if set(matched_image_ids) != set(expected_image_ids):
        raise RuntimeError("Matched LoRA predictions do not use the frozen image set")
    token_limits = {
        int(row.get("meta", {}).get("image_token_limit", -1)) for row in rows
    }
    if token_limits != {image_token_limit}:
        raise RuntimeError(
            "Matched LoRA image-token contract differs from the candidate: "
            f"{sorted(token_limits)} != [{image_token_limit}]"
        )
    return rows


def compact_round2_prompt(
    label: str, point: Sequence[float], width: int, height: int
) -> str:
    x, y = point_to_norm(point, width, height)
    return (
        f"Locate the single {label} containing point <box><{x}><{y}></box> "
        "in horizontal box format. Return None if absent."
    )


def mode_directory(prefix_cache_mode: str, wave_size: int) -> str:
    base = "sequential" if wave_size == 1 else f"wave{wave_size}"
    return base if prefix_cache_mode == "recompute" else f"shared_prefix_{base}"


def run_mode(
    *,
    decoder: PixelPIVRWaveDecoder,
    records: Sequence[Mapping[str, Any]],
    point_rows: Mapping[str, Mapping[str, Any]],
    alias_lookup: Mapping[str, str],
    class_set: set[str],
    wave_size: int,
    crop_side: int,
    nms_iou: float,
    class_specs: Sequence[Mapping[str, str]],
    benchmark: str,
    output: Path,
    visual_context: str,
) -> dict[str, Any]:
    from PIL import Image

    predictions_raw: list[dict[str, Any]] = []
    refinements: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    for image_index, record in enumerate(records, 1):
        image_id = str(record["image_id"])
        point_row = point_rows[image_id]
        image_path = Path(point_row["image_path"])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        branches: list[AddressedCrop] = []
        task_rows: list[dict[str, Any]] = []
        for point in point_row.get("points", []):
            crop_box = None
            crop = None
            frame_point = [float(value) for value in point["point"]]
            frame_width, frame_height = width, height
            if visual_context == "global_local":
                crop_box = point_centered_crop(
                    point["point"], width, height, crop_side
                )
                x1, y1, _x2, _y2 = crop_box
                crop = image.crop(crop_box)
                frame_point = [
                    float(point["point"][0]) - x1,
                    float(point["point"][1]) - y1,
                ]
                frame_width, frame_height = crop.width, crop.height
            point_id = str(point["point_id"])
            label = str(point["label"])
            question = compact_round2_prompt(
                label, frame_point, frame_width, frame_height
            )
            branches.append(
                AddressedCrop(
                    point_id,
                    label,
                    crop,
                    question,
                    point_normalized=point_to_norm(point["point"], width, height),
                )
            )
            task_rows.append(
                {
                    "point_id": point_id,
                    "label": label,
                    "global_point": list(point["point"]),
                    "frame_point": frame_point,
                    "crop_box": None if crop_box is None else list(crop_box),
                    "frame_size": [frame_width, frame_height],
                    "question": question,
                    "visual_context": visual_context,
                }
            )

        decoded, execution = decoder.decode_image(
            image,
            branches,
            requested_wave_size=wave_size,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            allow_none=True,
        )
        if len(decoded) != len(task_rows):
            raise RuntimeError(
                f"Address output mismatch for {image_id}: "
                f"{len(decoded)} != {len(task_rows)}"
            )
        image_predictions: list[dict[str, Any]] = []
        for task, result in zip(task_rows, decoded):
            frame_width, frame_height = task["frame_size"]
            prediction, diagnostics = choose_witness_box(
                str(result["answer"]),
                str(task["label"]),
                task["frame_point"],
                alias_lookup,
                class_set,
                int(frame_width),
                int(frame_height),
            )
            if prediction is not None:
                if task["crop_box"] is not None:
                    x1, y1, _x2, _y2 = task["crop_box"]
                    prediction["hbox"] = [
                        prediction["hbox"][0] + x1,
                        prediction["hbox"][1] + y1,
                        prediction["hbox"][2] + x1,
                        prediction["hbox"][3] + y1,
                    ]
                prediction["point_id"] = task["point_id"]
                prediction["witness_point"] = task["global_point"]
                prediction["source"] = (
                    f"pivr_{visual_context}_point_address_constrained_pbd6"
                )
                image_predictions.append(prediction)
            refinements.append(
                {
                    "image_id": image_id,
                    **task,
                    **result,
                    "prediction": prediction,
                    "diagnostics": diagnostics,
                    "requested_wave_size": wave_size,
                }
            )
        predictions_raw.append(
            {
                "dataset": benchmark,
                "image_id": image_id,
                "file_name": record["file_name"],
                "predictions": image_predictions,
            }
        )
        execution_rows.append({"image_id": image_id, **execution})
        if image_index % 10 == 0 or image_index == len(records):
            print(
                f"wave={wave_size} images={image_index}/{len(records)} "
                f"addresses={sum(row['addresses'] for row in execution_rows)}",
                flush=True,
            )

    predictions_nms = [
        {
            **row,
            "predictions": evaluator.nms_predictions(row["predictions"], nms_iou),
        }
        for row in predictions_raw
    ]
    metrics_raw = evaluator.evaluate_lae_style(
        records,
        predictions_raw,
        class_specs,
        benchmark,
        "lae-metainfo",
        "raw_class_name",
        0,
        prompt_mode=(
            f"pixel_pivr_atomic_wave{wave_size}_raw"
            if decoder.prefix_cache_mode == "recompute"
            else f"pixel_pivr_shared_prefix_atomic_wave{wave_size}_raw"
        ),
    )
    metrics_nms = evaluator.evaluate_lae_style(
        records,
        predictions_nms,
        class_specs,
        benchmark,
        "lae-metainfo",
        "raw_class_name",
        0,
        prompt_mode=(
            f"pixel_pivr_atomic_wave{wave_size}_nms{nms_iou}"
            if decoder.prefix_cache_mode == "recompute"
            else (
                f"pixel_pivr_shared_prefix_atomic_"
                f"wave{wave_size}_nms{nms_iou}"
            )
        ),
    )
    mode_dir = output / mode_directory(decoder.prefix_cache_mode, wave_size)
    atomic_jsonl(mode_dir / "predictions_raw.jsonl", predictions_raw)
    atomic_jsonl(mode_dir / "predictions_nms0p5.jsonl", predictions_nms)
    atomic_jsonl(mode_dir / "refinements.jsonl", refinements)
    atomic_jsonl(mode_dir / "execution.jsonl", execution_rows)
    atomic_json(
        mode_dir / "metrics.json", {"raw": metrics_raw, "nms_iou_0_5": metrics_nms}
    )
    return {
        "wave_size": wave_size,
        "prefix_cache_mode": decoder.prefix_cache_mode,
        "raw": compact_metrics(metrics_raw),
        "nms0p5": compact_metrics(metrics_nms),
        "predictions_nms": predictions_nms,
        "refinement_addresses": sum(row["addresses"] for row in execution_rows),
        "refinement_model_seconds": sum(row["model_seconds"] for row in execution_rows),
        "refinement_end_to_end_seconds": sum(
            row["end_to_end_seconds"] for row in execution_rows
        ),
        "maximum_effective_wave_size": max(
            (row.get("maximum_effective_wave_size", 0) for row in execution_rows),
            default=0,
        ),
        "peak_gpu_mb": max(
            (row.get("peak_gpu_mb", 0.0) for row in execution_rows), default=0.0
        ),
        "qwen_global_prefix_forwards": sum(
            int(row.get("qwen_global_prefix_forwards", 0))
            for row in execution_rows
        ),
        "qwen_branch_suffix_forwards": sum(
            int(row.get("qwen_branch_suffix_forwards", 0))
            for row in execution_rows
        ),
        "qwen_geometry_forwards": sum(
            int(row.get("qwen_geometry_forwards", 0))
            for row in execution_rows
        ),
        "qwen_total_forwards": sum(
            int(row.get("qwen_total_forwards", 0)) for row in execution_rows
        ),
        "maximum_persistent_prefix_kv_mb": max(
            (row.get("persistent_prefix_kv_mb", 0.0) for row in execution_rows),
            default=0.0,
        ),
        "fully_shared_prefix_images": sum(
            bool(row.get("fully_shared_qwen_prefix", False))
            for row in execution_rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "config/lae_benchmarks_balanced100.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT
            / "dataset/eval_balanced100/DOTAv2/manifest/"
            "dotav2_balanced_100_manifest.csv"
        ),
    )
    parser.add_argument("--benchmark", default="DOTAv2")
    parser.add_argument(
        "--model", type=Path, default=PROJECT / "model/LocateAnything-3B/checkpoint"
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        help=(
            "Optional standard LoRA checkpoint. Omit it to evaluate released "
            "Base LA without an adapter."
        ),
    )
    parser.add_argument("--point-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wave-size", type=int, default=200)
    parser.add_argument("--crop-side", type=int, default=384)
    parser.add_argument("--image-token-limit", type=int, default=1024)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument(
        "--visual-context",
        choices=("global_local", "global_only", "cached_projected_roi"),
        default="global_local",
    )
    parser.add_argument("--cached-roi-side-tokens", type=int, default=12)
    parser.add_argument(
        "--prefix-cache-mode",
        choices=("recompute", "shared"),
        default="recompute",
        help=(
            "recompute repeats the full Qwen prefix per address batch; shared "
            "stores one global-prefix KV cache per image and forks suffix caches"
        ),
    )
    parser.add_argument(
        "--geometry-prefix-mode",
        choices=("ref", "box_only"),
        default="ref",
        help=(
            "Tokens immediately before the six-token geometry block. Use box_only "
            "for unified CAW checkpoints trained to emit only <box>...</box>."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument(
        "--matched-lora-predictions",
        type=Path,
        default=(
            PROJECT
            / "runs/lae_style_predictions/"
            "DOTAv2_LocateAnything-3B_pivr_native6000_standard_directbox_lora_"
            "seed20260901_direct_multi_label_lae-metainfo_hybrid.jsonl"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.wave_size <= 200:
        parser.error("--wave-size must be in [2, 200]")
    required_paths = [
        args.config,
        args.manifest,
        args.model,
        args.point_predictions,
        args.matched_lora_predictions,
    ]
    if args.adapter is not None:
        required_paths.append(args.adapter)
    for path in required_paths:
        if not path.exists() and not path.is_symlink():
            raise FileNotFoundError(path)
    adapter_label = (
        str(args.adapter.resolve()) if args.adapter is not None else "released_base_la"
    )

    config = read_json(args.config)
    benchmark_config = dict(
        evaluator.selected_benchmarks(config, [args.benchmark])
    )[args.benchmark]
    class_specs = evaluator.benchmark_class_specs(
        args.benchmark, benchmark_config, "lae-metainfo", "raw_class_name"
    )
    alias_lookup = evaluator.build_alias_lookup(class_specs)
    class_set = {str(spec["canonical"]) for spec in class_specs}
    records = select_records(
        args.benchmark,
        benchmark_config,
        class_specs,
        load_manifest(args.manifest, args.benchmark, None),
        "raw_class_name",
    )
    if len(records) != 100 or sum(len(row.get("gt", [])) for row in records) != 9045:
        raise RuntimeError("Frozen DOTAv2 Balanced-100 protocol changed")
    point_rows = {
        str(row["image_id"]): row for row in read_jsonl(args.point_predictions)
    }
    missing = {str(row["image_id"]) for row in records} - set(point_rows)
    if missing:
        raise RuntimeError(f"Point scaffold is missing {len(missing)} images")
    matched_rows_raw = validate_matched_prediction_contract(
        args.matched_lora_predictions,
        records,
        args.image_token_limit,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "images": len(records),
                    "gt": 9045,
                    "point_addresses": sum(
                        len(point_rows[str(row["image_id"])].get("points", []))
                        for row in records
                    ),
                    "wave_sizes": [1, args.wave_size],
                    "prefix_cache_mode": args.prefix_cache_mode,
                    "adapter": adapter_label,
                },
                indent=2,
            )
        )
        return

    worker = LocateAnythingPixelPIVRWorker(
        args.model.resolve(),
        args.adapter.resolve() if args.adapter is not None else None,
        device="cuda",
        dtype="bfloat16",
    )
    if args.adapter is not None and worker.checkpoint is None:
        raise RuntimeError("Unified Pixel-PIVR LoRA adapter was not loaded")
    if args.adapter is None and worker.checkpoint is not None:
        raise RuntimeError("Released Base LA evaluation unexpectedly loaded an adapter")
    decoder = PixelPIVRWaveDecoder(
        worker,
        image_token_limit=args.image_token_limit,
        visual_context=args.visual_context,
        cached_roi_side_tokens=args.cached_roi_side_tokens,
        prefix_cache_mode=args.prefix_cache_mode,
        geometry_prefix_mode=args.geometry_prefix_mode,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    modes = {
        "sequential": run_mode(
            decoder=decoder,
            records=records,
            point_rows=point_rows,
            alias_lookup=alias_lookup,
            class_set=class_set,
            wave_size=1,
            crop_side=args.crop_side,
            nms_iou=args.nms_iou,
            class_specs=class_specs,
            benchmark=args.benchmark,
            output=args.output,
            visual_context=args.visual_context,
        ),
        "wave": run_mode(
            decoder=decoder,
            records=records,
            point_rows=point_rows,
            alias_lookup=alias_lookup,
            class_set=class_set,
            wave_size=args.wave_size,
            crop_side=args.crop_side,
            nms_iou=args.nms_iou,
            class_specs=class_specs,
            benchmark=args.benchmark,
            output=args.output,
            visual_context=args.visual_context,
        ),
    }
    matched_rows_nms = [
        {
            **row,
            "predictions": evaluator.nms_predictions(
                row.get("predictions", []), args.nms_iou
            ),
        }
        for row in matched_rows_raw
    ]
    matched_metrics_raw = evaluator.evaluate_lae_style(
        records,
        matched_rows_raw,
        class_specs,
        args.benchmark,
        "lae-metainfo",
        "raw_class_name",
        0,
        prompt_mode="matched_standard_lora_native6000_raw",
    )
    matched_metrics_nms = evaluator.evaluate_lae_style(
        records,
        matched_rows_nms,
        class_specs,
        args.benchmark,
        "lae-metainfo",
        "raw_class_name",
        0,
        prompt_mode=f"matched_standard_lora_native6000_nms{args.nms_iou}",
    )
    matched_per_image = per_image_counts(records, matched_rows_nms)
    point_seconds = sum(
        float(
            row.get(
                "point_generation_latency_ms",
                row.get("latency_ms", 0.0),
            )
        )
        for row in point_rows.values()
    ) / 1000
    point_latency_field = (
        "point_generation_latency_ms"
        if all("point_generation_latency_ms" in row for row in point_rows.values())
        else "latency_ms"
    )
    comparison: dict[str, Any] = {
        "schema_version": (
            "pivr-shared-prefix-wave-balanced100-v2"
            if args.prefix_cache_mode == "shared"
            else "pivr-visual-context-ablation-balanced100-v1"
        ),
        "protocol": {
            "benchmark": "DOTAv2 Balanced-100",
            "images": 100,
            "gt_instances": 9045,
            "prediction_cap": None,
            "matching": "class-aware one-to-one greedy IoU >= 0.5",
            "nms_iou": args.nms_iou,
            "crop_side": args.crop_side,
            "point_predictions": str(args.point_predictions.resolve()),
            "adapter": adapter_label,
            "visual_context": args.visual_context,
            "prefix_cache_mode": args.prefix_cache_mode,
            "qwen_prefix_scope": "one persistent global KV cache per image",
            "image_token_limit": args.image_token_limit,
            "round2_prompt_schema": "single-point-containment-compact-v1",
            "geometry_prefix_mode": args.geometry_prefix_mode,
            "allow_none": True,
            "single_training_seed": True,
            "point_latency_field": point_latency_field,
            "cached_roi_side_tokens": (
                args.cached_roi_side_tokens
                if args.visual_context == "cached_projected_roi"
                else None
            ),
            "cached_roi_visual_tokens": (
                args.cached_roi_side_tokens**2
                if args.visual_context == "cached_projected_roi"
                else None
            ),
        },
        "shared_point_scaffold_seconds": point_seconds,
        "matched_lora": {
            "prediction_path": str(args.matched_lora_predictions.resolve()),
            "prediction_sha256": hashlib.sha256(
                args.matched_lora_predictions.read_bytes()
            ).hexdigest(),
            "image_token_limit": args.image_token_limit,
            "raw": compact_metrics(matched_metrics_raw),
            "nms0p5": compact_metrics(matched_metrics_nms),
        },
        "modes": {},
    }
    for name, result in modes.items():
        predictions = result.pop("predictions_nms")
        candidate_per_image = per_image_counts(records, predictions)
        comparison["modes"][name] = {
            **result,
            "total_pipeline_seconds": point_seconds
            + result["refinement_end_to_end_seconds"],
            "mean_pipeline_seconds_per_image": (
                point_seconds + result["refinement_end_to_end_seconds"]
            )
            / len(records),
            "paired_bootstrap_vs_matched_lora": paired_bootstrap(
                candidate_per_image,
                matched_per_image,
                samples=args.bootstrap_samples,
                seed=20260901 + (0 if name == "sequential" else 1),
            ),
            "paired_randomization_vs_matched_lora": paired_randomization_test(
                candidate_per_image,
                matched_per_image,
                samples=args.bootstrap_samples,
                seed=20261901 + (0 if name == "sequential" else 1),
            ),
        }
    sequential_counts = per_image_counts(
        records,
        read_jsonl(
            args.output
            / mode_directory(args.prefix_cache_mode, 1)
            / "predictions_nms0p5.jsonl"
        ),
    )
    wave_counts = per_image_counts(
        records,
        read_jsonl(
            args.output
            / mode_directory(args.prefix_cache_mode, args.wave_size)
            / "predictions_nms0p5.jsonl"
        ),
    )
    comparison["wave_vs_sequential"] = {
        "paired_bootstrap": paired_bootstrap(
            wave_counts,
            sequential_counts,
            samples=args.bootstrap_samples,
            seed=20262901,
        ),
        "paired_randomization": paired_randomization_test(
            wave_counts,
            sequential_counts,
            samples=args.bootstrap_samples,
            seed=20263901,
        ),
        "refinement_model_speedup": (
            comparison["modes"]["sequential"]["refinement_model_seconds"]
            / max(1e-12, comparison["modes"]["wave"]["refinement_model_seconds"])
        ),
        "refinement_end_to_end_speedup": (
            comparison["modes"]["sequential"]["refinement_end_to_end_seconds"]
            / max(
                1e-12,
                comparison["modes"]["wave"]["refinement_end_to_end_seconds"],
            )
        ),
    }
    atomic_json(args.output / "comparison.json", comparison)
    lines = [
        "# Pixel-PIVR Sequential/Wave Balanced-100",
        "",
        "| Mode | P | R | F1 | Refine model s | Refine E2E s | Peak GPU MB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    matched_metric = comparison["matched_lora"]["nms0p5"]
    lines.append(
        f"| matched LoRA native-6000 | {100*matched_metric['precision']:.2f} | "
        f"{100*matched_metric['recall']:.2f} | {100*matched_metric['f1']:.2f} | "
        "n/a | n/a | n/a |"
    )
    for name in ("sequential", "wave"):
        row = comparison["modes"][name]
        metric = row["nms0p5"]
        lines.append(
            f"| {name} | {100*metric['precision']:.2f} | {100*metric['recall']:.2f} | "
            f"{100*metric['f1']:.2f} | {row['refinement_model_seconds']:.2f} | "
            f"{row['refinement_end_to_end_seconds']:.2f} | {row['peak_gpu_mb']:.1f} |"
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
