#!/usr/bin/env python3
"""Apply the preregistered paired gate to two SPWR 4K LoRA seeds."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import evaluate_locateanything_lae_style as evaluator  # noqa: E402
from summarize_causal_spatial_set_order_lora_4k import (  # noqa: E402
    paired_bootstrap,
    paired_randomization_test,
    per_image_counts,
    prediction_index,
)


SEEDS = (20260827, 20260828)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def resolve_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one match for {pattern}; found {matches}")
    return matches[0]


def compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    at50 = metrics["la_metrics"]["f1_iou_0.50"]
    return {
        "images": int(metrics["images"]),
        "gt": int(metrics["gt_instances"]),
        "predictions": int(metrics["predicted_boxes"]),
        "tp": int(at50["tp"]),
        "fp": int(at50["fp"]),
        "fn": int(at50["fn"]),
        "precision": float(at50["precision"]),
        "recall": float(at50["recall"]),
        "f1": float(at50["f1"]),
        "ap50": float(metrics["accuracy"]["ap50"]),
        "map_50_95": float(metrics["accuracy"]["map_50_95"]),
    }


def fixed_square_controls(
    records: Sequence[Mapping[str, Any]],
    point_rows: Sequence[Mapping[str, Any]],
    sides: Sequence[int] = (16, 20, 24, 32, 40, 48, 64),
) -> dict[str, dict[str, Any]]:
    """Measure whether generated points plus a fixed square explain the gain."""
    result = {}
    for side in sides:
        half = float(side) / 2.0
        prediction_rows = []
        for row in point_rows:
            width = int(row["meta"]["width"])
            height = int(row["meta"]["height"])
            predictions = []
            for point in row.get("points", []):
                x, y = map(float, point["point"])
                predictions.append(
                    {
                        "label": point["label"],
                        "hbox": [
                            max(0.0, x - half),
                            max(0.0, y - half),
                            min(float(width), x + half),
                            min(float(height), y + half),
                        ],
                        "score": 1.0,
                    }
                )
            prediction_rows.append(
                {
                    "dataset": "DOTAv2",
                    "image_id": row["image_id"],
                    "predictions": evaluator.nms_predictions(predictions, 0.5),
                }
            )
        counts = per_image_counts(records, prediction_rows)
        tp = sum(int(row["tp"]) for row in counts)
        fp = sum(int(row["fp"]) for row in counts)
        fn = sum(int(row["fn"]) for row in counts)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[str(side)] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": tp + fp,
        }
    return result


def evaluate_rows(
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    class_specs: Sequence[Mapping[str, str]],
    prompt_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = evaluator.evaluate_lae_style(
        records,
        rows,
        class_specs,
        "DOTAv2",
        "lae-metainfo",
        "raw_class_name",
        0,
        prompt_mode=prompt_mode,
    )
    compact = compact_metrics(metrics)
    compact["per_image"] = per_image_counts(records, rows)
    return compact, metrics


def diagnostic_slices(
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    class_specs: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Report class coverage and density slices hidden by micro-F1."""
    index = prediction_index(rows)
    class_names = [str(spec["canonical"]) for spec in class_specs]
    per_class = {
        class_name: evaluator.f1_for_threshold(
            records, index, [class_name], 0.5, 0
        )
        for class_name in class_names
    }
    density = {}
    for label, lower, upper in (
        ("sparse_lt20", 0, 20),
        ("medium_20_99", 20, 100),
        ("dense_ge100", 100, sys.maxsize),
    ):
        selected = [
            record for record in records
            if lower <= len(record.get("gt", [])) < upper
        ]
        density[label] = {
            "images": len(selected),
            "gt": sum(len(record.get("gt", [])) for record in selected),
            **evaluator.f1_for_threshold(selected, index, class_names, 0.5, 0),
        }
    return {
        "macro_f1": statistics.mean(row["f1"] for row in per_class.values()),
        "macro_recall": statistics.mean(row["recall"] for row in per_class.values()),
        "predicted_class_count": sum(
            row["tp"] + row["fp"] > 0 for row in per_class.values()
        ),
        "class_count": len(per_class),
        "per_class": per_class,
        "density": density,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path("/data1/tmp_shubham/self_conditioned_point_witness_lora_4k_v1"),
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/data1/tmp_shubham/self_conditioned_point_witness_eval_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "runs/self_conditioned_point_witness_lora_4k_v1",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = evaluator.read_json(PROJECT / "config/lae_benchmarks_balanced100.json")
    benchmark = config["benchmarks"]["DOTAv2"]
    class_specs = evaluator.benchmark_class_specs(
        "DOTAv2", benchmark, "lae-metainfo", "raw_class_name"
    )
    records = evaluator.load_lae_records(
        "DOTAv2", benchmark, class_specs, "raw_class_name"
    )
    if len(records) != 100 or sum(len(record["gt"]) for record in records) != 9045:
        raise RuntimeError("Frozen Balanced100 protocol changed")

    prediction_root = PROJECT / "runs/lae_style_predictions"
    metric_root = PROJECT / "runs/lae_style_metrics"
    standard_metric_path = resolve_one(
        metric_root,
        "*causal_pointrelay_4k_standard_best_native1024_uncapped_seed0*metrics.json",
    )
    standard_prediction_path = prediction_root / standard_metric_path.name.replace(
        "_metrics.json", ".jsonl"
    )
    standard_rows = read_jsonl(standard_prediction_path)
    standard_raw = compact_metrics(read_json(standard_metric_path))
    standard_raw["per_image"] = per_image_counts(records, standard_rows)
    standard_nms_rows = []
    for row in standard_rows:
        standard_nms_rows.append(
            {**row, "predictions": evaluator.nms_predictions(row.get("predictions", []), 0.5)}
        )
    standard_nms_path = args.output / "matched_standard_lora_predictions_nms0p5.jsonl"
    write_jsonl(standard_nms_path, standard_nms_rows)
    standard_nms, standard_nms_full = evaluate_rows(
        records, standard_nms_rows, class_specs, "matched_standard_nms0p5"
    )
    (args.output / "matched_standard_lora_metrics_nms0p5.json").write_text(
        json.dumps(standard_nms_full, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result: dict[str, Any] = {
        "schema_version": "self-conditioned-point-witness-lora-4k-gate-v1",
        "scope": "DOTAv2 Balanced100 architecture-development diagnostic",
        "protocol": {
            "images": 100,
            "gt_hbb": 9045,
            "classes": 18,
            "image_token_limit": 1024,
            "max_detections_per_image": 0,
            "matching": "class-aware one-to-one greedy IoU >= 0.5",
            "inference_seed": 0,
            "primary_set_assembly": "exact point deduplication plus class-wise NMS IoU 0.5",
        },
        "standard_lora_raw": standard_raw,
        "standard_lora_nms0p5": standard_nms,
        "spwr": {},
        "point_visual_reentry_crop384": {},
        "point_visual_reentry_crop384_nearest": {},
        "point_visual_reentry_scale_ablation_seed20260827": {},
        "fixed_square_point_controls": {},
        "diagnostic_slices": {
            "standard_lora_nms0p5": diagnostic_slices(
                records, standard_nms_rows, class_specs
            )
        },
    }
    same_protocol_path = args.eval_root / "matched_standard_same_protocol" / "metrics.json"
    if same_protocol_path.is_file():
        same_protocol_summary = read_json(same_protocol_path)
        same_protocol_raw_rows = read_jsonl(
            same_protocol_path.parent / "predictions_raw.jsonl"
        )
        same_protocol_nms_rows = read_jsonl(
            same_protocol_path.parent / "predictions_nms0p5.jsonl"
        )
        same_protocol_raw = compact_metrics(same_protocol_summary["raw"])
        same_protocol_raw["per_image"] = per_image_counts(records, same_protocol_raw_rows)
        same_protocol_nms = compact_metrics(same_protocol_summary["nms_iou_0_5"])
        same_protocol_nms["per_image"] = per_image_counts(records, same_protocol_nms_rows)
        result["standard_lora_same_protocol_raw"] = same_protocol_raw
        result["standard_lora_same_protocol_nms0p5"] = same_protocol_nms
    same_protocol_crop_path = (
        args.eval_root / "matched_standard_same_protocol_crop384" / "metrics.json"
    )
    if same_protocol_crop_path.is_file():
        same_protocol_crop_summary = read_json(same_protocol_crop_path)
        same_protocol_crop_raw_rows = read_jsonl(
            same_protocol_crop_path.parent / "predictions_raw.jsonl"
        )
        same_protocol_crop_nms_rows = read_jsonl(
            same_protocol_crop_path.parent / "predictions_nms0p5.jsonl"
        )
        same_protocol_crop_raw = compact_metrics(same_protocol_crop_summary["raw"])
        same_protocol_crop_raw["per_image"] = per_image_counts(
            records, same_protocol_crop_raw_rows
        )
        same_protocol_crop_nms = compact_metrics(
            same_protocol_crop_summary["nms_iou_0_5"]
        )
        same_protocol_crop_nms["per_image"] = per_image_counts(
            records, same_protocol_crop_nms_rows
        )
        result["standard_lora_same_protocol_crop384_raw"] = same_protocol_crop_raw
        result["standard_lora_same_protocol_crop384_nms0p5"] = same_protocol_crop_nms
    fixed_point_control_dirs = {
        "standard_refiner_spwr_points_crop384": (
            "standard_refiner_spwr_seed20260827_points_crop384"
        ),
        **{
            f"standard_direct_refiner_spwr_seed{seed}_points_crop384": (
                f"standard_direct_refiner_spwr_seed{seed}_points_crop384"
            )
            for seed in args.seeds
        },
    }
    for key, directory in fixed_point_control_dirs.items():
        metric_path = args.eval_root / directory / "metrics.json"
        if not metric_path.is_file():
            continue
        summary = read_json(metric_path)
        raw_rows = read_jsonl(metric_path.parent / "predictions_raw.jsonl")
        nms_rows = read_jsonl(metric_path.parent / "predictions_nms0p5.jsonl")
        raw = compact_metrics(summary["raw"])
        raw["per_image"] = per_image_counts(records, raw_rows)
        nms = compact_metrics(summary["nms_iou_0_5"])
        nms["per_image"] = per_image_counts(records, nms_rows)
        result[f"{key}_raw"] = raw
        result[f"{key}_nms0p5"] = nms
        nearest_metric_path = metric_path.parent / "nearest_point_control" / "metrics.json"
        if nearest_metric_path.is_file():
            nearest_summary = read_json(nearest_metric_path)
            nearest_raw_rows = read_jsonl(
                nearest_metric_path.parent / "predictions_raw.jsonl"
            )
            nearest_nms_rows = read_jsonl(
                nearest_metric_path.parent / "predictions_nms0p5.jsonl"
            )
            nearest_raw = compact_metrics(nearest_summary["raw"])
            nearest_raw["per_image"] = per_image_counts(records, nearest_raw_rows)
            nearest_nms = compact_metrics(nearest_summary["nms_iou_0_5"])
            nearest_nms["per_image"] = per_image_counts(records, nearest_nms_rows)
            result[f"{key}_nearest_raw"] = nearest_raw
            result[f"{key}_nearest_nms0p5"] = nearest_nms
            if key.startswith("standard_direct_refiner_spwr_seed"):
                result["diagnostic_slices"][f"{key}_nearest_nms0p5"] = (
                    diagnostic_slices(records, nearest_nms_rows, class_specs)
                )
    for seed in args.seeds:
        train_dir = args.train_root / f"seed{seed}"
        done = read_json(train_dir / "done.json")
        if not done.get("complete_one_pass") or int(done["record_exposures"]) != 16000:
            raise RuntimeError(f"SPWR seed {seed} is not a complete matched one-pass run")
        checkpoint = torch.load(train_dir / "best.pt", map_location="cpu", weights_only=False)
        eval_dir = args.eval_root / f"seed{seed}"
        summary = read_json(eval_dir / "metrics.json")
        point_rows = read_jsonl(eval_dir / "point_scaffold_predictions.jsonl")
        raw_rows = read_jsonl(eval_dir / "predictions_raw.jsonl")
        nms_rows = read_jsonl(eval_dir / "predictions_nms0p5.jsonl")
        raw = compact_metrics(summary["raw"])
        raw["per_image"] = per_image_counts(records, raw_rows)
        nms = compact_metrics(summary["nms_iou_0_5"])
        nms["per_image"] = per_image_counts(records, nms_rows)
        bootstrap = paired_bootstrap(
            nms["per_image"], standard_nms["per_image"],
            samples=args.bootstrap_samples, seed=seed,
        )
        randomization = paired_randomization_test(
            nms["per_image"], standard_nms["per_image"],
            samples=args.bootstrap_samples, seed=seed + 100000,
        )
        result["spwr"][str(seed)] = {
            "best_step": int(checkpoint["step"]),
            "best_validation_native_loss": float(checkpoint["validation"]["native_loss"]),
            "point_scaffold": summary["point_scaffold"],
            "refinement_tasks": int(summary["refinement_tasks"]),
            "successful_refinements": int(summary["successful_refinements"]),
            "raw": raw,
            "nms0p5": nms,
            "paired_bootstrap_vs_standard_nms0p5": bootstrap,
            "paired_randomization_vs_standard_nms0p5": randomization,
        }
        result["fixed_square_point_controls"][str(seed)] = fixed_square_controls(
            records, point_rows
        )

        crop_dir = args.eval_root / f"seed{seed}_crop384"
        crop_metric_path = crop_dir / "metrics.json"
        if crop_metric_path.is_file():
            crop_summary = read_json(crop_metric_path)
            crop_raw_rows = read_jsonl(crop_dir / "predictions_raw.jsonl")
            crop_nms_rows = read_jsonl(crop_dir / "predictions_nms0p5.jsonl")
            crop_raw = compact_metrics(crop_summary["raw"])
            crop_raw["per_image"] = per_image_counts(records, crop_raw_rows)
            crop_nms = compact_metrics(crop_summary["nms_iou_0_5"])
            crop_nms["per_image"] = per_image_counts(records, crop_nms_rows)
            result["point_visual_reentry_crop384"][str(seed)] = {
                "status": "posthoc_mechanism_test",
                "refine_crop_size": int(crop_summary["refine_crop_size"]),
                "raw": crop_raw,
                "nms0p5": crop_nms,
                "paired_bootstrap_vs_standard_nms0p5": paired_bootstrap(
                    crop_nms["per_image"], standard_nms["per_image"],
                    samples=args.bootstrap_samples, seed=seed + 200000,
                ),
                "paired_randomization_vs_standard_nms0p5": paired_randomization_test(
                    crop_nms["per_image"], standard_nms["per_image"],
                    samples=args.bootstrap_samples, seed=seed + 300000,
                ),
            }
            result["diagnostic_slices"][f"point_visual_reentry_crop384_seed{seed}"] = (
                diagnostic_slices(records, crop_nms_rows, class_specs)
            )
            nearest_metric_path = crop_dir / "nearest_point_control" / "metrics.json"
            if nearest_metric_path.is_file():
                nearest_summary = read_json(nearest_metric_path)
                nearest_raw_rows = read_jsonl(
                    nearest_metric_path.parent / "predictions_raw.jsonl"
                )
                nearest_nms_rows = read_jsonl(
                    nearest_metric_path.parent / "predictions_nms0p5.jsonl"
                )
                nearest_raw = compact_metrics(nearest_summary["raw"])
                nearest_raw["per_image"] = per_image_counts(records, nearest_raw_rows)
                nearest_nms = compact_metrics(nearest_summary["nms_iou_0_5"])
                nearest_nms["per_image"] = per_image_counts(records, nearest_nms_rows)
                result["point_visual_reentry_crop384_nearest"][str(seed)] = {
                    "status": "non_oracle_nearest_box_selection_stress_test",
                    "raw": nearest_raw,
                    "nms0p5": nearest_nms,
                    "paired_bootstrap_vs_standard_nms0p5": paired_bootstrap(
                        nearest_nms["per_image"], standard_nms["per_image"],
                        samples=args.bootstrap_samples, seed=seed + 800000,
                    ),
                    "paired_randomization_vs_standard_nms0p5": paired_randomization_test(
                        nearest_nms["per_image"], standard_nms["per_image"],
                        samples=args.bootstrap_samples, seed=seed + 900000,
                    ),
                }
            if "standard_lora_same_protocol_crop384_nms0p5" in result:
                control = result["standard_lora_same_protocol_crop384_nms0p5"]
                result["point_visual_reentry_crop384"][str(seed)][
                    "paired_bootstrap_vs_same_protocol_crop384"
                ] = paired_bootstrap(
                    crop_nms["per_image"], control["per_image"],
                    samples=args.bootstrap_samples, seed=seed + 400000,
                )
                result["point_visual_reentry_crop384"][str(seed)][
                    "paired_randomization_vs_same_protocol_crop384"
                ] = paired_randomization_test(
                    crop_nms["per_image"], control["per_image"],
                    samples=args.bootstrap_samples, seed=seed + 500000,
                )
            direct_control_key = (
                f"standard_direct_refiner_spwr_seed{seed}_points_crop384_nms0p5"
            )
            if direct_control_key in result:
                control = result[direct_control_key]
                result["point_visual_reentry_crop384"][str(seed)][
                    "paired_bootstrap_vs_standard_direct_same_points_crop384"
                ] = paired_bootstrap(
                    crop_nms["per_image"], control["per_image"],
                    samples=args.bootstrap_samples, seed=seed + 600000,
                )
                result["point_visual_reentry_crop384"][str(seed)][
                    "paired_randomization_vs_standard_direct_same_points_crop384"
                ] = paired_randomization_test(
                    crop_nms["per_image"], control["per_image"],
                    samples=args.bootstrap_samples, seed=seed + 700000,
                )
            direct_nearest_key = (
                f"standard_direct_refiner_spwr_seed{seed}_points_crop384_nearest_nms0p5"
            )
            nearest_treatment = result["point_visual_reentry_crop384_nearest"].get(
                str(seed)
            )
            if direct_nearest_key in result and nearest_treatment is not None:
                control = result[direct_nearest_key]
                nearest_treatment[
                    "paired_bootstrap_vs_standard_direct_same_points_crop384_nearest"
                ] = paired_bootstrap(
                    nearest_treatment["nms0p5"]["per_image"],
                    control["per_image"],
                    samples=args.bootstrap_samples,
                    seed=seed + 1000000,
                )
                nearest_treatment[
                    "paired_randomization_vs_standard_direct_same_points_crop384_nearest"
                ] = paired_randomization_test(
                    nearest_treatment["nms0p5"]["per_image"],
                    control["per_image"],
                    samples=args.bootstrap_samples,
                    seed=seed + 1100000,
                )

        if seed == args.seeds[0]:
            for crop_size in (256, 384, 512):
                scale_dir = args.eval_root / f"seed{seed}_crop{crop_size}"
                scale_metric_path = scale_dir / "metrics.json"
                if not scale_metric_path.is_file():
                    continue
                scale_summary = read_json(scale_metric_path)
                scale_raw = compact_metrics(scale_summary["raw"])
                scale_nms = compact_metrics(scale_summary["nms_iou_0_5"])
                result["point_visual_reentry_scale_ablation_seed20260827"][
                    str(crop_size)
                ] = {"raw": scale_raw, "nms0p5": scale_nms}

    seed_gates = {}
    treatment_f1 = []
    for seed in args.seeds:
        row = result["spwr"][str(seed)]["nms0p5"]
        treatment_f1.append(row["f1"])
        bootstrap = result["spwr"][str(seed)]["paired_bootstrap_vs_standard_nms0p5"]
        randomization = result["spwr"][str(seed)]["paired_randomization_vs_standard_nms0p5"]
        gates = {
            "beats_raw_matched_lora_f1": row["f1"] > standard_raw["f1"],
            "beats_nms_matched_lora_f1": row["f1"] > standard_nms["f1"],
            "increases_tp_vs_nms_control": row["tp"] > standard_nms["tp"],
            "paired_ci_lower_above_zero": bootstrap["ci95_delta_f1"][0] > 0,
            "paired_randomization_p_below_0_05": randomization["one_sided_p_value"] < 0.05,
            "predictions_within_2x_nms_control": row["predictions"] <= 2 * standard_nms["predictions"],
        }
        gates["passes"] = all(gates.values())
        seed_gates[str(seed)] = gates
    result["decision"] = {
        "preregistered_gate": (
            "Every independent seed must beat both raw and IoU-0.5-NMS matched LoRA in F1, "
            "increase TP over the NMS control, have paired-bootstrap CI lower bound > 0, "
            "one-sided paired-randomization p < 0.05, and emit <=2x NMS-control boxes."
        ),
        "mean_f1": statistics.mean(treatment_f1),
        "sample_std_f1": statistics.stdev(treatment_f1) if len(treatment_f1) > 1 else None,
        "seed_gates": seed_gates,
        "passes": all(row["passes"] for row in seed_gates.values()),
    }
    crop_gates = {}
    for seed, treatment in result["point_visual_reentry_crop384"].items():
        row = treatment["nms0p5"]
        bootstrap = treatment["paired_bootstrap_vs_standard_nms0p5"]
        randomization = treatment["paired_randomization_vs_standard_nms0p5"]
        crop_gates[seed] = {
            "beats_nms_matched_lora_f1": row["f1"] > standard_nms["f1"],
            "increases_tp_vs_nms_control": row["tp"] > standard_nms["tp"],
            "paired_ci_lower_above_zero": bootstrap["ci95_delta_f1"][0] > 0,
            "paired_randomization_p_below_0_05": randomization["one_sided_p_value"] < 0.05,
        }
        crop_gates[seed]["passes"] = all(crop_gates[seed].values())
    result["posthoc_visual_reentry_decision"] = {
        "status": "mechanism-discovery test; seed 20260828 is an independent-training-seed confirmation, not a new benchmark",
        "all_requested_seeds_available": len(crop_gates) == len(args.seeds),
        "seed_gates": crop_gates,
        "passes_confirmation_gate": (
            len(crop_gates) == len(args.seeds)
            and all(row["passes"] for row in crop_gates.values())
        ),
    }
    same_protocol_crop_gates = {}
    same_protocol_crop_control = result.get(
        "standard_lora_same_protocol_crop384_nms0p5"
    )
    if same_protocol_crop_control is not None:
        for seed, treatment in result["point_visual_reentry_crop384"].items():
            row = treatment["nms0p5"]
            bootstrap = treatment["paired_bootstrap_vs_same_protocol_crop384"]
            randomization = treatment[
                "paired_randomization_vs_same_protocol_crop384"
            ]
            same_protocol_crop_gates[seed] = {
                "beats_same_protocol_crop384_f1": (
                    row["f1"] > same_protocol_crop_control["f1"]
                ),
                "increases_tp_vs_same_protocol_crop384": (
                    row["tp"] > same_protocol_crop_control["tp"]
                ),
                "paired_ci_lower_above_zero": bootstrap["ci95_delta_f1"][0] > 0,
                "paired_randomization_p_below_0_05": (
                    randomization["one_sided_p_value"] < 0.05
                ),
            }
            same_protocol_crop_gates[seed]["passes"] = all(
                same_protocol_crop_gates[seed].values()
            )
    result["same_protocol_visual_reentry_decision"] = {
        "status": (
            "complete" if same_protocol_crop_control is not None else "pending control"
        ),
        "control": "matched standard LoRA with its own predicted points and crop384",
        "seed_gates": same_protocol_crop_gates,
        "passes": (
            same_protocol_crop_control is not None
            and len(same_protocol_crop_gates) == len(args.seeds)
            and all(row["passes"] for row in same_protocol_crop_gates.values())
        ),
    }
    fixed_point_refiner_gates = {}
    for seed in args.seeds:
        direct_control = result.get(
            f"standard_direct_refiner_spwr_seed{seed}_points_crop384_nms0p5"
        )
        treatment = result["point_visual_reentry_crop384"].get(str(seed))
        if direct_control is None or treatment is None:
            continue
        row = treatment["nms0p5"]
        bootstrap = treatment[
            "paired_bootstrap_vs_standard_direct_same_points_crop384"
        ]
        randomization = treatment[
            "paired_randomization_vs_standard_direct_same_points_crop384"
        ]
        gate = {
            "beats_direct_crop_control_f1": row["f1"] > direct_control["f1"],
            "increases_tp_vs_direct_crop_control": row["tp"] > direct_control["tp"],
            "paired_ci_lower_above_zero": bootstrap["ci95_delta_f1"][0] > 0,
            "paired_randomization_p_below_0_05": (
                randomization["one_sided_p_value"] < 0.05
            ),
        }
        gate["passes"] = all(gate.values())
        fixed_point_refiner_gates[str(seed)] = gate
    result["fixed_point_refiner_decision"] = {
        "status": (
            "complete"
            if len(fixed_point_refiner_gates) == len(args.seeds)
            else "pending control"
        ),
        "comparison": (
            "For each seed, SPWR and standard LoRA use identical predicted points, "
            "384px crops, containment filtering, and NMS; standard LoRA uses its "
            "native class-detection prompt."
        ),
        "seed_gates": fixed_point_refiner_gates,
        "passes": (
            len(fixed_point_refiner_gates) == len(args.seeds)
            and all(gate["passes"] for gate in fixed_point_refiner_gates.values())
        ),
    }
    nearest_refiner_gates = {}
    for seed in args.seeds:
        direct_control = result.get(
            f"standard_direct_refiner_spwr_seed{seed}_points_crop384_nearest_nms0p5"
        )
        treatment = result["point_visual_reentry_crop384_nearest"].get(str(seed))
        if direct_control is None or treatment is None:
            continue
        bootstrap = treatment[
            "paired_bootstrap_vs_standard_direct_same_points_crop384_nearest"
        ]
        randomization = treatment[
            "paired_randomization_vs_standard_direct_same_points_crop384_nearest"
        ]
        gate = {
            "beats_nearest_direct_crop_control_f1": (
                treatment["nms0p5"]["f1"] > direct_control["f1"]
            ),
            "increases_tp_vs_nearest_direct_crop_control": (
                treatment["nms0p5"]["tp"] > direct_control["tp"]
            ),
            "paired_ci_lower_above_zero": bootstrap["ci95_delta_f1"][0] > 0,
            "paired_randomization_p_below_0_05": (
                randomization["one_sided_p_value"] < 0.05
            ),
        }
        gate["passes"] = all(gate.values())
        nearest_refiner_gates[str(seed)] = gate
    result["nearest_point_refiner_decision"] = {
        "status": (
            "complete"
            if len(nearest_refiner_gates) == len(args.seeds)
            else "pending control"
        ),
        "comparison": (
            "For each seed, both adapters use identical predicted points, 384px crops, "
            "nearest-target-box selection without GT, and NMS."
        ),
        "seed_gates": nearest_refiner_gates,
        "passes": (
            len(nearest_refiner_gates) == len(args.seeds)
            and all(gate["passes"] for gate in nearest_refiner_gates.values())
        ),
    }
    factorized_gates = {}
    for seed in args.seeds:
        key = f"standard_direct_refiner_spwr_seed{seed}_points_crop384_nearest_nms0p5"
        factorized = result.get(key)
        if factorized is None:
            continue
        bootstrap = paired_bootstrap(
            factorized["per_image"],
            standard_nms["per_image"],
            samples=args.bootstrap_samples,
            seed=seed + 1200000,
        )
        randomization = paired_randomization_test(
            factorized["per_image"],
            standard_nms["per_image"],
            samples=args.bootstrap_samples,
            seed=seed + 1300000,
        )
        gate = {
            "metrics": factorized,
            "paired_bootstrap_vs_standard_nms0p5": bootstrap,
            "paired_randomization_vs_standard_nms0p5": randomization,
            "beats_full_image_standard_lora_f1": factorized["f1"] > standard_nms["f1"],
            "increases_tp_vs_full_image_standard_lora": factorized["tp"] > standard_nms["tp"],
            "paired_ci_lower_above_zero": bootstrap["ci95_delta_f1"][0] > 0,
            "paired_randomization_p_below_0_05": randomization["one_sided_p_value"] < 0.05,
        }
        gate["passes"] = all(
            gate[name]
            for name in (
                "beats_full_image_standard_lora_f1",
                "increases_tp_vs_full_image_standard_lora",
                "paired_ci_lower_above_zero",
                "paired_randomization_p_below_0_05",
            )
        )
        factorized_gates[str(seed)] = gate
    result["factorized_point_reentry_decision"] = {
        "status": (
            "complete" if len(factorized_gates) == len(args.seeds) else "pending control"
        ),
        "comparison": (
            "SPWR learns full-image labelled point discovery; the matched standard-LoRA "
            "expert decodes one local box from each predicted-point crop. The baseline is "
            "the same standard LoRA on the full image with identical NMS."
        ),
        "seed_gates": factorized_gates,
        "passes": (
            len(factorized_gates) == len(args.seeds)
            and all(gate["passes"] for gate in factorized_gates.values())
        ),
    }

    serializable = json.loads(json.dumps(result))
    for key in (
        "standard_lora_raw",
        "standard_lora_nms0p5",
        "standard_lora_same_protocol_raw",
        "standard_lora_same_protocol_nms0p5",
        "standard_lora_same_protocol_crop384_raw",
        "standard_lora_same_protocol_crop384_nms0p5",
        "standard_refiner_spwr_points_crop384_raw",
        "standard_refiner_spwr_points_crop384_nms0p5",
    ):
        if key not in serializable:
            continue
        serializable[key].pop("per_image", None)
    for key, row in serializable.items():
        if key.startswith("standard_direct_refiner_spwr_seed"):
            row.pop("per_image", None)
    for row in serializable["spwr"].values():
        row["raw"].pop("per_image", None)
        row["nms0p5"].pop("per_image", None)
    for row in serializable["point_visual_reentry_crop384"].values():
        row["raw"].pop("per_image", None)
        row["nms0p5"].pop("per_image", None)
    for row in serializable["point_visual_reentry_crop384_nearest"].values():
        row["raw"].pop("per_image", None)
        row["nms0p5"].pop("per_image", None)
    (args.output / "results.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = [
        ("Matched standard LoRA, raw", standard_raw),
        ("Matched standard LoRA, NMS@0.5", standard_nms),
    ]
    if "standard_lora_same_protocol_nms0p5" in result:
        rows.append(
            (
                "Matched standard LoRA, SPWR prompts + NMS@0.5",
                result["standard_lora_same_protocol_nms0p5"],
            )
        )
    if "standard_lora_same_protocol_crop384_nms0p5" in result:
        rows.append(
            (
                "Matched standard LoRA, SPWR prompts + crop384 + NMS@0.5",
                result["standard_lora_same_protocol_crop384_nms0p5"],
            )
        )
    if "standard_refiner_spwr_points_crop384_nms0p5" in result:
        rows.append(
            (
                "Standard LoRA witness refiner, SPWR points + crop384 + NMS@0.5",
                result["standard_refiner_spwr_points_crop384_nms0p5"],
            )
        )
    for seed in args.seeds:
        key = f"standard_direct_refiner_spwr_seed{seed}_points_crop384_nms0p5"
        if key in result:
            rows.append(
                (
                    f"Standard LoRA native detector, SPWR seed {seed} points + crop384 + NMS@0.5",
                    result[key],
                )
            )
        nearest_key = (
            f"standard_direct_refiner_spwr_seed{seed}_points_crop384_nearest_nms0p5"
        )
        if nearest_key in result:
            rows.append(
                (
                    f"Standard LoRA native detector, SPWR seed {seed} points + crop384 + nearest + NMS@0.5",
                    result[nearest_key],
                )
            )
    rows.extend(
        (f"SPWR seed {seed}, NMS@0.5", result["spwr"][str(seed)]["nms0p5"])
        for seed in args.seeds
    )
    rows.extend(
        (
            f"Point visual re-entry crop384 seed {seed}, NMS@0.5",
            result["point_visual_reentry_crop384"][str(seed)]["nms0p5"],
        )
        for seed in args.seeds
        if str(seed) in result["point_visual_reentry_crop384"]
    )
    rows.extend(
        (
            f"Point visual re-entry crop384 + nearest seed {seed}, NMS@0.5",
            result["point_visual_reentry_crop384_nearest"][str(seed)]["nms0p5"],
        )
        for seed in args.seeds
        if str(seed) in result["point_visual_reentry_crop384_nearest"]
    )
    lines = [
        "# Self-Conditioned Point-Witness Refinement: Matched 4K Gate",
        "",
        "| Model | TP | FP | FN | Precision (%) | Recall (%) | F1@0.5 (%) | AP50 (%) | Boxes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in rows:
        lines.append(
            f"| {label} | {row['tp']:,} | {row['fp']:,} | {row['fn']:,} | "
            f"{100*row['precision']:.2f} | {100*row['recall']:.2f} | "
            f"{100*row['f1']:.2f} | {100*row['ap50']:.2f} | {row['predictions']:,} |"
        )
    lines.extend(["", f"**Preregistered gate passed:** {result['decision']['passes']}.", ""])
    for seed in args.seeds:
        treatment = result["spwr"][str(seed)]
        bootstrap = treatment["paired_bootstrap_vs_standard_nms0p5"]
        randomization = treatment["paired_randomization_vs_standard_nms0p5"]
        lines.append(
            f"- Seed {seed}: point TP={treatment['point_scaffold']['tp']:,}; "
            f"delta F1={100*bootstrap['mean_delta_f1']:.2f} pp, "
            f"95% CI=[{100*bootstrap['ci95_delta_f1'][0]:.2f}, "
            f"{100*bootstrap['ci95_delta_f1'][1]:.2f}], "
            f"paired one-sided p={randomization['one_sided_p_value']:.5f}."
        )
    if result["point_visual_reentry_crop384"]:
        lines.extend(
            [
                "",
                "## Post-hoc Point-Conditioned Visual Re-entry Mechanism Test",
                "",
                "The 384-pixel crop was selected after inspecting the seed-20260827 full-image failure, "
                "so this is not part of the preregistered SPWR gate.",
            ]
        )
        for seed in args.seeds:
            if str(seed) not in result["point_visual_reentry_crop384"]:
                continue
            treatment = result["point_visual_reentry_crop384"][str(seed)]
            bootstrap = treatment["paired_bootstrap_vs_standard_nms0p5"]
            randomization = treatment["paired_randomization_vs_standard_nms0p5"]
            lines.append(
                f"- Seed {seed}: NMS F1={100*treatment['nms0p5']['f1']:.2f}%, "
                f"TP={treatment['nms0p5']['tp']:,}, "
                f"delta F1={100*bootstrap['mean_delta_f1']:.2f} pp, "
                f"95% CI=[{100*bootstrap['ci95_delta_f1'][0]:.2f}, "
                f"{100*bootstrap['ci95_delta_f1'][1]:.2f}], "
                f"paired one-sided p={randomization['one_sided_p_value']:.5f}."
            )
        if result["point_visual_reentry_crop384_nearest"]:
            lines.extend(
                [
                    "",
                    "### Non-Oracle Nearest-Box Selection Stress Test",
                    "",
                    "The saved crop responses are re-scored by selecting the target-class "
                    "box whose center is nearest to the predicted point. Containment is not "
                    "required and no GT is consulted.",
                ]
            )
            for seed in args.seeds:
                treatment = result["point_visual_reentry_crop384_nearest"].get(str(seed))
                if treatment is None:
                    continue
                bootstrap = treatment["paired_bootstrap_vs_standard_nms0p5"]
                randomization = treatment[
                    "paired_randomization_vs_standard_nms0p5"
                ]
                lines.append(
                    f"- Seed {seed}: NMS F1={100*treatment['nms0p5']['f1']:.2f}%, "
                    f"TP={treatment['nms0p5']['tp']:,}, "
                    f"delta F1={100*bootstrap['mean_delta_f1']:.2f} pp, "
                    f"95% CI=[{100*bootstrap['ci95_delta_f1'][0]:.2f}, "
                    f"{100*bootstrap['ci95_delta_f1'][1]:.2f}], "
                    f"paired one-sided p={randomization['one_sided_p_value']:.5f}."
                )
        scale_rows = result["point_visual_reentry_scale_ablation_seed20260827"]
        if scale_rows:
            lines.extend(
                [
                    "",
                    "### Fixed Point-Centered Crop Scale Ablation (Seed 20260827)",
                    "",
                    "| Crop side (px) | TP | FP | Precision (%) | Recall (%) | F1@0.5 (%) | AP50 (%) | Boxes |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for crop_size in sorted(scale_rows, key=int):
                row = scale_rows[crop_size]["nms0p5"]
                lines.append(
                    f"| {crop_size} | {row['tp']:,} | {row['fp']:,} | "
                    f"{100*row['precision']:.2f} | {100*row['recall']:.2f} | "
                    f"{100*row['f1']:.2f} | {100*row['ap50']:.2f} | "
                    f"{row['predictions']:,} |"
                )
        fixed_rows = result["fixed_square_point_controls"]
        if fixed_rows:
            lines.extend(
                [
                    "",
                    "### Fixed-Square Boundary Control",
                    "",
                ]
            )
            for seed in args.seeds:
                controls = fixed_rows[str(seed)]
                best_side, best = max(controls.items(), key=lambda item: item[1]["f1"])
                lines.append(
                    f"- Seed {seed}: best fixed square={best_side}px, "
                    f"F1={100*best['f1']:.2f}%, TP={best['tp']:,}, FP={best['fp']:,}."
                )
        lines.extend(
            [
                "",
                "### Class and Density Diagnostics",
                "",
                "| Method | Macro-F1 (%) | Macro recall (%) | Predicted classes | Sparse F1 (%) | Medium F1 (%) | Dense F1 (%) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        diagnostic_order = [
            ("Matched standard LoRA", "standard_lora_nms0p5"),
            *(
                (f"Point visual re-entry seed {seed}", f"point_visual_reentry_crop384_seed{seed}")
                for seed in args.seeds
            ),
        ]
        for label, key in diagnostic_order:
            if key not in result["diagnostic_slices"]:
                continue
            row = result["diagnostic_slices"][key]
            lines.append(
                f"| {label} | {100*row['macro_f1']:.2f} | "
                f"{100*row['macro_recall']:.2f} | "
                f"{row['predicted_class_count']}/{row['class_count']} | "
                f"{100*row['density']['sparse_lt20']['f1']:.2f} | "
                f"{100*row['density']['medium_20_99']['f1']:.2f} | "
                f"{100*row['density']['dense_ge100']['f1']:.2f} |"
            )
        lines.extend(
            [
                "",
                "## Same-Pipeline Training-Control Gate",
                "",
                "This stricter control gives the matched standard LoRA its own predicted "
                "points and the identical 384-pixel visual re-entry pipeline.",
                f"**Gate passed:** {result['same_protocol_visual_reentry_decision']['passes']} "
                f"({result['same_protocol_visual_reentry_decision']['status']}).",
            ]
        )
        if same_protocol_crop_control is not None:
            for seed in args.seeds:
                treatment = result["point_visual_reentry_crop384"][str(seed)]
                bootstrap = treatment["paired_bootstrap_vs_same_protocol_crop384"]
                randomization = treatment[
                    "paired_randomization_vs_same_protocol_crop384"
                ]
                lines.append(
                    f"- Seed {seed}: delta F1={100*bootstrap['mean_delta_f1']:.2f} pp, "
                    f"95% CI=[{100*bootstrap['ci95_delta_f1'][0]:.2f}, "
                    f"{100*bootstrap['ci95_delta_f1'][1]:.2f}], "
                    f"paired one-sided p={randomization['one_sided_p_value']:.5f}."
                )
        lines.extend(
            [
                "",
                "## Fixed-Point Refiner Gate",
                "",
                result["fixed_point_refiner_decision"]["comparison"],
                f"**Gate passed:** {result['fixed_point_refiner_decision']['passes']} "
                f"({result['fixed_point_refiner_decision']['status']}).",
            ]
        )
        for seed in args.seeds:
            direct_control = result.get(
                f"standard_direct_refiner_spwr_seed{seed}_points_crop384_nms0p5"
            )
            if direct_control is None:
                continue
            treatment = result["point_visual_reentry_crop384"][str(seed)]
            bootstrap = treatment[
                "paired_bootstrap_vs_standard_direct_same_points_crop384"
            ]
            randomization = treatment[
                "paired_randomization_vs_standard_direct_same_points_crop384"
            ]
            lines.append(
                f"- Seed {seed}: delta F1={100*bootstrap['mean_delta_f1']:.2f} pp, "
                f"95% CI=[{100*bootstrap['ci95_delta_f1'][0]:.2f}, "
                f"{100*bootstrap['ci95_delta_f1'][1]:.2f}], "
                f"paired one-sided p={randomization['one_sided_p_value']:.5f}."
            )
        lines.extend(
            [
                "",
                "## Nearest-Point Refiner Gate",
                "",
                result["nearest_point_refiner_decision"]["comparison"],
                f"**Gate passed:** {result['nearest_point_refiner_decision']['passes']} "
                f"({result['nearest_point_refiner_decision']['status']}).",
            ]
        )
        for seed in args.seeds:
            treatment = result["point_visual_reentry_crop384_nearest"].get(str(seed))
            control = result.get(
                f"standard_direct_refiner_spwr_seed{seed}_points_crop384_nearest_nms0p5"
            )
            if treatment is None or control is None:
                continue
            bootstrap = treatment[
                "paired_bootstrap_vs_standard_direct_same_points_crop384_nearest"
            ]
            randomization = treatment[
                "paired_randomization_vs_standard_direct_same_points_crop384_nearest"
            ]
            lines.append(
                f"- Seed {seed}: SPWR F1={100*treatment['nms0p5']['f1']:.2f}% "
                f"versus standard-direct F1={100*control['f1']:.2f}%; "
                f"delta F1={100*bootstrap['mean_delta_f1']:.2f} pp, "
                f"95% CI=[{100*bootstrap['ci95_delta_f1'][0]:.2f}, "
                f"{100*bootstrap['ci95_delta_f1'][1]:.2f}], "
                f"paired one-sided p={randomization['one_sided_p_value']:.5f}."
            )
        lines.extend(
            [
                "",
                "## Factorized Point-Discovery / Box-Refinement Gate",
                "",
                result["factorized_point_reentry_decision"]["comparison"],
                f"**Gate passed:** {result['factorized_point_reentry_decision']['passes']} "
                f"({result['factorized_point_reentry_decision']['status']}).",
            ]
        )
        for seed in args.seeds:
            gate = result["factorized_point_reentry_decision"]["seed_gates"].get(
                str(seed)
            )
            if gate is None:
                continue
            metrics = gate["metrics"]
            bootstrap = gate["paired_bootstrap_vs_standard_nms0p5"]
            randomization = gate["paired_randomization_vs_standard_nms0p5"]
            lines.append(
                f"- Seed {seed}: factorized F1={100*metrics['f1']:.2f}% "
                f"versus full-image standard LoRA={100*standard_nms['f1']:.2f}%; "
                f"delta F1={100*bootstrap['mean_delta_f1']:.2f} pp, "
                f"95% CI=[{100*bootstrap['ci95_delta_f1'][0]:.2f}, "
                f"{100*bootstrap['ci95_delta_f1'][1]:.2f}], "
                f"paired one-sided p={randomization['one_sided_p_value']:.5f}."
            )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
