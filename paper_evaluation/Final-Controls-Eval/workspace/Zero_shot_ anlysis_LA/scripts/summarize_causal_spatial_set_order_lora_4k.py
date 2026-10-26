#!/usr/bin/env python3
"""Summarize multi-seed spatial set ordering against the frozen matched LoRA."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from zero_shot_analysis.geometry import greedy_match, hbox_iou  # noqa: E402

from scripts.prepare_causal_spatial_set_order_lora_4k import hilbert_index  # noqa: E402


SEEDS = (20260825, 20260826, 20260827)
CLASSES = (
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool", "container crane", "airport", "helipad",
)


def load_evaluator():
    path = PROJECT / "scripts/evaluate_locateanything_lae_style.py"
    spec = importlib.util.spec_from_file_location("lae_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def resolve_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one file for {pattern}, found {len(matches)}: {matches}")
    return matches[0]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def prediction_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["image_id"]))
        if key in result:
            raise RuntimeError(f"Duplicate full-image prediction row: {key}")
        result[key] = row
    return result


def per_image_counts(
    records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    indexed = prediction_index(predictions)
    rows = []
    for record in records:
        key = (str(record["dataset"]), str(record["image_id"]))
        prediction = indexed.get(key, {})
        tp = fp = fn = 0
        for class_name in CLASSES:
            gt = [item["hbox"] for item in record["gt"] if item["label"] == class_name]
            pred = [
                item["hbox"]
                for item in prediction.get("predictions", [])
                if item.get("label") == class_name
            ]
            match = greedy_match(gt, pred, 0.5, [1.0] * len(pred))
            tp += match.tp
            fp += match.fp
            fn += match.fn
        rows.append({"image_id": key[1], "tp": tp, "fp": fp, "fn": fn})
    return rows


def paired_bootstrap(
    treatment: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if [row["image_id"] for row in treatment] != [row["image_id"] for row in control]:
        raise RuntimeError("Paired image order mismatch")
    treatment_array = np.asarray([[r["tp"], r["fp"], r["fn"]] for r in treatment], dtype=np.int64)
    control_array = np.asarray([[r["tp"], r["fp"], r["fn"]] for r in control], dtype=np.int64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    n = len(treatment_array)
    for index in range(samples):
        selected = rng.integers(0, n, size=n)
        t = treatment_array[selected].sum(axis=0)
        c = control_array[selected].sum(axis=0)
        deltas[index] = prf(*map(int, t))["f1"] - prf(*map(int, c))["f1"]
    return {
        "samples": samples,
        "seed": seed,
        "mean_delta_f1": float(deltas.mean()),
        "ci95_delta_f1": [float(v) for v in np.quantile(deltas, [0.025, 0.975])],
        "probability_delta_positive": float((deltas > 0).mean()),
    }


def paired_randomization_test(
    treatment: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """One-sided paired test for a positive corpus-level F1 difference.

    Under the sharp null, the treatment/control assignment is exchangeable
    within each image. Each Monte Carlo draw independently swaps the two
    per-image (TP, FP, FN) triples, then recomputes corpus F1. This preserves
    image difficulty and the non-linear corpus-F1 definition.
    """

    if [row["image_id"] for row in treatment] != [row["image_id"] for row in control]:
        raise RuntimeError("Paired image order mismatch")
    treatment_array = np.asarray(
        [[row["tp"], row["fp"], row["fn"]] for row in treatment], dtype=np.int64
    )
    control_array = np.asarray(
        [[row["tp"], row["fp"], row["fn"]] for row in control], dtype=np.int64
    )
    observed = (
        prf(*map(int, treatment_array.sum(axis=0)))["f1"]
        - prf(*map(int, control_array.sum(axis=0)))["f1"]
    )
    rng = np.random.default_rng(seed)
    greater_or_equal = 0
    for _ in range(samples):
        swap = rng.integers(0, 2, size=len(treatment_array), dtype=np.int8).astype(bool)
        randomized_treatment = np.where(swap[:, None], control_array, treatment_array)
        randomized_control = np.where(swap[:, None], treatment_array, control_array)
        delta = (
            prf(*map(int, randomized_treatment.sum(axis=0)))["f1"]
            - prf(*map(int, randomized_control.sum(axis=0)))["f1"]
        )
        greater_or_equal += int(delta >= observed)
    return {
        "samples": samples,
        "seed": seed,
        "observed_delta_f1": float(observed),
        "one_sided_p_value": float((greater_or_equal + 1) / (samples + 1)),
    }


def generation_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eos = 0
    duplicates = 0
    overlapping_same_class = 0
    adjacent = reversals = 0
    distances: list[float] = []
    non_eos_predictions = 0
    total_predictions = 0
    for row in rows:
        output = str(row.get("output") or "")
        ended = output.endswith("<|im_end|>")
        eos += int(ended)
        predictions = row.get("predictions", [])
        total_predictions += len(predictions)
        if not ended:
            non_eos_predictions += len(predictions)
        keys = [(p.get("label"), tuple(p.get("hbox", []))) for p in predictions]
        duplicates += len(keys) - len(set(keys))
        earlier_by_class: dict[str, list[Sequence[float]]] = defaultdict(list)
        for item in predictions:
            label = str(item.get("label"))
            box = item.get("hbox") or []
            if len(box) != 4:
                continue
            overlapping_same_class += int(
                any(hbox_iou(box, earlier) >= 0.5 for earlier in earlier_by_class[label])
            )
            earlier_by_class[label].append(box)
        by_class: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for item in predictions:
            box = item.get("hbox") or []
            if len(box) != 4:
                continue
            by_class[str(item.get("label"))].append(
                ((float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2)
            )
        for points in by_class.values():
            for left, right in zip(points, points[1:]):
                left_h = hilbert_index(round(left[0] * 1000 / 1024), round(left[1] * 1000 / 1024))
                right_h = hilbert_index(round(right[0] * 1000 / 1024), round(right[1] * 1000 / 1024))
                adjacent += 1
                reversals += int(right_h < left_h)
                distances.append(math.hypot(right[0] - left[0], right[1] - left[1]))
    return {
        "rows": len(rows),
        "eos_rows": eos,
        "non_eos_rows": len(rows) - eos,
        "total_predictions": total_predictions,
        "non_eos_prediction_fraction": non_eos_predictions / total_predictions if total_predictions else 0.0,
        "exact_duplicate_predictions": duplicates,
        "overlapping_same_class_predictions_iou50": overlapping_same_class,
        "within_class_adjacent_pairs": adjacent,
        "hilbert_reversal_fraction": reversals / adjacent if adjacent else 0.0,
        "mean_adjacent_center_distance_pixels": statistics.mean(distances) if distances else None,
    }


def metric_row(
    metric_path: Path,
    prediction_root: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    at50 = metrics["la_metrics"]["f1_iou_0.50"]
    prediction_path = prediction_root / metric_path.name.replace("_metrics.json", ".jsonl")
    predictions = read_jsonl(prediction_path)
    return {
        "metric_path": str(metric_path.resolve()),
        "prediction_path": str(prediction_path.resolve()),
        "images": int(metrics["images"]),
        "gt": int(metrics["gt_instances"]),
        "predictions": int(metrics["predicted_boxes"]),
        "tp": int(at50["tp"]),
        "fp": int(at50["fp"]),
        "fn": int(at50["fn"]),
        **{key: float(at50[key]) for key in ("precision", "recall", "f1")},
        "ap50": float(metrics["accuracy"]["ap50"]),
        "map_50_95": float(metrics["accuracy"]["map_50_95"]),
        "protocol": metrics["protocol"],
        "generation": generation_diagnostics(predictions),
        "per_image": per_image_counts(records, predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-base", type=Path,
        default=Path("/data1/tmp_shubham/causal_spatial_set_order_lora_4k_v1"),
    )
    parser.add_argument("--metric-root", type=Path, default=PROJECT / "runs/lae_style_metrics")
    parser.add_argument("--prediction-root", type=Path, default=PROJECT / "runs/lae_style_predictions")
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/causal_spatial_set_order_lora_4k_v1")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--run-prefix", default="causal_spatial_order_v1")
    parser.add_argument("--treatment-label", default="Causal spatial set ordering")
    args = parser.parse_args()
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds:
        raise ValueError("At least one treatment seed is required")

    evaluator = load_evaluator()
    config = evaluator.read_json(PROJECT / "config/lae_benchmarks_balanced100.json")
    benchmark = config["benchmarks"]["DOTAv2"]
    specs = evaluator.benchmark_class_specs("DOTAv2", benchmark, "lae-metainfo", "raw_class_name")
    records = evaluator.load_lae_records("DOTAv2", benchmark, specs, "raw_class_name")
    if len(records) != 100 or sum(len(row["gt"]) for row in records) != 9045:
        raise RuntimeError("Frozen Balanced100 annotation changed")

    base_metric = resolve_one(
        args.metric_root,
        "*causal_pointrelay_4k_base_best_native1024_uncapped_seed0*metrics.json",
    )
    standard_metric = resolve_one(
        args.metric_root,
        "*causal_pointrelay_4k_standard_best_native1024_uncapped_seed0*metrics.json",
    )
    result: dict[str, Any] = {
        "schema_version": "causal-spatial-set-order-lora-4k-summary-v1",
        "evaluation_scope": "fixed DOTAv2 Balanced100 architecture-development diagnostic",
        "base": metric_row(base_metric, args.prediction_root, records),
        "standard_lora": metric_row(standard_metric, args.prediction_root, records),
        "spatial_order": {},
    }
    protocol = result["standard_lora"]["protocol"]
    for seed in seeds:
        root = args.train_base / f"seed{seed}"
        done = json.loads((root / "done.json").read_text(encoding="utf-8"))
        if not done.get("complete_one_pass") or int(done["record_exposures"]) != 16000:
            raise RuntimeError(f"Incomplete training seed {seed}")
        metric = resolve_one(
            args.metric_root,
            f"*{args.run_prefix}_seed{seed}_best_native1024_uncapped_seed0*metrics.json",
        )
        row = metric_row(metric, args.prediction_root, records)
        if row["protocol"] != protocol or row["images"] != 100 or row["gt"] != 9045:
            raise RuntimeError(f"Evaluation contract mismatch for seed {seed}")
        checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
        row["best_step"] = int(checkpoint["step"])
        row["best_validation_native_loss"] = float(checkpoint["validation"]["native_loss"])
        row["paired_bootstrap_vs_standard"] = paired_bootstrap(
            row["per_image"], result["standard_lora"]["per_image"],
            samples=args.bootstrap_samples, seed=seed,
        )
        row["paired_randomization_vs_standard"] = paired_randomization_test(
            row["per_image"], result["standard_lora"]["per_image"],
            samples=args.bootstrap_samples, seed=seed + 100_000,
        )
        result["spatial_order"][str(seed)] = row

    seed_rows = list(result["spatial_order"].values())
    f1_values = [row["f1"] for row in seed_rows]
    standard = result["standard_lora"]
    primary = result["spatial_order"][str(seeds[0])]
    seed_gates = {}
    for seed in seeds:
        row = result["spatial_order"][str(seed)]
        ci = row["paired_bootstrap_vs_standard"]["ci95_delta_f1"]
        p_value = row["paired_randomization_vs_standard"]["one_sided_p_value"]
        seed_gates[str(seed)] = {
            "delta_f1": row["f1"] - standard["f1"],
            "ci95_delta_f1": ci,
            "one_sided_p_value": p_value,
            "tp_increases": row["tp"] > standard["tp"],
            "predictions_within_2x_control": row["predictions"] <= 2 * standard["predictions"],
            "passes": (
                row["f1"] > standard["f1"]
                and ci[0] > 0
                and p_value < 0.05
                and row["tp"] > standard["tp"]
                and row["predictions"] <= 2 * standard["predictions"]
            ),
        }
    result["decision"] = {
        "preregistered_gate": (
            "every independent treatment seed exceeds matched LoRA; every paired 95% image-bootstrap "
            "CI is above zero; every one-sided paired randomization p-value is below 0.05; every seed "
            "increases TP and emits no more than 2x the control predictions"
        ),
        "mean_f1": statistics.mean(f1_values),
        "sample_std_f1": statistics.stdev(f1_values) if len(f1_values) > 1 else None,
        "primary_delta_f1": primary["f1"] - standard["f1"],
        "primary_ci95_delta_f1": seed_gates[str(seeds[0])]["ci95_delta_f1"],
        "seed_gates": seed_gates,
        "passes": all(gate["passes"] for gate in seed_gates.values()),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(result))
    for key in ("base", "standard_lora"):
        serializable[key].pop("per_image", None)
    for row in serializable["spatial_order"].values():
        row.pop("per_image", None)
    (args.output / "results.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    table = [("Base LA", result["base"]), ("Matched standard LoRA", standard)] + [
        (f"{args.treatment_label} seed {seed}", result["spatial_order"][str(seed)]) for seed in seeds
    ]
    lines = [
        f"# {args.treatment_label}: Matched 4k Detection Gate", "",
        "All rows use 100 DOTAv2 images, 9,045 GT HBBs, one 18-class prompt, hybrid decoding, "
        "1,024 merged image-token limit, 2,048 generation tokens, and uncapped class-aware one-to-one scoring.",
        "",
        "| Model | TP | FP | Precision (%) | Recall (%) | F1@0.5 (%) | AP50 (%) | Predictions | EOS | Duplicates | Hilbert reversal (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in table:
        gen = row["generation"]
        lines.append(
            f"| {name} | {row['tp']:,} | {row['fp']:,} | {100*row['precision']:.2f} | "
            f"{100*row['recall']:.2f} | {100*row['f1']:.2f} | {100*row['ap50']:.2f} | "
            f"{row['predictions']:,} | {gen['eos_rows']}/100 | {gen['exact_duplicate_predictions']:,} | "
            f"{100*gen['hilbert_reversal_fraction']:.2f} |"
        )
    decision = result["decision"]
    lines.extend([
        "", "## Statistical gate", "",
        (
            f"- {len(seeds)}-seed F1: {100*decision['mean_f1']:.2f}"
            + (
                f" +/- {100*decision['sample_std_f1']:.2f} sample SD."
                if decision["sample_std_f1"] is not None else "."
            )
        ),
        f"- Primary minus matched LoRA: {100*decision['primary_delta_f1']:+.2f} F1 points.",
        f"- Paired image-bootstrap 95% CI: [{100*decision['primary_ci95_delta_f1'][0]:+.2f}, "
        f"{100*decision['primary_ci95_delta_f1'][1]:+.2f}] F1 points.",
        "- Per-seed paired randomization p-values: "
        + ", ".join(
            f"{seed}={decision['seed_gates'][str(seed)]['one_sided_p_value']:.4f}"
            for seed in seeds
        )
        + ".",
        f"- Preregistered pass: **{decision['passes']}**.", "",
        "Balanced100 is an architecture-development set. A pass licenses an untouched confirmation run; "
        "it is not itself a benchmark or ICLR claim.",
    ])
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
