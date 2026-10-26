"""Command line interface for the zero-shot analysis toolkit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset_builder import build_samples_from_records, load_annotation_records, load_config
from .experiment_matrix import load_matrix, preflight_matrix, write_preflight
from .metrics import evaluate_grounding, summarize_speed, write_metrics
from .schema import load_predictions, load_samples, write_jsonl


def _write_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_samples_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    records = load_annotation_records(args.annotations)
    samples = build_samples_from_records(records, config)
    write_jsonl(args.output, [sample.to_dict() for sample in samples])
    print(f"Wrote {len(samples)} evaluation samples to {args.output}")


def evaluate_command(args: argparse.Namespace) -> None:
    samples = load_samples(args.samples)
    predictions = load_predictions(args.predictions)
    metrics = evaluate_grounding(
        samples,
        predictions,
        small_max_area_ratio=args.small_max_area_ratio,
        medium_max_area_ratio=args.medium_max_area_ratio,
        sparse_max=args.sparse_max,
        medium_dense_max=args.medium_dense_max,
    )
    if args.output:
        write_metrics(args.output, metrics)
        print(f"Wrote metrics to {args.output}")
    else:
        print(json.dumps(metrics, indent=2, sort_keys=True))


def speed_command(args: argparse.Namespace) -> None:
    predictions = load_predictions(args.predictions)
    summary = summarize_speed(predictions)
    if args.output:
        _write_json(args.output, summary)
        print(f"Wrote speed summary to {args.output}")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


def validate_command(args: argparse.Namespace) -> None:
    sample_count = len(load_samples(args.samples))
    prediction_count = len(load_predictions(args.predictions)) if args.predictions else 0
    if args.predictions:
        print(f"Validated {sample_count} samples and {prediction_count} predictions")
    else:
        print(f"Validated {sample_count} samples")


def preflight_command(args: argparse.Namespace) -> None:
    matrix = load_matrix(args.matrix)
    result = preflight_matrix(matrix, args.root)
    if args.output:
        write_preflight(args.output, result)
        print(f"Wrote preflight report to {args.output}")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ready"] and args.strict:
        raise SystemExit(2)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zero-shot EO grounding analysis toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-samples", help="Convert unified annotations into evaluation JSONL")
    build.add_argument("--annotations", required=True, help="Input annotation JSONL")
    build.add_argument("--config", help="JSON config with target classes, relations, and generation options")
    build.add_argument("--output", required=True, help="Output evaluation JSONL")
    build.set_defaults(func=build_samples_command)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate model predictions against evaluation JSONL")
    evaluate.add_argument("--samples", required=True, help="Evaluation sample JSONL")
    evaluate.add_argument("--predictions", required=True, help="Prediction JSONL")
    evaluate.add_argument("--output", help="Output metrics JSON")
    evaluate.add_argument("--small-max-area-ratio", type=float, default=0.001)
    evaluate.add_argument("--medium-max-area-ratio", type=float, default=0.01)
    evaluate.add_argument("--sparse-max", type=int, default=5)
    evaluate.add_argument("--medium-dense-max", type=int, default=20)
    evaluate.set_defaults(func=evaluate_command)

    speed = subparsers.add_parser("speed-summary", help="Summarize decoding speed from prediction JSONL")
    speed.add_argument("--predictions", required=True, help="Prediction JSONL")
    speed.add_argument("--output", help="Output speed JSON")
    speed.set_defaults(func=speed_command)

    validate = subparsers.add_parser("validate", help="Validate JSONL files against the toolkit schema")
    validate.add_argument("--samples", required=True, help="Evaluation sample JSONL")
    validate.add_argument("--predictions", help="Prediction JSONL")
    validate.set_defaults(func=validate_command)

    preflight = subparsers.add_parser("preflight", help="Check whether real experiment assets are present")
    preflight.add_argument("--matrix", required=True, help="Experiment matrix JSON")
    preflight.add_argument("--root", default=".", help="Workspace root for relative paths")
    preflight.add_argument("--output", help="Output preflight JSON")
    preflight.add_argument("--strict", action="store_true", help="Exit nonzero when any required asset is missing")
    preflight.set_defaults(func=preflight_command)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
