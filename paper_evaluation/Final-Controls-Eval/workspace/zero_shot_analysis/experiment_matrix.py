"""Experiment matrix and real-run preflight checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


REQUIRED_EXPERIMENT_IDS = {
    "zero_shot_category_grounding",
    "strict_novel_class_grounding",
    "cross_dataset_transfer",
    "dense_object_analysis",
    "small_object_analysis",
    "negative_query_analysis",
    "relation_grounding",
    "speed_comparison",
    "geotoken_ablation",
}


def load_matrix(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _exists(path: str | Path, root: Path) -> bool:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.exists()


def _resolve_template(template: str, model_id: str, experiment_id: str, root: Path) -> Path:
    resolved = template.format(model_id=model_id, experiment_id=experiment_id)
    path = Path(resolved)
    if not path.is_absolute():
        path = root / path
    return path


def summarize_matrix(matrix: Mapping[str, Any]) -> Dict[str, Any]:
    datasets = matrix.get("datasets", {})
    models = matrix.get("models", {})
    experiments = matrix.get("experiments", [])
    ablations = matrix.get("ablations", {})
    experiment_ids = {experiment.get("id") for experiment in experiments}
    return {
        "datasets": sorted(datasets),
        "models": sorted(models),
        "experiments": sorted(experiment_ids),
        "ablations": sorted(ablations),
        "missing_required_experiments": sorted(REQUIRED_EXPERIMENT_IDS - experiment_ids),
    }


def preflight_matrix(matrix: Mapping[str, Any], root: str | Path = ".") -> Dict[str, Any]:
    root_path = Path(root)
    datasets: Mapping[str, Any] = matrix.get("datasets", {})
    models: Mapping[str, Any] = matrix.get("models", {})
    experiments: Iterable[Mapping[str, Any]] = matrix.get("experiments", [])
    ablations: Mapping[str, Any] = matrix.get("ablations", {})
    prediction_template = str(matrix.get("prediction_template", "runs/real_zero_shot_predictions/{model_id}/{experiment_id}.jsonl"))

    missing: List[Dict[str, str]] = []
    warnings: List[str] = []
    expected_predictions: List[str] = []

    summary = summarize_matrix(matrix)
    for experiment_id in summary["missing_required_experiments"]:
        missing.append({"kind": "experiment_definition", "id": experiment_id, "path": "<matrix>"})

    for dataset_id, dataset in datasets.items():
        for field in ["annotation_jsonl", "image_root"]:
            value = dataset.get(field)
            if not value:
                missing.append({"kind": "dataset_field", "id": dataset_id, "path": field})
            elif not _exists(value, root_path):
                missing.append({"kind": f"dataset_{field}", "id": dataset_id, "path": str(value)})

    for model_id, model in models.items():
        checkpoint = model.get("checkpoint")
        if not checkpoint:
            missing.append({"kind": "model_checkpoint_field", "id": model_id, "path": "checkpoint"})
        elif not _exists(checkpoint, root_path):
            missing.append({"kind": "model_checkpoint", "id": model_id, "path": str(checkpoint)})

    for ablation_id, ablation in ablations.items():
        checkpoint = ablation.get("checkpoint")
        if not checkpoint:
            missing.append({"kind": "ablation_checkpoint_field", "id": ablation_id, "path": "checkpoint"})
        elif not _exists(checkpoint, root_path):
            missing.append({"kind": "ablation_checkpoint", "id": ablation_id, "path": str(checkpoint)})

    for experiment in experiments:
        experiment_id = str(experiment.get("id"))
        experiment_datasets = experiment.get("datasets", [])
        experiment_models = experiment.get("models", [])
        if not experiment_id or experiment_id == "None":
            missing.append({"kind": "experiment_id", "id": "<missing>", "path": "<matrix>"})
            continue

        for dataset_id in experiment_datasets:
            if dataset_id not in datasets:
                missing.append({"kind": "experiment_dataset_reference", "id": experiment_id, "path": str(dataset_id)})

        for model_id in experiment_models:
            if model_id not in models and model_id not in ablations:
                missing.append({"kind": "experiment_model_reference", "id": experiment_id, "path": str(model_id)})
                continue
            prediction_path = _resolve_template(prediction_template, str(model_id), experiment_id, root_path)
            expected_predictions.append(str(prediction_path))
            if not prediction_path.exists():
                missing.append({"kind": "prediction_jsonl", "id": f"{model_id}:{experiment_id}", "path": str(prediction_path)})

        if experiment.get("requires_training_manifest"):
            manifest = experiment.get("training_manifest")
            if not manifest:
                missing.append({"kind": "training_manifest_field", "id": experiment_id, "path": "training_manifest"})
            elif not _exists(str(manifest), root_path):
                missing.append({"kind": "training_manifest", "id": experiment_id, "path": str(manifest)})

    if matrix.get("mode") != "real":
        warnings.append("matrix mode is not 'real'")

    return {
        "ready": not missing,
        "summary": summary,
        "prediction_template": prediction_template,
        "expected_prediction_files": expected_predictions,
        "missing": missing,
        "warnings": warnings,
    }


def write_preflight(path: str | Path, result: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
