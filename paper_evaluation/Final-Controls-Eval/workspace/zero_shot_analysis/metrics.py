"""Metrics for zero-shot EO grounding evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .geometry import dense_bucket, greedy_match, hbox_iou, size_bucket
from .schema import EvalSample, PredBox, PredictionRecord


IOU_THRESHOLDS = [round(value / 100.0, 2) for value in range(50, 100, 5)]


def sample_key(sample: EvalSample) -> Tuple[str, str, str]:
    return sample.dataset, sample.image_id, sample.query


def prediction_key(prediction: PredictionRecord) -> Tuple[Optional[str], str, str]:
    return prediction.dataset, prediction.image_id, prediction.query


class PredictionIndex:
    def __init__(self, predictions: Sequence[PredictionRecord]) -> None:
        self.exact: Dict[Tuple[Optional[str], str, str], PredictionRecord] = {}
        self.fallback: Dict[Tuple[str, str], PredictionRecord] = {}
        for prediction in predictions:
            self.exact[prediction_key(prediction)] = prediction
            self.fallback[(prediction.image_id, prediction.query)] = prediction

    def get(self, sample: EvalSample) -> Optional[PredictionRecord]:
        return self.exact.get((sample.dataset, sample.image_id, sample.query)) or self.fallback.get((sample.image_id, sample.query))


def _prediction_boxes_for_sample(prediction: Optional[PredictionRecord], sample: EvalSample) -> List[PredBox]:
    if prediction is None:
        return []
    boxes = []
    for box in prediction.pred_hboxes:
        if box.label and box.label != sample.target_class:
            continue
        boxes.append(box)
    return boxes


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _ap_from_pr(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    if not recalls:
        return 0.0
    ap = 0.0
    for recall_threshold in [idx / 100.0 for idx in range(101)]:
        eligible = [precision for recall, precision in zip(recalls, precisions) if recall >= recall_threshold]
        ap += max(eligible) if eligible else 0.0
    return ap / 101.0


def average_precision(
    samples: Sequence[EvalSample],
    pred_index: PredictionIndex,
    iou_threshold: float,
    gt_filter: Optional[Callable[[EvalSample, Sequence[float]], bool]] = None,
) -> float:
    class_names = sorted({sample.target_class for sample in samples if not sample.is_negative})
    class_aps: List[float] = []

    for class_name in class_names:
        gt_by_sample: Dict[Tuple[str, str, str], List[List[float]]] = {}
        detections: List[Tuple[float, Tuple[str, str, str], List[float]]] = []
        npos = 0

        for sample in samples:
            if sample.is_negative or sample.target_class != class_name:
                continue
            key = sample_key(sample)
            gt_boxes = [box for box in sample.gt_hboxes if gt_filter is None or gt_filter(sample, box)]
            gt_by_sample[key] = gt_boxes
            npos += len(gt_boxes)
            prediction = pred_index.get(sample)
            for pred_box in _prediction_boxes_for_sample(prediction, sample):
                detections.append((pred_box.score, key, pred_box.hbox))

        if npos == 0:
            continue

        detections.sort(key=lambda item: item[0], reverse=True)
        matched: Dict[Tuple[str, str, str], set[int]] = defaultdict(set)
        tps: List[int] = []
        fps: List[int] = []

        for _score, key, pred_box in detections:
            best_gt = None
            best_iou = 0.0
            for gt_idx, gt_box in enumerate(gt_by_sample.get(key, [])):
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
        class_aps.append(_ap_from_pr(recalls, precisions))

    return mean(class_aps) if class_aps else 0.0


def accuracy_summary(samples: Sequence[EvalSample], pred_index: PredictionIndex) -> Dict[str, Any]:
    positive_samples = [sample for sample in samples if not sample.is_negative]
    threshold_metrics: Dict[str, Dict[str, float]] = {}
    for threshold in [0.5, 0.75, 0.95]:
        tp = fp = fn = 0
        for sample in positive_samples:
            prediction = pred_index.get(sample)
            pred_boxes = _prediction_boxes_for_sample(prediction, sample)
            match = greedy_match(sample.gt_hboxes, [box.hbox for box in pred_boxes], threshold, [box.score for box in pred_boxes])
            tp += match.tp
            fp += match.fp
            fn += match.fn
        threshold_metrics[f"f1_iou_{threshold:.2f}"] = {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}

    return {
        "ap50": average_precision(positive_samples, pred_index, 0.50),
        "ap75": average_precision(positive_samples, pred_index, 0.75),
        "map_50_95": mean(average_precision(positive_samples, pred_index, threshold) for threshold in IOU_THRESHOLDS),
        **threshold_metrics,
    }


def evaluate_grounding(
    samples: Sequence[EvalSample],
    predictions: Sequence[PredictionRecord],
    small_max_area_ratio: float = 0.001,
    medium_max_area_ratio: float = 0.01,
    sparse_max: int = 5,
    medium_dense_max: int = 20,
) -> Dict[str, Any]:
    pred_index = PredictionIndex(predictions)
    positive_samples = [sample for sample in samples if not sample.is_negative]
    negative_samples = [sample for sample in samples if sample.is_negative]
    overall_accuracy = accuracy_summary(samples, pred_index)
    accuracy_by_query_type = {
        query_type: accuracy_summary([sample for sample in positive_samples if sample.query_type == query_type], pred_index)
        for query_type in sorted({sample.query_type for sample in positive_samples})
    }
    accuracy_by_class = {
        class_name: accuracy_summary([sample for sample in positive_samples if sample.target_class == class_name], pred_index)
        for class_name in sorted({sample.target_class for sample in positive_samples})
    }

    negative_correct = 0
    negative_false_positive_boxes = 0
    for sample in negative_samples:
        prediction = pred_index.get(sample)
        pred_boxes = _prediction_boxes_for_sample(prediction, sample)
        has_no_boxes = len(pred_boxes) == 0
        predicted_negative = bool(prediction and prediction.is_negative)
        if has_no_boxes or predicted_negative:
            negative_correct += 1
        negative_false_positive_boxes += len(pred_boxes)

    size_stats: Dict[str, Dict[str, Any]] = {
        "small": {"instances": 0, "tp": 0, "missed": 0},
        "medium": {"instances": 0, "tp": 0, "missed": 0},
        "large": {"instances": 0, "tp": 0, "missed": 0},
    }
    for sample in positive_samples:
        prediction = pred_index.get(sample)
        pred_boxes = _prediction_boxes_for_sample(prediction, sample)
        match = greedy_match(sample.gt_hboxes, [box.hbox for box in pred_boxes], 0.5, [box.score for box in pred_boxes])
        matched_gt = {gt_idx for gt_idx, _pred_idx in match.matched_pairs}
        for gt_idx, gt_box in enumerate(sample.gt_hboxes):
            bucket = size_bucket(gt_box, sample.image_width, sample.image_height, small_max_area_ratio, medium_max_area_ratio)
            size_stats[bucket]["instances"] += 1
            if gt_idx in matched_gt:
                size_stats[bucket]["tp"] += 1
            else:
                size_stats[bucket]["missed"] += 1
    for bucket, stats in size_stats.items():
        stats["recall"] = stats["tp"] / stats["instances"] if stats["instances"] else 0.0
        stats["ap50"] = average_precision(
            positive_samples,
            pred_index,
            0.50,
            gt_filter=lambda sample, box, target_bucket=bucket: size_bucket(
                box,
                sample.image_width,
                sample.image_height,
                small_max_area_ratio,
                medium_max_area_ratio,
            )
            == target_bucket,
        )

    dense_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"samples": 0, "gt_boxes": 0, "tp": 0, "fp": 0, "fn": 0})
    for sample in positive_samples:
        bucket = dense_bucket(len(sample.gt_hboxes), sparse_max, medium_dense_max)
        prediction = pred_index.get(sample)
        pred_boxes = _prediction_boxes_for_sample(prediction, sample)
        match = greedy_match(sample.gt_hboxes, [box.hbox for box in pred_boxes], 0.5, [box.score for box in pred_boxes])
        dense_stats[bucket]["samples"] += 1
        dense_stats[bucket]["gt_boxes"] += len(sample.gt_hboxes)
        dense_stats[bucket]["tp"] += match.tp
        dense_stats[bucket]["fp"] += match.fp
        dense_stats[bucket]["fn"] += match.fn
    for stats in dense_stats.values():
        stats.update(_prf(stats["tp"], stats["fp"], stats["fn"]))

    relation_samples = [sample for sample in positive_samples if sample.query_type == "relation"]
    relation_tp = relation_total = 0
    for sample in relation_samples:
        prediction = pred_index.get(sample)
        pred_boxes = _prediction_boxes_for_sample(prediction, sample)
        match = greedy_match(sample.gt_hboxes, [box.hbox for box in pred_boxes], 0.5, [box.score for box in pred_boxes])
        relation_tp += match.tp
        relation_total += len(sample.gt_hboxes)

    dataset_metrics = {}
    for dataset in sorted({sample.dataset for sample in samples}):
        subset = [sample for sample in samples if sample.dataset == dataset]
        tp = fp = fn = 0
        for sample in subset:
            if sample.is_negative:
                continue
            prediction = pred_index.get(sample)
            pred_boxes = _prediction_boxes_for_sample(prediction, sample)
            match = greedy_match(sample.gt_hboxes, [box.hbox for box in pred_boxes], 0.5, [box.score for box in pred_boxes])
            tp += match.tp
            fp += match.fp
            fn += match.fn
        dataset_metrics[dataset] = {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}

    query_type_metrics = {}
    for query_type in sorted({sample.query_type for sample in samples}):
        subset = [sample for sample in samples if sample.query_type == query_type and not sample.is_negative]
        tp = fp = fn = 0
        for sample in subset:
            prediction = pred_index.get(sample)
            pred_boxes = _prediction_boxes_for_sample(prediction, sample)
            match = greedy_match(sample.gt_hboxes, [box.hbox for box in pred_boxes], 0.5, [box.score for box in pred_boxes])
            tp += match.tp
            fp += match.fp
            fn += match.fn
        query_type_metrics[query_type] = {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}

    return {
        "samples": {
            "total": len(samples),
            "positive": len(positive_samples),
            "negative": len(negative_samples),
        },
        "accuracy": overall_accuracy,
        "accuracy_by_query_type": accuracy_by_query_type,
        "accuracy_by_class": accuracy_by_class,
        "negative_queries": {
            "total": len(negative_samples),
            "correct": negative_correct,
            "negative_accuracy": negative_correct / len(negative_samples) if negative_samples else 0.0,
            "false_positive_boxes": negative_false_positive_boxes,
            "false_positives_per_negative_query": negative_false_positive_boxes / len(negative_samples) if negative_samples else 0.0,
        },
        "small_object": size_stats,
        "dense_object": dict(dense_stats),
        "relation_grounding": {
            "samples": len(relation_samples),
            "targets": relation_total,
            "correct_at_iou_0_5": relation_tp,
            "relation_correctness": relation_tp / relation_total if relation_total else 0.0,
        },
        "by_dataset": dataset_metrics,
        "by_query_type": query_type_metrics,
        "speed": summarize_speed(predictions),
    }


def summarize_speed(predictions: Sequence[PredictionRecord]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], List[PredictionRecord]] = defaultdict(list)
    for prediction in predictions:
        model = prediction.model or "unknown_model"
        mode = prediction.decode_mode or "unknown_mode"
        groups[(model, mode)].append(prediction)

    summary = {}
    for (model, mode), group in groups.items():
        latencies = [item.latency_ms for item in group if item.latency_ms is not None]
        total_boxes = sum(len(item.pred_hboxes) for item in group)
        total_latency_s = sum(latencies) / 1000.0 if latencies else 0.0
        token_counts = [item.tokens_generated / max(1, len(item.pred_hboxes)) for item in group if item.tokens_generated is not None]
        gpu_values = [item.gpu_memory_mb for item in group if item.gpu_memory_mb is not None]
        key = f"{model}:{mode}"
        summary[key] = {
            "records": len(group),
            "boxes": total_boxes,
            "mean_latency_ms": mean(latencies) if latencies else None,
            "boxes_per_second": total_boxes / total_latency_s if total_latency_s > 0 else None,
            "tokens_per_object": mean(token_counts) if token_counts else None,
            "max_gpu_memory_mb": max(gpu_values) if gpu_values else None,
        }
    return summary


def write_metrics(path: str | Path, metrics: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
