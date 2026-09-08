"""HBB/point grammar, coordinate conversion, NMS, and diagnostic metrics."""

from __future__ import annotations
import re
from collections import defaultdict
from typing import Any, Mapping, Sequence

REF_POINT_OR_BOX = re.compile(
    "<ref>\\s*(.*?)\\s*</ref>|<box>\\s*<(\\d+)>\\s*<(\\d+)>(?:\\s*<(\\d+)>\\s*<(\\d+)>)?\\s*</box>",
    re.IGNORECASE | re.DOTALL,
)


def canonical_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def hbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    (ix1, iy1) = (max(left[0], right[0]), max(left[1], right[1]))
    (ix2, iy2) = (min(left[2], right[2]), min(left[3], right[3]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def detection_counts(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    matched: set[int] = set()
    tp = fp = 0
    for prediction in predictions:
        candidates = []
        label = canonical_label(prediction.get("label"))
        for index, target in enumerate(ground_truth):
            if index in matched or canonical_label(target.get("label")) != label:
                continue
            overlap = hbox_iou(prediction["hbox"], target["hbox"])
            if overlap >= iou_threshold:
                candidates.append((overlap, index))
        if candidates:
            (_, index) = max(candidates)
            matched.add(index)
            tp += 1
        else:
            fp += 1
    return (tp, fp, len(ground_truth) - len(matched))


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
