"""Geometry helpers for horizontal-box EO grounding evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MatchResult:
    tp: int
    fp: int
    fn: int
    matched_pairs: List[Tuple[int, int]]


def normalize_hbox(box: Sequence[float]) -> List[float]:
    if len(box) != 4:
        raise ValueError(f"Expected hbox with 4 values, got {len(box)}")
    x1, y1, x2, y2 = [float(v) for v in box]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return [left, top, right, bottom]


def hbox_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = normalize_hbox(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def hbox_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = normalize_hbox(box)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def hbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = normalize_hbox(a)
    bx1, by1, bx2, by2 = normalize_hbox(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = hbox_area(a) + hbox_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def obox_to_hbox(obox: Sequence[float]) -> List[float]:
    """Convert cx, cy, w, h, theta to an axis-aligned envelope.

    The theta value is intentionally ignored here. This keeps horizontal-box
    evaluation available even when a baseline cannot emit rotated boxes.
    """

    if len(obox) != 5:
        raise ValueError(f"Expected obox with 5 values, got {len(obox)}")
    cx, cy, width, height, _theta = [float(v) for v in obox]
    return [
        cx - width / 2.0,
        cy - height / 2.0,
        cx + width / 2.0,
        cy + height / 2.0,
    ]


def point_inside_hbox(point: Tuple[float, float], box: Sequence[float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = normalize_hbox(box)
    return x1 <= x <= x2 and y1 <= y <= y2


def center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = hbox_center(a)
    bx, by = hbox_center(b)
    return sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def greedy_match(
    gt_boxes: Sequence[Sequence[float]],
    pred_boxes: Sequence[Sequence[float]],
    iou_threshold: float,
    pred_scores: Optional[Sequence[float]] = None,
) -> MatchResult:
    if pred_scores is None:
        pred_scores = [1.0] * len(pred_boxes)
    order = sorted(range(len(pred_boxes)), key=lambda idx: pred_scores[idx], reverse=True)
    matched_gt = set()
    pairs: List[Tuple[int, int]] = []

    for pred_idx in order:
        best_gt_idx = None
        best_iou = 0.0
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            score = hbox_iou(gt_box, pred_boxes[pred_idx])
            if score > best_iou:
                best_iou = score
                best_gt_idx = gt_idx
        if best_gt_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
            pairs.append((best_gt_idx, pred_idx))

    tp = len(pairs)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return MatchResult(tp=tp, fp=fp, fn=fn, matched_pairs=pairs)


def size_bucket(
    box: Sequence[float],
    image_width: Optional[float],
    image_height: Optional[float],
    small_max_area_ratio: float = 0.001,
    medium_max_area_ratio: float = 0.01,
) -> str:
    if image_width and image_height and image_width > 0 and image_height > 0:
        ratio = hbox_area(box) / float(image_width * image_height)
    else:
        # Coordinates are often normalized to [0, 1000] for grounding models.
        ratio = hbox_area(box) / 1_000_000.0
    if ratio < small_max_area_ratio:
        return "small"
    if ratio < medium_max_area_ratio:
        return "medium"
    return "large"


def dense_bucket(object_count: int, sparse_max: int = 5, medium_max: int = 20) -> str:
    if object_count <= sparse_max:
        return "sparse"
    if object_count <= medium_max:
        return "medium_dense"
    return "dense"


def best_box_for_relation(
    subject_boxes: Sequence[Sequence[float]],
    anchor_boxes: Sequence[Sequence[float]],
    relation_type: str,
) -> List[List[float]]:
    if not subject_boxes or not anchor_boxes:
        return []

    relation_type = relation_type.lower()
    if relation_type in {"closest_to", "near"}:
        best = min(
            subject_boxes,
            key=lambda subject: min(center_distance(subject, anchor) for anchor in anchor_boxes),
        )
        return [normalize_hbox(best)]

    if relation_type == "inside":
        selected = []
        for subject in subject_boxes:
            center = hbox_center(subject)
            if any(point_inside_hbox(center, anchor) for anchor in anchor_boxes):
                selected.append(normalize_hbox(subject))
        return selected

    if relation_type == "left_of":
        selected = []
        for subject in subject_boxes:
            sx, _sy = hbox_center(subject)
            if any(sx < hbox_center(anchor)[0] for anchor in anchor_boxes):
                selected.append(normalize_hbox(subject))
        return selected

    if relation_type == "right_of":
        selected = []
        for subject in subject_boxes:
            sx, _sy = hbox_center(subject)
            if any(sx > hbox_center(anchor)[0] for anchor in anchor_boxes):
                selected.append(normalize_hbox(subject))
        return selected

    return []


def flatten(iterable: Iterable[Iterable[object]]) -> List[object]:
    return [item for group in iterable for item in group]
