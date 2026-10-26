"""Build unified zero-shot evaluation JSONL from object annotations."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .geometry import best_box_for_relation, normalize_hbox
from .prompts import (
    DEFAULT_RELATIONS,
    DEFAULT_TARGET_CLASSES,
    NEGATIVE_QUERY_CLASSES,
    STRICT_NOVEL_CLASSES,
    category_query,
    class_key,
    dense_query,
    negative_query,
    normalize_class_list,
)
from .schema import EvalSample, read_jsonl


def load_config(path: Optional[str | Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _object_class(obj: Mapping[str, Any]) -> str:
    for key in ("class_name", "category", "label", "name"):
        if key in obj:
            return class_key(str(obj[key]))
    raise ValueError("Annotation object must include class_name, category, label, or name")


def _object_hbox(obj: Mapping[str, Any]) -> List[float]:
    for key in ("hbox", "bbox", "box"):
        if key in obj:
            return normalize_hbox(obj[key])
    if "obox" in obj:
        from .geometry import obox_to_hbox

        return obox_to_hbox(obj["obox"])
    raise ValueError("Annotation object must include hbox, bbox, box, or obox")


def _group_objects(record: Mapping[str, Any]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for obj in record.get("objects", []):
        grouped[_object_class(obj)].append(obj)
    return grouped


def _boxes_for(objects: Iterable[Mapping[str, Any]]) -> List[List[float]]:
    return [_object_hbox(obj) for obj in objects]


def _oboxes_for(objects: Iterable[Mapping[str, Any]]) -> List[List[float]]:
    oboxes = []
    for obj in objects:
        if "obox" in obj:
            values = [float(value) for value in obj["obox"]]
            if len(values) == 5:
                oboxes.append(values)
    return oboxes


def build_samples_from_records(records: Iterable[Mapping[str, Any]], config: Optional[Mapping[str, Any]] = None) -> List[EvalSample]:
    cfg = dict(config or {})
    target_classes = normalize_class_list(cfg.get("target_classes", DEFAULT_TARGET_CLASSES))
    novel_classes = set(normalize_class_list(cfg.get("novel_classes", STRICT_NOVEL_CLASSES)))
    negative_classes = normalize_class_list(cfg.get("negative_classes", NEGATIVE_QUERY_CLASSES))
    relations = list(cfg.get("relations", DEFAULT_RELATIONS))
    include_category = bool(cfg.get("include_category_queries", True))
    include_novel = bool(cfg.get("include_novel_queries", True))
    include_dense = bool(cfg.get("include_dense_queries", True))
    include_negative = bool(cfg.get("include_negative_queries", True))
    include_relations = bool(cfg.get("include_relation_queries", True))
    dense_min_instances = int(cfg.get("dense_min_instances", 10))
    max_negative_per_image = int(cfg.get("max_negative_queries_per_image", 2))

    samples: List[EvalSample] = []
    for record in records:
        image_id = str(record["image_id"])
        dataset = str(record.get("dataset", "unknown"))
        split = str(record.get("split", "test"))
        width = float(record["width"]) if record.get("width") is not None else None
        height = float(record["height"]) if record.get("height") is not None else None
        source = (
            str(record["image_path"])
            if record.get("image_path") is not None
            else str(record["source"])
            if record.get("source") is not None
            else str(record["file_name"])
            if record.get("file_name") is not None
            else None
        )
        grouped = _group_objects(record)
        present = set(grouped)

        for target_class in sorted(present):
            if target_classes and target_class not in target_classes and target_class not in novel_classes:
                continue
            boxes = _boxes_for(grouped[target_class])
            oboxes = _oboxes_for(grouped[target_class])
            query_type = "novel_class" if target_class in novel_classes else "category"
            if include_category or (include_novel and query_type == "novel_class"):
                samples.append(
                    EvalSample(
                        image_id=image_id,
                        dataset=dataset,
                        split=split,
                        query=category_query(target_class),
                        target_class=target_class,
                        query_type=query_type,
                        gt_hboxes=boxes,
                        gt_oboxes=oboxes,
                        image_width=width,
                        image_height=height,
                        source=source,
                    )
                )
            if include_dense and len(boxes) >= dense_min_instances:
                samples.append(
                    EvalSample(
                        image_id=image_id,
                        dataset=dataset,
                        split=split,
                        query=dense_query(target_class),
                        target_class=target_class,
                        query_type="dense",
                        gt_hboxes=boxes,
                        gt_oboxes=oboxes,
                        image_width=width,
                        image_height=height,
                        source=source,
                        meta={"object_count": len(boxes)},
                    )
                )

        if include_negative:
            emitted = 0
            for target_class in negative_classes:
                if target_class in present:
                    continue
                samples.append(
                    EvalSample(
                        image_id=image_id,
                        dataset=dataset,
                        split=split,
                        query=negative_query(target_class),
                        target_class=target_class,
                        query_type="negative",
                        gt_hboxes=[],
                        is_negative=True,
                        image_width=width,
                        image_height=height,
                        source=source,
                    )
                )
                emitted += 1
                if emitted >= max_negative_per_image:
                    break

        if include_relations:
            for relation in relations:
                relation_type = str(relation["type"]).strip().lower().replace("-", "_").replace(" ", "_")
                subject = class_key(str(relation["subject"]))
                anchor = class_key(str(relation["anchor"]))
                if subject not in grouped or anchor not in grouped:
                    continue
                target_boxes = best_box_for_relation(_boxes_for(grouped[subject]), _boxes_for(grouped[anchor]), relation_type)
                if not target_boxes:
                    continue
                samples.append(
                    EvalSample(
                        image_id=image_id,
                        dataset=dataset,
                        split=split,
                        query=str(relation.get("query") or f"Locate the {subject} {relation_type} the {anchor}."),
                        target_class=subject,
                        query_type="relation",
                        gt_hboxes=target_boxes,
                        relation={"type": relation_type, "subject": subject, "anchor": anchor},
                        image_width=width,
                        image_height=height,
                        source=source,
                    )
                )

    return samples


def load_annotation_records(path: str | Path) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))
