"""Data contracts for zero-shot EO grounding analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from .geometry import normalize_hbox, obox_to_hbox
from .parsing import parse_model_output


class SchemaError(ValueError):
    """Raised when an evaluation or prediction record is malformed."""


def _as_float_list(values: Sequence[Any], expected_len: int, name: str) -> List[float]:
    if len(values) != expected_len:
        raise SchemaError(f"{name} must contain {expected_len} values")
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{name} must contain numeric values") from exc


def normalize_class_name(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


@dataclass
class EvalSample:
    image_id: str
    dataset: str
    split: str
    query: str
    target_class: str
    query_type: str
    gt_hboxes: List[List[float]]
    is_negative: bool = False
    gt_oboxes: List[List[float]] = field(default_factory=list)
    relation: Optional[Dict[str, Any]] = None
    image_width: Optional[float] = None
    image_height: Optional[float] = None
    source: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "EvalSample":
        required = ["image_id", "dataset", "split", "query", "target_class", "query_type"]
        for key in required:
            if key not in record:
                raise SchemaError(f"Missing required sample field: {key}")

        gt_hboxes = [normalize_hbox(_as_float_list(box, 4, "gt_hboxes[]")) for box in record.get("gt_hboxes", [])]
        gt_oboxes = [_as_float_list(box, 5, "gt_oboxes[]") for box in record.get("gt_oboxes", [])]
        is_negative = bool(record.get("is_negative", False))
        if is_negative and gt_hboxes:
            raise SchemaError("Negative samples must not include gt_hboxes")
        if not is_negative and not gt_hboxes:
            raise SchemaError("Positive samples must include at least one gt_hbox")

        return cls(
            image_id=str(record["image_id"]),
            dataset=str(record["dataset"]),
            split=str(record["split"]),
            query=str(record["query"]),
            target_class=normalize_class_name(str(record["target_class"])),
            query_type=str(record["query_type"]),
            gt_hboxes=gt_hboxes,
            is_negative=is_negative,
            gt_oboxes=gt_oboxes,
            relation=dict(record["relation"]) if record.get("relation") else None,
            image_width=float(record["image_width"]) if record.get("image_width") is not None else None,
            image_height=float(record["image_height"]) if record.get("image_height") is not None else None,
            source=str(record["source"]) if record.get("source") is not None else None,
            meta=dict(record.get("meta", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "image_id": self.image_id,
            "dataset": self.dataset,
            "split": self.split,
            "query": self.query,
            "target_class": self.target_class,
            "query_type": self.query_type,
            "gt_hboxes": self.gt_hboxes,
            "is_negative": self.is_negative,
        }
        if self.gt_oboxes:
            data["gt_oboxes"] = self.gt_oboxes
        if self.relation:
            data["relation"] = self.relation
        if self.image_width is not None:
            data["image_width"] = self.image_width
        if self.image_height is not None:
            data["image_height"] = self.image_height
        if self.source:
            data["source"] = self.source
        if self.meta:
            data["meta"] = self.meta
        return data


@dataclass
class PredBox:
    hbox: List[float]
    score: float = 1.0
    label: Optional[str] = None

    @classmethod
    def from_any(cls, value: Any) -> "PredBox":
        if isinstance(value, Mapping):
            raw_box = value.get("hbox", value.get("box"))
            if raw_box is None:
                raise SchemaError("Prediction box dict must include hbox or box")
            score = float(value.get("score", 1.0))
            label = normalize_class_name(str(value["label"])) if value.get("label") else None
            return cls(hbox=normalize_hbox(_as_float_list(raw_box, 4, "pred_hboxes[].hbox")), score=score, label=label)
        return cls(hbox=normalize_hbox(_as_float_list(value, 4, "pred_hboxes[]")))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"hbox": self.hbox, "score": self.score}
        if self.label:
            data["label"] = self.label
        return data


@dataclass
class PredictionRecord:
    image_id: str
    query: str
    dataset: Optional[str] = None
    target_class: Optional[str] = None
    output: Optional[str] = None
    pred_hboxes: List[PredBox] = field(default_factory=list)
    pred_oboxes: List[List[float]] = field(default_factory=list)
    is_negative: bool = False
    model: Optional[str] = None
    decode_mode: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens_generated: Optional[int] = None
    gpu_memory_mb: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "PredictionRecord":
        for key in ["image_id", "query"]:
            if key not in record:
                raise SchemaError(f"Missing required prediction field: {key}")

        parsed = parse_model_output(str(record.get("output", ""))) if record.get("output") else None
        pred_hboxes = [PredBox.from_any(box) for box in record.get("pred_hboxes", [])]
        pred_oboxes = [_as_float_list(box, 5, "pred_oboxes[]") for box in record.get("pred_oboxes", [])]

        if parsed and not record.get("pred_hboxes"):
            pred_hboxes.extend(PredBox(hbox=box, score=1.0, label=parsed.ref) for box in parsed.hboxes)
            pred_oboxes.extend(parsed.oboxes)
            for obox in parsed.oboxes:
                pred_hboxes.append(PredBox(hbox=obox_to_hbox(obox), score=1.0, label=parsed.ref))

        return cls(
            image_id=str(record["image_id"]),
            query=str(record["query"]),
            dataset=str(record["dataset"]) if record.get("dataset") is not None else None,
            target_class=normalize_class_name(str(record["target_class"])) if record.get("target_class") else None,
            output=str(record["output"]) if record.get("output") is not None else None,
            pred_hboxes=pred_hboxes,
            pred_oboxes=pred_oboxes,
            is_negative=bool(record.get("is_negative", False) or (parsed.negative if parsed else False)),
            model=str(record["model"]) if record.get("model") is not None else None,
            decode_mode=str(record["decode_mode"]) if record.get("decode_mode") is not None else None,
            latency_ms=float(record["latency_ms"]) if record.get("latency_ms") is not None else None,
            tokens_generated=int(record["tokens_generated"]) if record.get("tokens_generated") is not None else None,
            gpu_memory_mb=float(record["gpu_memory_mb"]) if record.get("gpu_memory_mb") is not None else None,
            meta=dict(record.get("meta", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "image_id": self.image_id,
            "query": self.query,
            "pred_hboxes": [box.to_dict() for box in self.pred_hboxes],
            "is_negative": self.is_negative,
        }
        optional = {
            "dataset": self.dataset,
            "target_class": self.target_class,
            "output": self.output,
            "model": self.model,
            "decode_mode": self.decode_mode,
            "latency_ms": self.latency_ms,
            "tokens_generated": self.tokens_generated,
            "gpu_memory_mb": self.gpu_memory_mb,
        }
        for key, value in optional.items():
            if value is not None:
                data[key] = value
        if self.pred_oboxes:
            data["pred_oboxes"] = self.pred_oboxes
        if self.meta:
            data["meta"] = self.meta
        return data


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{line_number} is not valid JSON") from exc


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_samples(path: str | Path) -> List[EvalSample]:
    return [EvalSample.from_dict(record) for record in read_jsonl(path)]


def load_predictions(path: str | Path) -> List[PredictionRecord]:
    return [PredictionRecord.from_dict(record) for record in read_jsonl(path)]
