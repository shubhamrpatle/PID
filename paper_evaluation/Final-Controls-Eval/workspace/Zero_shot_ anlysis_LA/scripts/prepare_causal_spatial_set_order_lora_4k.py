#!/usr/bin/env python3
"""Canonicalize HBB sets into a locality-preserving autoregressive order.

The source split and every native LocateAnything token are preserved.  Only the
order of numeric HBB blocks inside each ``<ref>class</ref>`` group changes.  Box
centers are sorted by a 10-bit Hilbert index with deterministic coordinate
tie-breakers.  Explicit ``<box>None</box>`` groups are copied byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT / "dataset/sft/causal_pointrelay_pbd_lora_4k_v1"
OUTPUT_ROOT = PROJECT / "dataset/sft/causal_spatial_set_order_lora_4k_v1"
HILBERT_BITS = 10
HBB_RE = re.compile(
    r"<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>"
)
REF_RE = re.compile(r"<ref>.*?</ref>")
NONE_BLOCK = "<box>None</box>"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def assistant_text(record: Mapping[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("Record has no conversations")
    turn = conversations[-1]
    if turn.get("from") != "gpt" or not isinstance(turn.get("value"), str):
        raise ValueError("Record does not end with a string GPT answer")
    return str(turn["value"])


def parse_box(block: str) -> tuple[int, int, int, int]:
    match = HBB_RE.fullmatch(block)
    if match is None:
        raise ValueError(f"Not a native HBB block: {block}")
    return tuple(int(match.group(key)) for key in ("x1", "y1", "x2", "y2"))


def center(box: Sequence[int]) -> tuple[int, int]:
    return ((int(box[0]) + int(box[2])) // 2, (int(box[1]) + int(box[3])) // 2)


def hilbert_index(x: int, y: int, bits: int = HILBERT_BITS) -> int:
    """Map an integer 2-D point to a Hilbert distance."""
    side = 1 << bits
    x = max(0, min(side - 1, int(x)))
    y = max(0, min(side - 1, int(y)))
    distance = 0
    scale = side // 2
    while scale:
        rx = 1 if x & scale else 0
        ry = 1 if y & scale else 0
        distance += scale * scale * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = side - 1 - x
                y = side - 1 - y
            x, y = y, x
        scale //= 2
    return distance


def path_length(boxes: Sequence[Sequence[int]]) -> float:
    points = [center(box) for box in boxes]
    return sum(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(points, points[1:])
    )


def spatial_key(block: str) -> tuple[int, int, int, int, int]:
    box = parse_box(block)
    cx, cy = center(box)
    return (hilbert_index(cx, cy), *box)


def grouped_blocks(answer: str) -> list[tuple[str, list[str]]]:
    refs = list(REF_RE.finditer(answer))
    if not refs:
        raise ValueError("Detection answer has no <ref> group")
    if refs[0].start() != 0:
        raise ValueError("Unexpected text before the first <ref> group")
    groups: list[tuple[str, list[str]]] = []
    for index, ref in enumerate(refs):
        end = refs[index + 1].start() if index + 1 < len(refs) else len(answer)
        body = answer[ref.end() : end]
        if body == NONE_BLOCK:
            groups.append((ref.group(0), [NONE_BLOCK]))
            continue
        blocks = [match.group(0) for match in HBB_RE.finditer(body)]
        if not blocks or "".join(blocks) != body:
            raise ValueError(f"Unsupported geometry body after {ref.group(0)}: {body[:160]}")
        groups.append((ref.group(0), blocks))
    return groups


def canonicalize_answer(answer: str) -> tuple[str, dict[str, Any]]:
    output: list[str] = []
    numeric_groups = changed_groups = boxes_total = 0
    before_path = after_path = 0.0
    for ref, blocks in grouped_blocks(answer):
        output.append(ref)
        if blocks == [NONE_BLOCK]:
            output.append(NONE_BLOCK)
            continue
        numeric_groups += 1
        boxes_total += len(blocks)
        ordered = sorted(blocks, key=spatial_key)
        if ordered != blocks:
            changed_groups += 1
        source_boxes = [parse_box(block) for block in blocks]
        ordered_boxes = [parse_box(block) for block in ordered]
        if Counter(source_boxes) != Counter(ordered_boxes):
            raise AssertionError("Canonical ordering changed the HBB multiset")
        before_path += path_length(source_boxes)
        after_path += path_length(ordered_boxes)
        output.extend(ordered)
    converted = "".join(output)
    if Counter(HBB_RE.findall(answer)) != Counter(HBB_RE.findall(converted)):
        raise AssertionError("Answer-level HBB multiset changed")
    if answer.count(NONE_BLOCK) != converted.count(NONE_BLOCK):
        raise AssertionError("Explicit None groups changed")
    return converted, {
        "numeric_groups": numeric_groups,
        "changed_groups": changed_groups,
        "boxes": boxes_total,
        "source_path_length": before_path,
        "ordered_path_length": after_path,
    }


def converted_record(
    record: Mapping[str, Any], *, split: str, source_line: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = json.loads(json.dumps(record))
    source = assistant_text(record)
    converted, audit = canonicalize_answer(source)
    output["conversations"][-1]["value"] = converted
    meta = dict(output.get("meta") or {})
    if int(meta.get("target_count") or 0) != audit["boxes"]:
        raise ValueError(
            f"Target count mismatch at {split}:{source_line}: "
            f"meta={meta.get('target_count')} boxes={audit['boxes']}"
        )
    meta.update(
        {
            "answer_format": "native_grouped_ref_hbb_hilbert_order_v1",
            "instance_order": "hilbert_center_10bit_with_coordinate_tiebreak",
            "instance_order_scope": "within_each_ref_group",
            "instance_order_changed_groups": audit["changed_groups"],
            "source_pilot": "causal_pointrelay_pbd_lora_4k_v1",
            "source_pilot_split": split,
            "source_pilot_line": source_line,
        }
    )
    output["meta"] = meta
    return output, audit


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def build_split(source: Path, destination: Path, split: str) -> dict[str, Any]:
    encoded: list[str] = []
    records = numeric_groups = changed_groups = boxes = none_groups = 0
    source_path = ordered_path = 0.0
    hashes: set[str] = set()
    for line_number, record in iter_jsonl(source):
        output, audit = converted_record(record, split=split, source_line=line_number)
        encoded.append(json.dumps(output, sort_keys=True, separators=(",", ":")))
        records += 1
        numeric_groups += int(audit["numeric_groups"])
        changed_groups += int(audit["changed_groups"])
        boxes += int(audit["boxes"])
        none_groups += assistant_text(output).count(NONE_BLOCK)
        source_path += float(audit["source_path_length"])
        ordered_path += float(audit["ordered_path_length"])
        digest = str((output.get("meta") or {}).get("image_content_sha256") or "")
        if len(digest) != 64:
            raise ValueError(f"Invalid image hash at {split}:{line_number}")
        hashes.add(digest)
    atomic_write(destination, "\n".join(encoded) + ("\n" if encoded else ""))
    return {
        "path": str(destination.resolve()),
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "sha256": sha256_file(destination),
        "records": records,
        "unique_image_hashes": len(hashes),
        "hbb_targets_preserved": boxes,
        "explicit_none_groups_preserved": none_groups,
        "numeric_ref_groups": numeric_groups,
        "changed_ref_groups": changed_groups,
        "source_path_length": source_path,
        "ordered_path_length": ordered_path,
        "path_length_ratio": ordered_path / source_path if source_path else 1.0,
        "image_hashes": sorted(hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    source_manifest_path = args.source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    built: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "smoke"):
        built[split] = build_split(
            args.source_root / f"{split}.jsonl",
            args.output_root / f"{split}.jsonl",
            split,
        )
        expected = int(source_manifest[split]["records"])
        if built[split]["records"] != expected:
            raise AssertionError(f"{split} record count changed")

    train_hashes = set(built["train"].pop("image_hashes"))
    validation_hashes = set(built["validation"].pop("image_hashes"))
    built["smoke"].pop("image_hashes")
    manifest = {
        "schema_version": "causal-spatial-set-order-lora-4k-v1",
        "research_scope": "detection-only matched mechanism gate; not a benchmark claim",
        "hypothesis": (
            "A locality-preserving canonical order reduces permutation entropy and "
            "autoregressive coordinate drift in dense set generation."
        ),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "conversion": {
            "scope": "numeric HBB blocks within each native <ref> group only",
            "order": "10-bit Hilbert index of integer HBB center",
            "tie_break": "x1,y1,x2,y2",
            "native_grammar_changed": False,
            "tokens_added": 0,
            "targets_added_or_removed": 0,
            "explicit_none_policy": "byte-identical",
        },
        "training_contract": {
            "initialization": "released Base LocateAnything-3B",
            "lora_scope": "Qwen q/k/v/o and gate/down/up projections only",
            "lora_rank": 16,
            "optimizer_steps": 4000,
            "gradient_accumulation": 4,
            "record_exposures": 16000,
            "complete_passes": 1,
            "image_token_limit": 1024,
            "sequence_budget": 8192,
            "matched_standard_control": "causal_pointrelay_pbd_lora_4k_v1/standard",
        },
        "train_validation_hash_overlap": len(train_hashes & validation_hashes),
        "benchmark_hash_overlap": int(source_manifest["benchmark_hash_overlap"]),
        **built,
    }
    if manifest["train_validation_hash_overlap"]:
        raise AssertionError("Train/validation leakage introduced")
    if manifest["benchmark_hash_overlap"]:
        raise AssertionError("Source pilot overlaps benchmark images")
    atomic_write(
        args.output_root / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
