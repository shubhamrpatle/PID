"""Prompt and class defaults for the zero-shot analysis suite."""

from __future__ import annotations

from typing import Dict, Iterable, List

from .schema import normalize_class_name


DEFAULT_TARGET_CLASSES = [
    "airplane",
    "ship",
    "storage tank",
    "bridge",
    "helicopter",
    "vehicle",
    "container crane",
    "dam",
    "chimney",
    "roundabout",
    "tower crane",
    "shipping container",
]

STRICT_NOVEL_CLASSES = [
    "helicopter",
    "container crane",
    "helipad",
    "chimney",
    "dam",
    "shipping container",
    "tower crane",
    "roundabout",
]

NEGATIVE_QUERY_CLASSES = [
    "airplane",
    "ship",
    "storage tank",
    "bridge",
    "helicopter",
    "container crane",
    "dam",
    "chimney",
]

DEFAULT_RELATIONS = [
    {
        "type": "closest_to",
        "subject": "airplane",
        "anchor": "runway",
        "query": "Locate the airplane closest to the runway.",
    },
    {
        "type": "near",
        "subject": "ship",
        "anchor": "dock",
        "query": "Locate the ship near the dock.",
    },
    {
        "type": "inside",
        "subject": "vehicle",
        "anchor": "parking area",
        "query": "Find the vehicle inside the parking area.",
    },
    {
        "type": "left_of",
        "subject": "building",
        "anchor": "bridge",
        "query": "Locate the building left of the bridge.",
    },
]


def class_key(value: str) -> str:
    return normalize_class_name(value)


def pluralize_class_name(class_name: str) -> str:
    name = class_key(class_name)
    if name.endswith("s"):
        return name
    if name.endswith("y") and len(name) > 1 and name[-2] not in "aeiou":
        return f"{name[:-1]}ies"
    if name.endswith(("ch", "sh", "x")):
        return f"{name}es"
    return f"{name}s"


def category_query(class_name: str) -> str:
    return f"Locate all {pluralize_class_name(class_name)}."


def dense_query(class_name: str) -> str:
    return f"Locate all {pluralize_class_name(class_name)} in this dense scene."


def negative_query(class_name: str) -> str:
    return f"Locate all {pluralize_class_name(class_name)}."


def normalize_class_list(values: Iterable[str]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for value in values:
        key = class_key(value)
        if key and key not in seen:
            seen.add(key)
            normalized.append(key)
    return normalized
