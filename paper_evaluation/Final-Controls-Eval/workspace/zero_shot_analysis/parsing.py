"""Parser for structured grounding outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .geometry import normalize_hbox


NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
HBOX_RE = re.compile(r"<hbox>\s*(.*?)\s*</hbox>", re.IGNORECASE | re.DOTALL)
OBOX_RE = re.compile(r"<obox>\s*(.*?)\s*</obox>", re.IGNORECASE | re.DOTALL)
BOX_RE = re.compile(r"<box>\s*(.*?)\s*</box>", re.IGNORECASE | re.DOTALL)
REF_RE = re.compile(r"<ref>\s*(.*?)\s*</ref>", re.IGNORECASE | re.DOTALL)
NEGATIVE_RE = re.compile(r"<negative>\s*(.*?)\s*</negative>", re.IGNORECASE | re.DOTALL)


@dataclass
class ParsedOutput:
    ref: Optional[str] = None
    hboxes: List[List[float]] = field(default_factory=list)
    oboxes: List[List[float]] = field(default_factory=list)
    negative: bool = False
    negative_text: Optional[str] = None


def _numbers(text: str) -> List[float]:
    return [float(match.group(0)) for match in re.finditer(NUMBER_PATTERN, text)]


def parse_model_output(text: str) -> ParsedOutput:
    result = ParsedOutput()

    ref_match = REF_RE.search(text)
    if ref_match:
        result.ref = " ".join(ref_match.group(1).strip().lower().split())

    negative_match = NEGATIVE_RE.search(text)
    if negative_match:
        result.negative = True
        result.negative_text = negative_match.group(1).strip()

    for match in HBOX_RE.finditer(text):
        values = _numbers(match.group(1))
        if len(values) >= 4:
            result.hboxes.append(normalize_hbox(values[:4]))

    for match in BOX_RE.finditer(text):
        values = _numbers(match.group(1))
        if len(values) >= 4:
            result.hboxes.append(normalize_hbox(values[:4]))

    for match in OBOX_RE.finditer(text):
        values = _numbers(match.group(1))
        if len(values) >= 5:
            result.oboxes.append(values[:5])

    return result
