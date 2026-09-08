from __future__ import annotations
import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping
import torch


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")


def read_jsonl_log(path: Path, *, repair_tail: bool = False) -> list[dict]:
    """Recover an interrupted append, never a corrupt complete/interior record."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows, offset = [], 0
    repaired = raw
    for index, line in enumerate(lines):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except (ValueError, UnicodeDecodeError) as error:
                if not (repair_tail and index == len(lines) - 1 and not line.endswith(b"\n")):
                    raise ValueError(f"Corrupt JSONL log {path}, line {index + 1}") from error
                repaired = raw[:offset]
                warnings.warn(f"Discarding unfinished final log entry in {path}", RuntimeWarning)
        offset += len(line)
    if repair_tail and repaired and not repaired.endswith(b"\n"):
        repaired += b"\n"
    if repair_tail and repaired != raw:
        temporary = path.with_name(f".{path.name}.repair-{os.getpid()}")
        temporary.write_bytes(repaired)
        os.replace(temporary, path)
    return rows


def atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def update_symlink(link: Path, target: Path) -> None:
    temporary = link.with_name(f".{link.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(target.name)
    os.replace(temporary, link)


def sha256_file(path: Path, block_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()
