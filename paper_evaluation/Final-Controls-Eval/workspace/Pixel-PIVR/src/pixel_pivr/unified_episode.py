"""Unified source-query loading and branch-isolated wave construction."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100
SCHEMA = "caw-pivr-unified-hbb-episode-v2"
ROUND1_ANSWER_FORMATS = ("reference_geometry", "geometry_only")


class IndexedJsonl:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.offsets: list[int] = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        self._handle = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._handle is None:
            self._handle = self.path.open(encoding="utf-8")
        self._handle.seek(self.offsets[index])
        return json.loads(self._handle.readline())

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        return state


def box_text(values: Sequence[int] | None) -> str:
    if values is None:
        return "<box>None</box>"
    return "<box>" + "".join(f"<{int(value)}>" for value in values) + "</box>"


def round1_conversation(
    branch: Mapping[str, Any], answer_format: str = "reference_geometry"
) -> list[dict[str, str]]:
    if answer_format not in ROUND1_ANSWER_FORMATS:
        raise ValueError(f"Unsupported Round-1 answer format: {answer_format}")
    conversations = copy.deepcopy(branch["conversations"])
    if answer_format == "reference_geometry":
        return conversations
    points = branch["target_points"]
    if bool(points) != (branch["polarity"] == "positive"):
        raise ValueError("Round-1 polarity and target points disagree")
    if any(len(point) != 2 or any(int(v) != v or not 0 <= v <= 1000 for v in point) for point in points):
        raise ValueError("Round-1 requires normalized integer point coordinates")
    geometry = "".join(box_text(point) for point in points) if points else box_text(None)
    # Verify the structured view preserves every original target. The reference
    # stays in the unchanged input and branch metadata, not the answer tokens.
    if (len(conversations) != 2 or conversations[0]["from"] != "human"
            or conversations[1]["from"] != "gpt"
            or conversations[1]["value"] != f"<ref>{branch['reference']}</ref>{geometry}"):
        raise ValueError("Round-1 structured targets do not match the source answer")
    conversations[1]["value"] = geometry
    return conversations


def round2_conversation(branch: Mapping[str, Any]) -> list[dict[str, str]]:
    reference = str(branch["reference"])
    point = list(map(int, branch["point"]))
    target = branch.get("target_hbb")
    prompt = (
        f"<image>\nLocate the single {reference} containing point "
        f"{box_text(point)} in horizontal box format. Return None if absent."
    )
    return [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": box_text(None if target is None else list(map(int, target)))},
    ]


def x0_length(position_ids: torch.Tensor) -> int:
    values = position_ids.reshape(-1)
    drops = torch.nonzero(values[1:] < values[:-1], as_tuple=False)
    return int(drops[0].item() + 1) if drops.numel() else int(values.numel())


def first_supervised(labels: torch.Tensor) -> int:
    values = torch.nonzero(labels.reshape(-1).ne(IGNORE_INDEX), as_tuple=False)
    if not values.numel():
        raise ValueError("Branch has no supervised token")
    return int(values[0].item())


def longest_common_prefix(branches: Sequence[Mapping[str, Any]]) -> int:
    if not branches:
        raise ValueError("Cannot find a prefix for no branches")
    sequences = [branch["input_ids"].reshape(-1) for branch in branches]
    limit = min(
        min(
            first_supervised(branch["labels"]),
            x0_length(branch["position_ids"]),
        )
        for branch in branches
    )
    limit = min(limit, *(int(sequence.numel()) for sequence in sequences))
    for index in range(limit):
        value = int(sequences[0][index])
        if any(int(sequence[index]) != value for sequence in sequences[1:]):
            return index
    return limit


def combined_length(branches: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    prefix = longest_common_prefix(branches)
    return prefix + sum(int(branch["input_ids"].numel()) - prefix for branch in branches), prefix


def split_waves(
    branches: Sequence[Mapping[str, Any]],
    *,
    max_sequence: int,
    maximum_wave_size: int,
) -> list[list[Mapping[str, Any]]]:
    waves: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    current_prefix = 0
    current_lengths = 0

    def prefix_with_first(branch: Mapping[str, Any], limit: int) -> int:
        first = current[0]["input_ids"].reshape(-1)
        value = branch["input_ids"].reshape(-1)
        stop = min(
            limit,
            first_supervised(branch["labels"]),
            x0_length(branch["position_ids"]),
            int(value.numel()),
        )
        unequal = torch.nonzero(first[:stop].ne(value[:stop]), as_tuple=False)
        return int(unequal[0].item()) if unequal.numel() else stop

    for branch in branches:
        branch_length = int(branch["input_ids"].numel())
        if current:
            candidate_prefix = prefix_with_first(branch, current_prefix)
            candidate_count = len(current) + 1
            length = current_lengths + branch_length - (candidate_count - 1) * candidate_prefix
        else:
            candidate_prefix = min(
                first_supervised(branch["labels"]),
                x0_length(branch["position_ids"]),
            )
            candidate_count = 1
            length = branch_length
        if current and (candidate_count > maximum_wave_size or length > max_sequence):
            waves.append(current)
            current = []
            current_prefix = min(
                first_supervised(branch["labels"]),
                x0_length(branch["position_ids"]),
            )
            current_lengths = 0
            length = branch_length
        if length > max_sequence:
            raise ValueError(
                f"One branch requires {length} tokens, exceeding {max_sequence}"
            )
        current.append(branch)
        current_lengths += branch_length
        if len(current) == 1:
            current_prefix = min(
                first_supervised(branch["labels"]),
                x0_length(branch["position_ids"]),
            )
        else:
            current_prefix = candidate_prefix
    if current:
        waves.append(current)
    return waves


def objective_branches_for_task(
    task: str,
    direct: Mapping[str, Any],
    round1: Sequence[Mapping[str, Any]],
    round2: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return only branches that belong to the declared episode objective."""
    if task == "pointing":
        return list(round1)
    return [direct, *round1, *round2]


def _expand_single_image_placeholder(
    text: str, processor: Any, image_tokens: int
) -> str:
    marker = "<image-1>"
    if text.count(marker) != 1:
        raise ValueError(f"Expected exactly one {marker}, got {text.count(marker)}")
    expansion = (
        f"<image 1>{processor.image_start_token}"
        f"{processor.image_token * int(image_tokens)}"
        f"{processor.image_end_token}"
    )
    return text.replace(marker, expansion)


class UnifiedEpisodeDataset(Dataset):
    """Compile one image and all direct/Round-1/Round-2 branches lazily."""

    def __init__(
        self,
        path: Path,
        processor: Any,
        *,
        data_root: Path,
        eagle_root: Path,
        max_sequence: int,
        maximum_wave_size: int = 200,
        round1_answer_format: str = "reference_geometry",
    ) -> None:
        if round1_answer_format not in ROUND1_ANSWER_FORMATS:
            raise ValueError(f"Unsupported Round-1 answer format: {round1_answer_format}")
        self.round1_answer_format = round1_answer_format
        self.rows = IndexedJsonl(path)
        self.processor = processor
        self.data_root = data_root.resolve()
        self.max_sequence = int(max_sequence)
        self.maximum_wave_size = int(maximum_wave_size)
        if str(eagle_root.resolve()) not in __import__("sys").path:
            __import__("sys").path.insert(0, str(eagle_root.resolve()))
        from eaglevl.train.locany_finetune_magi_stream import LazySupervisedDatasetMTP

        self.mtp = LazySupervisedDatasetMTP(
            "caw_pivr_unified_v2_helper",
            {
                "annotation": str(path.resolve()),
                "root": str(self.data_root),
                "repeat_time": 1.0,
                "data_augment": False,
            },
            processor,
            block_size=6,
        )
        from eaglevl.train.tools import process_multimodal_sample

        self.process_multimodal_sample = process_multimodal_sample

    def __len__(self) -> int:
        return len(self.rows)

    def _messages(self, image: str, conversations: Sequence[Mapping[str, str]]):
        raw = {"image": image, "conversations": copy.deepcopy(list(conversations))}
        return self.process_multimodal_sample(
            raw, str(self.data_root), 16, 2, int(32000 * 28 * 28 * 0.9)
        )

    def _prepare_image(self, image: str, conversations: Sequence[Mapping[str, str]]):
        messages = self._messages(image, conversations)
        image_inputs, video_inputs = self.processor.process_vision_info(messages)
        if video_inputs is not None or image_inputs is None or len(image_inputs) != 1:
            raise ValueError("Unified HBB episode must contain exactly one image")
        processed = self.processor.image_processor(images=image_inputs)
        grids = np.asarray(processed["image_grid_hws"])
        if grids.shape != (1, 2):
            raise ValueError(f"Unexpected single-image grid shape: {grids.shape}")
        merge_area = math.prod(self.processor.image_processor.merge_kernel_size)
        image_tokens = int(grids[0, 0] * grids[0, 1]) // int(merge_area)
        return processed["pixel_values"], grids, image_tokens

    def _compile(
        self,
        image: str,
        conversations: Sequence[Mapping[str, str]],
        image_tokens: int,
        *,
        route: str,
        branch_id: str,
        polarity: str,
        negative_type: str | None = None,
    ) -> dict[str, Any]:
        messages = self._messages(image, conversations)
        text = self.processor.py_apply_chat_template(messages, tokenize=False)
        text = _expand_single_image_placeholder(text, self.processor, image_tokens)
        tokenized = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        labels = self.mtp.get_targets_flag_with_mtp(tokenized["input_ids"][0], block_size=6)
        result = {
            "input_ids": labels["input_ids"],
            "labels": labels["labels"],
            "position_ids": labels["position_ids"],
            "route": route,
            "branch_id": branch_id,
            "polarity": polarity,
            "negative_type": negative_type,
        }
        if route == "round2":
            x0 = x0_length(result["position_ids"])
            # The released helper appends a second EOS/null MTP block even when
            # the answer is exactly one HBB. Round 2 is explicitly one PBD6
            # geometry decision, so retain the first six-token block and its
            # terminal pad while leaving the native conversation unchanged.
            if int(result["input_ids"].numel()) < x0 + 7:
                raise ValueError(f"Round-2 branch {branch_id} has no complete PBD6 block")
            first_block = slice(x0, x0 + 6)
            first_targets = slice(x0 + 1, x0 + 7)
            result["input_ids"] = torch.cat(
                (
                    result["input_ids"][:x0],
                    result["input_ids"][first_block],
                    result["input_ids"][-1:],
                )
            )
            result["labels"] = torch.cat(
                (
                    result["labels"][:x0],
                    torch.full((1,), IGNORE_INDEX, dtype=result["labels"].dtype),
                    result["labels"][first_targets],
                )
            )
            block_positions = result["position_ids"][first_block]
            result["position_ids"] = torch.cat(
                (
                    result["position_ids"][:x0],
                    block_positions,
                    block_positions[-1:] + 1,
                )
            )
            result["labels"] = result["labels"].clone()
            result["labels"][: x0 + 1] = IGNORE_INDEX
            tail = int(result["labels"].numel()) - (x0 + 1)
            if tail != 6:
                raise ValueError(
                    f"Round-2 branch {branch_id} produced {tail} PBD labels, expected 6"
                )
        if int(result["input_ids"].numel()) > self.max_sequence:
            raise ValueError(
                f"{route} branch {branch_id} has {result['input_ids'].numel()} tokens "
                f"> {self.max_sequence}"
            )
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if row.get("schema_version") != SCHEMA:
            raise ValueError(f"Unsupported unified schema at index {index}")
        image = str(row["image"])
        seed_conversation = row["direct"]["conversations"]
        pixel_values, grids, image_tokens = self._prepare_image(image, seed_conversation)

        direct = self._compile(
            image,
            row["direct"]["conversations"],
            image_tokens,
            route="direct",
            branch_id=f"{row['episode_id']}:direct",
            polarity="positive",
        )
        round1 = [
            self._compile(
                image,
                round1_conversation(branch, self.round1_answer_format),
                image_tokens,
                route="round1",
                branch_id=str(branch["branch_id"]),
                polarity=str(branch["polarity"]),
                negative_type="class_absent" if branch["polarity"] == "negative" else None,
            )
            for branch in row["round1"]["branches"]
        ]
        round2 = [
            self._compile(
                image,
                round2_conversation(branch),
                image_tokens,
                route="round2",
                branch_id=str(branch["branch_id"]),
                polarity=str(branch["polarity"]),
                negative_type=branch.get("negative_type"),
            )
            for branch in row["round2"]["branches"]
        ]
        if not round1:
            raise ValueError(f"Episode {row['episode_id']} has no retained Round-1 branch")
        # Pointing is supervised by its native point branch only.  Keep the
        # compiled direct branch available for auditing, but do not execute it
        # in the episode objective because branch_objective_weights deliberately
        # excludes direct-box and Round-2 supervision for pointing.
        objective_branches = objective_branches_for_task(
            str(row["task"]), direct, round1, round2
        )
        return {
            "episode_id": str(row["episode_id"]),
            "task": str(row["task"]),
            "stage": str(row["stage"]),
            "dataset": str(row["dataset"]),
            "pixel_values": pixel_values,
            "image_grid_hws": grids,
            "image_tokens": image_tokens,
            "direct_waves": split_waves([direct], max_sequence=self.max_sequence, maximum_wave_size=1),
            "round1_waves": split_waves(round1, max_sequence=self.max_sequence, maximum_wave_size=self.maximum_wave_size),
            "round2_waves": split_waves(round2, max_sequence=self.max_sequence, maximum_wave_size=self.maximum_wave_size) if round2 else [],
            "episode_waves": split_waves(
                objective_branches,
                max_sequence=self.max_sequence,
                maximum_wave_size=self.maximum_wave_size,
            ),
            "round1_branches": len(round1),
            "round2_branches": len(round2),
        }


def build_branch_isolation_mask(
    branches: Sequence[Mapping[str, Any]],
    prefix: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    block_size: int = 6,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Create prefix-causal, sibling-isolated native+PBD attention."""
    total = prefix + sum(int(branch["input_ids"].numel()) - prefix for branch in branches)
    allowed = torch.zeros((total, total), dtype=torch.bool, device=device)
    if prefix:
        allowed[:prefix, :prefix] = torch.ones((prefix, prefix), dtype=torch.bool, device=device).tril()
    ranges: list[tuple[int, int]] = []
    cursor = prefix
    for branch in branches:
        length = int(branch["input_ids"].numel())
        x0 = x0_length(branch["position_ids"])
        if not 0 <= prefix <= x0 <= length:
            raise ValueError(f"Invalid prefix/x0/length: {prefix}/{x0}/{length}")
        suffix = length - prefix
        start, stop = cursor, cursor + suffix
        ranges.append((start, stop))
        native = x0 - prefix
        if native:
            allowed[start : start + native, :prefix] = True
            allowed[start : start + native, start : start + native] = torch.ones(
                (native, native), dtype=torch.bool, device=device
            ).tril()
        mtp_start = start + native
        mtp_length = length - x0
        blocks = int(math.ceil(mtp_length / block_size))
        positions = branch["position_ids"].reshape(-1)
        for block_index in range(blocks):
            local_start = x0 + block_index * block_size
            local_stop = min(length, local_start + block_size)
            q_start = mtp_start + block_index * block_size
            q_stop = q_start + (local_stop - local_start)
            prefix_length = int(positions[local_start])
            allowed[q_start:q_stop, : min(prefix, prefix_length)] = True
            own_prefix = max(0, prefix_length - prefix)
            allowed[q_start:q_stop, start : start + min(native, own_prefix)] = True
            allowed[q_start:q_stop, q_start:q_stop] = True
        cursor = stop
    mask = torch.full((1, 1, total, total), -torch.inf, dtype=dtype, device=device)
    mask.masked_fill_(allowed.unsqueeze(0).unsqueeze(0), 0.0)
    return mask, ranges


def pack_wave(
    branches: Sequence[Mapping[str, Any]],
    *,
    image_token_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    length, prefix = combined_length(branches)
    ids = [branches[0]["input_ids"][:prefix]]
    labels = [branches[0]["labels"][:prefix]]
    positions = [branches[0]["position_ids"][:prefix]]
    for branch in branches:
        ids.append(branch["input_ids"][prefix:])
        labels.append(branch["labels"][prefix:])
        positions.append(branch["position_ids"][prefix:])
    input_ids = torch.cat(ids).unsqueeze(0).to(device)
    target = torch.cat(labels).unsqueeze(0).to(device)
    position_ids = torch.cat(positions).unsqueeze(0).to(device)
    if int(input_ids.shape[1]) != length:
        raise RuntimeError("Packed wave length mismatch")
    if target[:, :prefix].ne(IGNORE_INDEX).any():
        raise RuntimeError("Shared prefix unexpectedly contains supervised labels")
    image_count = int(input_ids.eq(int(image_token_index)).sum().item())
    if image_count <= 0:
        raise RuntimeError("Packed wave lost the global image prefix")
    mask, ranges = build_branch_isolation_mask(
        branches, prefix, device=device, dtype=dtype
    )
    return {
        "input_ids": input_ids,
        "labels": target,
        "position_ids": position_ids,
        "attention_mask": mask,
        "branch_ranges": ranges,
        "branches": list(branches),
        "prefix_length": prefix,
        "image_tokens": image_count,
    }
