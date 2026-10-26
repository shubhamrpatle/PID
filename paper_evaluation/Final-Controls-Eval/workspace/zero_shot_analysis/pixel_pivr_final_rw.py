"""Final detection-only Pixel-PIVR read/write training components.

One episode shares a single frozen MoonViT/projector result across three Qwen
LoRA objectives: direct HBB generation, point-address generation, and one
point-indexed HBB/None branch. The branch receives an unpadded ROI of up to
12x12 tokens gathered from the post-projector global grid; no second visual
encoding is performed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import Dataset

from scripts.train_pointrelay_pbd_lora_pilot import chunked_ce, make_dataset

from .cached_projected_roi import (
    extract_projected_roi_unpadded,
    insert_virtual_roi_image,
)
from .pointrelay_pbd import base_causal_lm


PHASES = ("direct", "address", "branch")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line: {path}:{line_number}")
            rows.append(json.loads(line))
    return rows


class FinalPixelPIVREpisodeDataset(Dataset):
    """Return three aligned LA-loader views for each canonical episode."""

    def __init__(self, root: Path, split: str, processor: Any) -> None:
        self.root = root.resolve()
        self.split = str(split)
        self.index = _read_jsonl(self.root / "index" / f"{self.split}.jsonl")
        self.episodes = _read_jsonl(self.root / f"{self.split}.jsonl")
        self.datasets = {
            phase: make_dataset(
                self.root / "views" / f"{self.split}_{phase}.jsonl", processor
            )
            for phase in PHASES
        }
        lengths = {len(self.index), len(self.episodes), *(len(value) for value in self.datasets.values())}
        if len(lengths) != 1:
            raise RuntimeError(f"Pixel-PIVR aligned-view length mismatch: {sorted(lengths)}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self.episodes[index]
        index_row = self.index[index]
        episode_id = str(episode.get("episode_id") or "")
        if index_row.get("episode_id") != episode_id:
            raise RuntimeError(f"Episode/index mismatch at {self.split}:{index + 1}")
        samples: dict[str, Any] = {}
        for phase, dataset in self.datasets.items():
            source_index = dataset.active_indices[index]
            raw = dataset.lazy_loader[source_index]
            meta = dict(raw.get("meta") or {})
            if meta.get("episode_id") != episode_id or meta.get("pivr_final_phase") != phase:
                raise RuntimeError(
                    f"Misaligned {phase} view at {self.split}:{index + 1}"
                )
            samples[phase] = dataset[index]
        branch_meta = dict(
            self.datasets["branch"].lazy_loader[
                self.datasets["branch"].active_indices[index]
            ].get("meta")
            or {}
        )
        return {
            "episode_id": episode_id,
            "dataset_index": int(index),
            "direct": samples["direct"],
            "address": samples["address"],
            "branch": samples["branch"],
            "branch_point": torch.tensor(
                branch_meta["pivr_branch_point"], dtype=torch.float32
            ),
            "branch_valid": bool(branch_meta["pivr_branch_valid"]),
            "branch_id": str(branch_meta["pivr_branch_id"]),
        }


class FinalPixelPIVRReadWriteModel(nn.Module):
    """One-encode Pixel-PIVR objective over aligned episode views."""

    def __init__(
        self,
        model: nn.Module,
        *,
        roi_side_tokens: int,
        image_start_id: int,
        image_end_id: int,
        max_sequence: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.roi_side_tokens = int(roi_side_tokens)
        self.image_start_id = int(image_start_id)
        self.image_end_id = int(image_end_id)
        self.max_sequence = int(max_sequence)
        self.vision_encode_calls = 0

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.vision_model.eval()
        self.model.mlp1.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode_global(self, sample: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        pixel_values = sample["pixel_values"].to(
            device=self.device, dtype=torch.bfloat16
        )
        image_grid_hws = torch.as_tensor(
            sample["image_grid_hws"], device=self.device, dtype=torch.int32
        )
        if image_grid_hws.ndim != 2 or tuple(image_grid_hws.shape) != (1, 2):
            raise RuntimeError(
                "Final Pixel-PIVR requires exactly one real global image per episode; "
                f"found grid {tuple(image_grid_hws.shape)}"
            )
        visual = self.model.extract_feature(pixel_values, image_grid_hws)
        if len(visual) != 1:
            raise RuntimeError(f"Expected one visual feature row, found {len(visual)}")
        projected = self.model.mlp1(visual[0]).detach()
        self.vision_encode_calls += 1
        return projected, image_grid_hws[0]

    def phase_loss(
        self,
        sample: Mapping[str, Any],
        projected_global: torch.Tensor,
        patch_grid_hw: torch.Tensor,
        *,
        phase: str,
        point_xy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if phase not in PHASES:
            raise ValueError(f"Unknown Pixel-PIVR phase: {phase}")
        input_ids = sample["input_ids"].to(self.device)
        labels = sample["labels"].to(self.device)
        position_ids = sample["position_ids"].to(self.device)
        projected = projected_global
        roi_tokens = 0
        if phase == "branch":
            if point_xy is None:
                raise ValueError("Branch phase requires a point")
            roi = extract_projected_roi_unpadded(
                projected_global,
                patch_grid_hw,
                point_xy,
                side_tokens=self.roi_side_tokens,
            )
            roi_tokens = int(roi.features.shape[0])
            input_ids, labels, position_ids = insert_virtual_roi_image(
                input_ids=input_ids,
                labels=labels,
                position_ids=position_ids,
                image_token_id=int(self.model.image_token_index),
                image_start_id=self.image_start_id,
                image_end_id=self.image_end_id,
                roi_token_count=roi_tokens,
            )
            projected = torch.cat((projected_global, roi.features), dim=0)
        if int(input_ids.numel()) > self.max_sequence:
            raise RuntimeError(
                f"{phase} sequence exceeds budget: {input_ids.numel()} > {self.max_sequence}"
            )

        batched_ids = input_ids.unsqueeze(0)
        selected = batched_ids.eq(int(self.model.image_token_index))
        if int(selected.sum().item()) != int(projected.shape[0]):
            raise RuntimeError(
                f"{phase} visual placeholder mismatch: "
                f"ids={int(selected.sum())}, projected={int(projected.shape[0])}"
            )
        embeddings = self.model.language_model.get_input_embeddings()(batched_ids)
        flat = embeddings.reshape(-1, embeddings.shape[-1]).clone()
        flat[selected.reshape(-1)] = projected
        embeddings = flat.view_as(embeddings)
        causal = base_causal_lm(self.model.language_model)
        outputs = causal.model(
            input_ids=None,
            inputs_embeds=embeddings,
            attention_mask=None,
            position_ids=position_ids.unsqueeze(0),
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        loss = chunked_ce(
            outputs.last_hidden_state, labels.unsqueeze(0), causal.lm_head.weight
        )
        supervised_tokens = int(labels.ne(-100).sum().item())
        return loss, {
            "loss": float(loss.detach().float()),
            "sequence_tokens": float(input_ids.numel()),
            "supervised_tokens": float(supervised_tokens),
            "global_visual_tokens": float(projected_global.shape[0]),
            "roi_visual_tokens": float(roi_tokens),
        }
