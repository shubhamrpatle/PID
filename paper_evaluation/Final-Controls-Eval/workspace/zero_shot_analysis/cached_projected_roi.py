"""Cached post-projector ROI support for point-indexed PIVR.

The global image is encoded once by MoonViT and LA's released projector.  For
Round-2 records, a fixed spatial window is gathered from the resulting
decoder-width grid and inserted as a virtual second image before Qwen.  This
concentrates decoder attention without a second pixel/MoonViT encode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from scripts.train_pointrelay_pbd_lora_pilot import chunked_ce

from .pointrelay_pbd import base_causal_lm


IGNORE_TOKEN_ID = -100
ROUND2_ROUTE = "point_indexed_visual_reentry"


@dataclass(frozen=True)
class ProjectedROI:
    features: torch.Tensor
    merged_grid_hw: tuple[int, int]
    center_rc: tuple[int, int]
    window_yx: tuple[int, int]


def _normalized_grid_index(value: float, size: int) -> int:
    if size <= 0:
        raise ValueError(f"Invalid projected-grid size: {size}")
    clipped = max(0.0, min(1000.0, float(value)))
    return min(size - 1, int(clipped * size / 1000.0))


def extract_projected_roi(
    projected_global: torch.Tensor,
    patch_grid_hw: Sequence[int] | torch.Tensor,
    point_normalized_xy: Sequence[float] | torch.Tensor,
    *,
    side_tokens: int,
) -> ProjectedROI:
    """Gather a fixed point-centred window from LA's row-major projected grid."""
    if projected_global.ndim != 2:
        raise ValueError(
            f"Expected [tokens, hidden] projected features, got {projected_global.shape}"
        )
    patch_hw = torch.as_tensor(patch_grid_hw, dtype=torch.long).flatten()
    if patch_hw.numel() != 2:
        raise ValueError(f"Expected a two-value patch grid, got {patch_hw.tolist()}")
    patch_h, patch_w = (int(value) for value in patch_hw.tolist())
    if patch_h % 2 or patch_w % 2:
        raise ValueError(f"MoonViT 2x2 merge requires an even grid, got {(patch_h, patch_w)}")
    grid_h, grid_w = patch_h // 2, patch_w // 2
    if int(projected_global.shape[0]) != grid_h * grid_w:
        raise ValueError(
            "Projected-token/grid mismatch: "
            f"{projected_global.shape[0]} != {grid_h}*{grid_w}"
        )
    side = int(side_tokens)
    if side <= 0:
        raise ValueError(f"ROI side must be positive, got {side}")
    point = torch.as_tensor(point_normalized_xy, dtype=torch.float32).flatten()
    if point.numel() != 2:
        raise ValueError(f"Expected normalized (x, y), got {point.tolist()}")

    row = _normalized_grid_index(float(point[1]), grid_h)
    col = _normalized_grid_index(float(point[0]), grid_w)
    grid = projected_global.view(grid_h, grid_w, projected_global.shape[-1])

    # Very narrow source images are padded by border replication.  Normal
    # target-1024 records are at least 12x12 and take the no-padding path.
    pad_top = max(0, (side - grid_h) // 2)
    pad_bottom = max(0, side - grid_h - pad_top)
    pad_left = max(0, (side - grid_w) // 2)
    pad_right = max(0, side - grid_w - pad_left)
    if pad_top or pad_bottom or pad_left or pad_right:
        channels_first = grid.permute(2, 0, 1).unsqueeze(0)
        channels_first = F.pad(
            channels_first,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="replicate",
        )
        grid = channels_first.squeeze(0).permute(1, 2, 0).contiguous()
        row += pad_top
        col += pad_left
        grid_h, grid_w = int(grid.shape[0]), int(grid.shape[1])

    start_y = min(max(row - side // 2, 0), grid_h - side)
    start_x = min(max(col - side // 2, 0), grid_w - side)
    roi = grid[start_y : start_y + side, start_x : start_x + side]
    if tuple(roi.shape[:2]) != (side, side):
        raise RuntimeError(f"Incomplete projected ROI: {tuple(roi.shape)}")
    return ProjectedROI(
        features=roi.reshape(side * side, projected_global.shape[-1]).contiguous(),
        merged_grid_hw=(grid_h, grid_w),
        center_rc=(row, col),
        window_yx=(start_y, start_x),
    )


def extract_projected_roi_unpadded(
    projected_global: torch.Tensor,
    patch_grid_hw: Sequence[int] | torch.Tensor,
    point_normalized_xy: Sequence[float] | torch.Tensor,
    *,
    side_tokens: int,
) -> ProjectedROI:
    """Gather up to ``side_tokens`` per axis without synthesizing border rows."""
    if projected_global.ndim != 2:
        raise ValueError(
            f"Expected [tokens, hidden] projected features, got {projected_global.shape}"
        )
    patch_hw = torch.as_tensor(patch_grid_hw, dtype=torch.long).flatten()
    if patch_hw.numel() != 2:
        raise ValueError(f"Expected a two-value patch grid, got {patch_hw.tolist()}")
    patch_h, patch_w = (int(value) for value in patch_hw.tolist())
    if patch_h % 2 or patch_w % 2:
        raise ValueError(f"MoonViT 2x2 merge requires an even grid, got {(patch_h, patch_w)}")
    grid_h, grid_w = patch_h // 2, patch_w // 2
    if int(projected_global.shape[0]) != grid_h * grid_w:
        raise ValueError(
            "Projected-token/grid mismatch: "
            f"{projected_global.shape[0]} != {grid_h}*{grid_w}"
        )
    side = int(side_tokens)
    if side <= 0:
        raise ValueError(f"ROI side must be positive, got {side}")
    point = torch.as_tensor(point_normalized_xy, dtype=torch.float32).flatten()
    if point.numel() != 2:
        raise ValueError(f"Expected normalized (x, y), got {point.tolist()}")

    row = _normalized_grid_index(float(point[1]), grid_h)
    col = _normalized_grid_index(float(point[0]), grid_w)
    window_h = min(side, grid_h)
    window_w = min(side, grid_w)
    start_y = min(max(row - window_h // 2, 0), grid_h - window_h)
    start_x = min(max(col - window_w // 2, 0), grid_w - window_w)
    grid = projected_global.view(grid_h, grid_w, projected_global.shape[-1])
    roi = grid[start_y : start_y + window_h, start_x : start_x + window_w]
    if tuple(roi.shape[:2]) != (window_h, window_w):
        raise RuntimeError(f"Incomplete projected ROI: {tuple(roi.shape)}")
    return ProjectedROI(
        features=roi.reshape(window_h * window_w, projected_global.shape[-1]).contiguous(),
        merged_grid_hw=(grid_h, grid_w),
        center_rc=(row, col),
        window_yx=(start_y, start_x),
    )


def insert_virtual_roi_image(
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    position_ids: torch.Tensor,
    image_token_id: int,
    image_start_id: int,
    image_end_id: int,
    roi_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Insert ``<img> ROI_CONTEXT... </img>`` after the real global image."""
    if input_ids.ndim != 1 or labels.ndim != 1 or position_ids.ndim != 1:
        raise ValueError("Cached ROI insertion expects unbatched one-dimensional tensors")
    selected = torch.where(input_ids.eq(int(image_token_id)))[0]
    if selected.numel() == 0:
        raise RuntimeError("Training record contains no global image placeholders")
    last_visual = int(selected[-1].item())
    if last_visual + 1 >= input_ids.numel() or int(input_ids[last_visual + 1]) != int(
        image_end_id
    ):
        raise RuntimeError("Global image placeholders are not followed by </img>")
    insert_at = last_visual + 2
    count = int(roi_token_count)
    virtual_ids = torch.cat(
        (
            input_ids.new_tensor([int(image_start_id)]),
            input_ids.new_full((count,), int(image_token_id)),
            input_ids.new_tensor([int(image_end_id)]),
        )
    )
    ignored = labels.new_full((count + 2,), IGNORE_TOKEN_ID)
    insertion_position = int(position_ids[insert_at - 1].item()) + 1
    virtual_positions = torch.arange(
        insertion_position,
        insertion_position + count + 2,
        dtype=position_ids.dtype,
        device=position_ids.device,
    )
    suffix_positions = position_ids[insert_at:].clone()
    suffix_positions[suffix_positions >= insertion_position] += count + 2
    return (
        torch.cat((input_ids[:insert_at], virtual_ids, input_ids[insert_at:])),
        torch.cat((labels[:insert_at], ignored, labels[insert_at:])),
        torch.cat(
            (position_ids[:insert_at], virtual_positions, suffix_positions)
        ),
    )


class CachedROIMetadataDataset(Dataset):
    """Attach route/point tensors to LA's exact existing tokenized records."""

    def __init__(self, base: Dataset) -> None:
        self.base = base
        if not hasattr(base, "lazy_loader") or not hasattr(base, "active_indices"):
            raise TypeError("Expected LazySupervisedDatasetMTP-compatible dataset")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base[index])
        source_index = self.base.active_indices[index]
        row = self.base.lazy_loader[source_index]
        meta = dict(row.get("meta") or {})
        enabled = str(meta.get("pivr_route") or "") == ROUND2_ROUTE
        point = meta.get("pivr_global_point") if enabled else (0, 0)
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise RuntimeError(
                f"Cached-ROI Round-2 record {index} has no global point: {point}"
            )
        raw_images = row.get("image")
        if isinstance(raw_images, list) or not isinstance(raw_images, str):
            raise RuntimeError(
                f"Cached-ROI records must contain exactly one real image: index={index}"
            )
        sample["cached_roi_enabled"] = torch.tensor(enabled, dtype=torch.bool)
        sample["cached_roi_point"] = torch.tensor(point, dtype=torch.float32)
        return sample


class CachedProjectedROIPilotModel(nn.Module):
    """Qwen-LoRA training wrapper with a frozen cached projected ROI."""

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

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.vision_model.eval()
        self.model.mlp1.eval()
        return self

    def forward(self, sample: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        device = next(self.model.parameters()).device
        input_ids = sample["input_ids"].to(device)
        labels = sample["labels"].to(device)
        position_ids = sample["position_ids"].to(device)
        pixel_values = sample["pixel_values"].to(device=device, dtype=torch.bfloat16)
        image_grid_hws = torch.as_tensor(
            sample["image_grid_hws"], device=device, dtype=torch.int32
        )
        if image_grid_hws.ndim != 2 or tuple(image_grid_hws.shape) != (1, 2):
            raise RuntimeError(
                "Cached projected ROI requires exactly one real global image; "
                f"found grid shape {tuple(image_grid_hws.shape)}"
            )
        with torch.no_grad():
            visual = self.model.extract_feature(pixel_values, image_grid_hws)
            if len(visual) != 1:
                raise RuntimeError(f"Expected one global visual feature list, found {len(visual)}")
            projected_global = self.model.mlp1(visual[0])

        roi_enabled = bool(torch.as_tensor(sample["cached_roi_enabled"]).item())
        projected = projected_global
        roi_count = 0
        if roi_enabled:
            roi = extract_projected_roi(
                projected_global,
                image_grid_hws[0],
                sample["cached_roi_point"],
                side_tokens=self.roi_side_tokens,
            )
            roi_count = int(roi.features.shape[0])
            input_ids, labels, position_ids = insert_virtual_roi_image(
                input_ids=input_ids,
                labels=labels,
                position_ids=position_ids,
                image_token_id=int(self.model.image_token_index),
                image_start_id=self.image_start_id,
                image_end_id=self.image_end_id,
                roi_token_count=roi_count,
            )
            projected = torch.cat((projected_global, roi.features), dim=0)
        if int(input_ids.numel()) > self.max_sequence:
            raise RuntimeError(
                f"Cached-ROI sequence exceeds budget: {input_ids.numel()} > {self.max_sequence}"
            )

        batched_ids = input_ids.unsqueeze(0)
        selected = batched_ids.eq(int(self.model.image_token_index))
        if int(selected.sum().item()) != int(projected.shape[0]):
            raise RuntimeError(
                "Cached-ROI visual placeholder mismatch: "
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
        native = chunked_ce(
            outputs.last_hidden_state, labels.unsqueeze(0), causal.lm_head.weight
        )
        zero = 0.0
        return native, {
            "loss": float(native.detach().float()),
            "native_loss": float(native.detach().float()),
            "point_loss": zero,
            "boundary_loss": zero,
            "eos_loss": zero,
            "none_loss": zero,
            "history_loss": zero,
            "cached_roi_records": float(roi_enabled),
            "cached_roi_tokens": float(roi_count),
        }
