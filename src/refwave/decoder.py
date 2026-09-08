"""Shared-prefix sequential/wave PBD6 geometry decoding for RefWave."""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import torch
from .lora import base_causal_lm
from .shared_prefix import SharedPrefixCache, legacy_key_values


@dataclass(frozen=True)
class AddressedCrop:
    """One instance address; global-only RefWave leaves the crop field empty."""

    address_id: str
    label: str
    crop: Any
    question: str
    point_normalized_xy: tuple[int, int] | None = None


def image_patch_lengths(image_grid_hws: Any) -> list[int]:
    grid = torch.as_tensor(image_grid_hws, dtype=torch.long)
    if grid.ndim != 2 or grid.shape[1] not in (2, 3):
        raise ValueError(f"Unexpected image_grid_hws shape: {tuple(grid.shape)}")
    return [int(row.prod().item()) for row in grid]


class GeometryDecoder:
    """Decode exactly one native PBD6 HBB for every known point address."""

    def __init__(
        self,
        worker: Any,
        *,
        image_token_limit: int = 6000,
        prefix_cache_mode: str = "shared",
        geometry_prefix_mode: str = "box_only",
    ) -> None:
        self.worker = worker
        self.model = worker.model
        self.processor = worker.processor
        self.tokenizer = worker.tokenizer
        self.device = worker.device
        self.dtype = worker.dtype
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None or not hasattr(image_processor, "in_token_limit"):
            raise RuntimeError("LocateAnything processor has no in_token_limit")
        image_processor.in_token_limit = int(image_token_limit)
        self.tokenizer.padding_side = "left"
        self.mask_id = int(self.model.token_ids["default_mask_token_id"])
        self.n_future_tokens = 6
        if prefix_cache_mode not in {"recompute", "shared"}:
            raise ValueError("prefix_cache_mode must be recompute or shared")
        self.prefix_cache_mode = prefix_cache_mode
        if geometry_prefix_mode not in {"ref", "box_only"}:
            raise ValueError("geometry_prefix_mode must be ref or box_only")
        self.geometry_prefix_mode = geometry_prefix_mode
        self.image_end_id = int(
            self.tokenizer.convert_tokens_to_ids(self.processor.image_end_token)
        )

    def _left_pad_rows(self, rows: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if not rows or any((int(row.numel()) == 0 for row in rows)):
            raise RuntimeError("Every address must have a nonempty suffix")
        maximum = max((int(row.numel()) for row in rows))
        pad_id = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        input_ids = rows[0].new_full((len(rows), maximum), pad_id)
        attention = rows[0].new_zeros((len(rows), maximum))
        for row_index, row in enumerate(rows):
            input_ids[row_index, -int(row.numel()) :] = row
            attention[row_index, -int(row.numel()) :] = 1
        return (input_ids, attention)

    def _prepare_geometry_inputs(
        self,
        branches: Sequence[AddressedCrop],
        *,
        input_dtype: torch.dtype,
        attention_dtype: torch.dtype,
        device: torch.device,
        prompt_last_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.geometry_prefix_mode == "box_only":
            if prompt_last_ids is None or tuple(prompt_last_ids.shape) != (len(branches),):
                raise RuntimeError("Box-only PBD6 requires one prompt-final token per branch")
            geometry_ids = torch.cat(
                (
                    prompt_last_ids.to(device=device, dtype=input_dtype).unsqueeze(1),
                    torch.full(
                        (len(branches), self.n_future_tokens - 1),
                        self.mask_id,
                        dtype=input_dtype,
                        device=device,
                    ),
                ),
                dim=1,
            )
            geometry_attention = torch.ones(
                (len(branches), self.n_future_tokens), dtype=attention_dtype, device=device
            )
            return (geometry_ids, geometry_attention)
        reference_rows = [
            self.tokenizer.encode(f"<ref>{branch.label}</ref>", add_special_tokens=False)
            for branch in branches
        ]
        if any((not row for row in reference_rows)):
            raise RuntimeError("A point-address reference tokenized to an empty sequence")
        maximum_reference = max((len(row) for row in reference_rows))
        pad_id = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        reference_ids = torch.full(
            (len(branches), maximum_reference), pad_id, dtype=input_dtype, device=device
        )
        reference_attention = torch.zeros(
            (len(branches), maximum_reference), dtype=attention_dtype, device=device
        )
        for index, row in enumerate(reference_rows):
            reference_ids[index, -len(row) :] = torch.as_tensor(
                row, dtype=input_dtype, device=device
            )
            reference_attention[index, -len(row) :] = 1
        geometry_ids = torch.cat(
            (
                reference_ids,
                reference_ids[:, -1:].clone(),
                torch.full(
                    (len(branches), self.n_future_tokens - 1),
                    self.mask_id,
                    dtype=input_dtype,
                    device=device,
                ),
            ),
            dim=1,
        )
        geometry_attention = torch.cat(
            (
                reference_attention,
                torch.ones(
                    (len(branches), self.n_future_tokens), dtype=attention_dtype, device=device
                ),
            ),
            dim=1,
        )
        return (geometry_ids, geometry_attention)

    def _results_from_logits(
        self, logits: torch.Tensor, branches: Sequence[AddressedCrop], *, allow_none: bool
    ) -> list[dict[str, Any]]:
        box_start = int(self.model.token_ids["box_start_token_id"])
        box_end = int(self.model.token_ids["box_end_token_id"])
        none_id = int(self.model.token_ids["none_token_id"])
        coord_start = int(self.model.token_ids["coord_start_token_id"])
        coord_end = int(self.model.token_ids["coord_end_token_id"])
        results: list[dict[str, Any]] = []
        for index, branch in enumerate(branches):
            row = logits[index].float()
            if allow_none and row[1, none_id] >= row[1, coord_start : coord_end + 1].max():
                tokens = torch.as_tensor(
                    [box_start, none_id, box_end], dtype=torch.long, device=logits.device
                )
                block_type = "empty_box"
            else:
                coordinates = []
                for lane in range(1, 5):
                    lane_logits = row[lane, coord_start : coord_end + 1]
                    (values, indices) = torch.topk(lane_logits, k=5)
                    coordinate = int(
                        torch.round((indices.float() * values.softmax(dim=0)).sum()).item()
                    )
                    coordinates.append(max(0, min(1000, coordinate)))
                (x1, x2) = sorted((coordinates[0], coordinates[2]))
                (y1, y2) = sorted((coordinates[1], coordinates[3]))
                tokens = torch.as_tensor(
                    [
                        box_start,
                        coord_start + x1,
                        coord_start + y1,
                        coord_start + x2,
                        coord_start + y2,
                        box_end,
                    ],
                    dtype=torch.long,
                    device=logits.device,
                )
                block_type = "coord_box"
            block = self.tokenizer.decode(tokens, skip_special_tokens=False)
            results.append(
                {
                    "address_id": branch.address_id,
                    "label": branch.label,
                    "block_type": block_type,
                    "block_tokens": [int(value) for value in tokens.tolist()],
                    "answer": (
                        block
                        if self.geometry_prefix_mode == "box_only"
                        else f"<ref>{branch.label}</ref>{block}"
                    ),
                }
            )
        return results

    def _decode_shared_atomic_blocks(
        self,
        prefix_ids: torch.Tensor,
        suffix_rows: Sequence[torch.Tensor],
        projected_locals: Sequence[torch.Tensor],
        prefix_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
        branches: Sequence[AddressedCrop],
        *,
        allow_none: bool,
    ) -> list[dict[str, Any]]:
        (suffix_ids, suffix_attention) = self._left_pad_rows(suffix_rows)
        suffix_ids = suffix_ids.to(self.device)
        suffix_attention = suffix_attention.to(self.device)
        batch_size = len(branches)
        prefix_attention = torch.ones(
            (batch_size, int(prefix_ids.numel())),
            dtype=suffix_attention.dtype,
            device=self.device,
        )
        context_attention = torch.cat((prefix_attention, suffix_attention), dim=1)
        context_positions = context_attention.long().cumsum(-1) - 1
        context_positions.masked_fill_(context_attention == 0, 1)
        projected = torch.cat(list(projected_locals), dim=0)
        if int(suffix_ids.eq(int(self.model.image_token_index)).sum()) != int(
            projected.shape[0]
        ):
            raise RuntimeError("Shared-prefix local placeholder count differs")
        causal = base_causal_lm(self.model.language_model)
        suffix_outputs = causal.model(
            input_ids=suffix_ids,
            visual_features=projected if projected.shape[0] else None,
            image_token_index=int(self.model.image_token_index) if projected.shape[0] else None,
            attention_mask=context_attention,
            position_ids=context_positions[:, -int(suffix_ids.shape[1]) :],
            past_key_values=SharedPrefixCache(prefix_key_values),
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        (geometry_ids, geometry_attention) = self._prepare_geometry_inputs(
            branches,
            input_dtype=suffix_ids.dtype,
            attention_dtype=suffix_attention.dtype,
            device=suffix_ids.device,
            prompt_last_ids=suffix_ids[:, -1],
        )
        full_attention = torch.cat((context_attention, geometry_attention), dim=1)
        positions = full_attention.long().cumsum(-1) - 1
        positions.masked_fill_(full_attention == 0, 1)
        positions[:, -self.n_future_tokens :] -= 1
        outputs = causal.model(
            input_ids=geometry_ids,
            visual_features=None,
            image_token_index=None,
            attention_mask=full_attention,
            position_ids=positions[:, -int(geometry_ids.shape[1]) :],
            past_key_values=suffix_outputs.past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state[:, -self.n_future_tokens :, :]
        logits = torch.nn.functional.linear(hidden, causal.lm_head.weight)
        return self._results_from_logits(logits, branches, allow_none=allow_none)
