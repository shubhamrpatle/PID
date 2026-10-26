"""Marked residual visual re-entry for LocateAnything's native PBD decoder.

The controller is a cap-free recurrent visual state, not an object-slot head. A
single MoonViT pass supplies a fine latent address field. At every native PBD
transition, the current Qwen state reads one residual location from that field.
After PBD accepts a geometry block, its complete spatial footprint is written
back as a soft mark. The remaining marked-field evidence then conditions both
the next geometry block and LA's native continue/stop decision.

This pilot deliberately preserves the released tokenizer, six-token HBB block,
Qwen decoder, LM head, and autoregressive grammar. Its object count is bounded
only by the language-model sequence budget; there is no top-k proposal list,
parallel slot bank, crop loop, or decoder copy.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .dense_address_field_pbd import DenseAddressFieldPBD
from .pointrelay_pbd import (
    RelayContext,
    base_causal_lm,
    configure_lora_only,
    restore_trainable_state,
)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(max(1e-6, float(value))))


class MarkedResidualAddressPBD(DenseAddressFieldPBD):
    """Hierarchical fine-field read, geometry-mark write-back, and learned stop."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vision_size: int,
        relay_layer: int,
        token_ids: Mapping[str, int],
        field_dim: int = 128,
        subpatch_dim: int = 32,
        subpatch_grid: int = 4,
        temperature: float = 0.12,
        target_sigma: float = 0.70,
        initial_gate_logit: float = -1.5,
        initial_coverage_strength: float = 6.0,
        residual_floor: float = 0.002,
        footprint_halo_cells: float = 0.35,
        initial_structure_scale: float = 1.5,
        absence_loss_weight: float = 2.0,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            vision_size=vision_size,
            relay_layer=relay_layer,
            token_ids=token_ids,
            field_dim=field_dim,
            subpatch_dim=subpatch_dim,
            subpatch_grid=subpatch_grid,
            temperature=temperature,
            target_sigma=target_sigma,
            initial_gate_logit=initial_gate_logit,
            initial_coverage_strength=initial_coverage_strength,
        )
        if not 0.0 < residual_floor < 1.0:
            raise ValueError("residual_floor must be in (0, 1)")
        if footprint_halo_cells < 0.0:
            raise ValueError("footprint_halo_cells must be non-negative")
        if absence_loss_weight <= 0.0:
            raise ValueError("absence_loss_weight must be positive")

        self.residual_floor = float(residual_floor)
        self.footprint_halo_cells = float(footprint_halo_cells)
        self.absence_loss_weight = float(absence_loss_weight)

        # A coarse semantic route first selects MoonViT patches; a fine route
        # then resolves one of the retained subpatch cells inside those patches.
        self.coarse_query_proj = nn.Linear(self.hidden_size, self.field_dim, bias=False)
        self.coarse_key_proj = nn.Linear(self.vision_size, self.field_dim, bias=False)
        self.existence_mlp = nn.Sequential(
            nn.Linear(2 * self.field_dim + 2, self.field_dim),
            nn.GELU(),
            nn.Linear(self.field_dim, 1),
        )
        self.structure_scale_logit = nn.Parameter(
            torch.tensor(_inverse_softplus(initial_structure_scale))
        )
        self.none_scale_logit = nn.Parameter(
            torch.tensor(_inverse_softplus(initial_structure_scale))
        )
        nn.init.xavier_uniform_(self.coarse_query_proj.weight)
        nn.init.xavier_uniform_(self.coarse_key_proj.weight)
        nn.init.xavier_uniform_(self.existence_mlp[0].weight)
        nn.init.zeros_(self.existence_mlp[0].bias)
        nn.init.normal_(self.existence_mlp[2].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.existence_mlp[2].bias)

        # These directions are derived from the released LM head at install
        # time. They do not add output classes or change the vocabulary.
        self.register_buffer(
            "continue_logit_direction", torch.zeros(self.hidden_size), persistent=True
        )
        self.register_buffer(
            "none_logit_direction", torch.zeros(self.hidden_size), persistent=True
        )

        self._target_boxes: tuple[tuple[int, int, int, int] | None, ...] = ()
        self._predecessor_kinds: tuple[str, ...] = ()
        self._pending_structure: tuple[
            tuple[tuple[int, int, int, int, int, int], str, torch.Tensor], ...
        ] = ()
        self.observed_box_count = 0
        self.observed_point_count = 0
        self.observed_ref_count = 0

    def _configure_output_directions(self, model: nn.Module) -> None:
        causal = base_causal_lm(model.language_model)
        weight = causal.lm_head.weight.detach().float()

        def stable_direction(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
            difference = positive - negative
            return difference / difference.square().sum().clamp_min(1e-6)

        box = weight[int(self.token_ids["box_start_token_id"])]
        stop_ids = [
            int(self.token_ids["ref_start_token_id"]),
            int(self.token_ids["im_end_token_id"]),
            int(self.token_ids["null_token_id"]),
        ]
        stop = weight[stop_ids].mean(dim=0)
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        coordinate = weight[coord_start : coord_end + 1].mean(dim=0)
        none = weight[int(self.token_ids["none_token_id"])]
        self.continue_logit_direction.copy_(
            stable_direction(box, stop).to(self.continue_logit_direction)
        )
        self.none_logit_direction.copy_(
            stable_direction(none, coordinate).to(self.none_logit_direction)
        )

    def _training_transitions(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        context: RelayContext,
    ) -> tuple[RelayContext, tuple[int, ...], tuple[float, ...], tuple[bool, ...]]:
        if input_ids.ndim != 2 or labels.ndim != 2 or input_ids.shape != labels.shape:
            raise ValueError("Marked residual transitions require matching [1, sequence] ids")
        ids = input_ids[0]
        targets_tensor = labels[0]
        mask_id = int(self.token_ids["default_mask_token_id"])
        ref_end = int(self.token_ids["ref_end_token_id"])
        box_end = int(self.token_ids["box_end_token_id"])
        box_start = int(self.token_ids["box_start_token_id"])
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])

        rows: list[tuple[int, int, int, int, int, int]] = []
        target_indices: list[int] = []
        existence_targets: list[float] = []
        reset_flags: list[bool] = []
        target_boxes: list[tuple[int, int, int, int] | None] = []
        predecessor_kinds: list[str] = []
        for start in range(max(0, int(ids.numel()) - 6)):
            if start + 6 >= int(targets_tensor.numel()):
                break
            predecessor = int(ids[start])
            if predecessor not in {ref_end, box_end}:
                continue
            if not bool(ids[start + 1 : start + 6].eq(mask_id).all()):
                continue
            block = [int(value) for value in targets_tensor[start + 1 : start + 7].tolist()]
            numeric = (
                len(block) == 6
                and block[0] == box_start
                and block[5] == box_end
                and all(coord_start <= value <= coord_end for value in block[1:5])
            )
            rows.append(tuple(range(start, start + 6)))
            reset_flags.append(predecessor == ref_end)
            predecessor_kinds.append("ref" if predecessor == ref_end else "box")
            if numeric:
                box = tuple(int(value - coord_start) for value in block[1:5])
                center = (
                    max(0, min(1000, int(round((box[0] + box[2]) / 2.0)))),
                    max(0, min(1000, int(round((box[1] + box[3]) / 2.0)))),
                )
                target_indices.append(self._center_index(center))
                target_boxes.append(box)
                existence_targets.append(1.0)
            else:
                target_indices.append(-1)
                target_boxes.append(None)
                existence_targets.append(0.0)

        self._target_boxes = tuple(target_boxes)
        self._predecessor_kinds = tuple(predecessor_kinds)
        transition_context = RelayContext(
            tuple(rows), (), context.image_columns, context.query_length, ()
        )
        return (
            transition_context,
            tuple(target_indices),
            tuple(existence_targets),
            tuple(reset_flags),
        )

    def set_training_context(
        self,
        context: RelayContext,
        image_grid_hws: Any | None = None,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> None:
        super().set_training_context(context, image_grid_hws, input_ids, labels)
        self._pending_structure = ()
        self.observed_box_count = 0
        self.observed_point_count = 0
        self.observed_ref_count = 0

    def set_inference_image_layout(self, input_ids: torch.Tensor, image_grid_hws: Any) -> None:
        super().set_inference_image_layout(input_ids, image_grid_hws)
        self._target_boxes = ()
        self._predecessor_kinds = ()
        self._pending_structure = ()
        self.observed_box_count = 0
        self.observed_point_count = 0
        self.observed_ref_count = 0

    def _inference_context(self, input_ids: torch.Tensor | None) -> RelayContext | None:
        context = super()._inference_context(input_ids)
        if context is None or input_ids is None:
            self._predecessor_kinds = ()
            return context
        predecessor = int(input_ids[0, context.rows[-1][0]])
        self._predecessor_kinds = (
            "ref"
            if predecessor == int(self.token_ids["ref_end_token_id"])
            else "box",
        )
        self._target_boxes = (None,)
        return context

    def _model_pre_hook(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if self._training_context:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        position_ids = kwargs.get("position_ids")
        self._inference_position_key = (
            tuple(int(value) for value in position_ids.detach().reshape(-1).tolist())
            if isinstance(position_ids, torch.Tensor)
            else ()
        )
        self._context = self._inference_context(input_ids)
        self._pending_structure = ()
        self._aux_loss = None
        return None

    def _box_support(
        self,
        box: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if self._field_positions is None or self._field_grid_hw is None:
            raise RuntimeError("Marked residual field is unavailable")
        positions = self._field_positions.to(device=device, dtype=dtype)
        grid_h, grid_w = self._field_grid_hw
        x1, y1, x2, y2 = [float(value) / 1000.0 for value in box]
        left, right = sorted((max(0.0, x1), min(1.0, x2)))
        top, bottom = sorted((max(0.0, y1), min(1.0, y2)))
        x = positions[:, 0]
        y = positions[:, 1]
        dx = (left - x).clamp_min(0.0) + (x - right).clamp_min(0.0)
        dy = (top - y).clamp_min(0.0) + (y - bottom).clamp_min(0.0)
        distance_sq = (dx * float(grid_w)).square() + (dy * float(grid_h)).square()
        halo = max(1e-3, self.footprint_halo_cells)
        support = torch.exp(-0.5 * distance_sq / (halo * halo))
        inside = (x >= left) & (x <= right) & (y >= top) & (y <= bottom)
        support = torch.where(inside, torch.ones_like(support), support)
        if float(support.max()) < 1e-6:
            center = (
                max(0, min(1000, int(round(500.0 * (left + right))))),
                max(0, min(1000, int(round(500.0 * (top + bottom))))),
            )
            support[self._center_index(center)] = 1.0
        return support.clamp(0.0, 1.0)

    def _point_support(self, point: Sequence[int], *, device: torch.device) -> torch.Tensor:
        if self._field_positions is None or self._field_grid_hw is None:
            raise RuntimeError("Marked residual field is unavailable")
        positions = self._field_positions.to(device=device, dtype=torch.float32)
        grid_h, grid_w = self._field_grid_hw
        center = positions.new_tensor([float(point[0]) / 1000.0, float(point[1]) / 1000.0])
        scale = positions.new_tensor([float(grid_w), float(grid_h)])
        distance_sq = ((positions - center) * scale).square().sum(dim=-1)
        return torch.exp(-0.5 * distance_sq / max(0.25, self.footprint_halo_cells) ** 2)

    def observe_generated_block(self, out_type: str, tokens: Any) -> None:
        """Write the geometry block accepted by PBD into residual visual state."""

        if self._training_context:
            return
        if out_type == "ref_object":
            self._coverage = None
            self.observed_ref_count += 1
        elif out_type in {"coord_box", "point_box"}:
            values = (
                [int(value) for value in tokens.detach().reshape(-1).tolist()]
                if isinstance(tokens, torch.Tensor)
                else [int(value.item() if hasattr(value, "item") else value) for value in tokens]
            )
            coord_start = int(self.token_ids["coord_start_token_id"])
            box_start = int(self.token_ids["box_start_token_id"])
            box_end = int(self.token_ids["box_end_token_id"])
            if not values or values[0] != box_start or values[-1] != box_end:
                return
            device = (
                self._field_positions.device
                if self._field_positions is not None
                else self.lane_gate_logits.device
            )
            if out_type == "coord_box" and len(values) == 6:
                support = self._box_support(
                    [value - coord_start for value in values[1:5]], device=device
                )
                self.observed_box_count += 1
                self.geometry_writeback_count += 1
            elif out_type == "point_box" and len(values) == 4:
                support = self._point_support(
                    [value - coord_start for value in values[1:3]], device=device
                )
                self.observed_point_count += 1
                self.geometry_writeback_count += 1
            else:
                return
            self._coverage = (
                support
                if self._coverage is None
                else torch.maximum(self._coverage.to(support), support)
            )
        self.last_diagnostics.update(
            {
                "marked_observed_boxes": float(self.observed_box_count),
                "marked_observed_points": float(self.observed_point_count),
                "marked_observed_refs": float(self.observed_ref_count),
                "marked_coverage_fraction": 0.0
                if self._coverage is None
                else float(self._coverage.gt(0.1).float().mean()),
            }
        )

    def _route(self, hidden_states: torch.Tensor, context: RelayContext) -> torch.Tensor:
        rows = tuple(row for row in context.rows if max(row) < int(hidden_states.shape[1]))
        field = self._field_hidden
        subpatch = self._subpatch_hidden
        positions = self._field_positions
        if (
            not rows
            or field is None
            or subpatch is None
            or positions is None
            or int(field.shape[0]) == 0
        ):
            self._pending_structure = ()
            return hidden_states

        self.route_call_count += 1
        route_dtype = self.query_proj.weight.dtype
        field = field.to(device=hidden_states.device, dtype=route_dtype)
        subpatch = subpatch.to(device=hidden_states.device, dtype=route_dtype)
        positions = positions.to(device=hidden_states.device, dtype=route_dtype)
        patches, cells = int(subpatch.shape[0]), int(subpatch.shape[1])
        if patches != int(field.shape[0]) or patches * cells != int(positions.shape[0]):
            raise RuntimeError("Marked residual patch/subcell field lost alignment")

        position_features = self.position_mlp(positions).float()
        semantic_keys = self.key_proj(field).float()
        semantic_values = self.value_proj(field).float()
        raw_keys = self.subpatch_key_proj(subpatch).float().reshape(-1, self.field_dim)
        raw_values = self.subpatch_value_proj(subpatch).float().reshape(-1, self.field_dim)
        fine_keys = F.normalize(
            semantic_keys[:, None, :].expand(-1, cells, -1).reshape(-1, self.field_dim)
            + raw_keys
            + position_features,
            dim=-1,
        )
        values = (
            semantic_values[:, None, :].expand(-1, cells, -1).reshape(-1, self.field_dim)
            + raw_values
            + position_features
        )
        coarse_positions = position_features.reshape(patches, cells, self.field_dim).mean(dim=1)
        coarse_keys = F.normalize(
            self.coarse_key_proj(field).float() + coarse_positions, dim=-1
        )
        field_cells = int(fine_keys.shape[0])
        coverage = (
            torch.zeros(field_cells, device=hidden_states.device, dtype=torch.float32)
            if self._coverage is None or int(self._coverage.numel()) != field_cells
            else self._coverage.to(device=hidden_states.device, dtype=torch.float32)
        )
        strength = self.coverage_strength().float()
        gates = torch.sigmoid(self.lane_gate_logits).to(hidden_states.dtype)
        updated = hidden_states.clone()
        route_losses: list[torch.Tensor] = []
        existence_losses: list[torch.Tensor] = []
        existence_probabilities: list[torch.Tensor] = []
        existence_labels: list[torch.Tensor] = []
        route_entropies: list[torch.Tensor] = []
        target_masses: list[torch.Tensor] = []
        pending: list[tuple[tuple[int, int, int, int, int, int], str, torch.Tensor]] = []

        for row_index, row in enumerate(rows):
            if row_index < len(self._reset_flags) and self._reset_flags[row_index]:
                coverage = torch.zeros_like(coverage)
            anchor = hidden_states[0, row[0]].to(route_dtype)
            fine_query = F.normalize(self.query_proj(anchor).float(), dim=-1)
            coarse_query = F.normalize(self.coarse_query_proj(anchor).float(), dim=-1)
            coarse_logits = torch.matmul(coarse_keys, coarse_query) / self.temperature
            local_logits = torch.matmul(fine_keys, fine_query) / self.temperature
            logits = (
                local_logits
                + coarse_logits.repeat_interleave(cells)
                - strength * coverage
            )
            probabilities = logits.softmax(dim=-1)
            selected = torch.matmul(probabilities, values)
            evidence_max = logits.max() / 10.0
            evidence_mean = (
                torch.logsumexp(logits, dim=0) - math.log(max(1, field_cells))
            ) / 10.0
            existence_input = torch.cat(
                [
                    fine_query,
                    F.normalize(selected, dim=-1),
                    evidence_max.reshape(1),
                    evidence_mean.reshape(1),
                ]
            )
            existence_logit = self.existence_mlp(existence_input).reshape(())
            existence_probability = torch.sigmoid(existence_logit)
            existence_probabilities.append(existence_probability)
            predecessor = (
                self._predecessor_kinds[row_index]
                if row_index < len(self._predecessor_kinds)
                else "box"
            )
            pending.append((row, predecessor, existence_logit))

            target_box = (
                self._target_boxes[row_index]
                if self._training_context and row_index < len(self._target_boxes)
                else None
            )
            if target_box is not None:
                support = self._box_support(target_box, device=logits.device)
                target = support / support.sum().clamp_min(1e-12)
                route_losses.append(-(target * logits.log_softmax(dim=-1)).sum())
                target_masses.append((probabilities * support).sum())
                coverage = torch.maximum(coverage, support)

            if self._training_context and row_index < len(self._existence_targets):
                label = existence_logit.new_tensor(self._existence_targets[row_index])
                weight = label.new_tensor(
                    1.0 if float(label) >= 0.5 else self.absence_loss_weight
                )
                existence_labels.append(label)
                existence_losses.append(
                    weight * F.binary_cross_entropy_with_logits(existence_logit, label)
                )

            lane_values = selected.unsqueeze(0) + self.lane_queries.float()
            lane_delta = self.value_up(F.silu(lane_values.to(route_dtype)))
            lane_delta = F.layer_norm(
                lane_delta.float(), (self.hidden_size,)
            ).to(hidden_states.dtype)
            for lane, position in enumerate(row[1:5]):
                updated[0, position] = updated[0, position] + gates[lane] * lane_delta[lane]
            route_entropies.append(
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            )

        route_loss = torch.stack(route_losses).mean() if route_losses else None
        existence_loss = (
            torch.stack(existence_losses).mean() if existence_losses else None
        )
        if route_loss is not None and existence_loss is not None:
            self._aux_loss = route_loss + existence_loss
        else:
            self._aux_loss = route_loss if route_loss is not None else existence_loss
        self._coverage = coverage.detach()
        self._pending_structure = tuple(pending)

        mean_existence = (
            torch.stack(existence_probabilities).mean()
            if existence_probabilities
            else hidden_states.new_zeros(())
        )
        if existence_labels:
            labels_tensor = torch.stack(existence_labels)
            probabilities_tensor = torch.stack(existence_probabilities[: len(existence_labels)])
            accuracy = probabilities_tensor.ge(0.5).eq(labels_tensor.ge(0.5)).float().mean()
            positive = probabilities_tensor[labels_tensor.ge(0.5)]
            negative = probabilities_tensor[labels_tensor.lt(0.5)]
        else:
            accuracy = mean_existence.new_zeros(())
            positive = negative = mean_existence.new_empty(0)
        entropy = (
            torch.stack(route_entropies).mean()
            if route_entropies
            else hidden_states.new_zeros(())
        )
        self.last_diagnostics = {
            "marked_field_cells": float(field_cells),
            "marked_route_loss": 0.0 if route_loss is None else float(route_loss.detach()),
            "marked_route_entropy_fraction": float(
                (entropy / math.log(max(2, field_cells))).detach()
            ),
            "marked_target_mass": 0.0
            if not target_masses
            else float(torch.stack(target_masses).mean().detach()),
            "marked_existence_loss": 0.0
            if existence_loss is None
            else float(existence_loss.detach()),
            "marked_existence_probability": float(mean_existence.detach()),
            "marked_existence_accuracy": float(accuracy.detach()),
            "marked_positive_existence": 0.0
            if int(positive.numel()) == 0
            else float(positive.mean().detach()),
            "marked_negative_existence": 0.0
            if int(negative.numel()) == 0
            else float(negative.mean().detach()),
            "marked_lane_gate": float(gates.mean().detach()),
            "marked_coverage_strength": float(strength.detach()),
            "marked_coverage_fraction": float(coverage.gt(0.1).float().mean().detach()),
            "marked_structure_scale": float(F.softplus(self.structure_scale_logit).detach()),
            "marked_none_scale": float(F.softplus(self.none_scale_logit).detach()),
            "marked_observed_boxes": float(self.observed_box_count),
            "marked_observed_points": float(self.observed_point_count),
            "marked_observed_refs": float(self.observed_ref_count),
        }
        return updated

    def _final_norm_hook(
        self, _module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor
    ) -> torch.Tensor:
        if not self._pending_structure or not isinstance(output, torch.Tensor):
            return output
        updated = output.clone()
        continue_scale = F.softplus(self.structure_scale_logit).to(output.dtype)
        none_scale = F.softplus(self.none_scale_logit).to(output.dtype)
        continue_direction = self.continue_logit_direction.to(output)
        none_direction = self.none_logit_direction.to(output)
        for row, predecessor, existence_logit in self._pending_structure:
            signal = existence_logit.float().clamp(-4.0, 4.0).to(output.dtype)
            if predecessor == "box" and row[0] < int(output.shape[1]):
                updated[0, row[0]] = (
                    updated[0, row[0]] + continue_scale * signal * continue_direction
                )
            elif predecessor == "ref" and row[1] < int(output.shape[1]):
                # Both present and absent classes begin with <box>. The second
                # lane chooses a coordinate versus the released None token.
                updated[0, row[1]] = (
                    updated[0, row[1]] - none_scale * signal * none_direction
                )
        return updated

    def finish_forward(self) -> None:
        super().finish_forward()
        self._target_boxes = ()
        self._predecessor_kinds = ()
        self._pending_structure = ()

    def clear_context(self) -> None:
        super().clear_context()
        self._target_boxes = ()
        self._predecessor_kinds = ()
        self._pending_structure = ()

    def install(self, model: nn.Module) -> None:
        self._configure_output_directions(model)
        super().install(model)
        causal = base_causal_lm(model.language_model)
        final_norm = getattr(causal.model, "norm", None)
        if final_norm is None:
            raise RuntimeError("Marked residual routing requires Qwen final norm")
        self._handles.append(final_norm.register_forward_hook(self._final_norm_hook))


def load_marked_residual_address_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "marked_residual_address":
        raise ValueError(f"Not a marked-residual checkpoint: {config.get('arm')!r}")
    configure_lora_only(model, int(config["lora_rank"]))
    controller = MarkedResidualAddressPBD(
        hidden_size=int(model.config.text_config.hidden_size),
        vision_size=int(model.config.vision_config.hidden_size),
        relay_layer=int(config["relay_layer"]),
        token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
        field_dim=int(config.get("field_dim", 128)),
        subpatch_dim=int(config.get("subpatch_dim", 32)),
        subpatch_grid=int(config.get("subpatch_grid", 4)),
        temperature=float(config.get("temperature", 0.12)),
        target_sigma=float(config.get("field_target_sigma", 0.70)),
        initial_gate_logit=float(config.get("initial_gate_logit", -1.5)),
        initial_coverage_strength=float(config.get("initial_coverage_strength", 6.0)),
        residual_floor=float(config.get("residual_floor", 0.002)),
        footprint_halo_cells=float(config.get("footprint_halo_cells", 0.35)),
        initial_structure_scale=float(config.get("initial_structure_scale", 1.5)),
        absence_loss_weight=float(config.get("absence_loss_weight", 2.0)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    structure_override = os.environ.get("LOCANY_MARKED_STRUCTURE_SCALE_OVERRIDE")
    none_override = os.environ.get("LOCANY_MARKED_NONE_SCALE_OVERRIDE")
    lane_gate_override = os.environ.get("LOCANY_MARKED_LANE_GATE_OVERRIDE")
    with torch.no_grad():
        if structure_override is not None:
            controller.structure_scale_logit.fill_(
                _inverse_softplus(float(structure_override))
            )
        if none_override is not None:
            controller.none_scale_logit.fill_(_inverse_softplus(float(none_override)))
        if lane_gate_override is not None:
            probability = min(max(float(lane_gate_override), 1e-6), 1.0 - 1e-6)
            controller.lane_gate_logits.fill_(math.log(probability / (1.0 - probability)))
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
