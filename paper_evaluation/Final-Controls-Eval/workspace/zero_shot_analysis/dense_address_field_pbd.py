"""Cap-free dense visual-address routing for LocateAnything PBD blocks.

The controller keeps MoonViT's pre-merge feature lattice as a high-resolution
latent address field. Every native six-token HBB block queries the same field,
reads one soft local feature, and writes emitted-geometry coverage back. The number
of generated boxes remains controlled by LocateAnything's native language
grammar and EOS token; there is no learned slot bank or fixed top-k object cap.

This is an intentionally compact detection gate.  It preserves the released
MoonViT, projector, Qwen decoder, vocabulary, and PBD geometry grammar while
testing whether fine visual re-entry can be learned without crop re-encoding.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .pointrelay_pbd import (
    RelayContext,
    base_causal_lm,
    configure_lora_only,
    restore_trainable_state,
)


class DenseAddressFieldPBD(nn.Module):
    """Route each native PBD box through MoonViT's full pre-merge lattice."""

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
        temperature: float = 0.10,
        target_sigma: float = 0.70,
        initial_gate_logit: float = -3.0,
        initial_coverage_strength: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or vision_size <= 0 or field_dim <= 0 or subpatch_dim <= 0:
            raise ValueError("Dense-address dimensions must be positive")
        if subpatch_grid <= 0:
            raise ValueError("Dense-address subpatch grid must be positive")
        if temperature <= 0 or target_sigma <= 0:
            raise ValueError("Dense-address temperature and sigma must be positive")

        self.hidden_size = int(hidden_size)
        self.vision_size = int(vision_size)
        self.relay_layer = int(relay_layer)
        self.field_dim = int(field_dim)
        self.subpatch_dim = int(subpatch_dim)
        self.subpatch_grid = int(subpatch_grid)
        self.temperature = float(temperature)
        self.target_sigma = float(target_sigma)
        self.token_ids = {str(key): int(value) for key, value in token_ids.items()}

        self.query_proj = nn.Linear(self.hidden_size, self.field_dim, bias=False)
        self.key_proj = nn.Linear(self.vision_size, self.field_dim, bias=False)
        self.value_proj = nn.Linear(self.vision_size, self.field_dim, bias=False)
        self.subpatch_stem = nn.Sequential(
            nn.Conv2d(3, self.subpatch_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.subpatch_dim, self.subpatch_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((self.subpatch_grid, self.subpatch_grid)),
        )
        self.subpatch_key_proj = nn.Linear(self.subpatch_dim, self.field_dim, bias=False)
        self.subpatch_value_proj = nn.Linear(self.subpatch_dim, self.field_dim, bias=False)
        self.position_mlp = nn.Sequential(
            nn.Linear(2, self.field_dim),
            nn.GELU(),
            nn.Linear(self.field_dim, self.field_dim),
        )
        self.lane_queries = nn.Parameter(torch.zeros(4, self.field_dim))
        self.value_up = nn.Linear(self.field_dim, self.hidden_size, bias=False)
        self.continuation_up = nn.Linear(self.field_dim, self.hidden_size, bias=False)
        self.lane_gate_logits = nn.Parameter(torch.full((4,), float(initial_gate_logit)))
        self.continuation_gate_logit = nn.Parameter(torch.tensor(float(initial_gate_logit)))
        unit_softplus_logit = math.log(math.expm1(1.0))
        self.existence_scale_logit = nn.Parameter(torch.tensor(unit_softplus_logit))
        self.existence_bias = nn.Parameter(torch.zeros(()))
        coverage_logit = math.log(
            math.expm1(max(1e-4, float(initial_coverage_strength)))
        )
        self.coverage_strength_logit = nn.Parameter(torch.tensor(coverage_logit))

        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.subpatch_key_proj.weight)
        nn.init.xavier_uniform_(self.subpatch_value_proj.weight)
        nn.init.normal_(self.value_up.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.continuation_up.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.lane_queries, mean=0.0, std=0.02)

        self._context: RelayContext | None = None
        self._training_context = False
        self._stored_image_columns: tuple[int, ...] = ()
        self._field_hidden: torch.Tensor | None = None
        self._subpatch_hidden: torch.Tensor | None = None
        self._field_positions: torch.Tensor | None = None
        self._field_grid_hw: tuple[int, int] | None = None
        self._target_indices: tuple[int, ...] = ()
        self._existence_targets: tuple[float, ...] = ()
        self._reset_flags: tuple[bool, ...] = ()
        self._coverage: torch.Tensor | None = None
        self._inference_position_key: tuple[int, ...] = ()
        self._last_inference_key: tuple[tuple[int, ...], tuple[tuple[int, ...], ...]] | None = None
        self._last_written_geometry_key: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        self._aux_loss: torch.Tensor | None = None
        self._handles: list[Any] = []
        self.last_diagnostics: dict[str, float] = {}
        self.route_call_count = 0
        self.geometry_writeback_count = 0

    @property
    def aux_loss(self) -> torch.Tensor | None:
        return self._aux_loss

    def coverage_strength(self) -> torch.Tensor:
        return F.softplus(self.coverage_strength_logit)

    def _vision_hook(
        self,
        _module: nn.Module,
        args: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        if len(args) < 2:
            raise RuntimeError("MoonViT encoder hook did not receive grid_hws")
        grids = torch.as_tensor(args[1], device=output.device, dtype=torch.long).reshape(-1, 2)
        if int(grids.shape[0]) != 1:
            raise ValueError("Dense-address pilot supports exactly one image per record")
        grid_h, grid_w = int(grids[0, 0]), int(grids[0, 1])
        if grid_h * grid_w != int(output.shape[0]):
            raise RuntimeError(
                f"Dense-address grid mismatch: grid={(grid_h, grid_w)} tokens={int(output.shape[0])}"
            )
        # MoonViT is frozen in the matched LoRA gate.  Retaining the graph here
        # would waste memory without creating useful gradients.
        self._field_hidden = output.detach()
        if self._subpatch_hidden is None:
            ys = torch.linspace(0.0, 1.0, grid_h, device=output.device, dtype=torch.float32)
            xs = torch.linspace(0.0, 1.0, grid_w, device=output.device, dtype=torch.float32)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            self._field_positions = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        elif int(self._subpatch_hidden.shape[0]) != int(output.shape[0]):
            raise RuntimeError(
                "Subpatch/MoonViT patch mismatch: "
                f"subpatch={int(self._subpatch_hidden.shape[0])} moonvit={int(output.shape[0])}"
            )
        self._field_grid_hw = (
            (grid_h * self.subpatch_grid, grid_w * self.subpatch_grid)
            if self._subpatch_hidden is not None
            else (grid_h, grid_w)
        )
        self._coverage = None
        self._last_inference_key = None
        self._last_written_geometry_key = None

    def prepare_pixel_field(self, pixel_values: torch.Tensor, image_grid_hws: Any) -> None:
        """Build a high-resolution latent field without crop re-encoding.

        The processor already exposes every 14x14 MoonViT input patch. A small
        shared stem retains a 4x4 (configurable) layout inside each patch, then
        the vision hook fuses those cells with the parent MoonViT representation.
        """

        grids = torch.as_tensor(
            image_grid_hws, device=pixel_values.device, dtype=torch.long
        ).reshape(-1, 2)
        if int(grids.shape[0]) != 1:
            raise ValueError("Dense-address pixel field supports one image per record")
        grid_h, grid_w = int(grids[0, 0]), int(grids[0, 1])
        if grid_h * grid_w != int(pixel_values.shape[0]):
            raise RuntimeError(
                f"Pixel patch grid mismatch: grid={(grid_h, grid_w)} patches={int(pixel_values.shape[0])}"
            )
        stem_dtype = next(self.subpatch_stem.parameters()).dtype
        features = self.subpatch_stem(pixel_values.to(dtype=stem_dtype))
        # [patch, channel, sub-y, sub-x] -> [patch, sub-y*sub-x, channel]
        features = features.permute(0, 2, 3, 1).reshape(
            int(pixel_values.shape[0]), self.subpatch_grid**2, self.subpatch_dim
        )
        self._subpatch_hidden = features

        sub = self.subpatch_grid
        patch_y = torch.arange(grid_h, device=pixel_values.device, dtype=torch.float32)
        patch_x = torch.arange(grid_w, device=pixel_values.device, dtype=torch.float32)
        sub_y = (torch.arange(sub, device=pixel_values.device, dtype=torch.float32) + 0.5) / sub
        sub_x = (torch.arange(sub, device=pixel_values.device, dtype=torch.float32) + 0.5) / sub
        py, px, sy, sx = torch.meshgrid(patch_y, patch_x, sub_y, sub_x, indexing="ij")
        positions = torch.stack(
            ((px + sx) / max(1, grid_w), (py + sy) / max(1, grid_h)), dim=-1
        )
        self._field_positions = positions.reshape(-1, 2)

    def _center_index(self, center: Sequence[int]) -> int:
        if self._field_grid_hw is None or self._field_positions is None:
            raise RuntimeError("Dense-address field was not captured before target setup")
        target = self._field_positions.new_tensor(
            [float(center[0]) / 1000.0, float(center[1]) / 1000.0]
        )
        return int((self._field_positions - target).square().sum(dim=-1).argmin().item())

    def set_training_context(
        self,
        context: RelayContext,
        _image_grid_hws: Any | None = None,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> None:
        transition_context, target_indices, existence_targets, reset_flags = (
            self._training_transitions(input_ids, labels, context)
            if input_ids is not None and labels is not None
            else (
                context,
                tuple(self._center_index(center) for center in context.center_targets),
                tuple(1.0 for _ in context.rows),
                tuple(index == 0 for index, _ in enumerate(context.rows)),
            )
        )
        self._context = transition_context
        self._training_context = True
        self._stored_image_columns = context.image_columns
        self._target_indices = target_indices
        self._existence_targets = existence_targets
        self._reset_flags = reset_flags
        self._coverage = None
        self._inference_position_key = ()
        self._last_inference_key = None
        self._last_written_geometry_key = None
        self._aux_loss = None
        self.last_diagnostics = {}

    def _training_transitions(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        context: RelayContext,
    ) -> tuple[RelayContext, tuple[int, ...], tuple[float, ...], tuple[bool, ...]]:
        """Collect every native class-conditioned continue/stop decision.

        Numeric HBB targets supervise both an address and positive residual
        existence. ``None``, a following ``<ref>``, and EOS supervise absence.
        The field coverage resets after ``</ref>`` so classes do not suppress one
        another even when their objects overlap spatially.
        """

        if input_ids.ndim != 2 or labels.ndim != 2 or input_ids.shape != labels.shape:
            raise ValueError("Dense-address transitions require matching [1, sequence] ids")
        ids = input_ids[0]
        target = labels[0]
        mask_id = int(self.token_ids["default_mask_token_id"])
        ref_end = int(self.token_ids["ref_end_token_id"])
        box_end = int(self.token_ids["box_end_token_id"])
        box_start = int(self.token_ids["box_start_token_id"])
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        valid_predecessors = {ref_end, box_end}

        rows: list[tuple[int, int, int, int, int, int]] = []
        targets: list[int] = []
        exists: list[float] = []
        resets: list[bool] = []
        for start in range(max(0, int(ids.numel()) - 6)):
            if start + 6 >= int(target.numel()):
                break
            predecessor = int(ids[start])
            if predecessor not in valid_predecessors:
                continue
            if not bool(ids[start + 1 : start + 6].eq(mask_id).all()):
                continue
            block = [int(value) for value in target[start + 1 : start + 7].tolist()]
            numeric_box = (
                len(block) == 6
                and block[0] == box_start
                and block[5] == box_end
                and all(coord_start <= value <= coord_end for value in block[1:5])
            )
            rows.append(tuple(range(start, start + 6)))
            resets.append(predecessor == ref_end)
            if numeric_box:
                x1, y1, x2, y2 = [value - coord_start for value in block[1:5]]
                center = (
                    max(0, min(1000, int(round((x1 + x2) / 2.0)))),
                    max(0, min(1000, int(round((y1 + y2) / 2.0)))),
                )
                targets.append(self._center_index(center))
                exists.append(1.0)
            else:
                targets.append(-1)
                exists.append(0.0)

        transition_context = RelayContext(
            tuple(rows), (), context.image_columns, context.query_length, ()
        )
        return transition_context, tuple(targets), tuple(exists), tuple(resets)

    def set_inference_image_layout(self, input_ids: torch.Tensor, _image_grid_hws: Any) -> None:
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("Dense-address inference expects input_ids with shape [1, sequence]")
        image_index = int(self.token_ids["image_token_index"])
        self._stored_image_columns = tuple(
            int(value)
            for value in input_ids[0].eq(image_index).nonzero(as_tuple=False).flatten().tolist()
        )
        self._context = None
        self._training_context = False
        self._field_hidden = None
        self._subpatch_hidden = None
        self._field_positions = None
        self._field_grid_hw = None
        self._target_indices = ()
        self._existence_targets = ()
        self._reset_flags = ()
        self._coverage = None
        self._inference_position_key = ()
        self._last_inference_key = None
        self._last_written_geometry_key = None
        self._aux_loss = None
        self.last_diagnostics = {}

    def _inference_context(self, input_ids: torch.Tensor | None) -> RelayContext | None:
        if input_ids is None or input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            return None
        ids = input_ids[0]
        mask_id = int(self.token_ids["default_mask_token_id"])
        valid_predecessors = {
            int(self.token_ids["ref_end_token_id"]),
            int(self.token_ids["box_end_token_id"]),
        }
        rows: list[tuple[int, int, int, int, int, int]] = []
        for start in range(max(0, int(ids.numel()) - 18), max(0, int(ids.numel()) - 5)):
            if start + 6 > int(ids.numel()) or int(ids[start]) not in valid_predecessors:
                continue
            if bool(ids[start + 1 : start + 6].eq(mask_id).all()):
                rows.append(tuple(range(start, start + 6)))
        if not rows:
            return None
        selected_row = rows[-1]
        self._target_indices = (-1,)
        self._existence_targets = ()
        self._reset_flags = (
            int(ids[selected_row[0]]) == int(self.token_ids["ref_end_token_id"]),
        )
        return RelayContext((selected_row,), (), self._stored_image_columns, int(ids.numel()))

    def _model_pre_hook(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if self._training_context:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        position_ids = kwargs.get("position_ids")
        if isinstance(position_ids, torch.Tensor):
            self._inference_position_key = tuple(
                int(value) for value in position_ids.detach().reshape(-1).tolist()
            )
        else:
            self._inference_position_key = ()
        self._context = self._inference_context(input_ids)
        self._write_emitted_geometry(input_ids, self._context)
        self._aux_loss = None
        return None

    def _write_emitted_geometry(
        self, input_ids: torch.Tensor | None, context: RelayContext | None
    ) -> None:
        """Write the model's preceding PBD box into residual coverage.

        With KV caching, LA exposes the six emitted geometry tokens immediately
        before the duplicated ``</box>`` predecessor and five mask tokens for the
        next PBD window. Training writes the teacher-forced GT center; inference
        writes this actually emitted center once per generation position.
        """

        if (
            self._training_context
            or input_ids is None
            or input_ids.ndim != 2
            or int(input_ids.shape[0]) != 1
            or context is None
            or not context.rows
            or self._field_positions is None
        ):
            return
        row_start = int(context.rows[-1][0])
        if row_start < 6:
            return
        block = tuple(int(value) for value in input_ids[0, row_start - 6 : row_start].tolist())
        box_start = int(self.token_ids["box_start_token_id"])
        box_end = int(self.token_ids["box_end_token_id"])
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        if (
            len(block) != 6
            or block[0] != box_start
            or block[5] != box_end
            or not all(coord_start <= value <= coord_end for value in block[1:5])
        ):
            return
        write_key = (self._inference_position_key, block)
        if write_key == self._last_written_geometry_key:
            return
        x1, y1, x2, y2 = [value - coord_start for value in block[1:5]]
        center = (
            max(0, min(1000, int(round((x1 + x2) / 2.0)))),
            max(0, min(1000, int(round((y1 + y2) / 2.0)))),
        )
        target_index = self._center_index(center)
        field_cells = int(self._field_positions.shape[0])
        coverage = (
            torch.zeros(
                field_cells,
                device=self._field_positions.device,
                dtype=torch.float32,
            )
            if self._coverage is None or int(self._coverage.numel()) != field_cells
            else self._coverage.to(device=self._field_positions.device, dtype=torch.float32)
        )
        coverage[target_index] += 1.0
        self._coverage = coverage
        self._last_written_geometry_key = write_key
        self.geometry_writeback_count += 1

    def _gaussian_target(self, target_index: int, device: torch.device) -> torch.Tensor:
        if self._field_grid_hw is None:
            raise RuntimeError("Dense-address field grid is unavailable")
        grid_h, grid_w = self._field_grid_hw
        if self._field_positions is None:
            raise RuntimeError("Dense-address field positions are unavailable")
        positions = self._field_positions.to(device=device, dtype=torch.float32)
        target_position = positions[int(target_index)]
        scale = positions.new_tensor([float(grid_w), float(grid_h)])
        squared = ((positions - target_position) * scale).square().sum(dim=-1)
        target = torch.exp(-0.5 * squared / (self.target_sigma**2))
        return target / target.sum().clamp_min(1e-12)

    def _route(self, hidden_states: torch.Tensor, context: RelayContext) -> torch.Tensor:
        rows = tuple(row for row in context.rows if max(row) < int(hidden_states.shape[1]))
        field = self._field_hidden
        subpatch = self._subpatch_hidden
        positions = self._field_positions
        if not rows or field is None or positions is None or int(field.shape[0]) == 0:
            return hidden_states

        self.route_call_count += 1
        route_dtype = self.query_proj.weight.dtype
        field = field.to(device=hidden_states.device, dtype=route_dtype)
        positions = positions.to(device=hidden_states.device, dtype=route_dtype)
        position_features = self.position_mlp(positions).float()
        semantic_keys = self.key_proj(field).float()
        semantic_values = self.value_proj(field).float()
        if subpatch is not None:
            subpatch = subpatch.to(device=hidden_states.device, dtype=route_dtype)
            cells = int(subpatch.shape[1])
            semantic_keys = semantic_keys.unsqueeze(1).expand(-1, cells, -1).reshape(-1, self.field_dim)
            semantic_values = semantic_values.unsqueeze(1).expand(-1, cells, -1).reshape(-1, self.field_dim)
            raw_keys = self.subpatch_key_proj(subpatch).float().reshape(-1, self.field_dim)
            raw_values = self.subpatch_value_proj(subpatch).float().reshape(-1, self.field_dim)
            semantic_keys = semantic_keys + raw_keys
            semantic_values = semantic_values + raw_values
        keys = F.normalize(semantic_keys + position_features, dim=-1)
        values = semantic_values + position_features
        field_cells = int(keys.shape[0])
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
        entropies: list[torch.Tensor] = []
        peaks: list[torch.Tensor] = []
        expected_errors: list[torch.Tensor] = []

        for row_index, row in enumerate(rows):
            if row_index < len(self._reset_flags) and self._reset_flags[row_index]:
                coverage = torch.zeros_like(coverage)
            anchor = hidden_states[0, row[0]].to(route_dtype)
            query = F.normalize(self.query_proj(anchor).float(), dim=-1)
            logits = torch.matmul(keys, query) / self.temperature - strength * coverage
            probabilities = logits.softmax(dim=-1)
            evidence = torch.logsumexp(logits, dim=0) - math.log(max(1, int(logits.numel())))
            existence_scale = F.softplus(self.existence_scale_logit).float()
            existence_logit = existence_scale * evidence + self.existence_bias.float()
            existence_probability = torch.sigmoid(existence_logit)
            existence_probabilities.append(existence_probability)
            peaks.append(probabilities.max())
            entropies.append(
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            )

            target_index = (
                self._target_indices[row_index]
                if self._training_context and row_index < len(self._target_indices)
                else -1
            )
            if target_index >= 0:
                target = self._gaussian_target(target_index, logits.device)
                route_losses.append(-(target * logits.log_softmax(dim=-1)).sum())
                target_position = positions[target_index].float()
                expected_position = torch.matmul(probabilities, positions.float())
                expected_errors.append((expected_position - target_position).abs().mean())
                coverage_update = torch.zeros_like(coverage)
                coverage_update[target_index] = 1.0
            else:
                # During inference, coverage is updated on the next cached step
                # from the geometry the model actually emitted. This avoids
                # suppressing an attended location when Qwen generated elsewhere.
                coverage_update = torch.zeros_like(coverage)

            if self._training_context and row_index < len(self._existence_targets):
                existence_label = existence_logit.new_tensor(self._existence_targets[row_index])
                existence_labels.append(existence_label)
                existence_losses.append(
                    F.binary_cross_entropy_with_logits(existence_logit, existence_label)
                )

            selected = torch.matmul(probabilities, values)
            continuation_delta = self.continuation_up(F.silu(selected.to(route_dtype)))
            continuation_delta = F.layer_norm(
                continuation_delta.float(), (self.hidden_size,)
            ).to(hidden_states.dtype)
            continuation_gate = torch.sigmoid(self.continuation_gate_logit).to(hidden_states.dtype)
            updated[0, row[0]] = (
                updated[0, row[0]]
                + continuation_gate
                * existence_probability.to(hidden_states.dtype)
                * continuation_delta
            )
            lane_values = selected.unsqueeze(0) + self.lane_queries.float()
            delta = self.value_up(F.silu(lane_values.to(route_dtype)))
            delta = F.layer_norm(delta.float(), (self.hidden_size,)).to(hidden_states.dtype)
            for lane, position in enumerate(row[1:5]):
                updated[0, position] = updated[0, position] + gates[lane] * delta[lane]
            coverage = coverage + coverage_update

        route_loss = torch.stack(route_losses).mean() if route_losses else None
        existence_loss = (
            torch.stack(existence_losses).mean() if existence_losses else None
        )
        if route_loss is not None and existence_loss is not None:
            self._aux_loss = route_loss + existence_loss
        else:
            self._aux_loss = route_loss if route_loss is not None else existence_loss
        self._coverage = coverage.detach()
        max_entropy = math.log(max(2, field_cells))
        mean_entropy = torch.stack(entropies).mean() if entropies else hidden_states.new_zeros(())
        mean_peak = torch.stack(peaks).mean() if peaks else hidden_states.new_zeros(())
        mean_error = (
            torch.stack(expected_errors).mean()
            if expected_errors
            else hidden_states.new_zeros(())
        )
        mean_existence = (
            torch.stack(existence_probabilities).mean()
            if existence_probabilities
            else hidden_states.new_zeros(())
        )
        if existence_labels:
            labels_tensor = torch.stack(existence_labels)
            probabilities_tensor = torch.stack(existence_probabilities[: len(existence_labels)])
            existence_accuracy = probabilities_tensor.ge(0.5).eq(labels_tensor.ge(0.5)).float().mean()
            positive = probabilities_tensor[labels_tensor.ge(0.5)]
            negative = probabilities_tensor[labels_tensor.lt(0.5)]
            positive_mean = positive.mean() if int(positive.numel()) else mean_existence.new_zeros(())
            negative_mean = negative.mean() if int(negative.numel()) else mean_existence.new_zeros(())
        else:
            existence_accuracy = mean_existence.new_zeros(())
            positive_mean = mean_existence.new_zeros(())
            negative_mean = mean_existence.new_zeros(())
        self.last_diagnostics = {
            "address_field_cells": float(field_cells),
            "address_route_loss": 0.0
            if self._aux_loss is None
            else float(self._aux_loss.detach()),
            "address_route_peak": float(mean_peak.detach()),
            "address_route_entropy_fraction": float((mean_entropy / max_entropy).detach()),
            "address_expected_l1_0_1000": float(1000.0 * mean_error.detach()),
            "address_gate": float(gates.mean().detach()),
            "address_coverage_strength": float(strength.detach()),
            "address_coverage_sum": float(coverage.sum().detach()),
            "address_existence_loss": 0.0
            if existence_loss is None
            else float(existence_loss.detach()),
            "address_existence_probability": float(mean_existence.detach()),
            "address_existence_accuracy": float(existence_accuracy.detach()),
            "address_positive_existence": float(positive_mean.detach()),
            "address_negative_existence": float(negative_mean.detach()),
            "address_continuation_gate": float(continuation_gate.detach()),
        }
        return updated

    def _layer_pre_hook(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ):
        context = self._context
        if context is None or not context.rows:
            return None
        inference_key = (self._inference_position_key, tuple(context.rows))
        prior_coverage = self._coverage
        repeated = not self._training_context and self._last_inference_key == inference_key
        updated = self._route(args[0], context)
        if not self._training_context:
            if repeated:
                self._coverage = prior_coverage
            self._last_inference_key = inference_key
        return (updated,) + args[1:], kwargs

    def finish_forward(self) -> None:
        self._context = None
        self._training_context = False
        self._target_indices = ()
        self._existence_targets = ()
        self._reset_flags = ()
        self._coverage = None
        self._inference_position_key = ()
        self._last_inference_key = None
        self._last_written_geometry_key = None
        self._aux_loss = None

    def clear_context(self) -> None:
        self.finish_forward()
        self._field_hidden = None
        self._subpatch_hidden = None
        self._field_positions = None
        self._field_grid_hw = None

    def install(self, model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("Dense-address hooks are already installed")
        causal = base_causal_lm(model.language_model)
        layers = causal.model.layers
        if not 0 <= self.relay_layer < len(layers):
            raise ValueError(f"relay_layer={self.relay_layer} outside 0..{len(layers)-1}")
        encoder = getattr(model.vision_model, "encoder", None)
        if encoder is None:
            raise RuntimeError("Dense-address routing requires MoonViT.encoder")
        self._handles.append(encoder.register_forward_hook(self._vision_hook))
        self._handles.append(
            causal.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        )
        self._handles.append(
            layers[self.relay_layer].register_forward_pre_hook(
                self._layer_pre_hook, with_kwargs=True
            )
        )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def load_dense_address_field_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "dense_address_field":
        raise ValueError(f"Not a dense-address checkpoint: {config.get('arm')!r}")
    configure_lora_only(model, int(config["lora_rank"]))
    controller = DenseAddressFieldPBD(
        hidden_size=int(model.config.text_config.hidden_size),
        vision_size=int(model.config.vision_config.hidden_size),
        relay_layer=int(config["relay_layer"]),
        token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
        field_dim=int(config.get("field_dim", 128)),
        subpatch_dim=int(config.get("subpatch_dim", 32)),
        subpatch_grid=int(config.get("subpatch_grid", 4)),
        temperature=float(config.get("temperature", 0.10)),
        target_sigma=float(config.get("field_target_sigma", 0.70)),
        initial_gate_logit=float(config.get("initial_gate_logit", -3.0)),
        initial_coverage_strength=float(config.get("initial_coverage_strength", 0.5)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
