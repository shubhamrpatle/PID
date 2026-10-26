"""Distribution-addressed visual re-entry for LocateAnything PBD.

Each native six-token HBB block first forms its ordinary four coordinate-token
distributions.  Their moments define a soft, uncertainty-aware address over a
cached high-resolution visual field.  Local MoonViT and sub-patch evidence is
then used to add a bounded residual only to the same four coordinate
vocabularies.

The released semantic grammar, ``None``, EOS, PBD block structure, tokenizer,
and decoder count policy are untouched.  There are no object slots, proposal
caps, crop calls, or decoder replicas.  The controller starts as the exact
parent model because its final residual layer is initialized to zero.
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


IGNORE_INDEX = -100


class DistributionAddressedVisualReentryPBD(nn.Module):
    """Refine native PBD geometry from its own provisional distribution."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vision_hidden_size: int,
        relay_layer: int,
        token_ids: Mapping[str, int],
        evidence_dim: int = 128,
        subpatch_dim: int = 32,
        subpatch_grid: int = 4,
        scale_multipliers: Sequence[float] = (0.75, 1.5, 3.0),
        minimum_sigma_cells: float = 0.75,
        uncertainty_scale: float = 2.0,
        initial_residual_strength: float = 0.10,
        maximum_residual_strength: float = 2.0,
    ) -> None:
        super().__init__()
        if min(hidden_size, vision_hidden_size, evidence_dim, subpatch_dim) <= 0:
            raise ValueError("Distribution-addressed dimensions must be positive")
        if subpatch_grid <= 0:
            raise ValueError("subpatch_grid must be positive")
        if not scale_multipliers or any(value <= 0 for value in scale_multipliers):
            raise ValueError("scale_multipliers must contain positive values")
        if minimum_sigma_cells <= 0 or uncertainty_scale < 0:
            raise ValueError("Invalid local-kernel parameters")
        if not 0 < initial_residual_strength < maximum_residual_strength:
            raise ValueError(
                "initial_residual_strength must be between zero and the maximum"
            )

        self.hidden_size = int(hidden_size)
        self.vision_hidden_size = int(vision_hidden_size)
        self.relay_layer = int(relay_layer)
        self.evidence_dim = int(evidence_dim)
        self.subpatch_dim = int(subpatch_dim)
        self.subpatch_grid = int(subpatch_grid)
        self.scale_multipliers = tuple(float(value) for value in scale_multipliers)
        self.minimum_sigma_cells = float(minimum_sigma_cells)
        self.uncertainty_scale = float(uncertainty_scale)
        self.maximum_residual_strength = float(maximum_residual_strength)
        self.token_ids = {str(key): int(value) for key, value in token_ids.items()}

        self.fine_value_proj = nn.Linear(
            self.vision_hidden_size, self.evidence_dim, bias=False
        )
        self.subpatch_stem = nn.Sequential(
            nn.Conv2d(3, self.subpatch_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.subpatch_dim, self.subpatch_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((self.subpatch_grid, self.subpatch_grid)),
        )
        self.subpatch_value_proj = nn.Linear(
            self.subpatch_dim, self.evidence_dim, bias=False
        )
        # x1/y1/x2/y2 means, variances, center, and extent.
        self.geometry_proj = nn.Sequential(
            nn.Linear(12, self.evidence_dim),
            nn.GELU(),
            nn.Linear(self.evidence_dim, self.evidence_dim),
        )
        self.lane_embedding = nn.Parameter(torch.zeros(4, self.evidence_dim))
        fusion_width = self.evidence_dim * (
            2 + 2 * len(self.scale_multipliers)
        )
        self.coordinate_head = nn.Sequential(
            nn.Linear(fusion_width, 2 * self.evidence_dim),
            nn.GELU(),
            nn.LayerNorm(2 * self.evidence_dim),
            nn.Linear(2 * self.evidence_dim, 1001),
        )

        nn.init.xavier_uniform_(self.fine_value_proj.weight)
        nn.init.xavier_uniform_(self.subpatch_value_proj.weight)
        nn.init.xavier_uniform_(self.geometry_proj[0].weight)
        nn.init.zeros_(self.geometry_proj[0].bias)
        nn.init.xavier_uniform_(self.geometry_proj[2].weight)
        nn.init.zeros_(self.geometry_proj[2].bias)
        nn.init.zeros_(self.coordinate_head[-1].weight)
        nn.init.zeros_(self.coordinate_head[-1].bias)
        ratio = initial_residual_strength / maximum_residual_strength
        self.residual_strength_logit = nn.Parameter(
            torch.full((4,), math.log(ratio / (1.0 - ratio)))
        )

        self._context: RelayContext | None = None
        self._training_context = False
        self._grid_hw: tuple[int, int] | None = None
        self._fine_tokens: torch.Tensor | None = None
        self._fine_positions: torch.Tensor | None = None
        self._subpatch_tokens: torch.Tensor | None = None
        self._subpatch_positions: torch.Tensor | None = None
        self._fine_values_cache: torch.Tensor | None = None
        self._subpatch_values_cache: torch.Tensor | None = None
        self._coordinate_residuals: torch.Tensor | None = None
        self._active_rows: tuple[tuple[int, int, int, int, int, int], ...] = ()
        self._handles: list[Any] = []
        self.last_diagnostics: dict[str, float] = {}
        self.route_call_count = 0
        self.logit_call_count = 0
        self.box_gated_blocks = 0

    @property
    def coordinate_residuals(self) -> torch.Tensor | None:
        return self._coordinate_residuals

    @property
    def active_rows(self) -> tuple[tuple[int, int, int, int, int, int], ...]:
        return self._active_rows

    def residual_strength(self) -> torch.Tensor:
        return self.maximum_residual_strength * torch.sigmoid(
            self.residual_strength_logit
        )

    def _set_grid(self, image_grid_hws: Any) -> None:
        values = torch.as_tensor(image_grid_hws, dtype=torch.long).reshape(-1, 2)
        if int(values.shape[0]) != 1:
            raise ValueError("Distribution-addressed pilot supports one image per record")
        height, width = int(values[0, 0]), int(values[0, 1])
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid MoonViT patch grid: {(height, width)}")
        self._grid_hw = (height, width)

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        self._fine_tokens = None
        self._fine_positions = None
        self._fine_values_cache = None
        if not preserve_pixel_field:
            self._subpatch_tokens = None
            self._subpatch_positions = None
        self._subpatch_values_cache = None
        self._coordinate_residuals = None
        self._active_rows = ()
        self.last_diagnostics = {}
        self.route_call_count = 0
        self.logit_call_count = 0
        self.box_gated_blocks = 0

    def set_training_context(self, context: RelayContext, image_grid_hws: Any) -> None:
        self._context = context
        self._training_context = True
        self._set_grid(image_grid_hws)
        self._reset_sample_state()

    def set_inference_image_layout(
        self, _input_ids: torch.Tensor, image_grid_hws: Any
    ) -> None:
        self._context = None
        self._training_context = False
        self._set_grid(image_grid_hws)
        self._reset_sample_state()

    def finish_forward(self) -> None:
        self._context = None
        self._training_context = False
        self._coordinate_residuals = None
        self._active_rows = ()

    def prepare_pixel_field(
        self, pixel_values: torch.Tensor, image_grid_hws: Any
    ) -> None:
        """Retain a small learned layout inside each processor patch."""

        self._set_grid(image_grid_hws)
        if self._grid_hw is None:
            raise RuntimeError("Image grid is unavailable")
        height, width = self._grid_hw
        if height * width != int(pixel_values.shape[0]):
            raise RuntimeError(
                f"Pixel patch mismatch: grid={(height, width)} "
                f"patches={int(pixel_values.shape[0])}"
            )
        stem_dtype = next(self.subpatch_stem.parameters()).dtype
        features = self.subpatch_stem(pixel_values.to(dtype=stem_dtype))
        # [patch, channel, sub-y, sub-x] -> global row-major sub-patch lattice.
        features = features.permute(0, 2, 3, 1).reshape(
            height, width, self.subpatch_grid, self.subpatch_grid, self.subpatch_dim
        )
        features = features.permute(0, 2, 1, 3, 4).reshape(
            height * self.subpatch_grid,
            width * self.subpatch_grid,
            self.subpatch_dim,
        )
        self._subpatch_tokens = features.reshape(-1, self.subpatch_dim)
        self._subpatch_values_cache = None
        sub_h, sub_w = features.shape[:2]
        yy, xx = torch.meshgrid(
            (torch.arange(sub_h, device=pixel_values.device, dtype=torch.float32) + 0.5)
            / sub_h,
            (torch.arange(sub_w, device=pixel_values.device, dtype=torch.float32) + 0.5)
            / sub_w,
            indexing="ij",
        )
        self._subpatch_positions = torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def _vision_hook(
        self, _module: nn.Module, _args: tuple[Any, ...], output: Any
    ) -> None:
        fine = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(fine, torch.Tensor) or fine.ndim != 2:
            raise RuntimeError("MoonViT encoder did not return [tokens, hidden]")
        if self._grid_hw is None:
            raise RuntimeError("Image grid was not configured before MoonViT")
        height, width = self._grid_hw
        if height * width != int(fine.shape[0]):
            raise RuntimeError(
                f"Fine-token mismatch: features={fine.shape[0]} "
                f"expected={height * width}"
            )
        self._fine_tokens = fine.detach()
        self._fine_values_cache = None
        yy, xx = torch.meshgrid(
            (torch.arange(height, device=fine.device, dtype=torch.float32) + 0.5)
            / height,
            (torch.arange(width, device=fine.device, dtype=torch.float32) + 0.5)
            / width,
            indexing="ij",
        )
        self._fine_positions = torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def _inference_context(self, input_ids: torch.Tensor | None) -> RelayContext | None:
        if input_ids is None or input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            return None
        ids = input_ids[0]
        mask_id = int(self.token_ids["default_mask_token_id"])
        predecessors = {
            int(self.token_ids["ref_end_token_id"]),
            int(self.token_ids["box_end_token_id"]),
        }
        rows: list[tuple[int, int, int, int, int, int]] = []
        for start in range(max(0, int(ids.numel()) - 18), max(0, int(ids.numel()) - 5)):
            if start + 6 > int(ids.numel()) or int(ids[start]) not in predecessors:
                continue
            if bool(ids[start + 1 : start + 6].eq(mask_id).all()):
                rows.append(tuple(range(start, start + 6)))
        if not rows:
            return None
        return RelayContext((rows[-1],), (), (), int(ids.numel()), ())

    def _model_pre_hook(
        self,
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if self._training_context:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        self._context = self._inference_context(input_ids)
        self._coordinate_residuals = None
        self._active_rows = ()
        self.last_diagnostics = {}
        return None

    @staticmethod
    def _distribution_moments(coord_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.linspace(
            0.0, 1.0, 1001, device=coord_logits.device, dtype=torch.float32
        )
        probabilities = coord_logits.float().softmax(dim=-1)
        means = torch.matmul(probabilities, values)
        variances = torch.matmul(probabilities, values.square()) - means.square()
        return means, variances.clamp_min(0.0)

    def _pool_scales(
        self,
        *,
        positions: torch.Tensor,
        values: torch.Tensor,
        centers: torch.Tensor,
        sigma_xy: torch.Tensor,
        grid_hw: tuple[int, int],
    ) -> list[torch.Tensor]:
        grid_h, grid_w = grid_hw
        delta_x = (positions[:, 0].unsqueeze(0) - centers[:, :1]) * grid_w
        delta_y = (positions[:, 1].unsqueeze(0) - centers[:, 1:2]) * grid_h
        outputs: list[torch.Tensor] = []
        for multiplier in self.scale_multipliers:
            sx = (sigma_xy[:, :1] * grid_w * multiplier).clamp_min(
                self.minimum_sigma_cells
            )
            sy = (sigma_xy[:, 1:2] * grid_h * multiplier).clamp_min(
                self.minimum_sigma_cells
            )
            logits = -0.5 * ((delta_x / sx).square() + (delta_y / sy).square())
            weights = logits.softmax(dim=-1).to(values.dtype)
            outputs.append(torch.matmul(weights, values))
        return outputs

    def prepare_coordinate_residuals(
        self,
        hidden_states: torch.Tensor,
        coordinate_logits: torch.Tensor,
        context: RelayContext | None = None,
    ) -> None:
        """Build coordinate corrections from detached native geometry moments."""

        active = self._context if context is None else context
        rows = () if active is None else tuple(
            row for row in active.rows if max(row) < int(hidden_states.shape[1])
        )
        fine = self._fine_tokens
        fine_positions = self._fine_positions
        raw = self._subpatch_tokens
        raw_positions = self._subpatch_positions
        if not rows or fine is None or fine_positions is None or raw is None or raw_positions is None:
            self._coordinate_residuals = None
            self._active_rows = ()
            return
        if coordinate_logits.shape[:2] != (len(rows), 4) or coordinate_logits.shape[-1] != 1001:
            raise ValueError(
                "coordinate_logits must have shape [native_blocks, 4, 1001]"
            )
        if self._grid_hw is None:
            raise RuntimeError("MoonViT grid is unavailable")

        self.route_call_count += 1
        # Geometry is a causal address, not an auxiliary target.  Detaching it
        # keeps Qwen's LoRA gradient identical to ordinary token CE except for
        # the direct corrected-coordinate logits below.
        means, variances = self._distribution_moments(coordinate_logits.detach())
        left = torch.minimum(means[:, 0], means[:, 2])
        right = torch.maximum(means[:, 0], means[:, 2])
        top = torch.minimum(means[:, 1], means[:, 3])
        bottom = torch.maximum(means[:, 1], means[:, 3])
        centers = torch.stack(((left + right) * 0.5, (top + bottom) * 0.5), dim=-1)
        extents = torch.stack((right - left, bottom - top), dim=-1)
        uncertainty = torch.stack(
            (
                (variances[:, 0] + variances[:, 2]).sqrt(),
                (variances[:, 1] + variances[:, 3]).sqrt(),
            ),
            dim=-1,
        )
        sigma_xy = (0.5 * extents + self.uncertainty_scale * uncertainty).clamp_min(1e-4)
        geometry = torch.cat((means, variances, centers, extents), dim=-1)

        route_dtype = self.fine_value_proj.weight.dtype
        fine_values = self._fine_values_cache
        if fine_values is None:
            fine_values = self.fine_value_proj(
                fine.to(device=hidden_states.device, dtype=route_dtype)
            )
            self._fine_values_cache = fine_values
        raw_values = self._subpatch_values_cache
        if raw_values is None:
            raw_values = self.subpatch_value_proj(
                raw.to(device=hidden_states.device, dtype=route_dtype)
            )
            self._subpatch_values_cache = raw_values
        fine_scales = self._pool_scales(
            positions=fine_positions.to(hidden_states.device),
            values=fine_values,
            centers=centers,
            sigma_xy=sigma_xy,
            grid_hw=self._grid_hw,
        )
        raw_hw = (
            self._grid_hw[0] * self.subpatch_grid,
            self._grid_hw[1] * self.subpatch_grid,
        )
        raw_scales = self._pool_scales(
            positions=raw_positions.to(hidden_states.device),
            values=raw_values,
            centers=centers,
            sigma_xy=sigma_xy,
            grid_hw=raw_hw,
        )
        geometry_features = self.geometry_proj(geometry.to(route_dtype))

        residual_lanes: list[torch.Tensor] = []
        for lane in range(4):
            lane_id = self.lane_embedding[lane].unsqueeze(0).expand(len(rows), -1)
            fused = torch.cat(
                (geometry_features, lane_id, *fine_scales, *raw_scales), dim=-1
            )
            residual_lanes.append(self.coordinate_head(fused).float())
        self._coordinate_residuals = torch.stack(residual_lanes, dim=1)
        self._active_rows = rows
        entropy = -(
            coordinate_logits.detach().float().softmax(-1)
            * coordinate_logits.detach().float().log_softmax(-1)
        ).sum(-1)
        self.last_diagnostics = {
            "geometry_blocks": float(len(rows)),
            "geometry_mean_entropy": float(entropy.mean()),
            "geometry_mean_extent": float(extents.mean()),
            "geometry_mean_uncertainty": float(uncertainty.mean()),
            "geometry_residual_strength": float(self.residual_strength().mean().detach()),
            "geometry_residual_abs_mean": float(self._coordinate_residuals.abs().mean().detach()),
            "geometry_fine_tokens": float(fine.shape[0]),
            "geometry_subpatch_tokens": float(raw.shape[0]),
        }

    def apply_coordinate_residual(
        self, logits: torch.Tensor, *, require_native_box: bool
    ) -> torch.Tensor:
        residuals = self._coordinate_residuals
        if residuals is None or not self._active_rows:
            return logits
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        if coord_end - coord_start + 1 != 1001:
            raise RuntimeError("Distribution-addressed pilot expects 1,001 coordinate tokens")
        box_start = int(self.token_ids["box_start_token_id"])
        updated = logits.clone()
        strength = self.residual_strength().to(logits.device, logits.dtype)
        for block_index, row in enumerate(self._active_rows):
            if block_index >= int(residuals.shape[0]) or max(row) >= int(logits.shape[-2]):
                continue
            if require_native_box:
                # ``<box>None</box>`` and a numeric HBB share the same first
                # token.  Requiring a native numeric decision in the first
                # coordinate lane prevents visual correction from turning an
                # absent-class answer into a false box.
                first_lane = int(logits[0, row[1]].argmax())
                if (
                    int(logits[0, row[0]].argmax()) != box_start
                    or not coord_start <= first_lane <= coord_end
                ):
                    continue
            self.box_gated_blocks += 1
            for lane, position in enumerate(row[1:5]):
                updated[0, position, coord_start : coord_end + 1] += (
                    strength[lane] * residuals[block_index, lane].to(logits.dtype)
                )
        self.logit_call_count += 1
        return updated

    def _lm_head_hook(
        self,
        _module: nn.Module,
        args: tuple[Any, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        active = self._context
        if active is None or not active.rows:
            return output
        rows = tuple(row for row in active.rows if max(row) < int(output.shape[1]))
        if not rows:
            return output
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        coordinate_logits = torch.stack(
            [
                torch.stack(
                    [output[0, position, coord_start : coord_end + 1] for position in row[1:5]]
                )
                for row in rows
            ]
        )
        self.prepare_coordinate_residuals(args[0], coordinate_logits, active)
        # The native block type is decided before any coordinate correction.
        return self.apply_coordinate_residual(output, require_native_box=True)

    def install(self, model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("Distribution-addressed hooks are already installed")
        causal = base_causal_lm(model.language_model)
        layers = causal.model.layers
        if not 0 <= self.relay_layer < len(layers):
            raise ValueError(f"relay_layer={self.relay_layer} outside 0..{len(layers)-1}")
        self._handles.append(
            model.vision_model.encoder.register_forward_hook(self._vision_hook)
        )
        self._handles.append(
            causal.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        )
        self._handles.append(causal.lm_head.register_forward_hook(self._lm_head_hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def distribution_addressed_coordinate_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    controller: DistributionAddressedVisualReentryPBD,
    *,
    chunk: int = 128,
) -> torch.Tensor:
    """Shifted native CE plus geometry-only distribution-addressed correction."""

    coord_start = int(controller.token_ids["coord_start_token_id"])
    coord_end = int(controller.token_ids["coord_end_token_id"])
    rows = tuple(
        row for row in (controller._context.rows if controller._context else ())
        if max(row) < int(hidden_states.shape[1])
    )
    if rows:
        coord_weight = weight[coord_start : coord_end + 1]
        coordinate_logits = torch.stack(
            [
                F.linear(
                    torch.stack([hidden_states[0, position] for position in row[1:5]]),
                    coord_weight,
                )
                for row in rows
            ]
        )
        controller.prepare_coordinate_residuals(
            hidden_states, coordinate_logits, controller._context
        )

    shifted_hidden = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
    shifted_labels = labels[:, 1:].reshape(-1)
    valid = shifted_labels.ne(IGNORE_INDEX)
    valid_hidden = shifted_hidden[valid]
    valid_labels = shifted_labels[valid]
    count = int(valid_labels.numel())
    if count == 0:
        raise RuntimeError("Distribution-addressed sample has no supervised tokens")

    total = hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, count, int(chunk)):
        stop = min(count, start + int(chunk))
        logits = F.linear(valid_hidden[start:stop], weight).float()
        total = total + F.cross_entropy(logits, valid_labels[start:stop], reduction="sum")

    residuals = controller.coordinate_residuals
    active_rows = controller.active_rows
    if residuals is None or not active_rows:
        return total / count
    strength = controller.residual_strength()
    corrections: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    for block_index, row in enumerate(active_rows):
        for lane, position in enumerate(row[1:5]):
            target_position = position + 1
            if target_position >= labels.shape[1]:
                continue
            target = labels[0, target_position]
            if int(target) == IGNORE_INDEX:
                continue
            corrections.append(
                (hidden_states[0, position], target, residuals[block_index, lane], lane)
            )

    for start in range(0, len(corrections), 32):
        group = corrections[start : start + 32]
        states = torch.stack([item[0] for item in group])
        targets = torch.stack([item[1] for item in group]).long()
        base_logits = F.linear(states, weight).float()
        modified = base_logits.clone()
        for index, (_state, _target, residual, lane) in enumerate(group):
            modified[index, coord_start : coord_end + 1] += (
                strength[lane].float() * residual.float()
            )
        total = total - F.cross_entropy(base_logits, targets, reduction="sum")
        total = total + F.cross_entropy(modified, targets, reduction="sum")
    return total / count


def load_distribution_addressed_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Load the geometry re-entry controller and its matched Qwen LoRA."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    configure_lora_only(model, int(config["lora_rank"]))
    controller = DistributionAddressedVisualReentryPBD(
        hidden_size=int(model.config.text_config.hidden_size),
        vision_hidden_size=int(model.config.vision_config.hidden_size),
        relay_layer=int(config.get("relay_layer", 18)),
        token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
        evidence_dim=int(config.get("evidence_dim", 128)),
        subpatch_dim=int(config.get("subpatch_dim", 32)),
        subpatch_grid=int(config.get("subpatch_grid", 4)),
        scale_multipliers=tuple(config.get("scale_multipliers", (0.75, 1.5, 3.0))),
        minimum_sigma_cells=float(config.get("minimum_sigma_cells", 0.75)),
        uncertainty_scale=float(config.get("uncertainty_scale", 2.0)),
        initial_residual_strength=float(config.get("initial_residual_strength", 0.10)),
        maximum_residual_strength=float(config.get("maximum_residual_strength", 2.0)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
