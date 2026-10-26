"""Cap-free residual-geometry visual re-entry for LocateAnything PBD.

This is the full RAVEL-style controller.  It extends residual-set continuation
with two tightly isolated operations: the uncovered visual field may decide
``None`` versus numeric geometry for the current reference and correct the four
numeric logits of the *next native geometry block*.  It never creates slots,
points, crops, decoder copies, semantic tokens, or a fixed count.

The final coordinate projection is zero initialized.  Before training, this
module is therefore exactly the released LocateAnything decoder.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .distribution_addressed_reentry_pbd import IGNORE_INDEX
from .pointrelay_pbd import configure_lora_only, restore_trainable_state
from .residual_set_reentry_pbd import (
    ResidualSetVisualReentryPBD,
    residual_set_reentry_ce,
)


class ResidualGeometryVisualReentryPBD(ResidualSetVisualReentryPBD):
    """Residual-set continuation plus uncovered-evidence coordinate readout."""

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
        field_temperature: float = 0.12,
        target_sigma_cells: float = 0.65,
        initial_coverage_strength: float = 5.0,
        maximum_continuation_bias: float = 3.0,
        initial_set_coordinate_strength: float = 0.10,
        maximum_set_coordinate_strength: float = 2.0,
        maximum_presence_bias: float = 3.0,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            vision_hidden_size=vision_hidden_size,
            relay_layer=relay_layer,
            token_ids=token_ids,
            evidence_dim=evidence_dim,
            subpatch_dim=subpatch_dim,
            subpatch_grid=subpatch_grid,
            scale_multipliers=scale_multipliers,
            minimum_sigma_cells=minimum_sigma_cells,
            uncertainty_scale=uncertainty_scale,
            initial_residual_strength=initial_residual_strength,
            maximum_residual_strength=maximum_residual_strength,
            field_temperature=field_temperature,
            target_sigma_cells=target_sigma_cells,
            initial_coverage_strength=initial_coverage_strength,
            maximum_continuation_bias=maximum_continuation_bias,
        )
        if not 0 < initial_set_coordinate_strength < maximum_set_coordinate_strength:
            raise ValueError("Set-coordinate strength must lie inside its maximum")
        if maximum_presence_bias <= 0:
            raise ValueError("Presence bias must be positive")
        self.maximum_set_coordinate_strength = float(maximum_set_coordinate_strength)
        self.maximum_presence_bias = float(maximum_presence_bias)
        self.set_lane_hidden_proj = nn.Linear(hidden_size, evidence_dim, bias=False)
        self.set_spatial_proj = nn.Sequential(
            nn.Linear(4, evidence_dim),
            nn.GELU(),
            nn.Linear(evidence_dim, evidence_dim),
        )
        self.set_coordinate_lane_embedding = nn.Parameter(
            torch.zeros(4, evidence_dim)
        )
        self.set_coordinate_head = nn.Sequential(
            nn.Linear(4 * evidence_dim, 2 * evidence_dim),
            nn.GELU(),
            nn.LayerNorm(2 * evidence_dim),
            nn.Linear(2 * evidence_dim, 1001),
        )
        self.presence_hidden_proj = nn.Linear(hidden_size, evidence_dim, bias=False)
        self.presence_head = nn.Sequential(
            nn.Linear(3 * evidence_dim, evidence_dim),
            nn.GELU(),
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, 1),
        )
        nn.init.xavier_uniform_(self.set_lane_hidden_proj.weight)
        nn.init.xavier_uniform_(self.set_spatial_proj[0].weight)
        nn.init.zeros_(self.set_spatial_proj[0].bias)
        nn.init.xavier_uniform_(self.set_spatial_proj[2].weight)
        nn.init.zeros_(self.set_spatial_proj[2].bias)
        nn.init.zeros_(self.set_coordinate_head[-1].weight)
        nn.init.zeros_(self.set_coordinate_head[-1].bias)
        nn.init.xavier_uniform_(self.presence_hidden_proj.weight)
        nn.init.zeros_(self.presence_head[-1].weight)
        nn.init.zeros_(self.presence_head[-1].bias)
        ratio = initial_set_coordinate_strength / maximum_set_coordinate_strength
        self.set_coordinate_strength_logit = nn.Parameter(
            torch.full((4,), math.log(ratio / (1.0 - ratio)))
        )
        self._set_coordinate_residuals: tuple[torch.Tensor, ...] = ()
        self._presence_deltas: tuple[torch.Tensor | None, ...] = ()
        self._presence_loss: torch.Tensor | None = None
        self.set_coordinate_gated_blocks = 0

    @property
    def set_coordinate_residuals(self) -> tuple[torch.Tensor, ...]:
        return self._set_coordinate_residuals

    def set_coordinate_strength(self) -> torch.Tensor:
        return self.maximum_set_coordinate_strength * torch.sigmoid(
            self.set_coordinate_strength_logit
        )

    @property
    def presence_loss(self) -> torch.Tensor | None:
        return self._presence_loss

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        super()._reset_sample_state(preserve_pixel_field=preserve_pixel_field)
        self._set_coordinate_residuals = ()
        self._presence_deltas = ()
        self._presence_loss = None
        self.set_coordinate_gated_blocks = 0

    def prepare_set_signals(self, hidden_states: torch.Tensor) -> None:
        super().prepare_set_signals(hidden_states)
        if not self._set_transitions or not self._set_selected_features:
            self._set_coordinate_residuals = ()
            self._presence_deltas = ()
            self._presence_loss = None
            return

        route_dtype = self.set_lane_hidden_proj.weight.dtype
        residuals: list[torch.Tensor] = []
        presence_deltas: list[torch.Tensor | None] = []
        positive_presence_losses: list[torch.Tensor] = []
        negative_presence_losses: list[torch.Tensor] = []
        for transition, selected, moments in zip(
            self._set_transitions,
            self._set_selected_features,
            self._set_spatial_moments,
        ):
            if max(transition.row) >= int(hidden_states.shape[1]):
                continue
            lane_hidden = self.set_lane_hidden_proj(
                torch.stack(
                    [hidden_states[0, position] for position in transition.row[1:5]]
                ).to(route_dtype)
            )
            selected_feature = selected.to(route_dtype).unsqueeze(0).expand(4, -1)
            spatial_feature = self.set_spatial_proj(
                moments.to(route_dtype)
            ).unsqueeze(0).expand(4, -1)
            lane_id = self.set_coordinate_lane_embedding.to(route_dtype)
            fused = torch.cat(
                (lane_hidden, selected_feature, spatial_feature, lane_id), dim=-1
            )
            residuals.append(self.set_coordinate_head(fused).float())
            if transition.predecessor == "ref":
                presence_hidden = self.presence_hidden_proj(
                    hidden_states[0, transition.row[0]].to(route_dtype)
                )
                presence_features = torch.cat(
                    (
                        presence_hidden,
                        selected.to(route_dtype),
                        self.set_spatial_proj(moments.to(route_dtype)),
                    )
                )
                raw_presence = self.presence_head(presence_features).reshape(())
                presence_delta = self.maximum_presence_bias * torch.tanh(
                    raw_presence
                )
                presence_deltas.append(presence_delta)
                label = 1.0 if transition.remaining_boxes else 0.0
                presence_decision_loss = F.binary_cross_entropy_with_logits(
                    raw_presence, raw_presence.new_tensor(label)
                )
                if label:
                    positive_presence_losses.append(presence_decision_loss)
                else:
                    negative_presence_losses.append(presence_decision_loss)
            else:
                presence_deltas.append(None)
        self._set_coordinate_residuals = tuple(residuals)
        self._presence_deltas = tuple(presence_deltas)
        balanced_presence = [
            torch.stack(group).mean()
            for group in (positive_presence_losses, negative_presence_losses)
            if group
        ]
        self._presence_loss = (
            torch.stack(balanced_presence).mean() if balanced_presence else None
        )
        self.last_diagnostics.update(
            {
                "set_coordinate_residual_abs_mean": 0.0
                if not residuals
                else float(
                    torch.stack([value.abs().mean() for value in residuals])
                    .mean()
                    .detach()
                ),
                "set_coordinate_strength": float(
                    self.set_coordinate_strength().mean().detach()
                ),
                "set_coordinate_gated_blocks": float(self.set_coordinate_gated_blocks),
                "presence_delta_abs": 0.0
                if not any(value is not None for value in presence_deltas)
                else float(
                    torch.stack(
                        [value.abs() for value in presence_deltas if value is not None]
                    ).mean().detach()
                ),
                "presence_loss": 0.0
                if self._presence_loss is None
                else float(self._presence_loss.detach()),
                "presence_positive_decisions": float(len(positive_presence_losses)),
                "presence_negative_decisions": float(len(negative_presence_losses)),
            }
        )

    def apply_set_coordinate_residual(
        self, logits: torch.Tensor, *, require_native_numeric: bool
    ) -> torch.Tensor:
        if not self._set_transitions or not self._set_coordinate_residuals:
            return logits
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        box_start = int(self.token_ids["box_start_token_id"])
        strength = self.set_coordinate_strength().to(logits.device, logits.dtype)
        updated = logits.clone()
        for transition, residual in zip(
            self._set_transitions, self._set_coordinate_residuals
        ):
            row = transition.row
            if max(row) >= int(logits.shape[1]):
                continue
            if require_native_numeric:
                first_lane = int(logits[0, row[1]].argmax())
                if (
                    int(logits[0, row[0]].argmax()) != box_start
                    or not coord_start <= first_lane <= coord_end
                ):
                    continue
            self.set_coordinate_gated_blocks += 1
            for lane, position in enumerate(row[1:5]):
                updated[0, position, coord_start : coord_end + 1] += (
                    strength[lane] * residual[lane].to(logits.dtype)
                )
        return updated

    def apply_presence_residual(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._set_transitions or not self._presence_deltas:
            return logits
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        none_token = int(self.token_ids["none_token_id"])
        updated = logits.clone()
        for transition, delta in zip(self._set_transitions, self._presence_deltas):
            if (
                transition.predecessor != "ref"
                or delta is None
                or transition.row[1] >= int(logits.shape[1])
            ):
                continue
            value = delta.to(logits.dtype)
            updated[0, transition.row[1], coord_start : coord_end + 1] += value
            updated[0, transition.row[1], none_token] -= value
        return updated

    def _lm_head_hook(
        self, module: nn.Module, args: tuple[Any, ...], output: torch.Tensor
    ) -> torch.Tensor:
        updated = super()._lm_head_hook(module, args, output)
        # Presence is isolated to the first geometry lane. Applying it before
        # coordinate refinement lets a visual positive recover from a native
        # ``None`` decision while still requiring numeric intent before any of
        # the four coordinate distributions can be changed.
        updated = self.apply_presence_residual(updated)
        return self.apply_set_coordinate_residual(
            updated, require_native_numeric=True
        )

    def finish_forward(self) -> None:
        super().finish_forward()
        self._set_coordinate_residuals = ()
        self._presence_deltas = ()
        self._presence_loss = None


def residual_geometry_reentry_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    controller: ResidualGeometryVisualReentryPBD,
    *,
    route_loss_weight: float = 0.03,
    continuation_loss_weight: float = 0.20,
    presence_loss_weight: float = 0.20,
) -> torch.Tensor:
    """Residual-set objective plus a numeric-only uncovered-field correction."""

    loss = residual_set_reentry_ce(
        hidden_states,
        labels,
        weight,
        controller,
        route_loss_weight=route_loss_weight,
        continuation_loss_weight=continuation_loss_weight,
    )
    residuals = controller.set_coordinate_residuals
    if not residuals:
        return loss

    valid_count = int(labels[:, 1:].ne(IGNORE_INDEX).sum())
    if valid_count <= 0:
        return loss
    coord_start = int(controller.token_ids["coord_start_token_id"])
    coord_end = int(controller.token_ids["coord_end_token_id"])
    set_strength = controller.set_coordinate_strength()
    parent_strength = controller.residual_strength()
    parent_by_row = {
        tuple(row): controller.coordinate_residuals[index]
        for index, row in enumerate(controller.active_rows)
        if controller.coordinate_residuals is not None
        and index < int(controller.coordinate_residuals.shape[0])
    }
    correction = hidden_states.new_zeros((), dtype=torch.float32)
    for transition, set_residual in zip(controller._set_transitions, residuals):
        parent_residual = parent_by_row.get(tuple(transition.row))
        for lane, position in enumerate(transition.row[1:5]):
            target_position = position + 1
            if target_position >= int(labels.shape[1]):
                continue
            target = labels[0, target_position]
            if not coord_start <= int(target) <= coord_end:
                continue
            base_logits = F.linear(hidden_states[0, position], weight).float()
            if parent_residual is not None:
                base_logits[coord_start : coord_end + 1] += (
                    parent_strength[lane].float() * parent_residual[lane].float()
                )
            modified = base_logits.clone()
            modified[coord_start : coord_end + 1] += (
                set_strength[lane].float() * set_residual[lane].float()
            )
            correction = correction + F.cross_entropy(
                modified.unsqueeze(0), target.reshape(1).long(), reduction="sum"
            ) - F.cross_entropy(
                base_logits.unsqueeze(0), target.reshape(1).long(), reduction="sum"
            )
    presence_correction = hidden_states.new_zeros((), dtype=torch.float32)
    none_token = int(controller.token_ids["none_token_id"])
    set_by_row = {
        tuple(transition.row): residual
        for transition, residual in zip(
            controller._set_transitions, controller.set_coordinate_residuals
        )
    }
    for transition, delta in zip(
        controller._set_transitions, controller._presence_deltas
    ):
        if transition.predecessor != "ref" or delta is None:
            continue
        position = transition.row[1]
        target_position = position + 1
        if target_position >= int(labels.shape[1]):
            continue
        target = labels[0, target_position]
        if int(target) == IGNORE_INDEX:
            continue
        base_logits = F.linear(hidden_states[0, position], weight).float()
        parent_residual = parent_by_row.get(tuple(transition.row))
        set_residual = set_by_row.get(tuple(transition.row))
        if coord_start <= int(target) <= coord_end:
            if parent_residual is not None:
                base_logits[coord_start : coord_end + 1] += (
                    parent_strength[0].float() * parent_residual[0].float()
                )
            if set_residual is not None:
                base_logits[coord_start : coord_end + 1] += (
                    set_strength[0].float() * set_residual[0].float()
                )
        modified = base_logits.clone()
        modified[coord_start : coord_end + 1] += delta.float()
        modified[none_token] -= delta.float()
        presence_correction = presence_correction + F.cross_entropy(
            modified.unsqueeze(0), target.reshape(1).long(), reduction="sum"
        ) - F.cross_entropy(
            base_logits.unsqueeze(0), target.reshape(1).long(), reduction="sum"
        )
    loss = loss + (correction + presence_correction) / valid_count
    if controller.presence_loss is not None:
        loss = loss + float(presence_loss_weight) * controller.presence_loss
    return loss


def load_residual_geometry_reentry_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "residual_geometry_visual_reentry":
        raise ValueError(f"Not a residual-geometry checkpoint: {config.get('arm')!r}")
    configure_lora_only(model, int(config["lora_rank"]))
    controller = ResidualGeometryVisualReentryPBD(
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
        field_temperature=float(config.get("field_temperature", 0.12)),
        target_sigma_cells=float(config.get("target_sigma_cells", 0.65)),
        initial_coverage_strength=float(config.get("initial_coverage_strength", 5.0)),
        maximum_continuation_bias=float(config.get("maximum_continuation_bias", 3.0)),
        initial_set_coordinate_strength=float(
            config.get("initial_set_coordinate_strength", 0.10)
        ),
        maximum_set_coordinate_strength=float(
            config.get("maximum_set_coordinate_strength", 2.0)
        ),
        maximum_presence_bias=float(config.get("maximum_presence_bias", 3.0)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
