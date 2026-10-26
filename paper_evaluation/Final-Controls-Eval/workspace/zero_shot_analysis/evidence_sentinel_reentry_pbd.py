"""Open-ended evidence-sentinel visual re-entry for LocateAnything PBD.

The controller represents the next unresolved instance as a class-conditioned
distribution over a cached fine visual field.  A learned STOP energy competes
with that field.  Consequently, presence and continuation are one decision:
either visual evidence remains, or generation returns to the native grammar.
No proposal slots, point list, crop policy, decoder copies, or object cap are
introduced.
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
from .residual_geometry_reentry_pbd import (
    ResidualGeometryVisualReentryPBD,
    residual_geometry_reentry_ce,
)


class EvidenceSentinelVisualReentryPBD(ResidualGeometryVisualReentryPBD):
    """Grammar-isolated visual pointer with an explicit learned STOP sentinel."""

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
        initial_coverage_strength: float = 8.0,
        maximum_continuation_bias: float = 2.0,
        initial_set_coordinate_strength: float = 0.50,
        maximum_set_coordinate_strength: float = 4.0,
        maximum_presence_bias: float = 2.0,
        initial_gate_scale: float = 0.05,
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
            initial_set_coordinate_strength=initial_set_coordinate_strength,
            maximum_set_coordinate_strength=maximum_set_coordinate_strength,
            maximum_presence_bias=maximum_presence_bias,
        )
        if not 0 < initial_gate_scale < 1:
            raise ValueError("initial_gate_scale must be in (0, 1)")
        # One global threshold keeps evidence monotonic under coverage.  The
        # tiny positive scale leaves the released decoder effectively intact
        # while retaining a useful gradient at initialization.
        self.stop_threshold = nn.Parameter(torch.zeros(()))
        self.gate_scale_logit = nn.Parameter(
            torch.tensor(math.log(initial_gate_scale / (1.0 - initial_gate_scale)))
        )
        self._sentinel_raw_gates: tuple[torch.Tensor, ...] = ()
        self._sentinel_loss: torch.Tensor | None = None

    @property
    def sentinel_loss(self) -> torch.Tensor | None:
        return self._sentinel_loss

    def gate_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scale_logit)

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        super()._reset_sample_state(preserve_pixel_field=preserve_pixel_field)
        self._sentinel_raw_gates = ()
        self._sentinel_loss = None

    def prepare_set_signals(self, hidden_states: torch.Tensor) -> None:
        # Build the shared visual pointer, selected witness, spatial moments,
        # and coordinate residuals.  We then replace the independent presence
        # and continuation predictions with one evidence-vs-STOP energy.
        super().prepare_set_signals(hidden_states)
        if not self._set_transitions or not self._set_logits:
            self._sentinel_raw_gates = ()
            self._sentinel_loss = None
            return

        raw_gates: list[torch.Tensor] = []
        positive_losses: list[torch.Tensor] = []
        negative_losses: list[torch.Tensor] = []
        deltas: list[torch.Tensor] = []
        presence: list[torch.Tensor | None] = []
        scale = self.gate_scale()
        for transition, field_logits in zip(self._set_transitions, self._set_logits):
            # log-mean-exp makes the evidence scale independent of image-grid
            # size. Coverage only subtracts from field logits, so evidence
            # cannot increase merely because prior boxes were emitted.
            evidence = torch.logsumexp(field_logits, dim=0) - math.log(
                max(1, int(field_logits.numel()))
            )
            raw_gate = evidence - self.stop_threshold
            label = 1.0 if transition.remaining_boxes else 0.0
            decision = F.binary_cross_entropy_with_logits(
                raw_gate, raw_gate.new_tensor(label)
            )
            (positive_losses if label else negative_losses).append(decision)
            maximum = (
                self.maximum_presence_bias
                if transition.predecessor == "ref"
                else self.maximum_continuation_bias
            )
            delta = maximum * torch.tanh(scale * raw_gate)
            raw_gates.append(raw_gate)
            deltas.append(delta)
            presence.append(delta if transition.predecessor == "ref" else None)

        balanced = [
            torch.stack(group).mean()
            for group in (positive_losses, negative_losses)
            if group
        ]
        self._sentinel_raw_gates = tuple(raw_gates)
        self._sentinel_loss = torch.stack(balanced).mean() if balanced else None
        self._continuation_deltas = tuple(deltas)
        self._presence_deltas = tuple(presence)
        self._continuation_loss = self._sentinel_loss
        self._presence_loss = None
        self.last_diagnostics.update(
            {
                "sentinel_loss": 0.0
                if self._sentinel_loss is None
                else float(self._sentinel_loss.detach()),
                "sentinel_raw_gate_mean": float(
                    torch.stack(raw_gates).mean().detach()
                ),
                "sentinel_raw_gate_abs_mean": float(
                    torch.stack([value.abs() for value in raw_gates]).mean().detach()
                ),
                "sentinel_positive_decisions": float(len(positive_losses)),
                "sentinel_negative_decisions": float(len(negative_losses)),
                "sentinel_gate_scale": float(scale.detach()),
                "sentinel_stop_threshold": float(self.stop_threshold.detach()),
                "set_continuation_loss": 0.0
                if self._sentinel_loss is None
                else float(self._sentinel_loss.detach()),
                "presence_loss": 0.0,
                "presence_positive_decisions": float(
                    sum(
                        transition.predecessor == "ref"
                        and bool(transition.remaining_boxes)
                        for transition in self._set_transitions
                    )
                ),
                "presence_negative_decisions": float(
                    sum(
                        transition.predecessor == "ref"
                        and not transition.remaining_boxes
                        for transition in self._set_transitions
                    )
                ),
                "set_continuation_delta_abs": float(
                    torch.stack([value.abs() for value in deltas]).mean().detach()
                ),
                "presence_delta_abs": 0.0
                if not any(value is not None for value in presence)
                else float(
                    torch.stack([value.abs() for value in presence if value is not None])
                    .mean()
                    .detach()
                ),
            }
        )

    def finish_forward(self) -> None:
        super().finish_forward()
        self._sentinel_raw_gates = ()
        self._sentinel_loss = None


def evidence_sentinel_reentry_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    controller: EvidenceSentinelVisualReentryPBD,
    *,
    route_loss_weight: float = 0.05,
    sentinel_loss_weight: float = 0.30,
    coordinate_aux_weight: float = 0.20,
) -> torch.Tensor:
    """Native LM loss plus pointer, STOP, and coordinate-only supervision."""

    loss = residual_geometry_reentry_ce(
        hidden_states,
        labels,
        weight,
        controller,
        route_loss_weight=route_loss_weight,
        continuation_loss_weight=sentinel_loss_weight,
        presence_loss_weight=0.0,
    )
    residuals = controller.set_coordinate_residuals
    if not residuals:
        return loss
    coord_start = int(controller.token_ids["coord_start_token_id"])
    coord_end = int(controller.token_ids["coord_end_token_id"])
    strength = controller.set_coordinate_strength().float()
    losses: list[torch.Tensor] = []
    for transition, residual in zip(controller._set_transitions, residuals):
        for lane, position in enumerate(transition.row[1:5]):
            target_position = position + 1
            if target_position >= int(labels.shape[1]):
                continue
            target = int(labels[0, target_position])
            if not coord_start <= target <= coord_end:
                continue
            base = F.linear(hidden_states[0, position], weight).float()[
                coord_start : coord_end + 1
            ]
            modified = base + strength[lane] * residual[lane].float()
            losses.append(
                F.cross_entropy(
                    modified.unsqueeze(0),
                    modified.new_tensor([target - coord_start], dtype=torch.long),
                )
            )
    if losses:
        loss = loss + float(coordinate_aux_weight) * torch.stack(losses).mean()
    return loss


def load_evidence_sentinel_reentry_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "evidence_sentinel_visual_reentry":
        raise ValueError(f"Not an evidence-sentinel checkpoint: {config.get('arm')!r}")
    configure_lora_only(model, int(config["lora_rank"]))
    controller = EvidenceSentinelVisualReentryPBD(
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
        initial_coverage_strength=float(config.get("initial_coverage_strength", 8.0)),
        maximum_continuation_bias=float(config.get("maximum_continuation_bias", 2.0)),
        initial_set_coordinate_strength=float(
            config.get("initial_set_coordinate_strength", 0.50)
        ),
        maximum_set_coordinate_strength=float(
            config.get("maximum_set_coordinate_strength", 4.0)
        ),
        maximum_presence_bias=float(config.get("maximum_presence_bias", 2.0)),
        initial_gate_scale=float(config.get("initial_gate_scale", 0.05)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
