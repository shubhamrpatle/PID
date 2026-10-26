"""Count-calibrated spatial-event re-entry for LocateAnything PBD.

The categorical visual-event controllers normalize every spatial field to one.
That is suitable for choosing one witness, but it cannot represent whether one
or one thousand unresolved instances remain.  This controller instead learns
an unnormalized language-conditioned intensity field.  Its integral predicts
remaining cardinality, while its local modes address one native PBD geometry
block at a time.

Training uses the same hidden GT posterior as the posterior-guided pilot only
to provide the visual mark for the current teacher-forced geometry block.  The
inference path sees only the learned intensity.  Accepted native geometry
blocks recurrently inhibit consumed evidence through the existing observer.
There are no proposal slots, explicit points, crops, count tokens, decoder
copies, or fixed object cap.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .pointrelay_pbd import configure_lora_only, restore_trainable_state
from .posterior_guided_visual_event_pbd import (
    PosteriorGuidedVisualEventReentryPBD,
    posterior_guided_visual_event_ce,
)


class IntensityGuidedVisualEventReentryPBD(
    PosteriorGuidedVisualEventReentryPBD
):
    """A marked spatial point process coupled to native PBD geometry."""

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
        selection_temperature: float = 0.50,
        posterior_logit_scale: float = 2.0,
        posterior_floor: float = 1e-4,
        initial_presence_rescue_scale: float = 0.50,
        initial_log_intensity_bias: float = -6.0,
        initial_intensity_decision_scale: float = 0.05,
        maximum_intensity_decision_bias: float = 3.0,
        intensity_stop_count: float = 0.50,
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
            initial_gate_scale=initial_gate_scale,
            selection_temperature=selection_temperature,
            posterior_logit_scale=posterior_logit_scale,
            posterior_floor=posterior_floor,
            initial_presence_rescue_scale=initial_presence_rescue_scale,
        )
        if not 0 < initial_intensity_decision_scale < 1:
            raise ValueError("initial_intensity_decision_scale must lie in (0, 1)")
        if maximum_intensity_decision_bias <= 0 or intensity_stop_count <= 0:
            raise ValueError("intensity decision constants must be positive")
        self.log_intensity_bias = nn.Parameter(
            torch.tensor(float(initial_log_intensity_bias))
        )
        self.intensity_decision_scale_logit = nn.Parameter(
            torch.tensor(
                math.log(
                    initial_intensity_decision_scale
                    / (1.0 - initial_intensity_decision_scale)
                )
            )
        )
        self.maximum_intensity_decision_bias = float(
            maximum_intensity_decision_bias
        )
        self.intensity_stop_count = float(intensity_stop_count)
        self._intensity_point_process_loss: torch.Tensor | None = None
        self._intensity_occupancy_loss: torch.Tensor | None = None
        self._intensity_count_loss: torch.Tensor | None = None
        self._intensity_expected_counts: tuple[torch.Tensor, ...] = ()

    @property
    def intensity_point_process_loss(self) -> torch.Tensor | None:
        return self._intensity_point_process_loss

    @property
    def intensity_occupancy_loss(self) -> torch.Tensor | None:
        return self._intensity_occupancy_loss

    @property
    def intensity_count_loss(self) -> torch.Tensor | None:
        return self._intensity_count_loss

    def intensity_decision_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.intensity_decision_scale_logit)

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        super()._reset_sample_state(preserve_pixel_field=preserve_pixel_field)
        self._intensity_point_process_loss = None
        self._intensity_occupancy_loss = None
        self._intensity_count_loss = None
        self._intensity_expected_counts = ()

    def _intensity_losses(
        self,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        tuple[torch.Tensor, ...],
    ]:
        point_process: list[torch.Tensor] = []
        occupancy: list[torch.Tensor] = []
        count_losses: list[torch.Tensor] = []
        expected_counts: list[torch.Tensor] = []

        for transition, field_logits in zip(
            self._set_transitions, self._set_logits
        ):
            log_rate = field_logits.float() + self.log_intensity_bias.float()
            rates = F.softplus(log_rate)
            expected = rates.sum()
            boxes = tuple(transition.remaining_boxes)
            target_count = len(boxes)
            expected_counts.append(expected)

            if boxes:
                supports = torch.stack(
                    [
                        self._center_support(box, device=field_logits.device)
                        for box in boxes
                    ]
                ).float()
                local_rates = torch.matmul(supports, rates).clamp_min(1e-8)
                # Discretized Poisson-process NLL, normalized per target so a
                # dense record cannot dominate merely because it is longer.
                point_process.append(
                    expected / float(target_count) - local_rates.log().mean()
                )
                target_union = supports.amax(dim=0)
                positive = target_union.gt(0.05)
                negative = ~positive
                groups: list[torch.Tensor] = []
                if bool(positive.any()):
                    groups.append(
                        F.binary_cross_entropy_with_logits(
                            log_rate[positive], torch.ones_like(log_rate[positive])
                        )
                    )
                if bool(negative.any()):
                    groups.append(
                        F.binary_cross_entropy_with_logits(
                            log_rate[negative], torch.zeros_like(log_rate[negative])
                        )
                    )
                occupancy.append(torch.stack(groups).mean())
            else:
                point_process.append(expected)
                occupancy.append(
                    F.binary_cross_entropy_with_logits(
                        log_rate, torch.zeros_like(log_rate)
                    )
                )

            target_log_count = expected.new_tensor(
                math.log1p(float(target_count))
            )
            count_losses.append(
                F.smooth_l1_loss(torch.log1p(expected), target_log_count)
            )

        def mean_or_none(values: list[torch.Tensor]) -> torch.Tensor | None:
            return torch.stack(values).mean() if values else None

        return (
            mean_or_none(point_process),
            mean_or_none(occupancy),
            mean_or_none(count_losses),
            tuple(expected_counts),
        )

    def prepare_set_signals(self, hidden_states: torch.Tensor) -> None:
        super().prepare_set_signals(hidden_states)
        (
            self._intensity_point_process_loss,
            self._intensity_occupancy_loss,
            self._intensity_count_loss,
            self._intensity_expected_counts,
        ) = self._intensity_losses()

        scale = self.intensity_decision_scale()
        threshold = math.log1p(self.intensity_stop_count)
        intensity_deltas = tuple(
            self.maximum_intensity_decision_bias
            * scale
            * torch.tanh(torch.log1p(expected) - threshold)
            for expected in self._intensity_expected_counts
        )
        self._continuation_deltas = tuple(
            original + extra
            for original, extra in zip(
                self._continuation_deltas, intensity_deltas
            )
        )
        self._presence_deltas = tuple(
            (
                None
                if original is None
                else original + extra.clamp_min(0.0)
            )
            for original, extra in zip(self._presence_deltas, intensity_deltas)
        )

        expected = self._intensity_expected_counts
        self.last_diagnostics.update(
            {
                "intensity_point_process_loss": 0.0
                if self._intensity_point_process_loss is None
                else float(self._intensity_point_process_loss.detach()),
                "intensity_occupancy_loss": 0.0
                if self._intensity_occupancy_loss is None
                else float(self._intensity_occupancy_loss.detach()),
                "intensity_count_loss": 0.0
                if self._intensity_count_loss is None
                else float(self._intensity_count_loss.detach()),
                "intensity_expected_count_mean": 0.0
                if not expected
                else float(torch.stack(expected).mean().detach()),
                "intensity_expected_count_max": 0.0
                if not expected
                else float(torch.stack(expected).max().detach()),
                "intensity_log_rate_bias": float(
                    self.log_intensity_bias.detach()
                ),
                "intensity_decision_scale": float(scale.detach()),
            }
        )

    def finish_forward(self) -> None:
        super().finish_forward()
        self._intensity_point_process_loss = None
        self._intensity_occupancy_loss = None
        self._intensity_count_loss = None
        self._intensity_expected_counts = ()


def intensity_guided_visual_event_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    controller: IntensityGuidedVisualEventReentryPBD,
    *,
    route_loss_weight: float = 0.20,
    continuation_loss_weight: float = 0.30,
    presence_loss_weight: float = 0.20,
    coordinate_aux_weight: float = 0.30,
    intensity_point_process_weight: float = 0.20,
    intensity_occupancy_weight: float = 0.20,
    intensity_count_weight: float = 0.10,
) -> torch.Tensor:
    """Native CE plus local geometry and count-calibrated event intensity."""

    loss = posterior_guided_visual_event_ce(
        hidden_states,
        labels,
        weight,
        controller,
        route_loss_weight=route_loss_weight,
        continuation_loss_weight=continuation_loss_weight,
        presence_loss_weight=presence_loss_weight,
        coordinate_aux_weight=coordinate_aux_weight,
    )
    for value, scale in (
        (controller.intensity_point_process_loss, intensity_point_process_weight),
        (controller.intensity_occupancy_loss, intensity_occupancy_weight),
        (controller.intensity_count_loss, intensity_count_weight),
    ):
        if value is not None:
            loss = loss + float(scale) * value
    return loss


def load_intensity_guided_visual_event_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "intensity_guided_visual_event_reentry":
        raise ValueError(
            f"Not an intensity-guided visual-event checkpoint: {config.get('arm')!r}"
        )
    configure_lora_only(model, int(config["lora_rank"]))
    controller = IntensityGuidedVisualEventReentryPBD(
        hidden_size=int(model.config.text_config.hidden_size),
        vision_hidden_size=int(model.config.vision_config.hidden_size),
        relay_layer=int(config.get("relay_layer", 18)),
        token_ids={
            **model.token_ids,
            "image_token_index": int(model.image_token_index),
        },
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
        selection_temperature=float(config.get("selection_temperature", 0.50)),
        posterior_logit_scale=float(config.get("posterior_logit_scale", 2.0)),
        posterior_floor=float(config.get("posterior_floor", 1e-4)),
        initial_presence_rescue_scale=float(
            config.get("initial_presence_rescue_scale", 0.50)
        ),
        initial_log_intensity_bias=float(
            config.get("initial_log_intensity_bias", -6.0)
        ),
        initial_intensity_decision_scale=float(
            config.get("initial_intensity_decision_scale", 0.05)
        ),
        maximum_intensity_decision_bias=float(
            config.get("maximum_intensity_decision_bias", 3.0)
        ),
        intensity_stop_count=float(config.get("intensity_stop_count", 0.50)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
