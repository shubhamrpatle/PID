"""Point-conditioned geometry marks for open-ended LocateAnything decoding.

The visual event field proposes an unbounded sequence of latent object centers.
For each selected event, a learned mark distribution predicts the corresponding
HBB relative to that center and biases only the four native PBD coordinate
lanes.  The ordinary Qwen stream still decides whether to emit a box, emits the
six-token atomic block, and decides when to stop.

This is intentionally not a proposal detector: there are no fixed slots,
counts, crops, decoder copies, or externally decoded boxes.  A zero mark
strength recovers the native LocateAnything coordinate logits exactly.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .open_ended_residual_event_pbd import (
    OpenEndedResidualEventFieldPBD,
    open_ended_residual_event_ce,
)
from .pointrelay_pbd import configure_lora_only, restore_trainable_state
from .residual_set_reentry_pbd import ResidualSetTransition


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class PointMarkedResidualEventPBD(OpenEndedResidualEventFieldPBD):
    """Bind each learned visual event to a probabilistic native PBD box mark."""

    def __init__(
        self,
        *,
        minimum_half_extent: float = 0.001,
        maximum_half_extent: float = 0.30,
        initial_half_extent: float = 0.02,
        maximum_center_offset: float = 0.10,
        minimum_mark_sigma: float = 0.004,
        maximum_mark_sigma: float = 0.12,
        initial_mark_sigma: float = 0.03,
        mark_logit_floor: float = -16.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not 0 < minimum_half_extent < initial_half_extent < maximum_half_extent:
            raise ValueError("Mark half-extent bounds are invalid")
        if not 0 < minimum_mark_sigma < initial_mark_sigma < maximum_mark_sigma:
            raise ValueError("Mark uncertainty bounds are invalid")
        if maximum_center_offset <= 0 or mark_logit_floor >= 0:
            raise ValueError("Mark offset/floor configuration is invalid")

        evidence_dim = int(self.set_lane_hidden_proj.out_features)
        self.mark_head = nn.Sequential(
            nn.Linear(3 * evidence_dim, 2 * evidence_dim),
            nn.GELU(),
            nn.LayerNorm(2 * evidence_dim),
            nn.Linear(2 * evidence_dim, 8),
        )
        nn.init.xavier_uniform_(self.mark_head[0].weight)
        nn.init.zeros_(self.mark_head[0].bias)
        nn.init.zeros_(self.mark_head[-1].weight)
        nn.init.zeros_(self.mark_head[-1].bias)

        extent_ratio = (initial_half_extent - minimum_half_extent) / (
            maximum_half_extent - minimum_half_extent
        )
        sigma_ratio = (initial_mark_sigma - minimum_mark_sigma) / (
            maximum_mark_sigma - minimum_mark_sigma
        )
        with torch.no_grad():
            self.mark_head[-1].bias[2:4].fill_(_logit(extent_ratio))
            self.mark_head[-1].bias[4:8].fill_(_logit(sigma_ratio))

        self.minimum_half_extent = float(minimum_half_extent)
        self.maximum_half_extent = float(maximum_half_extent)
        self.maximum_center_offset = float(maximum_center_offset)
        self.minimum_mark_sigma = float(minimum_mark_sigma)
        self.maximum_mark_sigma = float(maximum_mark_sigma)
        self.mark_logit_floor = float(mark_logit_floor)
        self.register_buffer(
            "mark_coordinate_values", torch.linspace(0.0, 1.0, 1001), persistent=False
        )

        self._mark_means: tuple[torch.Tensor, ...] = ()
        self._mark_sigmas: tuple[torch.Tensor, ...] = ()
        self._mark_regression_loss: torch.Tensor | None = None
        self._mark_uncertainty_loss: torch.Tensor | None = None
        self._mark_iou: tuple[torch.Tensor, ...] = ()

    @property
    def mark_regression_loss(self) -> torch.Tensor | None:
        return self._mark_regression_loss

    @property
    def mark_uncertainty_loss(self) -> torch.Tensor | None:
        return self._mark_uncertainty_loss

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        super()._reset_sample_state(preserve_pixel_field=preserve_pixel_field)
        self._clear_mark_state()

    def _clear_mark_state(self) -> None:
        self._mark_means = ()
        self._mark_sigmas = ()
        self._mark_regression_loss = None
        self._mark_uncertainty_loss = None
        self._mark_iou = ()

    def _predict_mark(
        self,
        hidden_states: torch.Tensor,
        transition: ResidualSetTransition,
        selected: torch.Tensor,
        moments: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = self.set_lane_hidden_proj.weight.dtype
        query = self.set_lane_hidden_proj(
            hidden_states[0, transition.row[0]].to(dtype)
        )
        spatial = self.set_spatial_proj(moments.to(dtype))
        raw = self.mark_head(
            torch.cat((query, selected.to(dtype), spatial), dim=-1)
        ).float()

        anchor = moments[:2].float()
        center = anchor + self.maximum_center_offset * torch.tanh(raw[:2])
        half_extent = self.minimum_half_extent + (
            self.maximum_half_extent - self.minimum_half_extent
        ) * torch.sigmoid(raw[2:4])
        means = torch.stack(
            (
                center[0] - half_extent[0],
                center[1] - half_extent[1],
                center[0] + half_extent[0],
                center[1] + half_extent[1],
            )
        ).clamp(0.0, 1.0)
        sigmas = self.minimum_mark_sigma + (
            self.maximum_mark_sigma - self.minimum_mark_sigma
        ) * torch.sigmoid(raw[4:8])
        return means, sigmas

    def _mark_coordinate_prior(
        self, means: torch.Tensor, sigmas: torch.Tensor
    ) -> torch.Tensor:
        values = self.mark_coordinate_values.to(means)
        distance = values.unsqueeze(0) - means.unsqueeze(1)
        prior = -0.5 * distance.square() / sigmas.unsqueeze(1).square()
        return prior.clamp_min(self.mark_logit_floor)

    @staticmethod
    def _box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        top_left = torch.maximum(first[:2], second[:2])
        bottom_right = torch.minimum(first[2:], second[2:])
        intersection = (bottom_right - top_left).clamp_min(0.0).prod()
        first_area = (first[2:] - first[:2]).clamp_min(0.0).prod()
        second_area = (second[2:] - second[:2]).clamp_min(0.0).prod()
        union = first_area + second_area - intersection
        return intersection / union.clamp_min(1e-8)

    def _set_coordinate_signals(
        self,
        hidden_states: torch.Tensor,
        transitions: Sequence[ResidualSetTransition],
        selected_rows: Sequence[torch.Tensor],
        moment_rows: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        priors: list[torch.Tensor] = []
        means_rows: list[torch.Tensor] = []
        sigma_rows: list[torch.Tensor] = []
        regression_losses: list[torch.Tensor] = []
        uncertainty_losses: list[torch.Tensor] = []
        ious: list[torch.Tensor] = []

        for transition, selected, moments in zip(
            transitions, selected_rows, moment_rows
        ):
            means, sigmas = self._predict_mark(
                hidden_states, transition, selected, moments
            )
            priors.append(self._mark_coordinate_prior(means, sigmas))
            means_rows.append(means)
            sigma_rows.append(sigmas)
            if transition.target_box is None:
                continue
            target = means.new_tensor(transition.target_box).float() / 1000.0
            iou = self._box_iou(means, target)
            ious.append(iou)
            regression_losses.append(
                F.smooth_l1_loss(means, target, beta=0.01) + (1.0 - iou)
            )
            uncertainty_losses.append(
                F.gaussian_nll_loss(
                    means,
                    target,
                    sigmas.square().clamp_min(1e-6),
                    full=False,
                    reduction="mean",
                )
            )

        self._mark_means = tuple(means_rows)
        self._mark_sigmas = tuple(sigma_rows)
        self._mark_regression_loss = (
            torch.stack(regression_losses).mean() if regression_losses else None
        )
        self._mark_uncertainty_loss = (
            torch.stack(uncertainty_losses).mean() if uncertainty_losses else None
        )
        self._mark_iou = tuple(ious)
        return tuple(priors)

    def prepare_set_signals(self, hidden_states: torch.Tensor) -> None:
        self._clear_mark_state()
        super().prepare_set_signals(hidden_states)
        self.last_diagnostics.update(
            {
                "point_mark_regression_loss": 0.0
                if self._mark_regression_loss is None
                else float(self._mark_regression_loss.detach()),
                "point_mark_uncertainty_loss": 0.0
                if self._mark_uncertainty_loss is None
                else float(self._mark_uncertainty_loss.detach()),
                "point_mark_mean_iou": 0.0
                if not self._mark_iou
                else float(torch.stack(self._mark_iou).mean().detach()),
                "point_mark_count": float(len(self._mark_means)),
            }
        )

    def finish_forward(self) -> None:
        super().finish_forward()
        self._clear_mark_state()


def point_marked_residual_event_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    controller: PointMarkedResidualEventPBD,
    *,
    mark_regression_weight: float = 0.75,
    mark_uncertainty_weight: float = 0.05,
    **kwargs: Any,
) -> torch.Tensor:
    """Native LA loss plus event intensity, termination, and anchored marks."""

    loss = open_ended_residual_event_ce(
        hidden_states, labels, weight, controller, **kwargs
    )
    if controller.mark_regression_loss is not None:
        loss = loss + float(mark_regression_weight) * controller.mark_regression_loss
    if controller.mark_uncertainty_loss is not None:
        loss = loss + float(mark_uncertainty_weight) * controller.mark_uncertainty_loss
    return loss


def load_point_marked_residual_event_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "point_marked_residual_event_reentry":
        raise ValueError(f"Not a point-marked event checkpoint: {config.get('arm')!r}")
    configure_lora_only(model, int(config["lora_rank"]))
    controller_class: type[PointMarkedResidualEventPBD]
    hazard_kwargs: dict[str, Any] = {}
    if bool(config.get("causal_hierarchical_hazard")):
        from .causal_hierarchical_point_reentry_pbd import (
            CausalHierarchicalPointReentryPBD,
        )

        controller_class = CausalHierarchicalPointReentryPBD
        hazard_kwargs = {
            "hazard_temperature": float(config.get("hazard_temperature", 1.0)),
            "hazard_exact_weight": float(config.get("hazard_exact_weight", 0.50)),
            "hazard_set_weight": float(config.get("hazard_set_weight", 0.50)),
            "hazard_stop_weight": float(config.get("hazard_stop_weight", 1.0)),
            "initial_revisit_penalty": float(
                config.get("initial_revisit_penalty", 0.10)
            ),
            "maximum_revisit_penalty": float(
                config.get("maximum_revisit_penalty", 2.0)
            ),
            "permutation_invariant_residual_set": bool(
                config.get("permutation_invariant_residual_set", False)
            ),
            "hazard_fused_token_mass": bool(
                config.get("hazard_fused_token_mass", False)
            ),
            "permutation_marginalized_pbd": bool(
                config.get("permutation_marginalized_pbd", False)
            ),
            "joint_residual_set_selection": bool(
                config.get("joint_residual_set_selection", False)
            ),
            "initial_hazard_logit_scale": float(
                config.get("initial_hazard_logit_scale", 8.0)
            ),
            "maximum_hazard_logit_scale": float(
                config.get("maximum_hazard_logit_scale", 100.0)
            ),
        }
    elif bool(config.get("residual_compensator")):
        from .residual_compensator_pbd import ResidualCompensatorPBD

        controller_class = ResidualCompensatorPBD
        hazard_kwargs = {
            "initial_log_rate_bias": float(
                config.get("initial_log_rate_bias", -6.0)
            ),
            "minimum_cell_rate": float(config.get("minimum_cell_rate", 1e-8)),
            "compensator_point_weight": float(
                config.get("compensator_point_weight", 1.0)
            ),
            "compensator_set_weight": float(
                config.get("compensator_set_weight", 0.5)
            ),
            "compensator_survival_weight": float(
                config.get("compensator_survival_weight", 1.0)
            ),
            "event_selection_mode": str(
                config.get("event_selection_mode", "argmax")
            ),
            "event_sampling_temperature": float(
                config.get("event_sampling_temperature", 1.0)
            ),
        }
    elif bool(config.get("normalized_residual_hazard")):
        from .normalized_residual_hazard_pbd import NormalizedResidualHazardPBD

        controller_class = NormalizedResidualHazardPBD
        hazard_kwargs = {
            "hazard_temperature": float(config.get("hazard_temperature", 1.0)),
            "hazard_exact_weight": float(config.get("hazard_exact_weight", 0.50)),
            "hazard_set_weight": float(config.get("hazard_set_weight", 0.50)),
            "hazard_stop_weight": float(config.get("hazard_stop_weight", 1.0)),
        }
    else:
        controller_class = PointMarkedResidualEventPBD
    controller = controller_class(
        hidden_size=int(model.config.text_config.hidden_size),
        vision_hidden_size=int(model.config.vision_config.hidden_size),
        relay_layer=int(config.get("relay_layer", 18)),
        token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
        evidence_dim=int(config.get("evidence_dim", 128)),
        subpatch_dim=int(config.get("subpatch_dim", 32)),
        subpatch_grid=int(config.get("subpatch_grid", 8)),
        scale_multipliers=tuple(config.get("scale_multipliers", (0.75, 1.5, 3.0))),
        minimum_sigma_cells=float(config.get("minimum_sigma_cells", 0.75)),
        uncertainty_scale=float(config.get("uncertainty_scale", 2.0)),
        initial_residual_strength=float(config.get("initial_residual_strength", 0.10)),
        maximum_residual_strength=float(config.get("maximum_residual_strength", 2.0)),
        field_temperature=float(config.get("field_temperature", 0.12)),
        target_sigma_cells=float(config.get("target_sigma_cells", 0.65)),
        initial_coverage_strength=float(config.get("initial_coverage_strength", 8.0)),
        maximum_continuation_bias=float(config.get("maximum_continuation_bias", 4.0)),
        initial_set_coordinate_strength=float(config.get("initial_set_coordinate_strength", 0.05)),
        maximum_set_coordinate_strength=float(config.get("maximum_set_coordinate_strength", 4.0)),
        maximum_presence_bias=float(config.get("maximum_presence_bias", 4.0)),
        initial_gate_scale=float(config.get("initial_gate_scale", 0.10)),
        selection_temperature=float(config.get("selection_temperature", 0.50)),
        posterior_logit_scale=float(config.get("posterior_logit_scale", 2.0)),
        posterior_floor=float(config.get("posterior_floor", 1e-4)),
        initial_presence_rescue_scale=float(config.get("initial_presence_rescue_scale", 0.50)),
        counterfactual_margin=float(config.get("counterfactual_margin", 0.25)),
        hard_negative_exclusion=float(config.get("hard_negative_exclusion", 0.05)),
        initial_event_logit_bias=float(config.get("initial_event_logit_bias", -2.0)),
        evidence_pool_temperature=float(config.get("evidence_pool_temperature", 0.15)),
        event_suppression=float(config.get("event_suppression", 30.0)),
        event_target_sigma_cells=float(config.get("event_target_sigma_cells", 1.0)),
        maximum_event_radius=int(config.get("maximum_event_radius", 2)),
        event_suppression_radius=int(config.get("event_suppression_radius", 0)),
        focal_alpha=float(config.get("focal_alpha", 2.0)),
        focal_beta=float(config.get("focal_beta", 4.0)),
        initial_event_gate_scale=float(config.get("initial_event_gate_scale", 0.10)),
        minimum_half_extent=float(config.get("minimum_half_extent", 0.001)),
        maximum_half_extent=float(config.get("maximum_half_extent", 0.30)),
        initial_half_extent=float(config.get("initial_half_extent", 0.02)),
        maximum_center_offset=float(config.get("maximum_center_offset", 0.10)),
        minimum_mark_sigma=float(config.get("minimum_mark_sigma", 0.004)),
        maximum_mark_sigma=float(config.get("maximum_mark_sigma", 0.12)),
        initial_mark_sigma=float(config.get("initial_mark_sigma", 0.03)),
        mark_logit_floor=float(config.get("mark_logit_floor", -16.0)),
        **hazard_kwargs,
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
