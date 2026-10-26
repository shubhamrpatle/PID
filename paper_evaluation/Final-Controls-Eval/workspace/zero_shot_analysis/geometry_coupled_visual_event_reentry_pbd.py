"""Accepted-geometry-coupled visual-event re-entry for LocateAnything PBD.

The controller preserves LocateAnything's native language and six-token HBB
grammar.  A query-conditioned latent visual event reads one cached fine visual
field before each geometry block.  The accepted geometry is then written back
to that field, so both fast PBD blocks and hybrid autoregressive fallbacks alter
the state used by the next event.  Training additionally contrasts the selected
event with a hard off-target event; this prevents the coordinate branch from
learning to ignore the re-entered local evidence.

There are no crops, detector proposals, parallel decoder replicas, object slots,
or fixed object-count caps.  Cardinality remains the native learned
autoregressive continue/stop process and is bounded only by the sequence budget.
"""

from __future__ import annotations

import os
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
from .residual_set_reentry_pbd import ResidualSetVisualReentryPBD


class GeometryCoupledVisualEventReentryPBD(
    PosteriorGuidedVisualEventReentryPBD
):
    """Closed-loop latent event read, native geometry emit, and spatial write-back."""

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
        counterfactual_margin: float = 0.25,
        hard_negative_exclusion: float = 0.05,
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
        if counterfactual_margin <= 0:
            raise ValueError("counterfactual_margin must be positive")
        if not 0 < hard_negative_exclusion < 1:
            raise ValueError("hard_negative_exclusion must lie in (0, 1)")
        self.counterfactual_margin = float(counterfactual_margin)
        self.hard_negative_exclusion = float(hard_negative_exclusion)
        self._geometry_cycle_loss: torch.Tensor | None = None
        self._counterfactual_event_loss: torch.Tensor | None = None
        self._counterfactual_pairs = 0

    @property
    def geometry_cycle_loss(self) -> torch.Tensor | None:
        return self._geometry_cycle_loss

    @property
    def counterfactual_event_loss(self) -> torch.Tensor | None:
        return self._counterfactual_event_loss

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        super()._reset_sample_state(preserve_pixel_field=preserve_pixel_field)
        self._geometry_cycle_loss = None
        self._counterfactual_event_loss = None
        self._counterfactual_pairs = 0

    def _coverage_write_support(
        self,
        transition: Any,
        selection_weights: torch.Tensor,
    ) -> torch.Tensor | None:
        """Mirror accepted-geometry feedback during teacher-forced training."""

        if not self._training_context or transition.target_box is None:
            return super()._coverage_write_support(transition, selection_weights)
        event_support = selection_weights.detach().float()
        geometry_support = self._center_support(
            transition.target_box, device=event_support.device
        ).float()
        geometry_support = geometry_support / geometry_support.max().clamp_min(1e-12)
        return torch.maximum(event_support, geometry_support)

    def _commit_pending_event(self) -> bool:
        support = self._pending_inference_event
        if support is None:
            return False
        support = support.detach().float()
        if self._coverage is not None:
            support = support.to(self._coverage.device)
        self._coverage = self._merge_coverage(self._coverage, support)
        self._pending_inference_event = None
        self._observed_inference_events += 1
        return True

    def _update_writeback_diagnostics(self) -> None:
        self.last_diagnostics.update(
            {
                "posterior_observed_events": float(self._observed_inference_events),
                "posterior_observed_refs": float(self._observed_inference_refs),
                "posterior_inference_coverage_sum": 0.0
                if self._coverage is None
                else float(self._coverage.detach().float().sum().cpu()),
                "accepted_geometry_writebacks": float(self.geometry_writeback_count),
            }
        )

    def observe_generated_block(self, out_type: str, _tokens: Any) -> None:
        """Commit the inferred event only when the fast PBD block is accepted."""

        if self._training_context:
            return
        if out_type == "ref_object":
            self._coverage = None
            self._pending_inference_event = None
            self._observed_inference_refs += 1
        elif out_type in {"coord_box", "point_box"}:
            self._commit_pending_event()
        elif out_type in {"empty_box", "error_box", "im_end"}:
            # An error block may subsequently complete through ordinary AR.  Its
            # accepted geometry is recovered by _write_previous_geometry below.
            self._pending_inference_event = None
        else:
            return
        self._update_writeback_diagnostics()

    def _write_previous_geometry(self, input_ids: torch.Tensor, row_start: int) -> None:
        """Write every accepted box, including boxes completed by AR fallback."""

        before = int(self.geometry_writeback_count)
        # LatentPointProcess deliberately disables this method.  Calling the
        # residual-set implementation restores actual accepted-geometry memory.
        ResidualSetVisualReentryPBD._write_previous_geometry(
            self, input_ids, row_start
        )
        if int(self.geometry_writeback_count) > before:
            # In a fast PBD path the observer has already consumed the event. In
            # an AR fallback this also safely consumes any still-pending event.
            self._commit_pending_event()
            self._update_writeback_diagnostics()

    def finish_forward(self) -> None:
        super().finish_forward()
        self._geometry_cycle_loss = None
        self._counterfactual_event_loss = None
        self._counterfactual_pairs = 0


def _event_coordinate_residual(
    controller: GeometryCoupledVisualEventReentryPBD,
    hidden_states: torch.Tensor,
    transition: Any,
    selected: torch.Tensor,
    moments: torch.Tensor,
) -> torch.Tensor:
    route_dtype = controller.set_lane_hidden_proj.weight.dtype
    lane_hidden = controller.set_lane_hidden_proj(
        torch.stack(
            [hidden_states[0, position] for position in transition.row[1:5]]
        ).to(route_dtype)
    )
    selected_feature = selected.to(route_dtype).unsqueeze(0).expand(4, -1)
    spatial_feature = controller.set_spatial_proj(
        moments.to(route_dtype)
    ).unsqueeze(0).expand(4, -1)
    fused = torch.cat(
        (
            lane_hidden,
            selected_feature,
            spatial_feature,
            controller.set_coordinate_lane_embedding.to(route_dtype),
        ),
        dim=-1,
    )
    return controller.set_coordinate_head(fused).float()


def geometry_coupled_visual_event_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    controller: GeometryCoupledVisualEventReentryPBD,
    *,
    route_loss_weight: float = 0.20,
    continuation_loss_weight: float = 0.30,
    presence_loss_weight: float = 0.20,
    coordinate_aux_weight: float = 0.30,
    geometry_cycle_weight: float = 0.20,
    counterfactual_event_weight: float = 0.20,
) -> torch.Tensor:
    """Native LA objective plus closed-loop event/geometry dependence losses."""

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
    return add_geometry_dependence_losses(
        hidden_states,
        weight,
        controller,
        loss,
        geometry_cycle_weight=geometry_cycle_weight,
        counterfactual_event_weight=counterfactual_event_weight,
    )


def add_geometry_dependence_losses(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    controller: GeometryCoupledVisualEventReentryPBD,
    loss: torch.Tensor,
    *,
    geometry_cycle_weight: float = 0.20,
    counterfactual_event_weight: float = 0.20,
) -> torch.Tensor:
    """Attach event/geometry consistency losses to any compatible base objective.

    Keeping this part separate lets a posterior-distilled controller use the
    same accepted-mark supervision without silently falling back to the weaker
    posterior-guided objective.
    """

    if (
        not controller._set_transitions
        or not controller._set_selected_features
        or not controller._set_spatial_moments
        or not controller._set_logits
        or controller._subpatch_positions is None
    ):
        return loss

    _, values = controller._field(hidden_states)
    positions = controller._subpatch_positions.to(
        hidden_states.device, dtype=torch.float32
    )
    coord_start = int(controller.token_ids["coord_start_token_id"])
    coord_end = int(controller.token_ids["coord_end_token_id"])
    coord_weight = weight[coord_start : coord_end + 1]
    coord_values = torch.linspace(
        0.0, 1.0, coord_end - coord_start + 1,
        device=hidden_states.device, dtype=torch.float32,
    )
    strength = controller.set_coordinate_strength().float().unsqueeze(1)
    cycle_losses: list[torch.Tensor] = []
    counterfactual_losses: list[torch.Tensor] = []

    for transition, selected, moments, field_logits, positive_residual in zip(
        controller._set_transitions,
        controller._set_selected_features,
        controller._set_spatial_moments,
        controller._set_logits,
        controller._set_coordinate_residuals,
    ):
        if transition.target_box is None or max(transition.row) >= int(
            hidden_states.shape[1]
        ):
            continue
        target = torch.as_tensor(
            transition.target_box, device=hidden_states.device, dtype=torch.long
        ).clamp_(0, 1000)
        lane_hidden = torch.stack(
            [hidden_states[0, position] for position in transition.row[1:5]]
        )
        base = F.linear(lane_hidden, coord_weight).float()
        positive_logits = base + strength * positive_residual.float()
        positive_log_probs = positive_logits.log_softmax(dim=-1)
        positive_nll = -positive_log_probs[
            torch.arange(4, device=target.device), target
        ].mean()

        expected = torch.matmul(positive_log_probs.exp(), coord_values)
        predicted_center = torch.stack(
            ((expected[0] + expected[2]) * 0.5, (expected[1] + expected[3]) * 0.5)
        )
        event_center = moments[:2].float()
        cycle_losses.append(
            F.smooth_l1_loss(predicted_center, event_center, beta=0.02)
        )

        target_support = controller._center_support(
            transition.target_box, device=hidden_states.device
        ).float()
        candidate_logits = field_logits.float().clone()
        candidate_logits[target_support >= controller.hard_negative_exclusion] = -torch.inf
        if not bool(torch.isfinite(candidate_logits).any()):
            continue
        negative_index = int(candidate_logits.argmax())
        negative_selected = values[negative_index]
        negative_moments = torch.cat(
            (positions[negative_index], positions.new_zeros(2))
        )
        negative_residual = _event_coordinate_residual(
            controller,
            hidden_states,
            transition,
            negative_selected,
            negative_moments,
        )
        negative_log_probs = (
            base + strength * negative_residual.float()
        ).log_softmax(dim=-1)
        negative_nll = -negative_log_probs[
            torch.arange(4, device=target.device), target
        ].mean()
        counterfactual_losses.append(
            F.relu(
                positive_nll.new_tensor(controller.counterfactual_margin)
                + positive_nll
                - negative_nll
            )
        )

    controller._geometry_cycle_loss = (
        torch.stack(cycle_losses).mean() if cycle_losses else None
    )
    controller._counterfactual_event_loss = (
        torch.stack(counterfactual_losses).mean()
        if counterfactual_losses
        else None
    )
    controller._counterfactual_pairs = len(counterfactual_losses)
    if controller.geometry_cycle_loss is not None:
        loss = loss + float(geometry_cycle_weight) * controller.geometry_cycle_loss
    if controller.counterfactual_event_loss is not None:
        loss = loss + float(
            counterfactual_event_weight
        ) * controller.counterfactual_event_loss
    controller.last_diagnostics.update(
        {
            "geometry_cycle_loss": 0.0
            if controller.geometry_cycle_loss is None
            else float(controller.geometry_cycle_loss.detach()),
            "counterfactual_event_loss": 0.0
            if controller.counterfactual_event_loss is None
            else float(controller.counterfactual_event_loss.detach()),
            "counterfactual_event_pairs": float(controller._counterfactual_pairs),
        }
    )
    return loss


def load_geometry_coupled_visual_event_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "geometry_coupled_visual_event_reentry":
        raise ValueError(
            f"Not a geometry-coupled visual-event checkpoint: {config.get('arm')!r}"
        )
    configure_lora_only(model, int(config["lora_rank"]))
    controller_class: type[GeometryCoupledVisualEventReentryPBD]
    variant = os.environ.get(
        "LOCANY_GEOMETRY_EVENT_CONTROLLER_VARIANT", "checkpoint"
    ).strip()
    if variant in {"", "checkpoint"}:
        controller_class = GeometryCoupledVisualEventReentryPBD
    elif variant == "cycle_guarded":
        from .cycle_guarded_geometry_coupled_pbd import (
            CycleGuardedGeometryCoupledVisualEventReentryPBD,
        )

        controller_class = CycleGuardedGeometryCoupledVisualEventReentryPBD
    else:
        raise ValueError(
            f"Unsupported geometry-coupled controller variant: {variant!r}"
        )
    controller = controller_class(
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
        selection_temperature=float(config.get("selection_temperature", 0.50)),
        posterior_logit_scale=float(config.get("posterior_logit_scale", 2.0)),
        posterior_floor=float(config.get("posterior_floor", 1e-4)),
        initial_presence_rescue_scale=float(
            config.get("initial_presence_rescue_scale", 0.50)
        ),
        counterfactual_margin=float(config.get("counterfactual_margin", 0.25)),
        hard_negative_exclusion=float(config.get("hard_negative_exclusion", 0.05)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {
        "checkpoint": checkpoint,
        "controller": controller,
        "controller_variant": variant or "checkpoint",
    }
