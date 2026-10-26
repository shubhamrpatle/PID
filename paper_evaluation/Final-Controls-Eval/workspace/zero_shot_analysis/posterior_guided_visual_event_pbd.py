"""Posterior-guided latent visual events for LocateAnything PBD.

Training uses a GT-centered posterior only to choose the local visual witness
that conditions the current native geometry block.  A language-conditioned
prior is trained to match that posterior and is the *only* address mechanism
available at inference.  After each emitted geometry, the selected latent
event writes inhibition and the same recurrent process either emits another
native PBD block or stops.

The external sequence contains no point tokens, slots, proposals, or counts.
The controller keeps one cached image encoding and one Qwen decoder, and its
cardinality is bounded only by LocateAnything's sequence budget.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .latent_point_process_reentry_pbd import (
    LatentPointProcessVisualReentryPBD,
)
from .pointrelay_pbd import configure_lora_only, restore_trainable_state
from .residual_geometry_reentry_pbd import (
    ResidualGeometryVisualReentryPBD,
    residual_geometry_reentry_ce,
)
from .residual_set_reentry_pbd import ResidualSetTransition


class PosteriorGuidedVisualEventReentryPBD(
    LatentPointProcessVisualReentryPBD
):
    """Amortized latent visual-event process with a learned open-ended stop."""

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
        )
        if posterior_logit_scale <= 0:
            raise ValueError("posterior_logit_scale must be positive")
        if not 0 < posterior_floor < 1:
            raise ValueError("posterior_floor must lie in (0, 1)")
        if not 0 < initial_presence_rescue_scale < 1:
            raise ValueError("initial_presence_rescue_scale must lie in (0, 1)")
        self.posterior_logit_scale = float(posterior_logit_scale)
        self.posterior_floor = float(posterior_floor)
        self.presence_rescue_scale_logit = nn.Parameter(
            torch.tensor(
                math.log(
                    initial_presence_rescue_scale
                    / (1.0 - initial_presence_rescue_scale)
                )
            )
        )
        self._posterior_target_masses: tuple[torch.Tensor, ...] = ()
        self._posterior_selected_masses: tuple[torch.Tensor, ...] = ()
        self._pending_inference_event: torch.Tensor | None = None
        self._observed_inference_events = 0
        self._observed_inference_refs = 0

    def presence_rescue_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.presence_rescue_scale_logit)

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        super()._reset_sample_state(preserve_pixel_field=preserve_pixel_field)
        self._posterior_target_masses = ()
        self._posterior_selected_masses = ()
        self._pending_inference_event = None
        self._observed_inference_events = 0
        self._observed_inference_refs = 0

    def _coverage_write_support(
        self,
        transition: ResidualSetTransition,
        selection_weights: torch.Tensor,
    ) -> torch.Tensor | None:
        """Commit inferred events only after PBD accepts their geometry block.

        LocateAnything's cached generation exposes the current six-token block
        to decoder hooks, but not the previously accepted block.  Writing
        coverage inside the hook can therefore consume one event multiple
        times (or fail to consume it at all).  Training keeps the ordinary
        straight-through write.  At inference, the selected event is staged
        here and committed by :meth:`observe_generated_block` exactly once.
        """

        support = selection_weights.detach().float()
        if self._training_context:
            return support
        self._pending_inference_event = support
        return None

    def observe_generated_block(self, out_type: str, _tokens: Any) -> None:
        """Advance recurrent event memory at the decoder acceptance boundary."""

        if self._training_context:
            return
        if out_type == "ref_object":
            self._coverage = None
            self._pending_inference_event = None
            self._observed_inference_refs += 1
        elif out_type in {"coord_box", "point_box"}:
            support = self._pending_inference_event
            if support is None:
                return
            support = support.to(
                device=(
                    self._coverage.device
                    if self._coverage is not None
                    else support.device
                ),
                dtype=torch.float32,
            )
            self._coverage = (
                support
                if self._coverage is None
                else self._coverage.to(support) + support
            )
            self._pending_inference_event = None
            self._observed_inference_events += 1
        elif out_type in {"empty_box", "error_box", "im_end"}:
            self._pending_inference_event = None
        else:
            return
        self.last_diagnostics.update(
            {
                "posterior_observed_events": float(
                    self._observed_inference_events
                ),
                "posterior_observed_refs": float(self._observed_inference_refs),
                "posterior_inference_coverage_sum": 0.0
                if self._coverage is None
                else float(self._coverage.detach().float().sum().cpu()),
            }
        )

    def _selection_weights_for_transition(
        self,
        field_logits: torch.Tensor,
        transition: ResidualSetTransition,
    ) -> torch.Tensor:
        if not self._training_context or transition.target_box is None:
            return super()._selection_weights_for_transition(
                field_logits, transition
            )

        target = self._center_support(
            transition.target_box, device=field_logits.device
        ).float()
        target = target / target.max().clamp_min(1e-12)
        posterior_logits = field_logits.float() + self.posterior_logit_scale * torch.log(
            target.clamp_min(self.posterior_floor)
        )
        weights = F.gumbel_softmax(
            posterior_logits,
            tau=self.selection_temperature,
            hard=True,
            dim=-1,
        ).to(field_logits.dtype)
        self._selected_point_indices = (
            *self._selected_point_indices,
            int(weights.detach().argmax()),
        )
        prior = field_logits.float().softmax(dim=-1)
        self._posterior_target_masses = (
            *self._posterior_target_masses,
            (prior * target).sum(),
        )
        self._posterior_selected_masses = (
            *self._posterior_selected_masses,
            (weights.float() * target).sum(),
        )
        return weights

    def prepare_set_signals(self, hidden_states: torch.Tensor) -> None:
        # Skip the inherited log-mean-exp sentinel. Sparse tiny-object evidence
        # should not be averaged away by thousands of background cells.
        self._selected_point_indices = ()
        self._posterior_target_masses = ()
        self._posterior_selected_masses = ()
        ResidualGeometryVisualReentryPBD.prepare_set_signals(self, hidden_states)

        # Preserve the released model's negative evidence: this branch can
        # rescue a numeric first geometry block, but cannot add a negative bias
        # that turns a valid class into None.
        rescue_scale = self.presence_rescue_scale()
        self._presence_deltas = tuple(
            None
            if value is None
            else rescue_scale * value.clamp_min(0.0)
            for value in self._presence_deltas
        )
        self._sentinel_raw_gates = ()
        self._sentinel_loss = None
        self.last_diagnostics.update(
            {
                "posterior_guided_transitions": float(
                    len(self._posterior_target_masses)
                ),
                "posterior_prior_target_mass": 0.0
                if not self._posterior_target_masses
                else float(
                    torch.stack(self._posterior_target_masses).mean().detach()
                ),
                "posterior_selected_target_mass": 0.0
                if not self._posterior_selected_masses
                else float(
                    torch.stack(self._posterior_selected_masses).mean().detach()
                ),
                "presence_rescue_scale": float(rescue_scale.detach()),
                "presence_delta_abs": 0.0
                if not any(value is not None for value in self._presence_deltas)
                else float(
                    torch.stack(
                        [
                            value.abs()
                            for value in self._presence_deltas
                            if value is not None
                        ]
                    )
                    .mean()
                    .detach()
                ),
                "sentinel_loss": 0.0,
            }
        )

    def finish_forward(self) -> None:
        super().finish_forward()
        self._posterior_target_masses = ()
        self._posterior_selected_masses = ()
        self._pending_inference_event = None


def posterior_guided_visual_event_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    controller: PosteriorGuidedVisualEventReentryPBD,
    *,
    route_loss_weight: float = 0.20,
    continuation_loss_weight: float = 0.30,
    presence_loss_weight: float = 0.20,
    coordinate_aux_weight: float = 0.30,
) -> torch.Tensor:
    """Native LM objective plus amortized event, stop, and local geometry loss."""

    loss = residual_geometry_reentry_ce(
        hidden_states,
        labels,
        weight,
        controller,
        route_loss_weight=route_loss_weight,
        continuation_loss_weight=continuation_loss_weight,
        presence_loss_weight=presence_loss_weight,
    )
    residuals = controller.set_coordinate_residuals
    if not residuals:
        return loss
    coord_start = int(controller.token_ids["coord_start_token_id"])
    coord_end = int(controller.token_ids["coord_end_token_id"])
    strength = controller.set_coordinate_strength().float()
    coordinate_losses: list[torch.Tensor] = []
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
            coordinate_losses.append(
                F.cross_entropy(
                    modified.unsqueeze(0),
                    modified.new_tensor([target - coord_start], dtype=torch.long),
                )
            )
    if coordinate_losses:
        loss = loss + float(coordinate_aux_weight) * torch.stack(
            coordinate_losses
        ).mean()
    return loss


def load_posterior_guided_visual_event_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "posterior_guided_visual_event_reentry":
        raise ValueError(
            f"Not a posterior-guided visual-event checkpoint: {config.get('arm')!r}"
        )
    configure_lora_only(model, int(config["lora_rank"]))
    controller = PosteriorGuidedVisualEventReentryPBD(
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
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
