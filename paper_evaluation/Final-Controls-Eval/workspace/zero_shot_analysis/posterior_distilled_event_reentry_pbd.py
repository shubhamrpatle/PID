"""Posterior-distilled recurrent visual events for LocateAnything PBD.

The released decoder keeps its native semantic and six-token geometry grammar.
For each geometry transition, a learned prior addresses one event in a cached
fine visual field.  During training only, a GT-conditioned posterior provides
an easier witness.  This controller explicitly transfers that posterior into
the inference prior and also trains the shared geometry/continue heads on the
prior-selected witness.  Accepted native geometry is written back into the
coverage state, so the same decoder can discover another event or stop without
fixed slots, counts, crops, or decoder replicas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .distribution_addressed_reentry_pbd import IGNORE_INDEX
from .pointrelay_pbd import configure_lora_only, restore_trainable_state
from .posterior_guided_visual_event_pbd import (
    PosteriorGuidedVisualEventReentryPBD,
    posterior_guided_visual_event_ce,
)
from .residual_set_reentry_pbd import ResidualSetTransition


class PosteriorDistilledVisualEventReentryPBD(
    PosteriorGuidedVisualEventReentryPBD
):
    """Align the latent event posterior, inference prior, and accepted mark."""

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
            posterior_logit_scale=posterior_logit_scale,
            posterior_floor=posterior_floor,
            initial_presence_rescue_scale=initial_presence_rescue_scale,
        )
        self._prior_selection_weights: tuple[torch.Tensor, ...] = ()
        self._prior_coordinate_residuals: tuple[torch.Tensor, ...] = ()
        self._posterior_prior_kls: tuple[torch.Tensor, ...] = ()
        self._prior_geometry_loss: torch.Tensor | None = None
        self._prior_decision_loss: torch.Tensor | None = None

    @property
    def posterior_prior_loss(self) -> torch.Tensor | None:
        if not self._posterior_prior_kls:
            return None
        return torch.stack(self._posterior_prior_kls).mean()

    @property
    def prior_geometry_loss(self) -> torch.Tensor | None:
        return self._prior_geometry_loss

    @property
    def prior_decision_loss(self) -> torch.Tensor | None:
        return self._prior_decision_loss

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        super()._reset_sample_state(preserve_pixel_field=preserve_pixel_field)
        self._prior_selection_weights = ()
        self._prior_coordinate_residuals = ()
        self._posterior_prior_kls = ()
        self._prior_geometry_loss = None
        self._prior_decision_loss = None

    def _selection_weights_for_transition(
        self,
        field_logits: torch.Tensor,
        transition: ResidualSetTransition,
    ) -> torch.Tensor:
        if not self._training_context:
            return super()._selection_weights_for_transition(
                field_logits, transition
            )

        prior = field_logits.float().softmax(dim=-1)
        prior_index = prior.argmax()
        prior_hard = F.one_hot(
            prior_index, num_classes=int(prior.numel())
        ).to(prior.dtype)
        prior_st = (prior_hard + prior - prior.detach()).to(field_logits.dtype)
        self._prior_selection_weights = (
            *self._prior_selection_weights,
            prior_st,
        )
        if transition.target_box is None:
            self._selected_point_indices = (
                *self._selected_point_indices,
                int(prior_index.detach()),
            )
            return prior_st

        target = self._center_support(
            transition.target_box, device=field_logits.device
        ).float()
        target = target / target.max().clamp_min(1e-12)
        posterior_logits = (
            field_logits.float()
            + self.posterior_logit_scale
            * torch.log(target.clamp_min(self.posterior_floor))
        )
        posterior = posterior_logits.softmax(dim=-1)
        posterior_hard = F.gumbel_softmax(
            posterior_logits,
            tau=self.selection_temperature,
            hard=True,
            dim=-1,
        ).to(field_logits.dtype)

        # The forward path uses the prior argmax, while its backward path
        # remains differentiable.  This is used only by auxiliary heads; the
        # native geometry path retains the teacher posterior during training.
        self._posterior_prior_kls = (
            *self._posterior_prior_kls,
            F.kl_div(
                prior.clamp_min(1e-12).log(),
                posterior.detach(),
                reduction="sum",
            ),
        )
        self._selected_point_indices = (
            *self._selected_point_indices,
            int(posterior_hard.detach().argmax()),
        )
        self._posterior_target_masses = (
            *self._posterior_target_masses,
            (prior * target).sum(),
        )
        self._posterior_selected_masses = (
            *self._posterior_selected_masses,
            (posterior_hard.float() * target).sum(),
        )
        return posterior_hard

    def _prior_signals(
        self, hidden_states: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
        values = self._set_values_cache
        positions = self._subpatch_positions
        if (
            not self._training_context
            or values is None
            or positions is None
            or not self._prior_selection_weights
        ):
            return (), None

        route_dtype = self.set_lane_hidden_proj.weight.dtype
        residuals: list[torch.Tensor] = []
        positive_decisions: list[torch.Tensor] = []
        negative_decisions: list[torch.Tensor] = []
        for transition, field_logits, weights in zip(
            self._set_transitions,
            self._set_logits,
            self._prior_selection_weights,
        ):
            if max(transition.row) >= int(hidden_states.shape[1]):
                continue
            selected = torch.matmul(weights.float(), values.float())
            mean = torch.matmul(weights.float(), positions.float())
            variance = torch.matmul(
                weights.float(), (positions.float() - mean).square()
            )
            moments = torch.cat((mean, variance))
            lane_hidden = self.set_lane_hidden_proj(
                torch.stack(
                    [
                        hidden_states[0, position]
                        for position in transition.row[1:5]
                    ]
                ).to(route_dtype)
            )
            selected_feature = selected.to(route_dtype).unsqueeze(0).expand(4, -1)
            spatial = self.set_spatial_proj(moments.to(route_dtype))
            spatial_feature = spatial.unsqueeze(0).expand(4, -1)
            fused = torch.cat(
                (
                    lane_hidden,
                    selected_feature,
                    spatial_feature,
                    self.set_coordinate_lane_embedding.to(route_dtype),
                ),
                dim=-1,
            )
            residuals.append(self.set_coordinate_head(fused).float())

            probabilities = field_logits.float().softmax(dim=-1)
            query = F.normalize(
                self.set_query_proj(
                    hidden_states[0, transition.row[0]].to(
                        self.set_query_proj.weight.dtype
                    )
                ).float(),
                dim=-1,
            )
            maximum = field_logits.float().max() / 10.0
            log_mean_exp = (
                torch.logsumexp(field_logits.float(), dim=0)
                - torch.log(
                    field_logits.new_tensor(
                        float(max(1, int(field_logits.numel())))
                    )
                )
            ) / 10.0
            entropy = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum() / torch.log(
                probabilities.new_tensor(float(max(2, int(probabilities.numel()))))
            )
            covered_fraction = (
                hidden_states.new_tensor(0.0)
                if self._coverage is None
                else self._coverage.to(hidden_states).gt(0.05).float().mean()
            )
            decision_features = torch.cat(
                (
                    query,
                    F.normalize(selected.float(), dim=-1),
                    maximum.reshape(1),
                    log_mean_exp.reshape(1),
                    entropy.reshape(1),
                    covered_fraction.reshape(1),
                )
            )
            raw_continue = self.continuation_head(
                decision_features.to(self.continuation_head[0].weight.dtype)
            ).reshape(())
            if transition.predecessor == "box":
                target = raw_continue.new_tensor(
                    1.0 if transition.remaining_boxes else 0.0
                )
                decision = F.binary_cross_entropy_with_logits(
                    raw_continue, target
                )
                (
                    positive_decisions
                    if bool(transition.remaining_boxes)
                    else negative_decisions
                ).append(decision)

        groups = [
            torch.stack(group).mean()
            for group in (positive_decisions, negative_decisions)
            if group
        ]
        decision_loss = torch.stack(groups).mean() if groups else None
        return tuple(residuals), decision_loss

    def prepare_set_signals(self, hidden_states: torch.Tensor) -> None:
        self._prior_selection_weights = ()
        self._posterior_prior_kls = ()
        super().prepare_set_signals(hidden_states)
        (
            self._prior_coordinate_residuals,
            self._prior_decision_loss,
        ) = self._prior_signals(hidden_states)
        self.last_diagnostics.update(
            {
                "distilled_posterior_prior_kl": 0.0
                if self.posterior_prior_loss is None
                else float(self.posterior_prior_loss.detach()),
                "distilled_prior_geometry_blocks": float(
                    len(self._prior_coordinate_residuals)
                ),
                "distilled_prior_decision_loss": 0.0
                if self._prior_decision_loss is None
                else float(self._prior_decision_loss.detach()),
            }
        )

    def observe_generated_block(self, out_type: str, tokens: Any) -> None:
        # Commit the selected latent event first.
        super().observe_generated_block(out_type, tokens)
        if self._training_context or out_type not in {"coord_box", "point_box"}:
            return
        values = (
            [int(value) for value in tokens.detach().reshape(-1).tolist()]
            if isinstance(tokens, torch.Tensor)
            else [
                int(value.item() if hasattr(value, "item") else value)
                for value in tokens
            ]
        )
        box_start = int(self.token_ids["box_start_token_id"])
        box_end = int(self.token_ids["box_end_token_id"])
        coord_start = int(self.token_ids["coord_start_token_id"])
        if not values or values[0] != box_start or values[-1] != box_end:
            return
        if out_type == "coord_box" and len(values) == 6:
            box = [value - coord_start for value in values[1:5]]
        elif out_type == "point_box" and len(values) == 4:
            x, y = [value - coord_start for value in values[1:3]]
            box = [x, y, x, y]
        else:
            return
        device = (
            self._subpatch_positions.device
            if self._subpatch_positions is not None
            else self.presence_rescue_scale_logit.device
        )
        support = self._center_support(box, device=device).float()
        self._coverage = (
            support
            if self._coverage is None
            else torch.maximum(self._coverage.to(support), support)
        )
        self.last_diagnostics["distilled_accepted_geometry_coverage_sum"] = float(
            self._coverage.detach().sum().cpu()
        )

    def finish_forward(self) -> None:
        super().finish_forward()
        self._prior_selection_weights = ()
        self._prior_coordinate_residuals = ()
        self._posterior_prior_kls = ()
        self._prior_geometry_loss = None
        self._prior_decision_loss = None


def posterior_distilled_visual_event_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    controller: PosteriorDistilledVisualEventReentryPBD,
    *,
    route_loss_weight: float = 0.20,
    continuation_loss_weight: float = 0.30,
    presence_loss_weight: float = 0.20,
    coordinate_aux_weight: float = 0.30,
    posterior_distill_weight: float = 0.10,
    prior_geometry_weight: float = 0.20,
    prior_decision_weight: float = 0.15,
) -> torch.Tensor:
    """Native objective plus posterior-to-prior and prior-roll-in losses."""

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
    if controller.posterior_prior_loss is not None:
        loss = loss + float(posterior_distill_weight) * controller.posterior_prior_loss

    coord_start = int(controller.token_ids["coord_start_token_id"])
    coord_end = int(controller.token_ids["coord_end_token_id"])
    strength = controller.set_coordinate_strength().float()
    coordinate_losses: list[torch.Tensor] = []
    for transition, residual in zip(
        controller._set_transitions, controller._prior_coordinate_residuals
    ):
        for lane, position in enumerate(transition.row[1:5]):
            target_position = position + 1
            if target_position >= int(labels.shape[1]):
                continue
            target = int(labels[0, target_position])
            if target == IGNORE_INDEX or not coord_start <= target <= coord_end:
                continue
            base = F.linear(hidden_states[0, position], weight).float()[
                coord_start : coord_end + 1
            ]
            modified = base + strength[lane] * residual[lane].float()
            coordinate_losses.append(
                F.cross_entropy(
                    modified.unsqueeze(0),
                    modified.new_tensor(
                        [target - coord_start], dtype=torch.long
                    ),
                )
            )
    controller._prior_geometry_loss = (
        torch.stack(coordinate_losses).mean() if coordinate_losses else None
    )
    if controller._prior_geometry_loss is not None:
        loss = loss + float(prior_geometry_weight) * controller._prior_geometry_loss
    if controller.prior_decision_loss is not None:
        loss = loss + float(prior_decision_weight) * controller.prior_decision_loss
    controller.last_diagnostics.update(
        {
            "distilled_prior_geometry_loss": 0.0
            if controller._prior_geometry_loss is None
            else float(controller._prior_geometry_loss.detach()),
        }
    )
    return loss


def load_posterior_distilled_visual_event_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "posterior_distilled_visual_event_reentry":
        raise ValueError(
            "Not a posterior-distilled visual-event checkpoint: "
            f"{config.get('arm')!r}"
        )
    configure_lora_only(model, int(config["lora_rank"]))
    controller = PosteriorDistilledVisualEventReentryPBD(
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
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
