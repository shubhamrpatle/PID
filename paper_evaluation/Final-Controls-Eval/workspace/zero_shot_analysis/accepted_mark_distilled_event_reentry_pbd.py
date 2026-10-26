"""Posterior-distilled accepted-mark visual re-entry for LocateAnything PBD.

This controller closes the supervision gap left by geometry-coupled visual
events.  A GT-guided event posterior teaches the inference-time visual-event
prior, while both posterior and prior witnesses receive native coordinate and
continue/stop supervision.  Every accepted PBD or autoregressive geometry is
then written back before the next event is addressed.

The external behavior remains one cached MoonViT encoding, one Qwen decoder,
and LocateAnything's native point/HBB grammar.  There are no proposals, crops,
parallel decoder replicas, count tokens, slots, or fixed object-count caps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .geometry_coupled_visual_event_reentry_pbd import (
    GeometryCoupledVisualEventReentryPBD,
    add_geometry_dependence_losses,
)
from .pointrelay_pbd import configure_lora_only, restore_trainable_state
from .posterior_distilled_event_reentry_pbd import (
    PosteriorDistilledVisualEventReentryPBD,
    posterior_distilled_visual_event_ce,
)


class AcceptedMarkDistilledVisualEventReentryPBD(
    GeometryCoupledVisualEventReentryPBD,
    PosteriorDistilledVisualEventReentryPBD,
):
    """Distill the latent point prior and close the loop with accepted marks."""


def accepted_mark_distilled_visual_event_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    controller: AcceptedMarkDistilledVisualEventReentryPBD,
    *,
    route_loss_weight: float = 0.20,
    continuation_loss_weight: float = 0.30,
    presence_loss_weight: float = 0.20,
    coordinate_aux_weight: float = 0.30,
    posterior_distill_weight: float = 0.10,
    prior_geometry_weight: float = 0.20,
    prior_decision_weight: float = 0.15,
    geometry_cycle_weight: float = 0.20,
    counterfactual_event_weight: float = 0.20,
) -> torch.Tensor:
    """Native CE plus prior distillation and accepted-mark dependence."""

    loss = posterior_distilled_visual_event_ce(
        hidden_states,
        labels,
        weight,
        controller,
        route_loss_weight=route_loss_weight,
        continuation_loss_weight=continuation_loss_weight,
        presence_loss_weight=presence_loss_weight,
        coordinate_aux_weight=coordinate_aux_weight,
        posterior_distill_weight=posterior_distill_weight,
        prior_geometry_weight=prior_geometry_weight,
        prior_decision_weight=prior_decision_weight,
    )
    return add_geometry_dependence_losses(
        hidden_states,
        weight,
        controller,
        loss,
        geometry_cycle_weight=geometry_cycle_weight,
        counterfactual_event_weight=counterfactual_event_weight,
    )


def load_accepted_mark_distilled_visual_event_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "accepted_mark_distilled_visual_event_reentry":
        raise ValueError(
            "Not an accepted-mark-distilled visual-event checkpoint: "
            f"{config.get('arm')!r}"
        )
    configure_lora_only(model, int(config["lora_rank"]))
    controller = AcceptedMarkDistilledVisualEventReentryPBD(
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
        counterfactual_margin=float(config.get("counterfactual_margin", 0.25)),
        hard_negative_exclusion=float(config.get("hard_negative_exclusion", 0.05)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
