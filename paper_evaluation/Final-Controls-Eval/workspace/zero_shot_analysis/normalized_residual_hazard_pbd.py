"""Permutation-symmetrized residual event hazards for LocateAnything.

This controller keeps the native LocateAnything Qwen/PBD stream and its
six-token HBB grammar.  It replaces the heuristic event-versus-NULL gate with
a normalized categorical hazard over every residual visual event and one
learned STOP state.  Training uses the same residual state as inference:

* every remaining GT center is a valid event;
* the serialized current center supplies an exact Monte-Carlo ordering sample;
* accepted centers are removed without replacement; and
* STOP is supervised only after the class-conditioned target set is exhausted.

There are no count targets, proposal slots, crop loops, decoder replicas, or
fixed object slots.  Cardinality is determined by recurrent event/STOP
competition and remains bounded only by the visual lattice and decoder budget.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .point_marked_residual_event_pbd import PointMarkedResidualEventPBD
from .pointrelay_pbd import configure_lora_only, restore_trainable_state
from .residual_set_reentry_pbd import ResidualSetTransition


class NormalizedResidualHazardPBD(PointMarkedResidualEventPBD):
    """One normalized no-replacement event process drives native PBD marks."""

    def __init__(
        self,
        *,
        hazard_temperature: float = 1.0,
        hazard_exact_weight: float = 0.50,
        hazard_set_weight: float = 0.50,
        hazard_stop_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if hazard_temperature <= 0:
            raise ValueError("hazard_temperature must be positive")
        if min(hazard_exact_weight, hazard_set_weight, hazard_stop_weight) < 0:
            raise ValueError("hazard loss weights must be non-negative")
        if hazard_exact_weight + hazard_set_weight <= 0 or hazard_stop_weight <= 0:
            raise ValueError("positive-event and STOP hazard supervision are required")
        self.hazard_temperature = float(hazard_temperature)
        self.hazard_exact_weight = float(hazard_exact_weight)
        self.hazard_set_weight = float(hazard_set_weight)
        self.hazard_stop_weight = float(hazard_stop_weight)
        self._hazard_exact_loss: torch.Tensor | None = None
        self._hazard_set_loss: torch.Tensor | None = None
        self._hazard_stop_loss: torch.Tensor | None = None
        self._hazard_event_probabilities: tuple[torch.Tensor, ...] = ()
        self._hazard_stop_probabilities: tuple[torch.Tensor, ...] = ()

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        super()._reset_sample_state(preserve_pixel_field=preserve_pixel_field)
        self._clear_hazard_state()

    def _clear_hazard_state(self) -> None:
        self._hazard_exact_loss = None
        self._hazard_set_loss = None
        self._hazard_stop_loss = None
        self._hazard_event_probabilities = ()
        self._hazard_stop_probabilities = ()

    def _joint_log_partition(
        self, event_logits: torch.Tensor, null_score: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temperature = self.hazard_temperature
        scaled_events = event_logits.float() / temperature
        scaled_null = null_score.float() / temperature
        event_partition = torch.logsumexp(scaled_events, dim=0)
        joint_partition = torch.logaddexp(event_partition, scaled_null)
        return event_partition, joint_partition

    def _prepare_training_event_signals(
        self,
        hidden_states: torch.Tensor,
        transitions: tuple[ResidualSetTransition, ...],
    ) -> None:
        keys, values = self._field(hidden_states)
        positions = self._subpatch_positions
        if positions is None:
            raise RuntimeError("Hazard event positions are unavailable")
        positions = positions.to(hidden_states.device, dtype=torch.float32)

        selected_rows: list[torch.Tensor] = []
        moment_rows: list[torch.Tensor] = []
        negative_rows: list[torch.Tensor | None] = []
        negative_moments: list[torch.Tensor | None] = []
        continuation: list[torch.Tensor] = []
        presence: list[torch.Tensor | None] = []
        raw_gates: list[torch.Tensor] = []
        exact_losses: list[torch.Tensor] = []
        set_losses: list[torch.Tensor] = []
        stop_losses: list[torch.Tensor] = []
        heatmap_losses: list[torch.Tensor] = []
        hard_negative_losses: list[torch.Tensor] = []
        peak_recalls: list[torch.Tensor] = []
        event_probabilities: list[torch.Tensor] = []
        stop_probabilities: list[torch.Tensor] = []
        base_rows: list[torch.Tensor] = []
        diagnostic_field: torch.Tensor | None = None
        gate_scale = self.event_gate_scale()
        temperature = self.hazard_temperature

        for group in self._transition_groups(transitions):
            first = group[0]
            base, null_score = self._event_query_base(hidden_states, first, keys)
            target, center_indices = self._event_target(
                first.remaining_boxes, device=base.device
            )
            heatmap_losses.append(self._hierarchical_event_loss(base, target))
            base_rows.append(base)

            negative_mask = target.lt(0.5)
            if bool(negative_mask.any()):
                negative_values = base[negative_mask]
                positive_count = max(1, int(center_indices.numel()))
                hard_count = min(
                    int(negative_values.numel()), max(64, 4 * positive_count)
                )
                hard_values = negative_values.topk(hard_count).values
                hard_negative_losses.append(
                    F.binary_cross_entropy_with_logits(
                        hard_values, torch.zeros_like(hard_values)
                    )
                )
                masked = base.masked_fill(~negative_mask, -torch.inf)
                negative_index = masked.argmax()
            else:
                negative_index = base.argmin()

            numeric = [
                transition for transition in group if transition.target_box is not None
            ]
            numeric_indices = (
                torch.stack(
                    [
                        self._event_cell_index(
                            transition.target_box, device=base.device
                        )
                        for transition in numeric
                    ]
                )
                if numeric
                else base.new_zeros(0, dtype=torch.long)
            )
            if int(center_indices.numel()):
                top_count = min(int(center_indices.numel()), int(base.numel()))
                top_indices = base.topk(top_count).indices
                peak_recalls.append(
                    torch.isin(center_indices, top_indices).float().mean()
                )

            coverage = torch.zeros_like(base, dtype=torch.float32)
            numeric_cursor = 0
            negative_feature = values[negative_index]
            negative_spatial = torch.cat(
                (positions[negative_index], positions.new_zeros(2))
            )
            for transition in group:
                residual = base - self.event_suppression * coverage
                event_partition, joint_partition = self._joint_log_partition(
                    residual, null_score
                )
                event_probability = (event_partition - joint_partition).exp()
                stop_probability = (
                    null_score.float() / temperature - joint_partition
                ).exp()
                raw_gate = temperature * (
                    event_partition - null_score.float() / temperature
                )
                if diagnostic_field is None:
                    diagnostic_field = residual.detach()
                # These probabilities are diagnostics only.  Detaching them avoids
                # retaining one full hazard-partition graph per emitted object.
                event_probabilities.append(event_probability.detach())
                stop_probabilities.append(stop_probability.detach())

                if transition.target_box is not None:
                    event_index = numeric_indices[numeric_cursor]
                    remaining = numeric_indices[numeric_cursor:]
                    exact_losses.append(
                        joint_partition - residual[event_index].float() / temperature
                    )
                    if int(remaining.numel()):
                        remaining_scores = residual[remaining.unique()].float() / temperature
                        set_losses.append(
                            joint_partition - torch.logsumexp(remaining_scores, dim=0)
                        )
                    numeric_cursor += 1
                    selected = values[event_index]
                    moments = torch.cat(
                        (positions[event_index], positions.new_zeros(2))
                    )
                    negative_rows.append(negative_feature)
                    negative_moments.append(negative_spatial)
                    support = self._event_support_for_index(
                        event_index, device=residual.device
                    )
                    coverage = self._merge_coverage(coverage, support)
                else:
                    stop_losses.append(
                        joint_partition - null_score.float() / temperature
                    )
                    selected = negative_feature
                    moments = negative_spatial
                    negative_rows.append(None)
                    negative_moments.append(None)

                maximum = (
                    self.maximum_presence_bias
                    if transition.predecessor == "ref"
                    else self.maximum_continuation_bias
                )
                delta = maximum * torch.tanh(gate_scale * raw_gate)
                selected_rows.append(selected)
                moment_rows.append(moments)
                continuation.append(delta)
                presence.append(delta if transition.predecessor == "ref" else None)
                raw_gates.append(raw_gate)

        # The normalized hazard losses already supervise the full residual field;
        # inherited set logits are not consumed by this arm.  Keeping every field
        # would otherwise scale retained training state with objects x lattice.
        self._set_logits = ()
        self._set_selected_features = tuple(selected_rows)
        self._set_spatial_moments = tuple(moment_rows)
        self._set_coordinate_residuals = self._set_coordinate_signals(
            hidden_states, transitions, selected_rows, moment_rows
        )
        self._event_negative_features = tuple(negative_rows)
        self._event_negative_moments = tuple(negative_moments)
        self._continuation_deltas = tuple(continuation)
        self._presence_deltas = tuple(presence)
        self._event_base_rows = [row.detach() for row in base_rows]
        self._event_residual_rows = (
            [] if diagnostic_field is None else [diagnostic_field]
        )
        self._event_raw_gates = tuple(raw_gates)
        self._event_peak_recalls = tuple(peak_recalls)
        self._event_heatmap_loss = (
            torch.stack(heatmap_losses).mean() if heatmap_losses else None
        )
        self._event_hard_negative_loss = (
            torch.stack(hard_negative_losses).mean()
            if hard_negative_losses
            else None
        )
        self._hazard_exact_loss = (
            torch.stack(exact_losses).mean() if exact_losses else None
        )
        self._hazard_set_loss = torch.stack(set_losses).mean() if set_losses else None
        self._hazard_stop_loss = (
            torch.stack(stop_losses).mean() if stop_losses else None
        )
        hazard_terms: list[torch.Tensor] = []
        if self._hazard_exact_loss is not None and self.hazard_exact_weight:
            hazard_terms.append(self.hazard_exact_weight * self._hazard_exact_loss)
        if self._hazard_set_loss is not None and self.hazard_set_weight:
            hazard_terms.append(self.hazard_set_weight * self._hazard_set_loss)
        if self._hazard_stop_loss is not None and self.hazard_stop_weight:
            hazard_terms.append(self.hazard_stop_weight * self._hazard_stop_loss)
        normalizer = (
            self.hazard_exact_weight
            + self.hazard_set_weight
            + self.hazard_stop_weight
        )
        self._event_gate_loss = (
            torch.stack(hazard_terms).sum() / max(normalizer, 1e-8)
            if hazard_terms
            else None
        )
        self._hazard_event_probabilities = tuple(event_probabilities)
        self._hazard_stop_probabilities = tuple(stop_probabilities)
        self._set_route_loss = None
        self._continuation_loss = None
        self._presence_loss = None

    def _prepare_inference_event_signals(
        self,
        hidden_states: torch.Tensor,
        transition: ResidualSetTransition,
    ) -> None:
        keys, values = self._field(hidden_states)
        positions = self._subpatch_positions
        if positions is None:
            raise RuntimeError("Hazard event positions are unavailable")
        positions = positions.to(hidden_states.device, dtype=torch.float32)
        if transition.reset_coverage or self._event_base_logits is None:
            self._event_base_logits, self._event_null_score = self._event_query_base(
                hidden_states, transition, keys
            )
        if self._event_null_score is None:
            raise RuntimeError("Hazard NULL score was not initialized")
        coverage = (
            torch.zeros_like(self._event_base_logits)
            if self._coverage is None
            else self._coverage.to(self._event_base_logits).float()
        )
        residual = self._event_base_logits - self.event_suppression * coverage
        event_index = residual.argmax()
        support = self._event_support_for_index(event_index, device=residual.device)
        self._pending_inference_event = support
        selected = values[event_index]
        moments = torch.cat((positions[event_index], positions.new_zeros(2)))
        event_partition, joint_partition = self._joint_log_partition(
            residual, self._event_null_score
        )
        raw_gate = self.hazard_temperature * (
            event_partition - self._event_null_score.float() / self.hazard_temperature
        )
        event_probability = (event_partition - joint_partition).exp()
        stop_probability = (
            self._event_null_score.float() / self.hazard_temperature - joint_partition
        ).exp()
        maximum = (
            self.maximum_presence_bias
            if transition.predecessor == "ref"
            else self.maximum_continuation_bias
        )
        delta = maximum * torch.tanh(self.event_gate_scale() * raw_gate)

        self._set_logits = (residual,)
        self._set_selected_features = (selected,)
        self._set_spatial_moments = (moments,)
        self._set_coordinate_residuals = self._set_coordinate_signals(
            hidden_states, (transition,), (selected,), (moments,)
        )
        self._continuation_deltas = (delta,)
        self._presence_deltas = (
            delta if transition.predecessor == "ref" else None,
        )
        self._event_base_rows = [self._event_base_logits]
        self._event_residual_rows = [residual]
        self._event_raw_gates = (raw_gate,)
        self._event_negative_features = (None,)
        self._event_negative_moments = (None,)
        self._hazard_event_probabilities = (event_probability,)
        self._hazard_stop_probabilities = (stop_probability,)
        self._set_route_loss = None
        self._continuation_loss = None
        self._presence_loss = None

    def _update_event_diagnostics(self) -> None:
        super()._update_event_diagnostics()
        self.last_diagnostics.update(
            {
                "hazard_exact_loss": 0.0
                if self._hazard_exact_loss is None
                else float(self._hazard_exact_loss.detach()),
                "hazard_set_loss": 0.0
                if self._hazard_set_loss is None
                else float(self._hazard_set_loss.detach()),
                "hazard_stop_loss": 0.0
                if self._hazard_stop_loss is None
                else float(self._hazard_stop_loss.detach()),
                "hazard_event_probability": 0.0
                if not self._hazard_event_probabilities
                else float(torch.stack(self._hazard_event_probabilities).mean().detach()),
                "hazard_stop_probability": 0.0
                if not self._hazard_stop_probabilities
                else float(torch.stack(self._hazard_stop_probabilities).mean().detach()),
                "hazard_temperature": float(self.hazard_temperature),
            }
        )

    def finish_forward(self) -> None:
        super().finish_forward()
        self._clear_hazard_state()


def load_normalized_residual_hazard_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if not bool(config.get("normalized_residual_hazard")):
        raise ValueError("Checkpoint does not enable normalized residual hazard")
    configure_lora_only(model, int(config["lora_rank"]))
    controller = NormalizedResidualHazardPBD(
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
        initial_set_coordinate_strength=float(config.get("initial_set_coordinate_strength", 0.50)),
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
        maximum_event_radius=int(config.get("maximum_event_radius", 0)),
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
        hazard_temperature=float(config.get("hazard_temperature", 1.0)),
        hazard_exact_weight=float(config.get("hazard_exact_weight", 0.50)),
        hazard_set_weight=float(config.get("hazard_set_weight", 0.50)),
        hazard_stop_weight=float(config.get("hazard_stop_weight", 1.0)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
