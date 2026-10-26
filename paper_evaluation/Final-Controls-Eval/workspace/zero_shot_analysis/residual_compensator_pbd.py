"""Multiplicity-aware residual point-process decoding for LocateAnything.

The controller keeps LocateAnything's native Qwen stream and atomic PBD
geometry grammar.  A phrase-conditioned visual field predicts a non-negative
event rate for every fine spatial cell.  Its integral is the point-process
compensator: it determines whether another object remains, without a count
token, fixed query slots, or a hand-set output cap.  Each accepted native
geometry block subtracts exactly one unit of rate at the selected event, so
multiple instances may share a cell and decoding remains open ended.

Training is permutation invariant at the set level.  The spatial rate uses a
Poisson point-process likelihood, while a remaining-set likelihood and the
Poisson survival function supervise recurrent continue/stop decisions.  The
selected event conditions the existing six-token PBD box mark; geometry is
still emitted by the ordinary language decoder.
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


def _stable_continue_logit(integrated_rate: torch.Tensor) -> torch.Tensor:
    """Return log odds of at least one event for a Poisson rate."""

    value = integrated_rate.float().clamp_min(1e-8)
    # P(N > 0) / P(N = 0) = exp(Lambda) - 1.  The large-rate branch
    # avoids overflow while preserving the exact asymptotic log odds.
    return torch.where(value > 20.0, value, torch.log(torch.expm1(value)))


class ResidualCompensatorPBD(PointMarkedResidualEventPBD):
    """A residual marked point process inside the native PBD decoder."""

    def __init__(
        self,
        *,
        initial_log_rate_bias: float = -6.0,
        minimum_cell_rate: float = 1e-8,
        compensator_point_weight: float = 1.0,
        compensator_set_weight: float = 0.5,
        compensator_survival_weight: float = 1.0,
        event_selection_mode: str = "argmax",
        event_sampling_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if minimum_cell_rate <= 0:
            raise ValueError("minimum_cell_rate must be positive")
        if min(
            compensator_point_weight,
            compensator_set_weight,
            compensator_survival_weight,
        ) < 0:
            raise ValueError("compensator loss weights must be non-negative")
        if compensator_point_weight + compensator_survival_weight <= 0:
            raise ValueError("point-process and survival supervision are required")
        if event_selection_mode not in {"argmax", "sample"}:
            raise ValueError("event_selection_mode must be 'argmax' or 'sample'")
        if event_sampling_temperature <= 0:
            raise ValueError("event_sampling_temperature must be positive")

        self.log_rate_bias = nn.Parameter(torch.tensor(float(initial_log_rate_bias)))
        # STOP is determined by the point-process survival function.  Keeping
        # the inherited learned NULL head trainable would add unused parameters
        # and make matched-capacity accounting needlessly ambiguous.
        self.null_event_head.requires_grad_(False)
        self.null_event_bias.requires_grad_(False)
        self.minimum_cell_rate = float(minimum_cell_rate)
        self.compensator_point_weight = float(compensator_point_weight)
        self.compensator_set_weight = float(compensator_set_weight)
        self.compensator_survival_weight = float(compensator_survival_weight)
        self.event_selection_mode = str(event_selection_mode)
        self.event_sampling_temperature = float(event_sampling_temperature)

        self._compensator_base_rates: torch.Tensor | None = None
        self._compensator_point_loss: torch.Tensor | None = None
        self._compensator_set_loss: torch.Tensor | None = None
        self._compensator_survival_loss: torch.Tensor | None = None
        self._compensator_integrals: tuple[torch.Tensor, ...] = ()
        self._compensator_event_probabilities: tuple[torch.Tensor, ...] = ()
        self._compensator_stop_probabilities: tuple[torch.Tensor, ...] = ()

    @property
    def compensator_point_loss(self) -> torch.Tensor | None:
        return self._compensator_point_loss

    @property
    def compensator_set_loss(self) -> torch.Tensor | None:
        return self._compensator_set_loss

    @property
    def compensator_survival_loss(self) -> torch.Tensor | None:
        return self._compensator_survival_loss

    def _clear_compensator_state(self) -> None:
        self._compensator_base_rates = None
        self._compensator_point_loss = None
        self._compensator_set_loss = None
        self._compensator_survival_loss = None
        self._compensator_integrals = ()
        self._compensator_event_probabilities = ()
        self._compensator_stop_probabilities = ()

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        super()._reset_sample_state(preserve_pixel_field=preserve_pixel_field)
        self._clear_compensator_state()

    def _merge_coverage(
        self,
        current: torch.Tensor | None,
        support: torch.Tensor,
    ) -> torch.Tensor:
        """Consume one event unit, retaining same-cell multiplicity."""

        support = support.detach().float()
        if current is None:
            return support
        return current.to(support).float() + support

    def _event_indices(
        self,
        boxes: Sequence[Sequence[int]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if not boxes:
            return torch.zeros(0, device=device, dtype=torch.long)
        return torch.stack(
            [self._event_cell_index(box, device=device) for box in boxes]
        )

    def _cell_rates(self, base_logits: torch.Tensor) -> torch.Tensor:
        return F.softplus(base_logits.float() + self.log_rate_bias.float()).clamp_min(
            self.minimum_cell_rate
        )

    def _residual_rates(
        self, rates: torch.Tensor, consumed: torch.Tensor | None
    ) -> torch.Tensor:
        if consumed is None:
            return rates
        return (rates - consumed.to(rates).float()).clamp_min(
            self.minimum_cell_rate
        )

    def _selection_index(self, residual_rates: torch.Tensor) -> torch.Tensor:
        if self.event_selection_mode == "argmax":
            return residual_rates.argmax()
        probability = residual_rates.pow(1.0 / self.event_sampling_temperature)
        probability = probability / probability.sum().clamp_min(1e-12)
        return torch.multinomial(probability, num_samples=1).reshape(())

    @staticmethod
    def _poisson_survival_loss(
        integrated_rate: torch.Tensor, *, has_event: bool
    ) -> torch.Tensor:
        value = integrated_rate.float().clamp_min(1e-8)
        if not has_event:
            # -log P(N=0) for N ~ Poisson(Lambda).
            return value
        # -log(1 - exp(-Lambda)), stable for both small and large rates.
        return -torch.log(-torch.expm1(-value)).clamp_min(1e-12)

    def _prepare_training_event_signals(
        self,
        hidden_states: torch.Tensor,
        transitions: tuple[ResidualSetTransition, ...],
    ) -> None:
        keys, values = self._field(hidden_states)
        positions = self._subpatch_positions
        if positions is None:
            raise RuntimeError("Residual-compensator positions are unavailable")
        positions = positions.to(hidden_states.device, dtype=torch.float32)

        selected_rows: list[torch.Tensor] = []
        moment_rows: list[torch.Tensor] = []
        negative_rows: list[torch.Tensor | None] = []
        negative_moments: list[torch.Tensor | None] = []
        continuation: list[torch.Tensor] = []
        presence: list[torch.Tensor | None] = []
        raw_gates: list[torch.Tensor] = []
        point_losses: list[torch.Tensor] = []
        set_losses: list[torch.Tensor] = []
        survival_losses: list[torch.Tensor] = []
        heatmap_losses: list[torch.Tensor] = []
        hard_negative_losses: list[torch.Tensor] = []
        peak_recalls: list[torch.Tensor] = []
        integrals: list[torch.Tensor] = []
        event_probabilities: list[torch.Tensor] = []
        stop_probabilities: list[torch.Tensor] = []
        base_rows: list[torch.Tensor] = []
        residual_rows: list[torch.Tensor] = []
        gate_scale = self.event_gate_scale()

        for group in self._transition_groups(transitions):
            first = group[0]
            base, _unused_null = self._event_query_base(hidden_states, first, keys)
            rates = self._cell_rates(base)
            all_boxes = tuple(first.remaining_boxes)
            all_indices = self._event_indices(all_boxes, device=base.device)
            target, unique_centers = self._event_target(all_boxes, device=base.device)
            heatmap_losses.append(self._hierarchical_event_loss(base, target))
            base_rows.append(base)
            residual_rows.append(rates)

            target_count = max(1, int(all_indices.numel()))
            if int(all_indices.numel()):
                # Discretized Poisson point-process NLL. Repeated indices are
                # intentionally retained, so the optimal rate at a shared cell
                # equals its object multiplicity.
                point_losses.append(
                    rates.sum() / float(target_count)
                    - rates[all_indices].log().mean()
                )
                top_count = min(int(unique_centers.numel()), int(rates.numel()))
                if top_count:
                    peak_recalls.append(
                        torch.isin(
                            unique_centers, rates.topk(top_count).indices
                        ).float().mean()
                    )
            else:
                point_losses.append(rates.sum())

            negative_mask = target.lt(0.05)
            if bool(negative_mask.any()):
                negative_values = base[negative_mask]
                hard_count = min(
                    int(negative_values.numel()), max(64, 4 * target_count)
                )
                hard_negative_losses.append(
                    F.binary_cross_entropy_with_logits(
                        negative_values.topk(hard_count).values,
                        torch.zeros(hard_count, device=base.device),
                    )
                )
                negative_index = base.masked_fill(~negative_mask, -torch.inf).argmax()
            else:
                negative_index = base.argmin()
            negative_feature = values[negative_index]
            negative_spatial = torch.cat(
                (positions[negative_index], positions.new_zeros(2))
            )

            consumed = torch.zeros_like(rates)
            numeric = [item for item in group if item.target_box is not None]
            numeric_indices = self._event_indices(
                [item.target_box for item in numeric if item.target_box is not None],
                device=base.device,
            )
            numeric_cursor = 0
            for transition in group:
                residual = self._residual_rates(rates, consumed)
                integrated = residual.sum()
                has_event = transition.target_box is not None
                raw_gate = _stable_continue_logit(integrated)
                integrals.append(integrated.detach())
                stop_probability = torch.exp(-integrated).clamp(0.0, 1.0)
                event_probabilities.append((1.0 - stop_probability).detach())
                stop_probabilities.append(stop_probability.detach())
                survival_losses.append(
                    self._poisson_survival_loss(integrated, has_event=has_event)
                )

                if has_event:
                    event_index = numeric_indices[numeric_cursor]
                    remaining = numeric_indices[numeric_cursor:]
                    remaining_mass = residual[remaining.unique()].sum()
                    set_losses.append(
                        integrated.log()
                        - remaining_mass.clamp_min(self.minimum_cell_rate).log()
                    )
                    numeric_cursor += 1
                    selected = values[event_index]
                    moments = torch.cat(
                        (positions[event_index], positions.new_zeros(2))
                    )
                    negative_rows.append(negative_feature)
                    negative_moments.append(negative_spatial)
                    consumed = consumed + self._event_support_for_index(
                        event_index, device=base.device
                    )
                else:
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
        self._event_residual_rows = [row.detach() for row in residual_rows]
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
        self._compensator_point_loss = (
            torch.stack(point_losses).mean() if point_losses else None
        )
        self._compensator_set_loss = (
            torch.stack(set_losses).mean() if set_losses else None
        )
        self._compensator_survival_loss = (
            torch.stack(survival_losses).mean() if survival_losses else None
        )
        weighted: list[torch.Tensor] = []
        denominator = 0.0
        for value, weight in (
            (self._compensator_point_loss, self.compensator_point_weight),
            (self._compensator_set_loss, self.compensator_set_weight),
            (self._compensator_survival_loss, self.compensator_survival_weight),
        ):
            if value is not None and weight:
                weighted.append(weight * value)
                denominator += weight
        self._event_gate_loss = (
            torch.stack(weighted).sum() / max(denominator, 1e-8)
            if weighted
            else None
        )
        self._compensator_integrals = tuple(integrals)
        self._compensator_event_probabilities = tuple(event_probabilities)
        self._compensator_stop_probabilities = tuple(stop_probabilities)
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
            raise RuntimeError("Residual-compensator positions are unavailable")
        positions = positions.to(hidden_states.device, dtype=torch.float32)
        if transition.reset_coverage or self._compensator_base_rates is None:
            base, _unused_null = self._event_query_base(hidden_states, transition, keys)
            self._event_base_logits = base
            self._compensator_base_rates = self._cell_rates(base)
        rates = self._compensator_base_rates
        residual = self._residual_rates(rates, self._coverage)
        event_index = self._selection_index(residual)
        self._pending_inference_event = self._event_support_for_index(
            event_index, device=residual.device
        )
        selected = values[event_index]
        moments = torch.cat((positions[event_index], positions.new_zeros(2)))
        integrated = residual.sum()
        raw_gate = _stable_continue_logit(integrated)
        stop_probability = torch.exp(-integrated).clamp(0.0, 1.0)
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
        self._compensator_integrals = (integrated.detach(),)
        self._compensator_event_probabilities = ((1.0 - stop_probability).detach(),)
        self._compensator_stop_probabilities = (stop_probability.detach(),)
        self._set_route_loss = None
        self._continuation_loss = None
        self._presence_loss = None

    def observe_generated_block(self, out_type: str, tokens: Any) -> None:
        if not self._training_context and out_type == "ref_object":
            self._compensator_base_rates = None
        super().observe_generated_block(out_type, tokens)

    def _update_event_diagnostics(self) -> None:
        super()._update_event_diagnostics()
        self.last_diagnostics.update(
            {
                "compensator_point_loss": 0.0
                if self._compensator_point_loss is None
                else float(self._compensator_point_loss.detach()),
                "compensator_set_loss": 0.0
                if self._compensator_set_loss is None
                else float(self._compensator_set_loss.detach()),
                "compensator_survival_loss": 0.0
                if self._compensator_survival_loss is None
                else float(self._compensator_survival_loss.detach()),
                "compensator_integrated_rate": 0.0
                if not self._compensator_integrals
                else float(torch.stack(self._compensator_integrals).mean()),
                "compensator_event_probability": 0.0
                if not self._compensator_event_probabilities
                else float(torch.stack(self._compensator_event_probabilities).mean()),
                "compensator_stop_probability": 0.0
                if not self._compensator_stop_probabilities
                else float(torch.stack(self._compensator_stop_probabilities).mean()),
                "compensator_log_rate_bias": float(self.log_rate_bias.detach()),
            }
        )

    def finish_forward(self) -> None:
        super().finish_forward()
        self._clear_compensator_state()


def load_residual_compensator_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if not bool(config.get("residual_compensator")):
        raise ValueError("Checkpoint does not enable the residual compensator")
    configure_lora_only(model, int(config["lora_rank"]))
    controller = ResidualCompensatorPBD(
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
        initial_log_rate_bias=float(config.get("initial_log_rate_bias", -6.0)),
        minimum_cell_rate=float(config.get("minimum_cell_rate", 1e-8)),
        compensator_point_weight=float(config.get("compensator_point_weight", 1.0)),
        compensator_set_weight=float(config.get("compensator_set_weight", 0.5)),
        compensator_survival_weight=float(config.get("compensator_survival_weight", 1.0)),
        event_selection_mode=str(config.get("event_selection_mode", "argmax")),
        event_sampling_temperature=float(config.get("event_sampling_temperature", 1.0)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
