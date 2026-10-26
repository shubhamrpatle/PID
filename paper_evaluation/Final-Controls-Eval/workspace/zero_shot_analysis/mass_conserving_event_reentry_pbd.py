"""Mass-conserving latent visual events for generative localization.

This extends accepted-mark visual re-entry in three ways required by very
dense scenes:

* recurrent memory stores additive consumed event mass instead of a binary
  covered mask, so multiple instances may originate from one visual cell;
* an unnormalized query-conditioned intensity is supervised against the full
  native target set and therefore retains multiplicity;
* a learned null event competes with residual spatial mass to control native
  continue/stop decisions without slots, count tokens, or a fixed object cap.

The external decoder remains LocateAnything's single Qwen/PBD stream.  One
accepted geometry block consumes one unit of latent mass and the next block is
generated from the updated field.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .accepted_mark_distilled_event_reentry_pbd import (
    AcceptedMarkDistilledVisualEventReentryPBD,
    accepted_mark_distilled_visual_event_ce,
)
from .distribution_addressed_reentry_pbd import IGNORE_INDEX
from .pointrelay_pbd import RelayContext, configure_lora_only, restore_trainable_state
from .residual_set_reentry_pbd import (
    ResidualSetTransition,
    residual_set_transitions_from_training,
)


def complete_residual_transitions_from_training(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    token_ids: Mapping[str, int],
) -> tuple[tuple[ResidualSetTransition, ...], tuple[int, ...]]:
    """Attach complete native target groups to capped auxiliary MTP rows.

    MTP copies may be stratified/capped to fit the sequence budget, but the
    original assistant answer is never truncated.  This routine recovers every
    native box first, then maps each retained MTP transition back to its group.
    """

    raw = residual_set_transitions_from_training(
        input_ids, labels, token_ids=token_ids
    )
    ids = input_ids[0]
    targets = labels[0]
    ref_end = int(token_ids["ref_end_token_id"])
    box_start = int(token_ids["box_start_token_id"])
    box_end = int(token_ids["box_end_token_id"])
    coord_start = int(token_ids["coord_start_token_id"])
    coord_end = int(token_ids["coord_end_token_id"])

    groups: list[list[tuple[int, int, int, int]]] = []
    active_group = -1
    position = 0
    while position < int(ids.numel()):
        if int(targets[position]) != IGNORE_INDEX and int(ids[position]) == ref_end:
            groups.append([])
            active_group += 1
            position += 1
            continue
        if active_group >= 0 and position + 6 <= int(ids.numel()):
            block = tuple(int(value) for value in ids[position : position + 6].tolist())
            supervised = bool(
                targets[position : position + 6].ne(IGNORE_INDEX).all()
            )
            numeric = (
                supervised
                and block[0] == box_start
                and block[5] == box_end
                and all(coord_start <= value <= coord_end for value in block[1:5])
            )
            if numeric:
                groups[active_group].append(
                    tuple(value - coord_start for value in block[1:5])
                )
                position += 6
                continue
        position += 1

    completed: list[ResidualSetTransition] = []
    group_ids: list[int] = []
    group_cursor = -1
    object_cursor: dict[int, int] = {}
    previous_group = -1
    for transition in raw:
        if transition.reset_coverage:
            group_cursor += 1
        group_cursor = min(group_cursor, len(groups) - 1)
        group = groups[group_cursor] if 0 <= group_cursor < len(groups) else []
        target_box = transition.target_box
        remaining: tuple[tuple[int, int, int, int], ...] = ()
        if target_box is not None and group:
            start = object_cursor.get(group_cursor, 0)
            match = next(
                (index for index in range(start, len(group)) if group[index] == target_box),
                None,
            )
            if match is None:
                match = next(
                    (index for index, box in enumerate(group) if box == target_box),
                    None,
                )
            if match is not None:
                remaining = tuple(group[match:])
                object_cursor[group_cursor] = match + 1
        reset = bool(transition.reset_coverage or group_cursor != previous_group)
        completed.append(
            ResidualSetTransition(
                row=transition.row,
                predecessor=transition.predecessor,
                reset_coverage=reset,
                target_box=target_box,
                remaining_boxes=remaining,
            )
        )
        group_ids.append(group_cursor)
        previous_group = group_cursor
    return tuple(completed), tuple(group_ids)


class MassConservingVisualEventReentryPBD(
    AcceptedMarkDistilledVisualEventReentryPBD
):
    """Posterior-distilled native PBD with residual event mass and null."""

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
        initial_log_intensity_bias: float = -6.0,
        maximum_null_event_bias: float = 3.0,
        initial_mass_route_modulation: float = 0.10,
        maximum_mass_route_modulation: float = 2.0,
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
            counterfactual_margin=counterfactual_margin,
            hard_negative_exclusion=hard_negative_exclusion,
        )
        if maximum_null_event_bias <= 0:
            raise ValueError("maximum_null_event_bias must be positive")
        if maximum_mass_route_modulation <= 0:
            raise ValueError("maximum_mass_route_modulation must be positive")
        if not 0 < initial_mass_route_modulation < maximum_mass_route_modulation:
            raise ValueError(
                "initial_mass_route_modulation must be between zero and its maximum"
            )
        self.log_intensity_bias = nn.Parameter(
            torch.tensor(float(initial_log_intensity_bias))
        )
        self.null_event_head = nn.Linear(hidden_size, 1)
        self.null_event_bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.null_event_head.weight)
        nn.init.zeros_(self.null_event_head.bias)
        self.maximum_null_event_bias = float(maximum_null_event_bias)
        self.maximum_mass_route_modulation = float(maximum_mass_route_modulation)
        modulation_ratio = (
            float(initial_mass_route_modulation)
            / self.maximum_mass_route_modulation
        )
        self.mass_route_modulation_logit = nn.Parameter(
            torch.tensor(math.log(modulation_ratio / (1.0 - modulation_ratio)))
        )
        self._mass_group_by_row: dict[tuple[int, ...], int] = {}
        self._mass_point_process_loss: torch.Tensor | None = None
        self._mass_count_loss: torch.Tensor | None = None
        self._null_event_loss: torch.Tensor | None = None
        self._mass_expected_counts: tuple[torch.Tensor, ...] = ()
        self._mass_residual_rates: tuple[torch.Tensor, ...] = ()
        self._mass_working_rates: list[torch.Tensor] = []
        self._mass_base_intensity: torch.Tensor | None = None
        self._mass_field_cursor = 0
        self._observed_mass_block: tuple[int, ...] | None = None
        self.exact_available_mass_consumption = os.environ.get(
            "LOCANY_MASS_EXACT_CONSUMPTION", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}

    @property
    def mass_point_process_loss(self) -> torch.Tensor | None:
        return self._mass_point_process_loss

    @property
    def mass_count_loss(self) -> torch.Tensor | None:
        return self._mass_count_loss

    @property
    def null_event_loss(self) -> torch.Tensor | None:
        return self._null_event_loss

    def mass_route_modulation(self) -> torch.Tensor:
        return self.maximum_mass_route_modulation * torch.sigmoid(
            self.mass_route_modulation_logit
        )

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        super()._reset_sample_state(preserve_pixel_field=preserve_pixel_field)
        self._mass_group_by_row = {}
        self._mass_point_process_loss = None
        self._mass_count_loss = None
        self._null_event_loss = None
        self._mass_expected_counts = ()
        self._mass_residual_rates = ()
        self._mass_working_rates = []
        self._mass_base_intensity = None
        self._mass_field_cursor = 0
        self._observed_mass_block = None

    def _merge_coverage(
        self,
        current: torch.Tensor | None,
        support: torch.Tensor,
    ) -> torch.Tensor:
        support = support.float()
        support = support / support.sum().clamp_min(1e-12)
        if self.exact_available_mass_consumption and self._mass_base_intensity is not None:
            base = self._mass_base_intensity.detach().to(support).float()
            consumed = torch.zeros_like(base) if current is None else current.to(base).float()
            available = (base - consumed).clamp_min(0.0)
            available_mass = available.sum()
            if float(available_mass.detach()) <= 1e-8:
                return consumed
            allocation = support * available
            if float(allocation.sum().detach()) <= 1e-8:
                allocation = available
            allocation = allocation / allocation.sum().clamp_min(1e-12)
            allocation = allocation * available_mass.clamp(max=1.0)
            return consumed + allocation
        if current is None:
            return support
        return current.to(support) + support

    def _residualize_field_logits(
        self,
        field_logits: torch.Tensor,
        coverage: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        """Route over a conserved phrase-level residual event measure.

        The first transition after ``<ref>`` creates one unnormalized event
        intensity. Subsequent autoregressive states may rerank that field, but
        they cannot change its integrated residual mass. This separates
        cardinality from Qwen's sequential choice of the next geometry.
        """

        del strength
        transition = (
            self._set_transitions[self._mass_field_cursor]
            if self._mass_field_cursor < len(self._set_transitions)
            else None
        )
        self._mass_field_cursor += 1
        raw = field_logits.float()
        # Teacher-forced transitions explicitly mark a new reference group.
        # Autoregressive inference has no transition table, so
        # ``transition is None`` is the normal case and must not recreate mass
        # after every emitted geometry block. New inference references clear
        # ``_mass_base_intensity`` in ``observe_generated_block`` below.
        reset = (
            (transition is not None and transition.reset_coverage)
            or self._mass_base_intensity is None
            or int(self._mass_base_intensity.numel()) != int(raw.numel())
        )
        if reset:
            self._mass_base_intensity = F.softplus(
                raw + self.log_intensity_bias.float()
            )
        residual_rates = (
            self._mass_base_intensity - coverage.float()
        ).clamp_min(1e-8)
        self._mass_working_rates.append(residual_rates)

        centered = raw - raw.mean()
        dynamic_route = centered / centered.square().mean().sqrt().clamp_min(1e-4)
        dynamic_route = dynamic_route.clamp(-4.0, 4.0)
        selection_logits = residual_rates.log()
        selection_logits = (
            selection_logits
            + self.mass_route_modulation().float() * dynamic_route
        )
        return selection_logits.to(field_logits.dtype)

    def _coverage_write_support(
        self,
        transition: ResidualSetTransition,
        selection_weights: torch.Tensor,
    ) -> torch.Tensor | None:
        support = super()._coverage_write_support(transition, selection_weights)
        if support is None:
            return None
        return support / support.sum().clamp_min(1e-12)

    def _route_target_boxes(
        self, transition: ResidualSetTransition
    ) -> tuple[tuple[int, int, int, int], ...]:
        # Current-mark routing stays cheap even when the complete residual set
        # contains 1,000 objects. Multiplicity is supervised by the intensity.
        return () if transition.target_box is None else (transition.target_box,)

    def _commit_mass_event(
        self,
        geometry_support: torch.Tensor,
        *,
        observed_block: tuple[int, ...] | None,
    ) -> None:
        """Consume exactly one event for one accepted native geometry block."""

        geometry_support = geometry_support.detach().float()
        joint = geometry_support
        if self._pending_inference_event is not None:
            pending = self._pending_inference_event.detach().to(joint).float()
            # In the exact ablation the accepted geometry is the event mark,
            # while the staged latent address identifies which event is
            # consumed. This avoids wasting conserved mass where no event rate
            # remains. The default path is preserved for the preregistered run.
            joint = (
                pending
                if self.exact_available_mass_consumption
                else torch.maximum(joint, pending)
            )
        joint = joint / joint.sum().clamp_min(1e-12)
        self._coverage = self._merge_coverage(self._coverage, joint)
        self._pending_inference_event = None
        self._observed_inference_events += 1
        self.geometry_writeback_count += 1
        self._observed_mass_block = observed_block
        self._update_writeback_diagnostics()

    def observe_generated_block(self, out_type: str, tokens: Any) -> None:
        """Fuse an accepted latent address and geometry into one consumed mark."""

        if self._training_context:
            return
        if out_type == "ref_object":
            self._coverage = None
            self._mass_base_intensity = None
            self._mass_field_cursor = 0
            self._pending_inference_event = None
            self._observed_mass_block = None
            self._observed_inference_refs += 1
            self._update_writeback_diagnostics()
            return
        if out_type in {"empty_box", "error_box", "im_end"}:
            self._pending_inference_event = None
            self._observed_mass_block = None
            self._update_writeback_diagnostics()
            return
        if out_type not in {"coord_box", "point_box"}:
            return
        values = (
            [int(value) for value in tokens.detach().reshape(-1).tolist()]
            if isinstance(tokens, torch.Tensor)
            else [int(value.item() if hasattr(value, "item") else value) for value in tokens]
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
            else self.null_event_bias.device
        )
        geometry = self._center_support(box, device=device)
        self._commit_mass_event(geometry, observed_block=tuple(values))

    def _write_previous_geometry(self, input_ids: torch.Tensor, row_start: int) -> None:
        """Handle AR fallback while de-duplicating an observed fast PBD block."""

        if self._subpatch_positions is None or row_start < 6:
            return
        block = tuple(
            int(value) for value in input_ids[0, row_start - 6 : row_start].tolist()
        )
        box_start = int(self.token_ids["box_start_token_id"])
        box_end = int(self.token_ids["box_end_token_id"])
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        valid = (
            len(block) == 6
            and block[0] == box_start
            and block[-1] == box_end
            and all(coord_start <= value <= coord_end for value in block[1:5])
        )
        if not valid:
            return
        if block == self._observed_mass_block:
            self._observed_mass_block = None
            return
        key = (self._inference_position_key, int(row_start), block)
        if key == self._last_written_geometry_key:
            return
        box = tuple(value - coord_start for value in block[1:5])
        geometry = self._center_support(box, device=self._subpatch_positions.device)
        self._last_written_geometry_key = key
        self._commit_mass_event(geometry, observed_block=None)

    def set_training_context(
        self,
        context: RelayContext,
        image_grid_hws: Any,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> None:
        if input_ids is None or labels is None:
            raise ValueError("Mass-conserving training requires input_ids and labels")
        super().set_training_context(
            context, image_grid_hws, input_ids=input_ids, labels=labels
        )
        transitions, group_ids = complete_residual_transitions_from_training(
            input_ids, labels, token_ids=self.token_ids
        )
        self._set_transitions = transitions
        self._mass_group_by_row = {
            tuple(transition.row): group_id
            for transition, group_id in zip(transitions, group_ids)
        }

    def _local_rates(
        self,
        boxes: Sequence[Sequence[int]],
        rates: torch.Tensor,
    ) -> torch.Tensor:
        values: list[torch.Tensor] = []
        for start in range(0, len(boxes), 128):
            supports = torch.stack(
                [
                    self._center_support(box, device=rates.device)
                    for box in boxes[start : start + 128]
                ]
            ).float()
            values.append(torch.matmul(supports, rates))
        return torch.cat(values).clamp_min(1e-8)

    def prepare_set_signals(self, hidden_states: torch.Tensor) -> None:
        self._mass_field_cursor = 0
        self._mass_working_rates = []
        super().prepare_set_signals(hidden_states)
        self._mass_residual_rates = tuple(self._mass_working_rates)
        if not self._set_transitions or not self._set_logits:
            return

        point_process: list[torch.Tensor] = []
        count_losses: list[torch.Tensor] = []
        positive_null: list[torch.Tensor] = []
        negative_null: list[torch.Tensor] = []
        expected_counts: list[torch.Tensor] = []
        null_deltas: list[torch.Tensor] = []
        seen_groups: set[int] = set()

        for transition, rates in zip(
            self._set_transitions, self._mass_residual_rates
        ):
            # Count, point-process and null losses use the conserved measure,
            # not Qwen's dynamic spatial reranking logits.
            rates = rates.float()
            expected = rates.sum()
            expected_counts.append(expected)
            null_score = (
                self.null_event_head(
                    hidden_states[0, transition.row[0]].to(
                        self.null_event_head.weight.dtype
                    )
                ).float().reshape(())
                + self.null_event_bias.float()
            )
            log_odds = torch.log(expected.clamp_min(1e-8)) - null_score
            has_remaining = bool(transition.remaining_boxes)
            decision = F.binary_cross_entropy_with_logits(
                log_odds, log_odds.new_tensor(1.0 if has_remaining else 0.0)
            )
            (positive_null if has_remaining else negative_null).append(decision)
            null_deltas.append(
                self.maximum_null_event_bias * torch.tanh(log_odds)
            )

            group_id = self._mass_group_by_row.get(tuple(transition.row), -1)
            if group_id in seen_groups:
                continue
            seen_groups.add(group_id)
            boxes = transition.remaining_boxes
            target_count = len(boxes)
            count_losses.append(
                F.smooth_l1_loss(
                    torch.log1p(expected),
                    expected.new_tensor(math.log1p(float(target_count))),
                )
            )
            if boxes:
                local = self._local_rates(boxes, rates)
                point_process.append(
                    expected / float(target_count) - local.log().mean()
                )

        groups = [
            torch.stack(group).mean()
            for group in (positive_null, negative_null)
            if group
        ]
        self._mass_point_process_loss = (
            torch.stack(point_process).mean() if point_process else None
        )
        self._mass_count_loss = (
            torch.stack(count_losses).mean() if count_losses else None
        )
        self._null_event_loss = torch.stack(groups).mean() if groups else None
        self._mass_expected_counts = tuple(expected_counts)
        self._continuation_deltas = tuple(
            original + extra
            for original, extra in zip(self._continuation_deltas, null_deltas)
        )
        self._presence_deltas = tuple(
            None if original is None else original + extra
            for original, extra in zip(self._presence_deltas, null_deltas)
        )
        self.last_diagnostics.update(
            {
                "mass_point_process_loss": 0.0
                if self._mass_point_process_loss is None
                else float(self._mass_point_process_loss.detach()),
                "mass_count_loss": 0.0
                if self._mass_count_loss is None
                else float(self._mass_count_loss.detach()),
                "null_event_loss": 0.0
                if self._null_event_loss is None
                else float(self._null_event_loss.detach()),
                "mass_expected_count_mean": 0.0
                if not expected_counts
                else float(torch.stack(expected_counts).mean().detach()),
                "mass_expected_count_max": 0.0
                if not expected_counts
                else float(torch.stack(expected_counts).max().detach()),
                "mass_route_modulation": float(
                    self.mass_route_modulation().detach()
                ),
                "exact_available_mass_consumption": float(
                    self.exact_available_mass_consumption
                ),
                "mass_base_expected_count": 0.0
                if self._mass_base_intensity is None
                else float(self._mass_base_intensity.sum().detach()),
                "consumed_event_mass": 0.0
                if self._coverage is None
                else float(self._coverage.float().sum().detach()),
            }
        )

    def finish_forward(self) -> None:
        super().finish_forward()
        self._mass_point_process_loss = None
        self._mass_count_loss = None
        self._null_event_loss = None
        self._mass_expected_counts = ()
        self._mass_residual_rates = ()
        self._mass_working_rates = []
        self._mass_field_cursor = 0


def mass_conserving_visual_event_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    controller: MassConservingVisualEventReentryPBD,
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
    mass_point_process_weight: float = 0.15,
    mass_count_weight: float = 0.15,
    null_event_weight: float = 0.20,
) -> torch.Tensor:
    loss = accepted_mark_distilled_visual_event_ce(
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
        geometry_cycle_weight=geometry_cycle_weight,
        counterfactual_event_weight=counterfactual_event_weight,
    )
    for value, scale in (
        (controller.mass_point_process_loss, mass_point_process_weight),
        (controller.mass_count_loss, mass_count_weight),
        (controller.null_event_loss, null_event_weight),
    ):
        if value is not None:
            loss = loss + float(scale) * value
    return loss


def load_mass_conserving_visual_event_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "mass_conserving_visual_event_reentry":
        raise ValueError(f"Not a mass-conserving checkpoint: {config.get('arm')!r}")
    configure_lora_only(model, int(config["lora_rank"]))
    controller = MassConservingVisualEventReentryPBD(
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
        initial_set_coordinate_strength=float(config.get("initial_set_coordinate_strength", 0.50)),
        maximum_set_coordinate_strength=float(config.get("maximum_set_coordinate_strength", 4.0)),
        maximum_presence_bias=float(config.get("maximum_presence_bias", 2.0)),
        initial_gate_scale=float(config.get("initial_gate_scale", 0.05)),
        selection_temperature=float(config.get("selection_temperature", 0.50)),
        posterior_logit_scale=float(config.get("posterior_logit_scale", 2.0)),
        posterior_floor=float(config.get("posterior_floor", 1e-4)),
        initial_presence_rescue_scale=float(config.get("initial_presence_rescue_scale", 0.50)),
        counterfactual_margin=float(config.get("counterfactual_margin", 0.25)),
        hard_negative_exclusion=float(config.get("hard_negative_exclusion", 0.05)),
        initial_log_intensity_bias=float(config.get("initial_log_intensity_bias", -6.0)),
        maximum_null_event_bias=float(config.get("maximum_null_event_bias", 3.0)),
        initial_mass_route_modulation=float(
            config.get("initial_mass_route_modulation", 0.10)
        ),
        maximum_mass_route_modulation=float(
            config.get("maximum_mass_route_modulation", 2.0)
        ),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
