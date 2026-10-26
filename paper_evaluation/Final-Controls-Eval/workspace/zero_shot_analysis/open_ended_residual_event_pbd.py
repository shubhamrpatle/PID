"""Open-ended residual visual-event fields for LocateAnything PBD.

The controller treats each queried instance as one latent visual event.  A
query-conditioned, high-resolution event field is created once at ``<ref>``.
The ordinary Qwen/PBD stream selects one residual event, emits one native
geometry block, and writes that accepted event back before decoding the next
block.  A learned NULL energy competes with the strongest residual evidence,
so cardinality is open ended and does not require count tokens, proposal
slots, decoder replicas, crops, or an object cap.

Unlike earlier normalized pointers, the field receives target-complete
independent-center supervision.  Its fine level preserves one-to-one events in
crowded scenes, while pooled auxiliary levels stabilize coarse and sparse
objects.  The external LocateAnything grammar and PBD geometry head remain
unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .accepted_mark_distilled_event_reentry_pbd import (
    AcceptedMarkDistilledVisualEventReentryPBD,
)
from .distribution_addressed_reentry_pbd import IGNORE_INDEX
from .geometry_coupled_visual_event_reentry_pbd import _event_coordinate_residual
from .mass_conserving_event_reentry_pbd import (
    complete_residual_transitions_from_training,
)
from .pointrelay_pbd import RelayContext, configure_lora_only, restore_trainable_state
from .residual_geometry_reentry_pbd import residual_geometry_reentry_ce
from .residual_set_reentry_pbd import ResidualSetTransition


def _penalty_reduced_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """CenterNet-style heatmap loss normalized by positive event centers."""

    logits = logits.float()
    target = target.float()
    probability = logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
    positive = target.eq(1.0)
    negative = target.lt(1.0)
    positive_loss = -(
        (1.0 - probability[positive]).pow(alpha)
        * probability[positive].log()
    ).sum()
    negative_weight = (1.0 - target[negative]).pow(beta)
    negative_loss = -(
        negative_weight
        * probability[negative].pow(alpha)
        * (1.0 - probability[negative]).log()
    ).sum()
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    # Normalize centers and background independently. A positive-normalized
    # background sum makes sparse examples grow with lattice resolution and
    # gives them much larger gradients than dense examples. Explicit hard-peak
    # supervision below retains pressure on difficult false events.
    terms: list[torch.Tensor] = []
    if positive_count:
        terms.append(positive_loss / float(positive_count))
    if negative_count:
        terms.append(negative_loss / float(negative_count))
    return torch.stack(terms).sum() if terms else logits.sum() * 0.0


class OpenEndedResidualEventFieldPBD(
    AcceptedMarkDistilledVisualEventReentryPBD
):
    """One learned visual-event ledger recurrently drives one PBD stream."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vision_hidden_size: int,
        relay_layer: int,
        token_ids: Mapping[str, int],
        evidence_dim: int = 128,
        subpatch_dim: int = 32,
        subpatch_grid: int = 8,
        scale_multipliers: Sequence[float] = (0.75, 1.5, 3.0),
        minimum_sigma_cells: float = 0.75,
        uncertainty_scale: float = 2.0,
        initial_residual_strength: float = 0.10,
        maximum_residual_strength: float = 2.0,
        field_temperature: float = 0.12,
        target_sigma_cells: float = 0.65,
        initial_coverage_strength: float = 8.0,
        maximum_continuation_bias: float = 4.0,
        initial_set_coordinate_strength: float = 0.50,
        maximum_set_coordinate_strength: float = 4.0,
        maximum_presence_bias: float = 4.0,
        initial_gate_scale: float = 0.10,
        selection_temperature: float = 0.50,
        posterior_logit_scale: float = 2.0,
        posterior_floor: float = 1e-4,
        initial_presence_rescue_scale: float = 0.50,
        counterfactual_margin: float = 0.25,
        hard_negative_exclusion: float = 0.05,
        initial_event_logit_bias: float = -2.0,
        evidence_pool_temperature: float = 0.15,
        event_suppression: float = 30.0,
        event_target_sigma_cells: float = 1.0,
        maximum_event_radius: int = 2,
        event_suppression_radius: int = 0,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
        initial_event_gate_scale: float = 0.10,
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
        if evidence_pool_temperature <= 0 or event_suppression <= 0:
            raise ValueError("Event evidence and suppression constants must be positive")
        if (
            event_target_sigma_cells <= 0
            or maximum_event_radius < 0
            or event_suppression_radius < 0
        ):
            raise ValueError("Event target geometry is invalid")
        if focal_alpha < 0 or focal_beta < 0:
            raise ValueError("Focal exponents must be non-negative")
        if not 0 < initial_event_gate_scale < 1:
            raise ValueError("initial_event_gate_scale must lie in (0, 1)")

        self.event_logit_bias = nn.Parameter(
            torch.tensor(float(initial_event_logit_bias))
        )
        self.null_event_head = nn.Linear(hidden_size, 1)
        self.null_event_bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.null_event_head.weight)
        nn.init.zeros_(self.null_event_head.bias)
        self.event_gate_scale_logit = nn.Parameter(
            torch.tensor(
                math.log(
                    initial_event_gate_scale / (1.0 - initial_event_gate_scale)
                )
            )
        )
        self.evidence_pool_temperature = float(evidence_pool_temperature)
        self.event_suppression = float(event_suppression)
        self.event_target_sigma_cells = float(event_target_sigma_cells)
        self.maximum_event_radius = int(maximum_event_radius)
        self.event_suppression_radius = int(event_suppression_radius)
        self.focal_alpha = float(focal_alpha)
        self.focal_beta = float(focal_beta)

        self._event_field_cursor = 0
        self._event_base_logits: torch.Tensor | None = None
        self._event_base_rows: list[torch.Tensor] = []
        self._event_residual_rows: list[torch.Tensor] = []
        self._event_heatmap_loss: torch.Tensor | None = None
        self._event_hard_negative_loss: torch.Tensor | None = None
        self._event_gate_loss: torch.Tensor | None = None
        self._event_peak_recalls: tuple[torch.Tensor, ...] = ()
        self._event_raw_gates: tuple[torch.Tensor, ...] = ()
        self._observed_event_block: tuple[int, ...] | None = None
        self._event_null_score: torch.Tensor | None = None
        self._event_negative_features: tuple[torch.Tensor | None, ...] = ()
        self._event_negative_moments: tuple[torch.Tensor | None, ...] = ()

    @property
    def event_heatmap_loss(self) -> torch.Tensor | None:
        return self._event_heatmap_loss

    @property
    def event_hard_negative_loss(self) -> torch.Tensor | None:
        return self._event_hard_negative_loss

    @property
    def event_gate_loss(self) -> torch.Tensor | None:
        return self._event_gate_loss

    def event_gate_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.event_gate_scale_logit)

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        super()._reset_sample_state(preserve_pixel_field=preserve_pixel_field)
        self._event_field_cursor = 0
        self._event_base_logits = None
        self._event_base_rows = []
        self._event_residual_rows = []
        self._event_heatmap_loss = None
        self._event_hard_negative_loss = None
        self._event_gate_loss = None
        self._event_peak_recalls = ()
        self._event_raw_gates = ()
        self._observed_event_block = None
        self._event_null_score = None
        self._event_negative_features = ()
        self._event_negative_moments = ()

    def set_training_context(
        self,
        context: RelayContext,
        image_grid_hws: Any,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> None:
        if input_ids is None or labels is None:
            raise ValueError("Residual-event training requires input_ids and labels")
        super().set_training_context(
            context, image_grid_hws, input_ids=input_ids, labels=labels
        )
        transitions, _group_ids = complete_residual_transitions_from_training(
            input_ids, labels, token_ids=self.token_ids
        )
        self._set_transitions = transitions

    def _merge_coverage(
        self,
        current: torch.Tensor | None,
        support: torch.Tensor,
    ) -> torch.Tensor:
        support = support.detach().float()
        support = support / support.max().clamp_min(1e-12)
        if current is None:
            return support
        return torch.maximum(current.to(support).float(), support)

    def _coverage_write_support(
        self,
        transition: ResidualSetTransition,
        selection_weights: torch.Tensor,
    ) -> torch.Tensor | None:
        if self._training_context:
            if transition.target_box is None:
                return None
            support = self._center_support(
                transition.target_box, device=selection_weights.device
            ).float()
            return support / support.max().clamp_min(1e-12)
        self._pending_inference_event = selection_weights.detach().float()
        return None

    def _residualize_field_logits(
        self,
        field_logits: torch.Tensor,
        coverage: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        del strength
        transition = (
            self._set_transitions[self._event_field_cursor]
            if self._event_field_cursor < len(self._set_transitions)
            else None
        )
        self._event_field_cursor += 1
        reset = (
            self._event_base_logits is None
            or (transition is not None and transition.reset_coverage)
            or int(self._event_base_logits.numel()) != int(field_logits.numel())
        )
        if reset:
            self._event_base_logits = field_logits.float() + self.event_logit_bias.float()
        base = self._event_base_logits
        residual = base - self.event_suppression * coverage.float().clamp(0.0, 1.0)
        self._event_base_rows.append(base)
        self._event_residual_rows.append(residual)
        return residual.to(field_logits.dtype)

    def _event_target(
        self,
        boxes: Sequence[Sequence[int]],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._grid_hw is None:
            raise RuntimeError("Event field grid is unavailable")
        height = int(self._grid_hw[0]) * self.subpatch_grid
        width = int(self._grid_hw[1]) * self.subpatch_grid
        cells = height * width
        target = torch.zeros(cells, device=device, dtype=torch.float32)
        if not boxes:
            return target, target.new_zeros(0, dtype=torch.long)

        values = torch.as_tensor(boxes, device=device, dtype=torch.float32)
        center_x = ((values[:, 0] + values[:, 2]) * 0.5 / 1000.0 * width)
        center_y = ((values[:, 1] + values[:, 3]) * 0.5 / 1000.0 * height)
        x = center_x.floor().long().clamp_(0, width - 1)
        y = center_y.floor().long().clamp_(0, height - 1)
        center_indices = y * width + x

        radius = self.maximum_event_radius
        sigma_sq = self.event_target_sigma_cells * self.event_target_sigma_cells
        for offset_y in range(-radius, radius + 1):
            for offset_x in range(-radius, radius + 1):
                shifted_x = (x + offset_x).clamp(0, width - 1)
                shifted_y = (y + offset_y).clamp(0, height - 1)
                indices = shifted_y * width + shifted_x
                value = math.exp(
                    -0.5 * (offset_x * offset_x + offset_y * offset_y) / sigma_sq
                )
                source = target.new_full((int(indices.numel()),), value)
                target.scatter_reduce_(
                    0, indices, source, reduce="amax", include_self=True
                )
        target[center_indices] = 1.0
        return target, center_indices.unique()

    def _hierarchical_event_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self._grid_hw is None:
            raise RuntimeError("Event field grid is unavailable")
        height = int(self._grid_hw[0]) * self.subpatch_grid
        width = int(self._grid_hw[1]) * self.subpatch_grid
        current_logits = logits.reshape(1, 1, height, width)
        current_target = target.reshape(1, 1, height, width)
        losses: list[torch.Tensor] = []
        for level in range(3):
            losses.append(
                _penalty_reduced_focal_loss(
                    current_logits.reshape(-1),
                    current_target.reshape(-1),
                    alpha=self.focal_alpha,
                    beta=self.focal_beta,
                )
            )
            if level < 2 and min(current_logits.shape[-2:]) >= 2:
                current_logits = F.max_pool2d(current_logits, kernel_size=2, stride=2)
                current_target = F.max_pool2d(current_target, kernel_size=2, stride=2)
        return torch.stack(losses).mean()

    def _residual_evidence(self, logits: torch.Tensor) -> torch.Tensor:
        """Pool the strongest learned event peaks without a count prediction.

        A small top-k is only a robust evidence statistic; it neither allocates
        output slots nor caps the number of events that can be consumed. Unlike
        a mean over every cell, its scale is stable when the spatial lattice is
        refined from 64x64 to 256x256.
        """

        temperature = self.evidence_pool_temperature
        count = min(8, int(logits.numel()))
        if count <= 0:
            return self.event_logit_bias.new_tensor(-30.0)
        strongest = logits.float().topk(count).values
        return temperature * (
            torch.logsumexp(strongest / temperature, dim=0) - math.log(count)
        )

    def _event_cell_index(
        self,
        box: Sequence[int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if self._grid_hw is None:
            raise RuntimeError("Event field grid is unavailable")
        height = int(self._grid_hw[0]) * self.subpatch_grid
        width = int(self._grid_hw[1]) * self.subpatch_grid
        values = torch.as_tensor(box, device=device, dtype=torch.float32)
        center_x = ((values[0] + values[2]) * 0.5 / 1000.0 * width).floor()
        center_y = ((values[1] + values[3]) * 0.5 / 1000.0 * height).floor()
        x = center_x.long().clamp_(0, width - 1)
        y = center_y.long().clamp_(0, height - 1)
        return y * width + x

    def _event_support_for_index(
        self,
        index: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if self._grid_hw is None:
            raise RuntimeError("Event field grid is unavailable")
        height = int(self._grid_hw[0]) * self.subpatch_grid
        width = int(self._grid_hw[1]) * self.subpatch_grid
        support = torch.zeros(height * width, device=device, dtype=torch.float32)
        center_y = torch.div(index.long(), width, rounding_mode="floor")
        center_x = index.long().remainder(width)
        # Consume only the selected learned event identity. Target heatmaps use
        # a wider Gaussian neighborhood for stable learning, but sharing that
        # radius here would impose a hidden minimum-distance/object-count cap in
        # crowded scenes.
        radius = self.event_suppression_radius
        sigma_sq = self.event_target_sigma_cells * self.event_target_sigma_cells
        indices: list[torch.Tensor] = []
        values: list[float] = []
        for offset_y in range(-radius, radius + 1):
            for offset_x in range(-radius, radius + 1):
                x = center_x + offset_x
                y = center_y + offset_y
                if not (0 <= int(x) < width and 0 <= int(y) < height):
                    continue
                indices.append(y * width + x)
                values.append(
                    math.exp(
                        -0.5
                        * (offset_x * offset_x + offset_y * offset_y)
                        / sigma_sq
                    )
                )
        if indices:
            flat = torch.stack(indices)
            support[flat] = support.new_tensor(values)
        return support

    def _event_query_base(
        self,
        hidden_states: torch.Tensor,
        transition: ResidualSetTransition,
        keys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = F.normalize(
            self.set_query_proj(
                hidden_states[0, transition.row[0]].to(
                    self.set_query_proj.weight.dtype
                )
            ).float(),
            dim=-1,
        )
        base = torch.matmul(keys, query) / self.field_temperature
        base = base.float() + self.event_logit_bias.float()
        query_hidden = hidden_states[0, transition.row[0]].to(
            self.null_event_head.weight.dtype
        )
        null_score = (
            self.null_event_head(query_hidden).float().reshape(())
            + self.null_event_bias.float()
        )
        return base, null_score

    @staticmethod
    def _transition_groups(
        transitions: Sequence[ResidualSetTransition],
    ) -> tuple[tuple[ResidualSetTransition, ...], ...]:
        groups: list[list[ResidualSetTransition]] = []
        current: list[ResidualSetTransition] = []
        for transition in transitions:
            if transition.reset_coverage and current:
                groups.append(current)
                current = []
            current.append(transition)
        if current:
            groups.append(current)
        return tuple(tuple(group) for group in groups)

    def _set_coordinate_signals(
        self,
        hidden_states: torch.Tensor,
        transitions: Sequence[ResidualSetTransition],
        selected_rows: Sequence[torch.Tensor],
        moment_rows: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            _event_coordinate_residual(
                self, hidden_states, transition, selected, moments
            )
            for transition, selected, moments in zip(
                transitions, selected_rows, moment_rows
            )
        )

    def _prepare_training_event_signals(
        self,
        hidden_states: torch.Tensor,
        transitions: tuple[ResidualSetTransition, ...],
    ) -> None:
        keys, values = self._field(hidden_states)
        positions = self._subpatch_positions
        if positions is None:
            raise RuntimeError("Event positions are unavailable")
        positions = positions.to(hidden_states.device, dtype=torch.float32)

        logits_rows: list[torch.Tensor] = []
        selected_rows: list[torch.Tensor] = []
        moment_rows: list[torch.Tensor] = []
        negative_rows: list[torch.Tensor | None] = []
        negative_moments: list[torch.Tensor | None] = []
        continuation: list[torch.Tensor] = []
        presence: list[torch.Tensor | None] = []
        raw_gates: list[torch.Tensor] = []
        positive_gate_losses: list[torch.Tensor] = []
        negative_gate_losses: list[torch.Tensor] = []
        heatmap_losses: list[torch.Tensor] = []
        hard_negative_losses: list[torch.Tensor] = []
        peak_recalls: list[torch.Tensor] = []
        base_rows: list[torch.Tensor] = []
        residual_rows: list[torch.Tensor] = []
        gate_scale = self.event_gate_scale()

        for group in self._transition_groups(transitions):
            first = group[0]
            base, null_score = self._event_query_base(hidden_states, first, keys)
            target, center_indices = self._event_target(
                first.remaining_boxes, device=base.device
            )
            heatmap_losses.append(self._hierarchical_event_loss(base, target))
            base_rows.append(base)
            residual_rows.append(base)

            negative_mask = target.lt(0.05)
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
                negative_evidence = self._residual_evidence(negative_values)
            else:
                negative_index = base.argmin()
                negative_evidence = base[negative_index]

            if int(center_indices.numel()):
                peak_recalls.append(
                    base[center_indices].sigmoid().gt(0.5).float().mean()
                )

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
            numeric_scores = base[numeric_indices] if numeric else base.new_zeros(0)
            suffix_max = (
                torch.flip(
                    torch.cummax(torch.flip(numeric_scores, dims=(0,)), dim=0).values,
                    dims=(0,),
                )
                if numeric
                else numeric_scores
            )
            numeric_cursor = 0
            negative_feature = values[negative_index]
            negative_spatial = torch.cat(
                (positions[negative_index], positions.new_zeros(2))
            )

            for transition in group:
                if transition.target_box is not None:
                    event_index = numeric_indices[numeric_cursor]
                    raw_gate = suffix_max[numeric_cursor] - null_score
                    numeric_cursor += 1
                    label = 1.0
                    selected = values[event_index]
                    moments = torch.cat(
                        (positions[event_index], positions.new_zeros(2))
                    )
                    negative_rows.append(negative_feature)
                    negative_moments.append(negative_spatial)
                else:
                    event_index = negative_index
                    raw_gate = negative_evidence - null_score
                    label = 0.0
                    selected = negative_feature
                    moments = negative_spatial
                    negative_rows.append(None)
                    negative_moments.append(None)

                decision = F.binary_cross_entropy_with_logits(
                    raw_gate, raw_gate.new_tensor(label)
                )
                (positive_gate_losses if label else negative_gate_losses).append(
                    decision
                )
                maximum = (
                    self.maximum_presence_bias
                    if transition.predecessor == "ref"
                    else self.maximum_continuation_bias
                )
                delta = maximum * torch.tanh(gate_scale * raw_gate)
                logits_rows.append(base)
                selected_rows.append(selected)
                moment_rows.append(moments)
                continuation.append(delta)
                presence.append(delta if transition.predecessor == "ref" else None)
                raw_gates.append(raw_gate)

        self._set_logits = tuple(logits_rows)
        self._set_selected_features = tuple(selected_rows)
        self._set_spatial_moments = tuple(moment_rows)
        self._set_coordinate_residuals = self._set_coordinate_signals(
            hidden_states, transitions, selected_rows, moment_rows
        )
        self._event_negative_features = tuple(negative_rows)
        self._event_negative_moments = tuple(negative_moments)
        self._continuation_deltas = tuple(continuation)
        self._presence_deltas = tuple(presence)
        self._event_base_rows = base_rows
        self._event_residual_rows = residual_rows
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
        balanced = [
            torch.stack(values_).mean()
            for values_ in (positive_gate_losses, negative_gate_losses)
            if values_
        ]
        self._event_gate_loss = torch.stack(balanced).mean() if balanced else None
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
            raise RuntimeError("Event positions are unavailable")
        positions = positions.to(hidden_states.device, dtype=torch.float32)
        if transition.reset_coverage or self._event_base_logits is None:
            self._event_base_logits, self._event_null_score = self._event_query_base(
                hidden_states, transition, keys
            )
        if self._event_null_score is None:
            raise RuntimeError("Event NULL score was not initialized")
        coverage = (
            torch.zeros_like(self._event_base_logits)
            if self._coverage is None
            else self._coverage.to(self._event_base_logits).float()
        )
        residual = self._event_base_logits - self.event_suppression * coverage
        event_index = residual.argmax()
        support = self._event_support_for_index(
            event_index, device=residual.device
        )
        self._pending_inference_event = support
        selected = values[event_index]
        moments = torch.cat((positions[event_index], positions.new_zeros(2)))
        raw_gate = self._residual_evidence(residual) - self._event_null_score
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
        self._set_route_loss = None
        self._continuation_loss = None
        self._presence_loss = None

    def _update_event_diagnostics(self) -> None:
        raw_gates = self._event_raw_gates
        peak_recalls = self._event_peak_recalls
        self.last_diagnostics.update(
            {
                "event_heatmap_loss": 0.0
                if self._event_heatmap_loss is None
                else float(self._event_heatmap_loss.detach()),
                "event_hard_negative_loss": 0.0
                if self._event_hard_negative_loss is None
                else float(self._event_hard_negative_loss.detach()),
                "event_gate_loss": 0.0
                if self._event_gate_loss is None
                else float(self._event_gate_loss.detach()),
                "event_peak_recall": 0.0
                if not peak_recalls
                else float(torch.stack(peak_recalls).mean().detach()),
                "event_gate_mean": 0.0
                if not raw_gates
                else float(torch.stack(raw_gates).mean().detach()),
                "event_gate_abs_mean": 0.0
                if not raw_gates
                else float(
                    torch.stack([value.abs() for value in raw_gates])
                    .mean()
                    .detach()
                ),
                "event_gate_scale": float(self.event_gate_scale().detach()),
                "event_field_cells": 0.0
                if not self._event_residual_rows
                else float(self._event_residual_rows[0].numel()),
                "event_positive_decisions": float(
                    sum(bool(transition.remaining_boxes) for transition in self._set_transitions)
                ),
                "event_negative_decisions": float(
                    sum(not bool(transition.remaining_boxes) for transition in self._set_transitions)
                ),
                "event_signal_complexity": float(
                    len(self._set_transitions)
                    + sum(row.numel() for row in self._event_base_rows)
                ),
            }
        )

    def prepare_set_signals(self, hidden_states: torch.Tensor) -> None:
        transitions = tuple(
            transition
            for transition in self._set_transitions
            if max(transition.row) < int(hidden_states.shape[1])
        )
        self._set_transitions = transitions
        self._event_field_cursor = 0
        self._event_base_rows = []
        self._event_residual_rows = []
        self._event_heatmap_loss = None
        self._event_hard_negative_loss = None
        self._event_gate_loss = None
        self._event_peak_recalls = ()
        self._event_raw_gates = ()
        if not transitions or self._subpatch_positions is None:
            self._set_logits = ()
            self._set_selected_features = ()
            self._set_spatial_moments = ()
            self._set_coordinate_residuals = ()
            self._continuation_deltas = ()
            self._presence_deltas = ()
            return
        if self._training_context:
            self._prepare_training_event_signals(hidden_states, transitions)
        else:
            self._prepare_inference_event_signals(hidden_states, transitions[-1])
        self._update_event_diagnostics()

    def _commit_pending_event(
        self, observed_block: tuple[int, ...] | None
    ) -> bool:
        support = self._pending_inference_event
        if support is None:
            return False
        self._coverage = self._merge_coverage(self._coverage, support)
        self._pending_inference_event = None
        self._observed_inference_events += 1
        self.geometry_writeback_count += 1
        self._observed_event_block = observed_block
        return True

    def observe_generated_block(self, out_type: str, tokens: Any) -> None:
        if self._training_context:
            return
        if out_type == "ref_object":
            self._coverage = None
            self._event_base_logits = None
            self._event_null_score = None
            self._pending_inference_event = None
            self._observed_event_block = None
            self._observed_inference_refs += 1
        elif out_type in {"coord_box", "point_box"}:
            values = (
                tuple(int(value) for value in tokens.detach().reshape(-1).tolist())
                if isinstance(tokens, torch.Tensor)
                else tuple(
                    int(value.item() if hasattr(value, "item") else value)
                    for value in tokens
                )
            )
            self._commit_pending_event(values)
        elif out_type in {"empty_box", "error_box", "im_end"}:
            self._pending_inference_event = None
            self._observed_event_block = None
        else:
            return
        self.last_diagnostics.update(
            {
                "event_observed_events": float(self._observed_inference_events),
                "event_observed_refs": float(self._observed_inference_refs),
                "event_consumed_cells": 0.0
                if self._coverage is None
                else float(self._coverage.gt(0.5).sum().detach().cpu()),
            }
        )

    def _write_previous_geometry(self, input_ids: torch.Tensor, row_start: int) -> None:
        # AR fallback commits the staged learned event. Geometry is the mark,
        # not a deterministic address policy.
        if self._pending_inference_event is None:
            return
        block: tuple[int, ...] = ()
        if row_start >= 6:
            block = tuple(
                int(value)
                for value in input_ids[0, row_start - 6 : row_start].tolist()
            )
        if block and block == self._observed_event_block:
            self._observed_event_block = None
            return
        self._commit_pending_event(None)

    def finish_forward(self) -> None:
        super().finish_forward()
        self._event_field_cursor = 0
        self._event_base_rows = []
        self._event_residual_rows = []
        self._event_heatmap_loss = None
        self._event_hard_negative_loss = None
        self._event_gate_loss = None
        self._event_peak_recalls = ()
        self._event_raw_gates = ()
        self._event_negative_features = ()
        self._event_negative_moments = ()


def open_ended_residual_event_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    controller: OpenEndedResidualEventFieldPBD,
    *,
    route_loss_weight: float = 0.20,
    coordinate_aux_weight: float = 0.30,
    posterior_distill_weight: float = 0.10,
    prior_geometry_weight: float = 0.20,
    prior_decision_weight: float = 0.15,
    geometry_cycle_weight: float = 0.20,
    counterfactual_event_weight: float = 0.20,
    event_heatmap_weight: float = 0.75,
    event_hard_negative_weight: float = 0.20,
    event_gate_weight: float = 0.35,
) -> torch.Tensor:
    # Use the native token objective plus PIVR's ordinary distribution-addressed
    # correction. ORE's own prepare_set_signals implementation is linear in the
    # number of targets and avoids inherited dense posterior work that would be
    # quadratic in a 1,000-instance answer.
    del route_loss_weight, posterior_distill_weight, prior_geometry_weight
    del prior_decision_weight
    loss = residual_geometry_reentry_ce(
        hidden_states,
        labels,
        weight,
        controller,
        route_loss_weight=0.0,
        continuation_loss_weight=0.0,
        presence_loss_weight=0.0,
    )

    coord_start = int(controller.token_ids["coord_start_token_id"])
    coord_end = int(controller.token_ids["coord_end_token_id"])
    coord_weight = weight[coord_start : coord_end + 1]
    coordinate_values = torch.linspace(
        0.0,
        1.0,
        coord_end - coord_start + 1,
        device=hidden_states.device,
        dtype=torch.float32,
    )
    strength = controller.set_coordinate_strength().float().unsqueeze(1)
    coordinate_losses: list[torch.Tensor] = []
    cycle_losses: list[torch.Tensor] = []
    counterfactual_losses: list[torch.Tensor] = []

    for transition, residual, moments, negative, negative_moments in zip(
        controller._set_transitions,
        controller.set_coordinate_residuals,
        controller._set_spatial_moments,
        controller._event_negative_features,
        controller._event_negative_moments,
    ):
        if transition.target_box is None:
            continue
        lane_hidden = torch.stack(
            [hidden_states[0, position] for position in transition.row[1:5]]
        )
        targets: list[int] = []
        for position in transition.row[1:5]:
            target_position = position + 1
            if target_position >= int(labels.shape[1]):
                targets = []
                break
            target = int(labels[0, target_position])
            if target == IGNORE_INDEX or not coord_start <= target <= coord_end:
                targets = []
                break
            targets.append(target - coord_start)
        if len(targets) != 4:
            continue

        base = F.linear(lane_hidden, coord_weight).float()
        positive_logits = base + strength * residual.float()
        target_tensor = positive_logits.new_tensor(targets, dtype=torch.long)
        positive_nll = F.cross_entropy(positive_logits, target_tensor)
        coordinate_losses.append(positive_nll)

        probabilities = positive_logits.softmax(dim=-1)
        expected = torch.matmul(probabilities, coordinate_values)
        predicted_center = torch.stack(
            ((expected[0] + expected[2]) * 0.5, (expected[1] + expected[3]) * 0.5)
        )
        cycle_losses.append(
            F.smooth_l1_loss(predicted_center, moments[:2].float(), beta=0.02)
        )

        if negative is not None and negative_moments is not None:
            negative_residual = _event_coordinate_residual(
                controller,
                hidden_states,
                transition,
                negative,
                negative_moments,
            )
            negative_nll = F.cross_entropy(
                base + strength * negative_residual.float(), target_tensor
            )
            counterfactual_losses.append(
                F.relu(
                    positive_nll.new_tensor(controller.counterfactual_margin)
                    + positive_nll
                    - negative_nll
                )
            )

    if coordinate_losses:
        loss = loss + float(coordinate_aux_weight) * torch.stack(
            coordinate_losses
        ).mean()
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

    for value, scale in (
        (controller.event_heatmap_loss, event_heatmap_weight),
        (controller.event_hard_negative_loss, event_hard_negative_weight),
        (controller.event_gate_loss, event_gate_weight),
    ):
        if value is not None:
            loss = loss + float(scale) * value
    controller.last_diagnostics.update(
        {
            "geometry_cycle_loss": 0.0
            if controller.geometry_cycle_loss is None
            else float(controller.geometry_cycle_loss.detach()),
            "counterfactual_event_loss": 0.0
            if controller.counterfactual_event_loss is None
            else float(controller.counterfactual_event_loss.detach()),
            "counterfactual_event_pairs": float(
                controller._counterfactual_pairs
            ),
        }
    )
    return loss


def load_open_ended_residual_event_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "open_ended_residual_event_reentry":
        raise ValueError(f"Not an open-ended residual-event checkpoint: {config.get('arm')!r}")
    configure_lora_only(model, int(config["lora_rank"]))
    controller = OpenEndedResidualEventFieldPBD(
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
        maximum_event_radius=int(config.get("maximum_event_radius", 2)),
        event_suppression_radius=int(config.get("event_suppression_radius", 0)),
        focal_alpha=float(config.get("focal_alpha", 2.0)),
        focal_beta=float(config.get("focal_beta", 4.0)),
        initial_event_gate_scale=float(config.get("initial_event_gate_scale", 0.10)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
