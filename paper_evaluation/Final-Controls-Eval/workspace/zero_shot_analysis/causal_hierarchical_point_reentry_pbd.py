"""Causal hierarchical point re-entry for LocateAnything PBD.

Each native geometry transition uses the *current* Qwen hidden state to choose
one MoonViT patch and then one fine subpatch inside that patch.  The selected
local feature conditions the existing atomic PBD geometry block.  A learned
STOP state is normalized jointly with the patch choices, so cardinality is
open ended and remains controlled by the native decoder history rather than a
fixed set of object slots.

The hierarchy is architectural, not an output cap: every fine cell remains
reachable, accepted geometry is written back only after a valid native block,
and a patch may be revisited arbitrarily many times.  The image field is cached
once; one Qwen decoder/KV stream emits the ordinary LocateAnything grammar.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .normalized_residual_hazard_pbd import NormalizedResidualHazardPBD
from .pointrelay_pbd import configure_lora_only, restore_trainable_state
from .residual_set_reentry_pbd import ResidualSetTransition


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class CausalHierarchicalPointReentryPBD(NormalizedResidualHazardPBD):
    """History-conditioned patch/subpatch event process with native PBD marks."""

    def __init__(
        self,
        *,
        initial_revisit_penalty: float = 0.10,
        maximum_revisit_penalty: float = 2.0,
        permutation_invariant_residual_set: bool = False,
        hazard_fused_token_mass: bool = False,
        permutation_marginalized_pbd: bool = False,
        joint_residual_set_selection: bool = False,
        initial_hazard_logit_scale: float = 8.0,
        maximum_hazard_logit_scale: float = 100.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not 0 < initial_revisit_penalty < maximum_revisit_penalty:
            raise ValueError("Causal revisit-penalty bounds are invalid")
        if not 1.0 <= initial_hazard_logit_scale < maximum_hazard_logit_scale:
            raise ValueError("Causal hazard-logit-scale bounds are invalid")
        self.maximum_revisit_penalty = float(maximum_revisit_penalty)
        ratio = initial_revisit_penalty / maximum_revisit_penalty
        self.revisit_penalty_logit = nn.Parameter(torch.tensor(_logit(ratio)))
        self.permutation_invariant_residual_set = bool(
            permutation_invariant_residual_set
        )
        self.hazard_fused_token_mass = bool(hazard_fused_token_mass)
        self.permutation_marginalized_pbd = bool(permutation_marginalized_pbd)
        self.joint_residual_set_selection = bool(joint_residual_set_selection)
        if self.hazard_fused_token_mass and not self.permutation_invariant_residual_set:
            raise ValueError(
                "Hazard-fused token mass requires permutation-invariant residual-set training"
            )
        if self.permutation_marginalized_pbd and not self.permutation_invariant_residual_set:
            raise ValueError(
                "Permutation-marginalized PBD requires residual-set training"
            )
        if self.joint_residual_set_selection and not self.permutation_invariant_residual_set:
            raise ValueError(
                "Joint residual-set selection requires residual-set training"
            )
        self.maximum_hazard_logit_scale = float(maximum_hazard_logit_scale)
        if self.permutation_invariant_residual_set:
            scale_ratio = (float(initial_hazard_logit_scale) - 1.0) / (
                self.maximum_hazard_logit_scale - 1.0
            )
            self.hazard_logit_scale_logit = nn.Parameter(
                torch.tensor(_logit(scale_ratio))
            )
        else:
            self.register_parameter("hazard_logit_scale_logit", None)
        self._causal_patch_loss: torch.Tensor | None = None
        self._causal_local_loss: torch.Tensor | None = None
        self._causal_set_loss: torch.Tensor | None = None
        self._causal_local_set_loss: torch.Tensor | None = None
        self._causal_stop_loss: torch.Tensor | None = None
        self._causal_exact_hits: tuple[torch.Tensor, ...] = ()
        self._causal_residual_set_hits: tuple[torch.Tensor, ...] = ()

    def revisit_penalty(self) -> torch.Tensor:
        return self.maximum_revisit_penalty * torch.sigmoid(
            self.revisit_penalty_logit
        )

    def hazard_logit_scale(self) -> torch.Tensor:
        if self.hazard_logit_scale_logit is None:
            return self.event_logit_bias.new_tensor(1.0 / self.hazard_temperature)
        return 1.0 + (self.maximum_hazard_logit_scale - 1.0) * torch.sigmoid(
            self.hazard_logit_scale_logit
        )

    def _clear_causal_state(self) -> None:
        self._causal_patch_loss = None
        self._causal_local_loss = None
        self._causal_set_loss = None
        self._causal_local_set_loss = None
        self._causal_stop_loss = None
        self._causal_exact_hits = ()
        self._causal_residual_set_hits = ()

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        super()._reset_sample_state(preserve_pixel_field=preserve_pixel_field)
        self._clear_causal_state()

    def _hierarchical_field(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys, values = self._field(hidden_states)
        if self._grid_hw is None:
            raise RuntimeError("Causal event grid is unavailable")
        grid_h, grid_w = self._grid_hw
        local_side = self.subpatch_grid
        expected = grid_h * grid_w * local_side * local_side
        if int(keys.shape[0]) != expected:
            raise RuntimeError(
                f"Hierarchical field mismatch: {int(keys.shape[0])} != {expected}"
            )
        dim = int(keys.shape[-1])
        key_grid = keys.reshape(
            grid_h, local_side, grid_w, local_side, dim
        ).permute(0, 2, 1, 3, 4)
        value_grid = values.reshape(
            grid_h, local_side, grid_w, local_side, dim
        ).permute(0, 2, 1, 3, 4)
        local_keys = key_grid.reshape(
            grid_h * grid_w, local_side * local_side, dim
        )
        local_values = value_grid.reshape(
            grid_h * grid_w, local_side * local_side, dim
        )
        patch_keys = F.normalize(local_keys.mean(dim=1), dim=-1)
        return patch_keys, local_keys, local_values

    def _hierarchical_indices(
        self, event_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._grid_hw is None:
            raise RuntimeError("Causal event grid is unavailable")
        grid_h, grid_w = self._grid_hw
        local_side = self.subpatch_grid
        raw_w = grid_w * local_side
        y = torch.div(event_index.long(), raw_w, rounding_mode="floor")
        x = event_index.long().remainder(raw_w)
        patch = torch.div(y, local_side, rounding_mode="floor") * grid_w
        patch = patch + torch.div(x, local_side, rounding_mode="floor")
        local = y.remainder(local_side) * local_side + x.remainder(local_side)
        return patch.clamp(0, grid_h * grid_w - 1), local.clamp(
            0, local_side * local_side - 1
        )

    def _flat_event_index(
        self, patch_index: torch.Tensor, local_index: torch.Tensor
    ) -> torch.Tensor:
        if self._grid_hw is None:
            raise RuntimeError("Causal event grid is unavailable")
        _grid_h, grid_w = self._grid_hw
        local_side = self.subpatch_grid
        patch_y = torch.div(patch_index.long(), grid_w, rounding_mode="floor")
        patch_x = patch_index.long().remainder(grid_w)
        local_y = torch.div(local_index.long(), local_side, rounding_mode="floor")
        local_x = local_index.long().remainder(local_side)
        raw_y = patch_y * local_side + local_y
        raw_x = patch_x * local_side + local_x
        return raw_y * (grid_w * local_side) + raw_x

    def _coverage_matrix(
        self, coverage: torch.Tensor | None, *, device: torch.device
    ) -> torch.Tensor:
        if self._grid_hw is None:
            raise RuntimeError("Causal event grid is unavailable")
        grid_h, grid_w = self._grid_hw
        local_cells = self.subpatch_grid * self.subpatch_grid
        cells = grid_h * grid_w * local_cells
        if coverage is None or int(coverage.numel()) != cells:
            return torch.zeros(
                grid_h * grid_w, local_cells, device=device, dtype=torch.float32
            )
        raw = coverage.to(device=device, dtype=torch.float32).reshape(
            grid_h,
            self.subpatch_grid,
            grid_w,
            self.subpatch_grid,
        )
        return raw.permute(0, 2, 1, 3).reshape(grid_h * grid_w, local_cells)

    def _query_scores(
        self,
        hidden_states: torch.Tensor,
        transition: ResidualSetTransition,
        patch_keys: torch.Tensor,
        local_keys: torch.Tensor,
        coverage: torch.Tensor,
        *,
        patch_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query_hidden = hidden_states[0, transition.row[0]]
        query = F.normalize(
            self.set_query_proj(query_hidden.to(self.set_query_proj.weight.dtype)).float(),
            dim=-1,
        )
        logit_scale = self.hazard_logit_scale().float()
        patch_logits = torch.matmul(patch_keys, query) * logit_scale
        patch_logits = patch_logits.float() + self.event_logit_bias.float()
        selected_patch = patch_logits.argmax() if patch_index is None else patch_index
        local_logits = torch.matmul(local_keys[selected_patch], query) * logit_scale
        local_logits = local_logits.float()
        local_logits = local_logits - self.revisit_penalty().float() * torch.log1p(
            coverage[selected_patch]
        )
        null_score = (
            self.null_event_head(query_hidden.to(self.null_event_head.weight.dtype))
            .float()
            .reshape(())
            + self.null_event_bias.float()
        )
        return patch_logits, local_logits, null_score, query

    def _joint_spatial_logits(
        self,
        query: torch.Tensor,
        patch_logits: torch.Tensor,
        local_keys: torch.Tensor,
        coverage: torch.Tensor,
    ) -> torch.Tensor:
        """Factor one open-ended event into patch and conditional local address."""

        local_logits = torch.einsum("pld,d->pl", local_keys, query)
        local_logits = local_logits * self.hazard_logit_scale().float()
        local_logits = local_logits.float() - self.revisit_penalty().float() * torch.log1p(
            coverage
        )
        return patch_logits[:, None] + F.log_softmax(local_logits, dim=-1)

    def _select_joint_spatial_event(
        self,
        joint_logits: torch.Tensor,
        local_values: torch.Tensor,
        positions: torch.Tensor,
        *,
        straight_through: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select one learned event without slots or a preset cardinality.

        Training uses a straight-through categorical variable: the forward pass
        reads one concrete local feature, while gradients flow through the full
        spatial distribution. Inference uses the same categorical argmax under
        ordinary greedy decoding.
        """

        patch_count, local_count = joint_logits.shape
        flat_logits = joint_logits.reshape(-1)
        hard_index = flat_logits.argmax()
        if straight_through:
            probabilities = F.softmax(
                flat_logits / float(self.selection_temperature), dim=0
            )
            hard = F.one_hot(hard_index, num_classes=int(flat_logits.numel())).to(
                probabilities.dtype
            )
            weights = hard + probabilities - probabilities.detach()
        else:
            weights = F.one_hot(
                hard_index, num_classes=int(flat_logits.numel())
            ).to(flat_logits.dtype)

        patch_index = torch.div(hard_index, local_count, rounding_mode="floor")
        local_index = hard_index.remainder(local_count)
        selected = torch.matmul(weights, local_values.reshape(-1, local_values.shape[-1]))

        patch_ids = torch.arange(patch_count, device=joint_logits.device)[:, None]
        local_ids = torch.arange(local_count, device=joint_logits.device)[None, :]
        raw_indices = self._flat_event_index(
            patch_ids.expand(-1, local_count).reshape(-1),
            local_ids.expand(patch_count, -1).reshape(-1),
        )
        selected_position = torch.matmul(weights.float(), positions[raw_indices].float())
        moments = torch.cat((selected_position, selected_position.new_zeros(2)))
        raw_index = self._flat_event_index(patch_index, local_index)
        return selected, moments, patch_index, local_index, raw_index

    def _prepare_training_event_signals(
        self,
        hidden_states: torch.Tensor,
        transitions: tuple[ResidualSetTransition, ...],
    ) -> None:
        patch_keys, local_keys, local_values = self._hierarchical_field(hidden_states)
        positions = self._subpatch_positions
        if positions is None:
            raise RuntimeError("Causal event positions are unavailable")
        positions = positions.to(hidden_states.device, dtype=torch.float32)

        selected_rows: list[torch.Tensor] = []
        moment_rows: list[torch.Tensor] = []
        continuation: list[torch.Tensor] = []
        presence: list[torch.Tensor | None] = []
        raw_gates: list[torch.Tensor] = []
        patch_losses: list[torch.Tensor] = []
        local_losses: list[torch.Tensor] = []
        set_losses: list[torch.Tensor] = []
        local_set_losses: list[torch.Tensor] = []
        stop_losses: list[torch.Tensor] = []
        exact_hits: list[torch.Tensor] = []
        residual_set_hits: list[torch.Tensor] = []
        event_probabilities: list[torch.Tensor] = []
        stop_probabilities: list[torch.Tensor] = []
        field_rows: list[torch.Tensor] = []
        gate_scale = self.event_gate_scale()
        temperature = self.hazard_temperature
        logit_scale = self.hazard_logit_scale().float()
        coverage = self._coverage_matrix(None, device=hidden_states.device)

        for group in self._transition_groups(transitions):
            coverage = torch.zeros_like(coverage)
            group_event_indices = tuple(
                None
                if transition.target_box is None
                else self._event_cell_index(
                    transition.target_box, device=hidden_states.device
                )
                for transition in group
            )
            positive_indices = tuple(
                index for index in group_event_indices if index is not None
            )
            if positive_indices:
                positive_tensor = torch.stack(positive_indices)
                positive_patches, positive_locals = self._hierarchical_indices(
                    positive_tensor
                )
                local_cells = int(local_keys.shape[1])
                address_counts = torch.bincount(
                    positive_patches * local_cells + positive_locals,
                    minlength=int(patch_keys.shape[0]) * local_cells,
                ).reshape(int(patch_keys.shape[0]), local_cells)
                patch_counts = address_counts.sum(dim=1)
            else:
                positive_patches = torch.empty(
                    0, device=hidden_states.device, dtype=torch.long
                )
                positive_locals = torch.empty_like(positive_patches)
                patch_counts = torch.zeros(
                    int(patch_keys.shape[0]),
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                address_counts = torch.zeros(
                    int(patch_keys.shape[0]),
                    int(local_keys.shape[1]),
                    device=hidden_states.device,
                    dtype=torch.long,
                )
            positive_offset = 0

            for transition, event_index in zip(group, group_event_indices):
                if event_index is not None:
                    target_patch = positive_patches[positive_offset]
                    target_local = positive_locals[positive_offset]
                else:
                    target_patch = None
                    target_local = None

                patch_logits, local_logits, null_score, query = self._query_scores(
                    hidden_states,
                    transition,
                    patch_keys,
                    local_keys,
                    coverage,
                    patch_index=target_patch,
                )
                patch_partition = torch.logsumexp(patch_logits, dim=0)
                if self.permutation_invariant_residual_set:
                    event_score = patch_partition - math.log(
                        max(int(patch_logits.numel()), 1)
                    )
                    raw_gate = event_score - null_score
                else:
                    event_score = patch_partition
                    raw_gate = temperature * (patch_partition - null_score)
                joint_partition = torch.logaddexp(event_score, null_score)
                event_probability = (event_score - joint_partition).exp()
                stop_probability = (null_score - joint_partition).exp()
                event_probabilities.append(
                    event_probability
                    if self.hazard_fused_token_mass
                    else event_probability.detach()
                )
                stop_probabilities.append(
                    stop_probability
                    if self.hazard_fused_token_mass
                    else stop_probability.detach()
                )

                if (
                    event_index is not None
                    and target_patch is not None
                    and target_local is not None
                ):
                    if self.joint_residual_set_selection:
                        joint_spatial_logits = self._joint_spatial_logits(
                            query, patch_logits, local_keys, coverage
                        )
                        remaining_mask = address_counts.gt(0)
                        if not bool(remaining_mask.any()):
                            raise RuntimeError(
                                "Positive transition has an empty residual target set"
                            )
                        spatial_partition = torch.logsumexp(
                            joint_spatial_logits.reshape(-1), dim=0
                        )
                        set_losses.append(
                            (joint_partition - event_score)
                            + spatial_partition
                            - torch.logsumexp(
                                joint_spatial_logits[remaining_mask], dim=0
                            )
                        )
                        (
                            selected,
                            moments,
                            predicted_patch,
                            predicted_local,
                            _predicted_raw,
                        ) = self._select_joint_spatial_event(
                            joint_spatial_logits,
                            local_values,
                            positions,
                            straight_through=True,
                        )
                        residual_set_hits.append(
                            remaining_mask[predicted_patch, predicted_local]
                        )
                    else:
                        local_partition = torch.logsumexp(local_logits, dim=0)
                        remaining_patches = patch_counts.gt(0).nonzero(
                            as_tuple=False
                        ).flatten()
                        if int(remaining_patches.numel()):
                            if self.permutation_invariant_residual_set:
                                set_losses.append(
                                    (joint_partition - event_score)
                                    + patch_partition
                                    - torch.logsumexp(
                                        patch_logits[remaining_patches], dim=0
                                    )
                                )
                            else:
                                set_losses.append(
                                    joint_partition
                                    - torch.logsumexp(
                                        patch_logits[remaining_patches], dim=0
                                    )
                                )

                        if self.permutation_invariant_residual_set:
                            remaining_locals = address_counts[target_patch].gt(0).nonzero(
                                as_tuple=False
                            ).flatten()
                            if int(remaining_locals.numel()):
                                local_set_losses.append(
                                    local_partition
                                    - torch.logsumexp(
                                        local_logits[remaining_locals], dim=0
                                    )
                                )
                        else:
                            patch_losses.append(
                                joint_partition - patch_logits[target_patch]
                            )
                            local_losses.append(
                                local_partition - local_logits[target_local]
                            )

                        predicted_patch = patch_logits.argmax()
                        predicted_local_logits = torch.matmul(
                            local_keys[predicted_patch], query
                        ) * logit_scale
                        predicted_local = predicted_local_logits.argmax()
                        selected = local_values[target_patch, target_local]
                        flat_index = self._flat_event_index(
                            target_patch, target_local
                        )
                        moments = torch.cat(
                            (positions[flat_index], positions.new_zeros(2))
                        )
                    exact_hits.append(
                        predicted_patch.eq(target_patch)
                        & predicted_local.eq(target_local)
                    )
                    coverage = coverage.clone()
                    coverage[target_patch, target_local] += 1.0
                    address_counts[target_patch, target_local] -= 1
                    patch_counts[target_patch] -= 1
                    positive_offset += 1
                else:
                    stop_losses.append(joint_partition - null_score)
                    if self.joint_residual_set_selection:
                        joint_spatial_logits = self._joint_spatial_logits(
                            query, patch_logits, local_keys, coverage
                        )
                        selected, moments, _patch, _local, _raw = (
                            self._select_joint_spatial_event(
                                joint_spatial_logits,
                                local_values,
                                positions,
                                straight_through=True,
                            )
                        )
                    else:
                        predicted_patch = patch_logits.argmax()
                        predicted_local = local_logits.argmax()
                        selected = local_values[predicted_patch, predicted_local]
                        flat_index = self._flat_event_index(
                            predicted_patch, predicted_local
                        )
                        moments = torch.cat(
                            (positions[flat_index], positions.new_zeros(2))
                        )

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
                if len(field_rows) < 4:
                    field_rows.append(patch_logits.detach())

            if positive_offset != len(positive_indices):
                raise RuntimeError(
                    "Causal transition group did not consume every positive target"
                )

        self._set_logits = ()
        self._set_selected_features = tuple(selected_rows)
        self._set_spatial_moments = tuple(moment_rows)
        self._set_coordinate_residuals = self._set_coordinate_signals(
            hidden_states, transitions, selected_rows, moment_rows
        )
        self._event_negative_features = tuple(None for _ in transitions)
        self._event_negative_moments = tuple(None for _ in transitions)
        self._continuation_deltas = tuple(continuation)
        self._presence_deltas = tuple(presence)
        self._event_base_rows = field_rows
        self._event_residual_rows = field_rows
        self._event_raw_gates = tuple(raw_gates)
        self._event_peak_recalls = tuple(value.float() for value in exact_hits)
        self._event_heatmap_loss = None
        self._event_hard_negative_loss = None
        self._causal_patch_loss = torch.stack(patch_losses).mean() if patch_losses else None
        self._causal_local_loss = torch.stack(local_losses).mean() if local_losses else None
        self._causal_set_loss = torch.stack(set_losses).mean() if set_losses else None
        self._causal_local_set_loss = (
            torch.stack(local_set_losses).mean() if local_set_losses else None
        )
        self._causal_stop_loss = torch.stack(stop_losses).mean() if stop_losses else None
        self._causal_exact_hits = tuple(value.float() for value in exact_hits)
        self._causal_residual_set_hits = tuple(
            value.float() for value in residual_set_hits
        )

        terms: list[torch.Tensor] = []
        weights: list[float] = []
        if self._causal_patch_loss is not None:
            terms.append(self.hazard_exact_weight * self._causal_patch_loss)
            weights.append(self.hazard_exact_weight)
        if self._causal_local_loss is not None:
            terms.append(self.hazard_exact_weight * self._causal_local_loss)
            weights.append(self.hazard_exact_weight)
        if self._causal_set_loss is not None:
            terms.append(self.hazard_set_weight * self._causal_set_loss)
            weights.append(self.hazard_set_weight)
        if self._causal_local_set_loss is not None:
            terms.append(self.hazard_set_weight * self._causal_local_set_loss)
            weights.append(self.hazard_set_weight)
        if self._causal_stop_loss is not None:
            terms.append(self.hazard_stop_weight * self._causal_stop_loss)
            weights.append(self.hazard_stop_weight)
        self._event_gate_loss = (
            torch.stack(terms).sum() / max(sum(weights), 1e-8) if terms else None
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
        patch_keys, local_keys, local_values = self._hierarchical_field(hidden_states)
        positions = self._subpatch_positions
        if positions is None:
            raise RuntimeError("Causal event positions are unavailable")
        positions = positions.to(hidden_states.device, dtype=torch.float32)
        coverage = self._coverage_matrix(self._coverage, device=hidden_states.device)
        patch_logits, _unused_local, null_score, query = self._query_scores(
            hidden_states, transition, patch_keys, local_keys, coverage
        )
        if self.joint_residual_set_selection:
            joint_spatial_logits = self._joint_spatial_logits(
                query, patch_logits, local_keys, coverage
            )
            selected, moments, patch_index, local_index, flat_index = (
                self._select_joint_spatial_event(
                    joint_spatial_logits,
                    local_values,
                    positions,
                    straight_through=False,
                )
            )
            local_logits = joint_spatial_logits[patch_index]
        else:
            patch_index = patch_logits.argmax()
            local_logits = (
                torch.matmul(local_keys[patch_index], query)
                * self.hazard_logit_scale().float()
            )
            local_logits = local_logits.float() - self.revisit_penalty().float() * torch.log1p(
                coverage[patch_index]
            )
            local_index = local_logits.argmax()
            flat_index = self._flat_event_index(patch_index, local_index)
            selected = local_values[patch_index, local_index]
            moments = torch.cat((positions[flat_index], positions.new_zeros(2)))
        support = self._event_support_for_index(flat_index, device=hidden_states.device)
        self._pending_inference_event = support
        patch_partition = torch.logsumexp(patch_logits, dim=0)
        if self.permutation_invariant_residual_set:
            event_score = patch_partition - math.log(
                max(int(patch_logits.numel()), 1)
            )
            raw_gate = event_score - null_score
        else:
            event_score = patch_partition
            raw_gate = self.hazard_temperature * (patch_partition - null_score)
        joint_partition = torch.logaddexp(event_score, null_score)
        event_probability = (event_score - joint_partition).exp()
        stop_probability = (null_score - joint_partition).exp()
        maximum = (
            self.maximum_presence_bias
            if transition.predecessor == "ref"
            else self.maximum_continuation_bias
        )
        delta = maximum * torch.tanh(self.event_gate_scale() * raw_gate)

        self._set_logits = (patch_logits, local_logits)
        self._set_selected_features = (selected,)
        self._set_spatial_moments = (moments,)
        self._set_coordinate_residuals = self._set_coordinate_signals(
            hidden_states, (transition,), (selected,), (moments,)
        )
        self._continuation_deltas = (delta,)
        self._presence_deltas = (
            delta if transition.predecessor == "ref" else None,
        )
        self._event_base_rows = [patch_logits]
        self._event_residual_rows = [patch_logits]
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
                "causal_patch_loss": 0.0
                if self._causal_patch_loss is None
                else float(self._causal_patch_loss.detach()),
                "causal_local_loss": 0.0
                if self._causal_local_loss is None
                else float(self._causal_local_loss.detach()),
                "causal_set_loss": 0.0
                if self._causal_set_loss is None
                else float(self._causal_set_loss.detach()),
                "causal_local_set_loss": 0.0
                if self._causal_local_set_loss is None
                else float(self._causal_local_set_loss.detach()),
                "causal_stop_loss": 0.0
                if self._causal_stop_loss is None
                else float(self._causal_stop_loss.detach()),
                "causal_exact_address_accuracy": 0.0
                if not self._causal_exact_hits
                else float(torch.stack(self._causal_exact_hits).mean().detach()),
                "causal_revisit_penalty": float(self.revisit_penalty().detach()),
                "causal_hazard_logit_scale": float(
                    self.hazard_logit_scale().detach()
                ),
                "causal_permutation_invariant_residual_set": float(
                    self.permutation_invariant_residual_set
                ),
                "causal_hazard_fused_token_mass": float(
                    self.hazard_fused_token_mass
                ),
                "causal_permutation_marginalized_pbd": float(
                    self.permutation_marginalized_pbd
                ),
                "causal_joint_residual_set_selection": float(
                    self.joint_residual_set_selection
                ),
                "causal_residual_set_hit_rate": 0.0
                if not self._causal_residual_set_hits
                else float(
                    torch.stack(self._causal_residual_set_hits).mean().detach()
                ),
                "causal_patch_count": 0.0
                if self._grid_hw is None
                else float(self._grid_hw[0] * self._grid_hw[1]),
                "causal_local_cells_per_patch": float(
                    self.subpatch_grid * self.subpatch_grid
                ),
            }
        )

    @staticmethod
    def _logsumexp_without_index(
        row_logits: torch.Tensor, excluded_index: int
    ) -> torch.Tensor:
        left = torch.logsumexp(row_logits[:excluded_index].float(), dim=0)
        right = torch.logsumexp(row_logits[excluded_index + 1 :].float(), dim=0)
        return torch.logaddexp(left, right)

    def continuation_box_shift(
        self, row_logits: torch.Tensor, transition_index: int
    ) -> torch.Tensor:
        """Return the post-box logit shift for the native ``<box>`` token.

        In hazard-fused mode this exactly assigns the learned event probability
        to the continue branch while preserving the native relative
        distribution over every stop/semantic-transition token.  It is a
        probabilistic grammar factorization, not a threshold or object cap.
        """

        if not self.hazard_fused_token_mass:
            return self._continuation_deltas[transition_index]
        if transition_index >= len(self._hazard_event_probabilities):
            raise RuntimeError("Missing causal event probability for token fusion")
        box_start = int(self.token_ids["box_start_token_id"])
        stop_lse = self._logsumexp_without_index(row_logits, box_start)
        native_log_odds = row_logits[box_start].float() - stop_lse
        event_probability = self._hazard_event_probabilities[transition_index].float()
        stop_probability = self._hazard_stop_probabilities[transition_index].float()
        target_log_odds = event_probability.clamp(1e-6, 1.0 - 1e-6).log()
        target_log_odds = target_log_odds - stop_probability.clamp(
            1e-6, 1.0 - 1e-6
        ).log()
        return target_log_odds - native_log_odds

    def apply_continuation_residual(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.hazard_fused_token_mass:
            return super().apply_continuation_residual(logits)
        if not self._set_transitions or not self._continuation_deltas:
            return logits
        box_start = int(self.token_ids["box_start_token_id"])
        updated = logits.clone()
        for transition_index, transition in enumerate(self._set_transitions):
            if (
                transition.predecessor != "box"
                or transition.row[0] >= int(logits.shape[1])
            ):
                continue
            row = updated[0, transition.row[0]]
            shift = self.continuation_box_shift(row, transition_index)
            updated[0, transition.row[0], box_start] += shift.to(logits.dtype)
        return updated

    def finish_forward(self) -> None:
        super().finish_forward()
        self._clear_causal_state()


def load_causal_hierarchical_point_reentry_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if not bool(config.get("causal_hierarchical_hazard")):
        raise ValueError("Checkpoint does not enable causal hierarchical re-entry")
    configure_lora_only(model, int(config["lora_rank"]))
    controller = CausalHierarchicalPointReentryPBD(
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
        initial_revisit_penalty=float(config.get("initial_revisit_penalty", 0.10)),
        maximum_revisit_penalty=float(config.get("maximum_revisit_penalty", 2.0)),
        permutation_invariant_residual_set=bool(
            config.get("permutation_invariant_residual_set", False)
        ),
        hazard_fused_token_mass=bool(config.get("hazard_fused_token_mass", False)),
        permutation_marginalized_pbd=bool(
            config.get("permutation_marginalized_pbd", False)
        ),
        joint_residual_set_selection=bool(
            config.get("joint_residual_set_selection", False)
        ),
        initial_hazard_logit_scale=float(
            config.get("initial_hazard_logit_scale", 8.0)
        ),
        maximum_hazard_logit_scale=float(
            config.get("maximum_hazard_logit_scale", 100.0)
        ),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
