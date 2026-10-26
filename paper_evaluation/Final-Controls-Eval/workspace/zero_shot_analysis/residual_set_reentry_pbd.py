"""Cap-free residual-set visual re-entry for LocateAnything PBD.

The controller keeps LocateAnything's native language grammar and one shared
decoder.  A native provisional box distribution addresses cached fine visual
evidence (via :mod:`distribution_addressed_reentry_pbd`).  In parallel, each
accepted geometry is written into a recurrent coverage field.  The uncovered
field predicts whether another geometry block remains for the current
reference.

Two constraints are deliberate:

* remaining-object supervision is permutation invariant: a transition is
  trained against the union of *all* still-unemitted GT centers, not a fixed
  next-object slot;
* the continuation controller can alter only the native ``<box>`` logit after
  ``</box>``.  It cannot rewrite semantic labels, ``None``, EOS, or hidden
  states.

The final coordinate and continuation projections are zero initialized, so an
untrained controller is exactly the parent model.  Object count remains dynamic
and is bounded only by the decoder sequence budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .distribution_addressed_reentry_pbd import (
    IGNORE_INDEX,
    DistributionAddressedVisualReentryPBD,
)
from .pointrelay_pbd import (
    RelayContext,
    configure_lora_only,
    restore_trainable_state,
)


@dataclass(frozen=True)
class ResidualSetTransition:
    """One native PBD decision and its order-free remaining target set."""

    row: tuple[int, int, int, int, int, int]
    predecessor: str
    reset_coverage: bool
    target_box: tuple[int, int, int, int] | None
    remaining_boxes: tuple[tuple[int, int, int, int], ...]


def residual_set_transitions_from_training(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    token_ids: Mapping[str, int],
) -> tuple[ResidualSetTransition, ...]:
    """Recover positive and stop transitions from LA's exact MTP layout.

    For a positive transition, ``remaining_boxes`` contains the current box and
    every later numeric box before the next semantic reference.  The target is
    therefore a set and does not privilege the serialized next-box order.
    """

    if input_ids.ndim != 2 or labels.ndim != 2 or input_ids.shape != labels.shape:
        raise ValueError("Residual-set transitions require matching [1, sequence] tensors")
    if int(input_ids.shape[0]) != 1:
        raise ValueError("Residual-set pilot supports one packed sample at a time")

    ids = input_ids[0]
    targets = labels[0]
    mask = int(token_ids["default_mask_token_id"])
    ref_end = int(token_ids["ref_end_token_id"])
    box_end = int(token_ids["box_end_token_id"])
    box_start = int(token_ids["box_start_token_id"])
    coord_start = int(token_ids["coord_start_token_id"])
    coord_end = int(token_ids["coord_end_token_id"])

    raw: list[
        tuple[
            tuple[int, int, int, int, int, int],
            str,
            bool,
            tuple[int, int, int, int] | None,
        ]
    ] = []
    for start in range(max(0, int(ids.numel()) - 6)):
        if start + 6 >= int(targets.numel()):
            break
        predecessor_id = int(ids[start])
        if predecessor_id not in {ref_end, box_end}:
            continue
        if not bool(ids[start + 1 : start + 6].eq(mask).all()):
            continue
        block = [int(value) for value in targets[start + 1 : start + 7].tolist()]
        numeric = (
            len(block) == 6
            and block[0] == box_start
            and block[5] == box_end
            and all(coord_start <= value <= coord_end for value in block[1:5])
        )
        box = (
            tuple(int(value - coord_start) for value in block[1:5])
            if numeric
            else None
        )
        raw.append(
            (
                tuple(range(start, start + 6)),
                "ref" if predecessor_id == ref_end else "box",
                predecessor_id == ref_end,
                box,
            )
        )

    transitions: list[ResidualSetTransition] = []
    for index, (row, predecessor, reset, target_box) in enumerate(raw):
        remaining: list[tuple[int, int, int, int]] = []
        for later_index in range(index, len(raw)):
            _later_row, _later_predecessor, later_reset, later_box = raw[later_index]
            if later_index > index and later_reset:
                break
            if later_box is None:
                break
            remaining.append(later_box)
        transitions.append(
            ResidualSetTransition(
                row=row,
                predecessor=predecessor,
                reset_coverage=reset,
                target_box=target_box,
                remaining_boxes=tuple(remaining),
            )
        )
    return tuple(transitions)


class ResidualSetVisualReentryPBD(DistributionAddressedVisualReentryPBD):
    """Distribution-addressed geometry plus isolated dynamic continuation."""

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
        initial_coverage_strength: float = 5.0,
        maximum_continuation_bias: float = 3.0,
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
        )
        if field_temperature <= 0 or target_sigma_cells <= 0:
            raise ValueError("Residual-set temperature and target sigma must be positive")
        if initial_coverage_strength <= 0 or maximum_continuation_bias <= 0:
            raise ValueError("Residual-set strengths must be positive")

        self.field_temperature = float(field_temperature)
        self.target_sigma_cells = float(target_sigma_cells)
        self.maximum_continuation_bias = float(maximum_continuation_bias)
        self.set_query_proj = nn.Linear(hidden_size, evidence_dim, bias=False)
        self.set_key_proj = nn.Linear(evidence_dim, evidence_dim, bias=False)
        self.set_value_proj = nn.Linear(evidence_dim, evidence_dim, bias=False)
        self.set_position_mlp = nn.Sequential(
            nn.Linear(2, evidence_dim),
            nn.GELU(),
            nn.Linear(evidence_dim, evidence_dim),
        )
        self.continuation_head = nn.Sequential(
            nn.Linear(2 * evidence_dim + 4, evidence_dim),
            nn.GELU(),
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, 1),
        )
        self.coverage_strength_logit = nn.Parameter(
            torch.tensor(math.log(math.expm1(initial_coverage_strength)))
        )
        nn.init.xavier_uniform_(self.set_query_proj.weight)
        nn.init.xavier_uniform_(self.set_key_proj.weight)
        nn.init.xavier_uniform_(self.set_value_proj.weight)
        nn.init.xavier_uniform_(self.set_position_mlp[0].weight)
        nn.init.zeros_(self.set_position_mlp[0].bias)
        nn.init.xavier_uniform_(self.set_position_mlp[2].weight)
        nn.init.zeros_(self.set_position_mlp[2].bias)
        nn.init.zeros_(self.continuation_head[-1].weight)
        nn.init.zeros_(self.continuation_head[-1].bias)

        self._set_transitions: tuple[ResidualSetTransition, ...] = ()
        self._set_logits: tuple[torch.Tensor, ...] = ()
        self._set_selected_features: tuple[torch.Tensor, ...] = ()
        self._set_spatial_moments: tuple[torch.Tensor, ...] = ()
        self._continuation_deltas: tuple[torch.Tensor, ...] = ()
        self._set_route_loss: torch.Tensor | None = None
        self._continuation_loss: torch.Tensor | None = None
        self._coverage: torch.Tensor | None = None
        self._set_keys_cache: torch.Tensor | None = None
        self._set_values_cache: torch.Tensor | None = None
        self._last_written_geometry_key: tuple[Any, ...] | None = None
        self._inference_position_key: int | None = None
        self.geometry_writeback_count = 0

    @property
    def set_route_loss(self) -> torch.Tensor | None:
        return self._set_route_loss

    @property
    def continuation_loss(self) -> torch.Tensor | None:
        return self._continuation_loss

    @property
    def continuation_deltas(self) -> tuple[torch.Tensor, ...]:
        return self._continuation_deltas

    def coverage_strength(self) -> torch.Tensor:
        return F.softplus(self.coverage_strength_logit)

    def _merge_coverage(
        self,
        current: torch.Tensor | None,
        support: torch.Tensor,
    ) -> torch.Tensor:
        """Merge one accepted event into recurrent visual memory.

        The released residual-set variants use binary-style max coverage.
        Count-preserving subclasses may override this operation without
        changing native geometry parsing or decoder hooks.
        """

        if current is None:
            return support
        return torch.maximum(current.to(support), support)

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        super()._reset_sample_state(preserve_pixel_field=preserve_pixel_field)
        self._set_transitions = ()
        self._set_logits = ()
        self._set_selected_features = ()
        self._set_spatial_moments = ()
        self._continuation_deltas = ()
        self._set_route_loss = None
        self._continuation_loss = None
        self._coverage = None
        self._set_keys_cache = None
        self._set_values_cache = None
        self._last_written_geometry_key = None
        self._inference_position_key = None
        self.geometry_writeback_count = 0

    def set_training_context(
        self,
        context: RelayContext,
        image_grid_hws: Any,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> None:
        super().set_training_context(context, image_grid_hws)
        if input_ids is None or labels is None:
            raise ValueError("Residual-set training requires input_ids and labels")
        self._set_transitions = residual_set_transitions_from_training(
            input_ids, labels, token_ids=self.token_ids
        )

    def prepare_pixel_field(self, pixel_values: torch.Tensor, image_grid_hws: Any) -> None:
        super().prepare_pixel_field(pixel_values, image_grid_hws)
        self._set_keys_cache = None
        self._set_values_cache = None

    def _vision_hook(
        self, module: nn.Module, args: tuple[Any, ...], output: Any
    ) -> None:
        super()._vision_hook(module, args, output)
        self._set_keys_cache = None
        self._set_values_cache = None

    def _inference_transition(self, input_ids: torch.Tensor) -> None:
        active = self._context
        if active is None or not active.rows:
            self._set_transitions = ()
            return
        row = active.rows[-1]
        predecessor_id = int(input_ids[0, row[0]])
        ref_end = int(self.token_ids["ref_end_token_id"])
        box_end = int(self.token_ids["box_end_token_id"])
        if predecessor_id == ref_end:
            self._coverage = None
            self._last_written_geometry_key = None
            predecessor = "ref"
        elif predecessor_id == box_end:
            predecessor = "box"
            self._write_previous_geometry(input_ids, row[0])
        else:
            self._set_transitions = ()
            return
        self._set_transitions = (
            ResidualSetTransition(row, predecessor, predecessor == "ref", None, ()),
        )

    def _model_pre_hook(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if self._training_context:
            return None
        super()._model_pre_hook(module, args, kwargs)
        position_ids = kwargs.get("position_ids")
        self._inference_position_key = (
            int(position_ids.reshape(-1)[-1])
            if isinstance(position_ids, torch.Tensor) and position_ids.numel()
            else None
        )
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if isinstance(input_ids, torch.Tensor):
            self._inference_transition(input_ids)
        else:
            self._set_transitions = ()
        return None

    def _center_support(
        self,
        box: Sequence[int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if self._subpatch_positions is None or self._grid_hw is None:
            raise RuntimeError("Residual-set field is unavailable")
        positions = self._subpatch_positions.to(device=device, dtype=torch.float32)
        raw_h = self._grid_hw[0] * self.subpatch_grid
        raw_w = self._grid_hw[1] * self.subpatch_grid
        x1, y1, x2, y2 = [float(value) / 1000.0 for value in box]
        center = positions.new_tensor([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
        scale = positions.new_tensor([float(raw_w), float(raw_h)])
        distance_sq = ((positions - center) * scale).square().sum(dim=-1)
        support = torch.exp(
            -0.5 * distance_sq / (self.target_sigma_cells * self.target_sigma_cells)
        )
        if float(support.max()) < 1e-8:
            support[distance_sq.argmin()] = 1.0
        return support / support.sum().clamp_min(1e-12)

    def _remaining_support(
        self,
        boxes: Sequence[Sequence[int]],
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not boxes:
            return None
        supports = torch.stack(
            [self._center_support(box, device=device) for box in boxes]
        )
        # A transition may select any still-unemitted object.  The max-union
        # is an order-free multiple-instance target rather than a normalized
        # average that would force attention to spread across every object.
        target = supports.amax(dim=0)
        return target / target.max().clamp_min(1e-12)

    def _selection_weights(self, field_logits: torch.Tensor) -> torch.Tensor:
        """Return the visual witness weights used by the geometry readout."""

        return field_logits.softmax(dim=-1)

    def _residualize_field_logits(
        self,
        field_logits: torch.Tensor,
        coverage: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the controller's recurrent consumed-event memory."""

        return field_logits - strength * coverage

    def _selection_weights_for_transition(
        self,
        field_logits: torch.Tensor,
        transition: ResidualSetTransition,
    ) -> torch.Tensor:
        """Return witness weights for one transition.

        Most controllers use only the learned prior field.  Posterior-guided
        variants override this hook so teacher information can affect the
        latent witness during training without entering the inference path.
        """

        del transition
        return self._selection_weights(field_logits)

    def _route_target_boxes(
        self, transition: ResidualSetTransition
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Return the set-valued route target for one transition."""

        return transition.remaining_boxes

    def _coverage_write_support(
        self,
        transition: ResidualSetTransition,
        selection_weights: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return the support written after a training transition."""

        if transition.target_box is None:
            return None
        return self._center_support(
            transition.target_box, device=selection_weights.device
        )

    def _write_previous_geometry(self, input_ids: torch.Tensor, row_start: int) -> None:
        if self._subpatch_positions is None:
            return
        box_start = int(self.token_ids["box_start_token_id"])
        box_end = int(self.token_ids["box_end_token_id"])
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        block: tuple[int, ...] = ()
        if row_start >= 6:
            candidate = tuple(
                int(value) for value in input_ids[0, row_start - 6 : row_start].tolist()
            )
            if (
                len(candidate) == 6
                and candidate[0] == box_start
                and candidate[-1] == box_end
                and all(coord_start <= value <= coord_end for value in candidate[1:5])
            ):
                block = candidate
        if not block:
            return
        key = (self._inference_position_key, int(row_start), block)
        if key == self._last_written_geometry_key:
            return
        box = tuple(value - coord_start for value in block[1:5])
        support = self._center_support(box, device=self._subpatch_positions.device)
        self._coverage = self._merge_coverage(self._coverage, support)
        self._last_written_geometry_key = key
        self.geometry_writeback_count += 1

    def _field(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._fine_tokens is None
            or self._subpatch_tokens is None
            or self._subpatch_positions is None
            or self._grid_hw is None
        ):
            raise RuntimeError("Residual-set visual field has not been prepared")
        if self._set_keys_cache is not None and self._set_values_cache is not None:
            return self._set_keys_cache, self._set_values_cache

        route_dtype = self.fine_value_proj.weight.dtype
        fine_values = self._fine_values_cache
        if fine_values is None:
            fine_values = self.fine_value_proj(
                self._fine_tokens.to(hidden_states.device, dtype=route_dtype)
            )
            self._fine_values_cache = fine_values
        raw_values = self._subpatch_values_cache
        if raw_values is None:
            raw_values = self.subpatch_value_proj(
                self._subpatch_tokens.to(hidden_states.device, dtype=route_dtype)
            )
            self._subpatch_values_cache = raw_values

        grid_h, grid_w = self._grid_hw
        raw_h, raw_w = grid_h * self.subpatch_grid, grid_w * self.subpatch_grid
        linear = torch.arange(raw_h * raw_w, device=hidden_states.device)
        cell_y = torch.div(linear, raw_w, rounding_mode="floor")
        cell_x = linear.remainder(raw_w)
        patch_index = torch.div(cell_y, self.subpatch_grid, rounding_mode="floor") * grid_w
        patch_index = patch_index + torch.div(
            cell_x, self.subpatch_grid, rounding_mode="floor"
        )
        positional = self.set_position_mlp(
            self._subpatch_positions.to(hidden_states.device, dtype=route_dtype)
        )
        fused = fine_values[patch_index] + raw_values + positional
        self._set_keys_cache = F.normalize(self.set_key_proj(fused).float(), dim=-1)
        self._set_values_cache = self.set_value_proj(fused).float()
        return self._set_keys_cache, self._set_values_cache

    def prepare_set_signals(self, hidden_states: torch.Tensor) -> None:
        transitions = tuple(
            transition
            for transition in self._set_transitions
            if max(transition.row) < int(hidden_states.shape[1])
        )
        if not transitions or self._subpatch_positions is None:
            self._set_logits = ()
            self._set_selected_features = ()
            self._set_spatial_moments = ()
            self._continuation_deltas = ()
            self._set_route_loss = None
            self._continuation_loss = None
            return

        keys, values = self._field(hidden_states)
        cells = int(keys.shape[0])
        coverage = (
            torch.zeros(cells, device=hidden_states.device, dtype=torch.float32)
            if self._coverage is None or int(self._coverage.numel()) != cells
            else self._coverage.to(hidden_states.device, dtype=torch.float32)
        )
        strength = self.coverage_strength().float()
        route_losses: list[torch.Tensor] = []
        positive_existence_losses: list[torch.Tensor] = []
        negative_existence_losses: list[torch.Tensor] = []
        logits_rows: list[torch.Tensor] = []
        selected_rows: list[torch.Tensor] = []
        moment_rows: list[torch.Tensor] = []
        deltas: list[torch.Tensor] = []
        labels: list[float] = []
        target_masses: list[torch.Tensor] = []

        for transition in transitions:
            if transition.reset_coverage:
                coverage = torch.zeros_like(coverage)
            query = F.normalize(
                self.set_query_proj(
                    hidden_states[0, transition.row[0]].to(
                        dtype=self.set_query_proj.weight.dtype
                    )
                ).float(),
                dim=-1,
            )
            field_logits = torch.matmul(keys, query) / self.field_temperature
            field_logits = self._residualize_field_logits(
                field_logits, coverage, strength
            )
            probabilities = field_logits.softmax(dim=-1)
            selection_weights = self._selection_weights_for_transition(
                field_logits, transition
            )
            selected = torch.matmul(selection_weights, values)
            positions = self._subpatch_positions.to(
                hidden_states.device, dtype=torch.float32
            )
            spatial_mean = torch.matmul(selection_weights, positions)
            spatial_variance = torch.matmul(
                selection_weights, (positions - spatial_mean).square()
            )
            maximum = field_logits.max() / 10.0
            log_mean_exp = (
                torch.logsumexp(field_logits, dim=0) - math.log(max(1, cells))
            ) / 10.0
            entropy = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum() / math.log(max(2, cells))
            covered_fraction = coverage.gt(0.05).float().mean()
            features = torch.cat(
                (
                    query,
                    F.normalize(selected, dim=-1),
                    maximum.reshape(1),
                    log_mean_exp.reshape(1),
                    entropy.reshape(1),
                    covered_fraction.reshape(1),
                )
            )
            raw_delta = self.continuation_head(features).reshape(())
            delta = self.maximum_continuation_bias * torch.tanh(raw_delta)
            logits_rows.append(field_logits)
            selected_rows.append(selected)
            moment_rows.append(torch.cat((spatial_mean, spatial_variance)))
            deltas.append(delta)

            target = self._remaining_support(
                self._route_target_boxes(transition), device=hidden_states.device
            )
            label = 1.0 if target is not None else 0.0
            labels.append(label)
            if target is not None:
                target_mass = (probabilities * target).sum().clamp_min(1e-12)
                route_losses.append(-target_mass.log())
                target_masses.append(target_mass)
            if transition.predecessor == "box":
                decision_loss = F.binary_cross_entropy_with_logits(
                    raw_delta,
                    raw_delta.new_tensor(label),
                )
                if label:
                    positive_existence_losses.append(decision_loss)
                else:
                    negative_existence_losses.append(decision_loss)

            emitted = self._coverage_write_support(
                transition, selection_weights
            )
            if emitted is not None:
                coverage = self._merge_coverage(coverage, emitted)

        self._set_logits = tuple(logits_rows)
        self._set_selected_features = tuple(selected_rows)
        self._set_spatial_moments = tuple(moment_rows)
        self._continuation_deltas = tuple(deltas)
        self._set_route_loss = (
            torch.stack(route_losses).mean() if route_losses else None
        )
        balanced_decisions = [
            torch.stack(group).mean()
            for group in (positive_existence_losses, negative_existence_losses)
            if group
        ]
        self._continuation_loss = (
            torch.stack(balanced_decisions).mean() if balanced_decisions else None
        )
        self._coverage = coverage.detach()
        self.last_diagnostics.update(
            {
                "set_transitions": float(len(transitions)),
                "set_positive_transitions": float(sum(labels)),
                "set_continue_decisions": float(len(positive_existence_losses)),
                "set_stop_decisions": float(len(negative_existence_losses)),
                "set_route_loss": 0.0
                if self._set_route_loss is None
                else float(self._set_route_loss.detach()),
                "set_continuation_loss": 0.0
                if self._continuation_loss is None
                else float(self._continuation_loss.detach()),
                "set_continuation_delta_abs": float(
                    torch.stack([value.abs() for value in deltas]).mean().detach()
                ),
                "set_target_mass": 0.0
                if not target_masses
                else float(torch.stack(target_masses).mean().detach()),
                "set_coverage_fraction": float(coverage.gt(0.05).float().mean().detach()),
                "set_coverage_strength": float(strength.detach()),
                "set_geometry_writebacks": float(self.geometry_writeback_count),
            }
        )

    def apply_continuation_residual(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._set_transitions or not self._continuation_deltas:
            return logits
        box_start = int(self.token_ids["box_start_token_id"])
        updated = logits.clone()
        for transition, delta in zip(self._set_transitions, self._continuation_deltas):
            # A class-level first block always begins with <box>, including
            # explicit None. Only a post-box decision means continue versus
            # semantic transition/EOS.
            if transition.predecessor != "box" or transition.row[0] >= int(logits.shape[1]):
                continue
            updated[0, transition.row[0], box_start] += delta.to(logits.dtype)
        return updated

    def _lm_head_hook(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        updated = super()._lm_head_hook(module, args, output)
        self.prepare_set_signals(args[0])
        return self.apply_continuation_residual(updated)

    def finish_forward(self) -> None:
        super().finish_forward()
        self._set_transitions = ()
        self._set_logits = ()
        self._set_selected_features = ()
        self._set_spatial_moments = ()
        self._continuation_deltas = ()
        self._set_route_loss = None
        self._continuation_loss = None


def residual_set_reentry_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    controller: ResidualSetVisualReentryPBD,
    *,
    route_loss_weight: float = 0.03,
    continuation_loss_weight: float = 0.20,
    chunk: int = 128,
) -> torch.Tensor:
    """Native token CE with isolated geometry/continuation corrections."""

    coord_start = int(controller.token_ids["coord_start_token_id"])
    coord_end = int(controller.token_ids["coord_end_token_id"])
    geometry_rows = tuple(
        row
        for row in (controller._context.rows if controller._context else ())
        if max(row) < int(hidden_states.shape[1])
    )
    if geometry_rows:
        coord_weight = weight[coord_start : coord_end + 1]
        coordinate_logits = torch.stack(
            [
                F.linear(
                    torch.stack(
                        [hidden_states[0, position] for position in row[1:5]]
                    ),
                    coord_weight,
                )
                for row in geometry_rows
            ]
        )
        controller.prepare_coordinate_residuals(
            hidden_states, coordinate_logits, controller._context
        )
    controller.prepare_set_signals(hidden_states)

    shifted_hidden = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
    shifted_labels = labels[:, 1:].reshape(-1)
    valid = shifted_labels.ne(IGNORE_INDEX)
    valid_hidden = shifted_hidden[valid]
    valid_labels = shifted_labels[valid]
    count = int(valid_labels.numel())
    if count == 0:
        raise RuntimeError("Residual-set sample has no supervised tokens")

    total = hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, count, int(chunk)):
        stop = min(count, start + int(chunk))
        logits = F.linear(valid_hidden[start:stop], weight).float()
        total = total + F.cross_entropy(
            logits, valid_labels[start:stop], reduction="sum"
        )

    residuals = controller.coordinate_residuals
    active_rows = controller.active_rows
    permutation_marginalized_pbd = bool(
        getattr(controller, "permutation_marginalized_pbd", False)
    )
    if residuals is not None and active_rows and permutation_marginalized_pbd:
        # RelayContext may contain native geometry rows that are not residual-set
        # transitions (for example, an auxiliary MTP view).  Match by the exact
        # six-token row instead of relying on positional list alignment.  Rows
        # outside the residual set retain their ordinary token CE unchanged.
        transition_by_row = {
            transition.row: transition for transition in controller._set_transitions
        }
        strength = controller.residual_strength()
        for block_index, row in enumerate(active_rows):
            transition = transition_by_row.get(row)
            if transition is None:
                continue
            if transition.target_box is None or not transition.remaining_boxes:
                continue
            states = torch.stack([hidden_states[0, position] for position in row[1:5]])
            exact_targets = torch.stack(
                [labels[0, position + 1] for position in row[1:5]]
            ).long()
            if bool(exact_targets.eq(IGNORE_INDEX).any()):
                continue
            base_logits = F.linear(states, weight).float()
            modified = base_logits.clone()
            for lane in range(4):
                modified[lane, coord_start : coord_end + 1] += (
                    strength[lane].float() * residuals[block_index, lane].float()
                )
            total = total - F.cross_entropy(
                base_logits, exact_targets, reduction="sum"
            )
            total = total + permutation_marginalized_coordinate_nll(
                modified[:, coord_start : coord_end + 1],
                transition.remaining_boxes,
            )
    elif residuals is not None and active_rows:
        strength = controller.residual_strength()
        corrections: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        for block_index, row in enumerate(active_rows):
            for lane, position in enumerate(row[1:5]):
                target_position = position + 1
                if target_position >= labels.shape[1]:
                    continue
                target = labels[0, target_position]
                if int(target) == IGNORE_INDEX:
                    continue
                corrections.append(
                    (hidden_states[0, position], target, residuals[block_index, lane], lane)
                )
        for start in range(0, len(corrections), 32):
            group = corrections[start : start + 32]
            states = torch.stack([item[0] for item in group])
            targets = torch.stack([item[1] for item in group]).long()
            base_logits = F.linear(states, weight).float()
            modified = base_logits.clone()
            for index, (_state, _target, residual, lane) in enumerate(group):
                modified[index, coord_start : coord_end + 1] += (
                    strength[lane].float() * residual.float()
                )
            total = total - F.cross_entropy(base_logits, targets, reduction="sum")
            total = total + F.cross_entropy(modified, targets, reduction="sum")

    box_start = int(controller.token_ids["box_start_token_id"])
    for transition_index, (transition, delta) in enumerate(
        zip(controller._set_transitions, controller.continuation_deltas)
    ):
        if transition.predecessor != "box":
            continue
        position = transition.row[0]
        target_position = position + 1
        if target_position >= labels.shape[1] or int(labels[0, target_position]) == IGNORE_INDEX:
            continue
        target = labels[0, target_position].reshape(1).long()
        base_logits = F.linear(hidden_states[0, position], weight).float().unsqueeze(0)
        modified = base_logits.clone()
        continuation_box_shift = getattr(controller, "continuation_box_shift", None)
        if callable(continuation_box_shift):
            delta = continuation_box_shift(base_logits[0], transition_index)
        modified[0, box_start] += delta.float()
        total = total - F.cross_entropy(base_logits, target, reduction="sum")
        total = total + F.cross_entropy(modified, target, reduction="sum")

    loss = total / count
    if controller.set_route_loss is not None:
        loss = loss + float(route_loss_weight) * controller.set_route_loss
    if controller.continuation_loss is not None:
        loss = loss + float(continuation_loss_weight) * controller.continuation_loss
    return loss


def permutation_marginalized_coordinate_nll(
    coordinate_logits: torch.Tensor,
    remaining_boxes: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Negative log probability that one PBD block is any residual GT box."""

    if coordinate_logits.ndim != 2 or int(coordinate_logits.shape[0]) != 4:
        raise ValueError("coordinate_logits must have shape [4, coordinate_bins]")
    if not remaining_boxes:
        raise ValueError("remaining_boxes must be non-empty")
    candidates = torch.as_tensor(
        remaining_boxes, device=coordinate_logits.device, dtype=torch.long
    ).reshape(-1, 4)
    if int(candidates.min()) < 0 or int(candidates.max()) >= int(
        coordinate_logits.shape[1]
    ):
        raise ValueError("Residual GT coordinate lies outside the PBD vocabulary")
    log_probabilities = coordinate_logits.float().log_softmax(dim=-1)
    lane_indices = torch.arange(4, device=coordinate_logits.device).reshape(1, 4)
    candidate_log_probabilities = log_probabilities[
        lane_indices, candidates
    ].sum(dim=-1)
    return -torch.logsumexp(candidate_log_probabilities, dim=0)


def load_residual_set_reentry_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Load a matched Qwen LoRA plus residual-set visual controller."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "residual_set_visual_reentry":
        raise ValueError(f"Not a residual-set checkpoint: {config.get('arm')!r}")
    configure_lora_only(model, int(config["lora_rank"]))
    controller = ResidualSetVisualReentryPBD(
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
        initial_coverage_strength=float(config.get("initial_coverage_strength", 5.0)),
        maximum_continuation_bias=float(config.get("maximum_continuation_bias", 3.0)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
