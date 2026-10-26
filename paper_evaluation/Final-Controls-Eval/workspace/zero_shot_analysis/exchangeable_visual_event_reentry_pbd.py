"""Exchangeable residual visual events for LocateAnything PBD.

The released LocateAnything decoder keeps its language reasoning and native PBD
geometry grammar.  This controller adds a latent, query-conditioned visual
event process over one cached fine MoonViT field.  Crucially, its training loss
does not assign an event to the next box in the annotation serialization.  It
marginalizes the geometry mark over every still-unclaimed target and removes
the claimed target from an autonomous residual set.

At inference there is one decoder and one cached image encoding.  An accepted
native geometry block inhibits its visual event, and a learned survival head
decides whether another event remains.  There are no object slots, count
tokens, crops, decoder copies, or fixed object-count cap.
"""

from __future__ import annotations

import math
import weakref
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .pointrelay_pbd import (
    base_causal_lm,
    configure_lora_only,
    restore_trainable_state,
)
from .distribution_addressed_reentry_pbd import IGNORE_INDEX
from .posterior_guided_visual_event_pbd import (
    PosteriorGuidedVisualEventReentryPBD,
)


def exchangeable_mark_nll(
    coordinate_log_probs: torch.Tensor,
    boxes: torch.Tensor,
    responsibilities: torch.Tensor,
) -> torch.Tensor:
    """Marginalize one four-coordinate mark over an unordered target set.

    ``coordinate_log_probs`` has shape ``[4, 1001]``, ``boxes`` has shape
    ``[N, 4]``, and ``responsibilities`` has shape ``[N]``.  Reordering the
    boxes and their responsibilities leaves the result unchanged.
    """

    if coordinate_log_probs.ndim != 2 or int(coordinate_log_probs.shape[0]) != 4:
        raise ValueError("coordinate_log_probs must have shape [4, vocabulary]")
    if boxes.ndim != 2 or tuple(boxes.shape[1:]) != (4,):
        raise ValueError("boxes must have shape [N, 4]")
    if responsibilities.ndim != 1 or int(responsibilities.numel()) != int(
        boxes.shape[0]
    ):
        raise ValueError("responsibilities must align with boxes")
    if int(boxes.shape[0]) == 0:
        raise ValueError("exchangeable mark likelihood requires at least one box")

    boxes = boxes.to(device=coordinate_log_probs.device, dtype=torch.long)
    if int(boxes.min()) < 0 or int(boxes.max()) >= int(
        coordinate_log_probs.shape[1]
    ):
        raise ValueError("box coordinates fall outside the coordinate vocabulary")
    normalized = responsibilities.float().clamp_min(1e-12)
    normalized = normalized / normalized.sum().clamp_min(1e-12)
    expanded = coordinate_log_probs.unsqueeze(0).expand(int(boxes.shape[0]), -1, -1)
    mark_log_prob = expanded.gather(2, boxes.unsqueeze(-1)).squeeze(-1).mean(dim=-1)
    return -torch.logsumexp(normalized.log() + mark_log_prob, dim=0)


class ExchangeableVisualEventReentryPBD(PosteriorGuidedVisualEventReentryPBD):
    """Order-free latent event assignment with an open-ended survival process."""

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
        maximum_continuation_bias: float = 1.5,
        initial_set_coordinate_strength: float = 0.25,
        maximum_set_coordinate_strength: float = 3.0,
        maximum_presence_bias: float = 1.5,
        initial_gate_scale: float = 0.05,
        selection_temperature: float = 0.50,
        posterior_logit_scale: float = 2.0,
        posterior_floor: float = 1e-4,
        initial_presence_rescue_scale: float = 0.10,
        assignment_temperature: float = 0.50,
        initial_survival_influence: float = 0.05,
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
        if assignment_temperature <= 0:
            raise ValueError("assignment_temperature must be positive")
        if not 0 < initial_survival_influence < 1:
            raise ValueError("initial_survival_influence must lie in (0, 1)")
        self.assignment_temperature = float(assignment_temperature)
        self.exchangeable_survival_head = nn.Sequential(
            nn.Linear(2 * evidence_dim + 4, evidence_dim),
            nn.GELU(),
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, 1),
        )
        nn.init.zeros_(self.exchangeable_survival_head[-1].weight)
        nn.init.zeros_(self.exchangeable_survival_head[-1].bias)
        self.survival_influence_logit = nn.Parameter(
            torch.tensor(
                math.log(
                    initial_survival_influence / (1.0 - initial_survival_influence)
                )
            )
        )
        self._exchangeable_event_loss: torch.Tensor | None = None
        self._exchangeable_mark_loss: torch.Tensor | None = None
        self._exchangeable_survival_loss: torch.Tensor | None = None
        self._exchangeable_assignments: tuple[int, ...] = ()
        self._lm_head_ref: Any | None = None

    @property
    def exchangeable_event_loss(self) -> torch.Tensor | None:
        return self._exchangeable_event_loss

    @property
    def exchangeable_mark_loss(self) -> torch.Tensor | None:
        return self._exchangeable_mark_loss

    @property
    def exchangeable_survival_loss(self) -> torch.Tensor | None:
        return self._exchangeable_survival_loss

    def survival_influence(self) -> torch.Tensor:
        return torch.sigmoid(self.survival_influence_logit)

    def install(self, model: nn.Module) -> None:
        self._lm_head_ref = weakref.ref(base_causal_lm(model.language_model).lm_head)
        super().install(model)

    def remove(self) -> None:
        super().remove()
        self._lm_head_ref = None

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        super()._reset_sample_state(preserve_pixel_field=preserve_pixel_field)
        self._exchangeable_event_loss = None
        self._exchangeable_mark_loss = None
        self._exchangeable_survival_loss = None
        self._exchangeable_assignments = ()

    def _survival_features(
        self,
        *,
        query: torch.Tensor,
        selected: torch.Tensor,
        field_logits: torch.Tensor,
        coverage: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = field_logits.float().softmax(dim=-1)
        maximum = field_logits.float().max() / 10.0
        log_mean_exp = (
            torch.logsumexp(field_logits.float(), dim=0)
            - math.log(max(1, int(field_logits.numel())))
        ) / 10.0
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum() / math.log(max(2, int(probabilities.numel())))
        covered_fraction = coverage.gt(0.05).float().mean()
        return torch.cat(
            (
                query.float(),
                F.normalize(selected.float(), dim=-1),
                maximum.reshape(1),
                log_mean_exp.reshape(1),
                entropy.reshape(1),
                covered_fraction.reshape(1),
            )
        )

    def _coordinate_log_probs(
        self,
        hidden_states: torch.Tensor,
        transition: Any,
        selected: torch.Tensor,
        moments: torch.Tensor,
    ) -> torch.Tensor:
        route_dtype = self.set_lane_hidden_proj.weight.dtype
        lane_hidden = self.set_lane_hidden_proj(
            torch.stack(
                [hidden_states[0, position] for position in transition.row[1:5]]
            ).to(route_dtype)
        )
        selected_feature = selected.to(route_dtype).unsqueeze(0).expand(4, -1)
        spatial_feature = self.set_spatial_proj(
            moments.to(route_dtype)
        ).unsqueeze(0).expand(4, -1)
        fused = torch.cat(
            (
                lane_hidden,
                selected_feature,
                spatial_feature,
                self.set_coordinate_lane_embedding.to(route_dtype),
            ),
            dim=-1,
        )
        residual = self.set_coordinate_head(fused).float()
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        lm_head = None if self._lm_head_ref is None else self._lm_head_ref()
        if lm_head is None:
            raise RuntimeError("Exchangeable controller has not been installed")
        weight = lm_head.weight[coord_start : coord_end + 1]
        base = F.linear(
            torch.stack(
                [hidden_states[0, position] for position in transition.row[1:5]]
            ),
            weight,
        ).float()
        strength = self.set_coordinate_strength().float().unsqueeze(1)
        return (base + strength * residual).log_softmax(dim=-1)

    def _target_center_cells(self, boxes: torch.Tensor) -> torch.Tensor:
        """Map normalized HBB centers to cached fine-field cells in O(N)."""

        if self._grid_hw is None:
            raise RuntimeError("Exchangeable visual field has no grid geometry")
        raw_h = int(self._grid_hw[0]) * int(self.subpatch_grid)
        raw_w = int(self._grid_hw[1]) * int(self.subpatch_grid)
        centers_x = 0.0005 * (boxes[:, 0].float() + boxes[:, 2].float())
        centers_y = 0.0005 * (boxes[:, 1].float() + boxes[:, 3].float())
        cell_x = torch.floor(centers_x * raw_w).long().clamp_(0, raw_w - 1)
        cell_y = torch.floor(centers_y * raw_h).long().clamp_(0, raw_h - 1)
        return cell_y * raw_w + cell_x

    def _autonomous_training_rollout(self, hidden_states: torch.Tensor) -> None:
        if not self._training_context or not self._set_transitions:
            return
        keys, values = self._field(hidden_states)
        positions = self._subpatch_positions
        if positions is None:
            return
        positions = positions.to(hidden_states.device, dtype=torch.float32)
        coverage = torch.zeros(
            int(keys.shape[0]), device=hidden_states.device, dtype=torch.float32
        )
        remaining: list[tuple[int, int, int, int]] = []
        event_losses: list[torch.Tensor] = []
        mark_losses: list[torch.Tensor] = []
        positive_survival: list[torch.Tensor] = []
        negative_survival: list[torch.Tensor] = []
        assignments: list[int] = []
        deltas: list[torch.Tensor] = []
        presence: list[torch.Tensor | None] = []
        influence = self.survival_influence()
        strength = self.coverage_strength().float()

        for transition in self._set_transitions:
            if max(transition.row) >= int(hidden_states.shape[1]):
                continue
            if transition.reset_coverage:
                coverage = torch.zeros_like(coverage)
                remaining = [tuple(box) for box in transition.remaining_boxes]

            query = F.normalize(
                self.set_query_proj(
                    hidden_states[0, transition.row[0]].to(
                        self.set_query_proj.weight.dtype
                    )
                ).float(),
                dim=-1,
            )
            field_logits = torch.matmul(keys, query) / self.field_temperature
            field_logits = field_logits - strength * coverage
            probabilities = field_logits.float().softmax(dim=-1)
            selection = F.gumbel_softmax(
                field_logits.float(),
                tau=self.selection_temperature,
                hard=True,
                dim=-1,
            )
            selected = torch.matmul(selection, values.float())
            mean = torch.matmul(selection, positions)
            variance = torch.matmul(selection, (positions - mean).square())
            moments = torch.cat((mean, variance))
            raw_survival = self.exchangeable_survival_head(
                self._survival_features(
                    query=query,
                    selected=selected,
                    field_logits=field_logits,
                    coverage=coverage,
                ).to(self.exchangeable_survival_head[0].weight.dtype)
            ).reshape(())
            exists = bool(remaining)
            survival_loss = F.binary_cross_entropy_with_logits(
                raw_survival, raw_survival.new_tensor(float(exists))
            )
            (positive_survival if exists else negative_survival).append(survival_loss)
            maximum = (
                self.maximum_presence_bias
                if transition.predecessor == "ref"
                else self.maximum_continuation_bias
            )
            delta = maximum * influence * torch.tanh(raw_survival)
            deltas.append(delta)
            presence.append(delta if transition.predecessor == "ref" else None)

            if not remaining:
                continue
            boxes = torch.as_tensor(
                remaining, device=hidden_states.device, dtype=torch.long
            )
            center_cells = self._target_center_cells(boxes)
            # Sparse center addressing avoids materializing an [N, cells]
            # support tensor at every event.  The rollout is therefore
            # O(events * (targets + cells)), not O(events * targets * cells).
            unique_cells = torch.unique(center_cells)
            event_mass = probabilities[unique_cells].sum().clamp(1e-12, 1.0)
            event_losses.append(-event_mass.log())
            masses = probabilities[center_cells].clamp_min(1e-12)
            responsibilities = (
                masses.log() / self.assignment_temperature
            ).softmax(dim=0)
            mark_losses.append(
                exchangeable_mark_nll(
                    self._coordinate_log_probs(
                        hidden_states, transition, selected, moments
                    ),
                    boxes,
                    responsibilities,
                )
            )

            assignment = F.gumbel_softmax(
                masses.log(),
                tau=self.assignment_temperature,
                hard=True,
                dim=0,
            )
            assigned_index = int(assignment.detach().argmax())
            assignments.append(assigned_index)
            assigned_box = torch.matmul(assignment, boxes.float())
            assigned_support = self._center_support(
                assigned_box, device=hidden_states.device
            )
            coverage = torch.maximum(coverage, assigned_support.detach())
            remaining.pop(assigned_index)

        balanced_survival = [
            torch.stack(group).mean()
            for group in (positive_survival, negative_survival)
            if group
        ]
        self._exchangeable_event_loss = (
            torch.stack(event_losses).mean() if event_losses else None
        )
        self._exchangeable_mark_loss = (
            torch.stack(mark_losses).mean() if mark_losses else None
        )
        self._exchangeable_survival_loss = (
            torch.stack(balanced_survival).mean() if balanced_survival else None
        )
        self._exchangeable_assignments = tuple(assignments)
        self._continuation_deltas = tuple(deltas)
        self._presence_deltas = tuple(presence)

    def _inference_survival(self, hidden_states: torch.Tensor) -> None:
        if self._training_context or not self._set_transitions or not self._set_logits:
            return
        cells = int(self._set_logits[0].numel())
        coverage = (
            torch.zeros(cells, device=hidden_states.device, dtype=torch.float32)
            if self._coverage is None or int(self._coverage.numel()) != cells
            else self._coverage.to(hidden_states.device, dtype=torch.float32)
        )
        influence = self.survival_influence()
        deltas: list[torch.Tensor] = []
        presence: list[torch.Tensor | None] = []
        for transition, field_logits, selected in zip(
            self._set_transitions, self._set_logits, self._set_selected_features
        ):
            query = F.normalize(
                self.set_query_proj(
                    hidden_states[0, transition.row[0]].to(
                        self.set_query_proj.weight.dtype
                    )
                ).float(),
                dim=-1,
            )
            raw_survival = self.exchangeable_survival_head(
                self._survival_features(
                    query=query,
                    selected=selected,
                    field_logits=field_logits,
                    coverage=coverage,
                ).to(self.exchangeable_survival_head[0].weight.dtype)
            ).reshape(())
            maximum = (
                self.maximum_presence_bias
                if transition.predecessor == "ref"
                else self.maximum_continuation_bias
            )
            delta = maximum * influence * torch.tanh(raw_survival)
            deltas.append(delta)
            presence.append(delta if transition.predecessor == "ref" else None)
        self._continuation_deltas = tuple(deltas)
        self._presence_deltas = tuple(presence)

    def prepare_set_signals(self, hidden_states: torch.Tensor) -> None:
        super().prepare_set_signals(hidden_states)
        if self._training_context:
            self._autonomous_training_rollout(hidden_states)
        else:
            self._inference_survival(hidden_states)
        self.last_diagnostics.update(
            {
                "exchangeable_event_loss": 0.0
                if self._exchangeable_event_loss is None
                else float(self._exchangeable_event_loss.detach()),
                "exchangeable_mark_loss": 0.0
                if self._exchangeable_mark_loss is None
                else float(self._exchangeable_mark_loss.detach()),
                "exchangeable_survival_loss": 0.0
                if self._exchangeable_survival_loss is None
                else float(self._exchangeable_survival_loss.detach()),
                "exchangeable_assignments": float(len(self._exchangeable_assignments)),
                "exchangeable_survival_influence": float(
                    self.survival_influence().detach()
                ),
            }
        )

    def finish_forward(self) -> None:
        super().finish_forward()
        self._exchangeable_event_loss = None
        self._exchangeable_mark_loss = None
        self._exchangeable_survival_loss = None
        self._exchangeable_assignments = ()


def exchangeable_visual_event_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    controller: ExchangeableVisualEventReentryPBD,
    *,
    coordinate_aux_weight: float = 0.20,
    exchangeable_event_weight: float = 0.25,
    exchangeable_mark_weight: float = 0.30,
    exchangeable_survival_weight: float = 0.30,
) -> torch.Tensor:
    """Native CE plus order-free event, geometry-mark, and survival losses."""

    # Preserve the released model's serialized language/grammar objective for
    # the Qwen LoRA branch.  Controller supervision below is deliberately
    # separate and permutation marginalized; otherwise a latent event that
    # correctly selects a different remaining object would be penalized
    # against the annotation's arbitrary next-box order.
    controller.prepare_set_signals(hidden_states)
    shifted_hidden = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
    shifted_labels = labels[:, 1:].reshape(-1)
    valid = shifted_labels.ne(IGNORE_INDEX)
    valid_hidden = shifted_hidden[valid]
    valid_labels = shifted_labels[valid]
    if not int(valid_labels.numel()):
        raise RuntimeError("Exchangeable visual-event sample has no supervised tokens")
    total = hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, int(valid_labels.numel()), 128):
        stop = min(int(valid_labels.numel()), start + 128)
        logits = F.linear(valid_hidden[start:stop], weight).float()
        total = total + F.cross_entropy(
            logits, valid_labels[start:stop], reduction="sum"
        )
    loss = total / int(valid_labels.numel())

    # Train the learned survival state through the exact logits it changes at
    # inference. Replacing only these CE terms preserves native grammar while
    # giving both the survival head and its bounded influence a real gradient.
    decision_correction = hidden_states.new_zeros((), dtype=torch.float32)
    coord_start = int(controller.token_ids["coord_start_token_id"])
    coord_end = int(controller.token_ids["coord_end_token_id"])
    none_token = int(controller.token_ids["none_token_id"])
    box_start = int(controller.token_ids["box_start_token_id"])
    for transition, continuation, presence in zip(
        controller._set_transitions,
        controller._continuation_deltas,
        controller._presence_deltas,
    ):
        if transition.predecessor == "ref":
            if presence is None:
                continue
            position = transition.row[1]
            delta = presence
        else:
            position = transition.row[0]
            delta = continuation
        target_position = position + 1
        if target_position >= int(labels.shape[1]):
            continue
        target = labels[0, target_position]
        if int(target) == IGNORE_INDEX:
            continue
        base = F.linear(hidden_states[0, position], weight).float()
        modified = base.clone()
        if transition.predecessor == "ref":
            modified[coord_start : coord_end + 1] += delta.float()
            modified[none_token] -= delta.float()
        else:
            modified[box_start] += delta.float()
        decision_correction = decision_correction + F.cross_entropy(
            modified.unsqueeze(0), target.reshape(1).long(), reduction="sum"
        ) - F.cross_entropy(
            base.unsqueeze(0), target.reshape(1).long(), reduction="sum"
        )
    decision_correction = decision_correction / int(valid_labels.numel())
    loss = loss + decision_correction
    controller.last_diagnostics["exchangeable_decision_correction"] = float(
        decision_correction.detach()
    )
    for value, scale in (
        (controller.exchangeable_event_loss, exchangeable_event_weight),
        (
            controller.exchangeable_mark_loss,
            exchangeable_mark_weight + coordinate_aux_weight,
        ),
        (controller.exchangeable_survival_loss, exchangeable_survival_weight),
    ):
        if value is not None:
            loss = loss + float(scale) * value
    return loss


def load_exchangeable_visual_event_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "exchangeable_visual_event_reentry":
        raise ValueError(
            f"Not an exchangeable visual-event checkpoint: {config.get('arm')!r}"
        )
    configure_lora_only(model, int(config["lora_rank"]))
    controller = ExchangeableVisualEventReentryPBD(
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
        maximum_continuation_bias=float(config.get("maximum_continuation_bias", 1.5)),
        initial_set_coordinate_strength=float(
            config.get("initial_set_coordinate_strength", 0.25)
        ),
        maximum_set_coordinate_strength=float(
            config.get("maximum_set_coordinate_strength", 3.0)
        ),
        maximum_presence_bias=float(config.get("maximum_presence_bias", 1.5)),
        initial_gate_scale=float(config.get("initial_gate_scale", 0.05)),
        selection_temperature=float(config.get("selection_temperature", 0.50)),
        posterior_logit_scale=float(config.get("posterior_logit_scale", 2.0)),
        posterior_floor=float(config.get("posterior_floor", 1e-4)),
        initial_presence_rescue_scale=float(
            config.get("initial_presence_rescue_scale", 0.10)
        ),
        assignment_temperature=float(config.get("assignment_temperature", 0.50)),
        initial_survival_influence=float(
            config.get("initial_survival_influence", 0.05)
        ),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
