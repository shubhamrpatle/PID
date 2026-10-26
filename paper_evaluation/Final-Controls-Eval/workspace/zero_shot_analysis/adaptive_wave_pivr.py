"""Adaptive-wave point-indexed visual re-entry for LocateAnything.

The controller keeps one MoonViT encoding and one recurrent Qwen decoder.  It
learns how many *additional* native PBD geometry blocks should be attempted in
the next execution wave.  A wave is a compute schedule, not an object slot set:
its branches share the same Qwen weights and prefix KV state, retrieve distinct
locations from the cached fine visual field through recurrent coverage, and are
serialized back into LocateAnything's ordinary output grammar.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .ravel_pbd import RavelContext, RavelEvent, RavelPBDController


class AdaptiveWavePIVRController(RavelPBDController):
    """Learned point addressing with adaptive sequential/wave PBD execution."""

    controller_kind = "adaptive_wave_pivr"
    integrated_generation_observer = True

    def __init__(
        self,
        *,
        hidden_size: int,
        vision_hidden_size: int,
        relay_layer: int,
        token_ids: Mapping[str, int],
        evidence_dim: int = 128,
        address_temperature: float = 0.10,
        local_sigmas: Sequence[float] = (0.75, 1.5, 3.0),
        coverage_penalty: float = 2.0,
        maximum_residual_strength: float = 1.0,
        isolate_controller_gradients: bool = True,
        routing_control: str = "learned",
        cached_crop_radius_cells: float = 1.5,
        wave_sizes: Sequence[int] = (1, 2, 4),
        wave_continue_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            vision_hidden_size=vision_hidden_size,
            relay_layer=relay_layer,
            token_ids=token_ids,
            evidence_dim=evidence_dim,
            address_temperature=address_temperature,
            local_sigmas=local_sigmas,
            coverage_penalty=coverage_penalty,
            maximum_residual_strength=maximum_residual_strength,
            isolate_controller_gradients=isolate_controller_gradients,
            routing_control=routing_control,
            cached_crop_radius_cells=cached_crop_radius_cells,
        )
        normalized_sizes = tuple(sorted({int(value) for value in wave_sizes}))
        if not normalized_sizes or normalized_sizes[0] != 1:
            raise ValueError("wave_sizes must contain 1")
        if any(value <= 0 for value in normalized_sizes):
            raise ValueError("wave_sizes must be positive")
        if not 0.0 < wave_continue_threshold < 1.0:
            raise ValueError("wave_continue_threshold must be in (0, 1)")
        self.wave_sizes = normalized_sizes
        self.wave_continue_threshold = float(wave_continue_threshold)
        self.wave_head = nn.Sequential(
            nn.Linear(self.common_feature_width, self.evidence_dim),
            nn.GELU(),
            nn.Linear(self.evidence_dim, len(self.wave_sizes)),
        )
        self._wave_logits: list[torch.Tensor] = []
        self._wave_targets: list[int] = []
        self._wave_loss: torch.Tensor | None = None
        self._next_wave_size = 1
        self._active_wave_size = 1
        self._wave_pending_attentions: tuple[torch.Tensor, ...] = ()
        self._wave_continue_probabilities: tuple[float, ...] = ()
        self._wave_size_predictions: tuple[int, ...] = ()
        self.parallel_wave_count = 0
        self.parallel_branch_count = 0

    @property
    def wave_loss(self) -> torch.Tensor | None:
        return self._wave_loss

    def inference_contract(self) -> dict[str, Any]:
        contract = super().inference_contract()
        contract.update(
            {
                "method": "PIVR-LA adaptive recurrent geometry waves",
                "controller_kind": self.controller_kind,
                "learned_point_addressing": True,
                "cached_premerge_visual_reentry": True,
                "recurrent_coverage_memory": True,
                "adaptive_sequential_parallel_decoding": True,
                "open_ended_continue_stop": True,
                "wave_sizes": list(self.wave_sizes),
                "wave_size_is_object_cap": False,
                "one_shared_qwen_weight_set": True,
                "canonical_serialized_output": True,
            }
        )
        return contract

    def _reset_state(self) -> None:
        super()._reset_state()
        self._wave_logits = []
        self._wave_targets = []
        self._wave_loss = None
        self._next_wave_size = 1
        self._active_wave_size = 1
        self._wave_pending_attentions = ()
        self._wave_continue_probabilities = ()
        self._wave_size_predictions = ()
        self.parallel_wave_count = 0
        self.parallel_branch_count = 0

    def finish_forward(self) -> None:
        super().finish_forward()
        self._wave_loss = None

    def _reset_event_auxiliary_state(self) -> None:
        self._wave_logits = []
        self._wave_targets = []
        self._wave_loss = None

    def _target_wave_index(self, remaining: int) -> int:
        target_size = max(1, int(remaining))
        selected = self.wave_sizes[0]
        for size in self.wave_sizes:
            if size <= target_size:
                selected = size
            else:
                break
        return self.wave_sizes.index(selected)

    def _observe_routed_event(
        self, common: torch.Tensor, event: RavelEvent
    ) -> None:
        logits = self.wave_head(common).float()
        if self._training_context and event.has_next_in_group:
            self._wave_logits.append(logits)
            self._wave_targets.append(
                self._target_wave_index(event.remaining_in_group)
            )
        elif not self._training_context:
            index = int(logits.argmax().detach())
            self._next_wave_size = int(self.wave_sizes[index])

    def _finalize_event_auxiliary_state(self) -> None:
        if not self._wave_logits:
            self._wave_loss = None
            return
        logits = torch.stack(self._wave_logits)
        targets = torch.tensor(
            self._wave_targets, dtype=torch.long, device=logits.device
        )
        self._wave_loss = F.cross_entropy(logits, targets)

    def requested_parallel_wave_size(self) -> int:
        """Return the learned next-wave width; 1 keeps native sequential PBD."""

        if self._next_continue_probability < self.wave_continue_threshold:
            return 1
        return int(self._next_wave_size)

    def begin_parallel_wave(self, wave_size: int) -> None:
        wave_size = int(wave_size)
        if wave_size not in self.wave_sizes or wave_size <= 1:
            raise ValueError(f"Invalid parallel wave size: {wave_size}")
        self._active_wave_size = wave_size
        self._wave_pending_attentions = ()
        self._wave_continue_probabilities = ()
        self._wave_size_predictions = ()

    def _model_pre_hook(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if self._training_context or self._active_wave_size <= 1:
            return super()._model_pre_hook(module, args, kwargs)
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None or input_ids.ndim != 2:
            self._context = None
            return None
        if int(input_ids.shape[0]) != self._active_wave_size:
            raise RuntimeError(
                "Adaptive-wave batch mismatch: "
                f"{int(input_ids.shape[0])} != {self._active_wave_size}"
            )
        self._context = self._inference_context(input_ids[:1])
        self._address_loss = None
        self._presence_loss = None
        self._continue_loss = None
        return None

    def _layer_pre_hook(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        if (
            not self._training_context
            and self._active_wave_size > 1
            and self._context is not None
            and self._context.events
            and args
        ):
            updated = self._route_parallel_wave(args[0], self._context)
            return (updated, *args[1:]), kwargs
        return super()._layer_pre_hook(module, args, kwargs)

    def _route_parallel_wave(
        self, hidden_states: torch.Tensor, context: RavelContext
    ) -> torch.Tensor:
        fine = self._fine_tokens
        event = context.events[-1]
        if fine is None or max(event.row) >= int(hidden_states.shape[1]):
            return hidden_states
        if int(hidden_states.shape[0]) != self._active_wave_size:
            raise RuntimeError("Parallel-wave hidden-state batch changed")
        if self._grid_hw is None:
            raise RuntimeError("Adaptive-wave patch grid is unavailable")

        self.route_call_count += 1
        route_dtype = self.key_proj.weight.dtype
        fine = fine.to(device=hidden_states.device, dtype=route_dtype)
        keys = F.normalize(self.key_proj(fine).float(), dim=-1)
        values = self.value_proj(fine)
        token_x, token_y = self._token_centers(hidden_states.device)
        coverage = (
            torch.zeros_like(token_x)
            if self._coverage is None
            else self._coverage.to(token_x)
        )
        updated = hidden_states.clone()
        strengths = self.residual_strength().to(
            hidden_states.device, hidden_states.dtype
        )
        pending: list[torch.Tensor] = []
        continue_probabilities: list[float] = []
        wave_predictions: list[int] = []
        entropies: list[torch.Tensor] = []
        height, width = self._grid_hw

        for branch in range(self._active_wave_size):
            anchor = hidden_states[branch, event.row[0]]
            normalized = self.query_norm(anchor.to(self.query_norm.weight.dtype))
            query = F.normalize(
                self.query_proj(normalized.to(self.query_proj.weight.dtype)).float(),
                dim=-1,
            )
            logits = torch.matmul(keys, query) / self.address_temperature
            logits = logits - self.coverage_penalty * coverage
            probability = logits.softmax(dim=0)
            predicted_point = torch.stack(
                (torch.dot(probability, token_x), torch.dot(probability, token_y))
            )
            routed = torch.matmul(probability.to(route_dtype), values)
            clipped_coverage = coverage.clamp_max(1)
            covered_value = (
                torch.matmul(clipped_coverage.to(route_dtype), values)
                / clipped_coverage.sum().clamp_min(1.0)
            )
            local_values: list[torch.Tensor] = []
            for sigma in self.local_sigmas:
                dx = (token_x - predicted_point[0]) * width
                dy = (token_y - predicted_point[1]) * height
                weights = (
                    -0.5 * (dx.square() + dy.square()) / (sigma**2)
                ).softmax(dim=0)
                local_values.append(torch.matmul(weights.to(route_dtype), values))
            common = torch.cat(
                (
                    query.to(route_dtype),
                    routed,
                    self.point_proj(predicted_point.to(route_dtype)),
                    self.coverage_proj(covered_value),
                    *local_values,
                ),
                dim=-1,
            )
            presence_probability = self.presence_head(common).squeeze(-1).sigmoid()
            continue_probability = self.continue_head(common).squeeze(-1).sigmoid()
            wave_index = int(self.wave_head(common).argmax().detach())
            wave_predictions.append(int(self.wave_sizes[wave_index]))
            continue_probabilities.append(float(continue_probability.detach()))

            lane_positions = event.row[1:-1]
            coordinate_count = {"point": 2, "hbb": 4, "obb": 8}.get(
                event.geometry, len(lane_positions)
            )
            for lane_index, position in enumerate(
                lane_positions[: min(8, coordinate_count)]
            ):
                fused = torch.cat((common, self.lane_embedding[lane_index]), dim=-1)
                residual = self.evidence_up(fused).to(updated.dtype)
                updated[branch, position] = (
                    updated[branch, position]
                    + strengths[lane_index]
                    * presence_probability.to(updated.dtype)
                    * residual
                )
            pending.append(probability.detach())
            coverage = torch.maximum(coverage, probability.detach())
            entropies.append(
                -(probability * probability.clamp_min(1e-8).log()).sum()
            )

        self._wave_pending_attentions = tuple(pending)
        self._wave_continue_probabilities = tuple(continue_probabilities)
        self._wave_size_predictions = tuple(wave_predictions)
        self._last_presence_probability = float(
            self.presence_head(common).squeeze(-1).sigmoid().detach()
        )
        self.last_diagnostics.update(
            {
                "ravel_events": float(self._active_wave_size),
                "ravel_fine_tokens": float(fine.shape[0]),
                "ravel_address_entropy": float(torch.stack(entropies).mean()),
                "ravel_coverage_sum": float(coverage.sum()),
                "ravel_residual_strength": float(strengths.mean()),
                "pivr_active_wave_size": float(self._active_wave_size),
            }
        )
        self._record_runtime_diagnostics()
        return updated

    def observe_generated_wave(self, accepted_branch_indices: Sequence[int]) -> None:
        accepted = tuple(int(value) for value in accepted_branch_indices)
        if not accepted:
            self._next_wave_size = 1
            self._active_wave_size = 1
            return
        self.parallel_wave_count += 1
        self.parallel_branch_count += len(accepted)
        self.accepted_geometry_block_count += len(accepted)
        self.geometry_writeback_count += len(accepted)
        for index in accepted:
            if not 0 <= index < len(self._wave_pending_attentions):
                raise RuntimeError(f"Accepted branch index outside wave: {index}")
            pending = self._wave_pending_attentions[index]
            self._coverage = (
                pending
                if self._coverage is None
                else torch.maximum(self._coverage.to(pending), pending)
            )
        last = accepted[-1]
        self._next_continue_probability = self._wave_continue_probabilities[last]
        self._next_wave_size = self._wave_size_predictions[last]
        self._geometry_route_gate = self._next_continue_probability
        self._active_wave_size = 1
        self._wave_pending_attentions = ()
        self._wave_continue_probabilities = ()
        self._wave_size_predictions = ()
        self._record_runtime_diagnostics()

    def observe_generated_block(self, block_type: str, tokens: Any) -> None:
        self._active_wave_size = 1
        self._wave_pending_attentions = ()
        self._wave_continue_probabilities = ()
        self._wave_size_predictions = ()
        super().observe_generated_block(block_type, tokens)
        if block_type in {"ref", "reference", "ref_object", "empty_box", "im_end"}:
            self._next_wave_size = 1
        self._record_runtime_diagnostics()

    def _record_runtime_diagnostics(self) -> None:
        super()._record_runtime_diagnostics()
        self.last_diagnostics.update(
            {
                "pivr_next_wave_size": float(self._next_wave_size),
                "pivr_active_wave_size": float(self._active_wave_size),
                "pivr_parallel_waves": float(self.parallel_wave_count),
                "pivr_parallel_branches": float(self.parallel_branch_count),
            }
        )

    def install(self, model: nn.Module) -> None:
        super().install(model)
        object.__setattr__(model, "_pivr_generation_controller", self)

    def remove(self) -> None:
        super().remove()
