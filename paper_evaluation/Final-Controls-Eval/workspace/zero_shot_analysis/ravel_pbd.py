"""Recurrent addressed visual-evidence localization for LocateAnything.

RAVEL keeps LocateAnything's tokenizer, Qwen decoder, PBD schedule, and native
geometry grammar unchanged.  It caches MoonViT features immediately before the
released 2x2 patch merger and lets one recurrent Qwen decoder address that fine
feature grid before emitting each point, HBB, OBB, or explicit ``None`` block.

The implementation is deliberately adapter-like:

* MoonViT is evaluated once per image and its fine feature map is cached.
* There is one Qwen decoder; no object slots or decoder replicas are created.
* A soft spatial address retrieves local evidence at several scales.
* A coverage state discourages repeatedly addressing already explained regions.
* The retrieved evidence is injected through a zero-initialized residual, so a
  fresh RAVEL controller is exactly equivalent to its parent LocateAnything.

This module provides the first research gate, not a benchmark claim.  In the
gate, a learned presence probability gates visual write-back and a soft
continuation probability carries the recurrent geometry state between accepted
blocks. Native PBD remains responsible for every output token and EOS choice.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
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


IGNORE_INDEX = -100

ROUTING_CONTROLS = (
    "learned",
    "random_point",
    "global_visual_reentry",
    "cached_feature_crop",
)


@dataclass(frozen=True)
class RavelEvent:
    """One supervised native geometry event in an MTP training sequence."""

    row: tuple[int, ...]
    geometry: str
    coordinates: tuple[int, ...]
    starts_group: bool
    has_next_in_group: bool
    remaining_in_group: int = 0


@dataclass(frozen=True)
class RavelContext:
    events: tuple[RavelEvent, ...]
    image_columns: tuple[int, ...]
    query_length: int


def _optional_token(token_ids: Mapping[str, int], name: str) -> int | None:
    value = token_ids.get(name)
    return None if value is None else int(value)


def _geometry_from_block(
    values: Sequence[int], token_ids: Mapping[str, int]
) -> tuple[str, tuple[int, ...]] | None:
    box_start = int(token_ids["box_start_token_id"])
    box_end = int(token_ids["box_end_token_id"])
    quad_start = _optional_token(token_ids, "quad_start_token_id")
    quad_end = _optional_token(token_ids, "quad_end_token_id")
    none_id = _optional_token(token_ids, "none_token_id")
    coord_start = int(token_ids["coord_start_token_id"])
    coord_end = int(token_ids["coord_end_token_id"])

    if not values:
        return None
    if values[0] == box_start:
        try:
            end = values.index(box_end, 1)
        except ValueError:
            return None
        payload = tuple(values[1:end])
        if none_id is not None and payload == (none_id,):
            return "none", ()
        if len(payload) not in (2, 4):
            return None
        if not all(coord_start <= value <= coord_end for value in payload):
            return None
        coordinates = tuple(value - coord_start for value in payload)
        return ("point" if len(coordinates) == 2 else "hbb"), coordinates
    if quad_start is not None and quad_end is not None and values[0] == quad_start:
        try:
            end = values.index(quad_end, 1)
        except ValueError:
            return None
        payload = tuple(values[1:end])
        if len(payload) != 8:
            return None
        if not all(coord_start <= value <= coord_end for value in payload):
            return None
        return "obb", tuple(value - coord_start for value in payload)
    return None


def ravel_context_from_training(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    token_ids: Mapping[str, int],
    maximum_block_size: int = 10,
) -> RavelContext:
    """Extract point/HBB/OBB/None events from LA's masked PBD supervision."""

    if input_ids.ndim != 2 or labels.ndim != 2 or input_ids.shape != labels.shape:
        raise ValueError("RAVEL expects matching [1, sequence] input ids and labels")
    if int(input_ids.shape[0]) != 1:
        raise ValueError("RAVEL's research gate supports one packed sequence at a time")
    ids = input_ids[0]
    targets = labels[0]
    mask_id = int(token_ids["default_mask_token_id"])
    ref_end = int(token_ids["ref_end_token_id"])
    box_end = int(token_ids["box_end_token_id"])
    quad_end = _optional_token(token_ids, "quad_end_token_id")
    predecessors = {ref_end, box_end}
    if quad_end is not None:
        predecessors.add(quad_end)

    provisional: list[tuple[tuple[int, ...], str, tuple[int, ...], bool]] = []
    sequence_length = int(ids.numel())
    for anchor in range(sequence_length - 1):
        if int(ids[anchor]) not in predecessors or int(ids[anchor + 1]) != mask_id:
            continue
        mask_count = 0
        while (
            mask_count < maximum_block_size - 1
            and anchor + 1 + mask_count < sequence_length
            and int(ids[anchor + 1 + mask_count]) == mask_id
        ):
            mask_count += 1
        if mask_count < 2:
            continue
        # The target block begins one position after the predecessor.  Try the
        # longest valid prefix so ten-token quadrilaterals are not read as HBBs.
        parsed: tuple[str, tuple[int, ...]] | None = None
        consumed = 0
        available = targets[anchor + 1 : anchor + 1 + mask_count + 1]
        raw = [int(value) for value in available.tolist() if int(value) != IGNORE_INDEX]
        for length in range(min(maximum_block_size, len(raw)), 1, -1):
            parsed = _geometry_from_block(raw[:length], token_ids)
            if parsed is not None:
                consumed = length
                break
        if parsed is None:
            continue
        geometry, coordinates = parsed
        row = tuple(range(anchor, anchor + max(consumed, 2)))
        provisional.append((row, geometry, coordinates, int(ids[anchor]) == ref_end))

    events: list[RavelEvent] = []
    for index, (row, geometry, coordinates, starts_group) in enumerate(provisional):
        remaining = 0
        for candidate in provisional[index + 1 :]:
            if candidate[3]:
                break
            remaining += 1
        events.append(
            RavelEvent(
                row=row,
                geometry=geometry,
                coordinates=coordinates,
                starts_group=starts_group,
                has_next_in_group=remaining > 0,
                remaining_in_group=remaining,
            )
        )
    image_index = int(token_ids["image_token_index"])
    image_columns = tuple(
        int(value)
        for value in ids.eq(image_index).nonzero(as_tuple=False).flatten().tolist()
    )
    return RavelContext(tuple(events), image_columns, sequence_length)


class RavelPBDController(nn.Module):
    """Retrieve fine visual evidence and write it into native PBD lanes."""

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
        additional_local_value_count: int = 0,
    ) -> None:
        super().__init__()
        if min(hidden_size, vision_hidden_size, evidence_dim) <= 0:
            raise ValueError("RAVEL dimensions must be positive")
        if address_temperature <= 0:
            raise ValueError("address_temperature must be positive")
        if not local_sigmas or any(value <= 0 for value in local_sigmas):
            raise ValueError("local_sigmas must be positive")
        if routing_control not in ROUTING_CONTROLS:
            raise ValueError(
                f"routing_control must be one of {ROUTING_CONTROLS}, got {routing_control!r}"
            )
        if cached_crop_radius_cells <= 0:
            raise ValueError("cached_crop_radius_cells must be positive")
        if additional_local_value_count < 0:
            raise ValueError("additional_local_value_count must be non-negative")
        self.hidden_size = int(hidden_size)
        self.vision_hidden_size = int(vision_hidden_size)
        self.relay_layer = int(relay_layer)
        self.evidence_dim = int(evidence_dim)
        self.address_temperature = float(address_temperature)
        self.local_sigmas = tuple(float(value) for value in local_sigmas)
        self.coverage_penalty = float(coverage_penalty)
        self.maximum_residual_strength = float(maximum_residual_strength)
        self.isolate_controller_gradients = bool(isolate_controller_gradients)
        self.routing_control = str(routing_control)
        self.cached_crop_radius_cells = float(cached_crop_radius_cells)
        self.additional_local_value_count = int(additional_local_value_count)
        self.token_ids = {str(key): int(value) for key, value in token_ids.items()}

        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.query_proj = nn.Linear(self.hidden_size, self.evidence_dim, bias=False)
        self.key_proj = nn.Linear(self.vision_hidden_size, self.evidence_dim, bias=False)
        self.value_proj = nn.Linear(self.vision_hidden_size, self.evidence_dim, bias=False)
        self.point_proj = nn.Sequential(
            nn.Linear(2, self.evidence_dim), nn.GELU(), nn.Linear(self.evidence_dim, self.evidence_dim)
        )
        self.coverage_proj = nn.Linear(self.evidence_dim, self.evidence_dim, bias=False)
        self.lane_embedding = nn.Parameter(torch.zeros(8, self.evidence_dim))
        fusion_width = self.evidence_dim * (
            5 + len(self.local_sigmas) + self.additional_local_value_count
        )
        self.common_feature_width = fusion_width - self.evidence_dim
        self.evidence_up = nn.Sequential(
            nn.Linear(fusion_width, 2 * self.evidence_dim),
            nn.GELU(),
            nn.LayerNorm(2 * self.evidence_dim),
            nn.Linear(2 * self.evidence_dim, self.hidden_size),
        )
        self.presence_head = nn.Sequential(
            nn.Linear(fusion_width - self.evidence_dim, self.evidence_dim),
            nn.GELU(),
            nn.Linear(self.evidence_dim, 1),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(fusion_width - self.evidence_dim, self.evidence_dim),
            nn.GELU(),
            nn.Linear(self.evidence_dim, 1),
        )
        # Zero output is an exact parent-model identity at initialization.
        nn.init.zeros_(self.evidence_up[-1].weight)
        nn.init.zeros_(self.evidence_up[-1].bias)
        self.residual_strength_logit = nn.Parameter(torch.full((8,), -3.0))

        self._context: RavelContext | None = None
        self._training_context = False
        self._grid_hw: tuple[int, int] | None = None
        self._fine_tokens: torch.Tensor | None = None
        self._coverage: torch.Tensor | None = None
        self._address_loss: torch.Tensor | None = None
        self._presence_loss: torch.Tensor | None = None
        self._continue_loss: torch.Tensor | None = None
        self._pending_attention: torch.Tensor | None = None
        self._geometry_route_gate = 0.0
        self._next_continue_probability = 0.0
        self._last_presence_probability = 0.0
        self._inference_geometry = "hbb"
        self._handles: list[Any] = []
        self.last_diagnostics: dict[str, float] = {}
        self.route_call_count = 0
        self.vision_capture_count = 0
        self.accepted_geometry_block_count = 0
        self.geometry_writeback_count = 0
        self.terminal_event_count = 0

    @property
    def address_loss(self) -> torch.Tensor | None:
        return self._address_loss

    @property
    def presence_loss(self) -> torch.Tensor | None:
        return self._presence_loss

    @property
    def continue_loss(self) -> torch.Tensor | None:
        return self._continue_loss

    def residual_strength(self) -> torch.Tensor:
        return self.maximum_residual_strength * torch.sigmoid(self.residual_strength_logit)

    def _continuation_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return the controller's continuation objective.

        Subclasses may rebalance this objective when ``continue`` and ``stop``
        events are strongly skewed.  The released RAVEL behavior remains the
        ordinary mean binary cross-entropy.
        """

        return F.binary_cross_entropy_with_logits(logits, targets)

    def _reset_event_auxiliary_state(self) -> None:
        """Reset optional per-event objectives implemented by a subclass."""

    def _observe_routed_event(
        self, common: torch.Tensor, event: RavelEvent
    ) -> None:
        """Expose one routed event to optional controller-specific objectives."""

        del common, event

    def _finalize_event_auxiliary_state(self) -> None:
        """Finalize optional objectives after all events in a sequence."""

    def inference_contract(self) -> dict[str, Any]:
        """Return the architectural contract used for benchmark accounting."""

        return {
            "method": "PIVR-LA single-pass (RAVEL)",
            "moonvit_encodings_per_image": 1,
            "qwen_decoder_streams": 1,
            "crop_reencoding": False,
            "decoder_replicas": 0,
            "fixed_object_slots": False,
            "fixed_object_cap": None,
            "native_geometry_grammar": True,
            "native_eos_termination": True,
            "presence_gated_visual_writeback": True,
            "soft_learned_geometry_continuation": True,
            "controller_auxiliary_gradients_into_qwen": not self.isolate_controller_gradients,
            "routing_control": self.routing_control,
            "routing_control_compute_matched": True,
        }

    def set_routing_control(self, routing_control: str) -> None:
        """Select a paired inference control without changing model weights.

        Every control keeps one MoonViT encoding, one Qwen stream, the same
        relay layer, and the same visual projections. Only the spatial measure
        used to read the cached fine-token grid changes.
        """

        if routing_control not in ROUTING_CONTROLS:
            raise ValueError(
                f"routing_control must be one of {ROUTING_CONTROLS}, got {routing_control!r}"
            )
        self.routing_control = str(routing_control)

    def _record_runtime_diagnostics(self) -> None:
        self.last_diagnostics.update(
            {
                "pivr_vision_encodings": float(self.vision_capture_count),
                "pivr_decoder_streams": 1.0,
                "pivr_crop_reencodings": 0.0,
                "pivr_accepted_geometry_blocks": float(
                    self.accepted_geometry_block_count
                ),
                "pivr_geometry_writebacks": float(
                    self.geometry_writeback_count
                ),
                "pivr_terminal_events": float(self.terminal_event_count),
                "pivr_geometry_route_gate": float(self._geometry_route_gate),
                "pivr_next_continue_probability": float(
                    self._next_continue_probability
                ),
                "pivr_last_presence_probability": float(
                    self._last_presence_probability
                ),
            }
        )

    def _set_grid(self, image_grid_hws: Any) -> None:
        values = torch.as_tensor(image_grid_hws, dtype=torch.long).reshape(-1, 2)
        if int(values.shape[0]) != 1:
            raise ValueError("RAVEL currently supports one image per record")
        height, width = int(values[0, 0]), int(values[0, 1])
        if min(height, width) <= 0:
            raise ValueError(f"Invalid MoonViT grid {(height, width)}")
        self._grid_hw = (height, width)

    def _reset_state(self) -> None:
        self._fine_tokens = None
        self._coverage = None
        self._address_loss = None
        self._presence_loss = None
        self._continue_loss = None
        self._pending_attention = None
        self._geometry_route_gate = 0.0
        self._next_continue_probability = 0.0
        self._last_presence_probability = 0.0
        self.last_diagnostics = {}
        self.route_call_count = 0
        self.vision_capture_count = 0
        self.accepted_geometry_block_count = 0
        self.geometry_writeback_count = 0
        self.terminal_event_count = 0

    def set_training_context(self, context: RavelContext, image_grid_hws: Any) -> None:
        self._context = context
        self._training_context = True
        self._set_grid(image_grid_hws)
        self._reset_state()

    def set_inference_image_layout(
        self, _input_ids: torch.Tensor, image_grid_hws: Any
    ) -> None:
        self._context = None
        self._training_context = False
        self._set_grid(image_grid_hws)
        self._reset_state()

    def set_inference_task(self, question: str, tokenizer: Any | None = None) -> None:
        """Select only the native lane width explicitly requested by the task."""

        normalized = " ".join(str(question).lower().split())
        if normalized.startswith("point to:"):
            self._inference_geometry = "point"
        elif "oriented box format" in normalized:
            self._inference_geometry = "obb"
        else:
            self._inference_geometry = "hbb"

    def finish_forward(self) -> None:
        self._context = None
        self._training_context = False
        self._address_loss = None
        self._presence_loss = None
        self._continue_loss = None
        self._pending_attention = None

    def _vision_hook(self, _module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        self.vision_capture_count += 1
        if self.vision_capture_count != 1:
            raise RuntimeError(
                "PIVR-LA single-pass contract violated: MoonViT encoder was "
                f"called {self.vision_capture_count} times for one query"
            )
        fine = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(fine, torch.Tensor) or fine.ndim != 2:
            raise RuntimeError("MoonViT encoder did not return [tokens, hidden]")
        if self._grid_hw is None:
            raise RuntimeError("RAVEL image grid was not configured")
        expected = self._grid_hw[0] * self._grid_hw[1]
        if int(fine.shape[0]) != expected:
            raise RuntimeError(f"Fine-token mismatch: {fine.shape[0]} != {expected}")
        self._fine_tokens = fine.detach()
        self._record_runtime_diagnostics()

    def _inference_context(self, input_ids: torch.Tensor | None) -> RavelContext | None:
        if self._geometry_route_gate <= 0.0:
            return None
        if input_ids is None or input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            return None
        ids = input_ids[0]
        mask_id = int(self.token_ids["default_mask_token_id"])
        ref_end = int(self.token_ids["ref_end_token_id"])
        box_end = int(self.token_ids["box_end_token_id"])
        predecessors = {ref_end, box_end}
        quad_end = _optional_token(self.token_ids, "quad_end_token_id")
        if quad_end is not None:
            predecessors.add(quad_end)
        candidates: list[RavelEvent] = []
        lower = max(0, int(ids.numel()) - 20)
        for anchor in range(lower, int(ids.numel()) - 1):
            if int(ids[anchor]) not in predecessors:
                continue
            masks = 0
            while anchor + 1 + masks < int(ids.numel()) and int(ids[anchor + 1 + masks]) == mask_id:
                masks += 1
            if masks >= 5:
                requested = {"point": 2, "hbb": 4, "obb": 8}[self._inference_geometry]
                lane_count = min(requested, masks - 1)
                row = tuple(range(anchor, anchor + lane_count + 2))
                candidates.append(
                    RavelEvent(
                        row,
                        self._inference_geometry,
                        (),
                        int(ids[anchor]) == ref_end,
                        False,
                        0,
                    )
                )
        if not candidates:
            return None
        image_index = int(self.token_ids["image_token_index"])
        columns = tuple(int(value) for value in ids.eq(image_index).nonzero(as_tuple=False).flatten().tolist())
        return RavelContext((candidates[-1],), columns, int(ids.numel()))

    def _model_pre_hook(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if self._training_context:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        self._context = self._inference_context(input_ids)
        self._address_loss = None
        self._presence_loss = None
        self._continue_loss = None
        return None

    def _token_centers(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._grid_hw is None:
            raise RuntimeError("RAVEL patch grid is unavailable")
        height, width = self._grid_hw
        yy, xx = torch.meshgrid(
            (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height,
            (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width,
            indexing="ij",
        )
        return xx.reshape(-1), yy.reshape(-1)

    def _support(self, event: RavelEvent, device: torch.device) -> torch.Tensor | None:
        if not event.coordinates or self._grid_hw is None:
            return None
        token_x, token_y = self._token_centers(device)
        values = torch.tensor(event.coordinates, device=device, dtype=torch.float32) / 1000.0
        if event.geometry == "point":
            distance = (token_x - values[0]).square() + (token_y - values[1]).square()
            support = torch.zeros_like(token_x)
            support[int(distance.argmin())] = 1.0
            return support
        if event.geometry == "hbb":
            left, right = torch.sort(values[[0, 2]])[0]
            top, bottom = torch.sort(values[[1, 3]])[0]
            if self._grid_hw is None:
                return None
            height, width = self._grid_hw
            # A tiny remote-sensing object is often smaller than one visual
            # cell, so no cell center lies inside its HBB. Supervise every
            # cell whose footprint overlaps the box instead. This guarantees
            # a non-empty, differentiable address target without enlarging or
            # changing the geometry emitted by LocateAnything.
            half_width = 0.5 / float(width)
            half_height = 0.5 / float(height)
            support = (
                (token_x + half_width >= left)
                & (token_x - half_width <= right)
                & (token_y + half_height >= top)
                & (token_y - half_height <= bottom)
            ).float()
            if float(support.sum()) > 0:
                return support
            center_x = (left + right) * 0.5
            center_y = (top + bottom) * 0.5
            distance = (token_x - center_x).square() + (token_y - center_y).square()
            support[int(distance.argmin())] = 1.0
            return support
        if event.geometry == "obb":
            polygon = values.view(4, 2)
            points = torch.stack((token_x, token_y), dim=-1)
            edges = torch.roll(polygon, shifts=-1, dims=0) - polygon
            relative = points[:, None, :] - polygon[None, :, :]
            cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
            support = ((cross >= -1e-6).all(dim=1) | (cross <= 1e-6).all(dim=1)).float()
            if float(support.sum()) > 0:
                return support
        center_x = values[0::2].mean()
        center_y = values[1::2].mean()
        distance = (token_x - center_x).square() + (token_y - center_y).square()
        support = torch.zeros_like(token_x)
        support[int(distance.argmin())] = 1.0
        return support

    def _controlled_probability(
        self,
        learned_probability: torch.Tensor,
        token_x: torch.Tensor,
        token_y: torch.Tensor,
    ) -> torch.Tensor:
        """Return the preregistered compute-matched visual-routing control."""

        if self._training_context or self.routing_control == "learned":
            return learned_probability
        if self.routing_control == "random_point":
            probability = torch.zeros_like(learned_probability)
            random_index = torch.randint(
                int(learned_probability.numel()),
                (),
                device=learned_probability.device,
            )
            probability[random_index] = 1.0
            return probability
        if self.routing_control == "global_visual_reentry":
            return torch.full_like(
                learned_probability, 1.0 / float(learned_probability.numel())
            )
        if self.routing_control == "cached_feature_crop":
            if self._grid_hw is None:
                raise RuntimeError("RAVEL patch grid is unavailable")
            center = int(learned_probability.argmax())
            height, width = self._grid_hw
            dx = (token_x - token_x[center]) * width
            dy = (token_y - token_y[center]) * height
            mask = (
                dx.abs().le(self.cached_crop_radius_cells)
                & dy.abs().le(self.cached_crop_radius_cells)
            ).to(learned_probability.dtype)
            return mask / mask.sum().clamp_min(1.0)
        raise AssertionError(f"Unhandled routing control: {self.routing_control}")

    def _additional_local_values(
        self,
        *,
        probability: torch.Tensor,
        predicted_point: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Return optional point-indexed evidence supplied by a subclass."""

        del probability, predicted_point, hidden_states
        return ()

    def _route(self, hidden_states: torch.Tensor, context: RavelContext) -> torch.Tensor:
        fine = self._fine_tokens
        events = tuple(event for event in context.events if event.row and max(event.row) < hidden_states.shape[1])
        if fine is None or not events:
            return hidden_states
        if self._grid_hw is None or self._grid_hw[0] * self._grid_hw[1] != int(fine.shape[0]):
            raise RuntimeError("RAVEL fine features do not match the patch grid")
        self.route_call_count += 1
        self._reset_event_auxiliary_state()
        route_dtype = self.key_proj.weight.dtype
        fine = fine.to(device=hidden_states.device, dtype=route_dtype)
        keys = F.normalize(self.key_proj(fine).float(), dim=-1)
        values = self.value_proj(fine)
        token_x, token_y = self._token_centers(hidden_states.device)
        coverage = torch.zeros_like(token_x) if self._training_context or self._coverage is None else self._coverage.to(token_x)
        updated = hidden_states.clone()
        address_losses: list[torch.Tensor] = []
        presence_logits: list[torch.Tensor] = []
        presence_targets: list[float] = []
        continue_logits: list[torch.Tensor] = []
        continue_targets: list[float] = []
        entropies: list[torch.Tensor] = []
        inside: list[torch.Tensor] = []
        strengths = self.residual_strength().to(hidden_states.device, hidden_states.dtype)

        for event in events:
            if event.starts_group:
                coverage = torch.zeros_like(coverage)
            anchor = hidden_states[0, event.row[0]]
            if self._training_context and self.isolate_controller_gradients:
                # Auxiliary address/presence/continuation targets should teach
                # the compact controller, not rewrite Qwen's semantic and EOS
                # states. Native LM loss still trains LoRA through the identity
                # path and through every generated geometry token.
                anchor = anchor.detach()
            normalized_anchor = self.query_norm(
                anchor.to(self.query_norm.weight.dtype)
            )
            query = self.query_proj(normalized_anchor.to(self.query_proj.weight.dtype))
            query = F.normalize(query.float(), dim=-1)
            logits = torch.matmul(keys, query) / self.address_temperature
            logits = logits - self.coverage_penalty * coverage
            learned_probability = logits.softmax(dim=0)
            probability = self._controlled_probability(
                learned_probability, token_x, token_y
            )
            predicted_point = torch.stack((torch.dot(probability, token_x), torch.dot(probability, token_y)))
            routed = torch.matmul(probability.to(route_dtype), values)
            covered_value = torch.matmul(coverage.clamp_max(1).to(route_dtype), values) / coverage.clamp_max(1).sum().clamp_min(1.0)
            covered_feature = self.coverage_proj(covered_value)
            local_values: list[torch.Tensor] = []
            height, width = self._grid_hw
            for sigma in self.local_sigmas:
                dx = (token_x - predicted_point[0]) * width
                dy = (token_y - predicted_point[1]) * height
                weights = (-0.5 * (dx.square() + dy.square()) / (sigma**2)).softmax(dim=0)
                local_values.append(torch.matmul(weights.to(route_dtype), values))
            additional_local_values = self._additional_local_values(
                probability=probability,
                predicted_point=predicted_point,
                hidden_states=hidden_states,
            )
            if len(additional_local_values) != self.additional_local_value_count:
                raise RuntimeError(
                    "RAVEL additional evidence count changed: "
                    f"{len(additional_local_values)} != "
                    f"{self.additional_local_value_count}"
                )
            point_feature = self.point_proj(predicted_point.to(route_dtype))
            query_feature = query.to(route_dtype)
            common = torch.cat(
                (
                    query_feature,
                    routed,
                    point_feature,
                    covered_feature,
                    *local_values,
                    *additional_local_values,
                ),
                dim=-1,
            )
            self._observe_routed_event(common, event)
            presence_logit = self.presence_head(common).squeeze(-1).float()
            continue_logit = self.continue_head(common).squeeze(-1).float()
            presence_logits.append(presence_logit)
            presence_targets.append(0.0 if event.geometry == "none" else 1.0)
            continue_logits.append(continue_logit)
            continue_targets.append(float(event.has_next_in_group))
            presence_probability = presence_logit.sigmoid()
            route_gate = (
                presence_probability
                if self._training_context
                else presence_probability * float(self._geometry_route_gate)
            )
            if not self._training_context:
                self._last_presence_probability = float(
                    presence_probability.detach()
                )
                self._next_continue_probability = float(
                    continue_logit.sigmoid().detach()
                )

            lane_positions = event.row[1:-1]
            if event.geometry == "none":
                coordinate_count = 0
            elif event.coordinates:
                coordinate_count = len(event.coordinates)
            else:
                # Inference events are created before native PBD emits the
                # coordinates, so they intentionally have no GT coordinates.
                # Route the task's native lane width rather than silently
                # disabling visual write-back.
                coordinate_count = {
                    "point": 2,
                    "hbb": 4,
                    "obb": 8,
                }.get(event.geometry, len(lane_positions))
            wrote_geometry = False
            for lane_index, position in enumerate(lane_positions[: min(8, coordinate_count)]):
                lane = self.lane_embedding[lane_index]
                fused = torch.cat((common, lane), dim=-1)
                residual = self.evidence_up(fused).to(updated.dtype)
                updated[0, position] = (
                    updated[0, position]
                    + strengths[lane_index] * route_gate.to(updated.dtype) * residual
                )
                wrote_geometry = True
            if wrote_geometry and not self._training_context:
                self.geometry_writeback_count += 1

            support = self._support(event, hidden_states.device)
            if support is not None:
                mass = (probability * support).sum()
                address_losses.append(-mass.clamp_min(1e-8).log())
                inside.append((support[int(probability.argmax())] > 0).float())
                coverage = torch.maximum(coverage, support)
            elif not self._training_context:
                self._pending_attention = probability.detach()
            entropies.append(-(probability * probability.clamp_min(1e-8).log()).sum())

        self._coverage = coverage.detach()
        self._address_loss = torch.stack(address_losses).mean() if address_losses else None
        self._presence_loss = F.binary_cross_entropy_with_logits(
            torch.stack(presence_logits),
            torch.tensor(presence_targets, device=hidden_states.device),
        ) if presence_logits else None
        self._continue_loss = self._continuation_loss(
            torch.stack(continue_logits),
            torch.tensor(continue_targets, device=hidden_states.device),
        ) if continue_logits else None
        self._finalize_event_auxiliary_state()
        self.last_diagnostics = {
            "ravel_events": float(len(events)),
            "ravel_fine_tokens": float(fine.shape[0]),
            "ravel_address_loss": 0.0 if self._address_loss is None else float(self._address_loss.detach()),
            "ravel_address_inside_rate": 0.0 if not inside else float(torch.stack(inside).mean().detach()),
            "ravel_address_entropy": 0.0 if not entropies else float(torch.stack(entropies).mean().detach()),
            "ravel_presence_loss": 0.0 if self._presence_loss is None else float(self._presence_loss.detach()),
            "ravel_continue_loss": 0.0 if self._continue_loss is None else float(self._continue_loss.detach()),
            "ravel_coverage_sum": float(coverage.sum().detach()),
            "ravel_residual_strength": float(strengths.mean().detach()),
        }
        self._record_runtime_diagnostics()
        return updated

    def _layer_pre_hook(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        if self._context is None or not self._context.events or not args:
            return None
        updated = self._route(args[0], self._context)
        return (updated, *args[1:]), kwargs

    def observe_generated_block(self, block_type: str, _tokens: Any) -> None:
        """Commit the last soft address after native PBD accepts a block."""

        if block_type in {"ref", "reference", "ref_object"}:
            self._coverage = None
            self._pending_attention = None
            self._geometry_route_gate = 1.0
            self._record_runtime_diagnostics()
            return
        if block_type in {"coord_box", "point_box", "empty_box"}:
            self.accepted_geometry_block_count += 1
            self._geometry_route_gate = (
                0.0
                if block_type == "empty_box"
                else float(self._next_continue_probability)
            )
        elif block_type == "im_end":
            self.terminal_event_count += 1
            self._geometry_route_gate = 0.0
        if self._pending_attention is not None:
            pending = self._pending_attention
            if self._coverage is None:
                self._coverage = pending
            else:
                self._coverage = torch.maximum(self._coverage.to(pending), pending)
            self._pending_attention = None
        self._record_runtime_diagnostics()

    def install(self, model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("RAVEL hooks are already installed")
        causal = base_causal_lm(model.language_model)
        layers = causal.model.layers
        if not 0 <= self.relay_layer < len(layers):
            raise ValueError(f"relay_layer={self.relay_layer} outside 0..{len(layers)-1}")
        self._handles.append(model.vision_model.encoder.register_forward_hook(self._vision_hook))
        self._handles.append(causal.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True))
        self._handles.append(layers[self.relay_layer].register_forward_pre_hook(self._layer_pre_hook, with_kwargs=True))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def load_ravel_adapter(model: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load RAVEL controller tensors and their matched Qwen LoRA weights."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    configure_lora_only(model, int(config["lora_rank"]))
    token_ids = {**model.token_ids, "image_token_index": int(model.image_token_index)}
    tokenizer_ids = config.get("extra_token_ids", {})
    token_ids.update({str(key): int(value) for key, value in tokenizer_ids.items()})
    controller_kind = str(config.get("controller_kind", "ravel_pre_layer"))
    controller_class: type[RavelPBDController] = RavelPBDController
    if controller_kind == "native_grammar_gated_history_isolated":
        from .grammar_gated_ravel_pbd import GrammarGatedRavelPBDController

        controller_class = GrammarGatedRavelPBDController
    elif controller_kind == "native_grammar_gated_exhaustion":
        variant = os.environ.get("LOCANY_PIVR_CONTROLLER_VARIANT", "checkpoint").strip()
        if variant in {"", "checkpoint", "boundary_handoff"}:
            from .exhaustion_gated_ravel_pbd import (
                ExhaustionGatedRavelPBDController,
            )

            controller_class = ExhaustionGatedRavelPBDController
        elif variant == "cycle_guarded":
            from .cycle_guarded_ravel_pbd import (
                CycleGuardedRavelPBDController,
            )

            controller_class = CycleGuardedRavelPBDController
        elif variant == "typed_lattice":
            from .evidence_typed_ravel_pbd import (
                EvidenceTypedRavelPBDController,
            )

            controller_class = EvidenceTypedRavelPBDController
        else:
            raise ValueError(f"Unsupported PIVR controller variant: {variant}")
    elif controller_kind == "opivr_open_ended_pixel_reentry":
        from .opivr_pbd import OpenEndedPointIndexedVisualReentryPBD

        controller_class = OpenEndedPointIndexedVisualReentryPBD
    elif controller_kind == "adaptive_wave_pivr":
        from .adaptive_wave_pivr import AdaptiveWavePIVRController

        controller_class = AdaptiveWavePIVRController
    elif controller_kind == "single_encode_point_address":
        from .single_encode_point_address_pbd import SingleEncodePointAddressPBD

        controller_class = SingleEncodePointAddressPBD
    elif controller_kind == "highres_point_evidence":
        from .highres_point_evidence_pbd import HighResolutionPointEvidencePBD

        controller_class = HighResolutionPointEvidencePBD
    elif controller_kind == "highres_point_presence_gate":
        from .highres_point_presence_gate_pbd import (
            HighResolutionPointPresenceGatePBD,
        )

        controller_class = HighResolutionPointPresenceGatePBD
    elif controller_kind == "provisional_box_multiscale_reentry":
        from .provisional_box_multiscale_reentry_pbd import (
            ProvisionalBoxMultiscaleReentryPBD,
        )

        controller_class = ProvisionalBoxMultiscaleReentryPBD
    elif controller_kind != "ravel_pre_layer":
        raise ValueError(f"Unsupported PIVR controller kind: {controller_kind}")
    controller = controller_class(
        hidden_size=int(model.config.text_config.hidden_size),
        vision_hidden_size=int(model.config.vision_config.hidden_size),
        relay_layer=int(config["relay_layer"]),
        token_ids=token_ids,
        evidence_dim=int(config.get("evidence_dim", 128)),
        address_temperature=float(config.get("address_temperature", 0.10)),
        local_sigmas=tuple(config.get("local_sigmas", (0.75, 1.5, 3.0))),
        coverage_penalty=float(config.get("coverage_penalty", 2.0)),
        maximum_residual_strength=float(config.get("maximum_residual_strength", 1.0)),
        isolate_controller_gradients=bool(
            config.get("isolate_controller_gradients", True)
        ),
        routing_control=str(config.get("routing_control", "learned")),
        cached_crop_radius_cells=float(
            config.get("cached_crop_radius_cells", 1.5)
        ),
        **(
            {
                "continuation_temperature": float(
                    config.get("continuation_temperature", 1.0)
                ),
                "coordinate_trust_topk": int(
                    config.get("coordinate_trust_topk", 5)
                ),
            }
            if controller_kind == "native_grammar_gated_exhaustion"
            else {}
        ),
        **(
            {
                "pixel_feature_dim": int(config.get("pixel_feature_dim", 32)),
                "pixel_subpatch_grid": int(
                    config.get("pixel_subpatch_grid", 4)
                ),
                "pixel_local_sigmas": tuple(
                    config.get("pixel_local_sigmas", (0.75, 1.5, 3.0))
                ),
            }
            if controller_kind == "opivr_open_ended_pixel_reentry"
            else {}
        ),
        **(
            {
                "wave_sizes": tuple(config.get("wave_sizes", (1, 2, 4))),
                "wave_continue_threshold": float(
                    config.get("wave_continue_threshold", 0.5)
                ),
            }
            if controller_kind == "adaptive_wave_pivr"
            else {}
        ),
        **(
            {
                "scale_anchors": tuple(
                    config.get("scale_anchors", (0.02, 0.05, 0.12, 0.30, 0.60))
                ),
            }
            if controller_kind
            in {
                "single_encode_point_address",
                "highres_point_evidence",
                "highres_point_presence_gate",
                "provisional_box_multiscale_reentry",
            }
            else {}
        ),
        **(
            {
                "pixel_feature_dim": int(config.get("pixel_feature_dim", 32)),
                "pixel_subpatch_grid": int(config.get("pixel_subpatch_grid", 4)),
                "pixel_roi_grid": int(config.get("pixel_roi_grid", 8)),
                "extent_residual_octaves": float(
                    config.get("extent_residual_octaves", 1.0)
                ),
            }
            if controller_kind
            in {
                "highres_point_evidence",
                "highres_point_presence_gate",
                "provisional_box_multiscale_reentry",
            }
            else {}
        ),
        **(
            {
                "maximum_presence_bias": float(
                    config.get("maximum_presence_bias", 4.0)
                ),
                "presence_temperature": float(
                    config.get("presence_temperature", 1.0)
                ),
            }
            if controller_kind
            in {"highres_point_presence_gate", "provisional_box_multiscale_reentry"}
            else {}
        ),
        **(
            {
                "proposal_scale_multipliers": tuple(
                    config.get("proposal_scale_multipliers", (0.75, 1.0, 1.5, 2.0))
                ),
                "proposal_roi_grid": int(config.get("proposal_roi_grid", 4)),
            }
            if controller_kind == "provisional_box_multiscale_reentry"
            else {}
        ),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {
        "checkpoint": checkpoint,
        "controller": controller,
        "inference_contract": controller.inference_contract(),
    }


def load_pivr_la_single_pass_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Canonical PIVR-LA name for the backward-compatible RAVEL loader."""

    return load_ravel_adapter(model, checkpoint_path)
