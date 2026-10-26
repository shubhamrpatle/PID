"""Single-encode point-addressed visual re-entry for LocateAnything.

One MoonViT pass supplies two caches through LocateAnything's released frozen
projector: the native non-overlapping stride-2 grid and an overlapping stride-1
dense grid.  An explicit point in the user prompt addresses those caches.  A
compact controller injects retrieved evidence only into the current native PBD
geometry block; it neither appends dense tokens to Qwen nor creates object slots,
decoder copies, crop images, or a second vision pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .pointrelay_pbd import base_causal_lm
from .ravel_pbd import RavelContext, RavelEvent


IGNORE_INDEX = -100


@dataclass(frozen=True)
class PointAddress:
    x: int
    y: int


def point_address_from_tokens(
    input_ids: torch.Tensor,
    *,
    token_ids: Mapping[str, int],
    labels: torch.Tensor | None = None,
) -> PointAddress | None:
    """Read the explicit two-coordinate point from the actual user prompt."""

    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("Point-address parsing expects [1, sequence] tokens")
    ids = input_ids[0]
    limit = int(ids.numel())
    if labels is not None:
        if labels.shape != input_ids.shape:
            raise ValueError("Labels and input ids must have the same shape")
        supervised = labels[0].ne(IGNORE_INDEX).nonzero(as_tuple=False).flatten()
        if supervised.numel():
            limit = int(supervised[0])
    box_start = int(token_ids["box_start_token_id"])
    box_end = int(token_ids["box_end_token_id"])
    coord_start = int(token_ids["coord_start_token_id"])
    coord_end = int(token_ids["coord_end_token_id"])
    matches: list[PointAddress] = []
    for index in range(max(0, limit - 3)):
        values = [int(value) for value in ids[index : index + 4].tolist()]
        if (
            values[0] == box_start
            and coord_start <= values[1] <= coord_end
            and coord_start <= values[2] <= coord_end
            and values[3] == box_end
        ):
            matches.append(PointAddress(values[1] - coord_start, values[2] - coord_start))
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"Expected one prompt point, found {len(matches)}")
    return matches[0]


class SingleEncodePointAddressPBD(nn.Module):
    """Point-conditioned global/dense cache reader for native PBD geometry."""

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
        coverage_penalty: float = 0.0,
        maximum_residual_strength: float = 1.0,
        isolate_controller_gradients: bool = True,
        scale_anchors: Sequence[float] = (0.02, 0.05, 0.12, 0.30, 0.60),
        **_unused: Any,
    ) -> None:
        super().__init__()
        del vision_hidden_size, local_sigmas, coverage_penalty
        if hidden_size <= 0 or evidence_dim <= 0:
            raise ValueError("Controller dimensions must be positive")
        if address_temperature <= 0:
            raise ValueError("address_temperature must be positive")
        anchors = tuple(float(value) for value in scale_anchors)
        if not anchors or any(not 0 < value <= 1 for value in anchors):
            raise ValueError("scale_anchors must lie in (0, 1]")
        self.hidden_size = int(hidden_size)
        self.relay_layer = int(relay_layer)
        self.evidence_dim = int(evidence_dim)
        self.address_temperature = float(address_temperature)
        self.maximum_residual_strength = float(maximum_residual_strength)
        self.isolate_controller_gradients = bool(isolate_controller_gradients)
        self.token_ids = {str(key): int(value) for key, value in token_ids.items()}
        self.register_buffer("scale_anchors", torch.tensor(anchors, dtype=torch.float32))

        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.query_proj = nn.Linear(self.hidden_size, self.evidence_dim, bias=False)
        self.global_key = nn.Linear(self.hidden_size, self.evidence_dim, bias=False)
        self.global_value = nn.Linear(self.hidden_size, self.evidence_dim, bias=False)
        self.dense_key = nn.Linear(self.hidden_size, self.evidence_dim, bias=False)
        self.dense_value = nn.Linear(self.hidden_size, self.evidence_dim, bias=False)
        self.point_proj = nn.Sequential(
            nn.Linear(2, self.evidence_dim),
            nn.GELU(),
            nn.Linear(self.evidence_dim, self.evidence_dim),
        )
        scale_input = self.evidence_dim * 2
        self.width_head = nn.Linear(scale_input, len(anchors))
        self.height_head = nn.Linear(scale_input, len(anchors))
        self.scale_proj = nn.Sequential(
            nn.Linear(2, self.evidence_dim),
            nn.GELU(),
            nn.Linear(self.evidence_dim, self.evidence_dim),
        )
        self.lane_embedding = nn.Parameter(torch.zeros(8, self.evidence_dim))
        fusion_width = self.evidence_dim * 6
        self.presence_head = nn.Sequential(
            nn.Linear(fusion_width - self.evidence_dim, self.evidence_dim),
            nn.GELU(),
            nn.Linear(self.evidence_dim, 1),
        )
        self.evidence_up = nn.Sequential(
            nn.Linear(fusion_width, 2 * self.evidence_dim),
            nn.GELU(),
            nn.LayerNorm(2 * self.evidence_dim),
            nn.Linear(2 * self.evidence_dim, self.hidden_size),
        )
        nn.init.zeros_(self.evidence_up[-1].weight)
        nn.init.zeros_(self.evidence_up[-1].bias)
        self.residual_strength_logit = nn.Parameter(torch.full((8,), -3.0))

        self._context: RavelContext | None = None
        self._training_context = False
        self._prompt_point: PointAddress | None = None
        self._grid_hw: tuple[int, int] | None = None
        self._dense_hw: tuple[int, int] | None = None
        self._global_projected: torch.Tensor | None = None
        self._dense_projected: torch.Tensor | None = None
        self._projector: nn.Module | None = None
        self._projecting_dense = False
        self._address_loss: torch.Tensor | None = None
        self._presence_loss: torch.Tensor | None = None
        self._scale_loss: torch.Tensor | None = None
        self._handles: list[Any] = []
        self._inference_geometry = "hbb"
        self._geometry_route_gate = 0.0
        self.vision_capture_count = 0
        self.route_call_count = 0
        self.last_diagnostics: dict[str, float] = {}

    @property
    def address_loss(self) -> torch.Tensor | None:
        return self._address_loss

    @property
    def presence_loss(self) -> torch.Tensor | None:
        return self._presence_loss

    @property
    def continue_loss(self) -> None:
        return None

    @property
    def wave_loss(self) -> None:
        return None

    @property
    def scale_loss(self) -> torch.Tensor | None:
        return self._scale_loss

    def requires_address_supervision(self, context: RavelContext) -> bool:
        return self._prompt_point is not None and any(
            event.geometry == "hbb" for event in context.events
        )

    def residual_strength(self) -> torch.Tensor:
        return self.maximum_residual_strength * torch.sigmoid(self.residual_strength_logit)

    def inference_contract(self) -> dict[str, Any]:
        return {
            "method": "single-encode point-addressed PBD",
            "moonvit_encodings_per_image": 1,
            "projected_visual_caches": ["native_stride2", "overlapping_stride1"],
            "shared_released_projector": True,
            "dense_tokens_appended_to_qwen": False,
            "crop_reencoding": False,
            "qwen_decoder_streams": 1,
            "decoder_replicas": 0,
            "fixed_object_slots": False,
            "native_geometry_grammar": True,
            "native_eos_termination": True,
            "explicit_point_address": True,
            "learned_anisotropic_scale_selection": True,
            "one_geometry_or_none_per_address": True,
        }

    def _set_grid(self, image_grid_hws: Any) -> None:
        values = torch.as_tensor(image_grid_hws, dtype=torch.long).reshape(-1, 2)
        if int(values.shape[0]) != 1:
            raise ValueError("Single-encode controller requires one image per record")
        height, width = int(values[0, 0]), int(values[0, 1])
        if min(height, width) < 2 or height % 2 or width % 2:
            raise ValueError(f"MoonViT grid must be even and >=2, got {(height, width)}")
        self._grid_hw = (height, width)
        self._dense_hw = (height - 1, width - 1)

    def _reset_state(self) -> None:
        self._global_projected = None
        self._dense_projected = None
        self._address_loss = None
        self._presence_loss = None
        self._scale_loss = None
        self.vision_capture_count = 0
        self.route_call_count = 0
        self.last_diagnostics = {}

    def set_addressed_training_context(
        self,
        context: RavelContext,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        image_grid_hws: Any,
    ) -> None:
        self._context = context
        self._training_context = True
        self._set_grid(image_grid_hws)
        self._reset_state()
        self._prompt_point = point_address_from_tokens(
            input_ids, labels=labels, token_ids=self.token_ids
        )
        has_hbb_refinement = any(event.geometry == "hbb" for event in context.events)
        if has_hbb_refinement and self._prompt_point is None:
            raise RuntimeError("HBB/None refinement record has no explicit prompt point")

    def set_training_context(self, context: RavelContext, image_grid_hws: Any) -> None:
        self._context = context
        self._training_context = True
        self._set_grid(image_grid_hws)
        self._reset_state()
        self._prompt_point = None

    def set_inference_image_layout(self, input_ids: torch.Tensor, image_grid_hws: Any) -> None:
        self._context = None
        self._training_context = False
        self._set_grid(image_grid_hws)
        self._reset_state()
        self._prompt_point = point_address_from_tokens(input_ids, token_ids=self.token_ids)

    def set_inference_task(self, question: str, tokenizer: Any | None = None) -> None:
        del tokenizer
        normalized = " ".join(str(question).lower().split())
        self._inference_geometry = "point" if normalized.startswith("point to:") else "hbb"

    def finish_forward(self) -> None:
        self._context = None
        self._training_context = False
        self._address_loss = None
        self._presence_loss = None
        self._scale_loss = None
        self._global_projected = None
        self._dense_projected = None

    def _vision_hook(self, _module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        self.vision_capture_count += 1
        if self.vision_capture_count != 1:
            raise RuntimeError("Single-encode contract violated: MoonViT ran more than once")
        if self._grid_hw is None or self._projector is None:
            raise RuntimeError("Grid/projector was not configured before MoonViT")
        fine = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(fine, torch.Tensor) or fine.ndim != 2:
            raise RuntimeError("MoonViT encoder did not return [tokens, hidden]")
        height, width = self._grid_hw
        if int(fine.shape[0]) != height * width:
            raise RuntimeError(f"Fine token mismatch: {fine.shape[0]} != {height * width}")
        if self._prompt_point is None:
            # Global point-discovery records train native LA LoRA only.  They
            # do not need the refinement-only dense side cache.
            return
        grid = fine.detach().view(height, width, -1)
        dense_windows = torch.cat(
            (grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]), dim=-1
        ).reshape((height - 1) * (width - 1), -1)
        self._projecting_dense = True
        try:
            with torch.no_grad():
                self._dense_projected = self._projector(dense_windows).detach()
        finally:
            self._projecting_dense = False

    def _projector_hook(self, _module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        if self._projecting_dense:
            return
        projected = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(projected, torch.Tensor) or projected.ndim != 2:
            raise RuntimeError("LocateAnything projector did not return [tokens, hidden]")
        self._global_projected = projected.detach()

    def _inference_context(self, input_ids: torch.Tensor | None) -> RavelContext | None:
        if self._prompt_point is None or self._geometry_route_gate <= 0:
            return None
        if input_ids is None or input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            return None
        ids = input_ids[0]
        mask_id = int(self.token_ids["default_mask_token_id"])
        predecessors = {
            int(self.token_ids["ref_end_token_id"]),
            int(self.token_ids["box_end_token_id"]),
        }
        rows: list[tuple[int, ...]] = []
        for anchor in range(max(0, int(ids.numel()) - 20), int(ids.numel()) - 1):
            if int(ids[anchor]) not in predecessors:
                continue
            count = 0
            while anchor + 1 + count < int(ids.numel()) and int(ids[anchor + 1 + count]) == mask_id:
                count += 1
            if count >= 5:
                rows.append(tuple(range(anchor, anchor + 6)))
        if not rows:
            return None
        event = RavelEvent(rows[-1], self._inference_geometry, (), True, False, 0)
        return RavelContext((event,), (), int(ids.numel()))

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
        self._scale_loss = None
        return None

    @staticmethod
    def _centers(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        yy, xx = torch.meshgrid(
            (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height,
            (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width,
            indexing="ij",
        )
        return xx.reshape(-1), yy.reshape(-1)

    def _dense_support(self, box: Sequence[int], device: torch.device) -> torch.Tensor:
        if self._dense_hw is None or self._grid_hw is None:
            raise RuntimeError("Dense grid is unavailable")
        height, width = self._dense_hw
        source_height, source_width = self._grid_hw
        y_grid, x_grid = torch.meshgrid(
            torch.arange(1, source_height, device=device, dtype=torch.float32)
            / source_height,
            torch.arange(1, source_width, device=device, dtype=torch.float32)
            / source_width,
            indexing="ij",
        )
        x, y = x_grid.reshape(-1), y_grid.reshape(-1)
        x1, y1, x2, y2 = [float(value) / 1000.0 for value in box]
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        support = ((x >= left) & (x <= right) & (y >= top) & (y <= bottom)).float()
        if float(support.sum()) == 0:
            distance = (x - (left + right) * 0.5).square() + (y - (top + bottom) * 0.5).square()
            support[int(distance.argmin())] = 1.0
        return support

    def _route(self, hidden_states: torch.Tensor, context: RavelContext) -> torch.Tensor:
        if self._prompt_point is None:
            return hidden_states
        if self._global_projected is None or self._dense_projected is None:
            return hidden_states
        events = tuple(
            event
            for event in context.events
            if event.geometry in {"hbb", "none"} and event.row and max(event.row) < hidden_states.shape[1]
        )
        if not events:
            return hidden_states
        if self._grid_hw is None or self._dense_hw is None:
            raise RuntimeError("Visual grid is unavailable")
        self.route_call_count += 1
        dtype = self.query_proj.weight.dtype
        global_tokens = self._global_projected.to(hidden_states.device, dtype)
        dense_tokens = self._dense_projected.to(hidden_states.device, dtype)
        global_keys = F.normalize(self.global_key(global_tokens).float(), dim=-1)
        dense_keys = F.normalize(self.dense_key(dense_tokens).float(), dim=-1)
        global_values = self.global_value(global_tokens)
        dense_values = self.dense_value(dense_tokens)
        gh, gw = self._grid_hw[0] // 2, self._grid_hw[1] // 2
        dh, dw = self._dense_hw
        global_x, global_y = self._centers(gh, gw, hidden_states.device)
        source_height, source_width = self._grid_hw
        dense_y_grid, dense_x_grid = torch.meshgrid(
            torch.arange(1, source_height, device=hidden_states.device, dtype=torch.float32)
            / source_height,
            torch.arange(1, source_width, device=hidden_states.device, dtype=torch.float32)
            / source_width,
            indexing="ij",
        )
        dense_x, dense_y = dense_x_grid.reshape(-1), dense_y_grid.reshape(-1)
        point = hidden_states.new_tensor(
            [self._prompt_point.x / 1000.0, self._prompt_point.y / 1000.0], dtype=dtype
        )
        point_feature = self.point_proj(point)
        updated = hidden_states.clone()
        address_losses: list[torch.Tensor] = []
        presence_logits: list[torch.Tensor] = []
        presence_targets: list[float] = []
        scale_losses: list[torch.Tensor] = []
        support_hits: list[torch.Tensor] = []
        expected_widths: list[torch.Tensor] = []
        expected_heights: list[torch.Tensor] = []
        strengths = self.residual_strength().to(hidden_states.device, hidden_states.dtype)

        for event in events:
            query_state = hidden_states[0, event.row[0]]
            if self._training_context and self.isolate_controller_gradients:
                query_state = query_state.detach()
            query = self.query_proj(
                self.query_norm(query_state.to(self.query_norm.weight.dtype)).to(dtype)
            )
            query = F.normalize(query.float(), dim=-1)
            scale_input = torch.cat((query.to(dtype), point_feature), dim=-1)
            width_logits = self.width_head(scale_input).float()
            height_logits = self.height_head(scale_input).float()
            anchors = self.scale_anchors.to(width_logits.device)
            expected_width = torch.dot(width_logits.softmax(-1), anchors)
            expected_height = torch.dot(height_logits.softmax(-1), anchors)
            expected_widths.append(expected_width)
            expected_heights.append(expected_height)
            scale_feature = self.scale_proj(
                torch.stack((expected_width, expected_height)).to(dtype)
            )

            global_semantic = torch.mv(global_keys, query) / self.address_temperature
            global_distance = (global_x - point[0].float()).square() + (global_y - point[1].float()).square()
            global_probability = (global_semantic - global_distance / 0.5**2).softmax(0)
            global_evidence = torch.matmul(global_probability.to(dtype), global_values)

            sigma_x = torch.maximum(expected_width * 0.5, expected_width.new_tensor(0.5 / dw))
            sigma_y = torch.maximum(expected_height * 0.5, expected_height.new_tensor(0.5 / dh))
            dense_prior = -0.5 * (
                ((dense_x - point[0].float()) / sigma_x).square()
                + ((dense_y - point[1].float()) / sigma_y).square()
            )
            dense_semantic = torch.mv(dense_keys, query) / self.address_temperature
            dense_probability = (dense_semantic + dense_prior).softmax(0)
            dense_evidence = torch.matmul(dense_probability.to(dtype), dense_values)
            common = torch.cat(
                (query.to(dtype), point_feature, global_evidence, dense_evidence, scale_feature), dim=-1
            )
            presence_logit = self.presence_head(common).squeeze(-1).float()
            presence_logits.append(presence_logit)
            presence_targets.append(0.0 if event.geometry == "none" else 1.0)
            gate = presence_logit.sigmoid().to(updated.dtype)

            lane_positions = event.row[1:-1]
            lane_count = 1 if event.geometry == "none" else min(4, len(lane_positions))
            for lane_index, position in enumerate(lane_positions[:lane_count]):
                fused = torch.cat((common, self.lane_embedding[lane_index]), dim=-1)
                residual = self.evidence_up(fused).to(updated.dtype)
                updated[0, position] = updated[0, position] + strengths[lane_index] * gate * residual

            if event.geometry == "hbb" and len(event.coordinates) == 4:
                x1, y1, x2, y2 = event.coordinates
                target_width = max(1e-3, abs(x2 - x1) / 1000.0)
                target_height = max(1e-3, abs(y2 - y1) / 1000.0)
                width_target = torch.argmin((anchors - target_width).abs()).view(1)
                height_target = torch.argmin((anchors - target_height).abs()).view(1)
                scale_losses.append(
                    F.cross_entropy(width_logits.view(1, -1), width_target)
                    + F.cross_entropy(height_logits.view(1, -1), height_target)
                )
                support = self._dense_support(event.coordinates, hidden_states.device)
                mass = torch.dot(dense_probability, support)
                address_losses.append(-mass.clamp_min(1e-8).log())
                support_hits.append(support[int(dense_probability.argmax())].gt(0).float())

        self._address_loss = torch.stack(address_losses).mean() if address_losses else None
        self._presence_loss = (
            F.binary_cross_entropy_with_logits(
                torch.stack(presence_logits),
                torch.tensor(presence_targets, device=hidden_states.device),
            )
            if presence_logits
            else None
        )
        self._scale_loss = torch.stack(scale_losses).mean() if scale_losses else None
        self.last_diagnostics = {
            "single_encode_events": float(len(events)),
            "single_encode_vision_captures": float(self.vision_capture_count),
            "single_encode_global_tokens": float(global_tokens.shape[0]),
            "single_encode_dense_tokens": float(dense_tokens.shape[0]),
            "single_encode_dense_tokens_in_qwen": 0.0,
            "single_encode_address_loss": 0.0 if self._address_loss is None else float(self._address_loss.detach()),
            "single_encode_presence_loss": 0.0 if self._presence_loss is None else float(self._presence_loss.detach()),
            "single_encode_scale_loss": 0.0 if self._scale_loss is None else float(self._scale_loss.detach()),
            "single_encode_support_hit_rate": 0.0 if not support_hits else float(torch.stack(support_hits).mean().detach()),
            "single_encode_expected_width": float(torch.stack(expected_widths).mean().detach()),
            "single_encode_expected_height": float(torch.stack(expected_heights).mean().detach()),
            "single_encode_residual_strength": float(strengths[:4].mean().detach()),
        }
        return updated

    def _layer_pre_hook(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        if self._context is None or not args:
            return None
        return (self._route(args[0], self._context), *args[1:]), kwargs

    def observe_generated_block(self, block_type: str, _tokens: Any) -> None:
        if block_type in {"ref", "reference", "ref_object"}:
            self._geometry_route_gate = 1.0
        elif block_type in {"coord_box", "point_box", "empty_box", "im_end"}:
            self._geometry_route_gate = 0.0

    def install(self, model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("Controller hooks are already installed")
        causal = base_causal_lm(model.language_model)
        layers = causal.model.layers
        if not 0 <= self.relay_layer < len(layers):
            raise ValueError(f"relay_layer={self.relay_layer} outside 0..{len(layers)-1}")
        self._projector = model.mlp1
        self._handles.append(model.vision_model.encoder.register_forward_hook(self._vision_hook))
        self._handles.append(model.mlp1.register_forward_hook(self._projector_hook))
        self._handles.append(causal.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True))
        self._handles.append(
            layers[self.relay_layer].register_forward_pre_hook(self._layer_pre_hook, with_kwargs=True)
        )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._projector = None
