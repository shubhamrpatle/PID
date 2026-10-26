"""Latent-address visual re-entry for LocateAnything's native PBD blocks.

The controller observes the four coordinate lanes halfway through Qwen's
released six-token HBB block.  Those lane states jointly address MoonViT's
pre-merger feature grid, retrieve local evidence, and produce a residual only
over the 1,001 native coordinate-token logits.  Semantic tokens, wrappers,
``None``, EOS, the tokenizer, and the PBD grammar are never modified.

There is no proposal list or fixed object slot count.  LocateAnything retains
its normal autoregressive class/object loop and native EOS decision; the
controller is invoked only for the current candidate geometry block.  This is
an experimental detection-only screening implementation, not a benchmark
claim by itself.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .pointrelay_pbd import (
    RelayContext,
    base_causal_lm,
    configure_lora_only,
    restore_trainable_state,
)


IGNORE_INDEX = -100


class LatentAddressVisualReentryPBD(nn.Module):
    """Read fine MoonViT evidence from a latent mid-block spatial address."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vision_hidden_size: int,
        relay_layer: int,
        token_ids: Mapping[str, int],
        evidence_dim: int = 128,
        point_temperature: float = 0.10,
        local_sigmas: Sequence[float] = (0.75, 1.5, 3.0),
        initial_residual_strength: float = 0.10,
        maximum_residual_strength: float = 2.0,
    ) -> None:
        super().__init__()
        if min(hidden_size, vision_hidden_size, evidence_dim) <= 0:
            raise ValueError("Latent-address dimensions must be positive")
        if point_temperature <= 0:
            raise ValueError("point_temperature must be positive")
        if not local_sigmas or any(value <= 0 for value in local_sigmas):
            raise ValueError("local_sigmas must contain positive values")
        if not 0 < initial_residual_strength < maximum_residual_strength:
            raise ValueError(
                "initial_residual_strength must be between zero and the maximum"
            )

        self.hidden_size = int(hidden_size)
        self.vision_hidden_size = int(vision_hidden_size)
        self.relay_layer = int(relay_layer)
        self.evidence_dim = int(evidence_dim)
        self.point_temperature = float(point_temperature)
        self.local_sigmas = tuple(float(value) for value in local_sigmas)
        self.maximum_residual_strength = float(maximum_residual_strength)
        self.token_ids = {str(key): int(value) for key, value in token_ids.items()}

        self.lane_norm = nn.LayerNorm(self.hidden_size)
        self.lane_proj = nn.Linear(self.hidden_size, self.evidence_dim, bias=False)
        self.query_fuse = nn.Sequential(
            nn.Linear(4 * self.evidence_dim, 2 * self.evidence_dim),
            nn.GELU(),
            nn.Linear(2 * self.evidence_dim, self.evidence_dim),
        )
        self.key_proj = nn.Linear(self.vision_hidden_size, self.evidence_dim, bias=False)
        self.value_proj = nn.Linear(
            self.vision_hidden_size, self.evidence_dim, bias=False
        )
        self.point_proj = nn.Sequential(
            nn.Linear(2, self.evidence_dim),
            nn.GELU(),
            nn.Linear(self.evidence_dim, self.evidence_dim),
        )
        self.lane_embedding = nn.Parameter(torch.zeros(4, self.evidence_dim))
        fusion_width = self.evidence_dim * (4 + len(self.local_sigmas))
        self.coordinate_head = nn.Sequential(
            nn.Linear(fusion_width, 2 * self.evidence_dim),
            nn.GELU(),
            nn.LayerNorm(2 * self.evidence_dim),
            nn.Linear(2 * self.evidence_dim, 1001),
        )

        for layer in (self.lane_proj, self.key_proj, self.value_proj):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.query_fuse[0].weight)
        nn.init.zeros_(self.query_fuse[0].bias)
        nn.init.xavier_uniform_(self.query_fuse[2].weight)
        nn.init.zeros_(self.query_fuse[2].bias)
        # Exact zero residual at initialization preserves the parent model.
        nn.init.zeros_(self.coordinate_head[-1].weight)
        nn.init.zeros_(self.coordinate_head[-1].bias)
        ratio = initial_residual_strength / maximum_residual_strength
        self.residual_strength_logit = nn.Parameter(
            torch.full((4,), math.log(ratio / (1.0 - ratio)))
        )

        self._context: RelayContext | None = None
        self._training_context = False
        self._grid_hw: tuple[int, int] | None = None
        self._fine_tokens: torch.Tensor | None = None
        self._coordinate_residuals: torch.Tensor | None = None
        self._active_rows: tuple[tuple[int, int, int, int, int, int], ...] = ()
        self._point_loss: torch.Tensor | None = None
        self._handles: list[Any] = []
        self.last_diagnostics: dict[str, float] = {}
        self.route_call_count = 0
        self.logit_call_count = 0
        self.box_gated_blocks = 0

    @property
    def point_loss(self) -> torch.Tensor | None:
        return self._point_loss

    @property
    def coordinate_residuals(self) -> torch.Tensor | None:
        return self._coordinate_residuals

    @property
    def active_rows(self) -> tuple[tuple[int, int, int, int, int, int], ...]:
        return self._active_rows

    def residual_strength(self) -> torch.Tensor:
        return self.maximum_residual_strength * torch.sigmoid(
            self.residual_strength_logit
        )

    def _set_grid(self, image_grid_hws: Any) -> None:
        values = torch.as_tensor(image_grid_hws, dtype=torch.long).reshape(-1, 2)
        if int(values.shape[0]) != 1:
            raise ValueError("Latent-address pilot supports one image per record")
        height, width = int(values[0, 0]), int(values[0, 1])
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid MoonViT patch grid: {(height, width)}")
        self._grid_hw = (height, width)

    def _reset_sample_state(self) -> None:
        self._fine_tokens = None
        self._coordinate_residuals = None
        self._active_rows = ()
        self._point_loss = None
        self.last_diagnostics = {}
        self.route_call_count = 0
        self.logit_call_count = 0
        self.box_gated_blocks = 0

    def set_training_context(self, context: RelayContext, image_grid_hws: Any) -> None:
        self._context = context
        self._training_context = True
        self._set_grid(image_grid_hws)
        self._reset_sample_state()

    def set_inference_image_layout(
        self, _input_ids: torch.Tensor, image_grid_hws: Any
    ) -> None:
        self._context = None
        self._training_context = False
        self._set_grid(image_grid_hws)
        self._reset_sample_state()

    def finish_forward(self) -> None:
        self._context = None
        self._training_context = False
        self._coordinate_residuals = None
        self._active_rows = ()
        self._point_loss = None

    def _vision_hook(
        self, _module: nn.Module, _args: tuple[Any, ...], output: Any
    ) -> None:
        fine = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(fine, torch.Tensor) or fine.ndim != 2:
            raise RuntimeError("MoonViT encoder did not return [tokens, hidden]")
        if self._grid_hw is None:
            raise RuntimeError("Image grid was not configured before MoonViT")
        expected = self._grid_hw[0] * self._grid_hw[1]
        if int(fine.shape[0]) != expected:
            raise RuntimeError(
                f"Fine-token mismatch: features={fine.shape[0]} expected={expected}"
            )
        self._fine_tokens = fine.detach()

    def _inference_context(self, input_ids: torch.Tensor | None) -> RelayContext | None:
        if input_ids is None or input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            return None
        ids = input_ids[0]
        mask_id = int(self.token_ids["default_mask_token_id"])
        predecessors = {
            int(self.token_ids["ref_end_token_id"]),
            int(self.token_ids["box_end_token_id"]),
        }
        rows: list[tuple[int, int, int, int, int, int]] = []
        lower = max(0, int(ids.numel()) - 18)
        upper = max(0, int(ids.numel()) - 5)
        for start in range(lower, upper):
            if start + 6 > int(ids.numel()) or int(ids[start]) not in predecessors:
                continue
            if bool(ids[start + 1 : start + 6].eq(mask_id).all()):
                rows.append(tuple(range(start, start + 6)))
        if not rows:
            return None
        return RelayContext((rows[-1],), (), (), int(ids.numel()), ())

    def _model_pre_hook(
        self,
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if self._training_context:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        self._context = self._inference_context(input_ids)
        self._coordinate_residuals = None
        self._active_rows = ()
        self._point_loss = None
        self.last_diagnostics = {}
        return None

    def _token_centers(self, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._grid_hw is None:
            raise RuntimeError("MoonViT grid is unavailable")
        height, width = self._grid_hw
        yy, xx = torch.meshgrid(
            (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height,
            (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width,
            indexing="ij",
        )
        return xx.reshape(-1), yy.reshape(-1)

    def _box_support(self, box: Sequence[int], *, device: torch.device) -> torch.Tensor:
        if self._grid_hw is None:
            raise RuntimeError("MoonViT grid is unavailable")
        height, width = self._grid_hw
        x1, y1, x2, y2 = [float(value) / 1000.0 for value in box]
        left, right = sorted((max(0.0, x1), min(1.0, x2)))
        top, bottom = sorted((max(0.0, y1), min(1.0, y2)))
        xs0 = torch.arange(width, device=device, dtype=torch.float32) / width
        xs1 = (torch.arange(width, device=device, dtype=torch.float32) + 1.0) / width
        ys0 = torch.arange(height, device=device, dtype=torch.float32) / height
        ys1 = (torch.arange(height, device=device, dtype=torch.float32) + 1.0) / height
        overlap_x = (
            torch.minimum(xs1, xs1.new_tensor(right))
            - torch.maximum(xs0, xs0.new_tensor(left))
        ).clamp_min(0)
        overlap_y = (
            torch.minimum(ys1, ys1.new_tensor(bottom))
            - torch.maximum(ys0, ys0.new_tensor(top))
        ).clamp_min(0)
        support = ((overlap_y[:, None] * overlap_x[None, :]) * height * width).clamp(0, 1)
        support = support.reshape(-1)
        if float(support.sum()) <= 1e-8:
            cx = max(0, min(width - 1, int(0.5 * (left + right) * width)))
            cy = max(0, min(height - 1, int(0.5 * (top + bottom) * height)))
            support[cy * width + cx] = 1.0
        return support

    def _route(self, hidden_states: torch.Tensor, context: RelayContext) -> None:
        rows = tuple(row for row in context.rows if max(row) < hidden_states.shape[1])
        fine = self._fine_tokens
        if not rows or fine is None:
            return
        if self._grid_hw is None or self._grid_hw[0] * self._grid_hw[1] != int(fine.shape[0]):
            raise RuntimeError("Fine features do not match the MoonViT patch grid")

        self.route_call_count += 1
        route_dtype = self.lane_proj.weight.dtype
        fine = fine.to(device=hidden_states.device, dtype=route_dtype)
        lanes = torch.stack(
            [torch.stack([hidden_states[0, position] for position in row[1:5]]) for row in rows]
        )
        normalized_lanes = self.lane_norm(
            lanes.to(dtype=self.lane_norm.weight.dtype)
        )
        lane_features = self.lane_proj(normalized_lanes.to(route_dtype))
        queries = self.query_fuse(lane_features.reshape(len(rows), -1))
        queries = F.normalize(queries.float(), dim=-1)
        keys = F.normalize(self.key_proj(fine).float(), dim=-1)
        point_logits = torch.matmul(queries, keys.transpose(0, 1)) / self.point_temperature
        point_probabilities = point_logits.softmax(dim=-1)

        token_x, token_y = self._token_centers(device=point_logits.device)
        predicted_points = torch.stack(
            (
                torch.matmul(point_probabilities, token_x),
                torch.matmul(point_probabilities, token_y),
            ),
            dim=-1,
        )

        point_losses: list[torch.Tensor] = []
        point_hits: list[torch.Tensor] = []
        target_boxes = context.box_targets[: len(rows)] if context.box_targets else ()
        for index, box in enumerate(target_boxes):
            support = self._box_support(box, device=point_logits.device)
            interior_mass = (point_probabilities[index] * support.gt(0).float()).sum()
            point_losses.append(-interior_mass.clamp_min(1e-8).log())
            x1, y1, x2, y2 = [float(value) / 1000.0 for value in box]
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            point_hits.append(
                (
                    predicted_points[index, 0].ge(left)
                    & predicted_points[index, 0].le(right)
                    & predicted_points[index, 1].ge(top)
                    & predicted_points[index, 1].le(bottom)
                ).float()
            )

        values = self.value_proj(fine)
        routed_evidence = torch.matmul(point_probabilities.to(route_dtype), values)
        pooled_scales: list[torch.Tensor] = []
        height, width = self._grid_hw
        for sigma in self.local_sigmas:
            dx = (token_x.unsqueeze(0) - predicted_points[:, :1]) * width
            dy = (token_y.unsqueeze(0) - predicted_points[:, 1:2]) * height
            local_logits = -0.5 * (dx.square() + dy.square()) / (sigma**2)
            local_weights = local_logits.softmax(dim=-1).to(route_dtype)
            pooled_scales.append(torch.matmul(local_weights, values))

        point_features = self.point_proj(predicted_points.to(route_dtype))
        residual_lanes: list[torch.Tensor] = []
        for lane_index in range(4):
            lane_id = self.lane_embedding[lane_index].unsqueeze(0).expand(len(rows), -1)
            fused = torch.cat(
                (
                    lane_features[:, lane_index],
                    routed_evidence,
                    point_features,
                    lane_id,
                    *pooled_scales,
                ),
                dim=-1,
            )
            residual_lanes.append(self.coordinate_head(fused).float())

        self._active_rows = rows
        self._coordinate_residuals = torch.stack(residual_lanes, dim=1)
        self._point_loss = torch.stack(point_losses).mean() if point_losses else None
        entropy = -(point_probabilities * point_probabilities.clamp_min(1e-8).log()).sum(-1)
        self.last_diagnostics = {
            "latent_point_loss": 0.0 if self._point_loss is None else float(self._point_loss.detach()),
            "latent_point_inside_rate": 0.0 if not point_hits else float(torch.stack(point_hits).mean().detach()),
            "latent_point_entropy": float(entropy.mean().detach()),
            "latent_residual_strength": float(self.residual_strength().mean().detach()),
            "latent_residual_abs_mean": float(self._coordinate_residuals.abs().mean().detach()),
            "latent_fine_tokens": float(fine.shape[0]),
            "latent_blocks": float(len(rows)),
        }

    def _layer_pre_hook(
        self,
        _module: nn.Module,
        args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        context = self._context
        if context is not None and context.rows:
            self._route(args[0], context)
        return None

    def apply_coordinate_residual(
        self, logits: torch.Tensor, *, require_native_box: bool
    ) -> torch.Tensor:
        residuals = self._coordinate_residuals
        if residuals is None or not self._active_rows:
            return logits
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        if coord_end - coord_start + 1 != 1001:
            raise RuntimeError("Latent-address pilot expects 1,001 coordinate tokens")
        box_start = int(self.token_ids["box_start_token_id"])
        updated = logits.clone()
        strength = self.residual_strength().to(logits.device, logits.dtype)
        for block_index, row in enumerate(self._active_rows):
            if block_index >= int(residuals.shape[0]) or max(row) >= int(logits.shape[-2]):
                continue
            if require_native_box and int(logits[0, row[0]].argmax()) != box_start:
                continue
            self.box_gated_blocks += 1
            for lane_index, position in enumerate(row[1:5]):
                updated[0, position, coord_start : coord_end + 1] += (
                    strength[lane_index] * residuals[block_index, lane_index].to(logits.dtype)
                )
        self.logit_call_count += 1
        return updated

    def _lm_head_hook(
        self,
        _module: nn.Module,
        _args: tuple[Any, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        # Hard grammar gate: if native LA did not choose <box>, no logit changes
        # are made anywhere in the block.
        return self.apply_coordinate_residual(output, require_native_box=True)

    def install(self, model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("Latent-address hooks are already installed")
        causal = base_causal_lm(model.language_model)
        layers = causal.model.layers
        if not 0 <= self.relay_layer < len(layers):
            raise ValueError(f"relay_layer={self.relay_layer} outside 0..{len(layers)-1}")
        self._handles.append(model.vision_model.encoder.register_forward_hook(self._vision_hook))
        self._handles.append(causal.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True))
        self._handles.append(
            layers[self.relay_layer].register_forward_pre_hook(
                self._layer_pre_hook, with_kwargs=True
            )
        )
        self._handles.append(causal.lm_head.register_forward_hook(self._lm_head_hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def coordinate_residual_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    controller: LatentAddressVisualReentryPBD,
    *,
    chunk: int = 128,
) -> torch.Tensor:
    """Shifted CE with residuals restricted to supervised coordinate lanes."""

    shifted_hidden = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
    shifted_labels = labels[:, 1:].reshape(-1)
    valid = shifted_labels.ne(IGNORE_INDEX)
    valid_hidden = shifted_hidden[valid]
    valid_labels = shifted_labels[valid]
    count = int(valid_labels.numel())
    if count == 0:
        raise RuntimeError("Latent-address sample has no supervised tokens")

    total = hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, count, int(chunk)):
        stop = min(count, start + int(chunk))
        logits = F.linear(valid_hidden[start:stop], weight).float()
        total = total + F.cross_entropy(logits, valid_labels[start:stop], reduction="sum")

    residuals = controller.coordinate_residuals
    rows = controller.active_rows
    if residuals is None or not rows:
        return total / count

    coord_start = int(controller.token_ids["coord_start_token_id"])
    coord_end = int(controller.token_ids["coord_end_token_id"])
    strength = controller.residual_strength()
    corrections: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    for block_index, row in enumerate(rows):
        for lane_index, position in enumerate(row[1:5]):
            target_position = position + 1
            if target_position >= labels.shape[1]:
                continue
            target = labels[0, target_position]
            if int(target) == IGNORE_INDEX:
                continue
            corrections.append(
                (
                    hidden_states[0, position],
                    target,
                    residuals[block_index, lane_index],
                    lane_index,
                )
            )

    for start in range(0, len(corrections), 32):
        group = corrections[start : start + 32]
        states = torch.stack([item[0] for item in group])
        targets = torch.stack([item[1] for item in group]).long()
        base_logits = F.linear(states, weight).float()
        modified = base_logits.clone()
        for index, (_state, _target, residual, lane_index) in enumerate(group):
            modified[index, coord_start : coord_end + 1] += (
                strength[lane_index].float() * residual.float()
            )
        total = total - F.cross_entropy(base_logits, targets, reduction="sum")
        total = total + F.cross_entropy(modified, targets, reduction="sum")
    return total / count


def load_latent_address_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Load a latent-address controller and its matched Qwen LoRA tensors."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    configure_lora_only(model, int(config["lora_rank"]))
    controller = LatentAddressVisualReentryPBD(
        hidden_size=int(model.config.text_config.hidden_size),
        vision_hidden_size=int(model.config.vision_config.hidden_size),
        relay_layer=int(config["relay_layer"]),
        token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
        evidence_dim=int(config.get("evidence_dim", 128)),
        point_temperature=float(config.get("point_temperature", 0.10)),
        local_sigmas=tuple(config.get("local_sigmas", (0.75, 1.5, 3.0))),
        initial_residual_strength=float(config.get("initial_residual_strength", 0.10)),
        maximum_residual_strength=float(config.get("maximum_residual_strength", 2.0)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
