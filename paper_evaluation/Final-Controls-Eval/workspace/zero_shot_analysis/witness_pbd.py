"""Point-routed, grammar-safe geometry adapter for LocateAnything.

The adapter treats an interior object point as an explicit visual witness.  A
Qwen geometry-block anchor predicts a distribution over MoonViT tokens before
the released 2x2 merger.  Multi-scale evidence is pooled around that point and
converted into a product-of-experts bias over the 1,001 native coordinate
tokens.  The bias is applied only at the four coordinate positions of an HBB
block; semantic text, ``<ref>`` transitions, wrappers, None and EOS logits are
left byte-for-byte untouched.

This module intentionally keeps the released checkpoint and tokenizer intact.
It is a detection-only screening implementation for the WitnessRoute-PBD
hypothesis, not a benchmark-independent claim.
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


class WitnessRoutePBD(nn.Module):
    """Route a semantic instance query through a point to coordinate logits."""

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
        coordinate_sigma: float = 0.08,
        initial_coordinate_strength: float = 0.05,
        initial_extent: float = 0.05,
    ) -> None:
        super().__init__()
        if min(hidden_size, vision_hidden_size, evidence_dim) <= 0:
            raise ValueError("WitnessRoute dimensions must be positive")
        if point_temperature <= 0 or coordinate_sigma <= 0:
            raise ValueError("WitnessRoute temperatures must be positive")
        if not local_sigmas or any(value <= 0 for value in local_sigmas):
            raise ValueError("WitnessRoute local sigmas must be positive")
        if not 0 < initial_coordinate_strength < 0.5:
            raise ValueError("initial_coordinate_strength must be in (0, 0.5)")
        if not 0 < initial_extent < 1:
            raise ValueError("initial_extent must be in (0, 1)")

        self.hidden_size = int(hidden_size)
        self.vision_hidden_size = int(vision_hidden_size)
        self.relay_layer = int(relay_layer)
        self.evidence_dim = int(evidence_dim)
        self.point_temperature = float(point_temperature)
        self.local_sigmas = tuple(float(value) for value in local_sigmas)
        self.token_ids = {str(key): int(value) for key, value in token_ids.items()}

        self.query_proj = nn.Linear(self.hidden_size, self.evidence_dim, bias=False)
        self.key_proj = nn.Linear(self.vision_hidden_size, self.evidence_dim, bias=False)
        self.value_proj = nn.Linear(self.vision_hidden_size, self.evidence_dim, bias=False)
        self.anchor_proj = nn.Linear(self.hidden_size, self.evidence_dim, bias=False)
        self.point_proj = nn.Sequential(
            nn.Linear(2, self.evidence_dim),
            nn.GELU(),
            nn.Linear(self.evidence_dim, self.evidence_dim),
        )
        fusion_width = self.evidence_dim * (2 + len(self.local_sigmas))
        self.extent_head = nn.Sequential(
            nn.Linear(fusion_width, self.evidence_dim),
            nn.GELU(),
            nn.Linear(self.evidence_dim, 4),
        )

        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.anchor_proj.weight)
        nn.init.zeros_(self.extent_head[-1].weight)
        extent_logit = math.log(initial_extent / (1.0 - initial_extent))
        nn.init.constant_(self.extent_head[-1].bias, extent_logit)

        strength_logit = math.log(
            initial_coordinate_strength / (0.5 - initial_coordinate_strength)
        )
        self.coordinate_strength_logit = nn.Parameter(
            torch.full((4,), float(strength_logit))
        )
        sigma_ratio = (coordinate_sigma - 0.015) / (0.25 - 0.015)
        sigma_ratio = min(1.0 - 1e-6, max(1e-6, sigma_ratio))
        self.coordinate_sigma_logit = nn.Parameter(
            torch.tensor(math.log(sigma_ratio / (1.0 - sigma_ratio)))
        )

        self._context: RelayContext | None = None
        self._training_context = False
        self._grid_hw: tuple[int, int] | None = None
        self._fine_tokens: torch.Tensor | None = None
        self._coordinate_biases: torch.Tensor | None = None
        self._active_rows: tuple[tuple[int, int, int, int, int, int], ...] = ()
        self._point_loss: torch.Tensor | None = None
        self._geometry_loss: torch.Tensor | None = None
        self._teacher_forcing_ratio = 0.0
        self._point_override: tuple[float, float] | None = None
        self._handles: list[Any] = []
        self.last_diagnostics: dict[str, float] = {}
        self.route_call_count = 0
        self.logit_call_count = 0

    @property
    def point_loss(self) -> torch.Tensor | None:
        return self._point_loss

    @property
    def geometry_loss(self) -> torch.Tensor | None:
        return self._geometry_loss

    @property
    def coordinate_biases(self) -> torch.Tensor | None:
        return self._coordinate_biases

    @property
    def active_rows(self) -> tuple[tuple[int, int, int, int, int, int], ...]:
        return self._active_rows

    def coordinate_strength(self) -> torch.Tensor:
        return 0.5 * torch.sigmoid(self.coordinate_strength_logit)

    def coordinate_sigma(self) -> torch.Tensor:
        return 0.015 + 0.235 * torch.sigmoid(self.coordinate_sigma_logit)

    def set_teacher_forcing_ratio(self, value: float) -> None:
        self._teacher_forcing_ratio = min(1.0, max(0.0, float(value)))

    def set_point_override(self, point_xy: tuple[float, float] | None) -> None:
        """Set a normalized intervention point for counterfactual diagnostics."""

        if point_xy is None:
            self._point_override = None
            return
        x, y = (float(point_xy[0]), float(point_xy[1]))
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("Witness intervention point must be normalized to [0, 1]")
        self._point_override = (x, y)

    def _set_grid(self, image_grid_hws: Any) -> None:
        values = torch.as_tensor(image_grid_hws, dtype=torch.long).reshape(-1, 2)
        if int(values.shape[0]) != 1:
            raise ValueError("WitnessRoute supports exactly one image per record")
        height, width = int(values[0, 0]), int(values[0, 1])
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid MoonViT patch grid: {(height, width)}")
        self._grid_hw = (height, width)

    def _reset_sample_state(self) -> None:
        self._fine_tokens = None
        self._coordinate_biases = None
        self._active_rows = ()
        self._point_loss = None
        self._geometry_loss = None
        self.last_diagnostics = {}
        self.route_call_count = 0
        self.logit_call_count = 0

    def set_training_context(self, context: RelayContext, image_grid_hws: Any) -> None:
        self._context = context
        self._training_context = True
        self._set_grid(image_grid_hws)
        self._reset_sample_state()

    def set_inference_image_layout(self, _input_ids: torch.Tensor, image_grid_hws: Any) -> None:
        self._context = None
        self._training_context = False
        self._set_grid(image_grid_hws)
        self._reset_sample_state()
        self._teacher_forcing_ratio = 0.0

    def finish_forward(self) -> None:
        self._context = None
        self._training_context = False
        self._coordinate_biases = None
        self._active_rows = ()
        self._point_loss = None
        self._geometry_loss = None

    def _vision_hook(self, _module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        fine = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(fine, torch.Tensor) or fine.ndim != 2:
            raise RuntimeError("MoonViT encoder did not return [tokens, hidden] features")
        if self._grid_hw is None:
            raise RuntimeError("WitnessRoute grid was not configured before visual encoding")
        expected = self._grid_hw[0] * self._grid_hw[1]
        if int(fine.shape[0]) != expected:
            raise RuntimeError(
                f"WitnessRoute fine-token mismatch: features={fine.shape[0]} expected={expected}"
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
        self._coordinate_biases = None
        self._active_rows = ()
        self._point_loss = None
        self._geometry_loss = None
        self.last_diagnostics = {}
        return None

    def _token_centers(self, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._grid_hw is None:
            raise RuntimeError("WitnessRoute grid is unavailable")
        height, width = self._grid_hw
        yy, xx = torch.meshgrid(
            (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height,
            (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width,
            indexing="ij",
        )
        return xx.reshape(-1), yy.reshape(-1)

    def _box_support(self, box: Sequence[int], *, device: torch.device) -> torch.Tensor:
        if self._grid_hw is None:
            raise RuntimeError("WitnessRoute grid is unavailable")
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
        support = ((overlap_y[:, None] * overlap_x[None, :]) * (height * width)).clamp(0, 1)
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
            raise RuntimeError("WitnessRoute fine features do not match the patch grid")

        self.route_call_count += 1
        fine = fine.to(device=hidden_states.device)
        route_dtype = self.query_proj.weight.dtype
        anchors = torch.stack([hidden_states[0, row[0]] for row in rows], dim=0)
        queries = F.normalize(self.query_proj(anchors.to(route_dtype)).float(), dim=-1)
        keys = F.normalize(self.key_proj(fine.to(route_dtype)).float(), dim=-1)
        point_logits = torch.matmul(queries, keys.transpose(0, 1)) / self.point_temperature
        point_probabilities = point_logits.softmax(dim=-1)
        token_x, token_y = self._token_centers(device=point_logits.device)
        predicted_x = torch.matmul(point_probabilities, token_x)
        predicted_y = torch.matmul(point_probabilities, token_y)
        predicted_points = torch.stack((predicted_x, predicted_y), dim=-1)

        target_boxes = context.box_targets[: len(rows)] if context.box_targets else ()
        point_losses: list[torch.Tensor] = []
        target_centers: list[tuple[float, float]] = []
        point_hit: list[torch.Tensor] = []
        for index, box in enumerate(target_boxes):
            support = self._box_support(box, device=point_logits.device)
            interior_mass = (point_probabilities[index] * support.gt(0).float()).sum()
            point_losses.append(-interior_mass.clamp_min(1e-8).log())
            x1, y1, x2, y2 = [float(value) / 1000.0 for value in box]
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            target_centers.append((0.5 * (left + right), 0.5 * (top + bottom)))
            point_hit.append(
                (
                    predicted_points[index, 0].ge(left)
                    & predicted_points[index, 0].le(right)
                    & predicted_points[index, 1].ge(top)
                    & predicted_points[index, 1].le(bottom)
                ).float()
            )

        route_points = predicted_points
        if target_centers and self._teacher_forcing_ratio > 0:
            targets = torch.tensor(target_centers, device=route_points.device, dtype=route_points.dtype)
            ratio = self._teacher_forcing_ratio
            route_points = ratio * targets + (1.0 - ratio) * route_points
        if self._point_override is not None:
            override = torch.tensor(
                self._point_override, device=route_points.device, dtype=route_points.dtype
            )
            route_points = override.unsqueeze(0).expand_as(route_points)

        values = self.value_proj(fine.to(route_dtype))
        pooled_scales: list[torch.Tensor] = []
        height, width = self._grid_hw
        for sigma in self.local_sigmas:
            dx = (token_x.unsqueeze(0) - route_points[:, :1]) * width
            dy = (token_y.unsqueeze(0) - route_points[:, 1:2]) * height
            local_logits = -0.5 * (dx.square() + dy.square()) / (sigma**2)
            local_weights = local_logits.softmax(dim=-1).to(route_dtype)
            pooled_scales.append(torch.matmul(local_weights, values))

        fused = torch.cat(
            (
                self.anchor_proj(anchors.to(route_dtype)),
                self.point_proj(route_points.to(route_dtype)),
                *pooled_scales,
            ),
            dim=-1,
        )
        raw_extents = self.extent_head(fused)
        extents = torch.sigmoid(raw_extents.float())
        x = route_points[:, 0]
        y = route_points[:, 1]
        coordinates = torch.stack(
            (
                (x - extents[:, 0]).clamp(0, 1),
                (y - extents[:, 1]).clamp(0, 1),
                (x + extents[:, 2]).clamp(0, 1),
                (y + extents[:, 3]).clamp(0, 1),
            ),
            dim=-1,
        )
        bins = torch.linspace(0.0, 1.0, 1001, device=coordinates.device)
        sigma = self.coordinate_sigma()
        biases = -0.5 * ((bins.view(1, 1, -1) - coordinates.unsqueeze(-1)) / sigma).square()

        geometry_loss = None
        geometry_l1 = None
        if target_boxes:
            targets = torch.tensor(target_boxes, device=biases.device, dtype=torch.long)
            geometry_loss = F.cross_entropy(biases.reshape(-1, 1001), targets.reshape(-1))
            geometry_l1 = (
                coordinates - targets.float().div(1000.0)
            ).abs().mean()

        self._active_rows = rows
        self._coordinate_biases = biases
        self._point_loss = torch.stack(point_losses).mean() if point_losses else None
        self._geometry_loss = geometry_loss
        self.last_diagnostics = {
            "witness_point_loss": 0.0
            if self._point_loss is None
            else float(self._point_loss.detach()),
            "witness_point_inside_rate": 0.0
            if not point_hit
            else float(torch.stack(point_hit).mean().detach()),
            "witness_geometry_l1_0_1000": 0.0
            if geometry_l1 is None
            else float(1000.0 * geometry_l1.detach()),
            "witness_coordinate_strength": float(self.coordinate_strength().mean().detach()),
            "witness_coordinate_sigma": float(sigma.detach()),
            "witness_teacher_forcing": float(self._teacher_forcing_ratio),
            "witness_fine_tokens": float(fine.shape[0]),
            "witness_blocks": float(len(rows)),
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

    def apply_coordinate_bias(self, logits: torch.Tensor) -> torch.Tensor:
        biases = self._coordinate_biases
        if biases is None or not self._active_rows:
            return logits
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        if coord_end - coord_start + 1 != 1001:
            raise RuntimeError("WitnessRoute expects the released 1,001 coordinate tokens")
        updated = logits.clone()
        strength = self.coordinate_strength().to(device=logits.device, dtype=logits.dtype)
        for block_index, row in enumerate(self._active_rows):
            if block_index >= int(biases.shape[0]):
                break
            for lane, position in enumerate(row[1:5]):
                if position >= int(updated.shape[-2]):
                    continue
                lane_bias = biases[block_index, lane].to(logits.dtype)
                updated[0, position, coord_start : coord_end + 1] += strength[lane] * (
                    lane_bias + 2.0
                )
        self.logit_call_count += 1
        return updated

    def _lm_head_hook(
        self,
        _module: nn.Module,
        _args: tuple[Any, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self.apply_coordinate_bias(output)

    def install(self, model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("WitnessRoute hooks are already installed")
        causal = base_causal_lm(model.language_model)
        layers = causal.model.layers
        if not 0 <= self.relay_layer < len(layers):
            raise ValueError(f"relay_layer={self.relay_layer} outside 0..{len(layers)-1}")
        self._handles.append(model.vision_model.encoder.register_forward_hook(self._vision_hook))
        self._handles.append(
            causal.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        )
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


def coordinate_adjusted_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    controller: WitnessRoutePBD,
    *,
    chunk: int = 128,
) -> torch.Tensor:
    """Native shifted CE with WitnessRoute applied only to coordinate rows."""

    shifted_hidden = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
    shifted_labels = labels[:, 1:].reshape(-1)
    valid = shifted_labels.ne(IGNORE_INDEX)
    valid_hidden = shifted_hidden[valid]
    valid_labels = shifted_labels[valid]
    count = int(valid_labels.numel())
    if count == 0:
        raise RuntimeError("WitnessRoute sample has no supervised tokens")

    total = hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, count, int(chunk)):
        stop = min(count, start + int(chunk))
        logits = F.linear(valid_hidden[start:stop], weight).float()
        total = total + F.cross_entropy(logits, valid_labels[start:stop], reduction="sum")

    biases = controller.coordinate_biases
    rows = controller.active_rows
    if biases is None or not rows:
        return total / count

    coord_start = int(controller.token_ids["coord_start_token_id"])
    coord_end = int(controller.token_ids["coord_end_token_id"])
    strength = controller.coordinate_strength()
    corrections: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    for block_index, row in enumerate(rows):
        for lane, position in enumerate(row[1:5]):
            target_position = position + 1
            if target_position >= labels.shape[1]:
                continue
            target = labels[0, target_position]
            if int(target) == IGNORE_INDEX:
                continue
            corrections.append(
                (hidden_states[0, position], target, biases[block_index, lane], lane)
            )

    for start in range(0, len(corrections), 32):
        group = corrections[start : start + 32]
        states = torch.stack([item[0] for item in group], dim=0)
        targets = torch.stack([item[1] for item in group], dim=0).long()
        base_logits = F.linear(states, weight).float()
        modified_logits = base_logits.clone()
        for index, (_state, _target, bias, lane) in enumerate(group):
            modified_logits[index, coord_start : coord_end + 1] += strength[lane].float() * (
                bias.float() + 2.0
            )
        total = total - F.cross_entropy(base_logits, targets, reduction="sum")
        total = total + F.cross_entropy(modified_logits, targets, reduction="sum")
    return total / count


def load_witness_adapter(model: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a compact WitnessRoute adapter plus its matched Qwen LoRA state."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    configure_lora_only(model, int(config["lora_rank"]))
    controller = WitnessRoutePBD(
        hidden_size=int(model.config.text_config.hidden_size),
        vision_hidden_size=int(model.config.vision_config.hidden_size),
        relay_layer=int(config["relay_layer"]),
        token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
        evidence_dim=int(config.get("evidence_dim", 128)),
        point_temperature=float(config.get("point_temperature", 0.10)),
        local_sigmas=tuple(config.get("local_sigmas", (0.75, 1.5, 3.0))),
        coordinate_sigma=float(config.get("coordinate_sigma", 0.08)),
        initial_coordinate_strength=float(config.get("initial_coordinate_strength", 0.05)),
        initial_extent=float(config.get("initial_extent", 0.05)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    controller.set_teacher_forcing_ratio(0.0)
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
