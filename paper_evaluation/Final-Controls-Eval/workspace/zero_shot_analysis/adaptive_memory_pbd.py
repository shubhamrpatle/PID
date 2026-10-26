"""Adaptive point-address memory for LocateAnything's native PBD decoder.

The module is a compact detection-gate implementation of a learned two-pass
localization architecture:

1. a point scaffold proposes semantic addresses;
2. MoonViT re-encodes a small patch window around every address;
3. four pooled visual slots per address form a compact external memory;
4. each native six-token PBD box block softly routes to that memory; and
5. differentiable coverage discourages consecutive blocks from reusing the
   same address.

Qwen remains a single sequential decoder.  The tokenizer, output grammar and
six-token PBD block are unchanged.  Local windows are encoded in bounded
micro-batches, so the number of Qwen decoder replicas never depends on the
number of proposed objects.
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


class AdaptivePointMemoryPBD(nn.Module):
    """Route native PBD box blocks through learned point-indexed visual memory."""

    def __init__(
        self,
        *,
        hidden_size: int,
        relay_layer: int,
        token_ids: Mapping[str, int],
        memory_dim: int = 128,
        local_patch_grid: int = 12,
        max_memory_addresses: int = 64,
        local_encode_batch: int = 8,
        temperature: float = 0.10,
        initial_gate_logit: float = -3.0,
        initial_coverage_strength: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or memory_dim <= 0:
            raise ValueError("Adaptive memory dimensions must be positive")
        if local_patch_grid < 2 or local_patch_grid % 2:
            raise ValueError("local_patch_grid must be an even integer >= 2")
        if max_memory_addresses <= 0 or local_encode_batch <= 0:
            raise ValueError("Adaptive memory limits must be positive")
        if temperature <= 0:
            raise ValueError("Adaptive memory temperature must be positive")

        self.hidden_size = int(hidden_size)
        self.relay_layer = int(relay_layer)
        self.memory_dim = int(memory_dim)
        self.local_patch_grid = int(local_patch_grid)
        self.max_memory_addresses = int(max_memory_addresses)
        self.local_encode_batch = int(local_encode_batch)
        self.temperature = float(temperature)
        self.token_ids = {str(key): int(value) for key, value in token_ids.items()}

        self.query_proj = nn.Linear(self.hidden_size, self.memory_dim, bias=False)
        self.key_proj = nn.Linear(self.hidden_size, self.memory_dim, bias=False)
        self.value_down = nn.Linear(self.hidden_size, self.memory_dim, bias=False)
        self.value_up = nn.Linear(self.memory_dim, self.hidden_size, bias=False)
        self.position_mlp = nn.Sequential(
            nn.Linear(2, self.memory_dim),
            nn.GELU(),
            nn.Linear(self.memory_dim, self.memory_dim),
        )
        self.lane_queries = nn.Parameter(torch.zeros(4, self.memory_dim))
        self.lane_gate_logits = nn.Parameter(torch.full((4,), float(initial_gate_logit)))
        coverage_logit = math.log(math.expm1(max(1e-4, float(initial_coverage_strength))))
        self.coverage_strength_logit = nn.Parameter(torch.tensor(coverage_logit))

        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_down.weight)
        nn.init.normal_(self.value_up.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.lane_queries, mean=0.0, std=0.02)

        self._context: RelayContext | None = None
        self._training_context = False
        self._stored_image_columns: tuple[int, ...] = ()
        self._memory_slots: torch.Tensor | None = None
        self._memory_points: torch.Tensor | None = None
        self._target_indices: tuple[int, ...] = ()
        self._pending_points: tuple[tuple[int, int], ...] = ()
        self._coverage: torch.Tensor | None = None
        self._inference_position_key: tuple[int, ...] = ()
        self._last_inference_key: tuple[tuple[int, ...], tuple[tuple[int, ...], ...]] | None = None
        self._aux_loss: torch.Tensor | None = None
        self._handles: list[Any] = []
        self.last_diagnostics: dict[str, float] = {}
        self.route_call_count = 0

    @property
    def aux_loss(self) -> torch.Tensor | None:
        return self._aux_loss

    def coverage_strength(self) -> torch.Tensor:
        return F.softplus(self.coverage_strength_logit)

    def set_pending_points(self, points: Sequence[Sequence[float]]) -> None:
        """Set normalized 0..1000 point addresses for the next inference call."""

        unique: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for point in points:
            if len(point) < 2:
                continue
            value = (
                max(0, min(1000, int(round(float(point[0]))))),
                max(0, min(1000, int(round(float(point[1]))))),
            )
            if value in seen:
                continue
            seen.add(value)
            unique.append(value)
        self._pending_points = tuple(unique[: self.max_memory_addresses])

    def set_training_context(
        self, context: RelayContext, image_grid_hws: Any | None = None
    ) -> None:
        self._context = context
        self._training_context = True
        self._stored_image_columns = context.image_columns
        self._memory_slots = None
        self._memory_points = None
        self._target_indices = ()
        self._coverage = None
        self._inference_position_key = ()
        self._last_inference_key = None
        self._aux_loss = None
        self.last_diagnostics = {}

    def set_inference_image_layout(self, input_ids: torch.Tensor, _image_grid_hws: Any) -> None:
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("Adaptive memory inference expects [1, sequence] input IDs")
        image_index = int(self.token_ids["image_token_index"])
        self._stored_image_columns = tuple(
            int(value)
            for value in input_ids[0].eq(image_index).nonzero(as_tuple=False).flatten().tolist()
        )
        self._context = None
        self._training_context = False
        self._memory_slots = None
        self._memory_points = None
        self._target_indices = ()
        self._coverage = None
        self._inference_position_key = ()
        self._last_inference_key = None
        self._aux_loss = None
        self.last_diagnostics = {}

    @staticmethod
    def _select_indices(count: int, maximum: int) -> list[int]:
        if count <= maximum:
            return list(range(count))
        # Even coverage is deterministic and does not privilege early grammar groups.
        values = torch.linspace(0, count - 1, maximum).round().long().tolist()
        return list(dict.fromkeys(int(value) for value in values))

    def _extract_local_slots(
        self,
        model: nn.Module,
        pixel_values: torch.Tensor,
        image_grid_hws: Any,
        points: Sequence[tuple[int, int]],
    ) -> torch.Tensor | None:
        if not points:
            return None
        grids = torch.as_tensor(image_grid_hws, device=pixel_values.device, dtype=torch.long).reshape(-1, 2)
        if int(grids.shape[0]) != 1:
            raise ValueError("Adaptive memory currently supports one global image per record")
        grid_h, grid_w = int(grids[0, 0]), int(grids[0, 1])
        if grid_h < 2 or grid_w < 2 or grid_h * grid_w != int(pixel_values.shape[0]):
            raise ValueError(
                f"Invalid global patch grid {(grid_h, grid_w)} for {int(pixel_values.shape[0])} patches"
            )
        local_h = min(self.local_patch_grid, grid_h - grid_h % 2)
        local_w = min(self.local_patch_grid, grid_w - grid_w % 2)
        if local_h < 2 or local_w < 2:
            raise ValueError("Global patch grid is too small for local memory")

        patch_grid = pixel_values.reshape(grid_h, grid_w, *pixel_values.shape[1:])
        windows: list[torch.Tensor] = []
        for x_norm, y_norm in points:
            center_x = int(round(float(x_norm) / 1000.0 * max(0, grid_w - 1)))
            center_y = int(round(float(y_norm) / 1000.0 * max(0, grid_h - 1)))
            left = max(0, min(grid_w - local_w, center_x - local_w // 2))
            top = max(0, min(grid_h - local_h, center_y - local_h // 2))
            windows.append(
                patch_grid[top : top + local_h, left : left + local_w].reshape(
                    local_h * local_w, *pixel_values.shape[1:]
                )
            )

        pooled: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(windows), self.local_encode_batch):
                active = windows[start : start + self.local_encode_batch]
                local_pixels = torch.cat(active, dim=0)
                local_grids = torch.tensor(
                    [[local_h, local_w]] * len(active),
                    device=pixel_values.device,
                    dtype=torch.int32,
                )
                encoded = model.extract_feature(local_pixels, local_grids)
                projected = [model.mlp1(item) for item in encoded]
                merged_h, merged_w = local_h // 2, local_w // 2
                for item in projected:
                    feature_grid = item.reshape(merged_h, merged_w, self.hidden_size)
                    quadrants = F.adaptive_avg_pool2d(
                        feature_grid.permute(2, 0, 1).unsqueeze(0).float(), (2, 2)
                    )
                    pooled.append(
                        quadrants.squeeze(0).permute(1, 2, 0).reshape(4, self.hidden_size)
                    )
        return torch.stack(pooled, dim=0).to(dtype=pixel_values.dtype)

    def prepare_training_memory(
        self,
        model: nn.Module,
        pixel_values: torch.Tensor,
        image_grid_hws: Any,
        context: RelayContext,
    ) -> None:
        centers = list(context.center_targets)
        selected = self._select_indices(len(centers), self.max_memory_addresses)
        points = [centers[index] for index in selected]
        self._memory_slots = self._extract_local_slots(
            model, pixel_values, image_grid_hws, points
        )
        if self._memory_slots is not None:
            self._memory_points = torch.tensor(
                points,
                device=self._memory_slots.device,
                dtype=torch.float32,
            ).div_(1000.0)
        reverse = {source_index: memory_index for memory_index, source_index in enumerate(selected)}
        self._target_indices = tuple(reverse.get(index, -1) for index in range(len(centers)))
        self._coverage = None

    def prepare_inference_memory(
        self,
        model: nn.Module,
        pixel_values: torch.Tensor,
        image_grid_hws: Any,
    ) -> None:
        points = self._pending_points
        self._pending_points = ()
        self._memory_slots = self._extract_local_slots(
            model, pixel_values, image_grid_hws, points
        )
        if self._memory_slots is None:
            self._memory_points = None
        else:
            self._memory_points = torch.tensor(
                points,
                device=self._memory_slots.device,
                dtype=torch.float32,
            ).div_(1000.0)
        self._coverage = None
        self._inference_position_key = ()
        self._last_inference_key = None

    def _inference_context(self, input_ids: torch.Tensor | None) -> RelayContext | None:
        if input_ids is None or input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            return None
        ids = input_ids[0]
        mask_id = int(self.token_ids["default_mask_token_id"])
        valid_predecessors = {
            int(self.token_ids["ref_end_token_id"]),
            int(self.token_ids["box_end_token_id"]),
        }
        rows: list[tuple[int, int, int, int, int, int]] = []
        for start in range(max(0, int(ids.numel()) - 18), max(0, int(ids.numel()) - 5)):
            if start + 6 > int(ids.numel()) or int(ids[start]) not in valid_predecessors:
                continue
            if bool(ids[start + 1 : start + 6].eq(mask_id).all()):
                rows.append(tuple(range(start, start + 6)))
        if not rows:
            return None
        # Generation may retain the preceding masked PBD block in the short
        # look-back window. Only the newest block is active; re-routing older
        # blocks corrupts cached geometry and advances coverage more than once.
        return RelayContext((rows[-1],), (), self._stored_image_columns, int(ids.numel()))

    def _model_pre_hook(self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if self._training_context:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        position_ids = kwargs.get("position_ids")
        if isinstance(position_ids, torch.Tensor):
            self._inference_position_key = tuple(
                int(value) for value in position_ids.detach().reshape(-1).tolist()
            )
        else:
            self._inference_position_key = ()
        self._context = self._inference_context(input_ids)
        self._aux_loss = None
        return None

    def _route(self, hidden_states: torch.Tensor, context: RelayContext) -> torch.Tensor:
        rows = tuple(row for row in context.rows if max(row) < int(hidden_states.shape[1]))
        slots = self._memory_slots
        points = self._memory_points
        if not rows or slots is None or points is None or int(slots.shape[0]) == 0:
            return hidden_states

        # Training provides one address for every supervised box (up to the
        # configured cap), so it never applies local memory after all entries
        # have accumulated one unit of coverage. Mirror that support at
        # inference: once the finite address memory is consumed, leave later
        # PBD blocks to the native decoder instead of recycling stale evidence.
        if (
            not self._training_context
            and self._coverage is not None
            and float(self._coverage.float().sum()) >= float(slots.shape[0]) - 1e-4
        ):
            self.last_diagnostics = {
                "memory_addresses": float(slots.shape[0]),
                "memory_slots": float(slots.shape[0] * slots.shape[1]),
                "memory_exhausted": 1.0,
                "memory_gate": float(torch.sigmoid(self.lane_gate_logits).mean().detach()),
                "memory_coverage_strength": float(self.coverage_strength().detach()),
            }
            return hidden_states

        self.route_call_count += 1
        route_dtype = self.query_proj.weight.dtype
        slots = slots.to(device=hidden_states.device, dtype=route_dtype)
        points = points.to(device=hidden_states.device, dtype=torch.float32)
        address_base = slots.mean(dim=1)
        keys = F.normalize(self.key_proj(address_base).float(), dim=-1)
        keys = F.normalize(keys + self.position_mlp(points.to(route_dtype)).float(), dim=-1)
        slot_values = self.value_down(slots).float()
        coverage = (
            torch.zeros(int(slots.shape[0]), device=hidden_states.device, dtype=torch.float32)
            if self._coverage is None or int(self._coverage.numel()) != int(slots.shape[0])
            else self._coverage.to(device=hidden_states.device, dtype=torch.float32)
        )
        strength = self.coverage_strength().float()
        gates = torch.sigmoid(self.lane_gate_logits).to(hidden_states.dtype)
        updated = hidden_states.clone()
        route_losses: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        selected_probabilities: list[torch.Tensor] = []

        for row_index, row in enumerate(rows):
            anchor = hidden_states[0, row[0]].to(route_dtype)
            query = F.normalize(self.query_proj(anchor).float(), dim=-1)
            logits = torch.matmul(keys, query) / self.temperature - strength * coverage
            probabilities = logits.softmax(dim=-1)
            selected_probabilities.append(probabilities)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            entropies.append(entropy)

            target_index = (
                self._target_indices[row_index]
                if self._training_context and row_index < len(self._target_indices)
                else -1
            )
            if target_index >= 0:
                target = torch.tensor([target_index], device=logits.device, dtype=torch.long)
                route_losses.append(F.cross_entropy(logits.unsqueeze(0), target))
                coverage_update = F.one_hot(target[0], int(slots.shape[0])).float()
            else:
                coverage_update = probabilities.detach()

            selected_slots = torch.einsum("k,ksd->sd", probabilities, slot_values)
            lane_query = query.to(selected_slots.dtype).unsqueeze(0) + self.lane_queries.float()
            slot_scores = torch.einsum("ld,sd->ls", lane_query, selected_slots)
            slot_attention = slot_scores.softmax(dim=-1)
            lane_memory = torch.einsum("ls,sd->ld", slot_attention, selected_slots)
            delta = self.value_up(F.silu(lane_memory.to(route_dtype)))
            delta = F.layer_norm(delta.float(), (self.hidden_size,)).to(hidden_states.dtype)
            for lane, position in enumerate(row[1:5]):
                updated[0, position] = updated[0, position] + gates[lane] * delta[lane]
            coverage = coverage + coverage_update

        self._aux_loss = torch.stack(route_losses).mean() if route_losses else None
        self._coverage = coverage.detach()
        max_entropy = math.log(max(2, int(slots.shape[0])))
        mean_entropy = torch.stack(entropies).mean() if entropies else hidden_states.new_zeros(())
        peak = (
            torch.stack([item.max() for item in selected_probabilities]).mean()
            if selected_probabilities
            else hidden_states.new_zeros(())
        )
        self.last_diagnostics = {
            "memory_addresses": float(slots.shape[0]),
            "memory_slots": float(slots.shape[0] * slots.shape[1]),
            "memory_route_loss": 0.0
            if self._aux_loss is None
            else float(self._aux_loss.detach()),
            "memory_route_peak": float(peak.detach()),
            "memory_route_entropy_fraction": float((mean_entropy / max_entropy).detach()),
            "memory_gate": float(gates.mean().detach()),
            "memory_coverage_strength": float(strength.detach()),
            "memory_exhausted": 0.0,
        }
        return updated

    def _layer_pre_hook(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ):
        context = self._context
        if context is None or not context.rows:
            return None
        # Cached generation presents each PBD block as the same local six-token
        # tensor. Absolute position IDs distinguish consecutive generated boxes.
        inference_key = (self._inference_position_key, tuple(context.rows))
        prior_coverage = self._coverage
        repeated = not self._training_context and self._last_inference_key == inference_key
        updated = self._route(args[0], context)
        if not self._training_context:
            if repeated:
                # Repeated cached forwards for the same block may reuse the
                # route, but must not consume the address twice.
                self._coverage = prior_coverage
            self._last_inference_key = inference_key
        return (updated,) + args[1:], kwargs

    def finish_forward(self) -> None:
        self._context = None
        self._training_context = False
        self._aux_loss = None
        self._memory_slots = None
        self._memory_points = None
        self._target_indices = ()
        self._coverage = None
        self._inference_position_key = ()
        self._last_inference_key = None

    def clear_context(self) -> None:
        self.finish_forward()
        self._pending_points = ()

    def install(self, model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("Adaptive memory hooks are already installed")
        causal = base_causal_lm(model.language_model)
        layers = causal.model.layers
        if not 0 <= self.relay_layer < len(layers):
            raise ValueError(f"relay_layer={self.relay_layer} outside 0..{len(layers)-1}")
        self._handles.append(
            causal.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        )
        self._handles.append(
            layers[self.relay_layer].register_forward_pre_hook(
                self._layer_pre_hook, with_kwargs=True
            )
        )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def load_adaptive_memory_adapter(model: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "adaptive_memory":
        raise ValueError(f"Not an adaptive-memory checkpoint: {config.get('arm')!r}")
    configure_lora_only(model, int(config["lora_rank"]))
    controller = AdaptivePointMemoryPBD(
        hidden_size=int(model.config.text_config.hidden_size),
        relay_layer=int(config["relay_layer"]),
        token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
        memory_dim=int(config.get("memory_dim", 128)),
        local_patch_grid=int(config.get("local_patch_grid", 12)),
        max_memory_addresses=int(config.get("max_memory_addresses", 64)),
        local_encode_batch=int(config.get("local_encode_batch", 8)),
        temperature=float(config.get("temperature", 0.10)),
        initial_gate_logit=float(config.get("initial_gate_logit", -3.0)),
        initial_coverage_strength=float(config.get("initial_coverage_strength", 1.0)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
