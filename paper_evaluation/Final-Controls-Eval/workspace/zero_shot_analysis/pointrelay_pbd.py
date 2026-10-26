"""PointRelay-PBD pilot utilities for LocateAnything.

This module intentionally leaves the released model code unchanged.  It installs
forward hooks on Qwen decoder layers for a small LoRA screening experiment and
stores only adapter/controller tensors.  The pilot is not the final leak-free
architecture: it tests whether an explicit, differentiable center relay helps
before committing to a larger model rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_INDEX = -100


def base_causal_lm(language_model: nn.Module) -> nn.Module:
    base = language_model.get_base_model() if hasattr(language_model, "get_base_model") else language_model
    if not hasattr(base, "model") or not hasattr(base, "lm_head"):
        raise RuntimeError("Cannot resolve Qwen decoder and LM head")
    return base


def qwen_lora_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    values = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "lora_" in name and "language_model" in name
    ]
    if not values:
        raise RuntimeError("No Qwen LoRA parameters were found")
    return values


def configure_lora_only(model: nn.Module, rank: int) -> dict[str, int]:
    model.requires_grad_(False)
    model.wrap_llm_lora(r=int(rank), lora_alpha=2 * int(rank), lora_dropout=0.05)
    model.requires_grad_(False)
    values = qwen_lora_parameters(model)
    for _name, parameter in values:
        parameter.requires_grad_(True)
    return {
        "rank": int(rank),
        "tensors": len(values),
        "parameters": sum(parameter.numel() for _name, parameter in values),
    }


@dataclass
class RelayContext:
    rows: tuple[tuple[int, int, int, int, int, int], ...]
    center_targets: tuple[tuple[int, int], ...]
    image_columns: tuple[int, ...]
    query_length: int
    box_targets: tuple[tuple[int, int, int, int], ...] = ()


def geometry_context_from_training(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    token_ids: Mapping[str, int],
) -> RelayContext:
    if input_ids.ndim != 2 or labels.ndim != 2 or input_ids.shape != labels.shape:
        raise ValueError("PointRelay pilot expects matching [1, sequence] ids and labels")
    if input_ids.shape[0] != 1:
        raise ValueError("PointRelay pilot currently supports batch size one")
    ids = input_ids[0]
    target = labels[0]
    mask_id = int(token_ids["default_mask_token_id"])
    box_start = int(token_ids["box_start_token_id"])
    box_end = int(token_ids["box_end_token_id"])
    coord_start = int(token_ids["coord_start_token_id"])
    coord_end = int(token_ids["coord_end_token_id"])
    rows: list[tuple[int, int, int, int, int, int]] = []
    centers: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for start in range(max(0, int(ids.numel()) - 6)):
        if start + 6 >= int(target.numel()):
            break
        if not bool(ids[start + 1 : start + 6].eq(mask_id).all()):
            continue
        block = target[start + 1 : start + 7]
        if int(block.numel()) != 6:
            continue
        values = [int(value) for value in block.tolist()]
        if values[0] != box_start or values[5] != box_end:
            continue
        if not all(coord_start <= value <= coord_end for value in values[1:5]):
            continue
        x1, y1, x2, y2 = [value - coord_start for value in values[1:5]]
        cx = max(0, min(1000, int(round((x1 + x2) / 2.0))))
        cy = max(0, min(1000, int(round((y1 + y2) / 2.0))))
        rows.append(tuple(range(start, start + 6)))
        centers.append((cx, cy))
        boxes.append((x1, y1, x2, y2))
    image_index = int(token_ids["image_token_index"])
    image_columns = tuple(int(value) for value in ids.eq(image_index).nonzero(as_tuple=False).flatten().tolist())
    return RelayContext(tuple(rows), tuple(centers), image_columns, int(ids.numel()), tuple(boxes))


class PointRelayController(nn.Module):
    """Intermediate point supervision and optional explicit center relay."""

    def __init__(
        self,
        *,
        hidden_size: int,
        relay_layer: int,
        token_ids: Mapping[str, int],
        mode: str,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in {"point_aux", "pointrelay"}:
            raise ValueError(f"Unsupported PointRelay mode: {mode}")
        self.hidden_size = int(hidden_size)
        self.relay_layer = int(relay_layer)
        self.mode = str(mode)
        self.temperature = float(temperature)
        self.token_ids = {str(key): int(value) for key, value in token_ids.items()}
        self.lane_embedding = nn.Parameter(torch.zeros(4, self.hidden_size))
        self.boundary_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 2 * self.hidden_size),
        )
        nn.init.zeros_(self.boundary_mlp[-1].weight)
        nn.init.zeros_(self.boundary_mlp[-1].bias)
        self._context: RelayContext | None = None
        self._stored_image_columns: tuple[int, ...] = ()
        self._training_context = False
        self._aux_loss: torch.Tensor | None = None
        self._handles: list[Any] = []
        # Keep this outside nn.Module registration. The released Qwen decoder is
        # frozen model state, not part of the compact PointRelay checkpoint.
        object.__setattr__(self, "_causal", None)

    @property
    def aux_loss(self) -> torch.Tensor | None:
        return self._aux_loss

    def set_training_context(self, context: RelayContext) -> None:
        self._context = context
        self._stored_image_columns = context.image_columns
        self._training_context = True
        self._aux_loss = None

    def clear_context(self) -> None:
        self._context = None
        self._training_context = False
        self._aux_loss = None

    def finish_forward(self) -> None:
        """Drop per-sample training state while retaining inference image columns."""
        self._context = None
        self._training_context = False
        self._aux_loss = None

    def _inference_context(self, input_ids: torch.Tensor | None) -> RelayContext | None:
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            return None
        ids = input_ids[0]
        image_index = int(self.token_ids["image_token_index"])
        image_columns = tuple(int(v) for v in ids.eq(image_index).nonzero(as_tuple=False).flatten().tolist())
        if image_columns:
            self._stored_image_columns = image_columns
        mask_id = int(self.token_ids["default_mask_token_id"])
        ref_end = int(self.token_ids["ref_end_token_id"])
        rows: list[tuple[int, int, int, int, int, int]] = []
        for start in range(max(0, int(ids.numel()) - 12), max(0, int(ids.numel()) - 5)):
            if start + 6 > int(ids.numel()):
                continue
            if int(ids[start]) == ref_end and bool(ids[start + 1 : start + 6].eq(mask_id).all()):
                rows.append(tuple(range(start, start + 6)))
        if not rows:
            return None
        return RelayContext(tuple(rows), (), self._stored_image_columns, int(ids.numel()))

    def _model_pre_hook(self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if self._training_context:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        self._context = self._inference_context(input_ids)
        self._aux_loss = None
        return None

    def _masks(self, base: torch.Tensor, context: RelayContext) -> tuple[torch.Tensor, torch.Tensor]:
        early = base.clone()
        post = base.clone()
        q_len = int(base.shape[-2])
        k_len = int(base.shape[-1])
        past = k_len - q_len
        neg = float("-inf")
        valid_image_columns = [value for value in context.image_columns if 0 <= value < k_len]
        for block in context.rows:
            if max(block) >= q_len:
                continue
            key_columns = [past + value for value in block if 0 <= past + value < k_len]
            blocked_peer_columns = [key_columns[index] for index in (0, 1, 2, 5) if index < len(key_columns)]
            for row in (block[3], block[4]):
                if valid_image_columns:
                    early[..., row, valid_image_columns] = neg
                if blocked_peer_columns:
                    early[..., row, blocked_peer_columns] = neg
            edge_columns = [key_columns[index] for index in (0, 5) if index < len(key_columns)]
            for row in block[1:5]:
                if edge_columns:
                    post[..., row, edge_columns] = neg
        return early, post

    def _relay(
        self,
        hidden_states: torch.Tensor,
        *,
        causal: nn.Module,
        context: RelayContext,
    ) -> torch.Tensor:
        rows = [block for block in context.rows if max(block) < hidden_states.shape[1]]
        if not rows:
            return hidden_states
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        coord_weight = causal.lm_head.weight[coord_start : coord_end + 1]
        embed_weight = causal.model.embed_tokens.weight[coord_start : coord_end + 1]
        point_hidden = torch.stack(
            [torch.stack((hidden_states[0, block[1]], hidden_states[0, block[2]]), dim=0) for block in rows],
            dim=0,
        )
        logits = F.linear(point_hidden, coord_weight).float() / self.temperature
        probabilities = logits.softmax(dim=-1)
        expected = torch.matmul(probabilities.to(embed_weight.dtype), embed_weight)
        if context.center_targets:
            targets = torch.tensor(context.center_targets[: len(rows)], device=logits.device, dtype=torch.long)
            self._aux_loss = 0.5 * (
                F.cross_entropy(logits[:, 0], targets[:, 0])
                + F.cross_entropy(logits[:, 1], targets[:, 1])
            )
        if self.mode != "pointrelay":
            return hidden_states
        updated = hidden_states.clone()
        boundary = self.boundary_mlp(expected.reshape(len(rows), -1)).reshape(len(rows), 2, self.hidden_size)
        mask_embedding = causal.model.embed_tokens.weight[int(self.token_ids["default_mask_token_id"])]
        for index, block in enumerate(rows):
            updated[0, block[1]] = expected[index, 0] + self.lane_embedding[0]
            updated[0, block[2]] = expected[index, 1] + self.lane_embedding[1]
            updated[0, block[3]] = mask_embedding + boundary[index, 0] + self.lane_embedding[2]
            updated[0, block[4]] = mask_embedding + boundary[index, 1] + self.lane_embedding[3]
        return updated

    def _layer_hook(self, layer_index: int):
        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            context = self._context
            if context is None or not context.rows:
                return None
            hidden_states = args[0]
            attention_mask = kwargs.get("attention_mask")
            if self.mode == "pointrelay" and attention_mask is not None and attention_mask.ndim == 4:
                early, post = self._masks(attention_mask, context)
                kwargs["attention_mask"] = early if layer_index < self.relay_layer else post
            if layer_index == self.relay_layer:
                hidden_states = self._relay(hidden_states, causal=self._causal, context=context)
                args = (hidden_states,) + args[1:]
                return args, kwargs
            return None

        return hook

    def install(self, model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("PointRelay hooks are already installed")
        causal = base_causal_lm(model.language_model)
        object.__setattr__(self, "_causal", causal)
        layers = causal.model.layers
        if not 0 <= self.relay_layer < len(layers):
            raise ValueError(f"relay_layer={self.relay_layer} outside 0..{len(layers)-1}")
        self._handles.append(causal.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True))
        indices: Sequence[int] = (self.relay_layer,) if self.mode == "point_aux" else range(len(layers))
        for index in indices:
            self._handles.append(layers[index].register_forward_pre_hook(self._layer_hook(index), with_kwargs=True))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        object.__setattr__(self, "_causal", None)


class DecoderScalarPointRelay:
    """Parameter-free point relay inserted between two frozen Qwen blocks.

    The only trainable tensors in this experiment are Qwen LoRA matrices.  At
    the relay boundary, q1/q2 are decoded with the frozen coordinate rows of
    the native LM head, reduced to two scalar expectations, and re-embedded
    with the frozen coordinate-token table.  The complete coordinate
    distributions and unrestricted hidden states do not cross the boundary.

    ``point_aux`` computes the intermediate center loss without changing the
    hidden states. ``pointrelay`` additionally performs the scalar relay and
    resets q3/q4 to the released ``<text_mask>`` embedding.  The released PBD
    attention mask, six-slot grammar, MoonViT, projector, embeddings and LM
    head are otherwise unchanged.
    """

    def __init__(
        self,
        *,
        relay_layer: int,
        token_ids: Mapping[str, int],
        mode: str,
        temperature: float = 1.0,
    ) -> None:
        if mode not in {"point_aux", "pointrelay"}:
            raise ValueError(f"Unsupported decoder relay mode: {mode}")
        self.relay_layer = int(relay_layer)
        self.mode = str(mode)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.token_ids = {str(key): int(value) for key, value in token_ids.items()}
        self._context: RelayContext | None = None
        self._stored_image_columns: tuple[int, ...] = ()
        self._training_context = False
        self._aux_loss: torch.Tensor | None = None
        self._causal: nn.Module | None = None
        self._handles: list[Any] = []
        self.last_scalar_points: torch.Tensor | None = None
        self.relay_call_count = 0

    @property
    def aux_loss(self) -> torch.Tensor | None:
        return self._aux_loss

    def set_training_context(self, context: RelayContext) -> None:
        self._context = context
        self._stored_image_columns = context.image_columns
        self._training_context = True
        self._aux_loss = None
        self.last_scalar_points = None

    def clear_context(self) -> None:
        self._context = None
        self._training_context = False
        self._aux_loss = None
        self.last_scalar_points = None

    def finish_forward(self) -> None:
        self._context = None
        self._training_context = False
        self._aux_loss = None

    def _inference_context(self, input_ids: torch.Tensor | None) -> RelayContext | None:
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            return None
        ids = input_ids[0]
        image_index = int(self.token_ids["image_token_index"])
        image_columns = tuple(int(v) for v in ids.eq(image_index).nonzero(as_tuple=False).flatten().tolist())
        if image_columns:
            self._stored_image_columns = image_columns
        mask_id = int(self.token_ids["default_mask_token_id"])
        ref_end = int(self.token_ids["ref_end_token_id"])
        rows: list[tuple[int, int, int, int, int, int]] = []
        lower = max(0, int(ids.numel()) - 12)
        upper = max(0, int(ids.numel()) - 5)
        for start in range(lower, upper):
            if start + 6 <= int(ids.numel()) and int(ids[start]) == ref_end:
                if bool(ids[start + 1 : start + 6].eq(mask_id).all()):
                    rows.append(tuple(range(start, start + 6)))
        if not rows:
            return None
        return RelayContext(tuple(rows), (), self._stored_image_columns, int(ids.numel()))

    def _model_pre_hook(self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if self._training_context:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        self._context = self._inference_context(input_ids)
        self._aux_loss = None
        return None

    def _scalar_coordinate_embedding(
        self,
        hidden: torch.Tensor,
        causal: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        coordinate_head = causal.lm_head.weight[coord_start : coord_end + 1]
        coordinate_embeddings = causal.model.embed_tokens.weight[coord_start : coord_end + 1]
        logits = F.linear(hidden, coordinate_head).float() / self.temperature
        probabilities = logits.softmax(dim=-1)
        values = torch.arange(
            coordinate_embeddings.shape[0], device=logits.device, dtype=probabilities.dtype
        )
        scalar = (probabilities * values).sum(dim=-1).clamp(0, coordinate_embeddings.shape[0] - 1)

        # Re-expand only the scalar expectation. Linear interpolation between
        # adjacent frozen token embeddings cannot carry distribution shape.
        lower = scalar.floor().long()
        upper = (lower + 1).clamp(max=coordinate_embeddings.shape[0] - 1)
        fraction = (scalar - lower.to(scalar.dtype)).to(coordinate_embeddings.dtype).unsqueeze(-1)
        embedding = (
            coordinate_embeddings[lower] * (1.0 - fraction)
            + coordinate_embeddings[upper] * fraction
        )
        return logits, scalar, embedding

    def _relay(self, hidden_states: torch.Tensor, context: RelayContext) -> torch.Tensor:
        if self._causal is None:
            raise RuntimeError("DecoderScalarPointRelay is not installed")
        rows = [block for block in context.rows if max(block) < hidden_states.shape[1]]
        if not rows:
            return hidden_states
        self.relay_call_count += 1
        point_hidden = torch.stack(
            [torch.stack((hidden_states[0, block[1]], hidden_states[0, block[2]]), dim=0) for block in rows],
            dim=0,
        )
        logits, scalars, point_embeddings = self._scalar_coordinate_embedding(point_hidden, self._causal)
        self.last_scalar_points = scalars.detach()
        if context.center_targets:
            targets = torch.tensor(context.center_targets[: len(rows)], device=logits.device, dtype=torch.long)
            self._aux_loss = 0.5 * (
                F.cross_entropy(logits[:, 0], targets[:, 0])
                + F.cross_entropy(logits[:, 1], targets[:, 1])
            )
        if self.mode != "pointrelay":
            return hidden_states

        updated = hidden_states.clone()
        mask_embedding = self._causal.model.embed_tokens.weight[
            int(self.token_ids["default_mask_token_id"])
        ]
        for index, block in enumerate(rows):
            updated[0, block[1]] = point_embeddings[index, 0]
            updated[0, block[2]] = point_embeddings[index, 1]
            updated[0, block[3]] = mask_embedding
            updated[0, block[4]] = mask_embedding
        return updated

    def _layer_pre_hook(self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        context = self._context
        if context is None or not context.rows:
            return None
        hidden_states = self._relay(args[0], context)
        if self.mode == "pointrelay":
            return (hidden_states,) + args[1:], kwargs
        return None

    def install(self, model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("DecoderScalarPointRelay hooks are already installed")
        causal = base_causal_lm(model.language_model)
        layers = causal.model.layers
        if not 0 <= self.relay_layer < len(layers):
            raise ValueError(f"relay_layer={self.relay_layer} outside 0..{len(layers)-1}")
        self._causal = causal
        self._handles.append(causal.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True))
        self._handles.append(
            layers[self.relay_layer].register_forward_pre_hook(self._layer_pre_hook, with_kwargs=True)
        )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._causal = None


class SetValuedPointRelayPBD:
    """Set-valued point supervision and a strict scalar relay for native PBD.

    The released six-position HBB grammar is preserved. At ``relay_layer``,
    the first two coordinate lanes are interpreted as a latent point. The
    point is supervised by probability mass inside the target box instead of
    an arbitrary center coordinate. Relay modes reduce each 1,001-way
    coordinate distribution to one scalar expectation and re-embed only those
    two scalars with the frozen native coordinate embedding table.

    ``setpoint_aux`` adds only the set-valued loss. ``setpoint_relay`` also
    performs the scalar read/re-embedding under the released attention mask.
    ``causal_pointrelay`` additionally prevents boundary lanes from seeing
    visual evidence before the relay and restricts their post-relay visual
    access to a point-conditioned local window. The controller has no learned
    tensors; all trainable parameters remain Qwen LoRA matrices.
    """

    MODES = {"setpoint_aux", "setpoint_relay", "causal_pointrelay"}

    def __init__(
        self,
        *,
        relay_layer: int,
        token_ids: Mapping[str, int],
        mode: str,
        temperature: float = 1.0,
        local_radius_fraction: float = 0.25,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported set-valued relay mode: {mode}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < local_radius_fraction <= 0.5:
            raise ValueError("local_radius_fraction must be in (0, 0.5]")
        self.relay_layer = int(relay_layer)
        self.mode = str(mode)
        self.temperature = float(temperature)
        self.local_radius_fraction = float(local_radius_fraction)
        self.token_ids = {str(key): int(value) for key, value in token_ids.items()}
        self._context: RelayContext | None = None
        self._stored_image_columns: tuple[int, ...] = ()
        self._image_grid_hw: tuple[int, int] | None = None
        self._training_context = False
        self._aux_loss: torch.Tensor | None = None
        self._causal: nn.Module | None = None
        self._handles: list[Any] = []
        self._local_image_columns: dict[tuple[int, ...], tuple[int, ...]] = {}
        self.last_scalar_points: torch.Tensor | None = None
        self.last_diagnostics: dict[str, float] = {}
        self.relay_call_count = 0
        self.mask_call_count = 0

    @property
    def aux_loss(self) -> torch.Tensor | None:
        return self._aux_loss

    def _set_grid(self, image_grid_hws: Any, expected_tokens: int | None = None) -> None:
        values = torch.as_tensor(image_grid_hws, dtype=torch.long).reshape(-1, 2)
        if int(values.shape[0]) != 1:
            raise ValueError("Set-valued PointRelay supports exactly one image per record")
        patch_h, patch_w = int(values[0, 0]), int(values[0, 1])
        if patch_h % 2 or patch_w % 2:
            raise ValueError(f"PointRelay requires an even 2x2 merge grid, got {(patch_h, patch_w)}")
        grid = (patch_h // 2, patch_w // 2)
        if expected_tokens is not None and grid[0] * grid[1] != int(expected_tokens):
            raise ValueError(f"PointRelay image-grid mismatch: grid={grid} tokens={expected_tokens}")
        self._image_grid_hw = grid

    def set_training_context(self, context: RelayContext, image_grid_hws: Any | None = None) -> None:
        self._context = context
        self._stored_image_columns = context.image_columns
        self._training_context = True
        self._aux_loss = None
        self._local_image_columns = {}
        self.last_scalar_points = None
        self.last_diagnostics = {}
        if image_grid_hws is not None:
            self._set_grid(image_grid_hws, len(context.image_columns))

    def set_inference_image_layout(self, input_ids: torch.Tensor, image_grid_hws: Any) -> None:
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("Set-valued PointRelay inference expects input_ids with shape [1, sequence]")
        columns = tuple(
            int(value)
            for value in input_ids[0]
            .eq(int(self.token_ids["image_token_index"]))
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        if not columns:
            raise ValueError("Set-valued PointRelay could not find image placeholders")
        self._stored_image_columns = columns
        self._set_grid(image_grid_hws, len(columns))
        self._context = None
        self._training_context = False
        self._aux_loss = None
        self._local_image_columns = {}
        self.last_scalar_points = None
        self.last_diagnostics = {}

    def clear_context(self) -> None:
        self._context = None
        self._training_context = False
        self._aux_loss = None
        self._local_image_columns = {}
        self.last_scalar_points = None
        self.last_diagnostics = {}

    def finish_forward(self) -> None:
        self._context = None
        self._training_context = False
        self._aux_loss = None
        self._local_image_columns = {}

    def _inference_context(self, input_ids: torch.Tensor | None) -> RelayContext | None:
        if input_ids is None or input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            return None
        ids = input_ids[0]
        image_index = int(self.token_ids["image_token_index"])
        image_columns = tuple(
            int(value) for value in ids.eq(image_index).nonzero(as_tuple=False).flatten().tolist()
        )
        if image_columns:
            self._stored_image_columns = image_columns
        mask_id = int(self.token_ids["default_mask_token_id"])
        valid_predecessors = {
            int(self.token_ids["ref_end_token_id"]),
            int(self.token_ids["box_end_token_id"]),
        }
        rows: list[tuple[int, int, int, int, int, int]] = []
        lower = max(0, int(ids.numel()) - 18)
        upper = max(0, int(ids.numel()) - 5)
        for start in range(lower, upper):
            if start + 6 > int(ids.numel()) or int(ids[start]) not in valid_predecessors:
                continue
            if bool(ids[start + 1 : start + 6].eq(mask_id).all()):
                rows.append(tuple(range(start, start + 6)))
        if not rows:
            return None
        return RelayContext(tuple(rows), (), self._stored_image_columns, int(ids.numel()))

    def _model_pre_hook(self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if self._training_context:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        self._context = self._inference_context(input_ids)
        self._aux_loss = None
        self._local_image_columns = {}
        self.last_scalar_points = None
        self.last_diagnostics = {}
        return None

    def _coordinate_distribution(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._causal is None:
            raise RuntimeError("Set-valued PointRelay is not installed")
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        coordinate_head = self._causal.lm_head.weight[coord_start : coord_end + 1]
        coordinate_embeddings = self._causal.model.embed_tokens.weight[coord_start : coord_end + 1]
        normalized = self._causal.model.norm(hidden)
        logits = F.linear(normalized, coordinate_head).float() / self.temperature
        probabilities = logits.softmax(dim=-1)
        values = torch.arange(
            coordinate_embeddings.shape[0], device=logits.device, dtype=probabilities.dtype
        )
        scalar = (probabilities * values).sum(dim=-1).clamp(0, coordinate_embeddings.shape[0] - 1)
        lower = scalar.floor().long()
        upper = (lower + 1).clamp(max=coordinate_embeddings.shape[0] - 1)
        fraction = (scalar - lower.to(scalar.dtype)).to(coordinate_embeddings.dtype).unsqueeze(-1)
        embedding = coordinate_embeddings[lower] * (1.0 - fraction) + coordinate_embeddings[upper] * fraction
        return logits, scalar, embedding

    @staticmethod
    def _interval_mass_loss(logits: torch.Tensor, intervals: Sequence[tuple[int, int]]) -> torch.Tensor:
        log_probabilities = logits.log_softmax(dim=-1)
        losses = []
        maximum = int(logits.shape[-1] - 1)
        for index, (lower, upper) in enumerate(intervals):
            lo = max(0, min(maximum, int(min(lower, upper))))
            hi = max(0, min(maximum, int(max(lower, upper))))
            losses.append(-torch.logsumexp(log_probabilities[index, lo : hi + 1], dim=-1))
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()

    def _point_loss_and_diagnostics(
        self,
        logits: torch.Tensor,
        scalars: torch.Tensor,
        boxes: Sequence[tuple[int, int, int, int]],
    ) -> torch.Tensor:
        selected = list(boxes[: int(logits.shape[0])])
        x_intervals = [(box[0], box[2]) for box in selected]
        y_intervals = [(box[1], box[3]) for box in selected]
        loss = 0.5 * (
            self._interval_mass_loss(logits[:, 0], x_intervals)
            + self._interval_mass_loss(logits[:, 1], y_intervals)
        )
        with torch.no_grad():
            target = torch.tensor(selected, device=scalars.device, dtype=scalars.dtype)
            lower = target[:, :2]
            upper = target[:, 2:]
            minimum = torch.minimum(lower, upper)
            maximum = torch.maximum(lower, upper)
            below = (minimum - scalars).clamp_min(0)
            above = (scalars - maximum).clamp_min(0)
            outside = below + above
            inside = ((scalars >= minimum) & (scalars <= maximum)).all(dim=-1)
            probabilities = logits.softmax(dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            max_entropy = torch.log(torch.tensor(float(logits.shape[-1]), device=logits.device))
            self.last_diagnostics = {
                "point_inside_box_rate": float(inside.float().mean()),
                "point_interval_error_0_1000": float(outside.mean()),
                "point_entropy_fraction": float((entropy / max_entropy).mean()),
            }
        return loss

    def _point_local_columns(self, scalars: torch.Tensor, rows: Sequence[tuple[int, ...]]) -> None:
        if self._image_grid_hw is None or not self._stored_image_columns:
            if self.mode == "causal_pointrelay":
                raise RuntimeError("Causal PointRelay requires a configured image grid")
            return
        grid_h, grid_w = self._image_grid_hw
        if grid_h * grid_w != len(self._stored_image_columns):
            raise RuntimeError("PointRelay image grid and image columns disagree")
        radius_y = max(1, int(round(self.local_radius_fraction * grid_h)))
        radius_x = max(1, int(round(self.local_radius_fraction * grid_w)))
        counts = []
        for block, point in zip(rows, scalars.detach()):
            center_x = int(round(float(point[0]) / 1000.0 * max(1, grid_w - 1)))
            center_y = int(round(float(point[1]) / 1000.0 * max(1, grid_h - 1)))
            x0, x1 = max(0, center_x - radius_x), min(grid_w - 1, center_x + radius_x)
            y0, y1 = max(0, center_y - radius_y), min(grid_h - 1, center_y + radius_y)
            indices = [row * grid_w + col for row in range(y0, y1 + 1) for col in range(x0, x1 + 1)]
            columns = tuple(self._stored_image_columns[index] for index in indices)
            self._local_image_columns[tuple(block)] = columns
            counts.append(len(columns))
        if counts:
            self.last_diagnostics["local_image_tokens"] = float(sum(counts) / len(counts))

    def _relay(self, hidden_states: torch.Tensor, context: RelayContext) -> torch.Tensor:
        if self._causal is None:
            raise RuntimeError("Set-valued PointRelay is not installed")
        rows = [block for block in context.rows if max(block) < hidden_states.shape[1]]
        if not rows:
            return hidden_states
        self.relay_call_count += 1
        point_hidden = torch.stack(
            [torch.stack((hidden_states[0, block[1]], hidden_states[0, block[2]]), dim=0) for block in rows],
            dim=0,
        )
        logits, scalars, point_embeddings = self._coordinate_distribution(point_hidden)
        self.last_scalar_points = scalars.detach()
        if context.box_targets:
            self._aux_loss = self._point_loss_and_diagnostics(logits, scalars, context.box_targets)
        if self.mode == "setpoint_aux":
            return hidden_states
        self._point_local_columns(scalars, rows)
        updated = hidden_states.clone()
        mask_embedding = self._causal.model.embed_tokens.weight[
            int(self.token_ids["default_mask_token_id"])
        ]
        for index, block in enumerate(rows):
            updated[0, block[1]] = point_embeddings[index, 0]
            updated[0, block[2]] = point_embeddings[index, 1]
            updated[0, block[3]] = mask_embedding
            updated[0, block[4]] = mask_embedding
        return updated

    def _causal_mask(
        self,
        base: torch.Tensor,
        context: RelayContext,
        *,
        before_relay: bool,
    ) -> torch.Tensor:
        if base.ndim != 4:
            raise ValueError(f"Causal PointRelay requires a 4D SDPA mask, got {tuple(base.shape)}")
        mask = base.clone()
        q_len, k_len = int(mask.shape[-2]), int(mask.shape[-1])
        past = k_len - q_len
        negative = float("-inf")
        image_columns = [column for column in context.image_columns if 0 <= column < k_len]
        for block in context.rows:
            if max(block) >= q_len:
                continue
            keys = [past + position for position in block]
            if before_relay:
                forbidden_point = [keys[index] for index in (0, 3, 4, 5)]
                for row in block[1:3]:
                    mask[..., row, forbidden_point] = negative
                for row in block[3:5]:
                    original = base[..., row, :].clone()
                    mask[..., row, :] = negative
                    for key in keys[3:5]:
                        mask[..., row, key] = original[..., key]
            else:
                local = set(self._local_image_columns.get(tuple(block), ()))
                forbidden_image = [column for column in image_columns if column not in local]
                wrapper_keys = [keys[0], keys[5]]
                for row in block[1:5]:
                    if forbidden_image:
                        mask[..., row, forbidden_image] = negative
                    mask[..., row, wrapper_keys] = negative
        self.mask_call_count += 1
        return mask

    def _layer_hook(self, layer_index: int):
        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            context = self._context
            if context is None or not context.rows:
                return None
            hidden_states = args[0]
            if layer_index == self.relay_layer:
                hidden_states = self._relay(hidden_states, context)
            if self.mode == "causal_pointrelay":
                attention_mask = kwargs.get("attention_mask")
                if attention_mask is None:
                    raise RuntimeError("Causal PointRelay did not receive the native attention mask")
                kwargs["attention_mask"] = self._causal_mask(
                    attention_mask,
                    context,
                    before_relay=layer_index < self.relay_layer,
                )
            if layer_index == self.relay_layer and self.mode != "setpoint_aux":
                return (hidden_states,) + args[1:], kwargs
            if self.mode == "causal_pointrelay":
                return args, kwargs
            return None

        return hook

    def install(self, model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("Set-valued PointRelay hooks are already installed")
        causal = base_causal_lm(model.language_model)
        layers = causal.model.layers
        if not 0 <= self.relay_layer < len(layers):
            raise ValueError(f"relay_layer={self.relay_layer} outside 0..{len(layers)-1}")
        self._causal = causal
        self._handles.append(causal.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True))
        indices: Sequence[int] = range(len(layers)) if self.mode == "causal_pointrelay" else (self.relay_layer,)
        for index in indices:
            self._handles.append(layers[index].register_forward_pre_hook(self._layer_hook(index), with_kwargs=True))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._causal = None


class PointRoutePBDController(nn.Module):
    """Soft latent-point routing for the native six-position PBD block.

    The context position that predicts ``<box>`` acts as an instance query.  It
    produces a supervised distribution over the *existing* image tokens.  A
    confidence-gated visual glimpse is then added residually to the four native
    coordinate lanes.  No hidden state is replaced and the released output
    vocabulary, PBD attention mask, and box grammar remain unchanged.

    During cached generation, image hidden states at the routing depth are
    captured once during prefill and reused by later MTP box blocks.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        relay_layer: int,
        token_ids: Mapping[str, int],
        mode: str,
        point_dim: int = 128,
        temperature: float = 0.10,
        target_sigma: float = 0.85,
        initial_gate_logit: float = -3.0,
    ) -> None:
        super().__init__()
        if mode not in {"pointroute_aux", "pointroute"}:
            raise ValueError(f"Unsupported PointRoute mode: {mode}")
        if point_dim <= 0 or hidden_size <= 0:
            raise ValueError("PointRoute dimensions must be positive")
        if temperature <= 0 or target_sigma <= 0:
            raise ValueError("PointRoute temperature and target sigma must be positive")
        self.hidden_size = int(hidden_size)
        self.relay_layer = int(relay_layer)
        self.mode = str(mode)
        self.point_dim = int(point_dim)
        self.temperature = float(temperature)
        self.target_sigma = float(target_sigma)
        self.token_ids = {str(key): int(value) for key, value in token_ids.items()}

        self.query_proj = nn.Linear(self.hidden_size, self.point_dim, bias=False)
        self.key_proj = nn.Linear(self.hidden_size, self.point_dim, bias=False)
        self.glimpse_down = nn.Linear(self.hidden_size, self.point_dim, bias=False)
        self.glimpse_up = nn.Linear(self.point_dim, self.hidden_size, bias=False)
        self.lane_gate_logits = nn.Parameter(torch.full((4,), float(initial_gate_logit)))
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.glimpse_down.weight)
        nn.init.normal_(self.glimpse_up.weight, mean=0.0, std=1e-3)

        self._context: RelayContext | None = None
        self._training_context = False
        self._stored_image_columns: tuple[int, ...] = ()
        self._image_grid_hw: tuple[int, int] | None = None
        self._cached_image_hidden: torch.Tensor | None = None
        self._handles: list[Any] = []
        self._aux_loss: torch.Tensor | None = None
        self.last_diagnostics: dict[str, float] = {}
        self.route_call_count = 0

    @property
    def aux_loss(self) -> torch.Tensor | None:
        return self._aux_loss

    def _set_grid(self, image_grid_hws: Any, expected_tokens: int | None = None) -> None:
        values = torch.as_tensor(image_grid_hws, dtype=torch.long).reshape(-1, 2)
        if int(values.shape[0]) != 1:
            raise ValueError("PointRoute pilot supports exactly one image per record")
        patch_h, patch_w = (int(values[0, 0]), int(values[0, 1]))
        if patch_h % 2 or patch_w % 2:
            raise ValueError(f"PointRoute requires an even 2x2 merge grid, got {(patch_h, patch_w)}")
        grid = (patch_h // 2, patch_w // 2)
        if expected_tokens is not None and grid[0] * grid[1] != int(expected_tokens):
            raise ValueError(
                f"PointRoute image-grid mismatch: grid={grid} tokens={expected_tokens}"
            )
        self._image_grid_hw = grid

    def set_training_context(self, context: RelayContext, image_grid_hws: Any | None = None) -> None:
        self._context = context
        self._stored_image_columns = context.image_columns
        self._training_context = True
        self._cached_image_hidden = None
        self._aux_loss = None
        self.last_diagnostics = {}
        if image_grid_hws is not None:
            self._set_grid(image_grid_hws, len(context.image_columns))

    def set_inference_image_layout(self, input_ids: torch.Tensor, image_grid_hws: Any) -> None:
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("PointRoute inference expects input_ids with shape [1, sequence]")
        columns = tuple(
            int(value)
            for value in input_ids[0]
            .eq(int(self.token_ids["image_token_index"]))
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        if not columns:
            raise ValueError("PointRoute could not find image placeholders during inference prefill")
        self._stored_image_columns = columns
        self._set_grid(image_grid_hws, len(columns))
        self._cached_image_hidden = None
        self._context = None
        self._training_context = False
        self._aux_loss = None
        self.last_diagnostics = {}

    def clear_context(self) -> None:
        self._context = None
        self._training_context = False
        self._cached_image_hidden = None
        self._aux_loss = None
        self.last_diagnostics = {}

    def finish_forward(self) -> None:
        self._context = None
        self._training_context = False
        self._aux_loss = None

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
        return RelayContext(tuple(rows), (), self._stored_image_columns, int(ids.numel()))

    def _model_pre_hook(self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if self._training_context:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        self._context = self._inference_context(input_ids)
        self._aux_loss = None
        self.last_diagnostics = {}
        return None

    def _image_hidden(self, hidden_states: torch.Tensor, context: RelayContext) -> torch.Tensor | None:
        valid = [column for column in context.image_columns if 0 <= column < hidden_states.shape[1]]
        if valid and len(valid) == len(context.image_columns):
            image_hidden = hidden_states[0, valid]
            if not self._training_context:
                self._cached_image_hidden = image_hidden.detach()
            return image_hidden
        if self._cached_image_hidden is not None:
            return self._cached_image_hidden.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return None

    def _gaussian_targets(
        self,
        centers: Sequence[tuple[int, int]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if self._image_grid_hw is None:
            raise RuntimeError("PointRoute image grid was not configured")
        grid_h, grid_w = self._image_grid_hw
        ys = torch.arange(grid_h, device=device, dtype=torch.float32)
        xs = torch.arange(grid_w, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        targets = []
        for cx, cy in centers:
            target_x = float(cx) / 1000.0 * max(1, grid_w - 1)
            target_y = float(cy) / 1000.0 * max(1, grid_h - 1)
            squared = (xx - target_x).square() + (yy - target_y).square()
            heatmap = torch.exp(-0.5 * squared / (self.target_sigma**2)).reshape(-1)
            targets.append(heatmap / heatmap.sum().clamp_min(1e-12))
        return torch.stack(targets, dim=0)

    def _route(self, hidden_states: torch.Tensor, context: RelayContext) -> torch.Tensor:
        rows = [block for block in context.rows if max(block) < hidden_states.shape[1]]
        if not rows:
            return hidden_states
        image_hidden = self._image_hidden(hidden_states, context)
        if image_hidden is None or int(image_hidden.shape[0]) == 0:
            return hidden_states
        if self._image_grid_hw is None or self._image_grid_hw[0] * self._image_grid_hw[1] != image_hidden.shape[0]:
            raise RuntimeError(
                f"PointRoute lacks a valid image grid for {int(image_hidden.shape[0])} image tokens"
            )

        self.route_call_count += 1
        anchors = torch.stack([hidden_states[0, block[0]] for block in rows], dim=0)
        route_dtype = self.query_proj.weight.dtype
        queries = F.normalize(self.query_proj(anchors.to(route_dtype)).float(), dim=-1)
        keys = F.normalize(self.key_proj(image_hidden.to(route_dtype)).float(), dim=-1)
        logits = torch.matmul(queries, keys.transpose(0, 1)) / self.temperature
        probabilities = logits.softmax(dim=-1)

        if context.center_targets:
            centers = context.center_targets[: len(rows)]
            targets = self._gaussian_targets(centers, device=logits.device)
            self._aux_loss = -(targets * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
            grid_h, grid_w = self._image_grid_hw
            ys = torch.arange(grid_h, device=logits.device, dtype=torch.float32)
            xs = torch.arange(grid_w, device=logits.device, dtype=torch.float32)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            expected_x = (probabilities * xx.reshape(1, -1)).sum(dim=-1)
            expected_y = (probabilities * yy.reshape(1, -1)).sum(dim=-1)
            target_x = torch.tensor([value[0] for value in centers], device=logits.device).float()
            target_y = torch.tensor([value[1] for value in centers], device=logits.device).float()
            target_x = target_x / 1000.0 * max(1, grid_w - 1)
            target_y = target_y / 1000.0 * max(1, grid_h - 1)
            normalized_mae = 0.5 * (
                (expected_x - target_x).abs().mean() / max(1, grid_w - 1)
                + (expected_y - target_y).abs().mean() / max(1, grid_h - 1)
            )
            self.last_diagnostics = {
                "point_mae_0_1000": float((1000.0 * normalized_mae).detach()),
            }

        if self.mode != "pointroute":
            return hidden_states

        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(float(probabilities.shape[-1]), device=entropy.device))
        certainty = (1.0 - entropy / max_entropy.clamp_min(1e-6)).clamp(0.0, 1.0)
        glimpse = torch.matmul(probabilities.to(route_dtype), image_hidden.to(route_dtype))
        delta = self.glimpse_up(F.silu(self.glimpse_down(glimpse)))
        delta = F.layer_norm(delta.float(), (self.hidden_size,)).to(hidden_states.dtype)
        gates = torch.sigmoid(self.lane_gate_logits).to(hidden_states.dtype)
        updated = hidden_states.clone()
        for index, block in enumerate(rows):
            scale = certainty[index].to(hidden_states.dtype)
            for lane, position in enumerate(block[1:5]):
                updated[0, position] = updated[0, position] + gates[lane] * scale * delta[index]
        self.last_diagnostics.update(
            {
                "point_certainty": float(certainty.mean().detach()),
                "route_gate": float(gates.mean().detach()),
            }
        )
        return updated

    def _layer_pre_hook(self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        hidden_states = args[0]
        context = self._context
        if context is None or not context.rows:
            if not self._training_context and self._stored_image_columns:
                valid = [column for column in self._stored_image_columns if column < hidden_states.shape[1]]
                if len(valid) == len(self._stored_image_columns):
                    self._cached_image_hidden = hidden_states[0, valid].detach()
            return None
        updated = self._route(hidden_states, context)
        if self.mode == "pointroute":
            return (updated,) + args[1:], kwargs
        return None

    def install(self, model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("PointRoute hooks are already installed")
        causal = base_causal_lm(model.language_model)
        layers = causal.model.layers
        if not 0 <= self.relay_layer < len(layers):
            raise ValueError(f"relay_layer={self.relay_layer} outside 0..{len(layers)-1}")
        self._handles.append(causal.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True))
        self._handles.append(
            layers[self.relay_layer].register_forward_pre_hook(self._layer_pre_hook, with_kwargs=True)
        )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def trainable_state(model: nn.Module, controller: nn.Module | None) -> dict[str, Any]:
    lora = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    }
    if not lora:
        raise RuntimeError("Refusing to save an empty LoRA state")
    return {
        "lora": lora,
        "controller": None if controller is None else {
            name: tensor.detach().cpu().clone() for name, tensor in controller.state_dict().items()
        },
    }


def restore_trainable_state(
    model: nn.Module,
    controller: nn.Module | None,
    payload: Mapping[str, Any],
) -> None:
    live = dict(model.named_parameters())
    saved = payload["lora"]
    missing = sorted(set(saved) - set(live))
    if missing:
        raise RuntimeError(f"Adapter keys are absent from live model: {missing[:3]}")
    with torch.no_grad():
        for name, tensor in saved.items():
            parameter = live[name]
            if tuple(parameter.shape) != tuple(tensor.shape):
                raise RuntimeError(f"Adapter shape mismatch for {name}")
            parameter.copy_(tensor.to(parameter.device, parameter.dtype))
    controller_state = payload.get("controller")
    if controller is None and controller_state:
        raise RuntimeError("Checkpoint contains PointRelay weights but no controller was created")
    if controller is not None:
        if not controller_state:
            raise RuntimeError("PointRelay checkpoint lacks controller weights")
        controller.load_state_dict(controller_state, strict=True)


def load_pilot_adapter(model: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    configure_lora_only(model, int(config["lora_rank"]))
    arm = str(config["arm"])
    controller = None
    if arm == "pointrelay":
        controller = PointRelayController(
            hidden_size=int(model.config.text_config.hidden_size),
            relay_layer=int(config["relay_layer"]),
            token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
            mode="pointrelay",
            temperature=float(config.get("temperature", 1.0)),
        ).to(next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        controller.install(model)
    elif arm == "decoder_pointrelay":
        controller = DecoderScalarPointRelay(
            relay_layer=int(config["relay_layer"]),
            token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
            mode="pointrelay",
            temperature=float(config.get("temperature", 1.0)),
        )
        controller.install(model)
    elif arm in {"pointroute_aux", "pointroute"}:
        controller = PointRoutePBDController(
            hidden_size=int(model.config.text_config.hidden_size),
            relay_layer=int(config["relay_layer"]),
            token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
            mode=arm,
            point_dim=int(config.get("point_dim", 128)),
            temperature=float(config.get("temperature", 0.10)),
            target_sigma=float(config.get("target_sigma", 0.85)),
            initial_gate_logit=float(config.get("initial_gate_logit", -3.0)),
        )
        if "controller_learning_rate" in config:
            controller = controller.to(next(model.parameters()).device)
        else:
            # Legacy v3 checkpoints were optimized and saved in BF16. Preserve
            # that precision for an exact low-learning-rate control replay.
            controller = controller.to(
                next(model.parameters()).device,
                dtype=next(model.parameters()).dtype,
            )
        controller.install(model)
    elif arm in {"setpoint_relay", "causal_pointrelay"}:
        controller = SetValuedPointRelayPBD(
            relay_layer=int(config["relay_layer"]),
            token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
            mode=arm,
            temperature=float(config.get("temperature", 1.0)),
            local_radius_fraction=float(config.get("local_radius_fraction", 0.25)),
        )
        controller.install(model)
    state_controller = controller if isinstance(controller, nn.Module) else None
    restore_trainable_state(model, state_controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
