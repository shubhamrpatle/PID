"""Fine-token evidence readers for a controlled LocateAnything detection pilot.

The pilot preserves LocateAnything's released MoonViT -> 2x2 merger -> projector
-> Qwen path.  A lightweight side reader observes MoonViT tokens immediately
before the 2x2 merger and adds a zero-initialized residual glimpse to the four
native PBD coordinate lanes.  ``fine_static`` always reads the full fine-token
grid.  ``fine_residual`` additionally suppresses regions covered by earlier
boxes in the same generated ``<ref>`` group.

This is intentionally a screening implementation.  It tests fine visual access
and geometry write-back without changing the released tokenizer, PBD grammar,
LM head, or model files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .pointrelay_pbd import (
    base_causal_lm,
    configure_lora_only,
    geometry_context_from_training,
    restore_trainable_state,
)


@dataclass
class EvidenceContext:
    rows: tuple[tuple[int, int, int, int, int, int], ...]
    box_targets: tuple[tuple[int, int, int, int], ...]
    group_ids: tuple[int, ...]


def evidence_context_from_training(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    token_ids: Mapping[str, int],
) -> EvidenceContext:
    """Extract ordered native HBB blocks and their ``<ref>`` groups."""

    base = geometry_context_from_training(input_ids, labels, token_ids=token_ids)
    ids = input_ids[0]
    ref_end = int(token_ids["ref_end_token_id"])
    group = -1
    groups: list[int] = []
    for row in base.rows:
        if group < 0 or int(ids[row[0]]) == ref_end:
            group += 1
        groups.append(group)
    if len(base.rows) != len(base.box_targets) or len(base.rows) != len(groups):
        raise RuntimeError("Evidence context lost alignment between rows, boxes, and groups")
    return EvidenceContext(base.rows, base.box_targets, tuple(groups))


class ResidualEvidencePBD(nn.Module):
    """Query-conditioned readout over pre-merge MoonViT tokens.

    During training, each native six-position HBB block is supervised to attend
    to its GT box support.  The residual arm masks support already explained by
    earlier GT boxes in the same reference group.  During generation, the same
    state is updated from the model's previously emitted box.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        vision_hidden_size: int,
        relay_layer: int,
        token_ids: Mapping[str, int],
        mode: str,
        evidence_dim: int = 128,
        temperature: float = 0.10,
        residual_floor: float = 0.02,
    ) -> None:
        super().__init__()
        if mode not in {"fine_static", "fine_residual"}:
            raise ValueError(f"Unsupported evidence mode: {mode}")
        if min(hidden_size, vision_hidden_size, evidence_dim) <= 0:
            raise ValueError("Evidence dimensions must be positive")
        if temperature <= 0 or not 0 < residual_floor < 1:
            raise ValueError("Invalid evidence temperature or residual floor")

        self.hidden_size = int(hidden_size)
        self.vision_hidden_size = int(vision_hidden_size)
        self.relay_layer = int(relay_layer)
        self.mode = str(mode)
        self.evidence_dim = int(evidence_dim)
        self.temperature = float(temperature)
        self.residual_floor = float(residual_floor)
        self.token_ids = {str(key): int(value) for key, value in token_ids.items()}

        self.query_proj = nn.Linear(self.hidden_size, self.evidence_dim, bias=False)
        self.key_proj = nn.Linear(self.vision_hidden_size, self.evidence_dim, bias=False)
        self.value_down = nn.Linear(self.vision_hidden_size, self.evidence_dim, bias=False)
        self.value_up = nn.Linear(self.evidence_dim, self.hidden_size, bias=False)
        self.lane_gate_logits = nn.Parameter(torch.zeros(4))
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_down.weight)
        nn.init.normal_(self.value_up.weight, mean=0.0, std=1e-3)

        self._context: EvidenceContext | None = None
        self._training_context = False
        self._grid_hw: tuple[int, int] | None = None
        self._fine_tokens: torch.Tensor | None = None
        self._coverage: torch.Tensor | None = None
        self._handles: list[Any] = []
        self._aux_loss: torch.Tensor | None = None
        self.last_diagnostics: dict[str, float] = {}
        self.route_call_count = 0
        self.observed_box_count = 0
        self.observed_ref_count = 0
        object.__setattr__(self, "_causal", None)

    @property
    def aux_loss(self) -> torch.Tensor | None:
        return self._aux_loss

    def _set_grid(self, image_grid_hws: Any) -> None:
        values = torch.as_tensor(image_grid_hws, dtype=torch.long).reshape(-1, 2)
        if int(values.shape[0]) != 1:
            raise ValueError("Evidence pilot supports exactly one image per record")
        height, width = int(values[0, 0]), int(values[0, 1])
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid MoonViT grid: {(height, width)}")
        self._grid_hw = (height, width)

    def set_training_context(self, context: EvidenceContext, image_grid_hws: Any) -> None:
        self._context = context
        self._training_context = True
        self._set_grid(image_grid_hws)
        self._fine_tokens = None
        self._coverage = None
        self._aux_loss = None
        self.last_diagnostics = {}
        self.route_call_count = 0
        self.observed_box_count = 0
        self.observed_ref_count = 0

    def set_inference_image_layout(self, _input_ids: torch.Tensor, image_grid_hws: Any) -> None:
        self._context = None
        self._training_context = False
        self._set_grid(image_grid_hws)
        self._fine_tokens = None
        self._coverage = None
        self._aux_loss = None
        self.last_diagnostics = {}
        self.route_call_count = 0
        self.observed_box_count = 0
        self.observed_ref_count = 0

    def finish_forward(self) -> None:
        self._context = None
        self._training_context = False
        self._fine_tokens = None
        self._coverage = None
        self._aux_loss = None

    def _vision_hook(self, _module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        fine = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(fine, torch.Tensor) or fine.ndim != 2:
            raise RuntimeError("MoonViT encoder did not return [tokens, hidden] features")
        if self._grid_hw is None:
            raise RuntimeError("MoonViT grid was not configured before visual encoding")
        expected = self._grid_hw[0] * self._grid_hw[1]
        if int(fine.shape[0]) != expected:
            raise RuntimeError(f"Fine-token grid mismatch: features={fine.shape[0]} expected={expected}")
        self._fine_tokens = fine.detach()

    def _box_support(
        self,
        box: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if self._grid_hw is None:
            raise RuntimeError("Evidence grid is unavailable")
        height, width = self._grid_hw
        x1, y1, x2, y2 = [float(value) / 1000.0 for value in box]
        left, right = sorted((max(0.0, x1), min(1.0, x2)))
        top, bottom = sorted((max(0.0, y1), min(1.0, y2)))
        xs0 = torch.arange(width, device=device, dtype=dtype) / float(width)
        xs1 = (torch.arange(width, device=device, dtype=dtype) + 1.0) / float(width)
        ys0 = torch.arange(height, device=device, dtype=dtype) / float(height)
        ys1 = (torch.arange(height, device=device, dtype=dtype) + 1.0) / float(height)
        overlap_x = (torch.minimum(xs1, torch.tensor(right, device=device, dtype=dtype)) -
                     torch.maximum(xs0, torch.tensor(left, device=device, dtype=dtype))).clamp_min(0)
        overlap_y = (torch.minimum(ys1, torch.tensor(bottom, device=device, dtype=dtype)) -
                     torch.maximum(ys0, torch.tensor(top, device=device, dtype=dtype))).clamp_min(0)
        support = (overlap_y[:, None] * overlap_x[None, :]) * float(height * width)
        support = support.clamp(0.0, 1.0).reshape(-1)
        if float(support.sum()) <= 1e-8:
            cx = max(0, min(width - 1, int(((left + right) * 0.5) * width)))
            cy = max(0, min(height - 1, int(((top + bottom) * 0.5) * height)))
            support[cy * width + cx] = 1.0
        return support

    def observe_generated_block(self, out_type: str, tokens: Any) -> None:
        """Update inference memory from the block accepted by PBD decoding.

        LocateAnything slices ``input_ids`` to the current six-token MTP window
        once KV caching starts.  The emitted coordinates therefore cannot be
        reconstructed reliably in a decoder pre-hook and must be observed at
        the generation decision itself.
        """

        if self._training_context or self.mode != "fine_residual":
            return
        if out_type == "ref_object":
            self._coverage = None
            self.observed_ref_count += 1
        elif out_type == "coord_box":
            if isinstance(tokens, torch.Tensor):
                values = tokens.detach().reshape(-1).tolist()
            else:
                values = [int(value.item() if hasattr(value, "item") else value) for value in tokens]
            box_start = int(self.token_ids["box_start_token_id"])
            box_end = int(self.token_ids["box_end_token_id"])
            coord_start = int(self.token_ids["coord_start_token_id"])
            coord_end = int(self.token_ids["coord_end_token_id"])
            if len(values) != 6 or int(values[0]) != box_start or int(values[-1]) != box_end:
                return
            coordinates = [int(value) for value in values[1:5]]
            if not all(coord_start <= value <= coord_end for value in coordinates):
                return
            box = tuple(value - coord_start for value in coordinates)
            device = self._fine_tokens.device if self._fine_tokens is not None else self.lane_gate_logits.device
            support = self._box_support(box, device=device)
            self._coverage = support if self._coverage is None else torch.maximum(self._coverage, support)
            self.observed_box_count += 1
        else:
            return
        self.last_diagnostics["observed_box_count"] = float(self.observed_box_count)
        self.last_diagnostics["observed_ref_count"] = float(self.observed_ref_count)
        self.last_diagnostics["covered_token_fraction"] = (
            0.0 if self._coverage is None else float(self._coverage.gt(0).float().mean())
        )

    def _inference_context(self, input_ids: torch.Tensor | None) -> EvidenceContext | None:
        if input_ids is None or input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            return None
        ids = input_ids[0]
        mask_id = int(self.token_ids["default_mask_token_id"])
        ref_end = int(self.token_ids["ref_end_token_id"])
        box_end = int(self.token_ids["box_end_token_id"])
        rows: list[tuple[int, int, int, int, int, int]] = []
        for start in range(max(0, int(ids.numel()) - 18), max(0, int(ids.numel()) - 5)):
            if start + 6 > int(ids.numel()):
                continue
            if int(ids[start]) not in {ref_end, box_end}:
                continue
            if bool(ids[start + 1 : start + 6].eq(mask_id).all()):
                rows.append(tuple(range(start, start + 6)))
        if not rows:
            return None

        current = rows[-1]
        if self.mode == "fine_residual" and int(ids[current[0]]) == ref_end:
            self._coverage = None
        return EvidenceContext((current,), (), (0,))

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

    def _read_one(
        self,
        anchor: torch.Tensor,
        fine: torch.Tensor,
        coverage: torch.Tensor | None,
        target_box: Sequence[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        route_dtype = self.query_proj.weight.dtype
        query = F.normalize(self.query_proj(anchor.to(route_dtype)).float(), dim=-1)
        keys = F.normalize(self.key_proj(fine.to(route_dtype)).float(), dim=-1)
        logits = torch.matmul(keys, query) / self.temperature
        if coverage is not None:
            residual = (1.0 - coverage.to(logits.device)).clamp_min(self.residual_floor)
            logits = logits + residual.log()
        probabilities = logits.softmax(dim=-1)
        values = self.value_down(fine.to(route_dtype))
        glimpse = torch.matmul(probabilities.to(route_dtype), values)
        delta = self.value_up(F.silu(glimpse))
        delta = F.layer_norm(delta.float(), (self.hidden_size,))

        loss = None
        diagnostics: dict[str, torch.Tensor] = {}
        if target_box is not None:
            support = self._box_support(target_box, device=logits.device)
            target = support / support.sum().clamp_min(1e-8)
            loss = -(target * logits.log_softmax(dim=-1)).sum()
            diagnostics["evidence_target_mass"] = (probabilities * support.gt(0).float()).sum()
            if self._grid_hw is not None:
                height, width = self._grid_hw
                yy, xx = torch.meshgrid(
                    torch.arange(height, device=logits.device, dtype=torch.float32),
                    torch.arange(width, device=logits.device, dtype=torch.float32),
                    indexing="ij",
                )
                expected_x = (probabilities * xx.reshape(-1)).sum() / max(1, width - 1)
                expected_y = (probabilities * yy.reshape(-1)).sum() / max(1, height - 1)
                x1, y1, x2, y2 = [float(value) / 1000.0 for value in target_box]
                diagnostics["evidence_center_error"] = 0.5 * (
                    (expected_x - (x1 + x2) * 0.5).abs()
                    + (expected_y - (y1 + y2) * 0.5).abs()
                )
        return delta, loss, diagnostics

    def _route(self, hidden_states: torch.Tensor, context: EvidenceContext) -> torch.Tensor:
        rows = [row for row in context.rows if max(row) < hidden_states.shape[1]]
        if not rows or self._fine_tokens is None:
            return hidden_states
        self.route_call_count += 1
        fine = self._fine_tokens.to(device=hidden_states.device)
        if self._grid_hw is None or self._grid_hw[0] * self._grid_hw[1] != int(fine.shape[0]):
            raise RuntimeError("Fine evidence does not match the configured MoonViT grid")

        updated = hidden_states.clone()
        losses: list[torch.Tensor] = []
        masses: list[torch.Tensor] = []
        errors: list[torch.Tensor] = []
        coverage_by_group: dict[int, torch.Tensor] = {}
        gates = torch.tanh(self.lane_gate_logits).to(hidden_states.dtype)

        for index, row in enumerate(rows):
            group = context.group_ids[index] if index < len(context.group_ids) else 0
            if self._training_context:
                coverage = coverage_by_group.get(group) if self.mode == "fine_residual" else None
                target_box = context.box_targets[index]
            else:
                coverage = self._coverage if self.mode == "fine_residual" else None
                target_box = None
            delta, loss, diagnostics = self._read_one(
                hidden_states[0, row[0]], fine, coverage, target_box
            )
            for lane, position in enumerate(row[1:5]):
                updated[0, position] = updated[0, position] + gates[lane] * delta.to(hidden_states.dtype)
            if loss is not None:
                losses.append(loss)
                masses.append(diagnostics["evidence_target_mass"])
                errors.append(diagnostics["evidence_center_error"])
            if self._training_context and self.mode == "fine_residual":
                support = self._box_support(target_box, device=hidden_states.device)
                previous = coverage_by_group.get(group)
                coverage_by_group[group] = support if previous is None else torch.maximum(previous, support)

        self._aux_loss = torch.stack(losses).mean() if losses else None
        self.last_diagnostics = {
            "evidence_target_mass": float(torch.stack(masses).mean().detach()) if masses else 0.0,
            "evidence_center_error_0_1000": float(1000.0 * torch.stack(errors).mean().detach()) if errors else 0.0,
            "evidence_gate": float(gates.abs().mean().detach()),
            "fine_visual_tokens": float(fine.shape[0]),
            "covered_token_fraction": 0.0 if self._coverage is None else float(self._coverage.gt(0).float().mean()),
            "observed_box_count": float(self.observed_box_count),
            "observed_ref_count": float(self.observed_ref_count),
        }
        return updated

    def _layer_pre_hook(self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        context = self._context
        if context is None or not context.rows:
            return None
        updated = self._route(args[0], context)
        return (updated,) + args[1:], kwargs

    def install(self, model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("Evidence hooks are already installed")
        causal = base_causal_lm(model.language_model)
        layers = causal.model.layers
        if not 0 <= self.relay_layer < len(layers):
            raise ValueError(f"relay_layer={self.relay_layer} outside 0..{len(layers)-1}")
        object.__setattr__(self, "_causal", causal)
        self._handles.append(model.vision_model.encoder.register_forward_hook(self._vision_hook))
        self._handles.append(causal.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True))
        self._handles.append(
            layers[self.relay_layer].register_forward_pre_hook(self._layer_pre_hook, with_kwargs=True)
        )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        object.__setattr__(self, "_causal", None)


def load_evidence_adapter(model: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a compact evidence-reader plus its matched Qwen LoRA weights."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    configure_lora_only(model, int(config["lora_rank"]))
    controller = ResidualEvidencePBD(
        hidden_size=int(model.config.text_config.hidden_size),
        vision_hidden_size=int(model.config.vision_config.hidden_size),
        relay_layer=int(config["relay_layer"]),
        token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
        mode=str(config["arm"]),
        evidence_dim=int(config.get("evidence_dim", 128)),
        temperature=float(config.get("temperature", 0.10)),
        residual_floor=float(config.get("residual_floor", 0.02)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
