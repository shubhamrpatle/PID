"""Semantic-aligned recurrent visual events for LocateAnything PBD.

The controller keeps LocateAnything's native autoregressive grammar and one
Qwen decoder.  It exposes spatial detail that is normally discarded by the
fixed 2x2 MoonViT merge by evaluating every overlapping 2x2 phase through the
released, frozen multimodal projector.  At a Qwen relay layer, these phase
tokens are aligned with the native image hidden states and queried by the
actual Qwen attention projections for the active semantic reference.

An accepted geometry block writes its soft visual event into recurrent
coverage.  The same decoder can therefore request another unresolved event or
stop through its ordinary token grammar.  There are no object slots, proposal
caps, crop calls, decoder replicas, or direct vocabulary-logit edits.
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
from .residual_set_reentry_pbd import (
    ResidualSetTransition,
    residual_set_transitions_from_training,
)


IGNORE_INDEX = -100


def overlapping_phase_windows(
    fine_tokens: torch.Tensor,
    grid_hw: tuple[int, int],
) -> torch.Tensor:
    """Return row-major TL/TR/BL/BR windows for every stride-one 2x2 phase."""

    height, width = grid_hw
    if fine_tokens.ndim != 2 or int(fine_tokens.shape[0]) != height * width:
        raise ValueError("Fine MoonViT tokens do not match the configured grid")
    if height < 2 or width < 2:
        raise ValueError("At least a 2x2 MoonViT grid is required")
    grid = fine_tokens.reshape(height, width, fine_tokens.shape[-1])
    windows = torch.stack(
        (grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]),
        dim=2,
    )
    return windows.reshape((height - 1) * (width - 1), -1)


def native_merge_windows(
    fine_tokens: torch.Tensor,
    grid_hw: tuple[int, int],
) -> torch.Tensor:
    """Reproduce MoonViT's released non-overlapping 2x2 merge exactly."""

    height, width = grid_hw
    if height % 2 or width % 2:
        raise ValueError(f"The released 2x2 merger requires an even grid, got {grid_hw}")
    if fine_tokens.ndim != 2 or int(fine_tokens.shape[0]) != height * width:
        raise ValueError("Fine MoonViT tokens do not match the configured grid")
    grid = fine_tokens.reshape(height // 2, 2, width // 2, 2, -1)
    return grid.permute(0, 2, 1, 3, 4).contiguous().reshape(
        (height // 2) * (width // 2), -1
    )


def interpolate_native_grid(
    values: torch.Tensor,
    native_hw: tuple[int, int],
    phase_hw: tuple[int, int],
) -> torch.Tensor:
    """Interpolate native merge-center values to all stride-one phase centers."""

    native_h, native_w = native_hw
    if values.ndim != 2 or int(values.shape[0]) != native_h * native_w:
        raise ValueError("Native values do not match their spatial grid")
    image = values.transpose(0, 1).reshape(1, values.shape[-1], native_h, native_w)
    resized = F.interpolate(
        image.float(), size=phase_hw, mode="bilinear", align_corners=True
    ).to(values.dtype)
    return resized.reshape(values.shape[-1], -1).transpose(0, 1).contiguous()


class SemanticAlignedEventReentryPBD(nn.Module):
    """Open-ended semantic event retrieval in LA's pretrained hidden space."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vision_hidden_size: int,
        relay_layer: int,
        token_ids: Mapping[str, int],
        phase_residual_dim: int = 256,
        event_dim: int = 256,
        route_temperature: float = 1.0,
        target_sigma_cells: float = 0.75,
        coverage_strength: float = 8.0,
        posterior_logit_scale: float = 4.0,
        posterior_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        if min(hidden_size, vision_hidden_size, phase_residual_dim, event_dim) <= 0:
            raise ValueError("Semantic event dimensions must be positive")
        if route_temperature <= 0 or target_sigma_cells <= 0 or coverage_strength <= 0:
            raise ValueError("Route, target, and coverage constants must be positive")
        if posterior_logit_scale <= 0 or not 0 < posterior_floor < 1:
            raise ValueError("Invalid posterior guidance constants")

        self.hidden_size = int(hidden_size)
        self.vision_hidden_size = int(vision_hidden_size)
        self.relay_layer = int(relay_layer)
        self.phase_residual_dim = int(phase_residual_dim)
        self.event_dim = int(event_dim)
        self.route_temperature = float(route_temperature)
        self.target_sigma_cells = float(target_sigma_cells)
        self.coverage_strength = float(coverage_strength)
        self.posterior_logit_scale = float(posterior_logit_scale)
        self.posterior_floor = float(posterior_floor)
        self.token_ids = {str(key): int(value) for key, value in token_ids.items()}

        # The projector residual adds only high-frequency phase information to
        # an interpolated Qwen image state.  The phase inputs themselves pass
        # through LA's frozen pretrained projector.
        self.phase_residual_down = nn.Linear(
            self.hidden_size, self.phase_residual_dim, bias=False
        )
        self.phase_residual_up = nn.Linear(
            self.phase_residual_dim, self.hidden_size, bias=False
        )
        self.event_query_down = nn.Linear(self.hidden_size, self.event_dim, bias=False)
        self.event_value_down = nn.Linear(self.hidden_size, self.event_dim, bias=False)
        self.event_up = nn.Linear(
            self.event_dim, 6 * self.hidden_size, bias=False
        )
        self.presence_head = nn.Sequential(
            nn.Linear(2 * self.event_dim + 1, self.event_dim),
            nn.GELU(),
            nn.LayerNorm(self.event_dim),
            nn.Linear(self.event_dim, 1),
        )
        nn.init.xavier_uniform_(self.phase_residual_down.weight)
        nn.init.normal_(self.phase_residual_up.weight, mean=0.0, std=1e-3)
        nn.init.xavier_uniform_(self.event_query_down.weight)
        nn.init.xavier_uniform_(self.event_value_down.weight)
        # Exact parent forward at initialization. Auxiliary route/presence
        # losses can learn before re-entry changes a decoder hidden state.
        nn.init.zeros_(self.event_up.weight)
        nn.init.xavier_uniform_(self.presence_head[0].weight)
        nn.init.zeros_(self.presence_head[0].bias)
        nn.init.zeros_(self.presence_head[-1].weight)
        nn.init.zeros_(self.presence_head[-1].bias)

        self._context: RelayContext | None = None
        self._transitions: tuple[ResidualSetTransition, ...] = ()
        self._training_context = False
        self._stored_image_columns: tuple[int, ...] = ()
        self._grid_hw: tuple[int, int] | None = None
        self._phase_positions: torch.Tensor | None = None
        self._fine_tokens: torch.Tensor | None = None
        self._phase_projected: torch.Tensor | None = None
        self._native_projected: torch.Tensor | None = None
        self._cached_memory: torch.Tensor | None = None
        self._coverage: torch.Tensor | None = None
        self._semantic_anchor: torch.Tensor | None = None
        self._pending_inference_event: torch.Tensor | None = None
        self._route_loss: torch.Tensor | None = None
        self._presence_loss: torch.Tensor | None = None
        self._handles: list[Any] = []
        self.last_diagnostics: dict[str, float] = {}
        self.route_call_count = 0
        self.geometry_writeback_count = 0
        object.__setattr__(self, "_host_model", None)
        object.__setattr__(self, "_relay_decoder_layer", None)

    @property
    def route_loss(self) -> torch.Tensor | None:
        return self._route_loss

    @property
    def presence_loss(self) -> torch.Tensor | None:
        return self._presence_loss

    def residual_strength(self) -> torch.Tensor:
        """Expose a common diagnostic without changing the native LM logits."""

        return self.event_up.weight.float().abs().mean()

    def _set_grid(self, image_grid_hws: Any) -> None:
        values = torch.as_tensor(image_grid_hws, dtype=torch.long).reshape(-1, 2)
        if int(values.shape[0]) != 1:
            raise ValueError("Semantic event re-entry supports one image per record")
        height, width = int(values[0, 0]), int(values[0, 1])
        if height < 2 or width < 2 or height % 2 or width % 2:
            raise ValueError(f"Expected an even MoonViT grid >=2, got {(height, width)}")
        self._grid_hw = (height, width)
        yy, xx = torch.meshgrid(
            (torch.arange(height - 1, dtype=torch.float32) + 1.0) / height,
            (torch.arange(width - 1, dtype=torch.float32) + 1.0) / width,
            indexing="ij",
        )
        self._phase_positions = torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def _reset_sample_state(self, *, preserve_fine: bool = False) -> None:
        if not preserve_fine:
            self._fine_tokens = None
            self._phase_projected = None
            self._native_projected = None
        self._cached_memory = None
        self._coverage = None
        self._semantic_anchor = None
        self._pending_inference_event = None
        self._route_loss = None
        self._presence_loss = None
        self.last_diagnostics = {}
        self.route_call_count = 0
        self.geometry_writeback_count = 0

    def set_training_context(
        self,
        context: RelayContext,
        image_grid_hws: Any,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> None:
        if input_ids is None or labels is None:
            raise ValueError("Semantic event training requires input_ids and labels")
        self._context = context
        self._transitions = residual_set_transitions_from_training(
            input_ids, labels, token_ids=self.token_ids
        )
        self._stored_image_columns = context.image_columns
        self._training_context = True
        self._set_grid(image_grid_hws)
        self._reset_sample_state()

    def set_inference_image_layout(
        self, input_ids: torch.Tensor, image_grid_hws: Any
    ) -> None:
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("Semantic event inference expects [1, sequence] input ids")
        columns = tuple(
            int(value)
            for value in input_ids[0]
            .eq(int(self.token_ids["image_token_index"]))
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        if not columns:
            raise ValueError("Image placeholders are absent from the inference prompt")
        self._stored_image_columns = columns
        self._context = None
        self._transitions = ()
        self._training_context = False
        self._set_grid(image_grid_hws)
        self._reset_sample_state()

    def prepare_pixel_field(self, _pixel_values: torch.Tensor, image_grid_hws: Any) -> None:
        # Kept for the common worker interface. Unlike older pilots, this path
        # uses only MoonViT and the released multimodal projector.
        self._set_grid(image_grid_hws)

    def _vision_hook(
        self, _module: nn.Module, _args: tuple[Any, ...], output: Any
    ) -> None:
        fine = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(fine, torch.Tensor) or fine.ndim != 2:
            raise RuntimeError("MoonViT encoder did not return [tokens, hidden]")
        if self._grid_hw is None or int(fine.shape[0]) != math.prod(self._grid_hw):
            raise RuntimeError("MoonViT fine tokens do not match image_grid_hws")
        self._fine_tokens = fine.detach()
        self._phase_projected = None
        self._native_projected = None
        self._cached_memory = None

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
        lower = max(0, int(ids.numel()) - 18)
        upper = max(0, int(ids.numel()) - 5)
        for start in range(lower, upper):
            if start + 6 > int(ids.numel()) or int(ids[start]) not in valid_predecessors:
                continue
            if bool(ids[start + 1 : start + 6].eq(mask_id).all()):
                rows.append(tuple(range(start, start + 6)))
        if not rows:
            return None
        predecessor = "ref" if int(ids[rows[-1][0]]) == int(
            self.token_ids["ref_end_token_id"]
        ) else "box"
        transition = ResidualSetTransition(
            row=rows[-1],
            predecessor=predecessor,
            reset_coverage=predecessor == "ref",
            target_box=None,
            remaining_boxes=(),
        )
        self._transitions = (transition,)
        return RelayContext((rows[-1],), (), self._stored_image_columns, int(ids.numel()), ())

    def _model_pre_hook(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if self._training_context:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        self._context = self._inference_context(input_ids)
        self._route_loss = None
        self._presence_loss = None
        return None

    def _projector_tokens(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._phase_projected is not None and self._native_projected is not None:
            return self._phase_projected, self._native_projected
        host = self._host_model
        if host is None or self._fine_tokens is None or self._grid_hw is None:
            raise RuntimeError("Semantic-aligned projector memory is unavailable")
        fine = self._fine_tokens
        phase_input = overlapping_phase_windows(fine, self._grid_hw)
        native_input = native_merge_windows(fine, self._grid_hw)
        with torch.no_grad():
            self._phase_projected = host.mlp1(phase_input)
            self._native_projected = host.mlp1(native_input)
        return self._phase_projected, self._native_projected

    def _build_memory(self, native_hidden: torch.Tensor) -> torch.Tensor:
        if self._cached_memory is not None:
            return self._cached_memory.to(native_hidden.device, native_hidden.dtype)
        if self._grid_hw is None:
            raise RuntimeError("Image grid is unavailable")
        height, width = self._grid_hw
        native_hw = (height // 2, width // 2)
        phase_hw = (height - 1, width - 1)
        if int(native_hidden.shape[0]) != math.prod(native_hw):
            raise RuntimeError(
                f"Native Qwen image states={native_hidden.shape[0]}, expected={math.prod(native_hw)}"
            )
        phase_projected, native_projected = self._projector_tokens()
        native_projected = native_projected.to(native_hidden.device, native_hidden.dtype)
        phase_projected = phase_projected.to(native_hidden.device, native_hidden.dtype)
        hidden_interp = interpolate_native_grid(native_hidden, native_hw, phase_hw)
        projector_interp = interpolate_native_grid(native_projected, native_hw, phase_hw)
        high_frequency = phase_projected - projector_interp
        adapter_dtype = self.phase_residual_down.weight.dtype
        adapted = self.phase_residual_up(
            F.gelu(self.phase_residual_down(high_frequency.to(adapter_dtype)))
        )
        memory = hidden_interp + adapted.to(hidden_interp.dtype)
        if not self._training_context:
            self._cached_memory = memory.detach()
        return memory

    def _local_rows(
        self, transitions: Sequence[ResidualSetTransition], sequence_length: int
    ) -> list[tuple[ResidualSetTransition, tuple[int, int, int, int, int, int]]]:
        values: list[tuple[ResidualSetTransition, tuple[int, int, int, int, int, int]]] = []
        context_length = self._context.query_length if self._context is not None else sequence_length
        offset = max(0, int(context_length) - int(sequence_length))
        for transition in transitions:
            if max(transition.row) < sequence_length:
                values.append((transition, transition.row))
                continue
            shifted = tuple(value - offset for value in transition.row)
            if min(shifted) >= 0 and max(shifted) < sequence_length:
                values.append((transition, shifted))
        return values

    def _center_support(
        self, box: Sequence[int], *, device: torch.device
    ) -> torch.Tensor:
        if self._phase_positions is None or self._grid_hw is None:
            raise RuntimeError("Phase positions are unavailable")
        positions = self._phase_positions.to(device=device)
        x1, y1, x2, y2 = (float(value) / 1000.0 for value in box)
        center = positions.new_tensor(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
        scale = positions.new_tensor((self._grid_hw[1], self._grid_hw[0]))
        distance = ((positions - center) * scale).square().sum(dim=-1)
        support = torch.exp(-0.5 * distance / (self.target_sigma_cells**2))
        if float(support.max()) < 1e-8:
            support[distance.argmin()] = 1.0
        return support / support.sum().clamp_min(1e-12)

    def _remaining_support(
        self, boxes: Sequence[Sequence[int]], *, device: torch.device
    ) -> torch.Tensor | None:
        if not boxes:
            return None
        supports = torch.stack([self._center_support(box, device=device) for box in boxes])
        union = supports.amax(dim=0)
        return union / union.max().clamp_min(1e-12)

    @staticmethod
    def _attention_content_scores(
        query_hidden: torch.Tensor,
        memory: torch.Tensor,
        layer: nn.Module,
    ) -> torch.Tensor:
        attention = layer.self_attn
        normalized_query = layer.input_layernorm(query_hidden.unsqueeze(0)).squeeze(0)
        normalized_memory = layer.input_layernorm(memory)
        query = attention.q_proj(normalized_query)
        keys = attention.k_proj(normalized_memory)
        heads = int(getattr(attention, "num_heads"))
        kv_heads = int(getattr(attention, "num_key_value_heads", heads))
        head_dim = int(getattr(attention, "head_dim", query.numel() // heads))
        query = query.reshape(heads, head_dim)
        keys = keys.reshape(keys.shape[0], kv_heads, head_dim)
        if kv_heads != heads:
            if heads % kv_heads:
                raise RuntimeError("Qwen attention heads are incompatible with KV heads")
            keys = keys.repeat_interleave(heads // kv_heads, dim=1)
        return torch.einsum("hd,nhd->n", query.float(), keys.float()) / math.sqrt(head_dim)

    def _native_image_hidden(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        columns = self._stored_image_columns
        valid = [value for value in columns if 0 <= value < int(hidden_states.shape[1])]
        if columns and len(valid) == len(columns):
            return hidden_states[0, valid]
        if self._cached_memory is not None:
            return None
        return None

    def _route_and_inject(
        self,
        hidden_states: torch.Tensor,
        local: Sequence[
            tuple[ResidualSetTransition, tuple[int, int, int, int, int, int]]
        ],
        memory: torch.Tensor,
    ) -> torch.Tensor:
        layer = self._relay_decoder_layer
        if layer is None:
            raise RuntimeError("Relay decoder layer is unavailable")
        if self._phase_positions is None:
            raise RuntimeError("Phase positions are unavailable")
        coverage = (
            torch.zeros(memory.shape[0], device=memory.device, dtype=torch.float32)
            if self._coverage is None or int(self._coverage.numel()) != int(memory.shape[0])
            else self._coverage.to(memory.device, dtype=torch.float32)
        )
        updated = hidden_states.clone()
        route_losses: list[torch.Tensor] = []
        positive_presence: list[torch.Tensor] = []
        negative_presence: list[torch.Tensor] = []
        target_masses: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []

        for transition, row in local:
            if transition.reset_coverage:
                coverage = torch.zeros_like(coverage)
                self._semantic_anchor = None
            anchor = updated[0, row[0]]
            if transition.predecessor == "ref" or self._semantic_anchor is None:
                self._semantic_anchor = anchor
            semantic = 0.5 * (anchor + self._semantic_anchor.to(anchor))
            scores = self._attention_content_scores(semantic, memory, layer)
            scores = scores / self.route_temperature - self.coverage_strength * coverage
            prior = scores.softmax(dim=-1)
            target = self._remaining_support(
                transition.remaining_boxes, device=memory.device
            )
            if target is not None:
                mass = (prior * target).sum().clamp_min(1e-12)
                route_losses.append(-mass.log())
                target_masses.append(mass)

            selection = prior
            if self._training_context and transition.target_box is not None:
                current = self._center_support(
                    transition.target_box, device=memory.device
                )
                posterior_scores = scores + self.posterior_logit_scale * torch.log(
                    current.clamp_min(self.posterior_floor)
                )
                selection = posterior_scores.softmax(dim=-1)
            selected = torch.matmul(selection.to(memory.dtype), memory)
            route_dtype = self.event_query_down.weight.dtype
            query_latent = self.event_query_down(semantic.to(route_dtype))
            value_latent = self.event_value_down(selected.to(route_dtype))
            covered = coverage.gt(0.05).float().mean().reshape(1).to(route_dtype)
            presence_logit = self.presence_head(
                torch.cat((query_latent, value_latent, covered), dim=-1)
            ).reshape(())
            label = 1.0 if target is not None else 0.0
            decision = F.binary_cross_entropy_with_logits(
                presence_logit, presence_logit.new_tensor(label)
            )
            (positive_presence if label else negative_presence).append(decision)
            gate = torch.sigmoid(presence_logit)
            event = self.event_up(F.gelu(query_latent + value_latent)).reshape(
                6, self.hidden_size
            )
            event = event.to(updated.dtype) * gate.to(updated.dtype)
            for lane, position in enumerate(row):
                updated[0, position] = updated[0, position] + event[lane]

            entropy = -(prior * prior.clamp_min(1e-12).log()).sum()
            entropies.append(entropy / math.log(max(2, int(prior.numel()))))
            gates.append(gate)
            if self._training_context and transition.target_box is not None:
                emitted = self._center_support(
                    transition.target_box, device=memory.device
                )
                coverage = torch.maximum(coverage, emitted)
            elif not self._training_context:
                self._pending_inference_event = selection.detach().float()

        groups = [
            torch.stack(values).mean()
            for values in (positive_presence, negative_presence)
            if values
        ]
        self._route_loss = torch.stack(route_losses).mean() if route_losses else None
        self._presence_loss = torch.stack(groups).mean() if groups else None
        self._coverage = coverage.detach()
        self.route_call_count += len(local)
        self.last_diagnostics.update(
            {
                "semantic_event_transitions": float(len(local)),
                "semantic_event_route_loss": 0.0
                if self._route_loss is None
                else float(self._route_loss.detach()),
                "semantic_event_presence_loss": 0.0
                if self._presence_loss is None
                else float(self._presence_loss.detach()),
                "semantic_event_target_mass": 0.0
                if not target_masses
                else float(torch.stack(target_masses).mean().detach()),
                "semantic_event_entropy": 0.0
                if not entropies
                else float(torch.stack(entropies).mean().detach()),
                "semantic_event_gate": 0.0
                if not gates
                else float(torch.stack(gates).mean().detach()),
                "semantic_event_coverage_fraction": float(
                    coverage.gt(0.05).float().mean().detach()
                ),
                "semantic_event_memory_tokens": float(memory.shape[0]),
                "semantic_event_writebacks": float(self.geometry_writeback_count),
            }
        )
        return updated

    def _layer_pre_hook(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        hidden_states = args[0]
        native_hidden = self._native_image_hidden(hidden_states)
        memory: torch.Tensor | None = None
        if native_hidden is not None:
            memory = self._build_memory(native_hidden)
        elif self._cached_memory is not None:
            memory = self._cached_memory.to(hidden_states.device, hidden_states.dtype)
        if memory is None:
            return None
        local = self._local_rows(self._transitions, int(hidden_states.shape[1]))
        if not local:
            return None
        updated = self._route_and_inject(hidden_states, local, memory)
        return (updated,) + args[1:], kwargs

    def observe_generated_block(self, out_type: str, _tokens: Any) -> None:
        """Commit a selected event only after native PBD accepts its geometry."""

        if self._training_context:
            return
        if out_type == "ref_object":
            self._coverage = None
            self._semantic_anchor = None
            self._pending_inference_event = None
        elif out_type in {"coord_box", "point_box"}:
            event = self._pending_inference_event
            if event is not None:
                event = event.to(
                    self._coverage.device if self._coverage is not None else event.device,
                    dtype=torch.float32,
                )
                self._coverage = (
                    event
                    if self._coverage is None
                    else torch.maximum(self._coverage.to(event), event)
                )
                self.geometry_writeback_count += 1
            self._pending_inference_event = None
        elif out_type in {"empty_box", "error_box", "im_end"}:
            self._pending_inference_event = None
        self.last_diagnostics["semantic_event_writebacks"] = float(
            self.geometry_writeback_count
        )

    def finish_forward(self) -> None:
        self._context = None
        self._transitions = ()
        self._training_context = False
        self._route_loss = None
        self._presence_loss = None
        self._phase_projected = None
        self._native_projected = None
        self._cached_memory = None
        self._fine_tokens = None

    def install(self, model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("Semantic event hooks are already installed")
        causal = base_causal_lm(model.language_model)
        layers = causal.model.layers
        if not 0 <= self.relay_layer < len(layers):
            raise ValueError(f"relay_layer={self.relay_layer} outside 0..{len(layers)-1}")
        object.__setattr__(self, "_host_model", model)
        object.__setattr__(self, "_relay_decoder_layer", layers[self.relay_layer])
        self._handles.append(
            model.vision_model.encoder.register_forward_hook(self._vision_hook)
        )
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
        object.__setattr__(self, "_host_model", None)
        object.__setattr__(self, "_relay_decoder_layer", None)


def semantic_aligned_event_reentry_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    controller: SemanticAlignedEventReentryPBD,
    *,
    route_loss_weight: float = 0.20,
    presence_loss_weight: float = 0.20,
    chunk: int = 128,
) -> torch.Tensor:
    """Native shifted token CE plus set-valued event amortization losses."""

    shifted_hidden = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
    shifted_labels = labels[:, 1:].reshape(-1)
    valid = shifted_labels.ne(IGNORE_INDEX)
    states = shifted_hidden[valid]
    targets = shifted_labels[valid]
    if int(targets.numel()) == 0:
        raise RuntimeError("Semantic event sample has no supervised tokens")
    total = hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, int(targets.numel()), int(chunk)):
        stop = min(int(targets.numel()), start + int(chunk))
        total = total + F.cross_entropy(
            F.linear(states[start:stop], weight).float(),
            targets[start:stop],
            reduction="sum",
        )
    loss = total / int(targets.numel())
    if controller.route_loss is not None:
        loss = loss + float(route_loss_weight) * controller.route_loss
    if controller.presence_loss is not None:
        loss = loss + float(presence_loss_weight) * controller.presence_loss
    return loss


def load_semantic_aligned_event_reentry_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "semantic_aligned_event_reentry":
        raise ValueError(f"Not a semantic-aligned event checkpoint: {config.get('arm')!r}")
    configure_lora_only(model, int(config["lora_rank"]))
    controller = SemanticAlignedEventReentryPBD(
        hidden_size=int(model.config.text_config.hidden_size),
        vision_hidden_size=int(model.config.vision_config.hidden_size),
        relay_layer=int(config.get("relay_layer", 18)),
        token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
        phase_residual_dim=int(config.get("phase_residual_dim", 256)),
        event_dim=int(config.get("event_dim", 256)),
        route_temperature=float(config.get("semantic_route_temperature", 1.0)),
        target_sigma_cells=float(config.get("target_sigma_cells", 0.75)),
        coverage_strength=float(config.get("semantic_coverage_strength", 8.0)),
        posterior_logit_scale=float(config.get("posterior_logit_scale", 4.0)),
        posterior_floor=float(config.get("posterior_floor", 1e-4)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
