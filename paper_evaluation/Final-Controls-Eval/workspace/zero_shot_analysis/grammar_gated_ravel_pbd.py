"""Native-grammar-gated, history-isolated visual re-entry for LocateAnything.

The original RAVEL pilot writes retrieved visual evidence before an intermediate
Qwen layer.  During PBD inference that write can influence a speculative mask
window even when the native decoder intends to emit a reference or EOS block.
This variant preserves the decoder's block decision and writes evidence only to
the coordinate lanes of a block that Qwen itself classifies as geometry.

The write is applied after the final Qwen decoder layer.  Consequently it can
change the current coordinate logits, but it cannot enter that layer's KV cache
or create a second decoder stream.  MoonViT is still evaluated once per image.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .pointrelay_pbd import base_causal_lm
from .ravel_pbd import RavelPBDController


class GrammarGatedRavelPBDController(RavelPBDController):
    """Route fine visual evidence without changing native block/EOS decisions."""

    controller_kind = "native_grammar_gated_history_isolated"

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
        self._native_norm: nn.Module | None = None
        self._native_lm_head: nn.Module | None = None
        self._native_geometry_gate = 0.0
        self._native_block_token = -1
        self._native_frame_kind = "unknown"
        self.native_gate_temperature = 0.7

    def inference_contract(self) -> dict[str, Any]:
        contract = super().inference_contract()
        contract.update(
            {
                "method": "PIVR-LA native-grammar-gated single-pass",
                "history_isolated_visual_writeback": True,
                "native_block_decision_preserved": True,
                "native_pbd_frame_decision_preserved": True,
                "native_eos_termination": True,
                "soft_learned_geometry_continuation": False,
                "controller_kind": self.controller_kind,
            }
        )
        return contract

    def _reset_state(self) -> None:
        super()._reset_state()
        self._native_geometry_gate = 0.0
        self._native_block_token = -1
        self._native_frame_kind = "unknown"

    def _record_runtime_diagnostics(self) -> None:
        super()._record_runtime_diagnostics()
        self.last_diagnostics.update(
            {
                "pivr_native_geometry_gate": float(self._native_geometry_gate),
                "pivr_native_block_token": float(self._native_block_token),
                "pivr_native_frame_is_geometry": float(
                    self._native_frame_kind == "geometry"
                ),
                "pivr_history_isolated_writeback": 1.0,
            }
        )

    def _native_pbd_frame_kind(
        self, hidden_states: torch.Tensor, event: Any
    ) -> str:
        """Reproduce LA's pre-write PBD frame classification.

        Detection uses the released six-position ``decode_bbox_avg`` contract.
        Point and quadrilateral branches use conservative native-frame checks
        so visual write-back cannot create their opening/closing grammar.
        """

        if self._native_norm is None or self._native_lm_head is None:
            raise RuntimeError("Native Qwen grammar modules were not installed")
        with torch.no_grad():
            positions = list(event.row)
            hidden = hidden_states[0, positions]
            normalized = self._native_norm(hidden)
            logits = self._native_lm_head(normalized)
            probabilities = torch.softmax(
                logits.float() / self.native_gate_temperature, dim=-1
            )
        self._native_block_token = int(probabilities[0].argmax())
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        keep_k = min(5, int(probabilities.shape[-1]))

        def coordinates_are_valid(start: int, count: int) -> bool:
            if len(positions) < start + count:
                return False
            ids = probabilities[start : start + count].topk(keep_k, dim=-1).indices
            return bool(((ids >= coord_start) & (ids <= coord_end)).any(dim=-1).all())

        kind = "other"
        if event.geometry == "hbb" and len(positions) >= 6:
            box_start = int(self.token_ids["box_start_token_id"])
            box_end = int(self.token_ids["box_end_token_id"])
            none = int(self.token_ids["none_token_id"])
            null = int(self.token_ids["null_token_id"])
            im_end = int(self.token_ids["im_end_token_id"])
            is_empty = bool(
                probabilities[0, box_start] >= 0.7
                and probabilities[1, none] > 0.2
                and probabilities[2, box_end] > 0.2
                and probabilities[3, null] > 0.1
                and probabilities[4, null] > 0.1
            )
            end_score = probabilities[5, [box_end, null, im_end]].sum()
            if is_empty:
                kind = "empty"
            elif bool(end_score >= 0.2) and coordinates_are_valid(1, 4):
                kind = "geometry"
        elif event.geometry == "point" and len(positions) >= 4:
            box_start = int(self.token_ids["box_start_token_id"])
            box_end = int(self.token_ids["box_end_token_id"])
            if (
                bool(probabilities[0, box_start] >= 0.35)
                and bool(probabilities[3, box_end] >= 0.15)
                and coordinates_are_valid(1, 2)
            ):
                kind = "geometry"
        elif event.geometry == "obb" and len(positions) >= 10:
            quad_start = self.token_ids.get("quad_start_token_id")
            quad_end = self.token_ids.get("quad_end_token_id")
            if (
                quad_start is not None
                and quad_end is not None
                and int(probabilities[0].argmax()) == int(quad_start)
                and int(probabilities[9].argmax()) == int(quad_end)
                and coordinates_are_valid(1, 8)
            ):
                kind = "geometry"
        self._native_frame_kind = kind
        self._native_geometry_gate = float(kind == "geometry")
        return kind

    def _layer_post_hook(
        self,
        _module: nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        if self._context is None or not self._context.events:
            return output
        if not isinstance(output, tuple) or not output or not isinstance(output[0], torch.Tensor):
            raise RuntimeError("Unexpected Qwen decoder-layer output for PIVR write-back")
        hidden_states = output[0]
        if not self._training_context:
            event = self._context.events[-1]
            if max(event.row) >= int(hidden_states.shape[1]):
                return output
            if self._native_pbd_frame_kind(hidden_states, event) != "geometry":
                self._record_runtime_diagnostics()
                return output
            # The native decoder has accepted geometry as the next block type.
            # The controller may now refine only its coordinate lanes.
            self._geometry_route_gate = 1.0
        updated = self._route(hidden_states, self._context)
        return (updated, *output[1:])

    def observe_generated_block(self, block_type: str, tokens: Any) -> None:
        super().observe_generated_block(block_type, tokens)
        if block_type in {"coord_box", "point_box"}:
            # Always inspect the next native block decision. The grammar gate,
            # rather than a count threshold or learned slot cap, decides whether
            # another geometry write is allowed.
            self._geometry_route_gate = 1.0
        self._record_runtime_diagnostics()

    def install(self, model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("PIVR hooks are already installed")
        causal = base_causal_lm(model.language_model)
        layers = causal.model.layers
        if self.relay_layer != len(layers) - 1:
            raise ValueError(
                "History-isolated PIVR must attach after the final Qwen layer; "
                f"expected relay_layer={len(layers) - 1}, got {self.relay_layer}"
            )
        # These are frozen aliases used only to inspect the native frame.  Use
        # object-level assignment so they are not registered a second time in
        # the controller state dict and serialized into every adapter.
        object.__setattr__(self, "_native_norm", causal.model.norm)
        object.__setattr__(self, "_native_lm_head", causal.lm_head)
        self._handles.append(model.vision_model.encoder.register_forward_hook(self._vision_hook))
        self._handles.append(
            causal.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        )
        self._handles.append(
            layers[self.relay_layer].register_forward_hook(
                self._layer_post_hook, with_kwargs=True
            )
        )
