"""Evidence-conditioned typed grammar for single-pass PIVR localization.

LocateAnything decides whether the next six-token PBD frame is geometry or an
explicit ``None`` before ordinary PIVR can write cached visual evidence. This
controller evaluates that decision as a two-state grammar lattice. Qwen's
native frame likelihood is combined with PIVR's learned visual-presence
likelihood, then the winning valid frame is decoded. Coordinates are still
selected from Qwen's coordinate logits after visual re-entry.

The controller retains one MoonViT encoding, one Qwen stream, and the exact
no-progress cycle guard. It has no object slots, count prediction, fixed object
cap, crop re-encoding, or free-form token repair.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .cycle_guarded_ravel_pbd import CycleGuardedRavelPBDController


class EvidenceTypedRavelPBDController(CycleGuardedRavelPBDController):
    """Fuse native and visual evidence inside a valid HBB/None grammar."""

    controller_kind = "evidence_typed_cycle_guarded"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._typed_frame_candidate = False
        self._typed_native_kind = "other"
        self._typed_geometry_choices = 0
        self._typed_empty_choices = 0
        self._typed_native_switches = 0

    def inference_contract(self) -> dict[str, Any]:
        contract = super().inference_contract()
        contract.update(
            {
                "method": "PIVR-LA evidence-typed cycle-guarded single-pass",
                "controller_kind": self.controller_kind,
                "evidence_conditioned_typed_grammar": True,
                "typed_states": ["hbb", "explicit_none"],
                "typed_decision": (
                    "native Qwen likelihood times learned visual-presence likelihood"
                ),
                "geometry_is_always_native_generated": True,
                "malformed_cross_state_frames_allowed": False,
            }
        )
        return contract

    def _reset_state(self) -> None:
        super()._reset_state()
        self._typed_frame_candidate = False
        self._typed_native_kind = "other"
        self._typed_geometry_choices = 0
        self._typed_empty_choices = 0
        self._typed_native_switches = 0

    def _layer_post_hook(
        self,
        _module: nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        if self._context is None or not self._context.events:
            return output
        if (
            not isinstance(output, tuple)
            or not output
            or not isinstance(output[0], torch.Tensor)
        ):
            raise RuntimeError("Unexpected Qwen decoder-layer output for typed PIVR")
        hidden_states = output[0]
        if self._training_context:
            return (self._route(hidden_states, self._context), *output[1:])

        event = self._context.events[-1]
        if max(event.row) >= int(hidden_states.shape[1]):
            return output
        kind = self._native_pbd_frame_kind(hidden_states, event)
        if kind not in {"geometry", "empty"}:
            self._typed_frame_candidate = False
            self._record_runtime_diagnostics()
            return output

        # Score both valid states after the same cached visual-evidence write.
        # Semantic-reference and EOS states never enter this branch.
        self._typed_frame_candidate = True
        self._typed_native_kind = kind
        self._geometry_route_gate = 1.0
        updated = self._route(hidden_states, self._context)
        return (updated, *output[1:])

    @staticmethod
    def _keep_only(
        logits: torch.Tensor,
        lane: int,
        token_ids: torch.Tensor,
    ) -> None:
        floor = torch.finfo(logits.dtype).min
        retained = logits[:, lane, token_ids].clone()
        logits[:, lane, :] = floor
        logits[:, lane, token_ids] = retained

    def _frame_log_scores(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(logits.shape[1]) < 6:
            raise RuntimeError("Typed HBB grammar requires the native six-lane PBD frame")
        logp = torch.log_softmax(logits[:, :6, :].float(), dim=-1)[0]
        box_start = int(self.token_ids["box_start_token_id"])
        box_end = int(self.token_ids["box_end_token_id"])
        none = int(self.token_ids["none_token_id"])
        null = int(self.token_ids["null_token_id"])
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])

        geometry = logp[0, box_start] + logp[5, box_end]
        for lane in range(1, 5):
            geometry = geometry + logp[lane, coord_start : coord_end + 1].max()
        empty = (
            logp[0, box_start]
            + logp[1, none]
            + logp[2, box_end]
            + logp[3, null]
            + logp[4, null]
            + logp[5, null]
        )
        probability = min(
            1.0 - 1e-6, max(1e-6, float(self._last_presence_probability))
        )
        present_prior = torch.tensor(probability, device=logits.device).log()
        absent_prior = torch.tensor(1.0 - probability, device=logits.device).log()
        return geometry + present_prior, empty + absent_prior

    def _mask_geometry_frame(self, logits: torch.Tensor) -> None:
        box_start = int(self.token_ids["box_start_token_id"])
        box_end = int(self.token_ids["box_end_token_id"])
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        device = logits.device
        self._keep_only(logits, 0, torch.tensor([box_start], device=device))
        keep = min(self.coordinate_trust_topk, coord_end - coord_start + 1)
        for lane in range(1, 5):
            candidates = (
                logits[0, lane, coord_start : coord_end + 1]
                .float()
                .topk(keep)
                .indices
                + coord_start
            )
            self._keep_only(logits, lane, candidates)
        self._keep_only(logits, 5, torch.tensor([box_end], device=device))

    def _mask_empty_frame(self, logits: torch.Tensor) -> None:
        tokens = (
            int(self.token_ids["box_start_token_id"]),
            int(self.token_ids["none_token_id"]),
            int(self.token_ids["box_end_token_id"]),
            int(self.token_ids["null_token_id"]),
            int(self.token_ids["null_token_id"]),
            int(self.token_ids["null_token_id"]),
        )
        for lane, token in enumerate(tokens):
            self._keep_only(
                logits, lane, torch.tensor([token], device=logits.device)
            )

    def adjust_pbd_logits(self, next_token_logits: torch.Tensor) -> torch.Tensor:
        adjusted = super().adjust_pbd_logits(next_token_logits)
        if not self._typed_frame_candidate:
            return adjusted
        self._typed_frame_candidate = False
        geometry_score, empty_score = self._frame_log_scores(adjusted)
        choose_geometry = bool(geometry_score >= empty_score)
        chosen_kind = "geometry" if choose_geometry else "empty"
        if chosen_kind != self._typed_native_kind:
            self._typed_native_switches += 1
        if choose_geometry:
            self._typed_geometry_choices += 1
            self._mask_geometry_frame(adjusted)
        else:
            self._typed_empty_choices += 1
            self._mask_empty_frame(adjusted)
        self._record_runtime_diagnostics()
        return adjusted

    def _record_runtime_diagnostics(self) -> None:
        super()._record_runtime_diagnostics()
        self.last_diagnostics.update(
            {
                "pivr_typed_geometry_choices": float(
                    self._typed_geometry_choices
                ),
                "pivr_typed_empty_choices": float(self._typed_empty_choices),
                "pivr_typed_native_switches": float(self._typed_native_switches),
                "pivr_typed_frame_pending": float(self._typed_frame_candidate),
            }
        )
