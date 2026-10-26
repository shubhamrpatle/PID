"""Soft-survival PIVR with a grammar-valid no-progress cycle guard.

The learned continuation posterior scales cached visual re-entry but never
samples an object count or forces a stop.  A semantic handoff is requested only
when LocateAnything emits the exact same quantized geometry twice consecutively
inside one reference group.  That second block is redundant under any one-to-one
localization metric and is a direct observable certificate of no progress.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .exhaustion_gated_ravel_pbd import ExhaustionGatedRavelPBDController
from .grammar_gated_ravel_pbd import GrammarGatedRavelPBDController


class CycleGuardedRavelPBDController(ExhaustionGatedRavelPBDController):
    """Preserve dense continuation while terminating exact geometry cycles."""

    controller_kind = "native_grammar_gated_cycle_guarded"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._last_geometry_signature: tuple[int, ...] | None = None
        self._no_progress_cycles = 0

    def inference_contract(self) -> dict[str, Any]:
        contract = super().inference_contract()
        contract.update(
            {
                "method": "PIVR-LA soft-survival cycle-guarded single-pass",
                "controller_kind": self.controller_kind,
                "stochastic_learned_visual_reentry_survival": False,
                "soft_learned_visual_reentry_survival": True,
                "exact_no_progress_cycle_guard": True,
                "fixed_stop_threshold": False,
                "fixed_object_cap": None,
                "fixed_object_slots": False,
                "geometry_is_always_native_generated": True,
            }
        )
        return contract

    def _reset_state(self) -> None:
        super()._reset_state()
        self._last_geometry_signature = None
        self._no_progress_cycles = 0

    @staticmethod
    def _signature(tokens: Any) -> tuple[int, ...]:
        values = torch.as_tensor(tokens).detach().reshape(-1).cpu().tolist()
        return tuple(int(value) for value in values)

    def observe_generated_block(self, block_type: str, tokens: Any) -> None:
        probability = min(
            1.0 - 1e-6, max(1e-6, float(self._next_continue_probability))
        )
        next_logit = math.log(probability) - math.log1p(-probability)

        # Bypass the stochastic subclass implementation.  Native grammar and
        # coverage bookkeeping are retained exactly.
        GrammarGatedRavelPBDController.observe_generated_block(
            self, block_type, tokens
        )
        if block_type in {"ref", "reference", "ref_object"}:
            self._current_reference_index += 1
            self._last_geometry_signature = None
            self._semantic_handoff_pending = False
            self._continuation_pending = False
            self._pending_continue_logit = 0.0
            self._geometry_route_gate = 1.0
        elif block_type in {"coord_box", "point_box"}:
            self._continuation_pending = True
            self._pending_continue_logit = float(next_logit)
            self._sampled_continue = float(probability)
            self._continuation_decisions += 1
            signature = self._signature(tokens)
            if signature == self._last_geometry_signature:
                self._no_progress_cycles += 1
                self._continuation_stops += 1
                self._semantic_handoff_pending = bool(
                    self._planned_reference_blocks
                )
                self._geometry_route_gate = 0.0
            else:
                self._last_geometry_signature = signature
                self._geometry_route_gate = float(probability)
        elif block_type == "empty_box":
            self._last_geometry_signature = None
            self._geometry_route_gate = 0.0
        else:
            self._continuation_pending = False
            self._pending_continue_logit = 0.0
        self._record_runtime_diagnostics()

    def _record_runtime_diagnostics(self) -> None:
        super()._record_runtime_diagnostics()
        self.last_diagnostics.update(
            {
                "pivr_no_progress_cycles": float(self._no_progress_cycles),
                "pivr_soft_survival_strength": float(
                    self._geometry_route_gate
                ),
            }
        )
