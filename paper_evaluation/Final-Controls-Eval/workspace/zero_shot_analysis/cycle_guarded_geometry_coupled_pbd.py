"""Cycle-safe geometry-coupled visual re-entry for LocateAnything.

This inference controller preserves the learned geometry-coupled visual-event
model and LocateAnything's native autoregressive cardinality.  It requests a
grammar-valid class handoff only after the model emits an *exact* duplicate of
an already accepted geometry in the current reference group.  Such a duplicate
cannot add a true positive under one-to-one matching, so the guard removes a
certified no-progress cycle without introducing slots, counts, or an object cap.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .geometry_coupled_visual_event_reentry_pbd import (
    GeometryCoupledVisualEventReentryPBD,
)


class CycleGuardedGeometryCoupledVisualEventReentryPBD(
    GeometryCoupledVisualEventReentryPBD
):
    """Geometry-coupled PIVR with an exact no-progress certificate."""

    integrated_generation_observer = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._planned_reference_blocks: tuple[tuple[int, ...], ...] = ()
        self._current_reference_index = -1
        self._accepted_signatures: set[tuple[int, ...]] = set()
        self._semantic_handoff_pending = False
        self._semantic_handoffs = 0
        self._exact_duplicate_cycles = 0
        # This is a runtime back-reference, not controller state.  Bypass
        # nn.Module.__setattr__ so the 3B generation model is not registered as
        # a child module and therefore is not expected in adapter checkpoints.
        object.__setattr__(self, "_generation_model", None)

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        super()._reset_sample_state(preserve_pixel_field=preserve_pixel_field)
        self._current_reference_index = -1
        self._accepted_signatures = set()
        self._semantic_handoff_pending = False
        self._semantic_handoffs = 0
        self._exact_duplicate_cycles = 0

    def set_inference_task(self, question: str, tokenizer: Any | None = None) -> None:
        """Cache only the semantic order already specified by the query."""

        self._planned_reference_blocks = ()
        normalized = " ".join(str(question).strip().split())
        prefix = "Locate all the instances that matches the following description:"
        if tokenizer is None or not normalized.startswith(prefix):
            return
        payload = normalized[len(prefix) :].strip()
        if payload.endswith("."):
            payload = payload[:-1].rstrip()
        categories = tuple(
            value.strip() for value in payload.split("</c>") if value.strip()
        )
        blocks: list[tuple[int, ...]] = []
        for category in categories:
            values = tokenizer.encode(
                f"<ref>{category}</ref>", add_special_tokens=False
            )
            if not values:
                raise RuntimeError(
                    f"Could not tokenize reference group: {category!r}"
                )
            blocks.append(tuple(int(value) for value in values))
        self._planned_reference_blocks = tuple(blocks)

    @staticmethod
    def _signature(tokens: Any) -> tuple[int, ...]:
        values = torch.as_tensor(tokens).detach().reshape(-1).cpu().tolist()
        return tuple(int(value) for value in values)

    def observe_generated_block(self, out_type: str, tokens: Any) -> None:
        super().observe_generated_block(out_type, tokens)
        if out_type in {"ref", "reference", "ref_object"}:
            self._current_reference_index += 1
            self._accepted_signatures.clear()
            self._semantic_handoff_pending = False
        elif out_type in {"coord_box", "point_box"}:
            signature = self._signature(tokens)
            if signature in self._accepted_signatures:
                self._exact_duplicate_cycles += 1
                self._semantic_handoff_pending = bool(
                    self._planned_reference_blocks
                )
            else:
                self._accepted_signatures.add(signature)
        elif out_type in {"empty_box", "im_end"}:
            self._semantic_handoff_pending = False
        self._update_cycle_diagnostics()

    def postprocess_generated_block(
        self, out_type: str, tokens: torch.Tensor
    ) -> tuple[str, torch.Tensor]:
        if not self._semantic_handoff_pending:
            return out_type, tokens
        self._semantic_handoff_pending = False
        next_index = self._current_reference_index + 1
        if next_index < len(self._planned_reference_blocks):
            self._semantic_handoffs += 1
            replacement = torch.tensor(
                self._planned_reference_blocks[next_index],
                dtype=tokens.dtype,
                device=tokens.device,
            )
            return "ref_object", replacement
        replacement = torch.tensor(
            [int(self.token_ids["im_end_token_id"])],
            dtype=tokens.dtype,
            device=tokens.device,
        )
        return "im_end", replacement

    def _update_cycle_diagnostics(self) -> None:
        self.last_diagnostics.update(
            {
                "geometry_cycle_guard_exact_duplicates": float(
                    self._exact_duplicate_cycles
                ),
                "geometry_cycle_guard_semantic_handoffs": float(
                    self._semantic_handoffs
                ),
                "geometry_cycle_guard_pending": float(
                    self._semantic_handoff_pending
                ),
                "geometry_cycle_guard_unbounded": 1.0,
            }
        )

    def install(self, model: nn.Module) -> None:
        super().install(model)
        object.__setattr__(self, "_generation_model", model)
        object.__setattr__(model, "_pivr_generation_controller", self)

    def remove(self) -> None:
        model = self._generation_model
        super().remove()
        if model is not None and getattr(
            model, "_pivr_generation_controller", None
        ) is self:
            object.__setattr__(model, "_pivr_generation_controller", None)
        object.__setattr__(self, "_generation_model", None)
