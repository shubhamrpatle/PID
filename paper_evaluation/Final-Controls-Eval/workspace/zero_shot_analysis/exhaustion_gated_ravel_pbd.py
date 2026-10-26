"""Learned exhaustion-gated PIVR for stable single-pass localization.

This controller keeps LocateAnything's native PBD grammar and one-pass visual
encoding, but closes two failure modes observed in dense all-class decoding:

* continuation is learned from whether another instance remains in the current
  reference group, with balanced supervision for the rarer ``stop`` event;
* visual coordinate re-entry is a reranker inside the native decoder's top-k
  coordinate support, so it cannot create an unrestricted token trajectory.

At inference the learned continuation posterior controls only whether cached
visual evidence is written into the next native geometry block.  A sampled
``stop`` returns generation unchanged to LocateAnything until the next
semantic reference; it never forces or suppresses an output token.  There is
no count prediction, fixed slot set, object cap, crop re-encoding, or second
decoder pass.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .grammar_gated_ravel_pbd import GrammarGatedRavelPBDController


class ExhaustionGatedRavelPBDController(GrammarGatedRavelPBDController):
    """PIVR with a learned, count-free native-block continuation posterior."""

    controller_kind = "native_grammar_gated_exhaustion"
    integrated_generation_observer = True

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
        continuation_temperature: float = 1.0,
        coordinate_trust_topk: int = 5,
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
        if continuation_temperature <= 0:
            raise ValueError("continuation_temperature must be positive")
        if coordinate_trust_topk <= 0:
            raise ValueError("coordinate_trust_topk must be positive")
        self.continuation_temperature = float(continuation_temperature)
        self.coordinate_trust_topk = int(coordinate_trust_topk)
        self._continuation_pending = False
        self._pending_continue_logit = 0.0
        self._applied_continue_logit = 0.0
        self._sampled_continue = -1.0
        self._continuation_decisions = 0
        self._continuation_stops = 0
        self._fresh_continuation_prediction = False
        self._native_coordinate_candidates: tuple[tuple[int, ...], ...] = ()
        self._planned_reference_blocks: tuple[tuple[int, ...], ...] = ()
        self._current_reference_index = -1
        self._semantic_handoff_pending = False
        self._semantic_handoffs = 0

    def inference_contract(self) -> dict[str, Any]:
        contract = super().inference_contract()
        contract.update(
            {
                "method": "PIVR-LA learned exhaustion-gated single-pass",
                "learned_count_free_continuation": True,
                "fixed_stop_threshold": False,
                "native_coordinate_trust_region": True,
                "coordinate_trust_topk": self.coordinate_trust_topk,
                "stochastic_learned_visual_reentry_survival": True,
                "almost_sure_count_free_termination": True,
                "soft_learned_geometry_continuation": False,
                "output_logits_are_never_forced_by_survival_gate": True,
                "learned_boundary_grammar_handoff": True,
                "geometry_is_always_native_generated": True,
                "controller_kind": self.controller_kind,
            }
        )
        return contract

    def _reset_state(self) -> None:
        super()._reset_state()
        self._continuation_pending = False
        self._pending_continue_logit = 0.0
        self._applied_continue_logit = 0.0
        self._sampled_continue = -1.0
        self._continuation_decisions = 0
        self._continuation_stops = 0
        self._fresh_continuation_prediction = False
        self._native_coordinate_candidates = ()
        self._current_reference_index = -1
        self._semantic_handoff_pending = False
        self._semantic_handoffs = 0

    def set_inference_task(self, question: str, tokenizer: Any | None = None) -> None:
        """Cache the requested detection-reference order for valid handoffs.

        The query already defines the semantic groups that LA must answer.  PIVR
        never invents a label here: after its learned exhaustion decision it may
        emit only the next requested ``<ref>...</ref>`` block, or native EOS when
        all requested groups are complete.  Geometry remains fully generated by
        LocateAnything.
        """

        super().set_inference_task(question, tokenizer=tokenizer)
        self._planned_reference_blocks = ()
        normalized = " ".join(str(question).strip().split())
        prefix = "Locate all the instances that matches the following description:"
        if tokenizer is None or not normalized.startswith(prefix):
            return
        payload = normalized[len(prefix) :].strip()
        if payload.endswith("."):
            payload = payload[:-1].rstrip()
        categories = tuple(value.strip() for value in payload.split("</c>") if value.strip())
        if not categories:
            return
        blocks: list[tuple[int, ...]] = []
        for category in categories:
            values = tokenizer.encode(
                f"<ref>{category}</ref>", add_special_tokens=False
            )
            if not values:
                raise RuntimeError(f"Could not tokenize reference group: {category!r}")
            blocks.append(tuple(int(value) for value in values))
        self._planned_reference_blocks = tuple(blocks)

    def _sample_continuation(
        self, probability: float, device: torch.device
    ) -> bool:
        """Draw the learned count-free survival decision for one native block."""

        return bool(torch.rand((), device=device) < probability)

    def _continuation_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Balance ``continue`` and the rarer class-exhaustion event."""

        losses = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        positive = targets.ge(0.5)
        positive_count = int(positive.sum())
        negative_count = int((~positive).sum())
        if positive_count == 0 or negative_count == 0:
            return losses.mean()
        weights = torch.where(
            positive,
            torch.full_like(losses, 0.5 / positive_count),
            torch.full_like(losses, 0.5 / negative_count),
        )
        return (losses * weights).sum()

    def _native_pbd_frame_kind(
        self, hidden_states: torch.Tensor, event: Any
    ) -> str:
        kind = super()._native_pbd_frame_kind(hidden_states, event)
        self._native_coordinate_candidates = ()
        if kind != "geometry" or self._native_norm is None or self._native_lm_head is None:
            return kind

        coordinate_count = {"point": 2, "hbb": 4, "obb": 8}.get(
            str(event.geometry), 0
        )
        available = min(coordinate_count, max(0, len(event.row) - 2))
        if available <= 0:
            return kind
        with torch.no_grad():
            positions = list(event.row)
            hidden = hidden_states[0, positions]
            logits = self._native_lm_head(self._native_norm(hidden)).float()
            coord_start = int(self.token_ids["coord_start_token_id"])
            coord_end = int(self.token_ids["coord_end_token_id"])
            keep = min(
                self.coordinate_trust_topk, coord_end - coord_start + 1
            )
            candidates: list[tuple[int, ...]] = []
            for lane in range(1, available + 1):
                ids = logits[lane, coord_start : coord_end + 1].topk(keep).indices
                candidates.append(
                    tuple(int(value) + coord_start for value in ids.tolist())
                )
        self._native_coordinate_candidates = tuple(candidates)
        return kind

    def adjust_pbd_logits(self, next_token_logits: torch.Tensor) -> torch.Tensor:
        """Apply only the native-coordinate trust region.

        The survival decision is committed before the next forward pass by
        :meth:`observe_generated_block`.  It therefore gates visual re-entry
        without converting a geometry-shaped speculative window into malformed
        semantic output.
        """

        if next_token_logits.ndim != 3 or int(next_token_logits.shape[0]) != 1:
            raise ValueError("PIVR expects PBD logits shaped [1, lanes, vocabulary]")
        adjusted = next_token_logits.clone()
        self._applied_continue_logit = 0.0
        self._sampled_continue = -1.0

        if self._native_coordinate_candidates:
            coord_start = int(self.token_ids["coord_start_token_id"])
            coord_end = int(self.token_ids["coord_end_token_id"])
            floor = torch.finfo(adjusted.dtype).min
            for lane, candidates in enumerate(
                self._native_coordinate_candidates, start=1
            ):
                if lane >= int(adjusted.shape[1]):
                    break
                allowed = torch.zeros(
                    coord_end - coord_start + 1,
                    dtype=torch.bool,
                    device=adjusted.device,
                )
                indices = torch.tensor(
                    [value - coord_start for value in candidates],
                    dtype=torch.long,
                    device=adjusted.device,
                )
                allowed[indices] = True
                coordinate_logits = adjusted[:, lane, coord_start : coord_end + 1]
                adjusted[:, lane, coord_start : coord_end + 1] = (
                    coordinate_logits.masked_fill(~allowed.unsqueeze(0), floor)
                )

        candidate_lanes = len(self._native_coordinate_candidates)
        # A trust region belongs to exactly one speculative native frame.
        self._native_coordinate_candidates = ()

        self.last_diagnostics.update(
            {
                "pivr_continuation_pending": float(self._continuation_pending),
                "pivr_pending_continue_logit": float(self._pending_continue_logit),
                "pivr_applied_continue_logit": float(self._applied_continue_logit),
                "pivr_sampled_continue": float(self._sampled_continue),
                "pivr_continuation_decisions": float(self._continuation_decisions),
                "pivr_continuation_stops": float(self._continuation_stops),
                "pivr_native_coordinate_candidate_lanes": float(
                    candidate_lanes
                ),
            }
        )
        return adjusted

    def observe_generated_block(self, block_type: str, tokens: Any) -> None:
        probability = min(
            1.0 - 1e-6, max(1e-6, float(self._next_continue_probability))
        )
        next_logit = math.log(probability) - math.log1p(-probability)
        super().observe_generated_block(block_type, tokens)
        if block_type in {"coord_box", "point_box"}:
            self._continuation_pending = True
            self._pending_continue_logit = float(next_logit)
            # Only a routed block has a fresh continuation posterior.  Once a
            # stop disables re-entry, subsequent native boxes cannot reactivate
            # it; the next semantic reference starts a new group.
            if self._fresh_continuation_prediction:
                sampled_continue = self._sample_continuation(
                    probability, self.residual_strength_logit.device
                )
                self._sampled_continue = float(sampled_continue)
                self._continuation_decisions += 1
                if not sampled_continue:
                    self._continuation_stops += 1
                    self._semantic_handoff_pending = bool(
                        self._planned_reference_blocks
                    )
                self._geometry_route_gate = float(sampled_continue)
            else:
                self._geometry_route_gate = 0.0
            self._fresh_continuation_prediction = False
        elif block_type in {"ref", "reference", "ref_object"}:
            self._current_reference_index += 1
            self._continuation_pending = False
            self._pending_continue_logit = 0.0
            self._sampled_continue = -1.0
            self._fresh_continuation_prediction = False
        else:
            self._continuation_pending = False
            self._pending_continue_logit = 0.0
            self._fresh_continuation_prediction = False
        self._record_runtime_diagnostics()

    def postprocess_generated_block(
        self, block_type: str, tokens: torch.Tensor
    ) -> tuple[str, torch.Tensor]:
        """Commit one learned class-boundary decision in valid LA grammar."""

        if not self._semantic_handoff_pending:
            return block_type, tokens
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

    def _route(self, hidden_states: torch.Tensor, context: Any) -> torch.Tensor:
        before = self.route_call_count
        updated = super()._route(hidden_states, context)
        if not self._training_context and self.route_call_count > before:
            self._fresh_continuation_prediction = True
        return updated

    def _record_runtime_diagnostics(self) -> None:
        super()._record_runtime_diagnostics()
        self.last_diagnostics.update(
            {
                "pivr_continuation_pending": float(self._continuation_pending),
                "pivr_pending_continue_logit": float(self._pending_continue_logit),
                "pivr_applied_continue_logit": float(self._applied_continue_logit),
                "pivr_sampled_continue": float(self._sampled_continue),
                "pivr_continuation_decisions": float(self._continuation_decisions),
                "pivr_continuation_stops": float(self._continuation_stops),
                "pivr_semantic_handoffs": float(self._semantic_handoffs),
                "pivr_semantic_handoff_pending": float(
                    self._semantic_handoff_pending
                ),
                "pivr_native_coordinate_candidate_lanes": float(
                    len(self._native_coordinate_candidates)
                ),
            }
        )

    def install(self, model: nn.Module) -> None:
        super().install(model)
        object.__setattr__(model, "_pivr_generation_controller", self)

    def remove(self) -> None:
        # Avoid keeping a stale callback on a model reused by another arm.
        model = None
        for handle in self._handles:
            module_ref = getattr(handle, "module", None)
            if module_ref is not None:
                model = module_ref
                break
        super().remove()
        if model is not None and getattr(model, "_pivr_generation_controller", None) is self:
            object.__setattr__(model, "_pivr_generation_controller", None)
