"""Evidence-typed HBB/None gate for high-resolution point re-entry.

The point-evidence controller retrieves and writes visual evidence as before.
This extension couples its learned presence probability to LocateAnything's
native six-lane PBD decision.  It compares exactly two legal frames, an HBB
and explicit ``None``, then masks the frame to the winning grammar state.
Coordinates remain selected from LA's native coordinate distributions.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .highres_point_evidence_pbd import HighResolutionPointEvidencePBD


class HighResolutionPointPresenceGatePBD(HighResolutionPointEvidencePBD):
    """Choose native HBB versus native explicit-None using visual evidence."""

    controller_kind = "highres_point_presence_gate"
    # LocateAnything calls observe_generated_block only for integrated
    # controllers. Without this flag, _point_address_complete is never set and
    # a one-point refinement can continue emitting unrelated geometry blocks.
    integrated_generation_observer = True

    def __init__(
        self,
        *,
        maximum_presence_bias: float = 4.0,
        presence_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if maximum_presence_bias <= 0 or presence_temperature <= 0:
            raise ValueError("Presence bias and temperature must be positive")
        self.maximum_presence_bias = float(maximum_presence_bias)
        self.presence_temperature = float(presence_temperature)
        # A 0.5 visual prior leaves the initial typed choice to native Qwen.
        torch.nn.init.zeros_(self.presence_head[-1].weight)
        torch.nn.init.zeros_(self.presence_head[-1].bias)
        self._typed_frame_pending = False
        self._typed_geometry_choices = 0
        self._typed_none_choices = 0
        self._point_address_complete = False
        object.__setattr__(self, "_generation_host", None)

    def _reset_state(self) -> None:
        super()._reset_state()
        self._typed_frame_pending = False
        self._typed_geometry_choices = 0
        self._typed_none_choices = 0
        self._point_address_complete = False

    def _route(self, hidden_states: torch.Tensor, context: Any) -> torch.Tensor:
        updated = super()._route(hidden_states, context)
        self._typed_frame_pending = bool(
            not self._training_context and self._presence_logits
        )
        return updated

    @staticmethod
    def _keep_only(
        logits: torch.Tensor, lane: int, token_ids: torch.Tensor
    ) -> None:
        floor = torch.finfo(logits.dtype).min
        retained = logits[:, lane, token_ids].clone()
        logits[:, lane, :] = floor
        logits[:, lane, token_ids] = retained

    def _frame_log_scores(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(logits.shape[1]) < 6:
            raise RuntimeError("Typed point refinement requires six PBD lanes")
        logp = torch.log_softmax(logits[:, :6, :].float(), dim=-1)[0]
        box_start = int(self.token_ids["box_start_token_id"])
        box_end = int(self.token_ids["box_end_token_id"])
        none = int(self.token_ids["none_token_id"])
        null = int(self.token_ids["null_token_id"])
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        geometry = logp[0, box_start] + logp[5, box_end]
        geometry = geometry + sum(
            logp[lane, coord_start : coord_end + 1].max()
            for lane in range(1, 5)
        )
        empty = (
            logp[0, box_start]
            + logp[1, none]
            + logp[2, box_end]
            + logp[3, null]
            + logp[4, null]
            + logp[5, null]
        )
        probability = min(
            1.0 - 1e-6, max(1e-6, self._last_presence_probability)
        )
        prior = logits.new_tensor(probability, dtype=torch.float32)
        return geometry + prior.log(), empty + (1.0 - prior).log()

    def _mask_geometry(self, logits: torch.Tensor) -> None:
        device = logits.device
        self._keep_only(
            logits,
            0,
            torch.tensor([int(self.token_ids["box_start_token_id"])], device=device),
        )
        coordinates = torch.arange(
            int(self.token_ids["coord_start_token_id"]),
            int(self.token_ids["coord_end_token_id"]) + 1,
            device=device,
        )
        for lane in range(1, 5):
            self._keep_only(logits, lane, coordinates)
        self._keep_only(
            logits,
            5,
            torch.tensor([int(self.token_ids["box_end_token_id"])], device=device),
        )

    def _mask_none(self, logits: torch.Tensor) -> None:
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
        if self._point_address_complete:
            adjusted = next_token_logits.clone()
            self._keep_only(
                adjusted,
                0,
                torch.tensor(
                    [int(self.token_ids["im_end_token_id"])],
                    device=adjusted.device,
                ),
            )
            self.last_diagnostics["typed_presence_forced_terminal"] = 1.0
            return adjusted
        if not self._typed_frame_pending:
            return next_token_logits
        self._typed_frame_pending = False
        adjusted = next_token_logits.clone()
        geometry, empty = self._frame_log_scores(adjusted)
        if bool(geometry >= empty):
            self._typed_geometry_choices += 1
            self._mask_geometry(adjusted)
        else:
            self._typed_none_choices += 1
            self._mask_none(adjusted)
        self.last_diagnostics.update(
            {
                "typed_presence_geometry_choices": float(
                    self._typed_geometry_choices
                ),
                "typed_presence_none_choices": float(self._typed_none_choices),
            }
        )
        return adjusted

    def observe_generated_block(self, block_type: str, tokens: Any) -> None:
        super().observe_generated_block(block_type, tokens)
        if block_type in {"coord_box", "empty_box"}:
            self._point_address_complete = True

    def install(self, model: torch.nn.Module) -> None:
        super().install(model)
        object.__setattr__(self, "_generation_host", model)
        object.__setattr__(model, "_pivr_generation_controller", self)

    def remove(self) -> None:
        model = self._generation_host
        if (
            model is not None
            and getattr(model, "_pivr_generation_controller", None) is self
        ):
            object.__setattr__(model, "_pivr_generation_controller", None)
        object.__setattr__(self, "_generation_host", None)
        super().remove()

    def geometry_decision_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor | None:
        """Train the same coordinate-vs-None margin used at inference."""

        if not self._presence_logits:
            return None
        coord_start = int(self.token_ids["coord_start_token_id"])
        coord_end = int(self.token_ids["coord_end_token_id"])
        none = int(self.token_ids["none_token_id"])
        losses: list[torch.Tensor] = []
        for position, presence_logit in zip(
            self._presence_frame_rows, self._presence_logits
        ):
            target_position = position + 1
            if target_position >= int(labels.shape[1]):
                continue
            target = labels[0, target_position]
            if int(target) < 0:
                continue
            logits = F.linear(hidden_states[0, position], weight).float()
            delta = self.maximum_presence_bias * torch.tanh(
                presence_logit / self.presence_temperature
            )
            logits = logits.clone()
            logits[coord_start : coord_end + 1] += delta
            logits[none] -= delta
            losses.append(
                F.cross_entropy(logits.unsqueeze(0), target.reshape(1).long())
            )
        return torch.stack(losses).mean() if losses else None

    def inference_contract(self) -> dict[str, Any]:
        contract = super().inference_contract()
        contract.update(
            {
                "method": "high-resolution point evidence with typed HBB/None gate",
                "controller_kind": self.controller_kind,
                "typed_states": ["native_hbb", "explicit_none"],
                "native_visual_joint_frame_decision": True,
                "posthoc_thresholding": False,
                "geometry_coordinates_remain_native": True,
                "at_most_one_geometry_block_per_point": True,
                "force_eos_after_geometry_or_none": True,
            }
        )
        return contract
