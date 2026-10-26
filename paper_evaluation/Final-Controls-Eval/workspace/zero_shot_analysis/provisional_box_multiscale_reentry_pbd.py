"""Provisional-box-conditioned multi-scale visual re-entry for LocateAnything.

Both matched arms receive the same explicit point and provisional HBB.  This
controller alone reads a fixed-budget set of interior, boundary, and context
features from the image field that was prepared for the single MoonViT pass.
The evidence is written into LA's native six-lane PBD block; native coordinate
tokens and the explicit ``None`` grammar are retained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .highres_point_presence_gate_pbd import HighResolutionPointPresenceGatePBD
from .ravel_pbd import RavelContext


IGNORE_INDEX = -100


@dataclass(frozen=True)
class ProvisionalBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def normalized(self) -> tuple[float, float, float, float]:
        left, right = sorted((self.x1, self.x2))
        top, bottom = sorted((self.y1, self.y2))
        return left / 1000.0, top / 1000.0, right / 1000.0, bottom / 1000.0


def provisional_box_from_tokens(
    input_ids: torch.Tensor,
    *,
    token_ids: Mapping[str, int],
    labels: torch.Tensor | None = None,
) -> ProvisionalBox | None:
    """Read one four-coordinate proposal from the user portion of a sample."""

    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("Provisional-box parsing expects [1, sequence] tokens")
    ids = input_ids[0]
    limit = int(ids.numel())
    if labels is not None:
        if labels.shape != input_ids.shape:
            raise ValueError("Labels and input ids must have the same shape")
        supervised = labels[0].ne(IGNORE_INDEX).nonzero(as_tuple=False).flatten()
        if supervised.numel():
            limit = int(supervised[0])
    box_start = int(token_ids["box_start_token_id"])
    box_end = int(token_ids["box_end_token_id"])
    coord_start = int(token_ids["coord_start_token_id"])
    coord_end = int(token_ids["coord_end_token_id"])
    matches: list[ProvisionalBox] = []
    for index in range(max(0, limit - 5)):
        values = [int(value) for value in ids[index : index + 6].tolist()]
        if (
            values[0] == box_start
            and all(coord_start <= value <= coord_end for value in values[1:5])
            and values[5] == box_end
        ):
            coordinates = [value - coord_start for value in values[1:5]]
            matches.append(ProvisionalBox(*coordinates))
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"Expected one provisional HBB, found {len(matches)}")
    proposal = matches[0]
    left, top, right, bottom = proposal.normalized()
    if right <= left or bottom <= top:
        raise RuntimeError(f"Degenerate provisional HBB: {proposal}")
    return proposal


class ProvisionalBoxMultiscaleReentryPBD(HighResolutionPointPresenceGatePBD):
    """Refine a shared non-oracle proposal with soft multi-scale local evidence."""

    controller_kind = "provisional_box_multiscale_reentry"

    def __init__(
        self,
        *,
        proposal_scale_multipliers: Sequence[float] = (0.75, 1.0, 1.5, 2.0),
        proposal_roi_grid: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        multipliers = tuple(float(value) for value in proposal_scale_multipliers)
        if not multipliers or any(value <= 0 for value in multipliers):
            raise ValueError("Proposal scale multipliers must be positive")
        if proposal_roi_grid < 2:
            raise ValueError("Proposal ROI grid must be at least 2")
        self.register_buffer(
            "proposal_scale_multipliers",
            torch.tensor(multipliers, dtype=torch.float32),
        )
        self.proposal_roi_grid = int(proposal_roi_grid)
        # center, size, point offset, and point position.
        self.proposal_proj = nn.Sequential(
            nn.Linear(8, self.evidence_dim),
            nn.GELU(),
            nn.Linear(self.evidence_dim, self.evidence_dim),
        )
        self.proposal_scale_selector = nn.Linear(
            2 * self.evidence_dim, len(multipliers)
        )
        self.proposal_scale_embedding = nn.Parameter(
            torch.zeros(len(multipliers), self.evidence_dim)
        )
        self._provisional_box: ProvisionalBox | None = None
        self._proposal_roi_reads = 0

    def _reset_state(self) -> None:
        super()._reset_state()
        self._provisional_box = None
        self._proposal_roi_reads = 0

    def set_addressed_training_context(
        self,
        context: RavelContext,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        image_grid_hws: Any,
    ) -> None:
        super().set_addressed_training_context(
            context, input_ids, labels, image_grid_hws
        )
        self._provisional_box = provisional_box_from_tokens(
            input_ids, labels=labels, token_ids=self.token_ids
        )
        if self._prompt_point is not None and self._provisional_box is None:
            raise RuntimeError("Point-addressed refinement has no provisional HBB")

    def set_inference_image_layout(
        self, input_ids: torch.Tensor, image_grid_hws: Any
    ) -> None:
        super().set_inference_image_layout(input_ids, image_grid_hws)
        self._provisional_box = provisional_box_from_tokens(
            input_ids, token_ids=self.token_ids
        )
        if self._prompt_point is not None and self._provisional_box is None:
            raise RuntimeError("Point-addressed inference has no provisional HBB")

    def finish_forward(self) -> None:
        super().finish_forward()
        self._provisional_box = None
        self._proposal_roi_reads = 0

    def _proposal_geometry(
        self, point: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if self._provisional_box is None:
            raise RuntimeError("Provisional HBB is unavailable")
        left, top, right, bottom = self._provisional_box.normalized()
        values = point.new_tensor([left, top, right, bottom], dtype=torch.float32)
        center_x = 0.5 * (values[0] + values[2])
        center_y = 0.5 * (values[1] + values[3])
        width = (values[2] - values[0]).clamp_min(0.002)
        height = (values[3] - values[1]).clamp_min(0.002)
        geometry = torch.stack(
            (
                center_x,
                center_y,
                width,
                height,
                point[0].float() - center_x,
                point[1].float() - center_y,
                point[0].float(),
                point[1].float(),
            )
        )
        return self.proposal_proj(geometry.to(dtype)), (
            center_x,
            center_y,
            width,
            height,
        )

    def _proposal_pixel_evidence(
        self,
        *,
        query: torch.Tensor,
        proposal_feature: torch.Tensor,
        center_x: torch.Tensor,
        center_y: torch.Tensor,
        width: torch.Tensor,
        height: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._pixel_field is None:
            raise RuntimeError("High-resolution pixel field is unavailable")
        route_dtype = self.pixel_key.weight.dtype
        offsets = torch.linspace(
            -0.5,
            0.5,
            self.proposal_roi_grid,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        scale_evidence: list[torch.Tensor] = []
        for scale_index, multiplier in enumerate(
            self.proposal_scale_multipliers.to(hidden_states.device)
        ):
            sample_x = center_x + xx * width * multiplier
            sample_y = center_y + yy * height * multiplier
            grid = torch.stack(
                (sample_x.clamp(0, 1) * 2 - 1, sample_y.clamp(0, 1) * 2 - 1),
                dim=-1,
            )
            sampled = F.grid_sample(
                self._pixel_field.to(hidden_states.device, route_dtype),
                grid.unsqueeze(0),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            sampled = sampled.squeeze(0).permute(1, 2, 0).reshape(
                -1, self.pixel_feature_dim
            )
            keys = F.normalize(self.pixel_key(sampled).float(), dim=-1)
            values = self.pixel_value(sampled)
            attention = (torch.mv(keys, query) / self.address_temperature).softmax(0)
            evidence = torch.matmul(attention.to(route_dtype), values)
            evidence = evidence + self.proposal_scale_embedding[scale_index].to(
                evidence.dtype
            )
            scale_evidence.append(evidence)
        scale_logits = self.proposal_scale_selector(
            torch.cat((query.to(route_dtype), proposal_feature), dim=-1)
        ).float()
        scale_weights = scale_logits.softmax(-1)
        stacked = torch.stack(scale_evidence)
        self._proposal_roi_reads += 1
        return torch.sum(scale_weights.to(stacked.dtype).unsqueeze(-1) * stacked, dim=0), scale_logits

    def _route(self, hidden_states: torch.Tensor, context: RavelContext) -> torch.Tensor:
        if self._prompt_point is None or self._provisional_box is None:
            return hidden_states
        if self._global_projected is None or self._dense_projected is None:
            return hidden_states
        if self._pixel_field is None or self._grid_hw is None or self._dense_hw is None:
            raise RuntimeError("Proposal re-entry visual caches are unavailable")
        events = tuple(
            event
            for event in context.events
            if event.geometry in {"hbb", "none"}
            and event.row
            and max(event.row) < hidden_states.shape[1]
        )
        if not events:
            return hidden_states

        self.route_call_count += 1
        dtype = self.query_proj.weight.dtype
        global_tokens = self._global_projected.to(hidden_states.device, dtype)
        dense_tokens = self._dense_projected.to(hidden_states.device, dtype)
        global_keys = F.normalize(self.global_key(global_tokens).float(), dim=-1)
        dense_keys = F.normalize(self.dense_key(dense_tokens).float(), dim=-1)
        global_values = self.global_value(global_tokens)
        dense_values = self.dense_value(dense_tokens)
        gh, gw = self._grid_hw[0] // 2, self._grid_hw[1] // 2
        dh, dw = self._dense_hw
        global_x, global_y = self._centers(gh, gw, hidden_states.device)
        source_height, source_width = self._grid_hw
        dense_y_grid, dense_x_grid = torch.meshgrid(
            torch.arange(1, source_height, device=hidden_states.device, dtype=torch.float32)
            / source_height,
            torch.arange(1, source_width, device=hidden_states.device, dtype=torch.float32)
            / source_width,
            indexing="ij",
        )
        dense_x, dense_y = dense_x_grid.reshape(-1), dense_y_grid.reshape(-1)
        point = hidden_states.new_tensor(
            [self._prompt_point.x / 1000.0, self._prompt_point.y / 1000.0],
            dtype=dtype,
        )
        point_feature = self.point_proj(point)
        proposal_feature, proposal_geometry = self._proposal_geometry(point, dtype)
        center_x, center_y, proposal_width, proposal_height = proposal_geometry
        updated = hidden_states.clone()
        address_losses: list[torch.Tensor] = []
        presence_logits: list[torch.Tensor] = []
        presence_frame_rows: list[int] = []
        presence_targets: list[float] = []
        scale_losses: list[torch.Tensor] = []
        support_hits: list[torch.Tensor] = []
        selected_scales: list[torch.Tensor] = []
        strengths = self.residual_strength().to(hidden_states.device, hidden_states.dtype)

        for event in events:
            query_state = hidden_states[0, event.row[0]]
            if self._training_context and self.isolate_controller_gradients:
                query_state = query_state.detach()
            query = self.query_proj(
                self.query_norm(query_state.to(self.query_norm.weight.dtype)).to(dtype)
            )
            query = F.normalize(query.float(), dim=-1)

            global_semantic = torch.mv(global_keys, query) / self.address_temperature
            global_distance = (global_x - point[0].float()).square() + (
                global_y - point[1].float()
            ).square()
            global_probability = (global_semantic - global_distance / 0.5**2).softmax(0)
            global_evidence = torch.matmul(global_probability.to(dtype), global_values)

            sigma_x = torch.maximum(
                proposal_width * 0.5,
                proposal_width.new_tensor(0.5 / dw),
            )
            sigma_y = torch.maximum(
                proposal_height * 0.5,
                proposal_height.new_tensor(0.5 / dh),
            )
            dense_prior = -0.5 * (
                ((dense_x - center_x) / sigma_x).square()
                + ((dense_y - center_y) / sigma_y).square()
            )
            dense_semantic = torch.mv(dense_keys, query) / self.address_temperature
            dense_probability = (dense_semantic + dense_prior).softmax(0)
            dense_evidence = torch.matmul(dense_probability.to(dtype), dense_values)
            pixel_evidence, scale_logits = self._proposal_pixel_evidence(
                query=query,
                proposal_feature=proposal_feature,
                center_x=center_x,
                center_y=center_y,
                width=proposal_width,
                height=proposal_height,
                hidden_states=hidden_states,
            )
            selected_scales.append(
                torch.dot(
                    scale_logits.softmax(-1),
                    self.proposal_scale_multipliers.to(scale_logits.device),
                )
            )

            common = torch.cat(
                (
                    query.to(dtype),
                    point_feature,
                    global_evidence,
                    dense_evidence,
                    pixel_evidence,
                    proposal_feature,
                ),
                dim=-1,
            )
            presence_logit = self.presence_head(common).squeeze(-1).float()
            presence_logits.append(presence_logit)
            presence_frame_rows.append(int(event.row[1]))
            presence_targets.append(0.0 if event.geometry == "none" else 1.0)
            gate = presence_logit.sigmoid().to(updated.dtype)

            lane_positions = event.row[1:-1]
            lane_count = 1 if event.geometry == "none" else min(4, len(lane_positions))
            for lane_index, position in enumerate(lane_positions[:lane_count]):
                fused = torch.cat((common, self.lane_embedding[lane_index]), dim=-1)
                residual = self.evidence_up(fused).to(updated.dtype)
                updated[0, position] = (
                    updated[0, position] + strengths[lane_index] * gate * residual
                )

            if event.geometry == "hbb" and len(event.coordinates) == 4:
                x1, y1, x2, y2 = [float(value) / 1000.0 for value in event.coordinates]
                target_left, target_right = sorted((x1, x2))
                target_top, target_bottom = sorted((y1, y2))
                target_center_x = 0.5 * (target_left + target_right)
                target_center_y = 0.5 * (target_top + target_bottom)
                required_x = (
                    (target_right - target_left)
                    + 2.0 * abs(target_center_x - float(center_x))
                ) / max(0.002, float(proposal_width))
                required_y = (
                    (target_bottom - target_top)
                    + 2.0 * abs(target_center_y - float(center_y))
                ) / max(0.002, float(proposal_height))
                required = max(required_x, required_y)
                multipliers = self.proposal_scale_multipliers.to(scale_logits.device)
                target_index = torch.argmin(
                    (multipliers.log() - math.log(max(0.01, required))).abs()
                ).view(1)
                scale_losses.append(
                    F.cross_entropy(scale_logits.view(1, -1), target_index)
                )
                support = self._dense_support(event.coordinates, hidden_states.device)
                mass = torch.dot(dense_probability, support)
                address_losses.append(-mass.clamp_min(1e-8).log())
                support_hits.append(
                    support[int(dense_probability.argmax())].gt(0).float()
                )

        self._address_loss = torch.stack(address_losses).mean() if address_losses else None
        self._presence_loss = (
            F.binary_cross_entropy_with_logits(
                torch.stack(presence_logits),
                torch.tensor(presence_targets, device=hidden_states.device),
            )
            if presence_logits
            else None
        )
        self._presence_frame_rows = tuple(presence_frame_rows)
        self._presence_logits = tuple(presence_logits)
        if presence_logits:
            self._last_presence_probability = float(
                presence_logits[-1].sigmoid().detach()
            )
        self._scale_loss = torch.stack(scale_losses).mean() if scale_losses else None
        self._typed_frame_pending = bool(
            not self._training_context and self._presence_logits
        )
        self.last_diagnostics = {
            "proposal_events": float(len(events)),
            "proposal_vision_captures": float(self.vision_capture_count),
            "proposal_global_tokens": float(global_tokens.shape[0]),
            "proposal_dense_tokens": float(dense_tokens.shape[0]),
            "proposal_roi_reads": float(self._proposal_roi_reads),
            "proposal_roi_cells_per_scale": float(self.proposal_roi_grid**2),
            "proposal_scale_count": float(len(self.proposal_scale_multipliers)),
            "proposal_selected_scale": float(torch.stack(selected_scales).mean().detach()),
            "proposal_width": float(proposal_width.detach()),
            "proposal_height": float(proposal_height.detach()),
            "proposal_address_loss": 0.0
            if self._address_loss is None
            else float(self._address_loss.detach()),
            "proposal_presence_loss": 0.0
            if self._presence_loss is None
            else float(self._presence_loss.detach()),
            "proposal_scale_loss": 0.0
            if self._scale_loss is None
            else float(self._scale_loss.detach()),
            "proposal_support_hit_rate": 0.0
            if not support_hits
            else float(torch.stack(support_hits).mean().detach()),
            "proposal_presence_probability": float(self._last_presence_probability),
            "proposal_residual_strength": float(strengths[:4].mean().detach()),
            "proposal_dense_tokens_in_qwen": 0.0,
        }
        return updated

    def inference_contract(self) -> dict[str, Any]:
        contract = super().inference_contract()
        contract.update(
            {
                "method": "provisional-box-conditioned multi-scale visual re-entry",
                "controller_kind": self.controller_kind,
                "proposal_is_explicit_input": True,
                "proposal_is_ground_truth_at_inference": False,
                "soft_multiscale_roi": True,
                "proposal_scale_multipliers": [
                    float(value) for value in self.proposal_scale_multipliers.tolist()
                ],
                "proposal_roi_grid": self.proposal_roi_grid,
                "interior_boundary_context_sampling": True,
                "second_moonvit_pass": False,
                "geometry_coordinates_remain_native": True,
            }
        )
        return contract
