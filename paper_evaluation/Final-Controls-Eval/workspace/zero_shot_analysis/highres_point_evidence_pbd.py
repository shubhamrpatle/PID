"""High-resolution point-conditioned evidence for native LA box decoding.

This controller keeps the one-MoonViT/one-Qwen contract of the single-encode
point-addressed pilot, while correcting two measured limitations:

1. logarithmic extent anchors plus a continuous log residual cover tiny through
   full-image objects; and
2. a compact shared stem retains within-patch pixel evidence that cannot be
   reconstructed from MoonViT's stride-14 token grid.

The pixel field is computed once from the processor's existing 14x14 patches.
A learned address-centered extent samples a fixed lattice from that field.  No
crop image, second MoonViT pass, extra Qwen visual tokens, decoder replica, or
fixed object slot is introduced.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .ravel_pbd import RavelContext
from .single_encode_point_address_pbd import SingleEncodePointAddressPBD


class HighResolutionPointEvidencePBD(SingleEncodePointAddressPBD):
    """Retrieve scale-adaptive raw-patch evidence at an explicit point."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vision_hidden_size: int,
        relay_layer: int,
        token_ids: Mapping[str, int],
        evidence_dim: int = 128,
        address_temperature: float = 0.10,
        maximum_residual_strength: float = 1.0,
        isolate_controller_gradients: bool = True,
        scale_anchors: Sequence[float] = (
            0.004,
            0.008,
            0.016,
            0.032,
            0.064,
            0.128,
            0.256,
            0.512,
            1.0,
        ),
        pixel_feature_dim: int = 32,
        pixel_subpatch_grid: int = 4,
        pixel_roi_grid: int = 8,
        extent_residual_octaves: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if pixel_feature_dim <= 0 or pixel_subpatch_grid <= 0 or pixel_roi_grid <= 0:
            raise ValueError("High-resolution pixel dimensions must be positive")
        if extent_residual_octaves <= 0:
            raise ValueError("extent_residual_octaves must be positive")
        super().__init__(
            hidden_size=hidden_size,
            vision_hidden_size=vision_hidden_size,
            relay_layer=relay_layer,
            token_ids=token_ids,
            evidence_dim=evidence_dim,
            address_temperature=address_temperature,
            maximum_residual_strength=maximum_residual_strength,
            isolate_controller_gradients=isolate_controller_gradients,
            scale_anchors=scale_anchors,
            **kwargs,
        )
        self.pixel_feature_dim = int(pixel_feature_dim)
        self.pixel_subpatch_grid = int(pixel_subpatch_grid)
        self.pixel_roi_grid = int(pixel_roi_grid)
        self.extent_residual_octaves = float(extent_residual_octaves)

        self.extent_residual_head = nn.Linear(self.evidence_dim * 2, 2)
        nn.init.zeros_(self.extent_residual_head.weight)
        nn.init.zeros_(self.extent_residual_head.bias)
        self.pixel_stem = nn.Sequential(
            nn.Conv2d(3, self.pixel_feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                self.pixel_feature_dim,
                self.pixel_feature_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((self.pixel_subpatch_grid, self.pixel_subpatch_grid)),
        )
        self.pixel_key = nn.Linear(self.pixel_feature_dim, self.evidence_dim, bias=False)
        self.pixel_value = nn.Linear(self.pixel_feature_dim, self.evidence_dim, bias=False)

        # The added pixel evidence expands the common feature from 5E to 6E.
        common_width = self.evidence_dim * 6
        self.presence_head = nn.Sequential(
            nn.Linear(common_width, self.evidence_dim),
            nn.GELU(),
            nn.Linear(self.evidence_dim, 1),
        )
        self.evidence_up = nn.Sequential(
            nn.Linear(common_width + self.evidence_dim, 2 * self.evidence_dim),
            nn.GELU(),
            nn.LayerNorm(2 * self.evidence_dim),
            nn.Linear(2 * self.evidence_dim, self.hidden_size),
        )
        # Preserve the released parent and matched LoRA behavior at initialization.
        nn.init.zeros_(self.evidence_up[-1].weight)
        nn.init.zeros_(self.evidence_up[-1].bias)

        self._pixel_field: torch.Tensor | None = None
        self._pixel_field_hw: tuple[int, int] | None = None
        self._pixel_reentry_reads = 0
        self._presence_frame_rows: tuple[int, ...] = ()
        self._presence_logits: tuple[torch.Tensor, ...] = ()
        self._last_presence_probability = 0.5

    def _reset_state(self) -> None:
        super()._reset_state()
        self._pixel_field = None
        self._pixel_field_hw = None
        self._pixel_reentry_reads = 0
        self._presence_frame_rows = ()
        self._presence_logits = ()
        self._last_presence_probability = 0.5

    def finish_forward(self) -> None:
        super().finish_forward()
        self._pixel_field = None
        self._pixel_field_hw = None
        self._pixel_reentry_reads = 0
        self._presence_frame_rows = ()
        self._presence_logits = ()
        self._last_presence_probability = 0.5

    def prepare_pixel_field(
        self, pixel_values: torch.Tensor, image_grid_hws: Any
    ) -> None:
        """Build one differentiable subpatch field from the existing image tensor."""

        self._set_grid(image_grid_hws)
        if self._grid_hw is None:
            raise RuntimeError("Image grid is unavailable")
        height, width = self._grid_hw
        if pixel_values.ndim != 4 or int(pixel_values.shape[0]) != height * width:
            raise RuntimeError(
                "Pixel/grid mismatch: "
                f"pixels={tuple(pixel_values.shape)} grid={(height, width)}"
            )
        stem_dtype = next(self.pixel_stem.parameters()).dtype
        features = self.pixel_stem(pixel_values.to(dtype=stem_dtype))
        features = features.permute(0, 2, 3, 1).reshape(
            height,
            width,
            self.pixel_subpatch_grid,
            self.pixel_subpatch_grid,
            self.pixel_feature_dim,
        )
        features = features.permute(0, 2, 1, 3, 4).reshape(
            height * self.pixel_subpatch_grid,
            width * self.pixel_subpatch_grid,
            self.pixel_feature_dim,
        )
        self._pixel_field = features.permute(2, 0, 1).unsqueeze(0).contiguous()
        self._pixel_field_hw = (int(features.shape[0]), int(features.shape[1]))

    def _predict_extent(
        self, scale_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        width_logits = self.width_head(scale_input).float()
        height_logits = self.height_head(scale_input).float()
        anchors = self.scale_anchors.to(width_logits.device).clamp_min(1e-4)
        log_anchors = anchors.log()
        residual = torch.tanh(self.extent_residual_head(scale_input).float())
        residual = residual * (math.log(2.0) * self.extent_residual_octaves)
        log_width = torch.dot(width_logits.softmax(-1), log_anchors) + residual[0]
        log_height = torch.dot(height_logits.softmax(-1), log_anchors) + residual[1]
        width = log_width.exp().clamp(0.002, 1.0)
        height = log_height.exp().clamp(0.002, 1.0)
        return width, height, width_logits, height_logits

    def _pixel_evidence(
        self,
        *,
        query: torch.Tensor,
        point: torch.Tensor,
        extent_width: torch.Tensor,
        extent_height: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self._pixel_field is None or self._pixel_field_hw is None:
            raise RuntimeError("High-resolution pixel field was not prepared")
        route_dtype = self.pixel_key.weight.dtype
        offsets = torch.linspace(
            -0.5,
            0.5,
            self.pixel_roi_grid,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        sample_x = point[0].float() + xx * extent_width
        sample_y = point[1].float() + yy * extent_height
        sample_grid = torch.stack((sample_x * 2.0 - 1.0, sample_y * 2.0 - 1.0), dim=-1)
        sampled = F.grid_sample(
            self._pixel_field.to(device=hidden_states.device, dtype=route_dtype),
            sample_grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled = sampled.squeeze(0).permute(1, 2, 0).reshape(-1, self.pixel_feature_dim)
        keys = F.normalize(self.pixel_key(sampled).float(), dim=-1)
        values = self.pixel_value(sampled)
        probabilities = (torch.mv(keys, query) / self.address_temperature).softmax(0)
        self._pixel_reentry_reads += 1
        return torch.matmul(probabilities.to(route_dtype), values)

    def _route(self, hidden_states: torch.Tensor, context: RavelContext) -> torch.Tensor:
        if self._prompt_point is None:
            return hidden_states
        if self._global_projected is None or self._dense_projected is None:
            return hidden_states
        events = tuple(
            event
            for event in context.events
            if event.geometry in {"hbb", "none"}
            and event.row
            and max(event.row) < hidden_states.shape[1]
        )
        if not events:
            return hidden_states
        if self._grid_hw is None or self._dense_hw is None:
            raise RuntimeError("Visual grid is unavailable")
        if self._pixel_field is None:
            raise RuntimeError("Point-address route requires the high-resolution pixel field")

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
            [self._prompt_point.x / 1000.0, self._prompt_point.y / 1000.0], dtype=dtype
        )
        point_feature = self.point_proj(point)
        updated = hidden_states.clone()
        address_losses: list[torch.Tensor] = []
        presence_logits: list[torch.Tensor] = []
        presence_frame_rows: list[int] = []
        presence_targets: list[float] = []
        scale_losses: list[torch.Tensor] = []
        support_hits: list[torch.Tensor] = []
        expected_widths: list[torch.Tensor] = []
        expected_heights: list[torch.Tensor] = []
        strengths = self.residual_strength().to(hidden_states.device, hidden_states.dtype)

        for event in events:
            query_state = hidden_states[0, event.row[0]]
            if self._training_context and self.isolate_controller_gradients:
                query_state = query_state.detach()
            query = self.query_proj(
                self.query_norm(query_state.to(self.query_norm.weight.dtype)).to(dtype)
            )
            query = F.normalize(query.float(), dim=-1)
            scale_input = torch.cat((query.to(dtype), point_feature), dim=-1)
            expected_width, expected_height, width_logits, height_logits = (
                self._predict_extent(scale_input)
            )
            expected_widths.append(expected_width)
            expected_heights.append(expected_height)
            scale_feature = self.scale_proj(
                torch.stack((expected_width, expected_height)).to(dtype)
            )

            global_semantic = torch.mv(global_keys, query) / self.address_temperature
            global_distance = (global_x - point[0].float()).square() + (
                global_y - point[1].float()
            ).square()
            global_probability = (global_semantic - global_distance / 0.5**2).softmax(0)
            global_evidence = torch.matmul(global_probability.to(dtype), global_values)

            sigma_x = torch.maximum(expected_width * 0.5, expected_width.new_tensor(0.5 / dw))
            sigma_y = torch.maximum(expected_height * 0.5, expected_height.new_tensor(0.5 / dh))
            dense_prior = -0.5 * (
                ((dense_x - point[0].float()) / sigma_x).square()
                + ((dense_y - point[1].float()) / sigma_y).square()
            )
            dense_semantic = torch.mv(dense_keys, query) / self.address_temperature
            dense_probability = (dense_semantic + dense_prior).softmax(0)
            dense_evidence = torch.matmul(dense_probability.to(dtype), dense_values)
            pixel_evidence = self._pixel_evidence(
                query=query,
                point=point,
                extent_width=expected_width,
                extent_height=expected_height,
                hidden_states=hidden_states,
            )

            common = torch.cat(
                (
                    query.to(dtype),
                    point_feature,
                    global_evidence,
                    dense_evidence,
                    pixel_evidence,
                    scale_feature,
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
                x1, y1, x2, y2 = event.coordinates
                point_x = float(self._prompt_point.x)
                point_y = float(self._prompt_point.y)
                # The ROI is centered at the address, not necessarily at the box center.
                target_width = max(
                    0.002,
                    2.0 * max(abs(point_x - x1), abs(point_x - x2)) / 1000.0,
                )
                target_height = max(
                    0.002,
                    2.0 * max(abs(point_y - y1), abs(point_y - y2)) / 1000.0,
                )
                anchors = self.scale_anchors.to(width_logits.device).clamp_min(1e-4)
                width_target = torch.argmin((anchors.log() - math.log(target_width)).abs()).view(1)
                height_target = torch.argmin((anchors.log() - math.log(target_height)).abs()).view(1)
                classification = F.cross_entropy(width_logits.view(1, -1), width_target)
                classification = classification + F.cross_entropy(
                    height_logits.view(1, -1), height_target
                )
                regression = F.smooth_l1_loss(
                    torch.stack((expected_width.log(), expected_height.log())),
                    hidden_states.new_tensor(
                        [math.log(target_width), math.log(target_height)],
                        dtype=torch.float32,
                    ),
                )
                scale_losses.append(regression + 0.25 * classification)
                support = self._dense_support(event.coordinates, hidden_states.device)
                mass = torch.dot(dense_probability, support)
                address_losses.append(-mass.clamp_min(1e-8).log())
                support_hits.append(support[int(dense_probability.argmax())].gt(0).float())

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
        self.last_diagnostics = {
            "highres_events": float(len(events)),
            "highres_vision_captures": float(self.vision_capture_count),
            "highres_global_tokens": float(global_tokens.shape[0]),
            "highres_dense_tokens": float(dense_tokens.shape[0]),
            "highres_pixel_cells": float(
                0
                if self._pixel_field_hw is None
                else self._pixel_field_hw[0] * self._pixel_field_hw[1]
            ),
            "highres_roi_cells_per_event": float(self.pixel_roi_grid**2),
            "highres_pixel_reentry_reads": float(self._pixel_reentry_reads),
            "highres_dense_tokens_in_qwen": 0.0,
            "highres_address_loss": 0.0
            if self._address_loss is None
            else float(self._address_loss.detach()),
            "highres_presence_loss": 0.0
            if self._presence_loss is None
            else float(self._presence_loss.detach()),
            "highres_scale_loss": 0.0
            if self._scale_loss is None
            else float(self._scale_loss.detach()),
            "highres_support_hit_rate": 0.0
            if not support_hits
            else float(torch.stack(support_hits).mean().detach()),
            "highres_expected_width": float(torch.stack(expected_widths).mean().detach()),
            "highres_expected_height": float(torch.stack(expected_heights).mean().detach()),
            "highres_residual_strength": float(strengths[:4].mean().detach()),
            "highres_presence_probability": float(
                self._last_presence_probability
            ),
        }
        return updated

    def inference_contract(self) -> dict[str, Any]:
        contract = super().inference_contract()
        contract.update(
            {
                "method": "high-resolution point-conditioned evidence PBD",
                "extent_anchors": [float(value) for value in self.scale_anchors.tolist()],
                "continuous_log_extent_residual": True,
                "address_centered_extent": True,
                "fresh_original_patch_pixels": True,
                "pixel_field_encodings_per_image": 1,
                "pixel_subpatch_grid": self.pixel_subpatch_grid,
                "pixel_roi_grid": self.pixel_roi_grid,
                "scale_adaptive_fixed_token_roi": True,
                "crop_reencoding": False,
                "second_moonvit_pass": False,
            }
        )
        return contract
