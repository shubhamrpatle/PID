"""Open-ended point-indexed pixel re-entry for LocateAnything.

OPIVR extends the single-stream RAVEL controller with a compact trainable pixel
field.  MoonViT and Qwen are still each executed once per image/query.  At every
native geometry event, Qwen's learned soft point address reads both cached
pre-merge MoonViT evidence and unmerged features derived directly from the
original processor patches.  The retrieved evidence is written back only to
the native PBD geometry lanes, while learned presence/continuation state and
the released EOS grammar keep the number of emitted objects open-ended.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .pointrelay_pbd import configure_lora_only, restore_trainable_state
from .ravel_pbd import RavelPBDController


class OpenEndedPointIndexedVisualReentryPBD(RavelPBDController):
    """Read fine original-pixel evidence at each learned recurrent address."""

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
        pixel_feature_dim: int = 32,
        pixel_subpatch_grid: int = 4,
        pixel_local_sigmas: Sequence[float] = (0.75, 1.5, 3.0),
    ) -> None:
        if pixel_feature_dim <= 0 or pixel_subpatch_grid <= 0:
            raise ValueError("OPIVR pixel dimensions must be positive")
        if not pixel_local_sigmas or any(value <= 0 for value in pixel_local_sigmas):
            raise ValueError("OPIVR pixel_local_sigmas must be positive")
        self.pixel_feature_dim = int(pixel_feature_dim)
        self.pixel_subpatch_grid = int(pixel_subpatch_grid)
        self.pixel_local_sigmas = tuple(float(value) for value in pixel_local_sigmas)
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
            additional_local_value_count=len(self.pixel_local_sigmas),
        )
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
            nn.AdaptiveAvgPool2d(
                (self.pixel_subpatch_grid, self.pixel_subpatch_grid)
            ),
        )
        self.pixel_value_proj = nn.Linear(
            self.pixel_feature_dim, self.evidence_dim, bias=False
        )
        self._pixel_tokens: torch.Tensor | None = None
        self._pixel_positions: torch.Tensor | None = None
        self._pixel_reentry_reads = 0

    def _reset_state(self) -> None:
        super()._reset_state()
        self._pixel_tokens = None
        self._pixel_positions = None
        self._pixel_reentry_reads = 0

    def finish_forward(self) -> None:
        super().finish_forward()
        self._pixel_tokens = None
        self._pixel_positions = None
        self._pixel_reentry_reads = 0

    def _record_runtime_diagnostics(self) -> None:
        super()._record_runtime_diagnostics()
        self.last_diagnostics.update(
            {
                "opivr_pixel_tokens": float(
                    0 if self._pixel_tokens is None else self._pixel_tokens.shape[0]
                ),
                "opivr_pixel_reentry_reads": float(self._pixel_reentry_reads),
            }
        )

    def prepare_pixel_field(
        self, pixel_values: torch.Tensor, image_grid_hws: Any
    ) -> None:
        """Encode an unmerged subpatch lattice from the original image patches."""

        self._set_grid(image_grid_hws)
        if self._grid_hw is None:
            raise RuntimeError("OPIVR image grid is unavailable")
        height, width = self._grid_hw
        if int(pixel_values.shape[0]) != height * width:
            raise RuntimeError(
                "OPIVR pixel/grid mismatch: "
                f"{int(pixel_values.shape[0])} != {height * width}"
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
        self._pixel_tokens = features.reshape(-1, self.pixel_feature_dim)
        pixel_h, pixel_w = features.shape[:2]
        yy, xx = torch.meshgrid(
            (
                torch.arange(
                    pixel_h, device=pixel_values.device, dtype=torch.float32
                )
                + 0.5
            )
            / pixel_h,
            (
                torch.arange(
                    pixel_w, device=pixel_values.device, dtype=torch.float32
                )
                + 0.5
            )
            / pixel_w,
            indexing="ij",
        )
        self._pixel_positions = torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def _additional_local_values(
        self,
        *,
        probability: torch.Tensor,
        predicted_point: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        del probability
        if self._pixel_tokens is None or self._pixel_positions is None:
            raise RuntimeError("OPIVR pixel field was not prepared")
        if self._grid_hw is None:
            raise RuntimeError("OPIVR image grid is unavailable")
        route_dtype = self.pixel_value_proj.weight.dtype
        values = self.pixel_value_proj(
            self._pixel_tokens.to(device=hidden_states.device, dtype=route_dtype)
        )
        positions = self._pixel_positions.to(hidden_states.device)
        pixel_h = self._grid_hw[0] * self.pixel_subpatch_grid
        pixel_w = self._grid_hw[1] * self.pixel_subpatch_grid
        dx = (positions[:, 0] - predicted_point[0]) * pixel_w
        dy = (positions[:, 1] - predicted_point[1]) * pixel_h
        outputs: list[torch.Tensor] = []
        for sigma in self.pixel_local_sigmas:
            weights = (
                -0.5 * (dx.square() + dy.square()) / (sigma * sigma)
            ).softmax(dim=0)
            outputs.append(torch.matmul(weights.to(route_dtype), values))
        self._pixel_reentry_reads += 1
        return tuple(outputs)

    def inference_contract(self) -> dict[str, Any]:
        contract = super().inference_contract()
        contract.update(
            {
                "method": "OPIVR-LA",
                "fresh_original_pixel_evidence": True,
                "pixel_field_encodings_per_image": 1,
                "pixel_subpatch_grid": self.pixel_subpatch_grid,
                "point_indexed_pixel_reentry": True,
                "crop_reencoding": False,
                "open_ended_continue_stop": True,
            }
        )
        return contract


def load_opivr_adapter(model: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load OPIVR controller tensors and their matched Qwen LoRA weights."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    configure_lora_only(model, int(config["lora_rank"]))
    token_ids = {**model.token_ids, "image_token_index": int(model.image_token_index)}
    token_ids.update(
        {
            str(key): int(value)
            for key, value in config.get("extra_token_ids", {}).items()
        }
    )
    controller = OpenEndedPointIndexedVisualReentryPBD(
        hidden_size=int(model.config.text_config.hidden_size),
        vision_hidden_size=int(model.config.vision_config.hidden_size),
        relay_layer=int(config["relay_layer"]),
        token_ids=token_ids,
        evidence_dim=int(config.get("evidence_dim", 128)),
        address_temperature=float(config.get("address_temperature", 0.10)),
        local_sigmas=tuple(config.get("local_sigmas", (0.75, 1.5, 3.0))),
        coverage_penalty=float(config.get("coverage_penalty", 2.0)),
        maximum_residual_strength=float(
            config.get("maximum_residual_strength", 1.0)
        ),
        isolate_controller_gradients=bool(
            config.get("isolate_controller_gradients", True)
        ),
        routing_control=str(config.get("routing_control", "learned")),
        cached_crop_radius_cells=float(
            config.get("cached_crop_radius_cells", 1.5)
        ),
        pixel_feature_dim=int(config.get("pixel_feature_dim", 32)),
        pixel_subpatch_grid=int(config.get("pixel_subpatch_grid", 4)),
        pixel_local_sigmas=tuple(
            config.get("pixel_local_sigmas", (0.75, 1.5, 3.0))
        ),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {
        "checkpoint": checkpoint,
        "controller": controller,
        "inference_contract": controller.inference_contract(),
    }
