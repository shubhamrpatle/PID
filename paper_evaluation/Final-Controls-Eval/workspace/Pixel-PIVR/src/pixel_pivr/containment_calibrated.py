"""Containment-calibrated pixel re-entry for Pixel-PIVR.

The controller predicts the smallest architecture-aligned field of view needed
to contain the complete object addressed by a class/point pair.  It operates on
the already projected global feature grid, so scale planning does not add a
second global MoonViT pass.  The selected pixel crop is then re-encoded once and
decoded with Pixel-PIVR's native one-address/one-PBD6 contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from .decoder import AddressedCrop, PixelPIVRWaveDecoder, image_patch_lengths
from .lora import base_causal_lm, restore_trainable_state
from .modeling import chunked_cross_entropy, weighted_training_objective


CONTAINMENT_CONTEXT = "containment_calibrated_pixel_reentry"
DEFAULT_SCALE_RATIOS = (
    0.0703125,
    0.09943689110435824,
    0.140625,
    0.1988737822087165,
    0.28125,
    0.397747564417433,
    0.5625,
    0.795495128834866,
    1.0,
)
DEFAULT_RING_RADII = (0, 1, 3, 7)


def parse_float_csv(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("Expected comma-separated floating-point values") from exc
    if len(parsed) < 3 or any(not 0.0 < item <= 1.0 for item in parsed):
        raise ValueError("Scale ratios require at least three values in (0,1]")
    if tuple(sorted(set(parsed))) != parsed or not math.isclose(parsed[-1], 1.0):
        raise ValueError("Scale ratios must be unique, increasing, and end at 1")
    return parsed


def parse_nonnegative_int_csv(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("Expected comma-separated integer radii") from exc
    if not parsed or any(item < 0 for item in parsed):
        raise ValueError("Ring radii must be non-negative")
    if tuple(sorted(set(parsed))) != parsed:
        raise ValueError("Ring radii must be unique and increasing")
    return parsed


def candidate_sides(
    width: int,
    height: int,
    ratios: Sequence[float],
    *,
    reference_pixels: int = 0,
) -> list[int]:
    """Materialize an ordered crop ladder for one source image.

    ``reference_pixels == 0`` preserves the original relative-to-image policy.
    A positive reference size anchors the ladder in source pixels before clipping
    it to the image boundary.  The latter is required for a true fixed-pixel
    magnification floor across mixed 600/800/1024-pixel source imagery.
    """
    maximum = max(int(width), int(height))
    if maximum <= 0:
        raise ValueError("Image dimensions must be positive")
    basis = int(reference_pixels) if int(reference_pixels) > 0 else maximum
    values = [
        min(maximum, max(2, int(round(float(ratio) * basis))))
        for ratio in ratios
    ]
    values[-1] = maximum
    return values


def point_centered_rectangular_crop(
    point_xy: Sequence[float], width: int, height: int, side: int
) -> tuple[int, int, int, int]:
    """Return an unpadded point-centred crop, shifting it at image boundaries."""
    crop_width = min(int(side), int(width))
    crop_height = min(int(side), int(height))
    center_x, center_y = float(point_xy[0]), float(point_xy[1])
    left = max(0, min(width - crop_width, int(round(center_x - crop_width / 2))))
    top = max(0, min(height - crop_height, int(round(center_y - crop_height / 2))))
    return left, top, left + crop_width, top + crop_height


def _raw_row(base: Any, index: int) -> Mapping[str, Any]:
    if not hasattr(base, "lazy_loader") or not hasattr(base, "active_indices"):
        raise TypeError("Expected a LazySupervisedDatasetMTP-compatible dataset")
    return base.lazy_loader[base.active_indices[index]]


class ContainmentScaleMetadataDataset(Dataset):
    """Attach scale targets and class tokens without altering signed JSONL rows."""

    def __init__(self, base: Dataset, tokenizer: Any, scale_count: int) -> None:
        self.base = base
        self.tokenizer = tokenizer
        self.scale_count = int(scale_count)
        if self.scale_count < 3:
            raise ValueError("Containment calibration needs at least three scales")
        if not hasattr(base, "lazy_loader") or not hasattr(base, "active_indices"):
            raise TypeError("Expected a LazySupervisedDatasetMTP-compatible dataset")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base[index])
        row = _raw_row(self.base, index)
        meta = dict(row.get("meta") or {})
        enabled = str(meta.get("pivr_route") or "") == "point_indexed_visual_reentry"
        point = meta.get("pivr_global_point") if enabled else (0, 0)
        categories = meta.get("query_categories") or []
        label = str(categories[0]).strip() if enabled and categories else ""
        if enabled and (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or any(not 0 <= float(value) <= 1000 for value in point)
            or not label
        ):
            raise RuntimeError(f"Invalid containment metadata at row {index}")
        target = int(meta.get("pivr_scale_target_index", -100)) if enabled else -100
        supervised = bool(meta.get("pivr_scale_supervised", False)) if enabled else False
        if supervised != (0 <= target < self.scale_count):
            raise RuntimeError(
                f"Scale target/supervision mismatch at row {index}: {target}, {supervised}"
            )
        images = row.get("image")
        if enabled and (not isinstance(images, list) or len(images) != 2):
            raise RuntimeError("Containment re-entry requires [global, local] images")
        if not enabled and not isinstance(images, str):
            raise RuntimeError("Global discovery requires one image")
        label_ids = self.tokenizer.encode(label or "object", add_special_tokens=False)
        if not label_ids:
            raise RuntimeError(f"Class label tokenized to an empty sequence: {label!r}")
        sample["containment_scale_enabled"] = torch.tensor(enabled, dtype=torch.bool)
        sample["containment_scale_supervised"] = torch.tensor(supervised, dtype=torch.bool)
        sample["containment_scale_target"] = torch.tensor(target, dtype=torch.long)
        sample["containment_global_point"] = torch.tensor(point, dtype=torch.float32)
        sample["containment_class_token_ids"] = torch.tensor(label_ids, dtype=torch.long)
        return sample


@dataclass(frozen=True)
class ScalePrediction:
    logits: torch.Tensor
    center: torch.Tensor
    spread: torch.Tensor


class OrderedContainmentScaleController(nn.Module):
    """Predict a unimodal ordered distribution over nested crop scales."""

    def __init__(
        self,
        hidden_size: int,
        scale_ratios: Sequence[float] = DEFAULT_SCALE_RATIOS,
        *,
        ring_radii: Sequence[int] = DEFAULT_RING_RADII,
        descriptor_hidden: int = 128,
        controller_hidden: int = 256,
        preferred_scale: int = 2,
        minimum_scale: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.scale_ratios = tuple(float(value) for value in scale_ratios)
        self.ring_radii = tuple(int(value) for value in ring_radii)
        self.descriptor_hidden = int(descriptor_hidden)
        self.controller_hidden = int(controller_hidden)
        self.preferred_scale = int(preferred_scale)
        self.minimum_scale = int(minimum_scale)
        if self.hidden_size <= 0 or self.descriptor_hidden <= 0 or self.controller_hidden <= 0:
            raise ValueError("Controller dimensions must be positive")
        if not 0 <= self.preferred_scale < len(self.scale_ratios):
            raise ValueError("Preferred scale is outside the scale ladder")
        if not 0 <= self.minimum_scale < len(self.scale_ratios):
            raise ValueError("Minimum scale is outside the scale ladder")
        if self.preferred_scale < self.minimum_scale:
            raise ValueError("Preferred scale cannot be below the minimum scale")
        parse_float_csv(",".join(str(value) for value in self.scale_ratios))
        parse_nonnegative_int_csv(",".join(str(value) for value in self.ring_radii))

        # One shared projection is deliberate: projected vision tokens and Qwen
        # class embeddings already occupy the same decoder-width feature space.
        self.descriptor_projection = nn.Linear(
            self.hidden_size, self.descriptor_hidden, bias=False
        )
        descriptor_count = len(self.ring_radii) + 2  # local windows, global, class
        position_width = 6
        self.fusion = nn.Sequential(
            nn.Linear(descriptor_count * self.descriptor_hidden + position_width, self.controller_hidden),
            nn.SiLU(),
            nn.LayerNorm(self.controller_hidden),
            nn.Linear(self.controller_hidden, self.controller_hidden),
            nn.SiLU(),
        )
        self.location_head = nn.Linear(self.controller_hidden, 1)
        self.spread_head = nn.Linear(self.controller_hidden, 1)
        nn.init.xavier_uniform_(self.descriptor_projection.weight)
        for module in self.fusion:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        scale_fraction = self.preferred_scale / max(1, len(self.scale_ratios) - 1)
        # A scale ladder may deliberately start at the proven fixed-crop
        # baseline.  Keep endpoint initializations finite while making index 0
        # (or the final index) overwhelmingly likely before learning.
        scale_fraction = min(1.0 - 1e-4, max(1e-4, scale_fraction))
        location_bias = math.log(scale_fraction / (1.0 - scale_fraction))
        nn.init.normal_(self.location_head.weight, std=0.01)
        nn.init.constant_(self.location_head.bias, location_bias)
        nn.init.normal_(self.spread_head.weight, std=0.01)
        nn.init.constant_(self.spread_head.bias, 0.0)

    @property
    def scale_count(self) -> int:
        return len(self.scale_ratios)

    @staticmethod
    def _grid_index(value: float, size: int) -> int:
        clipped = max(0.0, min(1000.0, float(value)))
        return min(size - 1, int(clipped * size / 1000.0))

    def descriptors(
        self,
        projected_global: torch.Tensor,
        grid_hw: Sequence[int],
        point_normalized_xy: Sequence[float] | torch.Tensor,
        class_embedding: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        height, width = (int(value) for value in grid_hw)
        if height <= 0 or width <= 0 or int(projected_global.shape[0]) != height * width:
            raise ValueError("Projected global features do not match their grid")
        if int(projected_global.shape[-1]) != self.hidden_size:
            raise ValueError("Projected feature width differs from controller width")
        point = torch.as_tensor(
            point_normalized_xy,
            dtype=torch.float32,
            device=projected_global.device,
        ).flatten()
        if point.numel() != 2:
            raise ValueError("A containment address must contain normalized x and y")
        row = self._grid_index(float(point[1]), height)
        col = self._grid_index(float(point[0]), width)
        grid = projected_global.view(height, width, self.hidden_size)
        values: list[torch.Tensor] = []
        for radius in self.ring_radii:
            top, bottom = max(0, row - radius), min(height, row + radius + 1)
            left, right = max(0, col - radius), min(width, col + radius + 1)
            values.append(grid[top:bottom, left:right].float().mean(dim=(0, 1)))
        values.append(projected_global.float().mean(dim=0))
        class_value = class_embedding.float().reshape(-1, self.hidden_size).mean(dim=0)
        values.append(class_value)
        xy = point.clamp(0, 1000) / 1000.0
        position = torch.stack(
            (
                2.0 * xy[0] - 1.0,
                2.0 * xy[1] - 1.0,
                torch.sin(math.pi * xy[0]),
                torch.cos(math.pi * xy[0]),
                torch.sin(math.pi * xy[1]),
                torch.cos(math.pi * xy[1]),
            )
        )
        return values, position

    def forward(
        self,
        projected_global: torch.Tensor,
        grid_hw: Sequence[int],
        point_normalized_xy: Sequence[float] | torch.Tensor,
        class_embedding: torch.Tensor,
    ) -> ScalePrediction:
        descriptors, position = self.descriptors(
            projected_global, grid_hw, point_normalized_xy, class_embedding
        )
        projected = [
            F.silu(self.descriptor_projection(F.layer_norm(value, value.shape)))
            for value in descriptors
        ]
        hidden = self.fusion(torch.cat((*projected, position), dim=0))
        center = (self.scale_count - 1) * torch.sigmoid(self.location_head(hidden)).squeeze(-1)
        spread = F.softplus(self.spread_head(hidden)).squeeze(-1) + 0.35
        indices = torch.arange(
            self.scale_count, dtype=center.dtype, device=center.device
        )
        logits = -0.5 * ((indices - center) / spread).square() - torch.log(spread)
        return ScalePrediction(logits=logits, center=center, spread=spread)

    def loss(
        self,
        prediction: ScalePrediction,
        target_index: int | torch.Tensor,
        *,
        regression_weight: float,
        undercrop_weight: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        target = torch.as_tensor(
            int(torch.as_tensor(target_index).item()),
            dtype=torch.long,
            device=prediction.logits.device,
        )
        if not 0 <= int(target) < self.scale_count:
            raise ValueError(f"Scale target is outside the ladder: {int(target)}")
        ce = F.cross_entropy(prediction.logits.unsqueeze(0), target.unsqueeze(0))
        target_value = target.to(dtype=prediction.center.dtype)
        regression = F.smooth_l1_loss(prediction.center, target_value)
        undercrop = F.relu(target_value - prediction.center).square()
        total = ce + float(regression_weight) * regression + float(undercrop_weight) * undercrop
        selected = self.select_index(prediction.center)
        return total, {
            "scale_ce": float(ce.detach()),
            "scale_regression": float(regression.detach()),
            "scale_under": float(undercrop.detach()),
            "scale_center": float(prediction.center.detach()),
            "scale_spread": float(prediction.spread.detach()),
            "scale_abs_error": float((prediction.center.detach() - target_value).abs()),
            "scale_exact": float(selected == int(target)),
        }

    def select_index(self, center: float | torch.Tensor) -> int:
        # Ceiling is an asymmetric, containment-preserving decision rule: a small
        # overestimate keeps extra context, while an underestimate can clip GT.
        value = float(torch.as_tensor(center).detach().cpu())
        return max(
            self.minimum_scale,
            min(self.scale_count - 1, int(math.ceil(value - 1e-6))),
        )

    def select_prediction(
        self,
        prediction: ScalePrediction,
        *,
        rule: str = "ceiling",
        ordinal_thresholds: Sequence[float] | None = None,
    ) -> int:
        """Turn the learned ordered distribution into one deployable crop scale.

        ``ceiling`` preserves the historical asymmetric decision rule. ``map``
        is the distribution-consistent rule used by the calibrated decoder: it
        selects the maximum-probability scale after the training objective has
        already imposed its explicit under-crop penalty.
        """

        if rule == "ceiling":
            return self.select_index(prediction.center)
        if rule == "map":
            selected = int(torch.argmax(prediction.logits.detach()).cpu())
            return max(self.minimum_scale, min(self.scale_count - 1, selected))
        if rule == "ordinal_thresholds":
            thresholds = tuple(float(value) for value in (ordinal_thresholds or ()))
            if len(thresholds) != self.scale_count - 1:
                raise ValueError(
                    "Ordinal selection requires one boundary between every pair "
                    "of crop scales"
                )
            if tuple(sorted(set(thresholds))) != thresholds:
                raise ValueError("Ordinal crop-scale boundaries must be unique and increasing")
            value = float(prediction.center.detach().cpu())
            selected = sum(value >= boundary for boundary in thresholds)
            return max(self.minimum_scale, min(self.scale_count - 1, selected))
        raise ValueError(f"Unsupported containment selection rule: {rule!r}")


def merged_grid_hw(image_grid_hw: Sequence[int] | torch.Tensor) -> tuple[int, int]:
    values = torch.as_tensor(image_grid_hw, dtype=torch.long).flatten()
    if values.numel() != 2 or int(values[0]) % 2 or int(values[1]) % 2:
        raise ValueError(f"Expected an even MoonViT grid, got {values.tolist()}")
    return int(values[0] // 2), int(values[1] // 2)


class ContainmentCalibratedTrainingModel(nn.Module):
    """Qwen LoRA plus a supervised point-conditioned crop-scale controller."""

    def __init__(
        self,
        model: nn.Module,
        *,
        scale_ratios: Sequence[float],
        ring_radii: Sequence[int],
        descriptor_hidden: int,
        controller_hidden: int,
        preferred_scale: int,
        minimum_scale: int,
        scale_loss_weight: float,
        scale_regression_weight: float,
        scale_undercrop_weight: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.scale_loss_weight = float(scale_loss_weight)
        self.scale_regression_weight = float(scale_regression_weight)
        self.scale_undercrop_weight = float(scale_undercrop_weight)
        hidden_size = int(base_causal_lm(model.language_model).config.hidden_size)
        self.controller = OrderedContainmentScaleController(
            hidden_size,
            scale_ratios,
            ring_radii=ring_radii,
            descriptor_hidden=descriptor_hidden,
            controller_hidden=controller_hidden,
            preferred_scale=preferred_scale,
            minimum_scale=minimum_scale,
        ).to(device=next(model.parameters()).device)
        model.pixel_pivr_controller = self.controller

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.vision_model.eval()
        self.model.mlp1.eval()
        self.controller.train(mode)
        return self

    def forward(self, sample: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        device = next(self.model.parameters()).device
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)
        position_ids = sample["position_ids"].unsqueeze(0).to(device)
        pixel_values = sample["pixel_values"].to(device=device, dtype=torch.bfloat16)
        image_grid_hws = torch.as_tensor(
            sample["image_grid_hws"], device=device, dtype=torch.int32
        )
        with torch.no_grad():
            visual = self.model.extract_feature(pixel_values, image_grid_hws)
            projected_images = [self.model.mlp1(value) for value in visual]
        projected = torch.cat(projected_images, dim=0)
        selected = input_ids.eq(int(self.model.image_token_index))
        if int(selected.sum()) != int(projected.shape[0]):
            raise RuntimeError("Containment-calibrated visual placeholder mismatch")
        embeddings = self.model.language_model.get_input_embeddings()(input_ids)
        flat = embeddings.reshape(-1, embeddings.shape[-1]).clone()
        flat[selected.reshape(-1)] = projected
        embeddings = flat.view_as(embeddings)
        causal = base_causal_lm(self.model.language_model)
        outputs = causal.model(
            input_ids=None,
            inputs_embeds=embeddings,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        native = chunked_cross_entropy(outputs.last_hidden_state, labels, causal.lm_head.weight)
        objective, metrics = weighted_training_objective(native, sample)
        enabled = bool(torch.as_tensor(sample["containment_scale_enabled"]).item())
        supervised = bool(torch.as_tensor(sample["containment_scale_supervised"]).item())
        metrics.update(
            {
                "scale_active": float(enabled),
                "scale_supervised": float(supervised),
                "scale_loss": 0.0,
                "scale_ce": 0.0,
                "scale_regression": 0.0,
                "scale_under": 0.0,
                "scale_center": 0.0,
                "scale_spread": 0.0,
                "scale_abs_error": 0.0,
                "scale_exact": 0.0,
            }
        )
        if supervised:
            class_ids = sample["containment_class_token_ids"].to(device)
            with torch.no_grad():
                class_embedding = self.model.language_model.get_input_embeddings()(class_ids)
            prediction = self.controller(
                projected_images[0].detach(),
                merged_grid_hw(image_grid_hws[0]),
                sample["containment_global_point"],
                class_embedding.detach(),
            )
            scale_loss, scale_metrics = self.controller.loss(
                prediction,
                sample["containment_scale_target"],
                regression_weight=self.scale_regression_weight,
                undercrop_weight=self.scale_undercrop_weight,
            )
            weighted_scale = self.scale_loss_weight * float(
                sample.get("pixel_pivr_loss_weight", 1.0)
            ) * scale_loss
            objective = objective + weighted_scale
            metrics.update(scale_metrics)
            metrics["scale_loss"] = float(scale_loss.detach())
            metrics["loss"] = float(objective.detach())
        return objective, metrics


class ContainmentCalibratedWaveDecoder(PixelPIVRWaveDecoder):
    """Plan learned variable pixel crops and decode them with one shared prefix."""

    def __init__(
        self,
        worker: Any,
        *,
        image_token_limit: int,
        prefix_cache_mode: str,
        geometry_prefix_mode: str,
        scale_selection_rule: str = "ceiling",
        scale_selection_thresholds: Sequence[float] | None = None,
    ) -> None:
        super().__init__(
            worker,
            image_token_limit=image_token_limit,
            prefix_cache_mode=prefix_cache_mode,
            geometry_prefix_mode=geometry_prefix_mode,
        )
        config = worker.checkpoint.get("config") or {}
        ratios = config.get("containment_scale_ratios") or DEFAULT_SCALE_RATIOS
        rings = config.get("containment_ring_radii") or DEFAULT_RING_RADII
        reference_pixels = int(config.get("containment_scale_reference_pixels", 0))
        if reference_pixels < 0:
            raise RuntimeError("containment_scale_reference_pixels cannot be negative")
        if scale_selection_rule not in {"ceiling", "map", "ordinal_thresholds"}:
            raise ValueError(
                "containment scale selection must be one of: ceiling, map, "
                "ordinal_thresholds"
            )
        hidden_size = int(base_causal_lm(self.model.language_model).config.hidden_size)
        self.controller = OrderedContainmentScaleController(
            hidden_size,
            ratios,
            ring_radii=rings,
            descriptor_hidden=int(config.get("containment_descriptor_hidden", 128)),
            controller_hidden=int(config.get("containment_controller_hidden", 256)),
            preferred_scale=int(config.get("containment_preferred_scale", 2)),
            minimum_scale=int(config.get("containment_minimum_scale", 0)),
        ).to(device=self.device)
        self.model.pixel_pivr_controller = self.controller
        restore_trainable_state(
            self.model, worker.checkpoint["trainable_state"], require_controller=True
        )
        self.controller.eval()
        self.scale_ratios = tuple(float(value) for value in ratios)
        self.scale_reference_pixels = reference_pixels
        self.scale_selection_rule = scale_selection_rule
        self.scale_selection_thresholds = tuple(
            float(value) for value in (scale_selection_thresholds or ())
        )
        if self.scale_selection_rule == "ordinal_thresholds":
            if len(self.scale_selection_thresholds) != len(self.scale_ratios) - 1:
                raise ValueError(
                    "ordinal_thresholds requires scale_count - 1 boundaries"
                )
            if tuple(sorted(set(self.scale_selection_thresholds))) != self.scale_selection_thresholds:
                raise ValueError("Ordinal crop-scale boundaries must be unique and increasing")
        self._planned_global: torch.Tensor | None = None
        self._planned_grid: tuple[int, int] | None = None
        self._selection_histogram = [0] * len(self.scale_ratios)

    @torch.no_grad()
    def plan_image(
        self,
        image: Any,
        points: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        self._selection_histogram = [0] * len(self.scale_ratios)
        image_inputs = self.processor.image_processor(images=[image], return_tensors="pt")
        grid = torch.as_tensor(image_inputs["image_grid_hws"], dtype=torch.int32)
        if tuple(grid.shape) != (1, 2):
            raise RuntimeError("Containment planning expects one global image")
        visual = self.model.extract_feature(
            image_inputs["pixel_values"].to(device=self.device, dtype=self.dtype),
            grid.to(device=self.device),
        )
        self._planned_global = self.model.mlp1(visual[0])
        self._planned_grid = tuple(int(value) for value in grid[0].tolist())
        width, height = image.size
        sides = candidate_sides(
            width,
            height,
            self.scale_ratios,
            reference_pixels=self.scale_reference_pixels,
        )
        plans = []
        for point in points:
            label_ids = self.tokenizer.encode(str(point["label"]), add_special_tokens=False)
            if not label_ids:
                raise RuntimeError(f"Empty class tokenization for {point['label']!r}")
            class_embedding = self.model.language_model.get_input_embeddings()(
                torch.as_tensor(label_ids, dtype=torch.long, device=self.device)
            )
            prediction = self.controller(
                self._planned_global,
                merged_grid_hw(grid[0]),
                point["point_norm"],
                class_embedding,
            )
            selected = self.controller.select_prediction(
                prediction,
                rule=self.scale_selection_rule,
                ordinal_thresholds=self.scale_selection_thresholds,
            )
            self._selection_histogram[selected] += 1
            plans.append(
                {
                    "scale_index": selected,
                    "scale_ratio": self.scale_ratios[selected],
                    "scale_side_pixels": sides[selected],
                    "scale_center": float(prediction.center),
                    "scale_spread": float(prediction.spread),
                    "scale_probabilities": torch.softmax(prediction.logits, dim=-1)
                    .float()
                    .cpu()
                    .tolist(),
                }
            )
        return plans

    def _verify_planned_grid(self, inputs: Mapping[str, Any]) -> None:
        grids = torch.as_tensor(inputs["image_grid_hws"], dtype=torch.int32)
        global_grids = grids[0::2]
        if self._planned_grid is None or any(
            tuple(int(value) for value in row.tolist()) != self._planned_grid
            for row in global_grids
        ):
            raise RuntimeError("Planned and decoded global MoonViT grids differ")

    def _project_shared_prefix_and_locals(
        self,
        inputs: Mapping[str, Any],
        patch_lengths: Sequence[int],
        cached_global: torch.Tensor | None,
        branches: Sequence[AddressedCrop],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self._planned_global is None:
            return super()._project_shared_prefix_and_locals(
                inputs, patch_lengths, cached_global, branches
            )
        self._verify_planned_grid(inputs)
        pixels = inputs["pixel_values"]
        chunks = list(torch.split(pixels, list(patch_lengths), dim=0))
        grids = torch.as_tensor(inputs["image_grid_hws"], dtype=torch.int32)
        local_chunks = chunks[1::2]
        local_grids = grids[1::2]
        features = self.model.extract_feature(
            torch.cat(local_chunks, dim=0).to(device=self.device, dtype=self.dtype),
            local_grids.to(device=self.device),
        )
        return self._planned_global, [self.model.mlp1(value) for value in features]

    def _project_cached_global_and_locals(
        self,
        inputs: Mapping[str, Any],
        patch_lengths: Sequence[int],
        cached_global: torch.Tensor | None,
        branches: Sequence[AddressedCrop],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_features, local_features = self._project_shared_prefix_and_locals(
            inputs, patch_lengths, cached_global, branches
        )
        rows = [value for local in local_features for value in (global_features, local)]
        return global_features, torch.cat(rows, dim=0)

    @torch.no_grad()
    def decode_image(self, *args: Any, **kwargs: Any):
        branches = args[1] if len(args) > 1 else kwargs.get("branches", ())
        if not branches:
            return super().decode_image(*args, **kwargs)
        if self._planned_global is None:
            raise RuntimeError("Call plan_image before containment-calibrated decoding")
        try:
            outputs, execution = super().decode_image(*args, **kwargs)
            execution.update(
                {
                    "visual_context": CONTAINMENT_CONTEXT,
                    "containment_scale_ratios": list(self.scale_ratios),
                    "containment_scale_reference_pixels": self.scale_reference_pixels,
                    "containment_scale_selection_rule": self.scale_selection_rule,
                    "containment_scale_selection_thresholds": list(
                        self.scale_selection_thresholds
                    ),
                    "containment_selection_histogram": list(self._selection_histogram),
                    "global_moonvit_encodes": 1,
                    "local_moonvit_encodes": int(execution.get("waves", 0)),
                }
            )
            return outputs, execution
        finally:
            self._planned_global = None
            self._planned_grid = None


__all__ = [
    "CONTAINMENT_CONTEXT",
    "DEFAULT_RING_RADII",
    "DEFAULT_SCALE_RATIOS",
    "ContainmentCalibratedTrainingModel",
    "ContainmentCalibratedWaveDecoder",
    "ContainmentScaleMetadataDataset",
    "OrderedContainmentScaleController",
    "ScalePrediction",
    "candidate_sides",
    "merged_grid_hw",
    "parse_float_csv",
    "parse_nonnegative_int_csv",
    "point_centered_rectangular_crop",
]
