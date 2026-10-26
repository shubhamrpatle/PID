"""Self-inhibiting latent-point visual re-entry for LocateAnything PBD.

Each native geometry transition chooses one query-conditioned visual witness
from a cached fine MoonViT field. During training, a straight-through Gumbel
sample provides a discrete witness with differentiable gradients; inference
uses the maximum-probability witness. The selected witness both conditions the
next native PBD geometry block and writes an inhibition state, so the next
transition searches elsewhere. A shared evidence-vs-STOP sentinel determines
whether the process opens or continues.

The process has no proposal slots, explicit point list, crop loop, decoder
copy, or learned count cap. Cardinality is determined by the learned STOP
event and bounded only by LocateAnything's native sequence budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .evidence_sentinel_reentry_pbd import EvidenceSentinelVisualReentryPBD
from .pointrelay_pbd import configure_lora_only, restore_trainable_state
from .residual_set_reentry_pbd import ResidualSetTransition


class LatentPointProcessVisualReentryPBD(EvidenceSentinelVisualReentryPBD):
    """One latent visual point, one native geometry mark, then STOP or repeat."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vision_hidden_size: int,
        relay_layer: int,
        token_ids: Mapping[str, int],
        evidence_dim: int = 128,
        subpatch_dim: int = 32,
        subpatch_grid: int = 4,
        scale_multipliers: Sequence[float] = (0.75, 1.5, 3.0),
        minimum_sigma_cells: float = 0.75,
        uncertainty_scale: float = 2.0,
        initial_residual_strength: float = 0.10,
        maximum_residual_strength: float = 2.0,
        field_temperature: float = 0.12,
        target_sigma_cells: float = 0.65,
        initial_coverage_strength: float = 8.0,
        maximum_continuation_bias: float = 2.0,
        initial_set_coordinate_strength: float = 0.50,
        maximum_set_coordinate_strength: float = 4.0,
        maximum_presence_bias: float = 2.0,
        initial_gate_scale: float = 0.05,
        selection_temperature: float = 0.50,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            vision_hidden_size=vision_hidden_size,
            relay_layer=relay_layer,
            token_ids=token_ids,
            evidence_dim=evidence_dim,
            subpatch_dim=subpatch_dim,
            subpatch_grid=subpatch_grid,
            scale_multipliers=scale_multipliers,
            minimum_sigma_cells=minimum_sigma_cells,
            uncertainty_scale=uncertainty_scale,
            initial_residual_strength=initial_residual_strength,
            maximum_residual_strength=maximum_residual_strength,
            field_temperature=field_temperature,
            target_sigma_cells=target_sigma_cells,
            initial_coverage_strength=initial_coverage_strength,
            maximum_continuation_bias=maximum_continuation_bias,
            initial_set_coordinate_strength=initial_set_coordinate_strength,
            maximum_set_coordinate_strength=maximum_set_coordinate_strength,
            maximum_presence_bias=maximum_presence_bias,
            initial_gate_scale=initial_gate_scale,
        )
        if selection_temperature <= 0:
            raise ValueError("selection_temperature must be positive")
        self.selection_temperature = float(selection_temperature)
        self._selected_point_indices: tuple[int, ...] = ()

    def _reset_sample_state(self, *, preserve_pixel_field: bool = False) -> None:
        super()._reset_sample_state(preserve_pixel_field=preserve_pixel_field)
        self._selected_point_indices = ()

    def _selection_weights(self, field_logits: torch.Tensor) -> torch.Tensor:
        if self._training_context:
            weights = F.gumbel_softmax(
                field_logits.float(),
                tau=self.selection_temperature,
                hard=True,
                dim=-1,
            )
        else:
            index = int(field_logits.argmax())
            weights = F.one_hot(
                field_logits.new_tensor(index, dtype=torch.long),
                num_classes=int(field_logits.numel()),
            ).to(field_logits.dtype)
        self._selected_point_indices = (
            *self._selected_point_indices,
            int(weights.detach().argmax()),
        )
        return weights

    def _route_target_boxes(
        self, transition: ResidualSetTransition
    ) -> tuple[tuple[int, int, int, int], ...]:
        # Geometry supervision is serialized one block at a time. Aligning the
        # latent witness to that same block prevents a route/geometry mismatch.
        return () if transition.target_box is None else (transition.target_box,)

    def _coverage_write_support(
        self,
        transition: ResidualSetTransition,
        selection_weights: torch.Tensor,
    ) -> torch.Tensor | None:
        # Write what the model actually selected, not a teacher-forced box.
        # The straight-through one-hot state makes training and inference agree.
        return selection_weights.detach().float()

    def _write_previous_geometry(self, input_ids: torch.Tensor, row_start: int) -> None:
        # The latent point is the sole identity state. Geometry still remains
        # in Qwen's token history, but it does not deterministically steer the
        # next address or create an external crop policy.
        return None

    def prepare_set_signals(self, hidden_states: torch.Tensor) -> None:
        self._selected_point_indices = ()
        super().prepare_set_signals(hidden_states)
        if self._selected_point_indices:
            self.last_diagnostics.update(
                {
                    "latent_point_selections": float(
                        len(self._selected_point_indices)
                    ),
                    "latent_point_unique_selections": float(
                        len(set(self._selected_point_indices))
                    ),
                }
            )


def load_latent_point_process_reentry_adapter(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config.get("arm")) != "latent_point_process_visual_reentry":
        raise ValueError(f"Not a latent-point-process checkpoint: {config.get('arm')!r}")
    configure_lora_only(model, int(config["lora_rank"]))
    controller = LatentPointProcessVisualReentryPBD(
        hidden_size=int(model.config.text_config.hidden_size),
        vision_hidden_size=int(model.config.vision_config.hidden_size),
        relay_layer=int(config.get("relay_layer", 18)),
        token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
        evidence_dim=int(config.get("evidence_dim", 128)),
        subpatch_dim=int(config.get("subpatch_dim", 32)),
        subpatch_grid=int(config.get("subpatch_grid", 4)),
        scale_multipliers=tuple(config.get("scale_multipliers", (0.75, 1.5, 3.0))),
        minimum_sigma_cells=float(config.get("minimum_sigma_cells", 0.75)),
        uncertainty_scale=float(config.get("uncertainty_scale", 2.0)),
        initial_residual_strength=float(config.get("initial_residual_strength", 0.10)),
        maximum_residual_strength=float(config.get("maximum_residual_strength", 2.0)),
        field_temperature=float(config.get("field_temperature", 0.12)),
        target_sigma_cells=float(config.get("target_sigma_cells", 0.65)),
        initial_coverage_strength=float(config.get("initial_coverage_strength", 8.0)),
        maximum_continuation_bias=float(config.get("maximum_continuation_bias", 2.0)),
        initial_set_coordinate_strength=float(
            config.get("initial_set_coordinate_strength", 0.50)
        ),
        maximum_set_coordinate_strength=float(
            config.get("maximum_set_coordinate_strength", 4.0)
        ),
        maximum_presence_bias=float(config.get("maximum_presence_bias", 2.0)),
        initial_gate_scale=float(config.get("initial_gate_scale", 0.05)),
        selection_temperature=float(config.get("selection_temperature", 0.50)),
    ).to(next(model.parameters()).device)
    controller.install(model)
    restore_trainable_state(model, controller, checkpoint["trainable_state"])
    model.eval()
    return {"checkpoint": checkpoint, "controller": controller}
