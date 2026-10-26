#!/usr/bin/env python3
"""Train one compact LoRA arm for the PointRelay-PBD detection pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
EAGLE = WORKSPACE / "Eagle/Embodied"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
if str(EAGLE) not in sys.path:
    sys.path.insert(0, str(EAGLE))

from zero_shot_analysis.pointrelay_pbd import (  # noqa: E402
    DecoderScalarPointRelay,
    PointRoutePBDController,
    PointRelayController,
    SetValuedPointRelayPBD,
    base_causal_lm,
    configure_lora_only,
    geometry_context_from_training,
    restore_trainable_state,
    trainable_state,
)
from zero_shot_analysis.adaptive_memory_pbd import AdaptivePointMemoryPBD  # noqa: E402
from zero_shot_analysis.dense_address_field_pbd import DenseAddressFieldPBD  # noqa: E402
from zero_shot_analysis.marked_residual_address_pbd import (  # noqa: E402
    MarkedResidualAddressPBD,
)


SEED = 20260825
IGNORE_INDEX = -100


def atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")


def chunked_ce(hidden_states: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    shifted_hidden = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
    shifted_labels = labels[:, 1:].reshape(-1)
    valid = shifted_labels.ne(IGNORE_INDEX)
    shifted_hidden = shifted_hidden[valid]
    shifted_labels = shifted_labels[valid]
    if shifted_labels.numel() == 0:
        raise RuntimeError("Pilot sample has no supervised tokens")
    total = hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, int(shifted_labels.numel()), int(chunk)):
        stop = min(int(shifted_labels.numel()), start + int(chunk))
        logits = F.linear(shifted_hidden[start:stop], weight).float()
        total = total + F.cross_entropy(logits, shifted_labels[start:stop], reduction="sum")
    return total / shifted_labels.numel()


def selected_token_ce(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    token_ids: tuple[int, ...],
) -> torch.Tensor:
    """Mean next-token CE restricted to an explicitly named token role."""

    if not token_ids:
        return hidden_states.new_zeros((), dtype=torch.float32)
    shifted_hidden = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
    shifted_labels = labels[:, 1:].reshape(-1)
    selected = shifted_labels.ne(IGNORE_INDEX)
    role_ids = torch.as_tensor(token_ids, device=shifted_labels.device, dtype=shifted_labels.dtype)
    selected &= torch.isin(shifted_labels, role_ids)
    if not bool(selected.any()):
        return hidden_states.new_zeros((), dtype=torch.float32)
    logits = F.linear(shifted_hidden[selected], weight).float()
    return F.cross_entropy(logits, shifted_labels[selected], reduction="mean")


def geometry_history_contrastive_loss(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    token_ids: Mapping[str, int],
    *,
    margin: float,
) -> torch.Tensor:
    """Rank each next GT box above every earlier same-ref geometry block.

    LocateAnything trains each six-token PBD block with a teacher-forced prefix.
    This loss makes prior boxes explicit hard negatives without changing the
    released grammar or inference path. It only compares the four coordinate
    lanes, so semantic labels and structural tokens remain under native CE.
    """

    if input_ids.ndim != 2 or labels.ndim != 2 or input_ids.shape != labels.shape:
        raise ValueError("History contrast expects matching [1, sequence] ids and labels")
    if int(input_ids.shape[0]) != 1:
        raise ValueError("History contrast currently supports batch size one")

    ids = input_ids[0]
    target = labels[0]
    mask_id = int(token_ids["default_mask_token_id"])
    ref_start = int(token_ids["ref_start_token_id"])
    box_start = int(token_ids["box_start_token_id"])
    box_end = int(token_ids["box_end_token_id"])
    coord_start = int(token_ids["coord_start_token_id"])
    coord_end = int(token_ids["coord_end_token_id"])

    # MTP blocks appear in the same causal order as the assistant response.
    # A <ref> block starts a new class group; numeric <box> blocks inherit it.
    group = -1
    boxes: list[tuple[int, int, torch.Tensor]] = []
    for start in range(max(0, int(ids.numel()) - 6)):
        if start + 6 >= int(target.numel()):
            break
        if not bool(ids[start + 1 : start + 6].eq(mask_id).all()):
            continue
        block = target[start + 1 : start + 7]
        if int(block.numel()) != 6:
            continue
        values = [int(value) for value in block.tolist()]
        if values[0] == ref_start:
            group += 1
            continue
        if (
            values[0] == box_start
            and values[5] == box_end
            and all(coord_start <= value <= coord_end for value in values[1:5])
        ):
            boxes.append(
                (
                    max(0, group),
                    start,
                    torch.as_tensor(
                        [value - coord_start for value in values[1:5]],
                        device=hidden_states.device,
                        dtype=torch.long,
                    ),
                )
            )

    losses: list[torch.Tensor] = []
    prior_by_group: dict[int, list[torch.Tensor]] = {}
    coordinate_weight = weight[coord_start : coord_end + 1]
    for group_id, start, current in boxes:
        previous = prior_by_group.setdefault(group_id, [])
        # Exact duplicate annotations do not define a learnable ordering
        # preference: the positive and negative token sequence would be
        # identical. Keep them out of the contrastive candidate set while the
        # native token CE still supervises the record unchanged.
        distinct_previous = [item for item in previous if not torch.equal(item, current)]
        if distinct_previous:
            # labels[start+2:start+6] are predicted by hidden[start+1:start+5].
            lane_hidden = hidden_states[0, start + 1 : start + 5]
            if int(lane_hidden.shape[0]) != 4:
                raise RuntimeError("Incomplete coordinate lanes in history contrast")
            log_probs = F.linear(lane_hidden, coordinate_weight).float().log_softmax(dim=-1)
            lane = torch.arange(4, device=hidden_states.device)
            target_score = log_probs[lane, current].mean()
            prior = torch.stack(distinct_previous, dim=0)
            expanded = log_probs.unsqueeze(0).expand(int(prior.shape[0]), -1, -1)
            prior_scores = expanded.gather(2, prior.unsqueeze(-1)).squeeze(-1).mean(dim=-1)
            losses.append(F.softplus(float(margin) + prior_scores.max() - target_score))
        previous.append(current)

    if not losses:
        return hidden_states.new_zeros((), dtype=torch.float32)
    return torch.stack(losses).mean()


def original_prefix_length(position_ids: torch.Tensor) -> int:
    """Return the pre-MTP sequence length from LocateAnything position IDs."""

    if position_ids.ndim != 2 or int(position_ids.shape[0]) != 1:
        raise ValueError("Expected position_ids with shape [1, sequence]")
    positions = position_ids[0]
    if int(positions.numel()) == 0:
        return 0
    resets = (positions[1:] <= positions[:-1]).nonzero(as_tuple=False)
    return int(resets[0, 0].item() + 1) if int(resets.numel()) else int(positions.numel())


def predicted_geometry_rollin(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    position_ids: torch.Tensor,
    weight: torch.Tensor,
    token_ids: Mapping[str, int],
    *,
    probability: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Replace teacher geometry history with detached model predictions.

    Only complete numeric HBBs in the original assistant sequence are eligible.
    Their four coordinate inputs are replaced by coordinate-vocabulary argmaxes
    from a no-gradient teacher pass. Labels and all native MTP targets stay
    unchanged, so the trainable pass learns the correct continuation from its own
    imperfect geometry state instead of an exclusively ground-truth prefix.
    """

    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("Geometry roll-in probability must be in [0, 1]")
    if float(probability) == 0.0:
        return input_ids, {"rollin_boxes": 0.0, "eligible_rollin_boxes": 0.0, "rollin_token_rate": 0.0}
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("Geometry roll-in expects matching [1, sequence] ids and labels")
    if int(input_ids.shape[0]) != 1:
        raise ValueError("Geometry roll-in currently supports batch size one")

    prefix_length = original_prefix_length(position_ids)
    ids = input_ids[0]
    target = labels[0]
    box_start = int(token_ids["box_start_token_id"])
    box_end = int(token_ids["box_end_token_id"])
    coord_start = int(token_ids["coord_start_token_id"])
    coord_end = int(token_ids["coord_end_token_id"])
    coordinate_weight = weight[coord_start : coord_end + 1]

    original_boxes: list[tuple[int, tuple[int, ...]]] = []
    upper = max(0, prefix_length - 5)
    for start in range(upper):
        block = target[start : start + 6]
        if int(block.numel()) != 6:
            continue
        values = [int(value) for value in block.tolist()]
        if (
            values[0] == box_start
            and values[5] == box_end
            and all(coord_start <= value <= coord_end for value in values[1:5])
            and bool(ids[start : start + 6].eq(block).all())
        ):
            original_boxes.append((start, tuple(values)))

    # Hybrid inference obtains each six-token geometry block from the parallel
    # MTP window. Use those exact lane states, then map each occurrence back to
    # its matching box in the original assistant prefix.
    mask_id = int(token_ids["default_mask_token_id"])
    mtp_by_target: dict[tuple[int, ...], list[int]] = {}
    for start in range(prefix_length, max(prefix_length, int(ids.numel()) - 6)):
        if start + 6 >= int(target.numel()):
            break
        if not bool(ids[start + 1 : start + 6].eq(mask_id).all()):
            continue
        block = target[start + 1 : start + 7]
        if int(block.numel()) != 6:
            continue
        values = tuple(int(value) for value in block.tolist())
        if (
            values[0] == box_start
            and values[5] == box_end
            and all(coord_start <= value <= coord_end for value in values[1:5])
        ):
            mtp_by_target.setdefault(values, []).append(start)

    eligible: list[tuple[int, int]] = []
    consumed: dict[tuple[int, ...], int] = {}
    for original_start, values in original_boxes:
        occurrence = consumed.get(values, 0)
        candidates = mtp_by_target.get(values, [])
        if occurrence < len(candidates):
            eligible.append((original_start, candidates[occurrence]))
            consumed[values] = occurrence + 1

    if not eligible:
        return input_ids, {"rollin_boxes": 0.0, "eligible_rollin_boxes": 0.0, "rollin_token_rate": 0.0}

    chosen = torch.rand(len(eligible), device=hidden_states.device) < float(probability)
    if not bool(chosen.any()):
        return input_ids, {
            "rollin_boxes": 0.0,
            "eligible_rollin_boxes": float(len(eligible)),
            "rollin_token_rate": 0.0,
        }

    rolled = input_ids.clone()
    changed_tokens = 0
    selected_boxes = 0
    for select, (original_start, mtp_start) in zip(chosen.tolist(), eligible):
        if not select:
            continue
        # MTP labels[start+2:start+6] are predicted by hidden[start+1:start+5].
        lane_hidden = hidden_states[0, mtp_start + 1 : mtp_start + 5]
        if int(lane_hidden.shape[0]) != 4:
            raise RuntimeError("Incomplete coordinate lanes in geometry roll-in")
        predicted = F.linear(lane_hidden, coordinate_weight).float().argmax(dim=-1) + coord_start
        old = rolled[0, original_start + 1 : original_start + 5]
        changed_tokens += int(predicted.ne(old).sum().item())
        rolled[0, original_start + 1 : original_start + 5] = predicted.detach()
        selected_boxes += 1

    return rolled, {
        "rollin_boxes": float(selected_boxes),
        "eligible_rollin_boxes": float(len(eligible)),
        "rollin_token_rate": float(changed_tokens / max(1, 4 * selected_boxes)),
    }


class PilotModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        controller: Any | None,
        point_weight: float,
        *,
        boundary_token_ids: tuple[int, ...] = (),
        boundary_loss_weight: float = 0.0,
        eos_token_id: int | None = None,
        eos_loss_weight: float = 0.0,
        none_token_id: int | None = None,
        none_loss_weight: float = 0.0,
        history_loss_weight: float = 0.0,
        history_margin: float = 0.5,
        geometry_rollin_probability: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.controller = controller
        self.point_weight = float(point_weight)
        self.boundary_token_ids = tuple(int(value) for value in boundary_token_ids)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.eos_token_ids = () if eos_token_id is None else (int(eos_token_id),)
        self.eos_loss_weight = float(eos_loss_weight)
        self.none_token_ids = () if none_token_id is None else (int(none_token_id),)
        self.none_loss_weight = float(none_loss_weight)
        self.history_loss_weight = float(history_loss_weight)
        self.history_margin = float(history_margin)
        self.geometry_rollin_probability = float(geometry_rollin_probability)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen visual modules must remain deterministic. Calling train() on
        # the wrapper would otherwise reactivate their training-time behavior.
        self.model.vision_model.eval()
        self.model.mlp1.eval()
        return self

    def forward(self, sample: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        device = next(self.model.parameters()).device
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)
        position_ids = sample["position_ids"].unsqueeze(0).to(device)
        pixel_values = sample["pixel_values"].to(device=device, dtype=torch.bfloat16)
        image_grid_hws = torch.as_tensor(sample["image_grid_hws"], device=device, dtype=torch.int32)
        if isinstance(self.controller, DenseAddressFieldPBD):
            self.controller.prepare_pixel_field(pixel_values, image_grid_hws)
        with torch.no_grad():
            visual = self.model.extract_feature(pixel_values, image_grid_hws)
            projected = self.model.mlp1(torch.cat(visual, dim=0))
        selected = input_ids.eq(int(self.model.image_token_index))
        if int(selected.sum().item()) != int(projected.shape[0]):
            raise RuntimeError(
                f"Visual placeholder mismatch: ids={int(selected.sum())}, projected={int(projected.shape[0])}"
            )
        causal = base_causal_lm(self.model.language_model)
        if self.controller is not None:
            context = geometry_context_from_training(
                input_ids,
                labels,
                token_ids={**self.model.token_ids, "image_token_index": int(self.model.image_token_index)},
            )
            if isinstance(self.controller, DenseAddressFieldPBD):
                self.controller.set_training_context(
                    context,
                    image_grid_hws,
                    input_ids=input_ids,
                    labels=labels,
                )
            elif isinstance(
                self.controller,
                (PointRoutePBDController, SetValuedPointRelayPBD, AdaptivePointMemoryPBD),
            ):
                self.controller.set_training_context(context, image_grid_hws)
            else:
                self.controller.set_training_context(context)
            if isinstance(self.controller, AdaptivePointMemoryPBD):
                self.controller.prepare_training_memory(
                    self.model,
                    pixel_values,
                    image_grid_hws,
                    context,
                )
        def decoder_forward(active_ids: torch.Tensor):
            embeddings = self.model.language_model.get_input_embeddings()(active_ids)
            active_selected = active_ids.eq(int(self.model.image_token_index))
            if int(active_selected.sum().item()) != int(projected.shape[0]):
                raise RuntimeError(
                    "Visual placeholders changed during geometry-state roll-in: "
                    f"ids={int(active_selected.sum())}, projected={int(projected.shape[0])}"
                )
            flat = embeddings.reshape(-1, embeddings.shape[-1]).clone()
            flat[active_selected.reshape(-1)] = projected
            embeddings = flat.view_as(embeddings)
            return causal.model(
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

        rollin_diagnostics = {
            "rollin_boxes": 0.0,
            "eligible_rollin_boxes": 0.0,
            "rollin_token_rate": 0.0,
        }
        active_input_ids = input_ids
        if self.geometry_rollin_probability > 0:
            if self.controller is not None:
                raise RuntimeError("Geometry-state roll-in is only defined for the standard LoRA arm")
            # PEFT's LoRA layers must execute this teacher pass with autograd
            # enabled. Running the adapter under torch.no_grad() causes the
            # following student pass to lose all adapter-parameter gradients.
            # Detach the teacher outputs explicitly and release their graph
            # before the trainable pass, so predictions are current-model
            # roll-ins without backpropagating through the teacher.
            teacher_outputs = decoder_forward(input_ids)
            active_input_ids, rollin_diagnostics = predicted_geometry_rollin(
                teacher_outputs.last_hidden_state.detach(),
                input_ids,
                labels,
                position_ids,
                causal.lm_head.weight.detach(),
                self.model.token_ids,
                probability=self.geometry_rollin_probability,
            )
            del teacher_outputs
        outputs = decoder_forward(active_input_ids)
        native = chunked_ce(outputs.last_hidden_state, labels, causal.lm_head.weight)
        boundary = selected_token_ce(
            outputs.last_hidden_state, labels, causal.lm_head.weight, self.boundary_token_ids
        )
        eos = selected_token_ce(
            outputs.last_hidden_state, labels, causal.lm_head.weight, self.eos_token_ids
        )
        none = selected_token_ce(
            outputs.last_hidden_state, labels, causal.lm_head.weight, self.none_token_ids
        )
        history = (
            geometry_history_contrastive_loss(
                outputs.last_hidden_state,
                active_input_ids,
                labels,
                causal.lm_head.weight,
                self.model.token_ids,
                margin=self.history_margin,
            )
            if self.history_loss_weight > 0
            else outputs.last_hidden_state.new_zeros((), dtype=torch.float32)
        )
        point = None if self.controller is None else self.controller.aux_loss
        if self.controller is not None:
            # Preserve the loss tensor for this backward pass, but do not let
            # sample-specific geometry leak into the next record.
            if isinstance(self.controller, (AdaptivePointMemoryPBD, DenseAddressFieldPBD)):
                self.controller.finish_forward()
            else:
                self.controller._context = None
                self.controller._training_context = False
        total = (
            native
            + self.boundary_loss_weight * boundary
            + self.eos_loss_weight * eos
            + self.none_loss_weight * none
            + self.history_loss_weight * history
        )
        if point is not None:
            total = total + self.point_weight * point
        values = {
            "loss": float(total.detach().float()),
            "native_loss": float(native.detach().float()),
            "point_loss": 0.0 if point is None else float(point.detach().float()),
            "boundary_loss": float(boundary.detach().float()),
            "eos_loss": float(eos.detach().float()),
            "none_loss": float(none.detach().float()),
            "history_loss": float(history.detach().float()),
            **rollin_diagnostics,
        }
        if hasattr(self.controller, "last_diagnostics"):
            values.update(self.controller.last_diagnostics)
        return total, values


def make_dataset(path: Path, processor: Any):
    from eaglevl.train.locany_finetune_magi_stream import LazySupervisedDatasetMTP

    return LazySupervisedDatasetMTP(
        "pointrelay_pilot",
        {"annotation": str(path), "root": str(WORKSPACE), "repeat_time": 1.0, "data_augment": False},
        processor,
        block_size=6,
    )


def make_loader(dataset: Any, indices: list[int], workers: int) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": Subset(dataset, indices),
        "batch_size": None,
        "shuffle": False,
        "num_workers": int(workers),
        "pin_memory": True,
    }
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def scheduler_factor(step: int, total_steps: int, warmup: int) -> float:
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = min(1.0, float(step - warmup) / max(1, total_steps - warmup))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(wrapper: PilotModel, dataset: Any, indices: list[int], workers: int) -> dict[str, float]:
    saved_rollin_probability = wrapper.geometry_rollin_probability
    wrapper.geometry_rollin_probability = 0.0
    wrapper.eval()
    # LocateAnything's custom Qwen builds the MTP block mask only when the
    # decoder model's own training flag is true. Keep child modules (including
    # LoRA dropout) in eval mode, while selecting that deterministic mask path.
    causal_model = base_causal_lm(wrapper.model.language_model).model
    causal_model.training = True
    losses: list[dict[str, float]] = []
    for sample in make_loader(dataset, indices, workers):
        _loss, values = wrapper(sample)
        losses.append(values)
    causal_model.training = False
    wrapper.train()
    wrapper.geometry_rollin_probability = saved_rollin_probability
    if not losses:
        raise RuntimeError("Validation subset is empty")
    return {
        key: sum(item[key] for item in losses) / len(losses)
        for key in (
            "loss", "native_loss", "point_loss", "boundary_loss", "eos_loss", "none_loss",
            "history_loss",
        )
    }


def checkpoint_payload(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    controller: Any | None,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    exposure: int,
    validation: Mapping[str, float] | None,
    audit: Mapping[str, int],
) -> dict[str, Any]:
    state_controller = controller if isinstance(controller, nn.Module) else None
    return {
        "schema_version": "pointrelay-pbd-lora-pilot-checkpoint-v1",
        "config": {
            "arm": args.arm,
            "base_model": str(args.model),
            "lora_rank": int(args.lora_rank),
            "relay_layer": int(args.relay_layer),
            "temperature": float(args.temperature),
            "point_dim": int(args.point_dim),
            "target_sigma": float(args.target_sigma),
            "initial_gate_logit": float(args.initial_gate_logit),
            "point_loss_weight": float(args.point_loss_weight),
            "boundary_loss_weight": float(args.boundary_loss_weight),
            "eos_loss_weight": float(args.eos_loss_weight),
            "none_loss_weight": float(args.none_loss_weight),
            "history_loss_weight": float(args.history_loss_weight),
            "history_margin": float(args.history_margin),
            "geometry_rollin_max_probability": float(args.geometry_rollin_max_probability),
            "geometry_rollin_start_step": int(args.geometry_rollin_start_step),
            "geometry_rollin_ramp_steps": int(args.geometry_rollin_ramp_steps),
            "best_validation_metric": str(args.best_validation_metric),
            "controller_learning_rate": float(args.controller_learning_rate),
            "joint_controller_optimizer": bool(args.joint_controller_optimizer),
            "local_radius_fraction": float(args.local_radius_fraction),
            "memory_dim": int(args.memory_dim),
            "field_dim": int(args.field_dim),
            "field_target_sigma": float(args.field_target_sigma),
            "subpatch_dim": int(args.subpatch_dim),
            "subpatch_grid": int(args.subpatch_grid),
            "local_patch_grid": int(args.local_patch_grid),
            "max_memory_addresses": int(args.max_memory_addresses),
            "local_encode_batch": int(args.local_encode_batch),
            "initial_coverage_strength": float(args.initial_coverage_strength),
            "residual_floor": float(args.residual_floor),
            "footprint_halo_cells": float(args.footprint_halo_cells),
            "initial_structure_scale": float(args.initial_structure_scale),
            "absence_loss_weight": float(args.absence_loss_weight),
            "image_token_limit": int(args.image_token_limit),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "gradient_accumulation": int(args.gradient_accumulation),
            "max_steps": int(args.max_steps),
            "seed": int(args.seed),
            "lora_audit": dict(audit),
        },
        "step": int(step),
        "exposure": int(exposure),
        "validation": None if validation is None else dict(validation),
        "trainable_state": trainable_state(model, state_controller),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
    }


def update_symlink(link: Path, target: Path) -> None:
    temporary = link.with_name(f".{link.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(target.name)
    os.replace(temporary, link)


def train(args: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoModel, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for name in (
        "boundary_loss_weight", "eos_loss_weight", "none_loss_weight", "history_loss_weight"
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be non-negative")
    if not 0.0 <= float(args.geometry_rollin_max_probability) <= 1.0:
        raise ValueError("geometry_rollin_max_probability must be in [0, 1]")
    if int(args.geometry_rollin_start_step) < 0 or int(args.geometry_rollin_ramp_steps) <= 0:
        raise ValueError("Geometry roll-in schedule steps must be non-negative/positive")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    processor.image_processor.in_token_limit = int(args.image_token_limit)
    processor.tokenizer.model_max_length = int(args.max_sequence)
    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    config._attn_implementation = "sdpa"
    config._attn_implementation_autoset = False
    config.text_config._attn_implementation = "sdpa"
    config.text_config._attn_implementation_autoset = False
    # MoonViT is frozen and runs without gradients in this LoRA pilot. Keep its
    # released efficient backend while Qwen uses the required training SDPA path.
    config.vision_config._attn_implementation = "flash_attention_2"
    config.vision_config._attn_implementation_autoset = False
    model = AutoModel.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    audit = configure_lora_only(model, args.lora_rank)
    if args.gradient_checkpointing:
        checkpointed_causal = base_causal_lm(model.language_model)
        checkpointed_causal.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        checkpointed_causal.config.use_cache = False
    controller = None
    if args.arm in {"point_aux", "pointrelay"}:
        controller = PointRelayController(
            hidden_size=int(model.config.text_config.hidden_size),
            relay_layer=args.relay_layer,
            token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
            mode=args.arm,
            temperature=args.temperature,
        ).to(device=device, dtype=torch.bfloat16)
        controller.install(model)
        controller.requires_grad_(args.arm == "pointrelay")
    elif args.arm in {"decoder_point_aux", "decoder_pointrelay"}:
        controller = DecoderScalarPointRelay(
            relay_layer=args.relay_layer,
            token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
            mode="point_aux" if args.arm == "decoder_point_aux" else "pointrelay",
            temperature=args.temperature,
        )
        controller.install(model)
    elif args.arm in {"pointroute_aux", "pointroute"}:
        controller = PointRoutePBDController(
            hidden_size=int(model.config.text_config.hidden_size),
            relay_layer=args.relay_layer,
            token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
            mode=args.arm,
            point_dim=args.point_dim,
            temperature=args.temperature,
            target_sigma=args.target_sigma,
            initial_gate_logit=args.initial_gate_logit,
        ).to(device=device)
        controller.install(model)
        if args.arm == "pointroute_aux" and not args.joint_controller_optimizer:
            controller.glimpse_down.requires_grad_(False)
            controller.glimpse_up.requires_grad_(False)
            controller.lane_gate_logits.requires_grad_(False)
    elif args.arm in {"setpoint_aux", "setpoint_relay", "causal_pointrelay"}:
        controller = SetValuedPointRelayPBD(
            relay_layer=args.relay_layer,
            token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
            mode=args.arm,
            temperature=args.temperature,
            local_radius_fraction=args.local_radius_fraction,
        )
        controller.install(model)
    elif args.arm == "adaptive_memory":
        controller = AdaptivePointMemoryPBD(
            hidden_size=int(model.config.text_config.hidden_size),
            relay_layer=args.relay_layer,
            token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
            memory_dim=args.memory_dim,
            local_patch_grid=args.local_patch_grid,
            max_memory_addresses=args.max_memory_addresses,
            local_encode_batch=args.local_encode_batch,
            temperature=args.temperature,
            initial_gate_logit=args.initial_gate_logit,
            initial_coverage_strength=args.initial_coverage_strength,
        ).to(device=device)
        controller.install(model)
    elif args.arm == "dense_address_field":
        controller = DenseAddressFieldPBD(
            hidden_size=int(model.config.text_config.hidden_size),
            vision_size=int(model.config.vision_config.hidden_size),
            relay_layer=args.relay_layer,
            token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
            field_dim=args.field_dim,
            subpatch_dim=args.subpatch_dim,
            subpatch_grid=args.subpatch_grid,
            temperature=args.temperature,
            target_sigma=args.field_target_sigma,
            initial_gate_logit=args.initial_gate_logit,
            initial_coverage_strength=args.initial_coverage_strength,
        ).to(device=device)
        controller.install(model)
    elif args.arm == "marked_residual_address":
        controller = MarkedResidualAddressPBD(
            hidden_size=int(model.config.text_config.hidden_size),
            vision_size=int(model.config.vision_config.hidden_size),
            relay_layer=args.relay_layer,
            token_ids={**model.token_ids, "image_token_index": int(model.image_token_index)},
            field_dim=args.field_dim,
            subpatch_dim=args.subpatch_dim,
            subpatch_grid=args.subpatch_grid,
            temperature=args.temperature,
            target_sigma=args.field_target_sigma,
            initial_gate_logit=args.initial_gate_logit,
            initial_coverage_strength=args.initial_coverage_strength,
            residual_floor=args.residual_floor,
            footprint_halo_cells=args.footprint_halo_cells,
            initial_structure_scale=args.initial_structure_scale,
            absence_loss_weight=args.absence_loss_weight,
        ).to(device=device)
        controller.install(model)
    model.vision_model.eval()
    model.mlp1.eval()
    tokenizer = processor.tokenizer

    def one_token_id(text: str) -> int:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"Expected one token for {text!r}, got {encoded}")
        return int(encoded[0])

    # Structural boundaries and sequence termination are deliberately separate
    # roles. This avoids accidentally applying both boundary and EOS weights to
    # the same token while covering the complete released LA tag grammar.
    boundary_tokens = tuple(
        dict.fromkeys(one_token_id(token) for token in ("<ref>", "</ref>", "<box>", "</box>"))
    )
    eos_token_id = one_token_id("<|im_end|>")
    none_token_id = one_token_id("None")
    wrapper = PilotModel(
        model,
        controller,
        args.point_loss_weight,
        boundary_token_ids=boundary_tokens if args.boundary_loss_weight > 0 else (),
        boundary_loss_weight=args.boundary_loss_weight,
        eos_token_id=(eos_token_id if args.eos_loss_weight > 0 else None),
        eos_loss_weight=args.eos_loss_weight,
        none_token_id=(none_token_id if args.none_loss_weight > 0 else None),
        none_loss_weight=args.none_loss_weight,
        history_loss_weight=args.history_loss_weight,
        history_margin=args.history_margin,
        geometry_rollin_probability=0.0,
    ).to(device)
    train_dataset = make_dataset(args.train_data, processor)
    validation_dataset = make_dataset(args.validation_data, processor)
    if len(train_dataset) != args.expected_train_records or len(validation_dataset) != args.expected_validation_records:
        raise RuntimeError(f"Frozen pilot sizes changed: train={len(train_dataset)}, val={len(validation_dataset)}")
    required_exposures = int(args.max_steps) * int(args.gradient_accumulation)
    if not args.smoke and required_exposures != len(train_dataset):
        raise RuntimeError(
            f"One-pass contract requires steps*accum={len(train_dataset)}, got {required_exposures}"
        )
    run_exposures = required_exposures if args.smoke else len(train_dataset)
    validation_indices = list(range(min(int(args.validation_records), len(validation_dataset))))
    lora_trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    controller_trainable = (
        [parameter for parameter in controller.parameters() if parameter.requires_grad]
        if isinstance(
            controller,
            (PointRoutePBDController, AdaptivePointMemoryPBD, DenseAddressFieldPBD),
        )
        else []
    )
    trainable = [*lora_trainable, *controller_trainable]
    if args.joint_controller_optimizer:
        parameter_groups: list[dict[str, Any]] = [
            {"params": trainable, "lr": args.learning_rate},
        ]
    else:
        parameter_groups = [{"params": lora_trainable, "lr": args.learning_rate}]
        if controller_trainable:
            parameter_groups.append(
                {"params": controller_trainable, "lr": args.controller_learning_rate}
            )
    optimizer = torch.optim.AdamW(
        parameter_groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: scheduler_factor(step, args.max_steps, args.warmup_steps)
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training_curve.jsonl"
    start_step = 0
    start_exposure = 0
    best_validation = float("inf")
    resume = output / "last.pt"
    if resume.is_file() or resume.is_symlink():
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        restore_trainable_state(
            model,
            controller if isinstance(controller, nn.Module) else None,
            payload["trainable_state"],
        )
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        start_step = int(payload["step"])
        start_exposure = int(payload["exposure"])
        torch.set_rng_state(payload["rng_cpu"])
        torch.cuda.set_rng_state(payload["rng_cuda"])
        if (output / "best.pt").exists():
            best_payload = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
            best_validation = float(best_payload["validation"][args.best_validation_metric])
    if start_step >= args.max_steps:
        print(f"{args.arm}: already complete at step {start_step}")
        return

    wrapper.train()
    optimizer.zero_grad(set_to_none=True)
    indices = list(range(start_exposure, run_exposures))
    loader = make_loader(train_dataset, indices, args.workers)
    started = time.time()
    running: list[dict[str, float]] = []
    exposure = start_exposure
    step = start_step
    for sample in loader:
        exposure += 1
        if args.geometry_rollin_max_probability > 0 and step >= args.geometry_rollin_start_step:
            ramp = min(
                1.0,
                float(step - args.geometry_rollin_start_step + 1)
                / max(1, int(args.geometry_rollin_ramp_steps)),
            )
            wrapper.geometry_rollin_probability = float(args.geometry_rollin_max_probability) * ramp
        else:
            wrapper.geometry_rollin_probability = 0.0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, values = wrapper(sample)
            scaled = loss / args.gradient_accumulation
        scaled.backward()
        optimized_grad_tensors = sum(parameter.grad is not None for parameter in trainable)
        optimized_finite_grad_tensors = sum(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        )
        if optimized_grad_tensors == 0:
            raise RuntimeError(
                "Backward produced no gradients for any optimized tensor; "
                f"step={step}, exposure={exposure}, "
                f"geometry_rollin_probability={wrapper.geometry_rollin_probability}"
            )
        if optimized_finite_grad_tensors != optimized_grad_tensors:
            raise RuntimeError(
                "Backward produced non-finite optimized gradients; "
                f"finite={optimized_finite_grad_tensors}, total={optimized_grad_tensors}"
            )
        values["optimized_grad_tensors"] = float(optimized_grad_tensors)
        running.append(values)
        if exposure % args.gradient_accumulation != 0:
            continue
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        record = {
            "step": step,
            "exposure": exposure,
            "loss": sum(item["loss"] for item in running) / len(running),
            "native_loss": sum(item["native_loss"] for item in running) / len(running),
            "point_loss": sum(item["point_loss"] for item in running) / len(running),
            "boundary_loss": sum(item["boundary_loss"] for item in running) / len(running),
            "eos_loss": sum(item["eos_loss"] for item in running) / len(running),
            "none_loss": sum(item["none_loss"] for item in running) / len(running),
            "history_loss": sum(item["history_loss"] for item in running) / len(running),
            "geometry_rollin_probability": wrapper.geometry_rollin_probability,
            "rollin_boxes": sum(item["rollin_boxes"] for item in running) / len(running),
            "eligible_rollin_boxes": sum(item["eligible_rollin_boxes"] for item in running) / len(running),
            "rollin_token_rate": sum(item["rollin_token_rate"] for item in running) / len(running),
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": time.time() - started,
            "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }
        if len(scheduler.get_last_lr()) > 1:
            record["controller_learning_rate"] = scheduler.get_last_lr()[1]
        for key in (
            "point_mae_0_1000",
            "point_certainty",
            "route_gate",
            "point_inside_box_rate",
            "point_interval_error_0_1000",
            "point_entropy_fraction",
            "local_image_tokens",
            "memory_addresses",
            "memory_slots",
            "memory_route_loss",
            "memory_route_peak",
            "memory_route_entropy_fraction",
            "memory_gate",
            "memory_coverage_strength",
            "address_field_cells",
            "address_route_loss",
            "address_route_peak",
            "address_route_entropy_fraction",
            "address_expected_l1_0_1000",
            "address_gate",
            "address_coverage_strength",
            "address_coverage_sum",
            "address_existence_loss",
            "address_existence_probability",
            "address_existence_accuracy",
            "address_positive_existence",
            "address_negative_existence",
            "address_continuation_gate",
            "marked_field_cells",
            "marked_route_loss",
            "marked_route_entropy_fraction",
            "marked_target_mass",
            "marked_existence_loss",
            "marked_existence_probability",
            "marked_existence_accuracy",
            "marked_positive_existence",
            "marked_negative_existence",
            "marked_lane_gate",
            "marked_coverage_strength",
            "marked_coverage_fraction",
            "marked_structure_scale",
            "marked_none_scale",
        ):
            present = [item[key] for item in running if key in item]
            if present:
                record[key] = sum(present) / len(present)
        running.clear()
        append_jsonl(log_path, record)
        if step == 1 or step % 10 == 0:
            print(json.dumps({"arm": args.arm, **record}, sort_keys=True), flush=True)
        if step % args.eval_steps == 0 or step == args.max_steps:
            metrics = validate(wrapper, validation_dataset, validation_indices, args.workers)
            checkpoint = output / f"checkpoint-step{step:04d}.pt"
            payload = checkpoint_payload(
                args=args,
                model=model,
                controller=controller,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                exposure=exposure,
                validation=metrics,
                audit=audit,
            )
            atomic_torch_save(checkpoint, payload)
            update_symlink(output / "last.pt", checkpoint)
            selected_validation = float(metrics[args.best_validation_metric])
            if selected_validation < best_validation:
                best_validation = selected_validation
                update_symlink(output / "best.pt", checkpoint)
            atomic_json(
                output / "status.json",
                {
                    "arm": args.arm,
                    "step": step,
                    "exposure": exposure,
                    "validation": metrics,
                    "best_validation_metric": args.best_validation_metric,
                    "best_validation_value": best_validation,
                    "best_validation_native_loss": (
                        best_validation if args.best_validation_metric == "native_loss" else None
                    ),
                    "complete": step == args.max_steps,
                },
            )
            print(json.dumps({"arm": args.arm, "validation": metrics, "step": step}, sort_keys=True), flush=True)
        if step >= args.max_steps:
            break
    if step != args.max_steps or exposure != run_exposures:
        raise RuntimeError(f"Coverage incomplete: step={step}, exposure={exposure}")
    atomic_json(
        output / "done.json",
        {
            "arm": args.arm,
            "steps": step,
            "record_exposures": exposure,
            "unique_training_records": run_exposures,
            "complete_one_pass": not args.smoke,
            "smoke": bool(args.smoke),
            "best": str((output / "best.pt").resolve()),
            "last": str((output / "last.pt").resolve()),
        },
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=[
            "standard",
            "point_aux",
            "pointrelay",
            "decoder_point_aux",
            "decoder_pointrelay",
            "pointroute_aux",
            "pointroute",
            "setpoint_aux",
            "setpoint_relay",
            "causal_pointrelay",
            "adaptive_memory",
            "dense_address_field",
            "marked_residual_address",
        ],
        required=True,
    )
    parser.add_argument("--model", type=Path, default=PROJECT / "model/LocateAnything-3B/checkpoint")
    parser.add_argument("--train-data", type=Path, default=PROJECT / "dataset/sft/pointrelay_pbd_lora_pilot_v1/train.jsonl")
    parser.add_argument("--validation-data", type=Path, default=PROJECT / "dataset/sft/pointrelay_pbd_lora_pilot_v1/validation.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--relay-layer", type=int, default=18)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--point-dim", type=int, default=128)
    parser.add_argument("--target-sigma", type=float, default=0.85)
    parser.add_argument("--initial-gate-logit", type=float, default=-3.0)
    parser.add_argument("--local-radius-fraction", type=float, default=0.25)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--field-dim", type=int, default=128)
    parser.add_argument("--field-target-sigma", type=float, default=0.70)
    parser.add_argument("--subpatch-dim", type=int, default=32)
    parser.add_argument("--subpatch-grid", type=int, default=4)
    parser.add_argument("--local-patch-grid", type=int, default=12)
    parser.add_argument("--max-memory-addresses", type=int, default=64)
    parser.add_argument("--local-encode-batch", type=int, default=8)
    parser.add_argument("--initial-coverage-strength", type=float, default=1.0)
    parser.add_argument("--residual-floor", type=float, default=0.002)
    parser.add_argument("--footprint-halo-cells", type=float, default=0.35)
    parser.add_argument("--initial-structure-scale", type=float, default=1.5)
    parser.add_argument("--absence-loss-weight", type=float, default=2.0)
    parser.add_argument("--point-loss-weight", type=float, default=0.2)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.0)
    parser.add_argument("--eos-loss-weight", type=float, default=0.0)
    parser.add_argument("--none-loss-weight", type=float, default=0.0)
    parser.add_argument("--history-loss-weight", type=float, default=0.0)
    parser.add_argument("--history-margin", type=float, default=0.5)
    parser.add_argument("--geometry-rollin-max-probability", type=float, default=0.0)
    parser.add_argument("--geometry-rollin-start-step", type=int, default=500)
    parser.add_argument("--geometry-rollin-ramp-steps", type=int, default=1000)
    parser.add_argument(
        "--best-validation-metric", choices=("native_loss", "loss"), default="native_loss"
    )
    parser.add_argument("--image-token-limit", type=int, default=1024)
    parser.add_argument("--max-sequence", type=int, default=8192)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--controller-learning-rate", type=float, default=3e-4)
    parser.add_argument("--joint-controller-optimizer", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--validation-records", type=int, default=40)
    parser.add_argument("--expected-train-records", type=int, default=600)
    parser.add_argument("--expected-validation-records", type=int, default=120)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true")
    return parser


if __name__ == "__main__":
    train(make_parser().parse_args())
