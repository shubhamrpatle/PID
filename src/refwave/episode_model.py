"""One-encode, branch-isolated CAW-PIVR episode objective."""

from __future__ import annotations
from collections import Counter
import sys
from typing import Any, Mapping, Sequence
import torch
import torch.nn.functional as F
from torch import nn
from .lora import base_causal_lm
from .episode import IGNORE_INDEX, pack_wave

OBJECTIVE_MODES = ("unified", "direct_only", "two_route_token_mean", "two_route_equal_mean")


def objective_waves(sample: Mapping[str, Any], objective_mode: str):
    if objective_mode == "direct_only":
        return sample["direct_waves"]
    if objective_mode == "unified":
        return sample["episode_waves"]
    if objective_mode in {"two_route_token_mean", "two_route_equal_mean"}:
        return [
            retained
            for wave in sample["episode_waves"]
            if (retained := [b for b in wave if b["route"] in {"round1", "round2"}])
        ]
    raise ValueError(f"Unsupported objective mode: {objective_mode!r}")


def supervised_token_count(branch: Mapping[str, Any]) -> int:
    return int(branch["labels"].reshape(-1)[1:].ne(IGNORE_INDEX).sum())


def _install_explicit_mask_hook(decoder: nn.Module) -> None:
    """Scope an explicit training mask to this dynamic Qwen model instance.

    Released LocateAnything replaces any caller-provided mask with its native
    PBD6 mask while training.  The unified objective still needs native PBD6,
    plus sibling-branch isolation.  Intercept only the mask factory while an
    explicit episode wave is active; every ordinary model call remains exact.
    """
    module = sys.modules.get(decoder.__class__.__module__)
    if module is None or not hasattr(module, "create_block_diff_mask_by_pe_4d"):
        raise RuntimeError("Cannot locate LocateAnything's PBD6 mask factory")
    owner = getattr(module, "_caw_pivr_explicit_mask_owner", None)
    if owner is not None:
        if owner is not decoder:
            raise RuntimeError("The Qwen module already has a different CAW-PIVR mask owner")
        return
    original = module.create_block_diff_mask_by_pe_4d

    def explicit_or_native(*args: Any, **kwargs: Any):
        explicit = getattr(decoder, "_caw_pivr_explicit_attention_mask", None)
        if explicit is None:
            return original(*args, **kwargs)
        return (explicit, explicit.eq(0))

    module._caw_pivr_native_mask_factory = original
    module._caw_pivr_explicit_mask_owner = decoder
    module.create_block_diff_mask_by_pe_4d = explicit_or_native


def _branch_cross_entropy(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    branch_range: tuple[int, int],
    lm_head_weight: torch.Tensor,
    *,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, int]:
    """Return token-mean next-token CE for one isolated branch suffix."""
    (start, stop) = map(int, branch_range)
    label_positions = torch.arange(start, stop, device=labels.device)
    branch_labels = labels[0, start:stop]
    valid = branch_labels.ne(IGNORE_INDEX) & label_positions.gt(0)
    label_positions = label_positions[valid]
    targets = branch_labels[valid]
    if not int(targets.numel()):
        raise RuntimeError(f"Branch range {branch_range} contains no supervised tokens")
    total = hidden_states.new_zeros((), dtype=torch.float32)
    for offset in range(0, int(targets.numel()), int(chunk_size)):
        end = min(int(targets.numel()), offset + int(chunk_size))
        states = hidden_states[0, label_positions[offset:end] - 1]
        logits = F.linear(states, lm_head_weight).float()
        total = total + F.cross_entropy(logits, targets[offset:end], reduction="sum")
    return (total / int(targets.numel()), int(targets.numel()))


def branch_objective_weights(
    task: str,
    waves_by_route: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]],
    *,
    objective_mode: str = "unified",
) -> dict[str, float]:
    """Assign a source-query-normalized weight to every branch."""
    branches = {
        route: [branch for wave in waves for branch in wave]
        for (route, waves) in waves_by_route.items()
    }
    if objective_mode == "two_route_equal_mean":
        routes = ("round1",) if task == "pointing" else ("round1", "round2")
        result = {}
        for route in routes:
            values = branches.get(route, [])
            counts = {str(b["branch_id"]): supervised_token_count(b) for b in values}
            if (
                not counts
                or len(counts) != len(values)
                or min(counts.values()) <= 0
                or set(counts).intersection(result)
            ):
                raise RuntimeError(
                    "Equal-route objective requires unique, supervised branches in each route"
                )
            denominator = sum(counts.values())
            result.update(
                {key: count / denominator / len(routes) for (key, count) in counts.items()}
            )
        return result
    if objective_mode == "two_route_token_mean":
        retained = [
            value
            for route in ("round1", "round2")
            for value in branches.get(route, [])
            if task != "pointing" or route == "round1"
        ]
        counts = {str(value["branch_id"]): supervised_token_count(value) for value in retained}
        if len(counts) != len(retained) or not counts or min(counts.values()) <= 0:
            raise RuntimeError("Two-route objective requires unique, supervised branches")
        denominator = sum(counts.values())
        return {key: count / denominator for (key, count) in counts.items()}
    if objective_mode == "direct_only":
        available = ["direct"] if branches.get("direct") else []
    elif objective_mode == "unified":
        available = [
            route
            for route in ("direct", "round1", "round2")
            if branches.get(route) and (task != "pointing" or route == "round1")
        ]
    else:
        raise ValueError(f"Unsupported objective mode: {objective_mode!r}")
    if not available:
        raise RuntimeError(f"Episode task {task!r} has no supervised route")
    route_weight = 1.0 / len(available)
    result: dict[str, float] = {}
    for route in available:
        values = branches[route]
        if route == "direct":
            each = route_weight / len(values)
            result.update({str(value["branch_id"]): each for value in values})
            continue
        if route == "round1":
            grouped_round1 = {
                "positive": [value for value in values if str(value.get("polarity")) == "positive"],
                "negative": [value for value in values if str(value.get("polarity")) != "positive"],
            }
            active_round1 = {key: group for (key, group) in grouped_round1.items() if group}
            stratum_weight = route_weight / len(active_round1)
            for group in active_round1.values():
                each = stratum_weight / len(group)
                result.update({str(value["branch_id"]): each for value in group})
            continue
        grouped: dict[str, list[Mapping[str, Any]]] = {
            "positive": [],
            "background": [],
            "wrong_class": [],
        }
        for value in values:
            if str(value.get("polarity")) == "positive":
                grouped["positive"].append(value)
            else:
                kind = str(value.get("negative_type") or "background")
                grouped[kind if kind in grouped else "background"].append(value)
        requested = {"positive": 0.5, "background": 0.25, "wrong_class": 0.25}
        active = {key: weight for (key, weight) in requested.items() if grouped[key]}
        normalizer = sum(active.values())
        if normalizer <= 0:
            raise RuntimeError("Round-2 route contains no valid branch stratum")
        for key, stratum_weight in active.items():
            each = route_weight * (stratum_weight / normalizer) / len(grouped[key])
            result.update({str(value["branch_id"]): each for value in grouped[key]})
    total = sum(result.values())
    if abs(total - 1.0) > 1e-06:
        raise RuntimeError(f"Episode branch weights sum to {total}, expected 1")
    if len(result) != sum((len(branches[route]) for route in available)):
        raise RuntimeError("A retained branch is missing from the episode objective")
    return result


class UnifiedEpisodeTrainingModel(nn.Module):
    """Frozen image encoder/projector with trainable Qwen LoRA branch waves."""

    def __init__(self, model: nn.Module, *, objective_mode: str = "unified") -> None:
        super().__init__()
        if objective_mode not in OBJECTIVE_MODES:
            raise ValueError(f"Unsupported objective mode: {objective_mode!r}")
        self.model = model
        self.objective_mode = objective_mode
        self.vision_encode_calls = 0
        self.causal = base_causal_lm(self.model.language_model)
        _install_explicit_mask_hook(self.causal.model)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.vision_model.eval()
        self.model.mlp1.eval()
        return self

    @torch.no_grad()
    def encode_global(self, sample: Mapping[str, Any]) -> torch.Tensor:
        pixel_values = sample["pixel_values"].to(device=self.device, dtype=torch.bfloat16)
        grids = torch.as_tensor(sample["image_grid_hws"], device=self.device, dtype=torch.int32)
        visual = self.model.extract_feature(pixel_values, grids)
        projected = self.model.mlp1(torch.cat(visual, dim=0))
        self.vision_encode_calls += 1
        return projected.detach()
