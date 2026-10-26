"""One-encode, branch-isolated CAW-PIVR episode objective."""

from __future__ import annotations

from collections import Counter
import sys
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .lora import base_causal_lm
from .unified_episode import IGNORE_INDEX, pack_wave

OBJECTIVE_MODES = ("unified", "direct_only", "two_route_token_mean", "two_route_equal_mean")


def objective_waves(sample: Mapping[str, Any], objective_mode: str):
    if objective_mode == "direct_only":
        return sample["direct_waves"]
    if objective_mode == "unified":
        return sample["episode_waves"]
    if objective_mode in {"two_route_token_mean", "two_route_equal_mean"}:
        # Removing branches cannot increase a wave's sequence length. Keep the
        # original wave ordering, but never forward an inactive direct branch.
        return [
            retained
            for wave in sample["episode_waves"]
            if (retained := [b for b in wave if b["route"] in {"round1", "round2"}])
        ]
    raise ValueError(f"Unsupported objective mode: {objective_mode!r}")


def supervised_token_count(branch: Mapping[str, Any]) -> int:
    # Label zero has no predecessor, matching _branch_cross_entropy exactly.
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
        return explicit, explicit.eq(0)

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
    start, stop = map(int, branch_range)
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
        total = total + F.cross_entropy(
            logits, targets[offset:end], reduction="sum"
        )
    return total / int(targets.numel()), int(targets.numel())


def branch_objective_weights(
    task: str,
    waves_by_route: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]],
    *,
    objective_mode: str = "unified",
) -> dict[str, float]:
    """Assign a source-query-normalized weight to every branch."""
    branches = {
        route: [branch for wave in waves for branch in wave]
        for route, waves in waves_by_route.items()
    }
    if objective_mode == "two_route_equal_mean":
        routes = ("round1",) if task == "pointing" else ("round1", "round2")
        result = {}
        for route in routes:
            values = branches.get(route, [])
            counts = {str(b["branch_id"]): supervised_token_count(b) for b in values}
            if (not counts or len(counts) != len(values) or min(counts.values()) <= 0
                    or set(counts).intersection(result)):
                raise RuntimeError("Equal-route objective requires unique, supervised branches in each route")
            denominator = sum(counts.values())
            result.update({key: count / denominator / len(routes) for key, count in counts.items()})
        return result
    if objective_mode == "two_route_token_mean":
        retained = [
            value for route in ("round1", "round2")
            for value in branches.get(route, [])
            if task != "pointing" or route == "round1"
        ]
        counts = {str(value["branch_id"]): supervised_token_count(value) for value in retained}
        if len(counts) != len(retained) or not counts or min(counts.values()) <= 0:
            raise RuntimeError("Two-route objective requires unique, supervised branches")
        denominator = sum(counts.values())
        # Each branch CE is already a token mean; these weights reconstruct one
        # episode-wide token mean, independent of how branches split into waves.
        return {key: count / denominator for key, count in counts.items()}
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
                "positive": [
                    value for value in values
                    if str(value.get("polarity")) == "positive"
                ],
                "negative": [
                    value for value in values
                    if str(value.get("polarity")) != "positive"
                ],
            }
            active_round1 = {
                key: group for key, group in grouped_round1.items() if group
            }
            stratum_weight = route_weight / len(active_round1)
            for group in active_round1.values():
                each = stratum_weight / len(group)
                result.update(
                    {str(value["branch_id"]): each for value in group}
                )
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
        requested = {"positive": 0.50, "background": 0.25, "wrong_class": 0.25}
        active = {key: weight for key, weight in requested.items() if grouped[key]}
        normalizer = sum(active.values())
        if normalizer <= 0:
            raise RuntimeError("Round-2 route contains no valid branch stratum")
        for key, stratum_weight in active.items():
            each = route_weight * (stratum_weight / normalizer) / len(grouped[key])
            result.update(
                {str(value["branch_id"]): each for value in grouped[key]}
            )

    total = sum(result.values())
    if abs(total - 1.0) > 1e-6:
        raise RuntimeError(f"Episode branch weights sum to {total}, expected 1")
    if len(result) != sum(len(branches[route]) for route in available):
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
        pixel_values = sample["pixel_values"].to(
            device=self.device, dtype=torch.bfloat16
        )
        grids = torch.as_tensor(
            sample["image_grid_hws"], device=self.device, dtype=torch.int32
        )
        visual = self.model.extract_feature(pixel_values, grids)
        projected = self.model.mlp1(torch.cat(visual, dim=0))
        self.vision_encode_calls += 1
        return projected.detach()

    def wave_losses(
        self,
        wave: Sequence[Mapping[str, Any]],
        projected: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[int], dict[str, float]]:
        causal = self.causal
        packed = pack_wave(
            wave,
            image_token_index=int(self.model.image_token_index),
            device=self.device,
            dtype=torch.bfloat16,
        )
        input_ids = packed["input_ids"]
        selected = input_ids.eq(int(self.model.image_token_index))
        if int(selected.sum()) != int(projected.shape[0]):
            raise RuntimeError(
                "Shared visual prefix mismatch: "
                f"placeholders={int(selected.sum())}, projected={int(projected.shape[0])}"
            )
        embeddings = self.model.language_model.get_input_embeddings()(input_ids)
        flat = embeddings.reshape(-1, embeddings.shape[-1]).clone()
        flat[selected.reshape(-1)] = projected.to(flat.dtype)
        embeddings = flat.view_as(embeddings)
        causal.model._caw_pivr_explicit_attention_mask = packed["attention_mask"]
        try:
            outputs = causal.model(
                input_ids=None,
                inputs_embeds=embeddings,
                attention_mask=None,
                position_ids=packed["position_ids"],
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            causal.model._caw_pivr_explicit_attention_mask = None
        losses: list[torch.Tensor] = []
        tokens: list[int] = []
        for branch_range in packed["branch_ranges"]:
            loss, count = _branch_cross_entropy(
                outputs.last_hidden_state,
                packed["labels"],
                branch_range,
                causal.lm_head.weight,
            )
            losses.append(loss)
            tokens.append(count)
        metrics = {
            "wave_sequence_tokens": float(input_ids.shape[1]),
            "wave_branches": float(len(wave)),
            "wave_shared_prefix_tokens": float(packed["prefix_length"]),
        }
        return losses, tokens, metrics

    def episode(
        self,
        sample: Mapping[str, Any],
        *,
        gradient_accumulation: int,
        backward: bool,
    ) -> dict[str, float]:
        waves_by_route = {
            "direct": sample["direct_waves"],
            "round1": sample["round1_waves"],
            "round2": sample["round2_waves"],
        }
        weights = branch_objective_weights(
            str(sample["task"]),
            waves_by_route,
            objective_mode=self.objective_mode,
        )
        before = self.vision_encode_calls
        projected = self.encode_global(sample)
        weighted_total = 0.0
        route_totals: Counter[str] = Counter()
        route_weights: Counter[str] = Counter()
        supervised_tokens = 0
        wave_count = 0
        maximum_wave_tokens = 0
        maximum_wave_branches = 0

        episode_waves = objective_waves(sample, self.objective_mode)
        executed_ids = [str(branch["branch_id"]) for wave in episode_waves for branch in wave]
        if len(executed_ids) != len(set(executed_ids)) or set(executed_ids) != set(weights):
            raise RuntimeError("Executed branches differ from the episode objective")
        for wave in episode_waves:
            route = str(wave[0]["route"])
            # Waves may contain several routes; every branch keeps its own
            # source-query-normalized objective weight.
            for branch in wave:
                if str(branch["route"]) not in waves_by_route:
                    raise RuntimeError(f"Unknown branch route: {branch['route']}")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses, token_counts, wave_metrics = self.wave_losses(wave, projected)
                objective = sum(
                    loss * weights[str(branch["branch_id"])]
                    for loss, branch in zip(losses, wave)
                )
                scaled = objective / int(gradient_accumulation)
            if not torch.isfinite(objective.detach()).item():
                raise FloatingPointError(f"Non-finite episode loss: {sample['episode_id']}")
            if backward:
                scaled.backward()
            for loss, branch in zip(losses, wave):
                branch_route = str(branch["route"])
                weight = weights[str(branch["branch_id"])]
                value = float(loss.detach().float())
                weighted_total += value * weight
                route_totals[branch_route] += value * weight
                route_weights[branch_route] += weight
            supervised_tokens += sum(token_counts)
            wave_count += 1
            maximum_wave_tokens = max(
                maximum_wave_tokens, int(wave_metrics["wave_sequence_tokens"])
            )
            maximum_wave_branches = max(
                maximum_wave_branches, int(wave_metrics["wave_branches"])
            )
            del losses, objective, scaled

        observed = self.vision_encode_calls - before
        if observed != 1:
            raise RuntimeError(f"Episode encoded global pixels {observed} times, expected 1")
        metrics = {
            "loss": weighted_total,
            "direct_loss": route_totals["direct"] / max(route_weights["direct"], 1e-12),
            "round1_loss": route_totals["round1"] / max(route_weights["round1"], 1e-12),
            "round2_loss": route_totals["round2"] / max(route_weights["round2"], 1e-12),
            "direct_active": float(bool(route_weights["direct"])),
            "round1_active": float(bool(route_weights["round1"])),
            "round2_active": float(bool(route_weights["round2"])),
            "round1_branches": float(sum(b["route"] == "round1" for w in episode_waves for b in w)),
            "round2_branches": float(sum(b["route"] == "round2" for w in episode_waves for b in w)),
            "supervised_tokens": float(supervised_tokens),
            "waves": float(wave_count),
            "maximum_wave_tokens": float(maximum_wave_tokens),
            "maximum_wave_branches": float(maximum_wave_branches),
            "global_visual_tokens": float(projected.shape[0]),
            "vision_encode_calls": float(observed),
        }
        return metrics
