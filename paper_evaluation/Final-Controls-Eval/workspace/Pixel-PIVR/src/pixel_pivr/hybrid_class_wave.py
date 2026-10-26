"""Experimental cached hybrid class waves; native MTP/AR, without prompt replay.

This is opt-in. It keeps one image prefix and independent suffix K/V per branch.
Qwen selects the MTP attention mask for an entire batch, so MTP and AR branches
execute in separate mode groups. No geometry is repaired or supplied externally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .class_wave import (
    ClassAddressedPointWaveDecoder, ClassPointBranch, _BranchState, _BranchRNG,
    _left_pad_rows,
)
from .lora import base_causal_lm
from .shared_prefix import SharedPrefixCache


@dataclass
class _HybridState(_BranchState):
    use_mtp: bool = True
    ar_transitions: int = 0
    suffix_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()


def _pack_suffix_cache(prefix_kv, states):
    lengths = [int(state.suffix_kv[0][0].shape[-2]) for state in states]
    width = max(lengths)
    cache = SharedPrefixCache(prefix_kv)
    for layer_index, layer in enumerate(cache.layers):
        layer.suffix_keys = torch.cat([
            F.pad(state.suffix_kv[layer_index][0], (0, 0, width - length, 0))
            for state, length in zip(states, lengths)
        ])
        layer.suffix_values = torch.cat([
            F.pad(state.suffix_kv[layer_index][1], (0, 0, width - length, 0))
            for state, length in zip(states, lengths)
        ])
    device = prefix_kv[0][0].device
    mask = torch.arange(width, device=device).unsqueeze(0) >= (
        width - torch.tensor(lengths, device=device).unsqueeze(1)
    )
    return cache, mask.long(), lengths


def _unpack_suffix_cache(cache, states, lengths, width, pending_lengths, pending_width=0):
    # Only accepted tokens persist. Discard left padding and speculative PBD6.
    for index, (state, length, pending) in enumerate(zip(states, lengths, pending_lengths)):
        start = width - length
        next_start, next_stop = width + pending_width - pending, width + pending_width
        state.suffix_kv = tuple((
            torch.cat((layer.suffix_keys[index:index + 1, :, start:width, :],
                       layer.suffix_keys[index:index + 1, :, next_start:next_stop, :]), dim=-2),
            torch.cat((layer.suffix_values[index:index + 1, :, start:width, :],
                       layer.suffix_values[index:index + 1, :, next_start:next_stop, :]), dim=-2),
        ) for layer in cache.layers)


class CachedHybridClassWaveDecoder(ClassAddressedPointWaveDecoder):
    """Continue hybrid AR branches from their own accepted-token caches."""

    def _pack_cache(self, prefix, states):
        return _pack_suffix_cache(prefix, states)

    def _unpack_cache(self, *args):
        return _unpack_suffix_cache(*args)

    def _sample_group(self, logits, states, rngs, kwargs, use_mtp):
        sampler = self._sample_row if use_mtp else self._sample_ar_row
        return [sampler(logits[i:i + 1],
                        torch.tensor([state.full_prompt_ids + state.answer_ids], device=self.device),
                        rngs[id(state)], kwargs)
                for i, state in enumerate(states)]

    def _close_truncated_answer(self, answer, fallback_reason):
        return answer, False

    def _sample_ar_row(self, logits, generated, rng, generation_kwargs):
        _, _, x0, _ = rng.call(lambda: self._sample_tokens(
            logits, generated, self.model.token_ids, **dict(generation_kwargs)))
        token = int(x0[0, 0])
        ids = self.model.token_ids
        if token == ids["box_end_token_id"]:
            kind = "box_end_ar"
        elif ids["coord_start_token_id"] <= token <= ids["coord_end_token_id"] or token == ids["none_token_id"]:
            kind = "coord_ar"
        else:
            kind = "im_end"
        return kind, [token]

    def _decode_wave(
        self,
        prefix: Sequence[int],
        prefix_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
        branches: Sequence[ClassPointBranch],
        full_rows: Sequence[Sequence[int]],
        suffix_rows: Sequence[Sequence[int]],
        *, max_new_tokens: int, temperature: float, top_p: float,
        repetition_penalty: float, generation_mode: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if generation_mode not in {"fast", "hybrid"}:
            raise ValueError("Class waves support fast or hybrid mode")
        if not branches or not (len(branches) == len(full_rows) == len(suffix_rows)):
            raise ValueError("Branch, prompt, and suffix counts must match and be nonzero")
        if len(branches) > self.max_wave_size or max_new_tokens < 1:
            raise ValueError("Invalid wave size or generation budget")
        states = [_HybridState(branch, list(full), list(suffix), [], [])
                  for branch, full, suffix in zip(branches, full_rows, suffix_rows)]
        rngs = {id(state): _BranchRNG(state.branch.seed, self.device) for state in states}
        causal = base_causal_lm(self.model.language_model)
        suffix_ids, suffix_mask = _left_pad_rows(suffix_rows, self.pad_id, self.device)
        prefix_mask = torch.ones((len(states), len(prefix)), device=self.device, dtype=torch.long)
        full_mask = torch.cat((prefix_mask, suffix_mask), dim=1)
        positions = (full_mask.cumsum(-1) - 1).masked_fill(full_mask == 0, 1)
        cache = SharedPrefixCache(prefix_key_values)
        outputs = causal.model(
            input_ids=suffix_ids, visual_features=None, image_token_index=None,
            attention_mask=full_mask, position_ids=positions[:, -suffix_ids.shape[1]:],
            past_key_values=cache, use_cache=True, output_attentions=False,
            output_hidden_states=False, return_dict=True,
        )
        self._unpack_cache(cache, states, [len(s) for s in suffix_rows],
                             suffix_ids.shape[1], [0] * len(states))
        del cache, outputs
        kwargs: Mapping[str, Any] = dict(generation_mode=generation_mode,
            temperature=float(temperature), top_p=float(top_p),
            repetition_penalty=float(repetition_penalty))
        calls = {True: 0, False: 0}
        iterations = 0
        while any(not state.finished for state in states):
            for state in states:
                if not state.finished and len(state.answer_ids) >= max_new_tokens:
                    state.finished = True
                    state.fallback_reason = "max_new_tokens"
            # Snapshot groups so each active branch advances once per iteration.
            groups = [(mode, [s for s in states if not s.finished and s.use_mtp == mode])
                      for mode in (True, False)]
            if not any(group for _, group in groups):
                break
            iterations += 1
            for use_mtp, group in groups:
                if not group:
                    continue
                cache, past_suffix_mask, lengths = self._pack_cache(prefix_key_values, group)
                width = past_suffix_mask.shape[1]
                pending_lengths = [len(state.pending_ids) for state in group]
                pending_width = max(pending_lengths)
                pending = torch.full((len(group), pending_width), self.pad_id,
                                     device=self.device, dtype=torch.long)
                pending_mask = torch.zeros_like(pending)
                # LA suppresses the column immediately preceding PBD6. Every
                # branch's last accepted token must occupy that column, not pad.
                for index, state in enumerate(group):
                    if state.pending_ids:
                        pending[index, -len(state.pending_ids):] = torch.tensor(state.pending_ids, device=self.device)
                        pending_mask[index, -len(state.pending_ids):] = 1
                if pending_width and min(pending_lengths) == 0:
                    raise RuntimeError("Cannot mix zero-pending and nonzero-pending PBD branches")
                if use_mtp:
                    block = torch.full((len(group), 6), self.mask_id, device=self.device, dtype=torch.long)
                    block[:, 0] = torch.tensor([
                        (state.pending_ids or state.suffix_ids)[-1] for state in group
                    ], device=self.device)
                    ids = torch.cat((pending, block), dim=1)
                    current_mask = torch.cat((pending_mask, torch.ones_like(block)), dim=1)
                else:
                    if min(pending_lengths) < 1:
                        raise RuntimeError("AR continuation requires accepted pending tokens")
                    ids, current_mask = pending, pending_mask
                mask = torch.cat((
                    torch.ones((len(group), len(prefix)), device=self.device, dtype=torch.long),
                    past_suffix_mask, current_mask), dim=1)
                pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
                if use_mtp:
                    pos[:, -6:] -= 1
                outputs = causal.model(
                    input_ids=ids, visual_features=None, image_token_index=None,
                    attention_mask=mask, position_ids=pos[:, -ids.shape[1]:],
                    past_key_values=cache, use_cache=True, output_attentions=False,
                    output_hidden_states=False, return_dict=True,
                )
                if use_mtp:
                    hidden = outputs.last_hidden_state[:, -6:]
                else:
                    hidden = outputs.last_hidden_state[:, -1:]
                logits = F.linear(hidden, causal.lm_head.weight)
                self._unpack_cache(cache, group, lengths, width, pending_lengths, pending_width)
                calls[use_mtp] += 1
                sampled = self._sample_group(logits, group, rngs, kwargs, use_mtp)
                for state, (kind, tokens) in zip(group, sampled):
                    state.forward_steps += 1
                    state.answer_ids.extend(tokens)
                    state.pending_ids = tokens
                    if kind == "im_end":
                        state.finished = True
                    elif kind == "error_box":
                        state.use_mtp = False
                        state.ar_transitions += 1
                    elif kind == "box_end_ar":
                        state.use_mtp = True
                del cache, outputs, hidden, logits
        results = [{"label": s.branch.label, "question": s.branch.question,
            "seed": s.branch.seed, "answer": self.tokenizer.decode(s.answer_ids, skip_special_tokens=False),
            "tokens_generated": len(s.answer_ids), "forward_steps": s.forward_steps,
            "fallback_reason": s.fallback_reason, "hybrid_ar_transitions": s.ar_transitions,
            "full_prompt_replays": 0} for s in states]
        return results, {"branches": len(states), "iterations": iterations,
            "fallbacks": sum(s.fallback_reason is not None for s in states),
            "prefix_tokens": len(prefix), "maximum_suffix_tokens": max(map(len, suffix_rows)),
            "mtp_forward_calls": calls[True], "ar_forward_calls": calls[False],
            "hybrid_ar_transitions": sum(s.ar_transitions for s in states), "full_prompt_replays": 0}
