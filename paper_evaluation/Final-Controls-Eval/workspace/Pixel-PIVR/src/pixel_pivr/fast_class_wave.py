"""Opt-in execution optimizations; preserve native LA probabilities and grammar."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .hybrid_class_wave import CachedHybridClassWaveDecoder
from .shared_prefix import SharedPrefixCache


def pack_suffix_views(prefix, states):
    lengths = [int(s.suffix_kv[0][0].shape[-2]) for s in states]
    width = max(lengths)
    cache = SharedPrefixCache(prefix)
    for layer_index, layer in enumerate(cache.layers):
        for lane, name in enumerate(("suffix_keys", "suffix_values")):
            rows = [s.suffix_kv[layer_index][lane] for s in states]
            # F.pad(..., 0) still allocates. Aligned histories need no padding.
            rows = [r if n == width else F.pad(r, (0, 0, width - n, 0))
                    for r, n in zip(rows, lengths)]
            setattr(layer, name, torch.cat(rows))
    device = prefix[0][0].device
    mask = torch.arange(width, device=device).unsqueeze(0) >= (
        width - torch.tensor(lengths, device=device).unsqueeze(1))
    return cache, mask.long(), lengths


def unpack_suffix_views(cache, states, lengths, width, pending_lengths, pending_width=0):
    for index, (state, length, pending) in enumerate(zip(states, lengths, pending_lengths)):
        start = width - length
        rows = []
        for layer in cache.layers:
            pair = []
            for storage in (layer.suffix_keys, layer.suffix_values):
                if not pending or pending == pending_width:
                    # The parent is immutable after this call. Exclude speculative
                    # PBD lanes, retain only the accepted history as a view.
                    value = storage[index:index + 1, :, start:width + pending, :]
                else:
                    value = torch.cat((storage[index:index + 1, :, start:width, :],
                        storage[index:index + 1, :, width + pending_width - pending:
                                width + pending_width, :]), dim=-2)
                pair.append(value)
            rows.append(tuple(pair))
        state.suffix_kv = tuple(rows)


class FastCachedHybridClassWaveDecoder(CachedHybridClassWaveDecoder):
    """Batch deterministic sampling operations, retain per-branch RNG streams."""

    def __init__(self, *args, batch_sampling=True, cache_views=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sampling = bool(batch_sampling)
        self.cache_views = bool(cache_views)

    def _pack_cache(self, prefix, states):
        return (pack_suffix_views(prefix, states) if self.cache_views
                else super()._pack_cache(prefix, states))

    def _unpack_cache(self, *args):
        return (unpack_suffix_views(*args) if self.cache_views
                else super()._unpack_cache(*args))

    def _sample_group(self, logits, states, rngs, kwargs, use_mtp):
        if not self.batch_sampling:
            return super()._sample_group(logits, states, rngs, kwargs, use_mtp)
        helpers = self._sample_tokens.__globals__
        histories = []
        for state in states:
            if not hasattr(state, "seen_tokens"):
                state.seen_tokens = set(state.full_prompt_ids)
                state.seen_answer_count = 0
            state.seen_tokens.update(state.answer_ids[state.seen_answer_count:])
            state.seen_answer_count = len(state.answer_ids)
            histories.append(sorted(state.seen_tokens))
        width = max(map(len, histories))
        # Padding repeats an already-present token, never a new pad-token penalty.
        history = torch.tensor([r + [r[0]] * (width - len(r)) for r in histories],
                               dtype=torch.long, device=logits.device)
        penalty = float(kwargs.get("repetition_penalty", 1.0))
        if penalty != 1.0:
            logits = helpers["apply_repetition_penalty"](logits, history, penalty)
        temperature = float(kwargs.get("temperature", 0))
        if temperature > 0:
            logits = logits / temperature
        top_p, top_k = kwargs.get("top_p"), kwargs.get("top_k")
        if top_p is not None and top_p < 1:
            # BF16 ties can be ordered differently by batched sort kernels.
            # Keep the native per-branch top-p call shape and threshold mask.
            logits = torch.cat([helpers["top_p_logits"](logits[i:i + 1], top_p)
                                for i in range(len(states))])
        if top_k is not None:
            logits = helpers["top_k_logits"](logits, top_k)
        probs = logits.softmax(dim=-1)
        results = []
        ids = self.model.token_ids
        for index, state in enumerate(states):
            row = probs[index:index + 1]

            def sample():
                if temperature > 0:
                    try:
                        return torch.distributions.Categorical(probs=row).sample()
                    except Exception:
                        return row.max(dim=-1)[1]
                return row.max(dim=-1)[1]

            x0 = rngs[id(state)].call(sample)
            if not use_mtp:
                token = int(x0[0, 0])
                kind = ("box_end_ar" if token == ids["box_end_token_id"] else
                        "coord_ar" if ids["coord_start_token_id"] <= token <= ids["coord_end_token_id"]
                        or token == ids["none_token_id"] else "im_end")
                results.append((kind, [token]))
                continue
            # Preserve native keep_k_avg=4, thresholds, and hybrid AR fallback.
            candidate = helpers["decode_bbox_avg"](logits[index], row[0], ids,
                keep_k=kwargs.get("keep_k_avg", 4), generation_mode=kwargs["generation_mode"])
            if candidate is None:
                candidate = helpers["decode_ref"](logits[index], row[0], ids)
            if candidate is None:
                candidate = x0[0]
            pattern = self._handle_pattern(candidate, ids, kwargs["generation_mode"])
            results.append((str(pattern["type"]), [int(t) for t in pattern["tokens"]]))
        return results
