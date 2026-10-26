"""Diagnostic PBD history control; not a default RefWave inference mode."""
from __future__ import annotations

import time

import torch

from .fused_pivr_geometry import FusedPIVRGeometryDecoder, fused_prompt_geometry_inputs
from .pixel_pivr_wave import _SharedPrefixCache
from .pointrelay_pbd import base_causal_lm


def discard_prediction_window(cache, width=6):
    """Retain real prompt/history states, never the speculative mask positions."""
    for layer in cache.layers:
        if layer.suffix_keys is None or layer.suffix_keys.shape[-2] < width:
            raise RuntimeError("Missing speculative geometry window")
        layer.suffix_keys = layer.suffix_keys[..., :-width, :]
        layer.suffix_values = layer.suffix_values[..., :-width, :]


@torch.no_grad()
def history_pbd_step(qwen, cache, suffix_ids, mask_id):
    if qwen.training or qwen._attn_implementation != "sdpa" or qwen.block_size != 6:
        raise ValueError("History diagnostic requires eval-mode native SDPA PBD6")
    if suffix_ids.ndim != 2 or suffix_ids.shape[0] != 1 or not suffix_ids.shape[1]:
        raise ValueError("History decoding consumes one nonempty instance suffix")
    geometry = torch.cat((suffix_ids[:, -1:], torch.full(
        (1, 5), mask_id, dtype=suffix_ids.dtype, device=suffix_ids.device)), dim=1)
    before = cache.get_seq_length()
    total = before + suffix_ids.shape[1] + 6
    limit = int(qwen.config.max_position_embeddings)
    if total > limit:
        raise ValueError(f"History requires {total} positions > {limit}; no targets truncated")
    ids, visibility, positions = fused_prompt_geometry_inputs(
        suffix_ids, torch.ones_like(suffix_ids), before,
        geometry, torch.ones_like(geometry), qwen.get_input_embeddings().weight.dtype,
        causal_attn=qwen.causal_attn)
    output = qwen(input_ids=ids, attention_mask=visibility, position_ids=positions,
        visual_features=None, image_token_index=None, past_key_values=cache,
        use_cache=True, output_attentions=False, output_hidden_states=False, return_dict=True)
    hidden = output.last_hidden_state[:, -6:, :]
    discard_prediction_window(cache)
    if cache.get_seq_length() != before + suffix_ids.shape[1]:
        raise RuntimeError("History cache retained speculative geometry positions")
    return hidden


def continuation_text(processor, question):
    """Use the processor's role delimiters without duplicating its system prompt."""
    system = {"role": "system", "content": ""}
    user = {"role": "user", "content": [{"type": "text", "text": question}]}
    prefix = processor.py_apply_chat_template([system], tokenize=False, add_generation_prompt=False)
    text = processor.py_apply_chat_template([system, user], tokenize=False, add_generation_prompt=True)
    if not text.startswith(prefix):
        raise ValueError("Processor chat template is not prefix-composable")
    return "<|im_end|>\n" + text[len(prefix):]


class CausalHistoryPIVRGeometryDecoder(FusedPIVRGeometryDecoder):
    """Keep earlier predicted answers as real causal context for subsequent boxes.

    Current reference wording and native geometry decoding remain unchanged.
    Cross-instance multi-turn history is new, untrained context: this is an
    inference-only diagnostic, not a matched trained mask ablation.
    """

    @torch.no_grad()
    def decode_prepared_image(self, prepared_image, branches, *, requested_wave_size=1,
                              allow_none=True):
        if requested_wave_size != 1:
            raise ValueError("A generated-history dependency requires serial decoding")
        if self.visual_context != "global_only" or self.prefix_cache_mode != "shared":
            raise ValueError("History diagnostic supports shared global-only context")
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        if not branches:
            return [], self._stats(0, [], 0, 0., 0.)
        prefix, suffixes, _ = self._prepared_global_suffixes(prepared_image, branches)
        cache = _SharedPrefixCache(prepared_image.key_values)
        if cache.get_seq_length() != prefix.numel():
            raise ValueError("Prepared prefix length does not match history cache")
        qwen = base_causal_lm(self.model.language_model).model
        continuation_ids = [None] + [self.tokenizer.encode(
            continuation_text(self.processor, b.question), add_special_tokens=False)
            for b in branches[1:]]
        outputs, lengths, previous_tokens = [], [], None
        preparation_seconds = time.perf_counter() - started
        for index, branch in enumerate(branches):
            if index == 0:
                suffix = suffixes[0].reshape(1, -1).to(device)
            else:
                suffix = torch.tensor([previous_tokens + continuation_ids[index]],
                                      dtype=torch.long, device=device)
            lengths.append(cache.get_seq_length() + suffix.shape[1] + 6)
            hidden = history_pbd_step(qwen, cache, suffix, self.mask_id)
            result = self._geometry_from_hidden(hidden, [branch], allow_none=allow_none)[0]
            previous_tokens = [int(v) for v in result["block_tokens"]]
            if self.mask_id in previous_tokens:
                raise RuntimeError("Speculative mask tokens cannot enter emitted history")
            result["history_instances"] = index
            result["context_positions"] = lengths[-1]
            outputs.append(result)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device)/1024**2 if device.type == "cuda" else 0.
        return outputs, self._stats(len(branches), lengths, peak, elapsed, preparation_seconds)

    @staticmethod
    def _stats(count, lengths, peak, elapsed, preparation):
        return dict(addresses=count, waves=count, requested_wave_size=1,
            effective_wave_sizes=[1]*count, qwen_total_forwards=count,
            qwen_fused_prompt_geometry_forwards=count,
            round2_execution="fused_one_pass_causal_history_diagnostic",
            history_policy="Previous model-predicted boxes/None and prompts in original point order",
            context_positions=lengths, max_context_positions=max(lengths, default=0),
            peak_gpu_mb=peak, end_to_end_seconds=elapsed, preparation_seconds=preparation,
            global_moonvit_encodes=0, local_moonvit_encodes=0,
            checkpoint_trained_for_history=False)
