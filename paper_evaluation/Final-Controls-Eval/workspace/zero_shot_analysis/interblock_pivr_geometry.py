"""Optional causal inter-box attention; reference prompts stay branch-isolated."""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import torch

from .fused_pivr_geometry import FusedPIVRGeometryDecoder
from .pixel_pivr_wave import _SharedPrefixCache
from .pointrelay_pbd import base_causal_lm


@dataclass
class InterBlockState:
    prefix_length: int
    box_spans: list[tuple[int, int]] = field(default_factory=list)
    previous_prompt: tuple[int, int] | None = None
    previous_box_position: int | None = None


def interblock_inputs(suffix_ids, pending_ids, past_length, state, mask_id, dtype,
                      *, attention='causal'):
    """Build one real-box catch-up + current prompt + PBD6 prediction window.

    Previous prompts are invisible to the current branch. Only their owning
    generated box may read them when that box is committed. Branch-relative
    RoPE positions match the unchanged isolated decoder in both attention modes.
    """
    if attention not in ('causal', 'isolated'):
        raise ValueError('Expected causal or isolated inter-block attention')
    if suffix_ids.ndim != 2 or suffix_ids.shape[0] != 1 or suffix_ids.shape[1] == 0:
        raise ValueError('Expected one nonempty reference suffix')
    device = suffix_ids.device
    if pending_ids is None:
        pending_ids = suffix_ids.new_empty((1, 0))
    pending = pending_ids.shape[1]
    if pending_ids.shape != (1, pending) or pending not in (0, 3, 6):
        raise ValueError('Catch-up must be one actual box or three-token None')
    if (state.previous_prompt is None) != (pending == 0):
        raise ValueError('Missing or unexpected generated box catch-up')
    if pending and (pending_ids == mask_id).any():
        raise ValueError('Prediction masks must never be committed as generated boxes')
    expected = state.prefix_length if state.previous_prompt is None else state.previous_prompt[1]
    if past_length != expected:
        raise ValueError('Cache length differs from the committed block layout')
    width = suffix_ids.shape[1]
    start, end = past_length + pending, past_length + pending + width
    geometry = torch.cat((suffix_ids[:, -1:], suffix_ids.new_full((1, 5), mask_id)), dim=1)
    ids = torch.cat((pending_ids, suffix_ids, geometry), dim=1)
    visible = torch.zeros((1, 1, pending + width + 6, end + 6), device=device, dtype=dtype)
    visible[..., :state.prefix_length] = 1
    if pending:
        left, right = state.previous_prompt
        visible[:, :, :pending, left:right] = 1
        visible[:, :, :pending, past_length:past_length + pending] = 1
    visible[:, :, pending:pending + width, start:end] = torch.ones(
        (width, width), device=device, dtype=dtype).tril()
    visible[:, :, -6:, start:end - 1] = 1
    visible[:, :, -6:, -6:] = 1
    if attention == 'causal':
        for left, right in state.box_spans:
            visible[:, :, :pending, left:right] = 1
            visible[:, :, -6:, left:right] = 1
        visible[:, :, -6:, past_length:past_length + pending] = 1
    positions = [torch.arange(state.prefix_length, state.prefix_length + width, device=device),
                 torch.arange(state.prefix_length + width - 1, state.prefix_length + width + 5, device=device)]
    if pending:
        positions.insert(0, torch.arange(state.previous_box_position,
                                        state.previous_box_position + pending, device=device))
    return ids, visible, torch.cat(positions)[None], (start, end)


@torch.no_grad()
def interblock_step(qwen, cache, suffix_ids, state, mask_id, pending_ids=None,
                    *, attention='causal'):
    if qwen.training or qwen._attn_implementation != 'sdpa' or qwen.block_size != 6 or qwen.causal_attn:
        raise ValueError('Requires eval-mode native SDPA with bidirectional PBD6')
    past_length = cache.get_seq_length()
    ids, mask, positions, prompt = interblock_inputs(suffix_ids, pending_ids, past_length,
        state, mask_id, qwen.get_input_embeddings().weight.dtype, attention=attention)
    if past_length + ids.shape[1] > qwen.config.max_position_embeddings:
        raise ValueError('Inter-block context limit exceeded; no targets truncated')
    output = qwen(input_ids=ids, attention_mask=mask, position_ids=positions,
        visual_features=None, image_token_index=None, past_key_values=cache,
        use_cache=True, output_attentions=False, output_hidden_states=False, return_dict=True)
    hidden = output.last_hidden_state[:, -6:]
    # Only real input tokens survive. Six speculative PBD states never become history.
    for layer in cache.layers:
        layer.suffix_keys = layer.suffix_keys[..., :-6, :]
        layer.suffix_values = layer.suffix_values[..., :-6, :]
    if prompt[0] > past_length:
        state.box_spans.append((past_length, prompt[0]))
    state.previous_prompt = prompt
    state.previous_box_position = state.prefix_length + suffix_ids.shape[1]
    if cache.get_seq_length() != prompt[1]:
        raise RuntimeError('Speculative geometry leaked into committed context')
    return hidden


class CausalInterBlockGeometryDecoder(FusedPIVRGeometryDecoder):
    """PTD-style causal inter-block control, adapted to existing LA references.

    The standard parallel path remains unchanged. This class never constructs
    chat continuations: each current reference is exactly its original suffix.
    Earlier generated geometry is additional attention context, not a new turn.
    """

    @torch.no_grad()
    def decode_prepared_image(self, prepared_image, branches, *, requested_wave_size=1,
                              allow_none=True):
        if requested_wave_size != 1:
            raise ValueError('Causal inter-block decoding requires serial prediction')
        device = torch.device(self.device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        if not branches:
            return [], self._stats(0, [], 0, 0.)
        prefix, suffixes, _ = self._prepared_global_suffixes(prepared_image, branches)
        state = InterBlockState(prefix.numel())
        cache = _SharedPrefixCache(prepared_image.key_values)
        qwen = base_causal_lm(self.model.language_model).model
        outputs, lengths, pending = [], [], None
        for branch, suffix in zip(branches, suffixes):
            suffix = suffix.reshape(1, -1).to(device)
            lengths.append(cache.get_seq_length() + (0 if pending is None else pending.numel()) + suffix.numel() + 6)
            hidden = interblock_step(qwen, cache, suffix, state, self.mask_id, pending)
            result = self._geometry_from_hidden(hidden, [branch], allow_none=allow_none)[0]
            result.update(previous_box_blocks=len(outputs), context_positions=lengths[-1])
            pending = torch.tensor([result['block_tokens']], dtype=torch.long, device=device)
            outputs.append(result)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)/1024**2 if device.type == 'cuda' else 0.
        return outputs, self._stats(len(outputs), lengths, peak, time.perf_counter() - started)

    @staticmethod
    def _stats(count, lengths, peak, elapsed):
        return dict(addresses=count, waves=count, requested_wave_size=1, effective_wave_sizes=[1]*count,
            qwen_total_forwards=count, qwen_fused_prompt_geometry_forwards=count,
            round2_execution='fused_one_pass_causal_interblock', inter_block_attention='causal',
            reference_prompts='Unchanged; current branch cannot attend to another reference prompt',
            previous_boxes='Model-predicted tokens only; speculative prediction states discarded',
            added_chat_turns=0, context_positions=lengths, max_context_positions=max(lengths, default=0),
            peak_gpu_mb=peak, end_to_end_seconds=elapsed, global_moonvit_encodes=0,
            local_moonvit_encodes=0, checkpoint_trained_for_causal_interblock=False)
