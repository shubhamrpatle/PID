"""Single-forward prompt + PBD6 execution; two-pass is an explicit reference mode."""
from __future__ import annotations

import torch

from .fast_pivr_geometry import FastPIVRGeometryDecoder
from .pixel_pivr_wave import _SharedPrefixCache
from .pointrelay_pbd import base_causal_lm


def fused_prompt_geometry_inputs(suffix_ids, suffix_attention, prefix_length,
                                 geometry_ids, geometry_attention, dtype, *, causal_attn=False):
    """Compose the binary 4D visibility mask expected by LA's mask converter."""
    batch, suffix_width = suffix_ids.shape
    if suffix_width == 0 or geometry_ids.shape != (batch, 6):
        raise ValueError('Expected nonempty address suffixes and exactly six PBD inputs')
    if suffix_attention.shape != suffix_ids.shape or geometry_attention.shape != geometry_ids.shape:
        raise ValueError('Input/mask dimensions differ')
    device = suffix_ids.device
    prefix_attention = torch.ones((batch, prefix_length), dtype=suffix_attention.dtype, device=device)
    full_attention = torch.cat((prefix_attention, suffix_attention, geometry_attention), dim=1)
    positions = full_attention.long().cumsum(-1) - 1
    positions.masked_fill_(full_attention == 0, 1)
    positions[:, -6:] -= 1
    length = suffix_width + 6
    q = torch.arange(length, device=device)[:, None] + prefix_length
    k = torch.arange(prefix_length + length, device=device)[None, :]
    allowed = (k <= q)[None, None].expand(batch, 1, -1, -1)
    allowed = allowed & full_attention[:, None, None, :].bool()
    mask = allowed.to(dtype=dtype)
    if not causal_attn:
        mask[:, :, -6:, -6:] = 1
    # The first geometry input repeats the final prompt token at the same RoPE ID.
    # Native PBD masks its original cached copy, but only for geometry queries.
    mask[:, :, -6:, -7] = 0
    return torch.cat((suffix_ids, geometry_ids), dim=1), mask, positions[:, -length:]


class FusedPIVRGeometryDecoder(FastPIVRGeometryDecoder):
    def __init__(self, *args, round2_execution='fused_one_pass', **kwargs):
        super().__init__(*args, **kwargs)
        if round2_execution not in ('two_pass', 'fused_one_pass'):
            raise ValueError('round2_execution must be two_pass or fused_one_pass')
        if self.visual_context != 'global_only' or self.prefix_cache_mode != 'shared':
            raise ValueError('This first fused experiment supports shared global-only RefWave')
        self.round2_execution = round2_execution

    def _decode_shared_atomic_blocks(self, prefix_ids, suffix_rows, branch_visuals,
                                     prefix_key_values, branches, *, temperature,
                                     top_p, repetition_penalty, allow_none):
        if self.round2_execution == 'two_pass':
            return super()._decode_shared_atomic_blocks(
                prefix_ids, suffix_rows, branch_visuals, prefix_key_values, branches,
                temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty,
                allow_none=allow_none)
        if not branches:
            return []
        if len(suffix_rows) != len(branches) or len(branch_visuals) != len(branches):
            raise ValueError('Inconsistent branch inputs')
        if any(value.numel() for value in branch_visuals):
            raise ValueError('Local visual re-entry is outside this execution-only experiment')
        causal = base_causal_lm(self.model.language_model)
        if causal.model.training or causal.model._attn_implementation != 'sdpa':
            raise ValueError('Fused execution requires evaluation-mode native LA SDPA')
        if causal.model.block_size != 6 or causal.model.text_mask_token_id != self.mask_id:
            raise ValueError('The native model does not use the expected PBD6 contract')
        suffix_ids, suffix_attention = self._left_pad_rows(suffix_rows)
        suffix_ids, suffix_attention = suffix_ids.to(self.device), suffix_attention.to(self.device)
        if suffix_ids.eq(int(self.model.image_token_index)).any():
            raise ValueError('Global visual tokens must already be in the shared prefix')
        geometry_ids, geometry_attention = self._prepare_geometry_inputs(
            branches, input_dtype=suffix_ids.dtype, attention_dtype=suffix_attention.dtype,
            device=self.device, prompt_last_ids=suffix_ids[:, -1])
        ids, mask, positions = fused_prompt_geometry_inputs(
            suffix_ids, suffix_attention, int(prefix_ids.numel()), geometry_ids,
            geometry_attention, causal.model.get_input_embeddings().weight.dtype,
            causal_attn=causal.model.causal_attn)
        # LA converts binary 4D visibility to additive bias, then reapplies its
        # native PBD tail rule. The tests compare the actual per-layer masks.
        # A fresh cache fork isolates this wave and leaves the image cache intact.
        outputs = causal.model(input_ids=ids, visual_features=None, image_token_index=None,
            attention_mask=mask, position_ids=positions,
            past_key_values=_SharedPrefixCache(prefix_key_values), use_cache=True,
            output_attentions=False, output_hidden_states=False, return_dict=True)
        return self._geometry_from_hidden(outputs.last_hidden_state[:, -6:, :], branches,
                                          allow_none=allow_none)

    @torch.no_grad()
    def decode_prepared_image(self, prepared_image, branches, *, requested_wave_size, allow_none=False):
        results, stats = super().decode_prepared_image(prepared_image, branches,
            requested_wave_size=requested_wave_size, allow_none=allow_none)
        stats['round2_execution'] = self.round2_execution
        stats['qwen_fused_prompt_geometry_forwards'] = 0
        if self.round2_execution == 'fused_one_pass':
            stats['qwen_branch_suffix_forwards'] = 0
            stats['qwen_geometry_forwards'] = 0
            stats['qwen_fused_prompt_geometry_forwards'] = stats['waves']
            stats['qwen_total_forwards'] = stats['waves']
        return results, stats
