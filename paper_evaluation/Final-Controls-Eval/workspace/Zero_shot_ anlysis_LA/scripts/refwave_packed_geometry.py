"""Opt-in, sequence-packed Round-2 geometry with unchanged branch semantics."""
from __future__ import annotations

import torch

from zero_shot_analysis.pixel_pivr_wave import _SharedPrefixCache
from zero_shot_analysis.pointrelay_pbd import base_causal_lm


def packed_inputs(suffix_rows, geometry_ids, prefix_length, dtype, *, causal_attn=False):
    """Pack unpadded prompts/PBD6 blocks; reset RoPE positions for every branch."""
    count = len(suffix_rows)
    if not count or geometry_ids.shape != (count, 6):
        raise ValueError('Expected nonempty prompts and exactly six geometry inputs per branch')
    if any(row.ndim != 1 or not row.numel() for row in suffix_rows):
        raise ValueError('Prompts must be nonempty unpadded token rows')
    device = geometry_ids.device
    lengths = [int(row.numel()) + 6 for row in suffix_rows]
    total = sum(lengths)
    mask = torch.zeros((1, 1, total, prefix_length + total), device=device, dtype=dtype)
    mask[:, :, :, :prefix_length] = 1
    rows, positions, indices = [], [], []
    offset = 0
    for row, geometry, length in zip(suffix_rows, geometry_ids, lengths):
        row = row.to(device=device, dtype=geometry_ids.dtype)
        prompt_length = length - 6
        if not torch.equal(row[-1], geometry[0]):
            raise ValueError('PBD6 must repeat the final prompt token')
        rows.append(torch.cat((row, geometry)))
        pos = torch.arange(prefix_length, prefix_length + length, device=device)
        pos[-6:] -= 1
        positions.append(pos)
        local = torch.ones((length, length), device=device, dtype=dtype).tril_()
        if not causal_attn:
            local[-6:, -6:] = 1
        local[-6:, -7] = 0
        mask[0, 0, offset:offset+length, prefix_length+offset:prefix_length+offset+length] = local
        indices.append(torch.arange(offset+prompt_length, offset+length, device=device))
        offset += length
    return (torch.cat(rows)[None], mask, torch.cat(positions)[None], torch.stack(indices))


def packed_hidden(decoder, prefix_ids, suffix_rows, branch_visuals, prefix_key_values, branches):
    """One native Qwen forward, batch size one, all independent segments packed."""
    if len(branches) != len(suffix_rows) or len(branches) != len(branch_visuals):
        raise ValueError('Inconsistent packed branch inputs')
    if not branches:
        raise ValueError('Empty addresses should bypass the decoder')
    if decoder.round2_execution != 'fused_one_pass':
        raise ValueError('Packed mode requires fused one-pass PBD')
    if any(value.numel() for value in branch_visuals):
        raise ValueError('This diagnostic preserves global-only RefWave')
    causal = base_causal_lm(decoder.model.language_model)
    if causal.model.training or causal.model._attn_implementation != 'sdpa':
        raise ValueError('Packed reference requires evaluation-mode native LA SDPA')
    if causal.model.block_size != 6 or causal.model.text_mask_token_id != decoder.mask_id:
        raise ValueError('Unexpected native geometry contract')
    device = torch.device(decoder.device)
    suffix_rows = [row.to(device) for row in suffix_rows]
    if any(row.eq(int(decoder.model.image_token_index)).any() for row in suffix_rows):
        raise ValueError('Global visual tokens must already be in the shared prefix')
    geometry, attention = decoder._prepare_geometry_inputs(branches,
        input_dtype=suffix_rows[0].dtype, attention_dtype=torch.long, device=device,
        prompt_last_ids=torch.stack([row[-1] for row in suffix_rows]))
    if not attention.bool().all():
        raise ValueError('Unexpected padded geometry block')
    ids, mask, positions, indices = packed_inputs(suffix_rows, geometry,
        int(prefix_ids.numel()), causal.model.get_input_embeddings().weight.dtype,
        causal_attn=causal.model.causal_attn)
    outputs = causal.model(input_ids=ids, visual_features=None, image_token_index=None,
        attention_mask=mask, position_ids=positions,
        past_key_values=_SharedPrefixCache(prefix_key_values), use_cache=True,
        output_attentions=False, output_hidden_states=False, return_dict=True)
    return outputs.last_hidden_state[0, indices, :]


def packed_atomic(decoder, prefix_ids, suffix_rows, branch_visuals, prefix_key_values,
                  branches, *, temperature, top_p, repetition_penalty, allow_none):
    del temperature, top_p, repetition_penalty
    if not branches:
        return []
    hidden = packed_hidden(decoder, prefix_ids, suffix_rows, branch_visuals, prefix_key_values, branches)
    return decoder._geometry_from_hidden(hidden, branches, allow_none=allow_none)
