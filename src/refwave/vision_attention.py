"""Memory-efficient MoonViT attention without changing Qwen or model weights."""

from contextlib import nullcontext
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


VISION_ATTENTION = "refwave_fused_per_image_sdpa"


def _backend_context(device):
    if device.type == "cuda":
        # Do not silently recreate the high-memory math attention on CUDA.
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    return nullcontext()


def _boundaries(cumulative, length):
    # RefWave encodes one image at a time; avoid a GPU synchronization per block.
    if cumulative is None or cumulative.numel() == 2:
        return [0, length]
    return cumulative.tolist()


def packed_vision_attention(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
    """Attend within each packed image, using [batch, heads, tokens, dim] inputs."""
    q_bounds = _boundaries(q_cu_seqlens, q.shape[0])
    k_bounds = _boundaries(k_cu_seqlens, k.shape[0])
    if len(q_bounds) != len(k_bounds):
        raise ValueError("MoonViT query and key image counts differ")
    outputs = []
    with _backend_context(q.device):
        for index in range(len(q_bounds) - 1):
            start, stop = q_bounds[index : index + 2]
            k_start, k_stop = k_bounds[index : index + 2]
            query = q[start:stop].transpose(0, 1).unsqueeze(0).contiguous()
            key = k[k_start:k_stop].transpose(0, 1).unsqueeze(0).contiguous()
            value = v[k_start:k_stop].transpose(0, 1).unsqueeze(0).contiguous()
            result = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
            outputs.append(result.squeeze(0).transpose(0, 1).reshape(stop - start, -1))
    return torch.cat(outputs, dim=0)


def enable_fused_vision_attention(model):
    """Select a separate attention entry only on this model's MoonViT blocks."""
    for block in model.vision_model.encoder.blocks:
        module = sys.modules[type(block).__module__]
        module.VL_VISION_ATTENTION_FUNCTIONS[VISION_ATTENTION] = packed_vision_attention
        block.attn_implementation = VISION_ATTENTION
    return VISION_ATTENTION
