"""Install after dtype conversion, before prefill. No trained weights change."""
from types import MethodType
import torch


def fp32_set_cos_sin_cache(self, seq_len, device, dtype):
    # Upcasting an already-rounded inverse-frequency buffer is not sufficient.
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        exponent = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim
        frequencies = 1.0 / (self.base ** exponent)
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)
        embedding = torch.cat((angles, angles), dim=-1)
        cosine, sine = embedding.cos().to(dtype), embedding.sin().to(dtype)
    self.register_buffer('cos_cached', cosine, persistent=False)
    self.register_buffer('sin_cached', sine, persistent=False)
    self.max_seq_len_cached = seq_len


def install_fp32_rope_cache(model):
    modules = [m for m in model.modules() if type(m).__name__ == 'Qwen2RotaryEmbedding']
    if not modules:
        raise ValueError('No supported legacy Qwen2RotaryEmbedding modules found')
    for module in modules:
        if not all(hasattr(module, n) for n in ('base', 'dim', 'cos_cached', 'sin_cached', 'max_seq_len_cached')):
            raise ValueError('Unsupported Qwen rotary implementation')
    for module in modules:
        module._set_cos_sin_cache = MethodType(fp32_set_cos_sin_cache, module)
        module._set_cos_sin_cache(module.max_seq_len_cached, module.cos_cached.device, module.cos_cached.dtype)
    return len(modules)
