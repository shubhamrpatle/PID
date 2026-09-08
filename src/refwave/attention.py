from __future__ import annotations


def locateanything_attention_dispatch(vision_attention: str) -> dict[str, str]:
    """Keep Qwen on SDPA while selecting MoonViT's tested implementation.

    Transformers recursively applies a scalar ``attn_implementation`` to every
    sub-config. LocateAnything needs a dictionary here; otherwise a final
    ``"sdpa"`` argument silently overrides the earlier MoonViT setting.
    """
    if vision_attention not in {"sdpa", "flash_attention_2"}:
        raise ValueError(f"Unsupported MoonViT attention: {vision_attention!r}")
    return {"": "sdpa", "text_config": "sdpa", "vision_config": vision_attention}
