from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import torch
from torch import nn


def base_causal_lm(language_model: nn.Module) -> nn.Module:
    base = (
        language_model.get_base_model()
        if hasattr(language_model, "get_base_model")
        else language_model
    )
    if not hasattr(base, "model") or not hasattr(base, "lm_head"):
        raise RuntimeError("Cannot resolve Qwen decoder and LM head")
    return base


def qwen_lora_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    values = [
        (name, parameter)
        for (name, parameter) in model.named_parameters()
        if "lora_" in name and "language_model" in name
    ]
    if not values:
        raise RuntimeError("No Qwen LoRA parameters were found")
    return values


def configure_lora_only(model: nn.Module, rank: int) -> dict[str, int]:
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    model.requires_grad_(False)
    model.wrap_llm_lora(r=int(rank), lora_alpha=2 * int(rank), lora_dropout=0.05)
    model.requires_grad_(False)
    values = qwen_lora_parameters(model)
    for _name, parameter in values:
        parameter.requires_grad_(True)
    return {
        "rank": int(rank),
        "tensors": len(values),
        "parameters": sum((parameter.numel() for (_name, parameter) in values)),
    }
