from contextlib import nullcontext
import copy
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from refwave import vision_attention
from refwave.vision_attention import (
    VISION_ATTENTION,
    enable_fused_vision_attention,
    packed_vision_attention,
)
from test_restart import tiny_trainer


def masked_reference(q, k, v, q_bounds, k_bounds):
    mask = torch.zeros(1, q.shape[0], k.shape[0], dtype=torch.bool, device=q.device)
    for i in range(len(q_bounds) - 1):
        mask[:, q_bounds[i] : q_bounds[i + 1], k_bounds[i] : k_bounds[i + 1]] = True
    with sdpa_kernel(SDPBackend.MATH):
        output = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), mask, dropout_p=0.0
        )
    return output.transpose(0, 1).reshape(q.shape[0], -1)


@pytest.mark.parametrize("q_lengths,k_lengths", [([9], [9]), ([5, 7], [5, 7]), ([3, 4], [5, 2])])
def test_outputs_and_gradients_match_masked_attention(q_lengths, k_lengths):
    torch.manual_seed(11)
    q_bounds = torch.tensor([0, *q_lengths], dtype=torch.int32).cumsum(0)
    k_bounds = torch.tensor([0, *k_lengths], dtype=torch.int32).cumsum(0)
    q = torch.randn(sum(q_lengths), 4, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(sum(k_lengths), 4, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    actual = packed_vision_attention(q, k, v, q_bounds, k_bounds)
    expected = masked_reference(q, k, v, q_bounds, k_bounds)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    upstream = torch.randn_like(actual)
    left = torch.autograd.grad((actual * upstream).sum(), (q, k, v))
    right = torch.autograd.grad((expected * upstream).sum(), (q, k, v))
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


def test_images_are_isolated_and_kernel_receives_four_dimensions(monkeypatch):
    torch.manual_seed(12)
    calls = []
    original = F.scaled_dot_product_attention

    def attention(q, k, v, **kwargs):
        calls.append(q.shape)
        assert q.ndim == k.ndim == v.ndim == 4
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert "attn_mask" not in kwargs
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(vision_attention.F, "scaled_dot_product_attention", attention)
    q = torch.randn(9, 2, 8, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    bounds = torch.tensor([0, 4, 9])
    actual = packed_vision_attention(q, k, v, bounds, bounds)
    assert calls == [torch.Size([1, 2, 4, 8]), torch.Size([1, 2, 5, 8])]
    changed_k, changed_v = k.detach().clone(), v.detach().clone()
    changed_k[4:] *= 100
    changed_v[4:] += 100
    changed = packed_vision_attention(q, changed_k, changed_v, bounds, bounds)
    torch.testing.assert_close(actual[:4], changed[:4])
    grads = torch.autograd.grad(actual[:4].square().sum(), (q, k, v))
    assert all(torch.count_nonzero(grad[4:]) == 0 for grad in grads)


def test_single_image_avoids_boundary_copy_and_cuda_excludes_math(monkeypatch):
    class Bounds:
        def numel(self):
            return 2

        def tolist(self):
            raise AssertionError("Single-image path must not copy boundaries to CPU")

    q = torch.randn(5, 2, 8)
    torch.testing.assert_close(
        packed_vision_attention(q, q, q, Bounds(), Bounds()),
        packed_vision_attention(q, q, q),
    )
    requests = []
    monkeypatch.setattr(vision_attention, "sdpa_kernel", lambda backends: requests.append(backends) or nullcontext())
    with vision_attention._backend_context(torch.device("cpu")):
        pass
    assert requests == []
    with vision_attention._backend_context(torch.device("cuda")):
        pass
    assert requests == [[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]]


def test_mismatched_image_counts_fail():
    q = torch.randn(5, 2, 8)
    with pytest.raises(ValueError, match="image counts"):
        packed_vision_attention(q, q, q, torch.tensor([0, 2, 5]), torch.tensor([0, 5]))


@pytest.mark.skipif(not os.getenv("LA_TEST_MODEL"), reason="Optional actual LA MoonViT source")
@pytest.mark.parametrize("checkpointed", [False, True])
def test_actual_moonvit_preserves_weights_gradients_and_other_instances(checkpointed):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    torch.manual_seed(14)
    encoder_type = get_class_from_dynamic_module("modeling_vit.MoonVitEncoder", os.environ["LA_TEST_MODEL"])
    block_cfg = dict(num_heads=4, hidden_dim=32, mlp_dim=64, attn_implementation="sdpa")
    reference = encoder_type(hidden_dim=32, num_layers=3, block_cfg=block_cfg).train()
    optimized = copy.deepcopy(reference)
    holder = SimpleNamespace(vision_model=SimpleNamespace(encoder=optimized), language_model=object())
    language_model = holder.language_model
    parameter_ids = [id(p) for p in optimized.parameters()]
    assert enable_fused_vision_attention(holder) == VISION_ATTENTION
    enable_fused_vision_attention(holder)
    assert [id(p) for p in optimized.parameters()] == parameter_ids
    assert holder.language_model is language_model
    assert all(b.attn_implementation == "sdpa" for b in reference.blocks)
    assert all(b.attn_implementation == VISION_ATTENTION for b in optimized.blocks)
    grids = torch.tensor([[4, 4], [2, 4]], dtype=torch.int32)
    original_input = torch.randn(24, 32, requires_grad=True)
    fused_input = original_input.detach().clone().requires_grad_()
    if checkpointed:
        expected = checkpoint(reference, original_input, grids, use_reentrant=False)
        actual = checkpoint(optimized, fused_input, grids, use_reentrant=False)
    else:
        expected, actual = reference(original_input, grids), optimized(fused_input, grids)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()
    torch.testing.assert_close(fused_input.grad, original_input.grad, rtol=3e-5, atol=5e-6)
    for (name, left), (other, right) in zip(reference.named_parameters(), optimized.named_parameters()):
        assert name == other and left.grad is not None and right.grad is not None
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(left.grad, right.grad, rtol=3e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a free CUDA GPU")
@pytest.mark.parametrize("head_dim", [72, 128])
def test_cuda_bf16_fused_output_and_backward(head_dim):
    torch.manual_seed(15)
    q = torch.randn(126, 4, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k, v = torch.randn_like(q, requires_grad=True), torch.randn_like(q, requires_grad=True)
    bounds = torch.tensor([0, 49, 126], dtype=torch.int32, device="cuda")
    actual = packed_vision_attention(q, k, v, bounds, bounds)
    expected = masked_reference(q, k, v, [0, 49, 126], [0, 49, 126])
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-3)
    upstream = torch.randn_like(actual)
    left = torch.autograd.grad((actual * upstream).sum(), (q, k, v))
    right = torch.autograd.grad((expected * upstream).sum(), (q, k, v))
    for a, b in zip(left, right):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=3e-2, atol=1e-2)


@pytest.mark.parametrize("tuning", ["lora", "full"])
def test_trainer_enables_fused_vision_only_for_full(tiny_trainer, monkeypatch, tuning):
    from refwave import train

    args, state = tiny_trainer
    args.tuning = tuning
    enabled = []

    def enable(model):
        assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
        enabled.append(model)
        return VISION_ATTENTION

    monkeypatch.setattr(train, "enable_fused_vision_attention", enable)
    train.run(args)
    assert len(enabled) == (1 if tuning == "full" else 0)
    config = json.loads((args.output / "config.json").read_text())
    if tuning == "full":
        assert config["trainable"]["vision_attention"] == VISION_ATTENTION
    else:
        assert "vision_attention" not in config["trainable"]
    checkpoint = torch.load(args.output / "last.pt", weights_only=False)
    assert "param_groups" in checkpoint["optimizer"]
    assert "shards" not in checkpoint["optimizer"]
    for item in checkpoint["optimizer"]["state"].values():
        assert item["exp_avg"].dtype == item["exp_avg_sq"].dtype == torch.float32
    assert state.updates == 2
    train.run(args)
    assert state.updates == 2
