"""One-pass geometry regressions; these tests are not part of the user launchers."""

import inspect
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from refwave.decoder import AddressedCrop, GeometryDecoder, fused_prompt_geometry_inputs
from refwave.shared_prefix import legacy_key_values


def test_one_pass_is_default():
    assert inspect.signature(GeometryDecoder).parameters["round2_execution"].default == "fused_one_pass"


@pytest.mark.parametrize("method,execution,folder", [
    ("refwave", None, "evaluation_fused_one_pass"),
    ("refwave", "two_pass", "evaluation_two_pass"),
    ("standard", None, "evaluation"),
    ("class_isolated", None, "evaluation"),
])
def test_launcher_passes_mode_and_separates_outputs(tmp_path, method, execution, folder):
    env = {k: v for k, v in os.environ.items()
           if k not in ("ROUND2_EXECUTION", "EVAL_OUTPUT", "SLURM_JOB_NUM_NODES")}
    env.update(PYTHON_BIN="/bin/echo", WORK_ROOT=str(tmp_path), METHOD=method)
    if execution:
        env["ROUND2_EXECUTION"] = execution
    checkpoint = tmp_path / "run" / "stage2" / "best.pt"
    result = subprocess.run(["bash", str(Path(__file__).resolve().parents[1] / "evaluate.sh"),
                             "4", str(checkpoint)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"--round2-execution {execution or 'fused_one_pass'}" in result.stdout
    assert f"--output {checkpoint.parent / folder} --method {method}" in result.stdout


def test_mask_preserves_causality_padding_and_pbd_visibility():
    suffixes = torch.tensor([[0, 11], [12, 13]])
    attention = torch.tensor([[0, 1], [1, 1]])
    geometry = torch.tensor([[11, 99, 99, 99, 99, 99], [13, 99, 99, 99, 99, 99]])
    ids, mask, positions = fused_prompt_geometry_inputs(
        suffixes, attention, 3, geometry, torch.ones_like(geometry), torch.float32,
    )
    assert ids.shape == (2, 8) and mask.shape == (2, 1, 8, 11)
    assert not mask[:, :, :2, -6:].any()
    assert not mask[:, :, -6:, -7].any()
    assert mask[:, :, -6:, -6:].all()
    assert not mask[0, :, :, 3].any()
    assert torch.equal(positions[:, 1], positions[:, 2])


@pytest.fixture
def native_decoder():
    path = os.getenv("LA_TEST_MODEL")
    if not path:
        pytest.skip("Set LA_TEST_MODEL to exercise actual released Qwen code on CPU")
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    torch.set_num_threads(2)
    config_class = get_class_from_dynamic_module("configuration_qwen2.Qwen2Config", path)
    model_class = get_class_from_dynamic_module("modeling_qwen2.Qwen2Model", path)
    config = config_class(vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, use_cache=True, block_size=6, text_mask_token_id=99)
    config._attn_implementation = "sdpa"
    torch.manual_seed(1729)
    qwen = model_class(config).eval()
    decoder = object.__new__(GeometryDecoder)
    decoder.model = SimpleNamespace(language_model=SimpleNamespace(
        model=qwen, lm_head=torch.nn.Linear(32, 128, bias=False)), image_token_index=2)
    decoder.tokenizer = SimpleNamespace(pad_token_id=0)
    decoder.device = torch.device("cpu")
    decoder.geometry_prefix_mode = "box_only"
    decoder.mask_id, decoder.n_future_tokens = 99, 6
    decoder._results_from_logits = lambda logits, *args, **kw: logits.detach().clone()
    prefix = torch.tensor([1, 2, 3])
    with torch.no_grad():
        cache = legacy_key_values(qwen(input_ids=prefix[None], visual_features=torch.randn(1, 32),
            image_token_index=2, attention_mask=torch.ones(1, 3), position_ids=torch.arange(3)[None],
            use_cache=True).past_key_values)
    return decoder, qwen, prefix, cache


def decode(fixture, suffixes, mode):
    decoder, _, prefix, cache = fixture
    decoder.round2_execution = mode
    branches = [AddressedCrop(str(i), "ship" if i % 2 else "small vehicle", None, "query")
                for i in range(len(suffixes))]
    with torch.no_grad():
        return decoder._decode_shared_atomic_blocks(prefix, suffixes,
            [torch.empty(0, 32) for _ in branches], cache, branches, allow_none=True)


@pytest.mark.parametrize("count", [1, 18, 200, 203])
def test_logits_waves_forward_counts_and_cache_reuse(native_decoder, count):
    decoder, qwen, _, cache = native_decoder
    snapshot = [(k.clone(), v.clone()) for k, v in cache]
    suffixes = [torch.arange(11, 12 + i % 7) for i in range(count)]
    calls = []
    handle = qwen.register_forward_pre_hook(lambda *args: calls.append(1))
    try:
        reference = torch.cat([decode(native_decoder, suffixes[i:i+200], "two_pass")
                               for i in range(0, count, 200)])
        assert len(calls) == 2 * ((count + 199) // 200)
        calls.clear()
        fused = torch.cat([decode(native_decoder, suffixes[i:i+200], "fused_one_pass")
                          for i in range(0, count, 200)])
        assert len(calls) == (count + 199) // 200
    finally:
        handle.remove()
    torch.testing.assert_close(reference, fused, atol=2e-6, rtol=2e-6)
    for before, after in zip(snapshot, cache):
        assert all(torch.equal(x, y) for x, y in zip(before, after))


def test_branch_isolation_empty_and_repeated_decode(native_decoder):
    a = decode(native_decoder, [torch.tensor([11, 12]), torch.tensor([13, 14])], "fused_one_pass")
    b = decode(native_decoder, [torch.tensor([11, 12]), torch.tensor([50, 51])], "fused_one_pass")
    assert torch.equal(a[0], b[0]) and not torch.equal(a[1], b[1])
    c = decode(native_decoder, [torch.tensor([11, 12]), torch.tensor([13, 14])], "fused_one_pass")
    assert torch.equal(a, c)
    assert decode(native_decoder, [], "fused_one_pass") == []
