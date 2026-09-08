"""Optional CPU integration with the released LA Qwen code, not a GPU precheck."""

import os
from pathlib import Path
import sys
import pytest
import torch
from refwave.model import branch_ce, pack_batch
from refwave.episode_model import _install_explicit_mask_hook


@pytest.mark.skipif(
    not os.getenv("LA_TEST_MODEL"), reason="Set LA_TEST_MODEL to the released LA directory"
)
def test_actual_qwen_padded_episode_batch_matches_separate_gradients():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    path = Path(os.environ["LA_TEST_MODEL"])
    config_cls = get_class_from_dynamic_module("configuration_qwen2.Qwen2Config", str(path))
    model_cls = get_class_from_dynamic_module("modeling_qwen2.Qwen2ForCausalLM", str(path))
    config = config_cls(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=128,
        attention_dropout=0,
        use_cache=False,
        tie_word_embeddings=False,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(17)
    model = model_cls(config).float().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    _install_explicit_mask_hook(model.model)

    def branch(name, token, shorter=False):
        ids = torch.tensor([1, 2, 3, token, 11, 90, 91, 92, 93, 94, 95, 0])
        labels = torch.tensor([-100] * 6 + [31, 32, 33, 34, 35, 36])
        pos = torch.tensor([0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8])
        return dict(branch_id=name, input_ids=ids, labels=labels, position_ids=pos)

    waves = [[branch("a", 10), branch("b", 20)], [branch("c", 30)]]

    def objective(waves):
        (ids, pos, mask, items) = pack_batch(waves, 2, torch.device("cpu"), torch.float32)
        model.model._caw_pivr_explicit_attention_mask = mask
        try:
            hidden = model.model(input_ids=ids, position_ids=pos, use_cache=False).last_hidden_state
        finally:
            model.model._caw_pivr_explicit_attention_mask = None
        total = hidden.new_zeros(())
        for i, item in enumerate(items):
            for span in item["branch_ranges"]:
                total = (
                    total
                    + branch_ce(hidden[i : i + 1], item["labels"], span, model.lm_head.weight)[0]
                    / 3
                )
        return total

    try:
        full = objective(waves)
        full.backward()
        observed = {n: p.grad.clone() for (n, p) in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        expected_loss = 0
        for wave in waves:
            value = objective([wave])
            expected_loss += float(value.detach())
            value.backward()
        assert float(full.detach()) == pytest.approx(expected_loss, abs=1e-05)
        for name, p in model.named_parameters():
            if p.grad is not None:
                torch.testing.assert_close(observed[name], p.grad, atol=2e-06, rtol=0.0001)
    finally:
        module = sys.modules[model.model.__class__.__module__]
        module.create_block_diff_mask_by_pe_4d = module._caw_pivr_native_mask_factory
        del module._caw_pivr_explicit_mask_owner
