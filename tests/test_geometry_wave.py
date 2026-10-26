import os
from types import SimpleNamespace
import pytest
import torch
import sys

from refwave.decoder import GeometryDecoder, AddressedCrop
from refwave.shared_prefix import legacy_key_values


@pytest.mark.parametrize("mode", ["box_only", "ref"])
@pytest.mark.parametrize("absent", [False, True])
def test_decoded_answer_matches_geometry_prefix_contract(mode, absent):
    decoder = object.__new__(GeometryDecoder)
    decoder.geometry_prefix_mode = mode
    decoder.model = SimpleNamespace(
        token_ids=dict(
            box_start_token_id=1,
            box_end_token_id=2,
            none_token_id=3,
            coord_start_token_id=10,
            coord_end_token_id=20,
        )
    )

    def decode(tokens, **kwargs):
        special = {1: "<box>", 2: "</box>", 3: "None"}
        return "".join(special.get(t, f"<{t - 10}>") for t in tokens.tolist())

    decoder.tokenizer = SimpleNamespace(decode=decode)
    logits = torch.full((1, 6, 32), -100.0)
    for lane, value in enumerate((1, 2, 6, 7), 1):
        logits[0, lane, 10 + value] = 100
    if absent:
        logits[0, 1, 3] = 200
    result = decoder._results_from_logits(
        logits, [AddressedCrop("a", "ship", None, "query")], allow_none=True
    )[0]
    block = "<box>None</box>" if absent else "<box><1><2><6><7></box>"
    assert result["answer"] == (block if mode == "box_only" else "<ref>ship</ref>" + block)
    assert result["label"] == "ship"
    assert len(result["block_tokens"]) == (3 if absent else 6)


@pytest.mark.skipif(not os.getenv("LA_TEST_MODEL"), reason="Optional actual LA Qwen code")
def test_box_only_wave_logits_equal_separate_addresses():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    path = os.environ["LA_TEST_MODEL"]
    config_class = get_class_from_dynamic_module("configuration_qwen2.Qwen2Config", path)
    model_class = get_class_from_dynamic_module("modeling_qwen2.Qwen2Model", path)
    config = config_class(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        use_cache=True,
        block_size=6,
        text_mask_token_id=99,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(31)
    qwen = model_class(config).eval()
    head = torch.nn.Linear(32, 128, bias=False)
    decoder = object.__new__(GeometryDecoder)
    decoder.model = SimpleNamespace(
        language_model=SimpleNamespace(model=qwen, lm_head=head), image_token_index=2
    )
    decoder.tokenizer = SimpleNamespace(pad_token_id=0)
    decoder.device = torch.device("cpu")
    decoder.geometry_prefix_mode = "box_only"
    decoder.round2_execution = "fused_one_pass"
    decoder.mask_id, decoder.n_future_tokens = 99, 6
    decoder._results_from_logits = lambda logits, branches, **kwargs: [
        r.clone() for r in logits
    ]
    prefix = torch.tensor([1, 2, 3])
    with torch.inference_mode():
        cache = legacy_key_values(
            qwen(
                input_ids=prefix[None],
                attention_mask=torch.ones(1, 3),
                position_ids=torch.arange(3)[None],
                use_cache=True,
            ).past_key_values
        )
        suffixes = [torch.tensor([11]), torch.tensor([12, 13]), torch.tensor([14, 15, 16])]
        branches = [AddressedCrop(str(i), "ship", None, "q") for i in range(3)]
        local = torch.empty(0, 32)
        batch = decoder._decode_shared_atomic_blocks(
            prefix, suffixes, [local] * 3, cache, branches, allow_none=True
        )
        for i in range(3):
            single = decoder._decode_shared_atomic_blocks(
                prefix,
                suffixes[i : i + 1],
                [local],
                cache,
                branches[i : i + 1],
                allow_none=True,
            )
            torch.testing.assert_close(batch[i], single[0], atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not os.getenv("LA_TEST_MODEL"), reason="Optional actual LA Qwen code")
def test_full_episode_backpropagates_to_vision_projector_qwen_and_head():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from refwave.model import ClusterEpisodeModel, configure_full

    path = os.environ["LA_TEST_MODEL"]
    config_class = get_class_from_dynamic_module("configuration_qwen2.Qwen2Config", path)
    model_class = get_class_from_dynamic_module("modeling_qwen2.Qwen2ForCausalLM", path)
    config = config_class(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        use_cache=False,
        block_size=6,
        text_mask_token_id=99,
    )
    config._attn_implementation = "sdpa"
    model = torch.nn.Module()
    model.language_model = model_class(config)
    model.vision_model = torch.nn.Linear(5, 32)
    model.mlp1 = torch.nn.Linear(32, 32)
    model.extract_feature = lambda pixels, grids: [model.vision_model(pixels)]
    model.image_token_index = 2
    configure_full(model)
    wrapper = ClusterEpisodeModel(model, objective_mode="two_route_equal_mean", full=True)

    def branch(name, route):
        return dict(
            branch_id=name,
            route=route,
            polarity="positive",
            input_ids=torch.tensor([1, 2, 3, 11, 11, 90, 91, 92, 93, 94, 95, 0]),
            labels=torch.tensor([-100] * 6 + [31, 32, 33, 34, 35, 36]),
            position_ids=torch.tensor([0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8]),
        )

    a, b = branch("a", "round1"), branch("b", "round2")
    sample = dict(
        task="detection",
        pixel_values=torch.randn(1, 5),
        image_grid_hws=[[2, 2]],
        direct_waves=[],
        round1_waves=[[a]],
        round2_waves=[[b]],
        episode_waves=[[a], [b]],
    )
    module = sys.modules[wrapper.causal.model.__class__.__module__]
    try:
        wrapper.train()
        rows = wrapper.batch([sample], token_budget=1)
        assert rows[0]["waves"] == 2
        assert wrapper.vision_encode_calls == 1
        for component in (
            model.vision_model,
            model.mlp1,
            model.language_model.model,
            model.language_model.lm_head,
        ):
            assert (
                sum(
                    float(p.grad.abs().sum())
                    for p in component.parameters()
                    if p.grad is not None
                )
                > 0
            )
    finally:
        module.create_block_diff_mask_by_pe_4d = module._caw_pivr_native_mask_factory
        del module._caw_pivr_explicit_mask_owner
