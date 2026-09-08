from types import SimpleNamespace
from pathlib import Path
import os
import pytest
import torch
from refwave.class_wave import ClassPointBranch, _BranchRNG
from refwave.hybrid_class_wave import (
    CachedHybridClassWaveDecoder,
    _HybridState,
    _pack_suffix_cache,
    _unpack_suffix_cache,
)
from refwave.shared_prefix import legacy_key_values


def test_cache_repacking_retains_only_own_accepted_tokens():
    prefix = ((torch.ones(1, 2, 5, 3), torch.ones(1, 2, 5, 3) * 2),)
    states = [_HybridState(ClassPointBranch(str(i), "q", i), [1], [1], [], []) for i in range(2)]
    for state, length, value in zip(states, (2, 4), (3.0, 7.0)):
        state.suffix_kv = (
            (torch.full((1, 2, length, 3), value), torch.full((1, 2, length, 3), value + 1)),
        )
    (cache, mask, lengths) = _pack_suffix_cache(prefix, states)
    assert cache.layers[0].prefix_keys is prefix[0][0]
    assert mask.tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]
    pending = torch.full((2, 2, 8, 3), 9.0)
    cache.layers[0].update(pending, pending)
    _unpack_suffix_cache(cache, states, lengths, 4, [1, 2], 2)
    assert states[0].suffix_kv[0][0].shape[-2] == 3
    assert states[1].suffix_kv[0][0].shape[-2] == 6
    assert states[0].suffix_kv[0][0][0, 0, :, 0].tolist() == [3.0, 3.0, 9.0]
    assert states[1].suffix_kv[0][0][0, 0, :, 0].tolist() == [7.0, 7.0, 7.0, 7.0, 9.0, 9.0]
    states[0].suffix_kv[0][0].zero_()
    assert torch.all(prefix[0][0] == 1)
    assert torch.all(states[1].suffix_kv[0][0] != 0)


class FakeQwen:

    def __init__(self, prefix):
        self.prefix = prefix
        self.calls = []

    def __call__(self, **kwargs):
        (ids, mask, pos) = (kwargs[k] for k in ("input_ids", "attention_mask", "position_ids"))
        cache = kwargs["past_key_values"]
        assert cache.layers[0].prefix_keys is self.prefix[0][0]
        assert kwargs["visual_features"] is None
        self.calls.append(ids.tolist())
        expected = mask.cumsum(-1) - 1
        expected.masked_fill_(mask == 0, 1)
        if ids[0, -1] == 99:
            expected[:, -6:] -= 1
            assert torch.all(ids[:, -1] == 99)
        assert torch.equal(pos, expected[:, -ids.shape[1] :])
        values = ids.float().unsqueeze(1).unsqueeze(-1)
        cache.layers[0].update(values, values)
        return SimpleNamespace(last_hidden_state=torch.zeros((*ids.shape, 1)))


def make_fake_decoder():
    decoder = object.__new__(CachedHybridClassWaveDecoder)
    prefix = ((torch.ones(1, 1, 2, 1), torch.ones(1, 1, 2, 1)),)
    qwen = FakeQwen(prefix)
    decoder.model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=qwen, lm_head=SimpleNamespace(weight=torch.zeros(100, 1))
        )
    )
    decoder.device = torch.device("cpu")
    (decoder.pad_id, decoder.mask_id, decoder.max_wave_size) = (0, 99, 18)
    decoder.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: ",".join(map(str, ids)))

    def sample_mtp(logits, history, rng, kwargs):
        values = history[0].tolist()
        assert 0 not in values
        if 11 in values:
            return ("error_box", [20, 30]) if 20 not in values else ("im_end", [2])
        if 12 in values:
            return ("coord_box", [20, 32, 33, 21]) if 20 not in values else ("im_end", [2])
        return ("ref_object", [77])

    def sample_ar(logits, history, rng, kwargs):
        return ("coord_ar", [31]) if history[0, -1] == 30 else ("box_end_ar", [21])

    (decoder._sample_row, decoder._sample_ar_row) = (sample_mtp, sample_ar)
    return (decoder, prefix, qwen)


def run_fake(decoder, prefix, suffixes, budget=30):
    branches = [ClassPointBranch(str(i), "q", i) for i in range(len(suffixes))]
    return decoder._decode_wave(
        [1, 1],
        prefix,
        branches,
        [[1, 1] + row for row in suffixes],
        suffixes,
        max_new_tokens=budget,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        generation_mode="hybrid",
    )


def test_hybrid_transition_and_wave_width_equivalence():
    (decoder, prefix, qwen) = make_fake_decoder()
    (wave, stats) = run_fake(decoder, prefix, [[11], [12, 13]])
    independent = [run_fake(decoder, prefix, [row])[0][0] for row in [[11], [12, 13]]]
    assert [r["answer"] for r in wave] == [r["answer"] for r in independent]
    assert wave[0]["answer"] == "20,30,31,21,2"
    assert wave[1]["answer"] == "20,32,33,21,2"
    assert stats["ar_forward_calls"] == 2
    assert stats["hybrid_ar_transitions"] == 1
    assert stats["full_prompt_replays"] == 0
    assert all((row["fallback_reason"] is None for row in wave))
    assert torch.all(prefix[0][0] == 1)


def test_token_budget_is_branch_local_and_does_not_fabricate_eos():
    (decoder, prefix, _) = make_fake_decoder()
    (result, _) = run_fake(decoder, prefix, [[11], [12, 13], [14]], budget=4)
    assert result[2]["answer"] == "77,77,77,77"
    assert result[2]["forward_steps"] == 4
    assert all((row["fallback_reason"] == "max_new_tokens" for row in result))
    assert decoder._close_truncated_answer("<box><1>", "max_new_tokens") == ("<box><1>", False)


@pytest.mark.parametrize(
    "token,kind",
    [(21, "box_end_ar"), (30, "coord_ar"), (40, "coord_ar"), (2, "im_end"), (88, "im_end")],
)
def test_ar_sampler_preserves_native_token_and_stop_rule(token, kind):
    (decoder, _, _) = make_fake_decoder()
    decoder.model.token_ids = dict(
        box_end_token_id=21, coord_start_token_id=30, coord_end_token_id=39, none_token_id=40
    )
    decoder._sample_tokens = lambda *args, **kwargs: (None, None, torch.tensor([[token]]), None)
    result = CachedHybridClassWaveDecoder._sample_ar_row(
        decoder, torch.zeros(1, 1, 100), torch.ones(1, 2), _BranchRNG(1, decoder.device), {}
    )
    assert result == (kind, [token])


def test_actual_la_qwen_hybrid_cache_matches_full_history_logits():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    if not os.getenv("LA_TEST_MODEL"):
        pytest.skip("Optional integration test requires LA_TEST_MODEL")
    checkpoint = Path(os.environ["LA_TEST_MODEL"])
    config_type = get_class_from_dynamic_module("configuration_qwen2.Qwen2Config", str(checkpoint))
    model_type = get_class_from_dynamic_module("modeling_qwen2.Qwen2Model", str(checkpoint))
    torch.manual_seed(19)
    config = config_type(
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
    qwen = model_type(config).eval()
    head = torch.nn.Linear(32, 128, bias=False)
    (decoder, _, _) = make_fake_decoder()
    decoder.model.language_model = SimpleNamespace(model=qwen, lm_head=head)
    checked = []

    def check_logits(logits, history, mtp):
        length = history.shape[1]
        native = qwen(
            input_ids=history,
            attention_mask=torch.ones_like(history),
            position_ids=torch.arange(length).unsqueeze(0),
            use_cache=True,
            return_dict=True,
        )
        if mtp:
            block = torch.tensor([[int(history[0, -1]), 99, 99, 99, 99, 99]])
            native = qwen(
                input_ids=block,
                attention_mask=torch.ones(1, length + 6, dtype=torch.long),
                position_ids=torch.arange(length - 1, length + 5).unsqueeze(0),
                past_key_values=native.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        expected = head(
            native.last_hidden_state[:, -6:] if mtp else native.last_hidden_state[:, -1:]
        )
        torch.testing.assert_close(logits, expected, atol=2e-06, rtol=2e-06)
        checked.append(float((logits - expected).abs().max()))

    (original_mtp, original_ar) = (decoder._sample_row, decoder._sample_ar_row)

    def sample_mtp(logits, history, rng, kwargs):
        check_logits(logits, history, True)
        return original_mtp(logits, history, rng, kwargs)

    def sample_ar(logits, history, rng, kwargs):
        check_logits(logits, history, False)
        return original_ar(logits, history, rng, kwargs)

    (decoder._sample_row, decoder._sample_ar_row) = (sample_mtp, sample_ar)
    with torch.inference_mode():
        prefix = legacy_key_values(
            qwen(
                input_ids=torch.tensor([[1, 1]]),
                attention_mask=torch.ones(1, 2, dtype=torch.long),
                position_ids=torch.tensor([[0, 1]]),
                use_cache=True,
                return_dict=True,
            ).past_key_values
        )
        run_fake(decoder, prefix, [[11], [12, 13], [14]], budget=8)
    assert len(checked) >= 10
