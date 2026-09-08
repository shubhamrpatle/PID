from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from refwave.model import (
    branch_ce,
    configure_lora_head,
    group_cross_entropy,
    group_jobs,
    pack_batch,
    update_indices,
)
from refwave.train import SCHEMA, restore, trainable_state, validate_parent
from refwave.data import _NativeMTP6
from refwave.episode import IGNORE_INDEX, pack_wave


def branch(name, length=8):
    return dict(
        branch_id=name,
        route="round1",
        polarity="positive",
        input_ids=torch.arange(length),
        position_ids=torch.arange(length),
        labels=torch.tensor([IGNORE_INDEX] * 4 + [2] * (length - 4)),
    )


@pytest.mark.parametrize("world", range(1, 9))
@pytest.mark.parametrize("size", [1, 3, 17, 32, 101])
def test_exact_one_pass_and_tail_gradient_normalization(world, size):
    order = list(reversed(range(size)))
    (seen, cursor) = ([], 0)
    while cursor < size:
        contributions = []
        for rank in range(world):
            (indices, end) = update_indices(order, cursor, 4, world, rank)
            seen.extend(indices)
            contributions.append(sum((x + 1 for x in indices)) / ((end - cursor) / world))
        assert sum(contributions) / world == pytest.approx(
            sum((x + 1 for x in order[cursor:end])) / (end - cursor)
        )
        cursor = end
    assert sorted(seen) == sorted(order)


def test_batched_masks_match_separate_and_dummy_rows_are_finite():
    waves = [
        [branch("a"), {**branch("b"), "input_ids": torch.tensor([0, 1, 2, 9, 4, 5, 6, 7])}],
        [branch("c", 6)],
    ]
    (ids, positions, mask, items) = pack_batch(waves, 1, torch.device("cpu"), torch.float32)
    for i, wave in enumerate(waves):
        separate = pack_wave(
            wave, image_token_index=1, device=torch.device("cpu"), dtype=torch.float32
        )
        n = separate["input_ids"].numel()
        torch.testing.assert_close(ids[i, :n], separate["input_ids"][0])
        torch.testing.assert_close(mask[i, :, :n, :n], separate["attention_mask"][0])
    assert torch.isfinite(mask).any(-1).all()
    q = torch.randn(2, 2, ids.shape[1], 4)
    output = torch.nn.functional.scaled_dot_product_attention(q, q, q, attn_mask=mask)
    for i, item in enumerate(items):
        n = item["input_ids"].numel()
        expected = torch.nn.functional.scaled_dot_product_attention(
            q[i : i + 1, :, :n],
            q[i : i + 1, :, :n],
            q[i : i + 1, :, :n],
            attn_mask=mask[i : i + 1, :, :n, :n],
        )
        torch.testing.assert_close(output[i : i + 1, :, :n], expected)


def test_chunked_full_head_loss_and_gradients():
    hidden = torch.randn(1, 11, 8, requires_grad=True)
    head = torch.randn(17, 8, requires_grad=True)
    labels = torch.tensor([[-100] * 3 + [2, 3, 4, 7, -100, 1, 2, 3]])
    (observed, count) = branch_ce(hidden, labels, (3, 11), head, chunk_size=2)
    expected = torch.nn.functional.cross_entropy(
        hidden[0, :-1] @ head.T, labels[0, 1:], ignore_index=-100
    )
    torch.testing.assert_close(observed, expected)
    left = torch.autograd.grad(observed, (hidden, head), retain_graph=True)
    right = torch.autograd.grad(expected, (hidden, head))
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b)
    assert count == 7


def test_fused_head_matches_branch_objective_and_gradients():
    hidden = torch.randn(2, 12, 8, requires_grad=True)
    head = torch.randn(17, 8, requires_grad=True)
    items = [
        dict(
            labels=torch.tensor([[-100] * 3 + [2, 3, 4, 7, -100, 1, 2, 3, 4]]),
            branch_ranges=[(3, 8), (8, 12)],
        ),
        dict(
            labels=torch.tensor([[-100] * 5 + [3, 5, 1, 2, 3, -100, -100]]), branch_ranges=[(5, 10)]
        ),
    ]
    waves = [[dict(branch_id="a"), dict(branch_id="b")], [dict(branch_id="c")]]
    weights = [dict(a=0.5, b=0.5), dict(c=1.0)]
    (actual, summary) = group_cross_entropy(
        hidden, items, waves, weights, head, divisor=2, chunk_size=3
    )
    expected = hidden.new_zeros(())
    for i, (item, wave) in enumerate(zip(items, waves)):
        for b, span in zip(wave, item["branch_ranges"]):
            (value, _) = branch_ce(hidden[i : i + 1], item["labels"], span, head)
            expected = expected + value * weights[i][b["branch_id"]] / 2
    torch.testing.assert_close(actual, expected)
    for left, right in zip(
        torch.autograd.grad(actual, (hidden, head), retain_graph=True),
        torch.autograd.grad(expected, (hidden, head)),
    ):
        torch.testing.assert_close(left, right)
    assert [n for (_, n) in summary] == [4, 4, 5]


class TinyModel(nn.Module):

    def __init__(self, tied=False):
        super().__init__()
        self.vision_model = nn.Linear(4, 4)
        self.mlp1 = nn.Linear(4, 4)
        self.language_model = nn.Module()
        lm = self.language_model
        lm.model = nn.Module()
        lm.model.embed_tokens = nn.Embedding(7, 4)
        lm.lm_head = nn.Linear(4, 7, bias=False)
        if tied:
            lm.lm_head.weight = lm.model.embed_tokens.weight
        lm.config = SimpleNamespace(tie_word_embeddings=tied)
        lm.get_input_embeddings = lambda: lm.model.embed_tokens

    def wrap_llm_lora(self, **kwargs):
        self.language_model.model.lora_A = nn.Linear(4, 2, bias=False)
        self.language_model.model.lora_B = nn.Linear(2, 4, bias=False)


@pytest.mark.parametrize("tied", [False, True])
def test_only_qwen_lora_and_head_trainable_and_restored(tied):
    model = TinyModel(tied)
    configure_lora_head(model)
    names = [n for (n, p) in model.named_parameters() if p.requires_grad]
    assert names == [
        "language_model.model.lora_A.weight",
        "language_model.model.lora_B.weight",
        "language_model.lm_head.weight",
    ]
    assert not model.language_model.get_input_embeddings().weight.requires_grad
    saved = {n: p.clone() for (n, p) in trainable_state(model).items()}
    with torch.no_grad():
        model.language_model.lm_head.weight.zero_()
    restore(model, saved)
    torch.testing.assert_close(
        model.language_model.lm_head.weight, saved["language_model.lm_head.weight"]
    )
    del saved["language_model.lm_head.weight"]
    with pytest.raises(ValueError, match="head"):
        restore(model, saved)


def test_bucket_respects_physical_batch_and_keeps_oversized_singletons():
    jobs = [(i, [branch(str(i), n)]) for (i, n) in enumerate([6, 10, 7, 50])]
    grouped = list(group_jobs(jobs, 3, 24))
    assert sorted((i for group in grouped for (i, _) in group)) == list(range(4))
    assert all((len(group) <= 3 for group in grouped))
    assert any((len(group) == 1 and group[0][0] == 3 for group in grouped))


def import_tool(name):
    path = Path(__file__).parents[1] / "tools" / (name + ".py")
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_refuses_archive_traversal(tmp_path):
    import io
    import tarfile

    tool = import_tool("download_refwave_assets")
    path = tmp_path / "evil.tar"
    with tarfile.open(path, "w") as tar:
        member = tarfile.TarInfo("images/../../escape")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        tool.extract_archive(path, tmp_path / "data")


def test_pinned_eagle_mtp_signature_uses_constructor_block_size():
    helper = SimpleNamespace(block_size=6, get_targets_flag_with_mtp=lambda ids: {"input_ids": ids})
    ids = torch.tensor([1, 2])
    assert _NativeMTP6(helper).get_targets_flag_with_mtp(ids, block_size=6)["input_ids"] is ids
    with pytest.raises(ValueError, match="PBD6"):
        _NativeMTP6(helper).get_targets_flag_with_mtp(ids, block_size=10)


@pytest.mark.parametrize("change", ["method", "stage", "cursor", "lora_rank"])
def test_stage2_cannot_take_wrong_method_or_incomplete_parent(change):
    args = SimpleNamespace(method="refwave", lora_rank=32)
    parent = dict(
        schema=SCHEMA,
        cursor=100,
        config=dict(method="refwave", stage="stage1", lora_rank=32, records=100),
    )
    validate_parent(parent, args)
    if change == "cursor":
        parent["cursor"] = 99
    else:
        parent["config"][change] = "wrong"
    with pytest.raises(ValueError, match="completed Stage-1"):
        validate_parent(parent, args)
