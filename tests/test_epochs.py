import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from refwave import train
from refwave.io import read_jsonl_log, sha256_file
from test_restart import tiny_trainer


@pytest.mark.parametrize("world", range(1, 9))
@pytest.mark.parametrize("size", [1, 3, 17, 32, 101])
def test_multi_epoch_coverage_and_tail_batches(world, size):
    epochs, per_gpu, seed = 3, 4, 123
    steps_per_epoch = math.ceil(size / (per_gpu * world))
    plans = [list(train.epoch_updates(size, epochs, seed, per_gpu, world, rank))
             for rank in range(world)]
    assert all(len(plan) == epochs * steps_per_epoch for plan in plans)
    for epoch in range(epochs):
        seen = []
        previous = epoch * size
        for index in range(epoch * steps_per_epoch, (epoch + 1) * steps_per_epoch):
            end = plans[0][index][1]
            indices = [i for plan in plans for i in plan[index][0]]
            assert all(plan[index][1] == end for plan in plans)
            assert len(indices) == end - previous
            assert end == train.cursor_after_step(index + 1, size, per_gpu * world)
            seen.extend(indices)
            previous = end
        assert sorted(seen) == list(range(size))
    for rank, plan in enumerate(plans):
        for boundary in [0, 1, steps_per_epoch, len(plan) - 1, len(plan)]:
            cursor = train.cursor_after_step(boundary, size, per_gpu * world)
            assert list(train.epoch_updates(size, epochs, seed, per_gpu, world, rank, cursor)) == plan[boundary:]


def test_epoch_shuffle_preserves_old_first_pass():
    size, seed = 29, 20260908
    passes = list(train.epoch_updates(size, 3, seed, size, 1, 0))
    assert passes[0][0] == np.random.default_rng(seed).permutation(size).tolist()
    assert len({tuple(indices) for indices, _ in passes}) == 3


@pytest.mark.parametrize("stop_step", [1, 3, 4, 9])
def test_multi_epoch_resume_matches_uninterrupted_training(tiny_trainer, monkeypatch, stop_step):
    args, state = tiny_trainer
    monkeypatch.setattr(train, "ClusterDataset", lambda *a, **k: list(range(5)))
    args.batch_size = 2
    args.epochs = 3
    reference_dir = args.output
    train.run(args)
    reference = torch.load(reference_dir / "last.pt", weights_only=False)
    expected_order = state.seen.copy()
    assert len(expected_order) == 15
    assert state.updates == 9
    assert state.validations == 3
    assert [row["step"] for row in read_jsonl_log(reference_dir / "validation_curve.jsonl")] == [3, 6, 9]
    args.output = reference_dir.with_name("interrupted")
    state.updates = state.validations = 0
    state.seen.clear()
    state.on_update = lambda: state.stop() if state.updates == stop_step else None
    with pytest.raises(SystemExit) as stopped:
        train.run(args)
    assert stopped.value.code == 130
    state.on_update = None
    train.run(args)
    resumed = torch.load(args.output / "last.pt", weights_only=False)
    assert state.seen == expected_order
    assert state.updates == 9
    assert state.validations == 3
    assert resumed["cursor"] == 15
    assert resumed["step"] == 9
    assert resumed["scheduler"] == reference["scheduler"]
    for name, value in resumed["trainable_state"].items():
        torch.testing.assert_close(value, reference["trainable_state"][name], rtol=0, atol=0)
    done = json.loads((args.output / "done.json").read_text())
    assert done == dict(steps=9, records=15, total_records=15, epochs=3)
    rows = read_jsonl_log(args.output / "training_curve.jsonl")
    assert [r["records_seen"] for r in rows] == [2, 4, 5, 7, 9, 10, 12, 14, 15]
    assert rows[-1]["epoch"] == 3
    train.run(args)
    assert state.updates == 9


def test_stage2_uses_early_best_from_completed_multi_epoch_stage1(tiny_trainer, monkeypatch):
    args, state = tiny_trainer
    args.epochs = 2
    args.eval_steps = 1
    monkeypatch.setattr(train, "validate", lambda *a: dict(detection_grounding_macro_loss=float(state.updates)))
    args.output = args.output / "stage1"
    train.run(args)
    best_path = args.output / "best.pt"
    parent = torch.load(best_path, weights_only=False)
    assert parent["step"] == 1
    assert parent["cursor"] < parent["config"]["records"] * parent["config"]["epochs"]
    restored = []
    original_restore = train.restore

    def capture_restore(model, values):
        restored.append({key: value.clone() for key, value in values.items()})
        original_restore(model, values)

    monkeypatch.setattr(train, "restore", capture_restore)
    args.stage = "stage2"
    args.init_checkpoint = best_path
    args.output = args.output.parent / "stage2"
    train.run(args)
    assert len(restored) == 1
    for key, value in restored[0].items():
        torch.testing.assert_close(value, parent["trainable_state"][key], rtol=0, atol=0)


def test_stage2_continuation_is_new_training_and_preserves_parent(tiny_trainer):
    args, state = tiny_trainer
    args.output = args.output / "stage1"
    train.run(args)
    args.init_checkpoint = args.output / "best.pt"
    args.stage = "stage2"
    args.output = args.output.parent / "stage2"
    train.run(args)
    parent = args.output / "best.pt"
    original_hash = sha256_file(parent)
    args.output = args.output.parent.parent / "extra" / "stage2"
    args.init_checkpoint = None
    args.continue_checkpoint = parent
    args.epochs = 2
    args.learning_rate /= 2
    args.head_lr /= 2
    state.updates = 0
    state.on_update = state.stop
    with pytest.raises(SystemExit):
        train.run(args)
    saved = torch.load(args.output / "last.pt", weights_only=False)
    assert saved["step"] == saved["cursor"] == 1
    assert saved["config"]["continue_checkpoint_sha256"] == original_hash
    assert saved["config"]["init_checkpoint_sha256"] is None
    # Resume does not need to reread a parent, and keeps the continuation provenance.
    args.continue_checkpoint = None
    state.on_update = None
    train.run(args)
    saved = torch.load(args.output / "last.pt", weights_only=False)
    assert saved["step"] == saved["cursor"] == state.updates == 4
    assert saved["config"]["continue_checkpoint_sha256"] == original_hash
    assert sha256_file(parent) == original_hash


def test_existing_stage2_resume_keeps_original_stage1_parent(tiny_trainer):
    args, state = tiny_trainer
    args.output = args.output / "stage1"
    train.run(args)
    args.init_checkpoint = args.output / "last.pt"
    args.stage = "stage2"
    args.output = args.output.parent / "stage2"
    state.on_update = state.stop
    with pytest.raises(SystemExit):
        train.run(args)
    parent_sha = json.loads((args.output / "config.json").read_text())["init_checkpoint_sha256"]
    args.init_checkpoint = None
    state.on_update = None
    train.run(args)
    saved = torch.load(args.output / "last.pt", weights_only=False)
    assert saved["config"]["init_checkpoint_sha256"] == parent_sha
    assert saved["cursor"] == 2


def test_epoch_change_cannot_overwrite_an_existing_run(tiny_trainer):
    args, _ = tiny_trainer
    train.run(args)
    original_hash = sha256_file(args.output / "last.pt")
    args.epochs = 2
    with pytest.raises(ValueError, match="new output directory"):
        train.run(args)
    assert sha256_file(args.output / "last.pt") == original_hash


def test_incomplete_multi_epoch_stage1_is_not_a_valid_parent():
    args = SimpleNamespace(method="refwave", lora_rank=32)
    parent = dict(schema=train.SCHEMA, cursor=100, step=10,
                  config=dict(method="refwave", lora_rank=32, stage="stage1",
                              epochs=2, records=100, steps=20))
    with pytest.raises(ValueError, match="completed Stage-1"):
        train.validate_parent(parent, args)
