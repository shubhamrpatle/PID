import json
import signal
from types import SimpleNamespace

import pytest
import torch

from refwave import train
from refwave.runtime import truncate_curve
from refwave.io import append_jsonl, read_jsonl_log
from refwave.class_wave import _BranchRNG
from refwave.loader import make_loader


def test_resume_recovers_only_unfinished_log_tail(tmp_path):
    path = tmp_path / "training_curve.jsonl"
    path.write_bytes(b'{"step":1}\n{"step":2}\n{"step":3')
    truncate_curve(path, "step", 1)
    assert [json.loads(line) for line in path.read_text().splitlines()] == [{"step": 1}]


def test_resume_does_not_hide_interior_log_corruption(tmp_path):
    path = tmp_path / "training_curve.jsonl"
    original = b'{"step":1}\nnot json\n{"step":2}\n'
    path.write_bytes(original)
    with pytest.raises(ValueError):
        truncate_curve(path, "step", 1)
    assert path.read_bytes() == original


@pytest.mark.parametrize("tail", [b'{"index":1', b'{"index":1,"text":"\xe2'])
def test_prediction_restart_discards_only_unfinished_entry(tmp_path, tail):
    path = tmp_path / "predictions.rank0.jsonl"
    path.write_bytes(b'{"index":0}\n' + tail)
    with pytest.warns(RuntimeWarning, match="unfinished"):
        assert read_jsonl_log(path, repair_tail=True) == [{"index": 0}]
    append_jsonl(path, {"index": 1})
    assert read_jsonl_log(path) == [{"index": 0}, {"index": 1}]


def test_complete_unterminated_prediction_is_not_lost_or_concatenated(tmp_path):
    path = tmp_path / "predictions.rank0.jsonl"
    path.write_bytes(b'{"index":0}')
    assert read_jsonl_log(path, repair_tail=True) == [{"index": 0}]
    append_jsonl(path, {"index": 1})
    assert read_jsonl_log(path) == [{"index": 0}, {"index": 1}]


def test_prediction_restart_rejects_corrupt_complete_line(tmp_path):
    path = tmp_path / "predictions.rank0.jsonl"
    original = b'{"index":0}\nbroken\n'
    path.write_bytes(original)
    with pytest.raises(ValueError, match="line 2"):
        read_jsonl_log(path, repair_tail=True)
    assert path.read_bytes() == original


def test_branch_rng_does_not_seed_other_devices_and_restores_outer_state(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))
    before = torch.get_rng_state()
    rng = _BranchRNG(101, torch.device("cpu"))
    assert calls == []
    torch.testing.assert_close(before, torch.get_rng_state())
    generator = torch.Generator().manual_seed(101)
    for _ in range(2):
        torch.testing.assert_close(rng.call(lambda: torch.rand(5)), torch.rand(5, generator=generator))
        torch.testing.assert_close(before, torch.get_rng_state())
    with pytest.raises(RuntimeError, match="sampling error"):
        rng.call(lambda: (_ for _ in ()).throw(RuntimeError("sampling error")))
    torch.testing.assert_close(before, torch.get_rng_state())


def test_loader_restart_does_not_advance_training_rng():
    before = torch.get_rng_state()
    assert list(make_loader([1, 2, 3], [0, 2], 0)) == [1, 3]
    assert list(make_loader([1, 2, 3], [1], 0)) == [2]
    torch.testing.assert_close(before, torch.get_rng_state())


@pytest.fixture
def tiny_trainer(tmp_path, monkeypatch):
    """Exercise the real training loop and checkpoints without allocating a GPU."""
    from test_cluster_training import TinyModel

    state = SimpleNamespace(updates=0, validations=0, seen=[], on_update=None, on_validation=None)
    handlers = {}

    class TorchProxy:
        cuda = SimpleNamespace(
            manual_seed_all=lambda seed: None,
            get_rng_state=torch.get_rng_state,
            set_rng_state=torch.set_rng_state,
            max_memory_allocated=lambda: 0,
        )

        def device(self, *args):
            return torch.device("cpu")

        def __getattr__(self, key):
            return getattr(torch, key)

    def load_base(*args):
        model = TinyModel()
        model.language_model.gradient_checkpointing_enable = lambda **kw: None
        model.language_model.enable_input_require_grads = lambda: None
        return model, None

    class Wrapper(torch.nn.Module):
        def __init__(self, model, **kwargs):
            super().__init__()
            self.model = model
            self.causal = model.language_model

        def batch(self, batch, divisor, **kwargs):
            state.seen.extend(batch)
            loss = sum(p.square().sum() for p in self.model.parameters() if p.requires_grad)
            (loss * len(batch) / divisor).backward()
            state.updates += 1
            if state.on_update:
                state.on_update()
            return [dict(loss=float(loss.detach()), round1_loss=1.0, round2_loss=1.0,
                         direct_loss=0.0, supervised_tokens=8, waves=2,
                         task="detection") for _ in batch]

    def validate(*args):
        state.validations += 1
        if state.on_validation:
            state.on_validation()
        return dict(detection_grounding_macro_loss=1.0)

    monkeypatch.setattr(train, "torch", TorchProxy())
    monkeypatch.setattr(train, "distributed_context", lambda: (0, 0, 1))
    monkeypatch.setattr(train, "load_base", load_base)
    monkeypatch.setattr(
        train, "enable_fused_vision_attention", lambda model: "refwave_fused_per_image_sdpa"
    )
    monkeypatch.setattr(train, "ClusterEpisodeModel", Wrapper)
    monkeypatch.setattr(train, "ClusterDataset", lambda path, *a, **k: [0, 1])
    monkeypatch.setattr(train, "make_loader", lambda dataset, indices, workers: iter(indices))
    monkeypatch.setattr(train, "validate", validate)
    monkeypatch.setattr(train.signal, "signal", lambda sig, handler: handlers.update({sig: handler}))
    state.stop = lambda: handlers[signal.SIGTERM](signal.SIGTERM, None)
    records = tmp_path / "records.jsonl"
    records.write_text("{}\n{}\n")
    args = SimpleNamespace(
        seed=1, model=tmp_path / "model", image_token_limit=6000,
        tuning="full", lora_rank=32, method="refwave", stage="stage1",
        data_root=tmp_path, train_data=records, validation_data=records,
        max_sequence=32768, batch_size=1, accumulation=1, batch_tokens=8192,
        learning_rate=2e-5, head_lr=1e-6, vision_lr=1e-6, projector_lr=1e-5,
        warmup_ratio=0.03, output=tmp_path / "run", init_checkpoint=None,
        workers=0, eval_steps=300, save_steps=300, log_steps=10,
        epochs=1, continue_checkpoint=None,
        wandb_mode="disabled",
    )
    return args, state


def test_final_update_interrupt_resumes_validation_before_completion(tiny_trainer):
    args, state = tiny_trainer
    state.on_update = lambda: state.stop() if state.updates == 2 else None
    with pytest.raises(SystemExit) as stopped:
        train.run(args)
    assert stopped.value.code == 130
    assert not (args.output / "done.json").exists()
    assert not (args.output / "best.pt").exists()
    # Validation may finish writing its log before checkpoint replacement fails.
    append_jsonl(args.output / "validation_curve.jsonl", {"step": 2, "loss": 999})
    state.on_update = None
    train.run(args)
    assert state.updates == 2
    assert state.validations == 1
    saved = torch.load(args.output / "last.pt", weights_only=False)
    assert saved["validation"]["detection_grounding_macro_loss"] == 1.0
    assert len(read_jsonl_log(args.output / "validation_curve.jsonl")) == 1
    assert (args.output / "best.pt").exists()
    assert json.loads((args.output / "done.json").read_text())["records"] == 2


def test_stop_during_validation_is_saved_and_aligned(tiny_trainer):
    args, state = tiny_trainer
    args.eval_steps = 1
    state.on_validation = state.stop
    with pytest.raises(SystemExit) as stopped:
        train.run(args)
    assert stopped.value.code == 130
    assert state.updates == 1
    saved = torch.load(args.output / "last.pt", weights_only=False)
    assert saved["step"] == saved["cursor"] == 1
    assert saved["validation"] is not None


def test_resume_keeps_updates_exact_and_logs_used_learning_rate(tiny_trainer):
    args, state = tiny_trainer
    args.warmup_ratio = 1.0
    state.on_update = state.stop
    with pytest.raises(SystemExit):
        train.run(args)
    state.on_update = None
    train.run(args)
    assert state.updates == 2
    rows = read_jsonl_log(args.output / "training_curve.jsonl")
    assert [r["step"] for r in rows] == [1, 2]
    assert [r["records_seen"] for r in rows] == [1, 2]
    assert [r["learning_rate"] for r in rows] == pytest.approx([args.learning_rate / 2] * 2)
    assert not (args.output / "interrupted.json").exists()
    train.run(args)
    assert state.updates == 2
    assert state.validations == 1


def test_best_pointer_is_recovered_after_interrupted_checkpoint_publication(tiny_trainer):
    args, state = tiny_trainer
    train.run(args)
    (args.output / "best.pt").unlink()
    train.run(args)
    assert state.updates == 2
    assert state.validations == 1
    assert (args.output / "best.pt").resolve() == (args.output / "last.pt").resolve()


@pytest.mark.parametrize("method,tuning", [("refwave", "full"), ("standard", "full"), ("class_isolated", "full")])
@pytest.mark.parametrize("epochs", [1, 2])
def test_new_run_tolerates_config_published_during_hashing(tiny_trainer, monkeypatch, method, tuning, epochs):
    args, state = tiny_trainer
    args.method, args.tuning, args.epochs = method, tuning, epochs
    train.run(args)
    expected = json.loads((args.output / "config.json").read_text())
    args.output = args.output.with_name("fresh_parallel_start")
    config_path = args.output / "config.json"
    real_hash = train.sha256_file
    published = []

    def publish_like_faster_rank(path):
        if not published:
            # The current rank has already cached the initial missing config.
            assert not config_path.exists()
            train.atomic_json(config_path, expected)
            published.append(True)
        return real_hash(path)

    monkeypatch.setattr(train, "sha256_file", publish_like_faster_rank)
    train.run(args)
    assert published == [True]
    assert json.loads(config_path.read_text()) == expected
    assert json.loads((args.output / "done.json").read_text())["records"] == 2 * epochs


def test_existing_empty_config_is_not_treated_as_a_fresh_run(tiny_trainer):
    args, _ = tiny_trainer
    args.output.mkdir()
    config_path = args.output / "config.json"
    config_path.write_text("{}\n")
    with pytest.raises(ValueError, match="Existing run settings differ"):
        train.run(args)
    assert config_path.read_text() == "{}\n"
    assert not (args.output / "last.pt").exists()


def test_actual_settings_mismatch_reports_fields_without_overwriting(tiny_trainer):
    args, _ = tiny_trainer
    train.run(args)
    before = (args.output / "config.json").read_bytes()
    checkpoint_sha = train.sha256_file(args.output / "last.pt")
    args.epochs = 2
    with pytest.raises(ValueError, match="Changed fields: epochs, steps") as error:
        train.run(args)
    assert '"epochs": {"requested": 2, "saved": 1}' in str(error.value)
    assert (args.output / "config.json").read_bytes() == before
    assert train.sha256_file(args.output / "last.pt") == checkpoint_sha
