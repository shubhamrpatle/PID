import hashlib
import sys
from types import SimpleNamespace

import pytest

from refwave import train
from test_restart import tiny_trainer


@pytest.fixture
def tracking(monkeypatch):
    runs = []

    def init(**config):
        run = SimpleNamespace(config=config, history={}, events=[], metrics={}, finished=False)

        def define_metric(name, **options):
            run.metrics[name] = options

        def log(data, *, commit):
            assert commit is True
            run.events.append(data.copy())
            run.history.setdefault(data["optimizer_step"], {}).update(data)

        def finish():
            run.finished = True

        run.log = log
        run.define_metric = define_metric
        run.finish = finish
        runs.append(run)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    return runs


@pytest.mark.parametrize("method,tuning", [("refwave", "full"), ("standard", "full"), ("class_isolated", "full")])
def test_wandb_names_separate_fresh_runs_and_group_their_stages(tiny_trainer, tracking, method, tuning):
    args, _ = tiny_trainer
    args.wandb_mode = "online"
    args.wandb_project = "test-refwave"
    args.method, args.tuning = method, tuning
    parent = args.output.parent
    for suffix in ("4gpu_20260908T120000Z_first", "8gpu_20260908T120000Z_second"):
        root = parent / f"{method}_{tuning}_{suffix}"
        for stage in ("stage1", "stage2"):
            args.stage = stage
            args.output = root / stage
            args.init_checkpoint = root / "stage1/last.pt" if stage == "stage2" else None
            train.run(args)
            run = tracking[-1]
            assert run.config["name"] == f"{root.name}_{stage}"
            assert run.config["group"] == root.name
            assert run.config["job_type"] == stage
            assert run.config["dir"] == str(args.output.resolve())
            assert run.config["project"] == "test-refwave"
            assert run.config["mode"] == "online"
            assert run.config["id"] == hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()[:16]
            assert run.finished
            assert "train/loss" in run.history[2]
            assert "train/round1_loss" in run.history[2]
            assert "train/round2_loss" in run.history[2]
            assert "validation/detection_grounding_macro_loss" in run.history[2]
            assert run.metrics["train/*"]["step_metric"] == "optimizer_step"
            assert run.metrics["validation/*"]["step_metric"] == "optimizer_step"
    assert len({run.config["id"] for run in tracking}) == 4
    assert len({run.config["name"] for run in tracking}) == 4
    assert len({run.config["group"] for run in tracking}) == 2


def test_wandb_resume_keeps_stage_identity(tiny_trainer, tracking):
    args, state = tiny_trainer
    args.output = args.output / "stage1"
    args.wandb_mode = "online"
    args.wandb_project = "test-refwave"
    state.on_update = state.stop
    with pytest.raises(SystemExit) as stopped:
        train.run(args)
    assert stopped.value.code == 130
    state.on_update = None
    train.run(args)
    assert state.updates == 2
    assert len(tracking) == 2
    for key in ("id", "name", "group", "project", "dir"):
        assert tracking[0].config[key] == tracking[1].config[key]
    assert all(run.config["resume"] == "allow" and run.finished for run in tracking)
    assert list(tracking[0].history) == [1]
    assert list(tracking[1].history) == [2]


def test_wandb_disabled_does_not_start_tracking(tiny_trainer, tracking):
    args, _ = tiny_trainer
    train.run(args)
    assert not tracking


def test_first_step_is_committed_before_next_update_and_validation_is_not_lost(tiny_trainer, tracking):
    args, state = tiny_trainer
    args.wandb_mode = "online"
    args.wandb_project = "test-refwave"

    def during_update():
        if state.updates == 2:
            assert len(tracking[0].events) == 1
            assert tracking[0].events[0]["optimizer_step"] == 1
            assert "train/loss" in tracking[0].events[0]
            assert not tracking[0].finished

    state.on_update = during_update
    train.run(args)
    events = tracking[0].events
    assert [event["optimizer_step"] for event in events] == [1, 2, 2]
    assert "train/loss" in events[1]
    assert "validation/detection_grounding_macro_loss" in events[2]
