import json
from pathlib import Path
import subprocess

import pytest

from test_simple_release import ROOT, fake_python


def launch(env, gpus=4, stage="stage1", tag=None):
    return subprocess.run(
        ["bash", str(ROOT / "train_lora.sh"), str(gpus), stage,
         *([tag] if tag is not None else [])],
        env=env, capture_output=True, text=True, timeout=30,
    )


def commands(env):
    return [json.loads(line) for line in Path(env["COMMAND_LOG"]).read_text().splitlines()]


@pytest.mark.parametrize("gpus", [4, 8])
@pytest.mark.parametrize("tag", [None, "ft_2"])
def test_fresh_runs_create_distinct_directories_even_with_identical_timestamp(tmp_path, gpus, tag):
    env = fake_python(tmp_path)
    env.update(METHOD="refwave", STAGE1_EPOCHS="2", STAGE2_EPOCHS="3")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    clock = bin_dir / "date"
    clock.write_text("#!/bin/sh\nprintf '20260908T120000Z\\n'\n")
    clock.chmod(0o755)
    env["PATH"] = str(bin_dir) + ":" + env["PATH"]
    for _ in range(2):
        result = launch(env, gpus, tag=tag)
        assert result.returncode == 0, result.stderr + result.stdout
    outputs = [Path(c[c.index("--output") + 1]) for c in commands(env)]
    assert len(outputs) == 2 and outputs[0].parent != outputs[1].parent
    for output in outputs:
        run = output.parent
        assert run.parent == Path(env["WORK_ROOT"]) / "runs"
        suffix = f"_{tag}" if tag else ""
        assert run.name.startswith(f"refwave_lora_{gpus}gpu_s1e2_s2e3_stage1{suffix}_20260908T120000Z_")
        log = (run / "console.log").read_text()
        assert f"RUN_ROOT={run}" in log
        assert "Resume with the same settings:" in log


@pytest.mark.parametrize("tag", [None, "new_label"])
def test_explicit_run_root_reuses_directory_and_appends_console(tmp_path, tag):
    env = fake_python(tmp_path)
    run = tmp_path / "existing run"
    env["RUN_ROOT"] = str(run)
    assert launch(env, tag=tag).returncode == 0
    previous = (run / "console.log").read_text()
    assert launch(env, tag=tag).returncode == 0
    log = (run / "console.log").read_text()
    assert log.startswith(previous) and len(log) > len(previous)
    assert all(c[c.index("--output") + 1] == str(run / "stage1") for c in commands(env))


def test_stage2_requires_explicit_existing_run(tmp_path):
    env = fake_python(tmp_path)
    result = launch(env, stage="stage2")
    assert result.returncode == 2
    assert "Set RUN_ROOT" in result.stderr
    assert not Path(env["COMMAND_LOG"]).exists()


@pytest.mark.parametrize("tag", [None, "ft_2"])
def test_both_stages_share_one_unique_directory_and_select_best(tmp_path, tag):
    env = fake_python(tmp_path)
    env.update(STAGE1_EPOCHS="2", STAGE2_EPOCHS="3")
    Path(env["PYTHON_BIN"]).write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        'with open(os.environ["COMMAND_LOG"], "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
        'out = Path(sys.argv[sys.argv.index("--output") + 1])\n'
        'out.mkdir(parents=True, exist_ok=True)\n'
        '(out / "done.json").write_text("{}")\n'
        '(out / "last.pt").write_text("fake checkpoint")\n'
        '(out / "best.pt").write_text("fake best checkpoint")\n'
        'print("worker stdout")\nprint("worker stderr", file=sys.stderr)\n'
    )
    result = launch(env, stage="both", tag=tag)
    assert result.returncode == 0, result.stderr + result.stdout
    first, second = commands(env)
    run = Path(first[first.index("--output") + 1]).parent
    assert run.name.startswith("refwave_lora_4gpu_s1e2_s2e3_both_")
    if tag:
        assert f"_both_{tag}_" in run.name
    assert second[second.index("--output") + 1] == str(run / "stage2")
    assert second[second.index("--init-checkpoint") + 1] == str(run / "stage1/best.pt")
    assert first[first.index("--epochs") + 1] == "2"
    assert second[second.index("--epochs") + 1] == "3"
    log = (run / "console.log").read_text()
    assert log.count("worker stdout") == log.count("worker stderr") == 2


def test_stage2_resume_does_not_change_original_parent(tmp_path):
    env = fake_python(tmp_path)
    run = tmp_path / "old run"
    (run / "stage2").mkdir(parents=True)
    (run / "stage2/last.pt").write_text("old checkpoint")
    env.update(RUN_ROOT=str(run), STAGE2_EPOCHS="2")
    result = launch(env, stage="stage2")
    assert result.returncode == 0, result.stderr + result.stdout
    command, = commands(env)
    assert "--init-checkpoint" not in command
    assert command[command.index("--epochs") + 1] == "2"


@pytest.mark.parametrize("tag", [None, "continue_ft2"])
def test_stage2_continuation_creates_new_run_without_stage1(tmp_path, tag):
    env = fake_python(tmp_path)
    parent = tmp_path / "old run/stage2/best.pt"
    env.update(CONTINUE_FROM=str(parent), STAGE2_EPOCHS="3")
    result = launch(env, stage="stage2", tag=tag)
    assert result.returncode == 0, result.stderr + result.stdout
    command, = commands(env)
    assert command[command.index("--continue-checkpoint") + 1] == str(parent)
    assert command[command.index("--epochs") + 1] == "3"
    assert "--init-checkpoint" not in command
    assert Path(command[command.index("--output") + 1]).parent != parent.parent.parent
    assert Path(command[command.index("--output") + 1]).parent.name.startswith(
        "refwave_lora_4gpu_stage2_extrae3_" + (f"{tag}_" if tag else "")
    )


@pytest.mark.parametrize("method,tuning", [
    ("refwave", "lora"), ("refwave", "full"), ("standard", "lora"),
    ("standard", "full"), ("class_isolated", "lora"), ("class_isolated", "full"),
])
@pytest.mark.parametrize("tag", [None, "ft_2", "run-2.v1", "a" * 64])
def test_run_name_identifies_method_tuning_gpus_and_epochs(tmp_path, method, tuning, tag):
    env = dict(fake_python(tmp_path), METHOD=method, STAGE1_EPOCHS="2", STAGE2_EPOCHS="4")
    result = subprocess.run(
        ["bash", str(ROOT / f"train_{tuning}.sh"), "8", "stage1",
         *([tag] if tag is not None else [])],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    command, = commands(env)
    run = Path(command[command.index("--output") + 1]).parent
    suffix = f"_{tag}" if tag else ""
    assert run.name.startswith(f"{method}_{tuning}_8gpu_s1e2_s2e4_stage1{suffix}_")
    assert command[command.index("--method") + 1] == method
    assert command[command.index("--tuning") + 1] == tuning
    assert command[command.index("--epochs") + 1] == "2"


@pytest.mark.parametrize("tag", ["../bad", "two words", "x/y", "x\ny", "a" * 65, "--bad", "$(id)"])
def test_invalid_run_label_never_launches_or_creates_run(tmp_path, tag):
    env = fake_python(tmp_path)
    result = launch(env, tag=tag)
    assert result.returncode == 2
    assert "Run label" in result.stderr
    assert not Path(env["COMMAND_LOG"]).exists()
    assert not (Path(env["WORK_ROOT"]) / "runs").exists()


def test_extra_arguments_are_not_silently_ignored(tmp_path):
    env = fake_python(tmp_path)
    result = subprocess.run(
        ["bash", str(ROOT / "train_full.sh"), "8", "both", "ft_2", "extra"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "at most one optional run label" in result.stderr
    assert not Path(env["COMMAND_LOG"]).exists()


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "abc"])
def test_invalid_epoch_count_never_launches_training(tmp_path, value):
    env = dict(fake_python(tmp_path), STAGE1_EPOCHS=value)
    assert launch(env).returncode == 2
    assert not Path(env["COMMAND_LOG"]).exists()


def test_continuation_cannot_accidentally_restart_both_stages(tmp_path):
    env = dict(fake_python(tmp_path), CONTINUE_FROM="/old/run/stage2/best.pt")
    assert launch(env, stage="both").returncode == 2
    assert not Path(env["COMMAND_LOG"]).exists()


def test_failed_training_stays_failed_with_console_logging(tmp_path):
    env = dict(fake_python(tmp_path), FAKE_STATUS="17")
    result = launch(env, stage="both")
    assert result.returncode == 17
    assert len(commands(env)) == 1
    output = commands(env)[0]
    run = Path(output[output.index("--output") + 1]).parent
    assert (run / "console.log").is_file()
