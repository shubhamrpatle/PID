import json
from pathlib import Path
import subprocess

import pytest

from test_simple_release import ROOT, fake_python


def launch(env, gpus=4, stage="stage1"):
    return subprocess.run(
        ["bash", str(ROOT / "train_lora.sh"), str(gpus), stage],
        env=env, capture_output=True, text=True, timeout=30,
    )


def commands(env):
    return [json.loads(line) for line in Path(env["COMMAND_LOG"]).read_text().splitlines()]


@pytest.mark.parametrize("gpus", [4, 8])
def test_fresh_runs_create_distinct_directories_even_with_identical_timestamp(tmp_path, gpus):
    env = fake_python(tmp_path)
    env["METHOD"] = "refwave"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    clock = bin_dir / "date"
    clock.write_text("#!/bin/sh\nprintf '20260908T120000Z\\n'\n")
    clock.chmod(0o755)
    env["PATH"] = str(bin_dir) + ":" + env["PATH"]
    for _ in range(2):
        result = launch(env, gpus)
        assert result.returncode == 0, result.stderr + result.stdout
    outputs = [Path(c[c.index("--output") + 1]) for c in commands(env)]
    assert len(outputs) == 2 and outputs[0].parent != outputs[1].parent
    for output in outputs:
        run = output.parent
        assert run.parent == Path(env["WORK_ROOT"]) / "runs"
        assert run.name.startswith(f"refwave_lora_{gpus}gpu_20260908T120000Z_")
        log = (run / "console.log").read_text()
        assert f"RUN_ROOT={run}" in log
        assert "Resume with the same settings:" in log


def test_explicit_run_root_reuses_directory_and_appends_console(tmp_path):
    env = fake_python(tmp_path)
    run = tmp_path / "existing run"
    env["RUN_ROOT"] = str(run)
    assert launch(env).returncode == 0
    previous = (run / "console.log").read_text()
    assert launch(env).returncode == 0
    log = (run / "console.log").read_text()
    assert log.startswith(previous) and len(log) > len(previous)
    assert all(c[c.index("--output") + 1] == str(run / "stage1") for c in commands(env))


def test_stage2_requires_explicit_existing_run(tmp_path):
    env = fake_python(tmp_path)
    result = launch(env, stage="stage2")
    assert result.returncode == 2
    assert "Set RUN_ROOT" in result.stderr
    assert not Path(env["COMMAND_LOG"]).exists()


def test_both_stages_share_one_unique_directory(tmp_path):
    env = fake_python(tmp_path)
    Path(env["PYTHON_BIN"]).write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        'with open(os.environ["COMMAND_LOG"], "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
        'out = Path(sys.argv[sys.argv.index("--output") + 1])\n'
        'out.mkdir(parents=True, exist_ok=True)\n'
        '(out / "done.json").write_text("{}")\n'
        '(out / "last.pt").write_text("fake checkpoint")\n'
        'print("worker stdout")\nprint("worker stderr", file=sys.stderr)\n'
    )
    result = launch(env, stage="both")
    assert result.returncode == 0, result.stderr + result.stdout
    first, second = commands(env)
    run = Path(first[first.index("--output") + 1]).parent
    assert second[second.index("--output") + 1] == str(run / "stage2")
    assert second[second.index("--init-checkpoint") + 1] == str(run / "stage1/last.pt")
    log = (run / "console.log").read_text()
    assert log.count("worker stdout") == log.count("worker stderr") == 2


def test_failed_training_stays_failed_with_console_logging(tmp_path):
    env = dict(fake_python(tmp_path), FAKE_STATUS="17")
    result = launch(env, stage="both")
    assert result.returncode == 17
    assert len(commands(env)) == 1
    output = commands(env)[0]
    run = Path(output[output.index("--output") + 1]).parent
    assert (run / "console.log").is_file()
