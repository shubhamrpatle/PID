import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "explicit,venv,conda,local,expected",
    [
        (True, True, True, True, "explicit"),
        (False, True, True, True, "venv"),
        (False, False, True, True, "conda"),
        (False, False, True, False, "conda"),
        (False, False, False, True, "local"),
        (False, False, False, False, "path"),
    ],
)
def test_interpreter_priority_without_package_checks(tmp_path, explicit, venv, conda, local, expected):
    repo = tmp_path / "repo with spaces"
    (repo / "scripts").mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/common.sh", repo / "scripts/common.sh")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHON_BIN", "VIRTUAL_ENV", "CONDA_PREFIX")}
    paths = {}
    for name, directory in (
        ("explicit", tmp_path / "explicit env"),
        ("venv", tmp_path / "active venv"),
        ("conda", tmp_path / "conda env"),
        ("local", repo / ".venv"),
        ("path", tmp_path / "path python"),
    ):
        path = directory / "bin/python"
        path.parent.mkdir(parents=True)
        if name != "local" or local:
            path.write_text("#!/bin/sh\nexit 99\n")
            path.chmod(0o755)
        paths[name] = path
    env["PATH"] = str(paths["path"].parent) + os.pathsep + env["PATH"]
    if explicit:
        env["PYTHON_BIN"] = str(paths["explicit"])
    if venv:
        env["VIRTUAL_ENV"] = str(paths["venv"].parent.parent)
    if conda:
        env["CONDA_PREFIX"] = str(paths["conda"].parent.parent)
    result = subprocess.run(
        ["bash", "-c", 'source "$1/scripts/common.sh"; printf "%s" "$PYTHON_BIN"', "bash", str(repo)],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(paths[expected])
