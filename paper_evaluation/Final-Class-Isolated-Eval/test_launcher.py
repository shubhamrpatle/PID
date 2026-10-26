import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("selection", ["explicit", "venv", "conda", "sibling", "python", "python3"])
def test_launcher_uses_available_environment_without_helper(tmp_path, selection):
    root = tmp_path/"path with spaces"
    folder = root/"Final-Class-Isolated-Eval"
    folder.mkdir(parents=True)
    launcher = folder/"run.sh"
    shutil.copyfile(Path(__file__).with_name("run.sh"), launcher)
    pathbin = root/"bin"
    pathbin.mkdir()
    (pathbin/"dirname").symlink_to(shutil.which("dirname"))
    env = dict(os.environ,PATH=str(pathbin),MAX_WORKERS="4")
    for name in ("PYTHON_BIN","VIRTUAL_ENV","CONDA_PREFIX","CUDA_VISIBLE_DEVICES"):
        env.pop(name,None)
    candidates = {"explicit":root/"chosen/python", "venv":root/"active/bin/python",
        "conda":root/"conda/bin/python", "sibling":root/".venv/bin/python",
        "python":pathbin/"python", "python3":pathbin/"python3"}
    path = candidates[selection]
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('#!/bin/bash\nprintf "%s\\n" "$PYTHON_BIN" "$*"\n')
    path.chmod(0o755)
    if selection == "explicit":
        env["PYTHON_BIN"] = str(path)
    elif selection == "venv":
        env["VIRTUAL_ENV"] = str(path.parent.parent)
    elif selection == "conda":
        env["CONDA_PREFIX"] = str(path.parent.parent)
    for allocation, args, expected in [(None,["2","7"],"2 7"),
            (None,[],"0 1 2 3 4 5 6 7"),("3,6",[],"3 6")]:
        if allocation:
            env["CUDA_VISIBLE_DEVICES"] = allocation
        result = subprocess.run(["/bin/bash",str(launcher),*args],env=env,capture_output=True,text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [str(path),
            "-u evaluate.py launch --max-workers 4 --gpus "+expected]


def test_invalid_explicit_python_is_not_silently_replaced():
    result = subprocess.run(["bash",str(Path(__file__).with_name("run.sh"))],
        env=dict(os.environ,PYTHON_BIN="/nonexistent/python"),capture_output=True,text=True)
    assert result.returncode == 1
    assert "set PYTHON_BIN=" in result.stderr
