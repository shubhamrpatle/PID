from pathlib import Path
import os
import subprocess
import sys


def test_two_cpu_ranks_consolidate_and_restore_optimizer(tmp_path):
    script = Path(__file__).with_name("zero_worker.py")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(script),
            str(tmp_path / "optimizer.pt"),
        ],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
