from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.mark.parametrize("method,tuning", [
    ("refwave", "lora"), ("standard", "lora"), ("refwave", "full"),
    ("class_isolated", "lora"), ("class_isolated", "full"),
])
def test_four_rank_start_stop_resume_and_stage_handoff(tmp_path, method, tuning):
    (tmp_path / "records.jsonl").write_text("{}\n" * 7)
    repo = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=4", str(Path(__file__).with_name("distributed_trainer_worker.py")),
         str(tmp_path), method, tuning],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                 PYTHONPATH=str(repo / "src")),
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    ranks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(4)]
    for stage, epochs in [("stage1", 2), ("stage2", 2), ("extra_stage2", 1)]:
        assert Counter(i for rank in ranks for i in rank[stage]) == Counter({i: epochs for i in range(7)})
        saved = torch.load(tmp_path / stage / "last.pt", map_location="cpu", weights_only=False)
        assert saved["step"] == epochs * 2
        assert saved["cursor"] == epochs * 7
        assert len(saved["rng"]) == 4
        assert saved["optimizer"]["state"]
        # A Stage-1 Adam step counter must not carry into Stage 2.
        assert all(int(s["step"]) == saved["step"] for s in saved["optimizer"]["state"].values())
        assert saved["scheduler"]["last_epoch"] == saved["step"]
        assert saved["config"]["optimizer_sharding"] == (tuning == "full")
        assert (tmp_path / stage / "best.pt").exists()
        assert (tmp_path / stage / "done.json").exists()
