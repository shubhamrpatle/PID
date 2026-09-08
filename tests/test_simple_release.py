import ast
import copy
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from refwave.evaluate import inference_query, parse_geometry, summarize
from refwave.messages import image_messages
from refwave.model import VisualGradientBridge, configure_full
from refwave.train import parameter_groups, validate_parent, SCHEMA

ROOT = Path(__file__).resolve().parents[1]


def test_visual_bridge_matches_joint_vision_and_decoder_gradient():
    torch.manual_seed(1)
    encoder = torch.nn.Sequential(torch.nn.Linear(5, 9), torch.nn.GELU(), torch.nn.Linear(9, 7))
    decoder = torch.nn.Linear(7, 4)
    image = torch.randn(3, 5)
    projected = encoder(image)
    bridge = VisualGradientBridge([projected])
    for weight, target in ((0.25, 1.0), (0.75, -1.0)):
        loss = weight * (decoder(bridge.leaves[0]) - target).square().mean()
        loss.backward()
    bridge.backward()
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    observed = [p.grad.clone() for p in parameters]
    for p in parameters:
        p.grad = None
    value = decoder(encoder(image))
    (0.25 * (value - 1).square().mean() + 0.75 * (value + 1).square().mean()).backward()
    for expected, p in zip(observed, parameters):
        torch.testing.assert_close(expected, p.grad)
    assert all(p.grad.abs().sum() > 0 for p in parameters)


def test_full_mode_covers_every_parameter_without_lora():
    from test_cluster_training import TinyModel

    model = TinyModel()
    model.requires_grad_(False)
    audit = configure_full(model)
    wrapper = SimpleNamespace(causal=model.language_model)
    args = SimpleNamespace(learning_rate=1e-6, head_lr=1e-6, vision_lr=1e-6, projector_lr=1e-5)
    groups = parameter_groups(model, wrapper, args)
    actual = [id(p) for group in groups for p in group["params"]]
    assert len(actual) == len(set(actual)) == len(list(model.parameters()))
    assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
    assert audit["trainable_parameters"] == sum(p.numel() for p in model.parameters())
    assert not any("lora_" in name for name, _ in model.named_parameters())


def test_parent_cannot_cross_full_and_lora():
    parent = dict(
        schema=SCHEMA,
        cursor=10,
        config=dict(method="refwave", stage="stage1", lora_rank=32, tuning="lora", records=10),
    )
    with pytest.raises(ValueError, match="tuning mode"):
        validate_parent(parent, SimpleNamespace(method="refwave", lora_rank=32, tuning="full"))


def test_single_image_messages_do_not_change_answers(tmp_path):
    conversation = [
        {"from": "human", "value": "<image>\nLocate ship."},
        {"from": "gpt", "value": "<box>None</box>"},
    ]
    original = copy.deepcopy(conversation)
    messages = image_messages("images/a.jpg", conversation, tmp_path)
    assert conversation == original
    assert messages == [
        dict(
            role="user",
            content=[
                dict(type="text", text="<image-1>\nLocate ship."),
                dict(type="image", image=str(tmp_path / "images/a.jpg")),
            ],
        ),
        dict(role="assistant", content="<box>None</box>"),
    ]
    with pytest.raises(ValueError, match="inside DATA_ROOT"):
        image_messages("../outside.jpg", conversation, tmp_path)


def test_inference_never_uses_gt_points_or_round2_targets():
    row = dict(
        image="images/a.jpg",
        task="detection",
        direct=dict(
            conversations=[dict(value="<image>\nLocate ship and bridge."), dict(value="GT SECRET")]
        ),
        round1=dict(
            branches=[
                dict(
                    reference="ship",
                    target_points=[[1, 2]],
                    polarity="positive",
                    conversations=[dict(value="<image>\nPoint to ship."), dict(value="GT SECRET")],
                ),
                dict(
                    reference="bridge",
                    target_points=[],
                    polarity="negative",
                    conversations=[
                        dict(value="<image>\nPoint to bridge."),
                        dict(value="GT SECRET"),
                    ],
                ),
            ]
        ),
        round2=dict(branches=[dict(point=[1, 2], target_hbb=[0, 0, 4, 4])]),
    )
    query = inference_query(row)
    assert "SECRET" not in json.dumps(query)
    row["round1"]["branches"][0]["target_points"] = [[900, 900]]
    row["round2"] = {"poison": "must never read"}
    assert inference_query(row) == query
    assert len(query["references"]) == 2


def test_parsing_does_not_turn_boxes_into_points_or_drop_duplicate_boxes():
    text = "<ref>ship</ref><box><1><2><5><6></box><box><1><2><5><6></box>"
    assert parse_geometry(text, 2) == []
    assert len(parse_geometry(text, 4)) == 2
    assert parse_geometry("<box><1001><2></box>", 2, "ship") == []
    assert parse_geometry("<box>None</box><box>none</box>", 4) == []


def test_summary_is_micro_f1_not_average_per_image_f1():
    rows = [
        dict(
            task="detection",
            tp=1,
            fp=1,
            fn=0,
            seconds=1.0,
            peak_gpu_gib=2,
            predictions=[{}, {}],
            first_iou=None,
        ),
        dict(
            task="detection",
            tp=1,
            fp=0,
            fn=8,
            seconds=3.0,
            peak_gpu_gib=3,
            predictions=[{}],
            first_iou=None,
        ),
    ]
    values = summarize(rows)["detection"]
    assert values["precision"] == pytest.approx(2 / 3)
    assert values["recall"] == pytest.approx(2 / 10)
    assert values["f1"] == pytest.approx(4 / 13)


def fake_python(tmp_path):
    binary = tmp_path / "fake python"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        'with open(os.environ["COMMAND_LOG"], "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
        'sys.exit(int(os.environ.get("FAKE_STATUS", 0)))\n'
    )
    binary.chmod(0o755)
    return dict(
        os.environ,
        PYTHON_BIN=str(binary),
        WORK_ROOT=str(tmp_path / "assets with spaces"),
        COMMAND_LOG=str(tmp_path / "commands.jsonl"),
        SLURM_JOB_NUM_NODES="1",
    )


@pytest.mark.parametrize("gpus", [1, 2, 4, 8])
@pytest.mark.parametrize("mode", ["lora", "full"])
def test_launchers_quote_paths_and_never_call_old_scripts(tmp_path, gpus, mode):
    env = fake_python(tmp_path)
    result = subprocess.run(
        ["bash", str(ROOT / f"train_{mode}.sh"), str(gpus), "stage1"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in Path(env["COMMAND_LOG"]).read_text().splitlines()]
    assert len(commands) == 1
    command = commands[0]
    assert f"--nproc_per_node={gpus}" in command
    assert command[command.index("--tuning") + 1] == mode
    assert command[command.index("--data-root") + 1] == str(Path(env["WORK_ROOT"]) / "dataset")
    assert not any("eagle" in part or "pixel_pivr" in part or "audit" in part for part in command)


def test_failed_stage1_never_launches_stage2(tmp_path):
    env = dict(fake_python(tmp_path), FAKE_STATUS="1")
    result = subprocess.run(
        ["bash", str(ROOT / "train_lora.sh"), "2", "both"], env=env, capture_output=True
    )
    assert result.returncode == 1
    assert len(Path(env["COMMAND_LOG"]).read_text().splitlines()) == 1


def test_evaluation_has_independent_gpu_count(tmp_path):
    env = dict(fake_python(tmp_path), DATA_ROOT=str(tmp_path / "existing data"))
    result = subprocess.run(
        ["bash", str(ROOT / "evaluate.sh"), "2", "baseline"], env=env, capture_output=True
    )
    assert result.returncode == 0
    command = json.loads(Path(env["COMMAND_LOG"]).read_text())
    assert "--nproc_per_node=2" in command
    assert command[command.index("--method") + 1] == "standard"
    assert command[command.index("--data-root") + 1] == env["DATA_ROOT"]
    assert command[command.index("--records") + 1] == str(
        Path(env["DATA_ROOT"]) / "refwave_v1/validation.jsonl"
    )


def test_no_legacy_or_heavy_trainer_imports_in_clean_process():
    code = """
import importlib.abc, importlib.util, sys
blocked = {'pixel_pivr', 'eaglevl', 'deepspeed', 'liger_kernel'}
find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a, **k: None if name.split('.')[0] in blocked else find_spec(name, *a, **k)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ImportError('legacy dependency: ' + fullname)
sys.meta_path.insert(0, Block())
import refwave.train, refwave.evaluate, refwave.export
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
