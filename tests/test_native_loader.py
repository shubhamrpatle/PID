"""Optional CPU checks with actual LA assets; never called by launchers."""

import ast
import copy
import json
import os
import os.path as osp
from pathlib import Path
import re
import subprocess
from typing import Dict

import numpy as np
import pytest
import torch

from refwave.data import ClusterDataset
from refwave.episode import combined_length
from refwave.messages import image_messages
from refwave.native_mtp import NativeMTP6


@pytest.mark.skipif(
    not os.getenv("LA_TEST_EAGLE"), reason="Optional pinned upstream source comparison"
)
def test_vendored_mtp_is_exact_upstream_ast_and_messages_match(tmp_path):
    root = os.environ["LA_TEST_EAGLE"]
    rev = "8442db3b79f7fd2357e468e6eecdd9b6a82049ff"

    def original(path):
        return subprocess.check_output(["git", "-C", root, "show", f"{rev}:{path}"], text=True)

    tree = ast.parse(original("Embodied/eaglevl/train/locany_finetune_magi_stream.py"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LazySupervisedDatasetMTP"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_targets_flag_with_mtp"
    )
    import refwave.native_mtp

    copied = next(
        n
        for n in ast.parse(Path(refwave.native_mtp.__file__).read_text()).body
        if isinstance(n, ast.ClassDef)
    )
    actual = next(
        n for n in copied.body if isinstance(n, ast.FunctionDef) and n.name == method.name
    )
    assert ast.dump(actual) == ast.dump(method)
    tree = ast.parse(original("Embodied/eaglevl/train/tools.py"))
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "process_multimodal_sample"
    )
    namespace = dict(osp=osp, re=re)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<upstream>", "exec"), namespace)
    for question in (
        "Point to ship.",
        "Locate the single ship containing point <box><3><4></box>.",
        "Locate all instances: ship</c>bridge.",
    ):
        for answer in (
            "<box>None</box>",
            "<box><1><2></box>",
            "<ref>ship</ref><box><1><2><3><4></box>",
        ):
            conv = [
                dict(from_="human", value="<image>\n" + question),
                dict(from_="gpt", value=answer),
            ]
            conv = [{"from": r["from_"], "value": r["value"]} for r in conv]
            expected = namespace["process_multimodal_sample"](
                dict(image="images/test.jpg", conversations=copy.deepcopy(conv)),
                str(tmp_path),
                16,
                2,
                1,
            )
            assert image_messages("images/test.jpg", conv, tmp_path) == expected


@pytest.mark.skipif(
    not os.getenv("LA_TEST_RECORD_ROOT"), reason="Optional full-scale records CPU compilation"
)
@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_representative_loader_matches_previously_measured_sequences(stage):
    from transformers import AutoProcessor

    root = Path(os.environ["LA_TEST_RECORD_ROOT"])
    image_root = Path(os.environ["LA_TEST_IMAGE_ROOT"])
    model = os.environ["LA_TEST_MODEL"]
    report = json.loads(
        Path(os.environ["LA_TEST_REPORT_ROOT"], f"refwave_{stage}_loader.json").read_text()
    )
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True, use_fast=False)
    processor.image_processor.in_token_limit = 6000
    for method in ("refwave", "standard"):
        dataset = ClusterDataset(
            root / (stage + ".jsonl"),
            processor,
            data_root=image_root,
            max_sequence=32768,
            maximum_wave_size=200,
            method=method,
        )
        for expected in report["rows"]:
            if expected["method"] != method:
                continue
            sample = dataset[expected["index"]]
            assert sample["episode_id"] == expected["episode_id"]
            assert sample["image_tokens"] == expected["projected_image_tokens"]
            assert len(sample["episode_waves"]) == expected["waves"]
            assert (
                max(combined_length(w)[0] for w in sample["episode_waves"])
                == expected["maximum_sequence"]
            )
            refinements = [b for wave in sample["round2_waves"] for b in wave]
            assert len(refinements) == expected["round2_branches"]
            assert all(int(b["labels"].ne(-100).sum()) == 6 for b in refinements)
