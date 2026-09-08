"""Opt-in real-model CPU export test; creates and removes only its own temp files."""

import gc
import os
from pathlib import Path
import sys
import tempfile
import pytest
import torch


@pytest.mark.skipif(
    not os.getenv("LA_TEST_EXPORT"), reason="Real-model CPU export requires LA_TEST_EXPORT"
)
def test_lora_and_modified_head_survive_native_export(monkeypatch):
    from refwave import export as cluster_export
    from refwave.model import configure_lora_head
    from refwave.train import SCHEMA, load_base, trainable_state
    from refwave.lora import base_causal_lm

    base = Path(os.environ["LA_TEST_EXPORT"])
    with tempfile.TemporaryDirectory(prefix="refwave-head-export-") as temporary:
        root = Path(temporary)
        (model, processor) = load_base(base, torch.device("cpu"), 6000)
        configure_lora_head(model, 32)
        with torch.no_grad():
            base_causal_lm(model.language_model).lm_head.weight[0, 0] = 5.25
        torch.save(
            dict(
                schema=SCHEMA,
                config=dict(method="refwave", lora_rank=32, image_token_limit=6000),
                step=1,
                optimizer={},
                trainable_state=trainable_state(model),
            ),
            root / "last.pt",
        )
        del model, processor
        gc.collect()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "export",
                "--model",
                str(base),
                "--checkpoint",
                str(root / "last.pt"),
                "--output",
                str(root / "merged"),
            ],
        )
        cluster_export.main()
        gc.collect()
        (restored, processor) = load_base(root / "merged", torch.device("cpu"), 6000)
        assert float(base_causal_lm(restored.language_model).lm_head.weight[0, 0]) == 5.25
        assert not any(("lora_" in name for (name, _) in restored.named_parameters()))
        assert restored.config.refwave_round1_answer_format == "geometry_only"
        assert processor.image_processor.in_token_limit == 6000
        del restored, processor
        gc.collect()
