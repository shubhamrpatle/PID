"""Optional real-model CPU decoding smoke test, never called by launchers."""

import gc
import os
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from refwave.class_wave import ClassPointBranch
from refwave.decoder import AddressedCrop, GeometryDecoder
from refwave.evaluate import CachedImageDecoder
from refwave.train import load_base


@pytest.mark.skipif(
    not os.getenv("LA_TEST_RUNTIME_MODEL"), reason="Optional actual-model decoding smoke test"
)
def test_public_model_direct_class_wave_and_geometry_decode(tmp_path, monkeypatch):
    model, processor = load_base(os.environ["LA_TEST_RUNTIME_MODEL"], torch.device("cpu"), 6000)
    # This CPU-only API test avoids software-emulated BF16 on AVX2 machines.
    model.float().eval()
    image = tmp_path / "synthetic.png"
    Image.new("RGB", (56, 56), (90, 130, 160)).save(image)
    calls = []
    original = model.extract_feature

    def record_encode(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "extract_feature", record_encode)
    worker = SimpleNamespace(
        model=model,
        processor=processor,
        tokenizer=processor.tokenizer,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    points = CachedImageDecoder(worker, image_token_limit=6000, max_wave_size=2)
    generation = dict(
        max_new_tokens=6,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        generation_mode="hybrid",
    )
    with torch.inference_mode():
        direct = ClassPointBranch(
            "ship", "Locate all the instances that match the following description: ship.", 3
        )
        rows, projected = points._prepare_rows(str(image), [direct])
        answer = points._independent_fallback(direct, rows[0], projected, **generation)
        assert isinstance(answer, str) and answer
        branches = [
            ClassPointBranch(c, f"Point to: {c}.", 3 + i) for i, c in enumerate(("ship", "car"))
        ]
        answers, _ = points.generate(str(image), branches, **generation)
        assert len(answers) == 2
        assert all(isinstance(a["answer"], str) for a in answers)

        # Synthetic addresses exercise the API, not localization accuracy.
        addresses = [
            AddressedCrop(
                str(i),
                "ship",
                None,
                f"Locate the single ship containing point <box><{p}><500></box> in horizontal box format. Return None if absent.",
            )
            for i, p in enumerate((400, 600))
        ]
        rows, reused = points._prepare_rows(
            str(image), [ClassPointBranch(a.address_id, a.question, 3) for a in addresses]
        )
        assert reused is projected
        prefix, suffixes = points._split_rows(rows, reused)
        cache = points._encode_prefix(prefix, reused)
        geometry = GeometryDecoder(
            worker,
            image_token_limit=6000,
            prefix_cache_mode="shared",
            geometry_prefix_mode="box_only",
        )
        results = geometry._decode_shared_atomic_blocks(
            torch.tensor(prefix),
            [torch.tensor(s) for s in suffixes],
            [projected[:0] for _ in addresses],
            cache,
            addresses,
            allow_none=True,
        )
        assert len(results) == 2
        assert all(
            r["answer"].startswith("<box>") and r["answer"].endswith("</box>") for r in results
        )
        assert len(calls) == 1
    del geometry, points, worker, model, processor
    gc.collect()
