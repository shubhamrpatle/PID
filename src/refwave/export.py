"""Merge a trusted cluster LoRA+head checkpoint into a portable LA model."""

import argparse
from pathlib import Path
import shutil
import torch
from .model import configure_lora_head, configure_full
from .train import SCHEMA, load_base, restore
from .io import atomic_json, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Export needs a new output directory")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if saved.get("schema") != SCHEMA:
        raise ValueError("Expected a cluster LoRA+head checkpoint")
    (model, processor) = load_base(
        args.model, torch.device("cpu"), saved["config"]["image_token_limit"]
    )
    tuning = saved["config"].get("tuning", "lora")
    if tuning == "full":
        configure_full(model)
    else:
        configure_lora_head(model, saved["config"]["lora_rank"])
    restore(model, saved["trainable_state"])
    del saved["optimizer"]
    if tuning == "lora":
        model.language_model = model.language_model.merge_and_unload()
    if hasattr(model.config, "use_llm_lora"):
        model.config.use_llm_lora = 0
    model.config.text_config = model.language_model.config
    model.config.refwave_round1_answer_format = "geometry_only"
    model.config.refwave_objective_mode = (
        "direct_only" if saved["config"]["method"] == "standard" else "two_route_equal_mean"
    )
    model.language_model.config.use_cache = True
    model.to(dtype=torch.bfloat16).eval()
    args.output.mkdir(parents=True)
    for source in args.model.rglob("*.py"):
        if ".cache" in source.parts or ".git" in source.parts:
            continue
        destination = args.output / source.relative_to(args.model)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="4GB")
    processor.save_pretrained(args.output)
    for name in ("chat_template.json", "generation_config.json"):
        if (args.model / name).exists():
            shutil.copy2(args.model / name, args.output / name)
    atomic_json(
        args.output / "refwave_export.json",
        dict(
            schema=SCHEMA,
            method=saved["config"]["method"],
            step=saved["step"],
            checkpoint_sha256=sha256_file(args.checkpoint),
            full_head_included=True,
        ),
    )
    print("Exported", tuning, "model:", args.output)


if __name__ == "__main__":
    main()
