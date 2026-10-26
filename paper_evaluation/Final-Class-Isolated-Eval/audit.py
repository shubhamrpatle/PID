"""Audit the supplied training evidence without loading weights or using a GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "CaseStudyRef/RefWave-Cluster-Runs"
REVISION = "266d967359913eeaa9e184651e787b67341d744b"
SPEC = dict(
    run="class_isolated_full_8gpu_s1e2_s2e2_both_both_ft_ciso_ep2_v32_20260917T115718Z_TvBPX9",
    sha256="eb662db4078042c39eb55b8d8760472bb053fc4d372c87ec613426a9e8750185",
    bytes=42216772310, step=1246,
    parent_sha256="a4462be8d4fce6832cdfd8789c4f1c95bc0dc5f99a60943a99470f18c1a6f86d")
DATA_HASHES = dict(
    stage1="2fd6c5e8df294d968fbe5b1381fadc5eeb795ea3b1fc1848627e480064c8ae52",
    stage2="c5d0e3156f2c3d55f9fddc573fcbdeedb5a1427ae80314b2fc45fb37ee140bd7",
    validation="00e14c17087bfb8bddd4299b3f3d663138658783d200c3534a4bacd57ea58a91")
MATCH_KEYS = ("tuning", "epochs", "records", "steps", "global_batch", "world_size",
    "batch_size", "accumulation", "batch_tokens", "max_sequence", "image_token_limit",
    "seed", "learning_rate", "head_lr", "vision_lr", "projector_lr", "warmup_ratio",
    "train_sha256", "validation_sha256", "validation_records", "trainable",
    "optimizer_sharding", "synchronized_trainable_initialization")
FILES = tuple(f"{stage}/{name}" for stage in ("stage1", "stage2") for name in
    ("config.json", "done.json", "validation_curve.jsonl", "training_curve.jsonl"))


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(directory, references=None):
    configs, completions, curves, stages = {}, {}, {}, {}
    for stage, records in (("stage1", 116386), ("stage2", 39851)):
        folder = Path(directory)/stage
        c, done = read(folder/"config.json"), read(folder/"done.json")
        train, val = jsonl(folder/"training_curve.jsonl"), jsonl(folder/"validation_curve.jsonl")
        if (c["method"], c["tuning"], c["stage"], c["epochs"], c["world_size"],
                c["records"], c["global_batch"], c["validation_records"]) != (
                "class_isolated", "full", stage, 2, 8, records, 32, 740):
            raise ValueError(f"Unexpected training configuration: {stage}")
        if (c["train_sha256"], c["validation_sha256"]) != (DATA_HASHES[stage], DATA_HASHES["validation"]):
            raise ValueError(f"Training/validation hash mismatch: {stage}")
        expected = 2 * math.ceil(records / c["global_batch"])
        if (c["steps"], done["steps"], done["records"], done["epochs"], done["total_records"]) != (
                expected, expected, 2*records, 2.0, 2*records):
            raise ValueError(f"Incomplete two-epoch schedule: {stage}")
        if [r["step"] for r in train] != list(range(1, expected+1)):
            raise ValueError(f"Missing/repeated training updates: {stage}")
        if not val or [r["step"] for r in val] != sorted({r["step"] for r in val}):
            raise ValueError(f"Missing/repeated validation observations: {stage}")
        if val[-1]["step"] != expected:
            raise ValueError(f"Missing final validation: {stage}")
        for row in train:
            epoch, offset = divmod(row["step"]-1, expected//2)
            seen = epoch*records + min((offset+1)*c["global_batch"], records)
            if row["records_seen"] != seen:
                raise ValueError(f"Incorrect recorded source-episode exposure: {stage}")
        for row in train + val:
            if any(isinstance(x, float) and not math.isfinite(x) for x in row.values()):
                raise ValueError(f"Nonfinite logged value: {stage}")
            if row["round1_loss"] != 0 or row["round2_loss"] != 0 or not math.isclose(
                    row["loss"], row["direct_loss"], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Unexpected non-direct objective: {stage}")
        if train[-1]["records_seen"] != 2*records or train[-1]["epoch"] != 2.0:
            raise ValueError(f"Final training cursor differs: {stage}")
        if not c["synchronized_trainable_initialization"] or c["image_token_limit"] != 6000:
            raise ValueError(f"Unexpected initialization or image budget: {stage}")
        best = min(val, key=lambda r: r["detection_grounding_macro_loss"])
        if best["step"] != (7276 if stage == "stage1" else SPEC["step"]):
            raise ValueError(f"Validation-selected checkpoint differs: {stage}")
        configs[stage], completions[stage], curves[stage] = c, done, val
        stages[stage] = dict(records=records, epochs=2, steps=expected, training_rows=len(train),
            validation_rows=len(val), selected_step=best["step"],
            selected_macro_loss=best["detection_grounding_macro_loss"],
            last_macro_loss=val[-1]["detection_grounding_macro_loss"],
            observed_nonfinite_values=0, direct_only_logged_objective=True)
    if configs["stage2"]["init_checkpoint_sha256"] != SPEC["parent_sha256"]:
        raise ValueError("Stage 2 did not initialize from the pinned Stage-1 best")
    matches = {}
    for name, ref in (references or {}).items():
        differences = {f"{stage}.{key}": [ref["configs"][stage].get(key), configs[stage].get(key)]
            for stage in configs for key in MATCH_KEYS
            if ref["configs"][stage].get(key) != configs[stage].get(key)}
        if differences:
            raise ValueError(f"Training recipe differs from {name}: {differences}")
        matches[name] = list(MATCH_KEYS)
    evidence = dict(repo=REPO, revision=REVISION, checkpoint=SPEC, stages=stages,
        full_trainable_parameters=configs["stage2"]["trainable"]["trainable_parameters"],
        data_hashes=DATA_HASHES, matched_recipe_fields=matches,
        source_hashes={p:sha(Path(directory)/p) for p in FILES},
        limitations=["Logs/configs establish recorded execution, not semantic correctness of every annotation.",
            "No executed training-source snapshot was supplied; exact remote implementation is not independently proven.",
            "The full weight file and GPU inference are verified on the evaluation machine before full evaluation.",
            "Matched source episodes and optimizer schedule do not equate supervised token counts across formulations."])
    return evidence, configs, completions, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-selection", type=Path)
    args = parser.parse_args()
    references = read(args.reference_selection) if args.reference_selection else None
    evidence, _, _, _ = inspect(HERE/"metadata/runs"/SPEC["run"], references)
    (HERE/"TRAINING_AUDIT.json").write_text(json.dumps(evidence, indent=2)+"\n")
    lines = ["# Corrected class-isolated checkpoint: training evidence", "",
        f"HF revision: `{REVISION}`. Run: `{SPEC['run']}`.", "",
        "| Stage | Source records | Epochs | Updates | Selected update | Selected macro loss |",
        "|---|---:|---:|---:|---:|---:|"]
    for stage, v in evidence["stages"].items():
        lines.append(f"| {stage} | {v['records']} | 2 | {v['steps']} | {v['selected_step']} | {v['selected_macro_loss']:.9f} |")
    lines += ["", "Both complete training curves contain every update, finite logged values, and direct-only loss.",
        "Full MoonViT + projector + Qwen + LM head: 3,517,975,280 trainable parameters, as recorded in configs.",
        f"Stage-2 best is update 1246 (macro validation loss {evidence['stages']['stage2']['selected_macro_loss']:.9f}), "
        f"not last update 2492 ({evidence['stages']['stage2']['last_macro_loss']:.9f}).",
        "Stage-2 initialization hash equals the pinned Stage-1 best checkpoint hash.", "",
        "Recipe compared to: "+(", ".join(matches for matches in evidence["matched_recipe_fields"]) or "not supplied")+".",
        "Exact compared fields, training/validation hashes and evidence hashes are in TRAINING_AUDIT.json.", "",
        "## Scope of verification", "", *["- "+x for x in evidence["limitations"]], ""]
    (HERE/"TRAINING_AUDIT.md").write_text("\n".join(lines))
    print(json.dumps(evidence["stages"], indent=2))


if __name__ == "__main__":
    main()
