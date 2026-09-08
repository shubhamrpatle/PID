"""Generated HBB evaluation on explicit queries, sharded across allocated GPUs."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from .class_wave import ClassPointBranch
from .hybrid_class_wave import CachedHybridClassWaveDecoder
from .decoder import AddressedCrop, GeometryDecoder
from .episode import IndexedJsonl, box_text
from .geometry import canonical_label, detection_counts, precision_recall_f1, hbox_iou
from .io import atomic_json, append_jsonl, read_jsonl_log, sha256_file
from .model import configure_lora_head, configure_full
from .runtime import distributed_context
from .train import SCHEMA, load_base, restore

GEOMETRY = re.compile(r"<ref>(.*?)</ref>|<box>(.*?)</box>", re.DOTALL)
COORDINATES = re.compile(r"<(\d+)>")


def parse_geometry(answer, dimensions, default_label=""):
    label, result = canonical_label(default_label), []
    for match in GEOMETRY.finditer(answer):
        if match.group(1) is not None:
            label = canonical_label(match.group(1))
            continue
        body = match.group(2).strip()
        if body in ("None", "none"):
            continue
        values = [int(v) for v in COORDINATES.findall(body)]
        if (
            len(values) != dimensions
            or any(v > 1000 for v in values)
            or COORDINATES.sub("", body).strip()
        ):
            continue
        if dimensions == 4 and (values[0] >= values[2] or values[1] >= values[3]):
            continue
        result.append(dict(label=label, **{"point" if dimensions == 2 else "hbox": values}))
    return result


def inference_query(row):
    """Do not pass targets, GT points, polarity or stored Round-2 branches to inference."""
    return dict(
        image=row["image"],
        task=row["task"],
        direct_question=row["direct"]["conversations"][0]["value"].replace("<image>", "").strip(),
        references=[
            dict(
                reference=b["reference"],
                question=b["conversations"][0]["value"].replace("<image>", "").strip(),
            )
            for b in row["round1"]["branches"]
        ],
    )


def ground_truth(row):
    return [
        dict(label=canonical_label(group["reference"]), hbox=instance["geometry_global"])
        for group in row["target_groups"]
        for instance in group["instances"]
    ]


class CachedImageDecoder(CachedHybridClassWaveDecoder):
    """Reuse vision across the two rounds; Qwen prefix is rebuilt per round."""

    def clear_image(self):
        self.cached_image = None
        self.projected = None

    def _prepare_rows(self, image, branches):
        if getattr(self, "cached_image", None) != image:
            rows, projected = super()._prepare_rows(image, branches)
            self.cached_image, self.projected = image, projected
            self.visual_tokens = rows[0].count(self.image_token_id)
            return rows, projected
        placeholder = (
            "<image 1>"
            + self.processor.image_start_token
            + self.processor.image_token * self.visual_tokens
            + self.processor.image_end_token
        )
        rows = []
        for branch in branches:
            text = self.processor.py_apply_chat_template(
                self._message(image, branch.question), tokenize=False, add_generation_prompt=True
            )
            if text.count("<image-1>") != 1:
                raise ValueError("Expected one image in inference prompt")
            rows.append(
                self.tokenizer(text.replace("<image-1>", placeholder), add_special_tokens=True)[
                    "input_ids"
                ]
            )
        return rows, self.projected


class Evaluator:
    def __init__(self, model, processor, args, device):
        worker = SimpleNamespace(
            model=model,
            processor=processor,
            tokenizer=processor.tokenizer,
            device=device,
            dtype=torch.bfloat16,
        )
        self.points = CachedImageDecoder(
            worker,
            image_token_limit=args.image_token_limit,
            max_wave_size=args.class_wave,
            bucket_suffix_lengths=False,
        )
        self.geometry = GeometryDecoder(
            worker,
            image_token_limit=args.image_token_limit,
            prefix_cache_mode="shared",
            geometry_prefix_mode="box_only",
        )
        self.args, self.device = args, device

    @torch.no_grad()
    def predict(self, image, query, seed):
        self.points.clear_image()
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        generation = dict(
            max_new_tokens=self.args.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            generation_mode="hybrid",
        )
        if self.args.method == "standard":
            branch = ClassPointBranch("direct", query["direct_question"], seed)
            rows, projected = self.points._prepare_rows(image, [branch])
            answer = self.points._independent_fallback(branch, rows[0], projected, **generation)
            default = query["references"][0]["reference"] if len(query["references"]) == 1 else ""
            predictions = parse_geometry(answer, 4, default)
            raw, points, round1_seconds = [dict(answer=answer)], [], None
        else:
            branches = [
                ClassPointBranch(b["reference"], b["question"], seed + i)
                for i, b in enumerate(query["references"])
            ]
            answers, stats = self.points.generate(image, branches, **generation)
            points, seen = [], set()
            for branch, answer in zip(branches, answers):
                for point in parse_geometry(answer["answer"], 2, branch.label):
                    # Reference identity belongs to the input branch, not output text.
                    point["label"] = canonical_label(branch.label)
                    point["reference"] = branch.label
                    key = (point["label"], *point["point"])
                    if key not in seen:
                        seen.add(key)
                        points.append(point)
            torch.cuda.synchronize(self.device)
            round1_seconds = time.perf_counter() - started
            raw, predictions = [dict(round1=answers, stats=stats)], []
            addresses = [
                AddressedCrop(
                    str(i),
                    p["label"],
                    None,
                    f"Locate the single {p['reference']} containing point {box_text(p['point'])} "
                    "in horizontal box format. Return None if absent.",
                )
                for i, p in enumerate(points)
            ]
            if addresses:
                r2 = [ClassPointBranch(a.address_id, a.question, seed) for a in addresses]
                rows, projected = self.points._prepare_rows(image, r2)
                prefix, suffixes = self.points._split_rows(rows, projected)
                prefix_kv = self.points._encode_prefix(prefix, projected)
                for start in range(0, len(addresses), self.args.box_wave):
                    group = addresses[start : start + self.args.box_wave]
                    output = self.geometry._decode_shared_atomic_blocks(
                        torch.tensor(prefix, device=self.device),
                        [
                            torch.tensor(s, device=self.device)
                            for s in suffixes[start : start + len(group)]
                        ],
                        [projected[:0] for _ in group],
                        prefix_kv,
                        group,
                        allow_none=True,
                    )
                    for result in output:
                        predictions.extend(parse_geometry(result["answer"], 4, result["label"]))
                    raw.append(dict(round2=output))
        torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(self.device) / 1024**3
        self.points.clear_image()
        return dict(
            predictions=predictions,
            predicted_points=points,
            raw=raw,
            seconds=seconds,
            round1_seconds=round1_seconds,
            round2_seconds=seconds - round1_seconds if round1_seconds is not None else None,
            peak_gpu_gib=peak,
        )


def summarize(rows):
    result = {}
    for task in ("detection", "grounding"):
        subset = [r for r in rows if r["task"] == task]
        if not subset:
            continue
        tp, fp, fn = [sum(r[k] for r in subset) for k in ("tp", "fp", "fn")]
        seconds = [r["seconds"] for r in subset]
        metrics = dict(
            queries=len(subset),
            tp=tp,
            fp=fp,
            fn=fn,
            **precision_recall_f1(tp, fp, fn),
            latency_mean_seconds=float(np.mean(seconds)),
            latency_p50_seconds=float(np.median(seconds)),
            latency_p95_seconds=float(np.percentile(seconds, 95)),
            peak_gpu_gib=max(r["peak_gpu_gib"] for r in subset),
            predicted_boxes_per_second=sum(len(r["predictions"]) for r in subset)
            / max(sum(seconds), 1e-9),
        )
        single = [r["first_iou"] for r in subset if r["first_iou"] is not None]
        if task == "grounding" and single:
            metrics.update(
                single_gt_queries=len(single),
                first_acc_0_5=float(np.mean(np.array(single) >= 0.5)),
                first_acc_0_7=float(np.mean(np.array(single) >= 0.7)),
                first_mean_iou=float(np.mean(single)),
            )
        result[task] = metrics
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "data-root", "records", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--method", choices=("refwave", "standard"), default="refwave")
    p.add_argument("--class-wave", type=int, default=18)
    p.add_argument("--box-wave", type=int, default=200)
    p.add_argument("--image-token-limit", type=int, default=6000)
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260908)
    args = p.parse_args()
    if (
        not (1 <= args.class_wave <= 200 and 1 <= args.box_wave <= 200)
        or args.limit < 0
        or args.max_new_tokens <= 0
    ):
        p.error("Wave sizes must be 1-200; limit must be nonnegative")
    rank, local, world = distributed_context()
    device = torch.device("cuda", local)
    model, processor = load_base(args.model, device, args.image_token_limit)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if saved.get("schema") != SCHEMA or saved["config"]["method"] != args.method:
            raise ValueError("Evaluation method does not match this checkpoint")
        if saved["config"]["tuning"] == "full":
            configure_full(model)
        else:
            configure_lora_head(model, saved["config"]["lora_rank"])
        restore(model, saved["trainable_state"])
        del saved
    model.to(dtype=torch.bfloat16).eval()
    rows = IndexedJsonl(args.records)
    count = min(len(rows), args.limit) if args.limit else len(rows)
    contract = dict(
        model=str(args.model.resolve()),
        checkpoint_sha256=sha256_file(args.checkpoint) if args.checkpoint else None,
        records_sha256=sha256_file(args.records),
        records=count,
        method=args.method,
        world_size=world,
        class_wave=args.class_wave,
        box_wave=args.box_wave,
        image_token_limit=args.image_token_limit,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        iou=0.5,
        nms=False,
        max_detections=None,
        note="Generated validation-query evaluation; not an official full benchmark unless supplied records implement it.",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    config = args.output / "config.json"
    if config.exists() and json.loads(config.read_text()) != contract:
        raise ValueError("Output belongs to another evaluation; choose a new EVAL_OUTPUT")
    if rank == 0:
        atomic_json(config, contract)
    shard = args.output / f"predictions.rank{rank}.jsonl"
    done = {row["index"] for row in read_jsonl_log(shard, repair_tail=True)}
    evaluator = Evaluator(model, processor, args, device)
    for index in range(rank, count, world):
        if index in done:
            continue
        row = rows[index]
        query = inference_query(row)
        image = (args.data_root / query["image"]).resolve()
        if not image.is_relative_to(args.data_root.resolve()) or not image.is_file():
            raise FileNotFoundError(f"Invalid or missing evaluation image: {image}")
        seed = args.seed + int(hashlib.sha256(row["episode_id"].encode()).hexdigest()[:8], 16)
        output = evaluator.predict(str(image), query, seed)
        gt = ground_truth(row)
        tp, fp, fn = detection_counts(gt, output["predictions"])
        first = (
            (
                hbox_iou(output["predictions"][0]["hbox"], gt[0]["hbox"])
                if output["predictions"]
                else 0.0
            )
            if len(gt) == 1
            else None
        )
        append_jsonl(
            shard,
            dict(
                index=index,
                episode_id=row["episode_id"],
                task=row["task"],
                gt=gt,
                tp=tp,
                fp=fp,
                fn=fn,
                first_iou=first,
                **output,
            ),
        )
        print(f"[rank {rank}] completed query {index + 1}/{count}", flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        results = [
            row
            for r in range(world)
            if (args.output / f"predictions.rank{r}.jsonl").exists()
            for row in read_jsonl_log(args.output / f"predictions.rank{r}.jsonl")
        ]
        if sorted(r["index"] for r in results) != list(range(count)):
            raise RuntimeError("Evaluation output is incomplete or duplicated")
        atomic_json(
            args.output / "metrics.json", dict(protocol=contract, results=summarize(results))
        )
        atomic_json(args.output / "done.json", dict(queries=count))
        print(json.dumps(summarize(results), indent=2), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
