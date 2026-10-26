#!/usr/bin/env python3
"""Evaluate the corrected, data-matched class-isolated full-FT checkpoint."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import importlib.metadata
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import audit

HERE = Path(__file__).resolve().parent
PID_DIR = Path(os.environ.get("PID_DIR", HERE.parent/"Final-RefWave-Eval")).resolve()
OUTPUT = Path(os.environ.get("CLASS_ISOLATED_OUTPUT", HERE/"results")).resolve()
DOWNLOAD = Path(os.environ.get("CLASS_ISOLATED_CHECKPOINTS", HERE/"checkpoints")).resolve()
os.environ.setdefault("EVAL_HF_HOME", str(HERE/".hf-cache"))
sys.path.insert(0, str(PID_DIR))
import remote_eval as previous
import dynamic_queue as scheduler

if Path(previous.__file__).resolve() != PID_DIR/"remote_eval.py":
    raise RuntimeError("Wrong remote_eval.py imported; set PID_DIR to Final-RefWave-Eval")
evaluation, base, queue, release = previous.evaluation, previous.base, previous.queue, previous.release
CONTROLS, PID_OUTPUT = previous.CONTROLS, previous.OUTPUT
REPO, REVISION, SPEC = audit.REPO, audit.REVISION, audit.SPEC
MODEL = "class_isolated"
PROTOCOL = "corrected_class_isolated_direct_v1"
EXPECTED = previous.EXPECTED
GENERATION = dict(max_new_tokens=8192, temperature=.7, top_p=.9,
    repetition_penalty=1.1, generation_mode="hybrid")
release.REPO, release.REVISION, release.DOWNLOAD = REPO, REVISION, DOWNLOAD
release.SPECS = {MODEL: SPEC}
evaluation.METHODS = (MODEL,)
read, write = audit.read, base.common.write


def software():
    import torch
    from pixel_pivr.worker import choose_vision_attention
    result = dict(torch=str(torch.__version__), python=sys.version, cuda=torch.version.cuda,
        vision_attention=choose_vision_attention(), text_attention="sdpa")
    for name in ("transformers", "flash-attn", "numpy", "pillow"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def metadata(references):
    from huggingface_hub import HfApi, hf_hub_download
    prefix = f"runs/{SPEC['run']}/"
    paths = [prefix+"stage2/best.pt", prefix+"stage2/checkpoint-0001246.pt", prefix+"stage1/best.pt"]
    info = {p.path:p for p in HfApi(token=False).get_paths_info(
        REPO, paths=paths, repo_type="dataset", revision=REVISION)}
    for path, expected in zip(paths, (SPEC["sha256"], SPEC["sha256"], SPEC["parent_sha256"])):
        if path not in info or info[path].lfs.sha256 != expected:
            raise ValueError("Pinned remote parent/best/numbered checkpoint identity differs")
    if info[paths[0]].size != SPEC["bytes"]:
        raise ValueError("Pinned checkpoint size differs")
    root = HERE/"metadata"
    trusted = read(HERE/"TRAINING_AUDIT.json")["source_hashes"]
    for filename in audit.FILES:
        path = root/prefix/filename
        if not path.exists():
            hf_hub_download(REPO, prefix+filename, repo_type="dataset", revision=REVISION,
                local_dir=root, token=False)
        if audit.sha(path) != trusted[filename]:
            raise ValueError("Training evidence differs from the reviewed HF revision: "+filename)
    evidence, configs, completions, curves = audit.inspect(root/prefix, references)
    best = release.validate_metadata(MODEL, configs, completions, curves)
    return dict(SPEC, method=MODEL, repo=REPO, revision=REVISION,
        checkpoint=str(release.checkpoint(MODEL)), configs=configs, completions=completions,
        curves=curves, validation=best, criterion="minimum validation detection_grounding_macro_loss"), evidence


def validate_metric(metric, kind, suite, manifest_sha, selected, hardware):
    c = metric["contract"]
    if (not metric["complete"] or metric["records"] != EXPECTED[suite] or c["limit"] != 0
            or c["manifest_sha256"] != manifest_sha or c["checkpoint_sha256"] != selected["sha256"]
            or c["step"] != selected["step"] or c["generation"] != GENERATION
            or c["method"] != selected["method"]
            or c["image_token_limit"] != 6000 or c["added_nms"] or c["max_detections"] is not None
            or c["shard_count"] != 2 or c["warmup_records"] != 2
            or c["suite"] != suite or c["kind"] != kind
            or (c["gpu"], c["torch"]) != (hardware["gpu_model"], hardware["torch"])):
        raise ValueError(f"Incomplete or incompatible result: {suite}, {kind}")
    if kind == "boxes" and suite in ("DIOR", "DOTAv2") and len(metric["raw"]) != 10:
        raise ValueError("Detection requires all ten IoU thresholds")
    if selected["method"] == "refwave" and (
            c["protocol"] != previous.pid.PROTOCOL_ID
            or (kind == "boxes" and c["round2"] != "packed_all_one_forward")):
        raise ValueError("Historical PID is not the final packed-all protocol")
    if selected["method"] == MODEL and (c.get("class_isolated_max_concurrent_references") != 32
            or c.get("native_query_batch") != 1):
        raise ValueError("Class-isolated/native pointing execution differs from the frozen control")


def prepare():
    for old in (CONTROLS, PID_DIR):
        if OUTPUT == old or OUTPUT.is_relative_to(old) or old.is_relative_to(OUTPUT):
            raise ValueError("CLASS_ISOLATED_OUTPUT must be separate from existing installations")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DOWNLOAD.mkdir(parents=True, exist_ok=True)
    signatures = previous.verify_runtime()
    references, archives = {}, {}
    hardware = scheduler.control_hardware()
    current_software = software()
    if current_software["torch"] != hardware["torch"]:
        raise ValueError("Use the same evaluation environment/Torch version as the completed controls")
    for model, root in (("standard", CONTROLS/"results"), ("refwave", PID_OUTPUT)):
        selected = read(root/"selection.json")[model]
        references[model] = selected
        for kind in ("boxes", "pointing"):
            for suite in EXPECTED:
                path = root/"results"/kind/model/suite/"metrics.json"
                if path.is_file():
                    archives[str(path)] = audit.sha(path)
    # This helper validates and copies the identical controls manifests, without rewriting paths.
    saved_output = previous.OUTPUT
    try:
        previous.OUTPUT = OUTPUT
        manifests = previous.copy_manifests()
    finally:
        previous.OUTPUT = saved_output
    for path in archives:
        metric = read(path)
        c = metric["contract"]
        validate_metric(metric, c["kind"], c["suite"], manifests[c["suite"]]["sha256"],
            references[c["method"]], hardware)
    selected, evidence = metadata(references)
    base.freeze(OUTPUT/"selection.json", {MODEL:selected})
    base.freeze(OUTPUT/"training_audit.json", evidence)
    for path in (HERE/"evaluate.py", HERE/"audit.py", HERE/"TRAINING_AUDIT.json",
            PID_DIR/"remote_eval.py", PID_DIR/"dynamic_queue.py", PID_DIR/"runtime_hashes.json"):
        signatures[str(path)] = audit.sha(path)
    signatures.update(archives)
    plan = dict(protocol=PROTOCOL, repo=REPO, revision=REVISION, models={MODEL:SPEC},
        manifests=manifests, sources=signatures, historical_selections=references, archives=archives,
        generation=GENERATION, image_token_limit=6000, added_nms=False, max_detections=None,
        shards=2, warmup_records_per_shard=2, hardware=hardware, software=current_software,
        timing="synchronized preprocessing + generation + parsing, excluding disk IO, load, warmup and scoring",
        isolated_detection="All 20 DIOR / 18 DOTAv2 category queries concurrently, direct boxes, shared prefix",
        grounding="Full original expression; native direct-box generation; first-box metrics",
        pointing="Native point queries, sequential per reference, unchanged Standard/control protocol",
        training_comparison="Same source/validation hashes, full scope, epochs, optimizer schedule and seed",
        comparison_limits=["Reference separation changes total available output budget and supervised token counts.",
            "Pointing is a native diagnostic on localization annotations, not an official pointing benchmark.",
            "Shared-server timings match GPU model/Torch but do not control all node-level contention.",
            "The new trained class-isolated model is not the inference-only per-reference Standard ablation."])
    base.freeze(OUTPUT/"plan.json", plan)
    verify_checkpoint(selected)
    print("Prepared corrected Stage-2 best 1246; frozen benchmarks unchanged.", flush=True)


def verify_checkpoint(selected):
    import torch
    from huggingface_hub import hf_hub_download
    path = release.checkpoint(MODEL)
    ready = OUTPUT/"checkpoints_ready.json"
    if ready.is_file() and path.is_file():
        ready_info = read(ready)
        info = ready_info["models"][MODEL]
        if (ready_info.get("complete") and info.get("verified") and info["sha256"] == SPEC["sha256"] and info["step"] == SPEC["step"]
                and info["path"] == str(path) and info["bytes"] == SPEC["bytes"]
                and (path.stat().st_size, path.stat().st_mtime_ns) == (info["bytes"], info["mtime_ns"])):
            return
    if not path.exists():
        if shutil.disk_usage(DOWNLOAD).free < SPEC["bytes"] + 8*1024**3:
            raise RuntimeError("Need approximately 51 GB free for one checkpoint plus download headroom")
        print("Downloading only Stage-2 best.pt (42.2 GB); no Stage-1 or last weights.", flush=True)
        hf_hub_download(REPO, f"runs/{SPEC['run']}/stage2/best.pt", repo_type="dataset",
            revision=REVISION, local_dir=DOWNLOAD, token=False)
    print("Checking checkpoint SHA256 and embedded training state.", flush=True)
    if path.stat().st_size != SPEC["bytes"] or audit.sha(path) != SPEC["sha256"]:
        raise ValueError("Checkpoint hash/size mismatch; existing file preserved")
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    tied = read(base.BASE/"config.json")["text_config"].get("tie_word_embeddings", False)
    best, components = release.validate_saved(saved, MODEL, selected["configs"],
        selected["completions"], selected["curves"], tied)
    info = dict(path=str(path), sha256=SPEC["sha256"], bytes=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns, step=best["step"], components=components, verified=True)
    del saved
    gc.collect()
    write(OUTPUT/"class_isolated.checkpoint_verified.json", info)
    write(ready, dict(complete=True, models={MODEL:info}))


def check_plan():
    plan = read(OUTPUT/"plan.json")
    if plan["protocol"] != PROTOCOL or plan["models"] != {MODEL:SPEC}:
        raise ValueError("Different experiment in output directory")
    evaluation.check_sources(plan)
    return plan


def infer(args):
    import torch
    plan = check_plan()
    if software() != plan["software"]:
        raise ValueError("Evaluation environment changed after preparation")
    if (torch.cuda.get_device_name(0), str(torch.__version__)) != (
            plan["hardware"]["gpu_model"], plan["hardware"]["torch"]):
        raise ValueError("GPU/Torch differs from completed controls")
    # Reuse the unchanged trained class-isolated/native grounding/pointing implementation.
    evaluation.infer(args)


def merge(args):
    plan = check_plan()
    directory = base.folder(args)
    for i in range(args.shard_count):
        c = read(directory/f"shard{i:02d}of{args.shard_count:02d}.contract.json")
        if (c["protocol"] != PROTOCOL or c["method"] != MODEL
                or c["plan_sha256"] != audit.sha(OUTPUT/"plan.json")
                or c["checkpoint_sha256"] != SPEC["sha256"]
                or c["manifest_sha256"] != plan["manifests"][args.suite]["sha256"]):
            raise ValueError("Stale or different checkpoint/manifest shard")
    base.merge(args)


def report():
    plan = check_plan()
    cells, lines = [], ["# Corrected class-isolated comparison", "",
        "Detection: F1@.50 / F1@.75 / mean F1 across ten IoUs .50:.05:.95.",
        "Grounding: first-box Acc@.5 / Acc@.7 / mIoU. Pointing: point-inside-HBB F1.",
        "Percentages; seconds per complete image or expression. No NMS or evaluator top-K.", "",
        "| Model | Task | Benchmark | Records | Scores (%) | Mean s/record | Outputs/s | Correct outputs/s @.5 |",
        "|---|---|---|---:|---|---:|---:|---:|"]
    for model, root in (("standard", CONTROLS/"results"), ("refwave", PID_OUTPUT), (MODEL, OUTPUT)):
        selected = (read(OUTPUT/"selection.json")[MODEL] if model == MODEL
                    else plan["historical_selections"][model])
        for kind in ("boxes", "pointing"):
            for suite, count in EXPECTED.items():
                path = root/"results"/kind/model/suite/"metrics.json"
                values = ["-"]*4
                if path.exists():
                    m = read(path)
                    validate_metric(m, kind, suite, plan["manifests"][suite]["sha256"], selected, plan["hardware"])
                    if model == MODEL and (m["contract"]["protocol"] != PROTOCOL or
                            m["contract"]["plan_sha256"] != audit.sha(OUTPUT/"plan.json")):
                        raise ValueError("New metrics use a different plan")
                    for filename, digest in m["prediction_files"].items():
                        if audit.sha(path.parent/filename) != digest:
                            raise ValueError("Predictions changed after scoring: "+str(path))
                    if kind == "pointing":
                        scores = [m["pointing"]["f1"]]
                        bps, cbps = m["points_per_second"], m["correct_points_per_second"]
                    elif suite in ("DIOR", "DOTAv2"):
                        scores = [m["raw"]["0.5"]["f1"], m["raw"]["0.75"]["f1"], m["mean_f1_iou_0p50_to_0p95"]]
                        bps, cbps = m["bps"], m["cbps_iou0p5"]
                    else:
                        scores = [m["grounding"][k] for k in ("first_acc_0_5", "first_acc_0_7", "first_mean_iou")]
                        bps, cbps = None, scores[0]/m["mean_seconds"]
                    values = [" / ".join(f"{x*100:.2f}" for x in scores), f"{m['mean_seconds']:.3f}",
                        "-" if bps is None else f"{bps:.3f}", f"{cbps:.3f}"]
                    cells.append(dict(model=model, kind=kind, suite=suite, metrics=m, path=str(path)))
                task = "pointing" if kind == "pointing" else ("detection" if suite in ("DIOR", "DOTAv2") else "grounding")
                label = "class-isolated (corrected, trained)" if model == MODEL else model
                lines.append("| "+" | ".join([label, task, suite, str(count), *values])+" |")
    lines += ["", "Standard/PID rows reuse completed frozen results, not newly measured timings.",
        "Grounding correct outputs/s counts correct first-box answers; no boxes/s is invented.",
        "Pointing correct outputs/s uses point-inside-HBB matching, not IoU.",
        "Native control pointing is sequential per reference; PID pointing is concurrent.",
        "Matched training recipe hashes are recorded in training_audit.json.", "", *plan["comparison_limits"], ""]
    (OUTPUT/"comparison.md").write_text("\n".join(lines))
    completed = sum(c["model"] == MODEL for c in cells)
    write(OUTPUT/"comparison.json", dict(complete=completed == 8,
        new_completed_cells=completed, expected_new_cells=8, cells=cells))
    print(f"Corrected class-isolated: {completed}/8 completed result cells.", flush=True)
    return completed


def jobs(smoke=False):
    return [dict(model=MODEL, kind=kind, suite=suite, shard_index=i,
        shard_count=1 if smoke else 2, limit=2 if smoke else 0)
        for kind in ("boxes", "pointing") for suite in EXPECTED for i in range(1 if smoke else 2)]


def command(output, job, mode):
    cmd = [sys.executable, "-u", str(HERE/"evaluate.py"), "worker", mode]
    for key in ("model", "kind", "suite", "shard_index", "shard_count", "limit"):
        cmd += ["--"+key.replace("_", "-"), str(job[key])]
    return cmd


def environment(model, gpu=None):
    env = previous.environment(model, gpu)
    env.update(PID_DIR=str(PID_DIR), CLASS_ISOLATED_OUTPUT=str(OUTPUT), CLASS_ISOLATED_CHECKPOINTS=str(DOWNLOAD))
    return env


@contextmanager
def queue_adapter():
    old = queue.command, queue.jobs, queue.scheduling.environment, queue.write_status
    last = [None, 0.]
    def status(output, state, **kwargs):
        old[3](output, state, **kwargs)
        current = state, len(kwargs.get("active", [])), kwargs.get("remaining_shards", 0)
        now = time.monotonic()
        if current != last[0] or now-last[1] >= 60:
            print(f"GPU QUEUE: {current[1]} running; {current[2]} pending; {state}; maximum {kwargs.get('max_workers',4)}", flush=True)
            last[:] = [current, now]
    queue.command, queue.jobs, queue.scheduling.environment, queue.write_status = command, jobs, environment, status
    try:
        yield
    finally:
        queue.command, queue.jobs, queue.scheduling.environment, queue.write_status = old


def run(policy, smoke_only=False):
    if not read(OUTPUT/"checkpoints_ready.json")["complete"]:
        raise ValueError("Checkpoint preparation incomplete")
    (OUTPUT/"logs").mkdir(exist_ok=True)
    write(OUTPUT/"scheduled_jobs.json", dict(smoke=jobs(True), full=jobs(), policy=policy))
    active = []
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        with queue_adapter(), scheduler.capped_scheduler(policy["max_workers"], policy):
            args = argparse.Namespace(output=OUTPUT, gpus=policy["eligible_gpus"])
            queue.run_phase(args, True, active)
            if not smoke_only:
                queue.run_phase(args, False, active)
                if report() != 8:
                    raise RuntimeError("Not all eight corrected model result cells completed")
                write(OUTPUT/"all.done.json", dict(complete=True, cells=8))
            queue.write_status(OUTPUT, "smoke_complete" if smoke_only else "complete", active=[])
    except (KeyboardInterrupt, Exception):
        # Only stop children created by this launcher, never unrelated GPU processes.
        for task in active:
            if task["process"].poll() is None:
                try:
                    os.killpg(task["process"].pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        for task in active:
            try:
                task["process"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                task["process"].terminate()
                task["process"].wait()
            task["log"].close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "prepare", "run", "smoke", "report", "worker"))
    parser.add_argument("worker_mode", nargs="?", choices=("infer", "merge"))
    parser.add_argument("--gpus", nargs="+", default=list(map(str, range(8))))
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--model", choices=(MODEL,), default=MODEL)
    parser.add_argument("--kind", choices=("boxes", "pointing"), default="boxes")
    parser.add_argument("--suite", choices=tuple(EXPECTED), default="DIOR")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.output = OUTPUT
    if not 1 <= args.max_workers <= 4:
        parser.error("--max-workers must be between 1 and 4")
    if not 0 <= args.shard_index < args.shard_count or args.limit < 0:
        parser.error("Invalid shard index/count or record limit")
    if args.mode == "worker":
        if not args.worker_mode:
            parser.error("worker requires infer or merge")
        (infer if args.worker_mode == "infer" else merge)(args)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/"queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode in ("prepare", "launch"):
            prepare()
        if args.mode == "prepare":
            return
        if args.mode == "report":
            report()
            return
        plan = check_plan()
        if software() != plan["software"]:
            raise ValueError("Environment changed; do not mix software within a result")
        policy = scheduler.prepare_schedule(args.gpus, args.max_workers)
        if policy["hardware"] != plan["hardware"]:
            raise ValueError("Reference hardware changed")
        with (OUTPUT/"scheduling_history.jsonl").open("a") as stream:
            import json
            stream.write(json.dumps(dict(started=time.time(), pid=os.getpid(), **policy))+"\n")
        run(policy, smoke_only=args.mode == "smoke")


if __name__ == "__main__":
    main()
