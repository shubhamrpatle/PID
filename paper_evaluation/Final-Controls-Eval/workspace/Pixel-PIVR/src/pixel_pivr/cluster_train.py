"""One-pass RefWave/standard LA training on a single node with 1-8 GPUs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.distributed as dist

from .attention import locateanything_attention_dispatch
from .cluster_modeling import ClusterEpisodeModel, configure_lora_head, update_indices
from .data import make_loader
from .io import append_jsonl, atomic_json, atomic_torch_save, sha256_file, update_symlink
from .train import choose_vision_attention, distributed_context, synchronize_trainable_parameters, truncate_curve
from .cluster_data import ClusterDataset

SCHEMA = "refwave-cluster-lora-head-v1"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def trainable_state(model):
    return {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}


def restore(model, state):
    values = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(values) != set(state):
        raise ValueError("Checkpoint does not contain this model's LoRA AND head weights")
    with torch.no_grad():
        for name, p in values.items():
            if tuple(p.shape) != tuple(state[name].shape):
                raise ValueError(f"Checkpoint tensor shape differs: {name}")
            p.copy_(state[name])


def validate_parent(parent, args):
    config = parent.get("config", {})
    if (parent.get("schema") != SCHEMA or config.get("lora_rank") != args.lora_rank
            or config.get("method") != args.method or config.get("stage") != "stage1"
            or parent.get("cursor") != config.get("records") or not config.get("records")):
        raise ValueError("Stage 2 requires the completed Stage-1 checkpoint of the same method and rank")


def average_gradients(parameters, world):
    if world == 1:
        return
    # Bounded transfers avoid making another full-head gradient allocation.
    for p in parameters:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        flat = p.grad.reshape(-1)
        for start in range(0, flat.numel(), 16 * 1024 * 1024):
            piece = flat[start:start + 16 * 1024 * 1024]
            dist.all_reduce(piece)
            piece.div_(world)


def metrics(rows, device, world):
    keys = ("loss", "round1_loss", "round2_loss", "direct_loss", "supervised_tokens", "waves")
    tasks = ("detection", "grounding", "pointing")
    values = [sum(r[k] for r in rows) for k in keys] + [len(rows)]
    for task in tasks:
        subset = [r for r in rows if r["task"] == task]
        values += [sum(r["loss"] for r in subset), len(subset)]
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(tensor)
    data = tensor.cpu().tolist()
    result = {k: data[i] / max(data[len(keys)], 1) for i, k in enumerate(keys)}
    means = []
    for i, task in enumerate(tasks):
        total, count = data[len(keys) + 1 + 2*i:len(keys) + 3 + 2*i]
        if count:
            result[task + "_loss"] = total / count
            if task != "pointing":
                means.append(total / count)
    result["detection_grounding_macro_loss"] = sum(means) / max(len(means), 1)
    return result


@torch.no_grad()
def validate(wrapper, dataset, args, rank, world):
    wrapper.eval()
    wrapper.causal.model.training = True
    rows, batch = [], []
    for sample in make_loader(dataset, list(range(rank, len(dataset), world)), args.workers):
        batch.append(sample)
        if len(batch) == args.batch_size:
            rows.extend(wrapper.batch(batch, backward=False, token_budget=args.batch_tokens))
            batch = []
    rows.extend(wrapper.batch(batch, backward=False, token_budget=args.batch_tokens))
    result = metrics(rows, wrapper.device, world)
    wrapper.train()
    return result


def load_base(path, device, image_limit):
    from transformers import AutoConfig, AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, use_fast=False)
    processor.image_processor.in_token_limit = image_limit
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    vision = choose_vision_attention("auto") if device.type == "cuda" else "sdpa"
    for child, kind in ((config, "sdpa"), (config.text_config, "sdpa"), (config.vision_config, vision)):
        child._attn_implementation = kind
        child._attn_implementation_autoset = False
    model = AutoModel.from_pretrained(path, config=config, trust_remote_code=True,
        dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=locateanything_attention_dispatch(vision)).to(device)
    return model, processor


def run(args):
    rank, local, world = distributed_context()
    seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", local)
    model, processor = load_base(args.model, device, args.image_token_limit)
    audit = configure_lora_head(model, args.lora_rank)
    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.language_model.enable_input_require_grads()
    model.language_model.config.use_cache = False
    wrapper = ClusterEpisodeModel(model, objective_mode=("direct_only" if args.method == "standard" else "two_route_equal_mean"))
    def dataset(path):
        return ClusterDataset(path, processor, data_root=args.data_root, eagle_root=args.eagle_root,
            max_sequence=args.max_sequence, maximum_wave_size=200, round1_answer_format="geometry_only",
            method=args.method)
    train, val = dataset(args.train_data), dataset(args.validation_data)
    if not len(train) or not len(val):
        raise ValueError("Training and validation must be nonempty")
    order = np.random.default_rng(args.seed).permutation(len(train)).tolist()
    per_gpu = args.batch_size * args.accumulation
    total_steps = math.ceil(len(train) / (per_gpu * world))
    contract = dict(schema=SCHEMA, method=args.method, base_model=str(args.model.resolve()),
        stage=args.stage, train_sha256=sha256_file(args.train_data),
        validation_sha256=sha256_file(args.validation_data), records=len(train),
        validation_records=len(val), epochs=1, steps=total_steps, world_size=world,
        batch_size=args.batch_size, accumulation=args.accumulation,
        global_batch=per_gpu * world, batch_tokens=args.batch_tokens, max_sequence=args.max_sequence,
        lora_rank=args.lora_rank, learning_rate=args.learning_rate, head_lr=args.head_lr,
        warmup_ratio=args.warmup_ratio, seed=args.seed, image_token_limit=args.image_token_limit,
        trainable=audit, synchronized_trainable_initialization=True,
        init_checkpoint_sha256=sha256_file(args.init_checkpoint) if args.init_checkpoint else None)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "config.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Existing run settings differ. Use a new output directory for a new experiment.")
    if rank == 0:
        atomic_json(contract_path, contract)
        print(json.dumps(contract, indent=2), flush=True)
    checkpoint_path = output / "last.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False) if checkpoint_path.exists() else None
    if saved:
        if saved.get("schema") != SCHEMA or saved.get("config") != contract:
            raise ValueError("Checkpoint schema/settings differ from this run")
        restore(model, saved["trainable_state"])
    elif args.init_checkpoint:
        parent = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        validate_parent(parent, args)
        restore(model, parent["trainable_state"])
        del parent
    elif args.stage == "stage2":
        raise ValueError("Stage 2 needs --init-checkpoint from completed Stage 1")
    synchronize_trainable_parameters(model, world)
    head_ids = {id(p) for p in wrapper.causal.lm_head.parameters()}
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": [p for p in trainable if id(p) not in head_ids], "lr": args.learning_rate},
        {"params": [p for p in trainable if id(p) in head_ids], "lr": args.head_lr},
    ], weight_decay=0.01, fused=True)
    warmup = max(1, round(total_steps * args.warmup_ratio))
    def factor(step):
        if step < warmup:
            return max(step, 1) / warmup
        progress = min(1.0, (step - warmup) / max(total_steps - warmup, 1))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    cursor, step, best_loss = 0, 0, math.inf
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        cursor, step, best_loss = saved["cursor"], saved["step"], saved["best_loss"]
        if not 0 <= step <= total_steps or cursor != min(step*per_gpu*world, len(train)):
            raise ValueError("Checkpoint cursor does not match its completed optimizer updates")
        rng = saved["rng"][rank]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"])
        if rank == 0:
            truncate_curve(output / "training_curve.jsonl", "step", step)
            truncate_curve(output / "validation_curve.jsonl", "step", step)
        del saved
    if cursor == len(train):
        if rank == 0:
            atomic_json(output / "done.json", dict(steps=step, records=cursor, epochs=1))
        if world > 1:
            dist.destroy_process_group()
        return
    # Flat rank-local index stream, no DistributedSampler padding or replacement.
    indices = []
    for start in range(cursor, len(train), per_gpu * world):
        selected, _ = update_indices(order, start, per_gpu, world, rank)
        indices.extend(selected)
    loader = iter(make_loader(train, indices, args.workers))
    interrupted = False
    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    wandb_run = None
    if rank == 0 and args.wandb_mode != "disabled":
        import wandb
        run_id = hashlib.sha256(str(output).encode()).hexdigest()[:16]
        wandb_run = wandb.init(project=args.wandb_project, name=output.name + "_" + args.method,
            id=run_id, resume="allow", dir=str(output), mode=args.wandb_mode, config=contract)
    wrapper.train()
    started = time.time()
    while cursor < len(train):
        selected, end = update_indices(order, cursor, per_gpu, world, rank)
        optimizer.zero_grad(set_to_none=True)
        rows = []
        for start in range(0, len(selected), args.batch_size):
            batch = [next(loader) for _ in selected[start:start + args.batch_size]]
            rows.extend(wrapper.batch(batch, divisor=(end - cursor) / world,
                                      token_budget=args.batch_tokens))
        average_gradients(trainable, world)
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        if not math.isfinite(norm):
            raise FloatingPointError("Non-finite gradient norm")
        optimizer.step()
        scheduler.step()
        cursor, step = end, step + 1
        row = dict(step=step, records_seen=cursor, elapsed_seconds=time.time() - started,
            gradient_norm=norm, learning_rate=scheduler.get_last_lr()[0],
            head_lr=scheduler.get_last_lr()[1], peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
            **metrics(rows, device, world))
        if rank == 0:
            append_jsonl(output / "training_curve.jsonl", row)
            if step == 1 or step % args.log_steps == 0:
                print(json.dumps(row), flush=True)
            if wandb_run:
                wandb_run.log({"train/" + k: v for k, v in row.items()}, step=step)
        flag = torch.tensor(int(interrupted), device=device)
        if world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        interrupted = bool(flag.item())
        validation = None
        if not interrupted and (step % args.eval_steps == 0 or cursor == len(train)):
            validation = validate(wrapper, val, args, rank, world)
            if rank == 0:
                append_jsonl(output / "validation_curve.jsonl", dict(step=step, **validation))
                print(json.dumps(dict(validation_step=step, **validation)), flush=True)
                if wandb_run:
                    wandb_run.log({"validation/" + k: v for k, v in validation.items()}, step=step)
        if validation is not None or interrupted or step % args.save_steps == 0:
            improved = validation is not None and validation["detection_grounding_macro_loss"] < best_loss
            if improved:
                best_loss = validation["detection_grounding_macro_loss"]
            rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                       python=random.getstate(), numpy=np.random.get_state())
            states = [None] * world
            if world > 1:
                dist.all_gather_object(states, rng)
            else:
                states[0] = rng
            if rank == 0:
                path = output / f"checkpoint-{step:07d}.pt"
                atomic_torch_save(path, dict(schema=SCHEMA, config=contract, step=step, cursor=cursor,
                    best_loss=best_loss, validation=validation, trainable_state=trainable_state(model),
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=states))
                update_symlink(output / "last.pt", path)
                if improved:
                    update_symlink(output / "best.pt", path)
                keep = {p.resolve() for p in (output / "last.pt", output / "best.pt") if p.exists()}
                for old in output.glob("checkpoint-*.pt"):
                    if old.resolve() not in keep:
                        old.unlink()
            if world > 1:
                dist.barrier()
        if interrupted:
            break
    if rank == 0:
        atomic_json(output / ("interrupted.json" if interrupted else "done.json"),
                    dict(steps=step, records=cursor, total_records=len(train), epochs=cursor/len(train)))
    if wandb_run:
        wandb_run.finish()
    if world > 1:
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "eagle-root", "data-root", "train-data", "validation-data", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    p.add_argument("--method", choices=("refwave", "standard"), default="refwave")
    p.add_argument("--init-checkpoint", type=Path)
    for name, value in (("batch-size", 4), ("accumulation", 1), ("batch-tokens", 8192),
                        ("max-sequence", 32768), ("image-token-limit", 6000), ("lora-rank", 32),
                        ("workers", 2), ("eval-steps", 300), ("save-steps", 300), ("log-steps", 10), ("seed", 20260908)):
        p.add_argument("--" + name, type=int, default=value)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-6)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--wandb-project", default="refwave-fullscale")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    args = p.parse_args()
    if args.stage == "stage1" and args.init_checkpoint:
        p.error("Stage 1 starts from the released base; omit --init-checkpoint")
    if any(getattr(args, k) <= 0 for k in ("batch_size", "accumulation", "batch_tokens", "max_sequence", "eval_steps", "save_steps", "log_steps", "lora_rank")):
        p.error("Batch sizes and intervals must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
