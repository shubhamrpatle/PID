"""Epoch-based RefWave/standard LA training on a single node with 1-8 GPUs."""

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
from .model import ClusterEpisodeModel, configure_lora_head, configure_full, update_indices
from .loader import make_loader
from .io import append_jsonl, atomic_json, atomic_torch_save, sha256_file, update_symlink
from .runtime import (
    choose_vision_attention,
    distributed_context,
    synchronize_trainable_parameters,
    truncate_curve,
    StopRequest,
)
from .data import ClusterDataset
from .vision_attention import enable_fused_vision_attention

SCHEMA = "refwave-simple-lora-head-v1"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def trainable_state(model):
    return {name: p.detach().cpu() for (name, p) in model.named_parameters() if p.requires_grad}


def restore(model, state):
    values = {n: p for (n, p) in model.named_parameters() if p.requires_grad}
    if set(values) != set(state):
        raise ValueError("Checkpoint does not contain this model's LoRA AND head weights")
    with torch.no_grad():
        for name, p in values.items():
            if tuple(p.shape) != tuple(state[name].shape):
                raise ValueError(f"Checkpoint tensor shape differs: {name}")
            p.copy_(state[name])


def validate_parent(parent, args):
    config = parent.get("config", {})
    continuation = getattr(args, "continue_checkpoint", None)
    expected_stage = "stage2" if continuation else "stage1"
    records = config.get("records", 0)
    total_records = records * config.get("epochs", 1)
    cursor = parent.get("cursor", 0)
    completed = bool(total_records) and cursor == total_records
    if config.get("steps") is not None:
        completed = completed and parent.get("step") == config["steps"]
    # Best validation weights may precede the final update of a completed stage.
    path = continuation or getattr(args, "init_checkpoint", None)
    if not completed and path and (path.parent / "done.json").is_file():
        done = json.loads((path.parent / "done.json").read_text())
        completed = (
            done.get("records") == total_records
            and done.get("steps") == config.get("steps")
            and done.get("epochs", 1) == config.get("epochs", 1)
        )
    if (
        parent.get("schema") != SCHEMA
        or config.get("lora_rank") != args.lora_rank
        or config.get("method") != args.method
        or config.get("stage") != expected_stage
        or not completed
        or not 0 < cursor <= total_records
    ):
        stage_name = "Stage-2" if continuation else "Stage-1"
        raise ValueError(
            f"Stage 2 requires weights from a completed {stage_name} run of the same method and rank"
        )
    if config.get("tuning", "lora") != getattr(args, "tuning", "lora"):
        raise ValueError("Stage 2 must use the same tuning mode as its parent")


def epoch_updates(size, epochs, seed, per_gpu, world, rank, cursor=0):
    """Keep every epoch's tail; distribute each record exactly once per epoch."""
    for epoch in range(cursor // size, epochs):
        order = np.random.default_rng(seed + epoch).permutation(size).tolist()
        offset = epoch * size
        for start in range(max(0, cursor - offset), size, per_gpu * world):
            selected, end = update_indices(order, start, per_gpu, world, rank)
            yield selected, offset + end


def cursor_after_step(step, size, global_batch):
    epoch, within = divmod(step, math.ceil(size / global_batch))
    return epoch * size + min(within * global_batch, size)


def average_gradients(parameters, world):
    if world == 1:
        return
    for p in parameters:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        flat = p.grad.reshape(-1)
        for start in range(0, flat.numel(), 16 * 1024 * 1024):
            piece = flat[start : start + 16 * 1024 * 1024]
            dist.all_reduce(piece)
            piece.div_(world)


def metrics(rows, device, world):
    keys = ("loss", "round1_loss", "round2_loss", "direct_loss", "supervised_tokens", "waves")
    tasks = ("detection", "grounding", "pointing")
    values = [sum((r[k] for r in rows)) for k in keys] + [len(rows)]
    for task in tasks:
        subset = [r for r in rows if r["task"] == task]
        values += [sum((r["loss"] for r in subset)), len(subset)]
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(tensor)
    data = tensor.cpu().tolist()
    result = {k: data[i] / max(data[len(keys)], 1) for (i, k) in enumerate(keys)}
    means = []
    for i, task in enumerate(tasks):
        (total, count) = data[len(keys) + 1 + 2 * i : len(keys) + 3 + 2 * i]
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
    (rows, batch) = ([], [])
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
    for child, kind in (
        (config, "sdpa"),
        (config.text_config, "sdpa"),
        (config.vision_config, vision),
    ):
        child._attn_implementation = kind
        child._attn_implementation_autoset = False
    model = AutoModel.from_pretrained(
        path,
        config=config,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=locateanything_attention_dispatch(vision),
    ).to(device)
    return (model, processor)


def run(args):
    (rank, local, world) = distributed_context()
    seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", local)
    (model, processor) = load_base(args.model, device, args.image_token_limit)
    audit = (
        configure_full(model)
        if args.tuning == "full"
        else configure_lora_head(model, args.lora_rank)
    )
    if args.tuning == "full":
        audit["vision_attention"] = enable_fused_vision_attention(model)
    model.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.language_model.enable_input_require_grads()
    model.language_model.config.use_cache = False
    wrapper = ClusterEpisodeModel(
        model,
        objective_mode="direct_only" if args.method in ("standard", "class_isolated") else "two_route_equal_mean",
        full=args.tuning == "full",
    )

    def dataset(path):
        return ClusterDataset(
            path,
            processor,
            data_root=args.data_root,
            max_sequence=args.max_sequence,
            maximum_wave_size=200,
            round1_answer_format="geometry_only",
            method=args.method,
        )

    (train, val) = (dataset(args.train_data), dataset(args.validation_data))
    if not len(train) or not len(val):
        raise ValueError("Training and validation must be nonempty")
    epochs = getattr(args, "epochs", 1)
    total_records = len(train) * epochs
    per_gpu = args.batch_size * args.accumulation
    total_steps = epochs * math.ceil(len(train) / (per_gpu * world))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "config.json"
    checkpoint_path = output / "last.pt"
    previous = json.loads(contract_path.read_text()) if contract_path.exists() else {}
    init_sha = sha256_file(args.init_checkpoint) if args.init_checkpoint else None
    continuation = getattr(args, "continue_checkpoint", None)
    continue_sha = sha256_file(continuation) if continuation else None
    if checkpoint_path.exists() and not args.init_checkpoint and not continuation:
        # Resume restores last.pt, not a newly selected parent checkpoint.
        init_sha = previous.get("init_checkpoint_sha256")
        continue_sha = previous.get("continue_checkpoint_sha256")
    contract = dict(
        schema=SCHEMA,
        method=args.method,
        base_model=str(args.model.resolve()),
        stage=args.stage,
        train_sha256=sha256_file(args.train_data),
        validation_sha256=sha256_file(args.validation_data),
        records=len(train),
        validation_records=len(val),
        epochs=epochs,
        steps=total_steps,
        world_size=world,
        batch_size=args.batch_size,
        accumulation=args.accumulation,
        global_batch=per_gpu * world,
        batch_tokens=args.batch_tokens,
        max_sequence=args.max_sequence,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        head_lr=args.head_lr,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        image_token_limit=args.image_token_limit,
        trainable=audit,
        synchronized_trainable_initialization=True,
        init_checkpoint_sha256=init_sha,
    )
    if continue_sha:
        contract["continue_checkpoint_sha256"] = continue_sha
    contract.update(
        tuning=args.tuning,
        vision_lr=args.vision_lr,
        projector_lr=args.projector_lr,
        optimizer_sharding=args.tuning == "full" and world > 1,
    )
    # Another rank may publish config.json while this rank hashes the inputs.
    persisted = json.loads(contract_path.read_text()) if contract_path.exists() else None
    if persisted is not None and persisted != contract:
        missing = object()
        changed = sorted(
            key for key in persisted.keys() | contract.keys()
            if persisted.get(key, missing) != contract.get(key, missing)
        )
        raise ValueError(
            "Existing run settings differ. Use a new output directory for a new experiment."
            f" Changed fields: {', '.join(changed)}"
            " Saved/requested values: " + json.dumps({
                key: {"saved": persisted.get(key), "requested": contract.get(key)}
                for key in changed
            }, sort_keys=True)
        )
    if rank == 0:
        atomic_json(contract_path, contract)
        print(json.dumps(contract, indent=2), flush=True)
    saved = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        if checkpoint_path.exists()
        else None
    )
    if saved:
        if saved.get("schema") != SCHEMA or saved.get("config") != contract:
            raise ValueError("Checkpoint schema/settings differ from this run")
        restore(model, saved["trainable_state"])
    elif args.init_checkpoint or continuation:
        parent = torch.load(
            args.init_checkpoint or continuation, map_location="cpu", weights_only=False, mmap=True
        )
        validate_parent(parent, args)
        restore(model, parent["trainable_state"])
        del parent
    elif args.stage == "stage2":
        raise ValueError("Stage 2 needs --init-checkpoint from completed Stage 1")
    synchronize_trainable_parameters(model, world)
    head_ids = {id(p) for p in wrapper.causal.lm_head.parameters()}
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = parameter_groups(model, wrapper, args)
    if args.tuning == "full" and world > 1:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        optimizer = ZeroRedundancyOptimizer(
            groups, optimizer_class=torch.optim.AdamW, weight_decay=0.01, fused=True
        )
    else:
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01, fused=True)
    warmup = max(1, round(total_steps * args.warmup_ratio))

    def factor(step):
        if step < warmup:
            return max(step, 1) / warmup
        progress = min(1.0, (step - warmup) / max(total_steps - warmup, 1))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    (cursor, step, best_loss) = (0, 0, math.inf)
    resume_validation = None
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        (cursor, step, best_loss) = (saved["cursor"], saved["step"], saved["best_loss"])
        resume_validation = saved.get("validation")
        if not 0 <= step <= total_steps or cursor != cursor_after_step(step, len(train), per_gpu * world):
            raise ValueError("Checkpoint cursor does not match its completed optimizer updates")
        rng = saved["rng"][rank]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"])
        if rank == 0:
            truncate_curve(output / "training_curve.jsonl", "step", step)
            truncate_curve(
                output / "validation_curve.jsonl", "step",
                step if resume_validation is not None else step - 1,
            )
            if (
                resume_validation is not None
                and resume_validation["detection_grounding_macro_loss"] == best_loss
            ):
                update_symlink(output / "best.pt", checkpoint_path.resolve())
        del saved
    interrupted = False
    stop = StopRequest()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    wandb_run = None
    if rank == 0:
        print(f"W&B tracking mode: {args.wandb_mode}", flush=True)
    if rank == 0 and args.wandb_mode != "disabled":
        import wandb

        run_id = hashlib.sha256(str(output).encode()).hexdigest()[:16]
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"{output.parent.name}_{output.name}",
            group=output.parent.name,
            job_type=args.stage,
            id=run_id,
            resume="allow",
            dir=str(output),
            mode=args.wandb_mode,
            config=contract,
        )
        # Training and validation can publish separately at the same optimizer step.
        wandb_run.define_metric("optimizer_step")
        wandb_run.define_metric("train/*", step_metric="optimizer_step")
        wandb_run.define_metric("validation/*", step_metric="optimizer_step")

    def save_checkpoint(validation):
        nonlocal best_loss
        improved = (
            validation is not None and validation["detection_grounding_macro_loss"] < best_loss
        )
        if improved:
            best_loss = validation["detection_grounding_macro_loss"]
        rng = dict(
            cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
            python=random.getstate(), numpy=np.random.get_state(),
        )
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, rng)
        else:
            states[0] = rng
        if args.tuning == "full" and world > 1:
            optimizer.consolidate_state_dict(to=0)
        if rank == 0:
            path = output / f"checkpoint-{step:07d}.pt"
            atomic_torch_save(
                path,
                dict(
                    schema=SCHEMA, config=contract, step=step, cursor=cursor,
                    best_loss=best_loss, validation=validation,
                    trainable_state=trainable_state(model), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), rng=states,
                ),
            )
            update_symlink(output / "last.pt", path)
            if improved:
                update_symlink(output / "best.pt", path)
            keep = {p.resolve() for p in (output / "last.pt", output / "best.pt") if p.exists()}
            for old in output.glob("checkpoint-*.pt"):
                if old.resolve() not in keep:
                    old.unlink()
        if world > 1:
            dist.barrier()

    def run_validation():
        validation = validate(wrapper, val, args, rank, world)
        if rank == 0:
            append_jsonl(output / "validation_curve.jsonl", dict(step=step, **validation))
            print(json.dumps(dict(validation_step=step, **validation)), flush=True)
            if wandb_run:
                wandb_run.log(
                    {"optimizer_step": step, **{"validation/" + k: v for k, v in validation.items()}},
                    commit=True,
                )
        return validation

    if cursor == total_records:
        # A signal on the last update may have saved weights before validation.
        if resume_validation is None:
            save_checkpoint(run_validation())
        if rank == 0:
            atomic_json(
                output / "done.json",
                dict(steps=step, records=cursor, total_records=total_records, epochs=epochs),
            )
            (output / "interrupted.json").unlink(missing_ok=True)
        if wandb_run:
            wandb_run.finish()
        if world > 1:
            dist.destroy_process_group()
        return
    if cursor and cursor % len(train) == 0 and resume_validation is None:
        save_checkpoint(run_validation())
    updates = list(epoch_updates(len(train), epochs, args.seed, per_gpu, world, rank, cursor))
    indices = [index for selected, _ in updates for index in selected]
    loader = iter(make_loader(train, indices, args.workers))
    wrapper.train()
    started = time.time()
    for selected, end in updates:
        optimizer.zero_grad(set_to_none=True)
        rows = []
        for start in range(0, len(selected), args.batch_size):
            batch = [next(loader) for _ in selected[start : start + args.batch_size]]
            rows.extend(
                wrapper.batch(batch, divisor=(end - cursor) / world, token_budget=args.batch_tokens)
            )
        average_gradients(trainable, world)
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        if not math.isfinite(norm):
            raise FloatingPointError("Non-finite gradient norm")
        used_rates = [group["lr"] for group in optimizer.param_groups]
        optimizer.step()
        scheduler.step()
        (cursor, step) = (end, step + 1)
        row = dict(
            step=step,
            records_seen=cursor,
            epoch=cursor / len(train),
            elapsed_seconds=time.time() - started,
            gradient_norm=norm,
            learning_rate=used_rates[0],
            head_lr=used_rates[1],
            peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3,
            **metrics(rows, device, world),
        )
        if rank == 0:
            append_jsonl(output / "training_curve.jsonl", row)
            if step == 1 or step % args.log_steps == 0:
                print(json.dumps(row), flush=True)
            if wandb_run:
                wandb_run.log(
                    {"optimizer_step": step, **{"train/" + k: v for k, v in row.items()}},
                    commit=True,
                )
        interrupted = stop.synchronize(device, world)
        validation = None
        if not interrupted and (step % args.eval_steps == 0 or cursor % len(train) == 0):
            validation = run_validation()
        interrupted = stop.synchronize(device, world)
        if validation is not None or interrupted or step % args.save_steps == 0:
            save_checkpoint(validation)
        if interrupted:
            break
    if rank == 0:
        atomic_json(
            output / ("interrupted.json" if interrupted else "done.json"),
            dict(steps=step, records=cursor, total_records=total_records, epochs=cursor / len(train)),
        )
        if not interrupted:
            (output / "interrupted.json").unlink(missing_ok=True)
    if wandb_run:
        wandb_run.finish()
    if world > 1:
        dist.destroy_process_group()
    if interrupted:
        raise SystemExit(130)


def parameter_groups(model, wrapper, args):
    groups = []
    head_ids = {id(p) for p in wrapper.causal.lm_head.parameters()}
    vision_ids = {id(p) for p in model.vision_model.parameters()}
    projector_ids = {id(p) for p in model.mlp1.parameters()}
    assigned = head_ids | vision_ids | projector_ids
    entries = [
        ([p for p in model.parameters() if id(p) not in assigned], args.learning_rate),
        (wrapper.causal.lm_head.parameters(), args.head_lr),
        (model.vision_model.parameters(), args.vision_lr),
        (model.mlp1.parameters(), args.projector_lr),
    ]
    seen = set()
    for parameters, lr in entries:
        selected = [p for p in parameters if p.requires_grad and id(p) not in seen]
        if selected:
            seen.update(map(id, selected))
            groups.append(dict(params=selected, lr=lr))
    if seen != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError("Optimizer does not cover every trainable tensor")
    return groups


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "data-root", "train-data", "validation-data", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    p.add_argument("--method", choices=("refwave", "standard", "class_isolated"), default="refwave")
    p.add_argument("--tuning", choices=("lora", "full"), default="lora")
    parent = p.add_mutually_exclusive_group()
    parent.add_argument("--init-checkpoint", type=Path)
    parent.add_argument("--continue-checkpoint", type=Path)
    for name, value in (
        ("epochs", 1),
        ("batch-size", 4),
        ("accumulation", 1),
        ("batch-tokens", 8192),
        ("max-sequence", 32768),
        ("image-token-limit", 6000),
        ("lora-rank", 32),
        ("workers", 2),
        ("eval-steps", 300),
        ("save-steps", 300),
        ("log-steps", 10),
        ("seed", 20260908),
    ):
        p.add_argument("--" + name, type=int, default=value)
    p.add_argument("--learning-rate", type=float, default=2e-05)
    p.add_argument("--head-lr", type=float, default=1e-06)
    p.add_argument("--vision-lr", type=float, default=1e-06)
    p.add_argument("--projector-lr", type=float, default=1e-05)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--wandb-project", default="refwave-fullscale")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    args = p.parse_args()
    if args.stage == "stage1" and args.init_checkpoint:
        p.error("Stage 1 starts from the released base; omit --init-checkpoint")
    if args.stage != "stage2" and args.continue_checkpoint:
        p.error("--continue-checkpoint is for a new Stage-2 continuation run")
    if any(
        (
            getattr(args, k) <= 0
            for k in (
                "epochs",
                "batch_size",
                "accumulation",
                "batch_tokens",
                "max_sequence",
                "eval_steps",
                "save_steps",
                "log_steps",
                "lora_rank",
            )
        )
    ):
        p.error("Epoch count, batch sizes and intervals must be positive")
    if args.workers < 0 or args.image_token_limit <= 0:
        p.error("Workers must be nonnegative; image-token-limit must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
