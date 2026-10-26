"""Exact-coverage two-stage LoRA training for unified CAW-PIVR episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist

from .attention import locateanything_attention_dispatch, verify_locateanything_attention
from .data import make_loader
from .io import (
    append_jsonl,
    atomic_json,
    atomic_torch_save,
    data_signature,
    sha256_file,
    update_symlink,
)
from .lora import configure_lora_only, restore_trainable_state, trainable_state
from .train import (
    choose_vision_attention,
    distributed_context,
    gather_rng_states,
    rank_log,
    rank_training_indices,
    scheduler_factor,
    synchronize_gradients,
    synchronize_trainable_parameters,
    truncate_curve,
)
from .unified_episode import SCHEMA, ROUND1_ANSWER_FORMATS, UnifiedEpisodeDataset
from .unified_modeling import OBJECTIVE_MODES, UnifiedEpisodeTrainingModel


CHECKPOINT_SCHEMA = "caw-pivr-unified-hbb-lora-checkpoint-v2"
METRICS = (
    "loss",
    "direct_loss",
    "round1_loss",
    "round2_loss",
    "direct_active",
    "round1_active",
    "round2_active",
    "round1_branches",
    "round2_branches",
    "supervised_tokens",
    "waves",
    "maximum_wave_tokens",
    "maximum_wave_branches",
    "global_visual_tokens",
    "vision_encode_calls",
)


def seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))


def reduce_rows(
    rows: list[Mapping[str, Any]], device: torch.device, world_size: int
) -> dict[str, float]:
    route_denominators = {
        "direct_loss": "direct_active",
        "round1_loss": "round1_active",
        "round2_loss": "round2_active",
    }
    numerators: list[float] = []
    denominators: list[float] = []
    for key in METRICS:
        denominator_key = route_denominators.get(key)
        if denominator_key:
            numerators.append(
                sum(float(row[key]) * float(row[denominator_key]) for row in rows)
            )
            denominators.append(sum(float(row[denominator_key]) for row in rows))
        else:
            numerators.append(sum(float(row[key]) for row in rows))
            denominators.append(float(len(rows)))
    task_values = []
    for task in ("detection", "grounding", "pointing"):
        selected = [row for row in rows if row.get("task") == task]
        task_values.extend((sum(float(row["loss"]) for row in selected), len(selected)))
    values = torch.tensor(
        [*numerators, *denominators, *task_values], dtype=torch.float64, device=device
    )
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    size = len(METRICS)
    result = {
        key: float(values[index] / max(float(values[size + index]), 1e-12))
        for index, key in enumerate(METRICS)
    }
    task_offset = size * 2
    task_losses = []
    for task_index, task in enumerate(("detection", "grounding", "pointing")):
        total = float(values[task_offset + task_index * 2])
        count = float(values[task_offset + task_index * 2 + 1])
        result[f"{task}_loss"] = total / count if count else 0.0
        if count:
            task_losses.append(total / count)
    result["macro_task_loss"] = sum(task_losses) / len(task_losses)
    box_task_losses = [
        result[f"{task}_loss"] for index, task in enumerate(("detection", "grounding"))
        if float(values[task_offset + index * 2 + 1]) > 0
    ]
    result["detection_grounding_macro_loss"] = (
        sum(box_task_losses) / len(box_task_losses) if box_task_losses else 0.0
    )
    return result


@torch.no_grad()
def validate(
    wrapper: UnifiedEpisodeTrainingModel,
    dataset: UnifiedEpisodeDataset,
    indices: list[int],
    workers: int,
    *,
    rank: int,
    world_size: int,
) -> dict[str, float]:
    wrapper.eval()
    # The released Qwen implementation selects its PBD6 training mask through
    # this module-level flag. Child modules remain in eval mode, so LoRA dropout
    # stays disabled while the explicit branch mask hook remains active.
    wrapper.causal.model.training = True
    rows: list[dict[str, Any]] = []
    for sample in make_loader(dataset, indices[rank::world_size], workers):
        metrics = wrapper.episode(sample, gradient_accumulation=1, backward=False)
        metrics["task"] = str(sample["task"])
        rows.append(metrics)
    result = reduce_rows(rows, wrapper.device, world_size)
    wrapper.causal.model.training = False
    wrapper.train()
    return result


def checkpoint_payload(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    step: int,
    exposure: int,
    validation: Mapping[str, float] | None,
    train_signature: list[dict[str, Any]],
    validation_signature: list[dict[str, Any]],
    lora_audit: Mapping[str, Any],
    parameter_sync: Mapping[str, Any],
    rng_states: list[dict[str, torch.Tensor]] | None,
    world_size: int,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "config": {
            "experiment": args.experiment_name,
            "stage": args.stage,
            "base_model": str(args.model.resolve()),
            "data_root": str(args.data_root.resolve()),
            "train_data": train_signature,
            "validation_data": validation_signature,
            "data_manifest_sha256": sha256_file(args.data_manifest.resolve()),
            "validation_report_sha256": sha256_file(args.validation_report.resolve()),
            "exact_loader_report": str(args.exact_loader_report.resolve()),
            "exact_loader_report_sha256": sha256_file(
                args.exact_loader_report.resolve()
            ),
            "loader_audit_policy": args.loader_audit_policy,
            "exhaustive_loader_audit_completed": args.loader_audit_policy == "full",
            "init_adapter": (
                None
                if args.init_adapter is None
                else {
                    "path": str(args.init_adapter.resolve()),
                    "sha256": sha256_file(args.init_adapter.resolve()),
                }
            ),
            "lora_rank": int(args.lora_rank),
            "image_token_limit": int(args.image_token_limit),
            "max_sequence": int(args.max_sequence),
            "maximum_wave_size": int(args.maximum_wave_size),
            "objective_mode": str(args.objective_mode),
            "round1_answer_format": args.round1_answer_format,
            "checkpoint_selection_metric": args.selection_metric,
            "gradient_accumulation": int(args.gradient_accumulation),
            "max_steps": int(args.max_steps),
            "world_size": int(world_size),
            "seed": int(args.seed),
            "learning_rate": float(args.learning_rate),
            "warmup_steps": int(args.warmup_steps),
            "vision_attention": str(args.resolved_vision_attention),
            "sample_order": "deterministic_shuffled_exact_one_pass",
            "one_global_image_encode_per_episode": True,
            "branch_attention": (
                "single_direct_branch_native_pbd6"
                if args.objective_mode == "direct_only"
                else "shared_prefix_sibling_isolated_pbd6"
            ),
            "episode_loss": (
                "direct_la_conversation_only"
                if args.objective_mode == "direct_only"
                else (
                    "sum_round1_round2_token_CE_over_episode_supervised_tokens"
                    if args.objective_mode == "two_route_token_mean"
                    else (
                        "0.5_round1_token_mean_plus_0.5_round2_token_mean_then_episode_mean"
                        if args.objective_mode == "two_route_equal_mean"
                        else "source_query_normalized_direct_round1_round2"
                    )
                )
            ),
            "trainable_scope": "Qwen LoRA only",
            "synchronized_trainable_initialization": True,
            "lora_audit": dict(lora_audit),
            "parameter_sync": dict(parameter_sync),
            "source_code_sha256": {
                name: sha256_file(Path(__file__).with_name(name))
                for name in (
                    "unified_train.py",
                    "unified_episode.py",
                    "unified_modeling.py",
                    "lora.py",
                    "data.py",
                )
            },
        },
        "step": int(step),
        "exposure": int(exposure),
        "validation": None if validation is None else dict(validation),
        "trainable_state": trainable_state(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "distributed_rng_states": rng_states,
    }


def prune(output: Path, keep: int) -> list[str]:
    checkpoints = sorted(
        output.glob("checkpoint-step*.pt"),
        key=lambda path: int(path.stem.removeprefix("checkpoint-step")),
    )
    protected = set()
    for name in ("last.pt", "best.pt"):
        link = output / name
        if link.exists() or link.is_symlink():
            protected.add(link.resolve())
    removable = [path for path in checkpoints[:-keep] if path.resolve() not in protected]
    for path in removable:
        path.unlink()
    return [str(path) for path in removable]


def validate_loader_gate(
    args: argparse.Namespace, *, filtered_source: bool, source_report: Mapping[str, Any]
) -> None:
    loader_report = json.loads(args.exact_loader_report.read_text())
    if loader_report.get("round1_answer_format", "reference_geometry") != getattr(args, "round1_answer_format", "reference_geometry"):
        raise RuntimeError("Loader audit Round-1 answer format does not match this run")
    loader_files = loader_report.get("files") or {}
    if (
        loader_report.get("status") != "passed"
        or int(loader_report.get("image_token_limit", -1)) != args.image_token_limit
        or int(loader_report.get("max_sequence", -1)) != args.max_sequence
        or int(loader_report.get("sequence_overflows", -1)) != 0
        or int(loader_report.get("round2_non_six_label_branches", -1)) != 0
    ):
        raise RuntimeError("Loader audit settings or safety checks do not match this run")
    if args.loader_audit_policy == "smoke_only":
        if not filtered_source:
            raise RuntimeError("Smoke-only startup requires a verified source-filter export")
        if (
            loader_report.get("maximum_wave_size") != args.maximum_wave_size
            or loader_report.get("loader_errors") != 0
            or loader_report.get("silent_replacements") != 0
            or not loader_files
            or sum(entry.get("episodes", 0) for entry in loader_files.values()) < 8
        ):
            raise RuntimeError("Smoke loader report is incomplete or incompatible")
        for name, entry in loader_files.items():
            if entry.get("sha256") != sha256_file(Path(name)):
                raise RuntimeError(f"Smoke data changed since audit: {name}")
        # The source audit verifies annotations/checksums, not full loader coverage.
        for path, expected in (
            (args.train_data, args.expected_train_records),
            (args.validation_data, args.expected_validation_records),
        ):
            entry = source_report.get("splits", {}).get(path.stem, {})
            if (
                path.resolve().parent != args.data_manifest.resolve().parent
                or entry.get("sha256") != sha256_file(path)
                or entry.get("records") != expected
            ):
                raise RuntimeError(f"Source integrity does not match this run: {path}")
    elif args.loader_audit_policy == "full":
        entry = loader_files.get(str(args.train_data.resolve()), {})
        if int(entry.get("episodes", -1)) != args.expected_train_records:
            raise RuntimeError("Exact loader audit does not cover this training file")
        if filtered_source:
            for path in (args.train_data, args.validation_data):
                entry = loader_files.get(str(path.resolve()), {})
                if entry.get("sha256") != sha256_file(path):
                    raise RuntimeError(f"File changed since exact loader audit: {path}")
    else:
        raise RuntimeError("Unknown loader audit policy")
    if filtered_source:
        if args.objective_mode not in loader_report.get("audited_objective_modes", []):
            raise RuntimeError("Loader did not audit the requested objective")
        for name in ("unified_episode.py", "unified_modeling.py"):
            if loader_report.get("source_code_sha256", {}).get(name) != sha256_file(Path(__file__).with_name(name)):
                raise RuntimeError(f"Loader implementation changed since audit: {name}")


def run(args: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoModel, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rank, local_rank, world_size = distributed_context()
    device = torch.device(f"cuda:{local_rank}")
    seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    manifest = json.loads(args.data_manifest.read_text())
    report = json.loads(args.validation_report.read_text())
    filtered_source = manifest.get("schema_version") == "caw-pivr-matched-source-filter-v1"
    if filtered_source:
        if (
            report.get("schema_version") != manifest["schema_version"]
            or report.get("status") != "passed_source_filter_audit"
            or report.get("manifest_sha256") != sha256_file(args.data_manifest)
            or report.get("training_benchmark_image_hash_overlap") != 0
            or report.get("training_validation_image_hash_overlap") != 0
            or args.objective_mode not in {"two_route_token_mean", "two_route_equal_mean", "direct_only"}
        ):
            raise RuntimeError("Filtered source requires its verified two-route/direct-only recipe")
    else:
        if manifest.get("schema_version") != SCHEMA:
            raise RuntimeError("Unexpected unified data manifest schema")
        if report.get("status") != "passed" or report.get("schema_version") != SCHEMA:
            raise RuntimeError("Unified data has not passed independent validation")
    validate_loader_gate(args, filtered_source=filtered_source, source_report=report)
    rank_log(rank, f"loader audit policy: {args.loader_audit_policy}; runtime token checks remain strict")

    train_signature = data_signature([args.train_data.resolve()])
    validation_signature = data_signature([args.validation_data.resolve()])
    if train_signature[0]["records"] != args.expected_train_records:
        raise RuntimeError("Training record count differs from the run contract")
    if validation_signature[0]["records"] != args.expected_validation_records:
        raise RuntimeError("Validation record count differs from the run contract")
    required = int(args.max_steps) * int(args.gradient_accumulation) * world_size
    if required != int(args.expected_train_records):
        raise RuntimeError(
            "Exact coverage requires steps*accumulation*world_size == records: "
            f"{required} != {args.expected_train_records}"
        )

    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    processor.image_processor.in_token_limit = int(args.image_token_limit)
    processor.tokenizer.model_max_length = int(args.max_sequence)
    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    vision_attention = choose_vision_attention(args.vision_attention)
    args.resolved_vision_attention = vision_attention
    config._attn_implementation = "sdpa"
    config._attn_implementation_autoset = False
    config.text_config._attn_implementation = "sdpa"
    config.text_config._attn_implementation_autoset = False
    config.vision_config._attn_implementation = vision_attention
    config.vision_config._attn_implementation_autoset = False
    dispatch = locateanything_attention_dispatch(vision_attention)

    def load_model():
        value = AutoModel.from_pretrained(
            args.model,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=dispatch,
        ).to(device)
        verify_locateanything_attention(value, vision_attention)
        return value

    model = None
    for loading_rank in range(world_size):
        if rank == loading_rank:
            rank_log(rank, "loading LocateAnything")
            model = load_model()
        if world_size > 1:
            dist.barrier(device_ids=[local_rank])
    if model is None:
        raise RuntimeError("Model loading failed")
    lora_audit = configure_lora_only(model, args.lora_rank)
    if args.init_adapter is not None:
        initial = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
        if int((initial.get("config") or {}).get("lora_rank", -1)) != args.lora_rank:
            raise RuntimeError("Stage-1 adapter rank differs from Stage-2")
        restore_trainable_state(model, initial["trainable_state"])
    model.vision_model.eval()
    model.mlp1.eval()
    model.language_model.gradient_checkpointing_enable()
    model.language_model.config.use_cache = False
    wrapper = UnifiedEpisodeTrainingModel(
        model, objective_mode=args.objective_mode
    ).to(device)
    parameter_sync = synchronize_trainable_parameters(wrapper, world_size)
    seed_all(args.seed + rank)

    train_dataset = UnifiedEpisodeDataset(
        args.train_data,
        processor,
        data_root=args.data_root,
        eagle_root=args.eagle_root,
        max_sequence=args.max_sequence,
        maximum_wave_size=args.maximum_wave_size,
        round1_answer_format=args.round1_answer_format,
    )
    validation_dataset = UnifiedEpisodeDataset(
        args.validation_data,
        processor,
        data_root=args.data_root,
        eagle_root=args.eagle_root,
        max_sequence=args.max_sequence,
        maximum_wave_size=args.maximum_wave_size,
        round1_answer_format=args.round1_answer_format,
    )
    trainable = [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda current: scheduler_factor(current, args.max_steps, args.warmup_steps),
    )

    output = args.output.resolve()
    contract = {
        "schema_version": "caw-pivr-unified-run-contract-v2",
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "stage": args.stage,
        "experiment": args.experiment_name,
        "base_model": str(args.model.resolve()),
        "data_root": str(args.data_root.resolve()),
        "train_data": train_signature,
        "validation_data": validation_signature,
        "data_manifest_sha256": sha256_file(args.data_manifest.resolve()),
        "validation_report_sha256": sha256_file(args.validation_report.resolve()),
        "exact_loader_report": str(args.exact_loader_report.resolve()),
        "exact_loader_report_sha256": sha256_file(args.exact_loader_report.resolve()),
        "loader_audit_policy": args.loader_audit_policy,
        "exhaustive_loader_audit_completed": args.loader_audit_policy == "full",
        "init_adapter_sha256": (
            None if args.init_adapter is None else sha256_file(args.init_adapter.resolve())
        ),
        "world_size": world_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_episode_batch": world_size * args.gradient_accumulation,
        "physical_episodes_per_gpu": 1,
        "batch_implementation": "one_episode_per_gpu_with_gradient_accumulation",
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "image_token_limit": args.image_token_limit,
        "max_sequence": args.max_sequence,
        "maximum_wave_size": args.maximum_wave_size,
        "objective_mode": args.objective_mode,
        "round1_answer_format": args.round1_answer_format,
        "checkpoint_selection_metric": args.selection_metric,
        "seed": args.seed,
        "lora": lora_audit,
        "parameter_sync": parameter_sync,
        "vision_attention": vision_attention,
        "source_code_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "unified_train.py",
                "unified_episode.py",
                "unified_modeling.py",
                "lora.py",
                "data.py",
            )
        },
    }
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        contract_path = output / "run_contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise RuntimeError(f"Unsafe resume: run contract changed at {contract_path}")
        atomic_json(contract_path, contract)
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])

    start_step = 0
    start_exposure = 0
    best_validation = math.inf
    last = output / "last.pt"
    if last.exists() or last.is_symlink():
        payload = torch.load(last, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA:
            raise RuntimeError("Existing checkpoint schema differs")
        if payload["config"]["data_manifest_sha256"] != contract["data_manifest_sha256"]:
            raise RuntimeError("Cannot resume after data changed")
        restore_trainable_state(model, payload["trainable_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        start_step = int(payload["step"])
        start_exposure = int(payload["exposure"])
        states = payload.get("distributed_rng_states") or []
        if rank < len(states):
            torch.set_rng_state(states[rank]["cpu"])
            torch.cuda.set_rng_state(states[rank]["cuda"])
        best = output / "best.pt"
        if best.exists() or best.is_symlink():
            best_payload = torch.load(best, map_location="cpu", weights_only=False)
            best_validation = float(best_payload["validation"][args.selection_metric])
        if rank == 0:
            truncate_curve(output / "training_curve.jsonl", "step", start_step)
            truncate_curve(output / "validation_curve.jsonl", "validation_step", start_step)
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])
    if start_step >= args.max_steps:
        rank_log(rank, f"stage already complete at step {start_step}")
        if world_size > 1:
            dist.destroy_process_group()
        return

    local_start = start_exposure // world_size
    local_total = args.max_steps * args.gradient_accumulation
    local_indices = rank_training_indices(
        len(train_dataset), 0, args.seed, "shuffled", rank, world_size,
        local_start, local_total,
    )
    validation_indices = list(range(len(validation_dataset)))
    loader = make_loader(train_dataset, local_indices, args.workers)

    interrupted = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    wandb_run = None
    if rank == 0 and args.wandb_project:
        import wandb

        run_id = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:24]
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or output.name,
            id=run_id,
            resume="allow",
            config=contract,
            dir=str(output),
            mode=args.wandb_mode,
        )

    step = start_step
    local_exposure = local_start
    exposure = start_exposure
    running: list[dict[str, Any]] = []
    started = time.time()
    wrapper.train()
    optimizer.zero_grad(set_to_none=True)

    def save(validation: Mapping[str, float] | None, reason: str) -> None:
        nonlocal best_validation
        states = gather_rng_states(rank, world_size)
        if rank == 0:
            path = output / f"checkpoint-step{step:07d}.pt"
            payload = checkpoint_payload(
                args, model, optimizer, scheduler,
                step=step, exposure=exposure, validation=validation,
                train_signature=train_signature,
                validation_signature=validation_signature,
                lora_audit=lora_audit, parameter_sync=parameter_sync,
                rng_states=states, world_size=world_size,
            )
            atomic_torch_save(path, payload)
            update_symlink(output / "last.pt", path)
            if validation is not None and validation[args.selection_metric] < best_validation:
                best_validation = float(validation[args.selection_metric])
                update_symlink(output / "best.pt", path)
            removed = prune(output, args.keep_recent_checkpoints)
            atomic_json(
                output / "status.json",
                {
                    "step": step,
                    "episode_exposure": exposure,
                    "reason": reason,
                    "validation": validation,
                    "best_validation_macro_task_loss": (
                        best_validation if args.selection_metric == "macro_task_loss" else None
                    ),
                    "best_validation_selection_loss": best_validation,
                    "checkpoint_selection_metric": args.selection_metric,
                    "removed_checkpoints": removed,
                    "interrupted": interrupted,
                },
            )
        if world_size > 1:
            dist.barrier(device_ids=[local_rank])

    for index, sample in zip(local_indices, loader):
        sample_seed = int.from_bytes(
            hashlib.sha256(f"{args.seed}|{index}|episode".encode()).digest()[:4], "big"
        )
        seed_all(sample_seed)
        metrics = wrapper.episode(
            sample,
            gradient_accumulation=args.gradient_accumulation,
            backward=True,
        )
        metrics["task"] = str(sample["task"])
        running.append(metrics)
        local_exposure += 1
        exposure = local_exposure * world_size
        if local_exposure % args.gradient_accumulation:
            continue

        synchronize_gradients(trainable, world_size)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm))
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(f"Non-finite gradient norm before optimizer step {step + 1}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        averaged = reduce_rows(running, device, world_size)
        running.clear()
        if rank == 0:
            row = {
                "step": step,
                "episode_exposure": exposure,
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": gradient_norm,
                "elapsed_seconds": time.time() - started,
                "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1024**2,
                **averaged,
            }
            append_jsonl(output / "training_curve.jsonl", row)
            if wandb_run is not None:
                wandb_run.log(
                    {f"train/{key}": value for key, value in row.items() if isinstance(value, (int, float))},
                    step=step,
                )
            if step == 1 or step % args.log_steps == 0:
                print(json.dumps(row, sort_keys=True), flush=True)

        flag = torch.tensor(int(interrupted), dtype=torch.int32, device=device)
        if world_size > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        should_stop = bool(flag.item())
        should_validate = (
            step % args.eval_steps == 0 or step == args.max_steps
        ) and not should_stop
        validation = None
        if should_validate:
            validation = validate(
                wrapper, validation_dataset, validation_indices, args.workers,
                rank=rank, world_size=world_size,
            )
            if rank == 0:
                value = {"validation_step": step, **validation}
                append_jsonl(output / "validation_curve.jsonl", value)
                print(json.dumps(value, sort_keys=True), flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"validation/{key}": metric for key, metric in validation.items()},
                        step=step,
                    )
        if should_stop or step % args.checkpoint_steps == 0 or should_validate:
            save(validation, "signal" if should_stop else "scheduled")
        if should_stop or step >= args.max_steps:
            break

    if interrupted:
        if rank == 0:
            atomic_json(output / "interrupted.json", {"step": step, "exposure": exposure})
    elif step != args.max_steps or exposure != len(train_dataset):
        raise RuntimeError(
            f"Incomplete exact pass: step={step}/{args.max_steps}, "
            f"exposure={exposure}/{len(train_dataset)}"
        )
    elif rank == 0:
        atomic_json(
            output / "done.json",
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "stage": args.stage,
                "steps": step,
                "episode_exposure": exposure,
                "unique_training_episodes": len(train_dataset),
                "complete_exact_one_pass": True,
                "effective_episode_batch": world_size * args.gradient_accumulation,
                "best": str((output / "best.pt").resolve()),
                "last": str((output / "last.pt").resolve()),
            },
        )
        (output / "interrupted.json").unlink(missing_ok=True)
    if wandb_run is not None:
        wandb_run.finish()
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eagle-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--exact-loader-report", type=Path, required=True)
    parser.add_argument(
        "--loader-audit-policy", choices=("full", "smoke_only"), default="full",
        help="Use full loader coverage, or explicit smoke coverage plus source integrity checks.",
    )
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--expected-train-records", type=int, required=True)
    parser.add_argument("--expected-validation-records", type=int, required=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--image-token-limit", type=int, default=6000)
    parser.add_argument("--max-sequence", type=int, default=8192)
    parser.add_argument("--maximum-wave-size", type=int, default=200)
    parser.add_argument("--round1-answer-format", choices=ROUND1_ANSWER_FORMATS, default="reference_geometry")
    parser.add_argument(
        "--objective-mode",
        choices=OBJECTIVE_MODES,
        default="unified",
        help=(
            "Use all unified CAW routes, or train the matched standard-LA "
            "control from only each episode's original direct conversation."
        ),
    )
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument(
        "--selection-metric", choices=("macro_task_loss", "detection_grounding_macro_loss"),
        default="macro_task_loss",
    )
    parser.add_argument("--gradient-accumulation", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--keep-recent-checkpoints", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--vision-attention", choices=("auto", "sdpa", "flash_attention_2"), default="auto"
    )
    parser.add_argument("--wandb-project", default="caw-pivr-unified-v2")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    args = parser.parse_args()
    positive = (
        args.expected_train_records,
        args.expected_validation_records,
        args.lora_rank,
        args.image_token_limit,
        args.max_sequence,
        args.maximum_wave_size,
        args.max_steps,
        args.gradient_accumulation,
        args.learning_rate,
        args.warmup_steps,
        args.keep_recent_checkpoints,
    )
    if any(value <= 0 for value in positive):
        parser.error("Counts, limits, steps, learning rate, and warm-up must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
