"""Small distributed helpers with no experiment or optional-kernel imports."""

import json
import os
from pathlib import Path
import torch
import torch.distributed as dist


def choose_vision_attention(requested):
    # Fixed SDPA avoids depending on a node-specific FlashAttention build.
    if requested not in ("auto", "sdpa"):
        raise ValueError("This release uses SDPA for Qwen and MoonViT")
    return "sdpa"


def distributed_context():
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device is visible. Run on an allocated GPU compute node.")
    if local >= torch.cuda.device_count():
        raise RuntimeError("Requested more processes than allocated visible GPUs")
    torch.cuda.set_device(local)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local, world


def synchronize_trainable_parameters(module, world_size):
    named = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
    if not named:
        raise RuntimeError("No trainable parameters")
    metadata = [(name, tuple(p.shape), str(p.dtype)) for name, p in named]
    if world_size > 1:
        replicas = [None] * world_size
        dist.all_gather_object(replicas, metadata)
        if any(replica != metadata for replica in replicas):
            raise RuntimeError("Trainable parameter metadata differs across ranks")
        with torch.no_grad():
            for _, parameter in named:
                dist.broadcast(parameter.data, src=0)


def truncate_curve(path: Path, step_key, maximum_step):
    if not path.is_file():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    temporary = path.with_name(f".{path.name}.resume-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row) + "\n" for row in rows if row[step_key] <= maximum_step)
    )
    os.replace(temporary, path)
