"""Exercise the real trainer on four CPU ranks, with tiny model/data substitutes."""

from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from refwave import train
from test_cluster_training import TinyModel


class TorchProxy:
    cuda = SimpleNamespace(
        manual_seed_all=lambda seed: None,
        get_rng_state=torch.get_rng_state,
        set_rng_state=torch.set_rng_state,
        max_memory_allocated=lambda: 0,
    )

    def device(self, *args):
        return torch.device("cpu")

    def __getattr__(self, key):
        return getattr(torch, key)


rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
root = Path(sys.argv[1])
method, tuning = sys.argv[2:4]
handlers = {}
seen = []
interrupt = True
delayed = False
group_number = 0


def context():
    global group_number
    if not dist.is_initialized():
        # Each production stage gets a new torchrun; use a fresh rendezvous here too.
        rendezvous = (root / f"rendezvous-{group_number}").as_uri()
        group_number += 1
        dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                                timeout=timedelta(seconds=60))
    return rank, rank, world


def load_base(*args):
    model = TinyModel()
    model.language_model.gradient_checkpointing_enable = lambda **kw: None
    model.language_model.enable_input_require_grads = lambda: None
    return model, None


class Wrapper(torch.nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.causal = model.language_model
        self.device = torch.device("cpu")

    def batch(self, batch, divisor=1, backward=True, **kwargs):
        global interrupt
        loss = sum(p.square().sum() for p in self.model.parameters() if p.requires_grad)
        if backward and batch:
            (loss * len(batch) / divisor).backward()
            seen.extend(batch)
            if rank == 1 and interrupt:
                interrupt = False
                handlers[signal.SIGTERM](signal.SIGTERM, None)
        return [dict(loss=float(loss.detach()), round1_loss=1., round2_loss=1.,
                     direct_loss=0., supervised_tokens=8, waves=2, task="detection")
                for _ in batch]


real_hash = train.sha256_file


def staggered_hash(path):
    global delayed
    if rank == 1 and not delayed:
        delayed = True
        deadline = time.monotonic() + 30
        while not (root / "stage1/config.json").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (root / "stage1/config.json").exists()
    return real_hash(path)


train.torch = TorchProxy()
train.distributed_context = context
train.load_base = load_base
train.ClusterEpisodeModel = Wrapper
train.ClusterDataset = lambda *a, **k: list(range(7))
train.make_loader = lambda dataset, indices, workers: iter(indices)
train.enable_fused_vision_attention = lambda model: "refwave_fused_per_image_sdpa"
train.signal.signal = lambda sig, handler: handlers.update({sig: handler})
train.sha256_file = staggered_hash
args = SimpleNamespace(
    seed=1, model=root / "model", image_token_limit=6000,
    tuning=tuning, lora_rank=32, method=method, stage="stage1",
    data_root=root, train_data=root / "records.jsonl", validation_data=root / "records.jsonl",
    max_sequence=32768, batch_size=1, accumulation=1, batch_tokens=8192,
    learning_rate=2e-5, head_lr=1e-6, vision_lr=1e-6, projector_lr=1e-5,
    warmup_ratio=.03, output=root / "stage1", init_checkpoint=None,
    workers=0, eval_steps=300, save_steps=300, log_steps=10,
    epochs=2, continue_checkpoint=None, wandb_mode="disabled",
)
try:
    train.run(args)
except SystemExit as error:
    assert error.code == 130
else:
    raise AssertionError("A stop on rank 1 must stop all ranks")
train.run(args)
stage1_seen = seen.copy()
seen.clear()
args.init_checkpoint = root / "stage1/best.pt"
args.output = root / "stage2"
args.stage = "stage2"
train.run(args)
stage2_seen = seen.copy()
seen.clear()
args.continue_checkpoint = root / "stage2/best.pt"
args.init_checkpoint = None
args.output = root / "extra_stage2"
args.epochs = 1
train.run(args)
(root / f"rank{rank}.json").write_text(json.dumps(
    dict(stage1=stage1_seen, stage2=stage2_seen, extra_stage2=seen)
))
