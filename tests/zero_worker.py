"""Two CPU processes test PyTorch optimizer consolidation and restoration."""

import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from refwave.runtime import StopRequest

dist.init_process_group("gloo")
rank = dist.get_rank()
torch.manual_seed(2)
model = torch.nn.Linear(3, 2)
optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW, lr=0.01)
model(torch.ones(2, 3)).square().mean().backward()
for p in model.parameters():
    dist.all_reduce(p.grad)
    p.grad.div_(dist.get_world_size())
optimizer.step()
optimizer.consolidate_state_dict(to=0)
path = Path(sys.argv[1])
if rank == 0:
    torch.save(optimizer.state_dict(), path)
dist.barrier()
saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
optimizer.load_state_dict(saved)
for p in model.parameters():
    values = [torch.empty_like(p) for _ in range(dist.get_world_size())]
    dist.all_gather(values, p.detach())
    for v in values:
        torch.testing.assert_close(v, p)
assert len(optimizer.optim.state) > 0
stop = StopRequest()
assert not stop.synchronize(torch.device("cpu"), dist.get_world_size())
if rank == 1:
    stop(15, None)
assert stop.synchronize(torch.device("cpu"), dist.get_world_size())
assert stop.synchronize(torch.device("cpu"), dist.get_world_size())
dist.destroy_process_group()
