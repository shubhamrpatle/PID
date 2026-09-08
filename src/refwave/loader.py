"""CPU workers compile episodes; only the trainer touches CUDA."""

import torch
from torch.utils.data import DataLoader, Subset


def make_loader(dataset, indices, workers):
    # Worker startup and validation must not consume the trainer's RNG on resume.
    options = dict(
        dataset=Subset(dataset, indices),
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(0),
    )
    if workers:
        options.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    return DataLoader(**options)
