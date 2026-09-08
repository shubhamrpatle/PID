"""CPU workers compile episodes; only the trainer touches CUDA."""

from torch.utils.data import DataLoader, Subset


def make_loader(dataset, indices, workers):
    options = dict(
        dataset=Subset(dataset, indices),
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    if workers:
        options.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    return DataLoader(**options)
