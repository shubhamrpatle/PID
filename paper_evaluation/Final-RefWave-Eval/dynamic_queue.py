#!/usr/bin/env python3
"""Cap concurrent workers without changing the frozen RefWave inference protocol."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import io
import json
import os
import subprocess
import time

import remote_eval as evaluation


def control_hardware():
    """Require a single hardware/software reference from the completed controls."""
    identities, paths = set(), []
    for model in ('standard', 'class_isolated'):
        for kind in ('boxes', 'pointing'):
            for suite, count in evaluation.EXPECTED.items():
                path = evaluation.CONTROLS/'results/results'/kind/model/suite/'metrics.json'
                if not path.is_file():
                    continue
                metric = evaluation.read(path)
                contract = metric['contract']
                if not metric.get('complete') or metric['records'] != count or contract['limit']:
                    continue
                if not contract.get('gpu') or not contract.get('torch'):
                    raise ValueError(f'Missing GPU/Torch identity in {path}')
                identities.add((contract['gpu'].strip(), str(contract['torch'])))
                paths.append(str(path))
    if len(identities) != 1:
        raise ValueError('Need completed controls with one GPU model and Torch version. '
                         f'Found {sorted(identities)}. Do not silently mix hardware for latency.')
    name, version = next(iter(identities))
    return dict(gpu_model=name, torch=version, control_metric_files=paths)


def gpu_inventory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    devices = {}
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row:
            continue
        index, uuid, name, memory = (value.strip() for value in row)
        devices[index] = dict(index=index, uuid=uuid, name=name, memory_mb=int(float(memory)))
    return devices


def select_pool(requested, inventory, hardware):
    evaluation.validate_gpus(requested)
    eligible, skipped, seen = [], [], set()
    for requested_id in requested:
        matches = [d for d in inventory.values() if requested_id in (d['index'], d['uuid'])]
        if len(matches) != 1:
            raise ValueError(f'GPU {requested_id} does not uniquely identify a physical GPU')
        device = dict(matches[0], requested_id=requested_id)
        if device['uuid'] in seen:
            raise ValueError('The same physical GPU was requested twice using different aliases')
        seen.add(device['uuid'])
        (eligible if device['name'] == hardware['gpu_model'] else skipped).append(device)
    if not eligible:
        raise ValueError(f"No requested GPU matches the controls: {hardware['gpu_model']}")
    if len({d['memory_mb'] for d in eligible}) != 1:
        raise ValueError('Matching GPU names have different VRAM capacities; choose a homogeneous pool')
    return eligible, skipped


def validate_max_workers(value):
    if not 1 <= value <= 8:
        raise ValueError('MAX_WORKERS / --max-workers must be between 1 and 8')
    return value


def low_utilization(gpu):
    try:
        result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
        return result.returncode == 0 and 0 <= float(result.stdout.strip()) < 5
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


@contextmanager
def capped_scheduler(max_workers, policy):
    """The existing queue rechecks admission after every spawn, updating active first."""
    validate_max_workers(max_workers)
    queue = evaluation.queue
    original_phase, original_idle, original_status = queue.run_phase, queue.scheduling.idle, queue.write_status
    max_observed = 0

    def write_status(output, state, **kwargs):
        nonlocal max_observed
        max_observed = max(max_observed, len(kwargs.get('active', [])))
        original_status(output, state, max_workers=max_workers,
            candidate_gpus=policy['candidate_gpus'], eligible_gpus=policy['eligible_gpus'],
            gpu_model=policy['hardware']['gpu_model'], max_observed_workers=max_observed, **kwargs)

    def run_phase(args, smoke, active):
        def admit(gpu):
            try:
                return len(active) < max_workers and original_idle(gpu) and low_utilization(gpu)
            except (OSError, subprocess.SubprocessError, ValueError):
                # An unreadable GPU is unavailable, never permission to schedule onto it.
                return False

        queue.scheduling.idle = admit
        try:
            return original_phase(args, smoke, active)
        finally:
            queue.scheduling.idle = original_idle

    queue.run_phase, queue.write_status = run_phase, write_status
    try:
        yield
    finally:
        queue.run_phase, queue.scheduling.idle, queue.write_status = original_phase, original_idle, original_status


def prepare_schedule(gpus, max_workers):
    import torch
    validate_max_workers(max_workers)
    hardware = control_hardware()
    if str(torch.__version__) != hardware['torch']:
        raise ValueError(f"Torch differs from controls: {torch.__version__} vs {hardware['torch']}")
    eligible, skipped = select_pool(gpus, gpu_inventory(), hardware)
    return dict(candidate_gpus=gpus, eligible_gpus=[d['requested_id'] for d in eligible],
        max_workers=max_workers, hardware=hardware, devices=eligible, excluded_devices=skipped,
        admission='below 500 MiB used, no compute processes, and GPU utilization below 5%',
        timing='Queue waiting, checkpoint loading and warmup are excluded from sample latency.',
        limits='Matching GPU model/Torch does not eliminate node-level contention; not isolated timing.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['launch', 'run', 'smoke', 'inspect'])
    parser.add_argument('--gpus', nargs='+', default=['0', '1', '2', '3'])
    parser.add_argument('--max-workers', type=int, default=4)
    args = parser.parse_args()
    validate_max_workers(args.max_workers)
    evaluation.validate_gpus(args.gpus)
    policy = prepare_schedule(args.gpus, args.max_workers)
    print(json.dumps(dict(event='gpu_pool', **policy), indent=2), flush=True)
    if args.mode == 'inspect':
        return
    evaluation.OUTPUT.mkdir(parents=True, exist_ok=True)
    # Same lock as the old launcher: a second queue cannot overlap this evaluation.
    with (evaluation.OUTPUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log = evaluation.OUTPUT/'scheduling_history.jsonl'
        with log.open('a') as stream:
            stream.write(json.dumps(dict(started=time.time(), pid=os.getpid(), mode=args.mode,
                scheduler_sha256=evaluation.base.common.sha(__file__), **policy))+'\n')
        if args.mode == 'launch':
            evaluation.prepare()
        with capped_scheduler(args.max_workers, policy):
            evaluation.run(policy['eligible_gpus'], smoke_only=args.mode == 'smoke')


if __name__ == '__main__':
    main()
