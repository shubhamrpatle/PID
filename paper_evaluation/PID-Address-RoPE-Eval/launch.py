#!/usr/bin/env python3
"""Resume the acceptance gate and eight paired shards on at most four idle GPUs."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from rerun_common import (HERE, DEFAULT_OUTPUT, DATASETS, prepare, finished, gate_done,
                         atomic, require)
from gpu_pool import GpuPool, query_smi


def eligible_devices(inventory, hardware, candidates):
    return {uuid.strip() for index, uuid, name, memory in inventory
            if int(index.strip()) in candidates
            and dict(name=name.strip(), memory_mb=int(memory.strip())) == hardware}


def jobs_for_phase(phase, output, plan, data):
    if phase == 'gate':
        return [] if gate_done(output, plan) else [('gate', None, 0)]
    require(gate_done(output, plan), 'Full evaluation cannot bypass the GPU acceptance gate')
    return [('worker', ds, shard) for ds in DATASETS for shard in range(plan['shards'])
            if not finished(output, ds, shard, plan, data)]


def complete(job, output, plan, data):
    mode, ds, shard = job
    return gate_done(output, plan) if mode == 'gate' else finished(output, ds, shard, plan, data)


def command_for(job, output):
    mode, ds, shard = job
    command = [sys.executable, '-u', str(HERE / 'address_eval.py'), mode, '--output', str(output)]
    if mode == 'worker':
        command += ['--dataset', ds, '--shard', str(shard)]
    return command


def queue(phase, args, plan, data, stopped):
    pending = jobs_for_phase(phase, args.output, plan, data)
    active, failures = [], []
    pool = GpuPool(max_gpus=args.max_workers, min_free_mb=30000, max_used_mb=500, max_utilization=5)
    logs = args.output / 'logs'
    logs.mkdir(exist_ok=True)
    while pending or active:
        for item in list(active):
            rc = item['process'].poll()
            if rc is None:
                continue
            item['log'].close()
            item['lease'].close()
            active.remove(item)
            if rc or not complete(item['job'], args.output, plan, data):
                failures.append(dict(job=item['job'], returncode=rc, log=str(item['path'])))
                print('FAILED', failures[-1], flush=True)
            else:
                print('DONE', item['job'], flush=True)
        if pending and not stopped and not failures:
            inventory = query_smi('--query-gpu=index,uuid,name,memory.total')
            eligible = eligible_devices(inventory, plan['hardware'], args.gpus)
            for device in pool.available({i['gpu'] for i in active}, eligible_uuids=eligible):
                if not pending:
                    break
                lease = pool.claim(device)
                if lease is None:
                    continue
                job = pending.pop(0)
                tag = 'gate' if job[0] == 'gate' else f'paired_{job[1]}_shard{job[2]:02d}of{plan["shards"]:02d}'
                path = logs / (tag + '.log')
                log = path.open('a')
                command = command_for(job, args.output)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device['uuid'], TOKENIZERS_PARALLELISM='false',
                           OMP_NUM_THREADS='4', RS_PROTOCOL_HF_HOME=str(HERE / '.hf-cache'))
                try:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                        env=env, start_new_session=True, pass_fds=(lease.fileno(),))
                except BaseException:
                    log.close()
                    lease.close()
                    raise
                active.append(dict(job=job, process=process, log=log, lease=lease, path=path, gpu=device['uuid']))
                print('START', tag, 'gpu', device['uuid'], 'pid', process.pid, flush=True)
        atomic(args.output / 'queue_status.json', dict(phase=phase, pending=pending, active=[
            dict(job=i['job'], pid=i['process'].pid, gpu=i['gpu']) for i in active],
            failures=failures, draining=bool(stopped), maximum_workers=args.max_workers, hardware=plan['hardware']))
        if (failures or stopped) and not active:
            raise SystemExit(1 if failures else 130)
        if pending or active:
            print(f'GPU QUEUE: {len(active)} running; {len(pending)} pending; phase {phase}; maximum {args.max_workers}', flush=True)
            time.sleep(args.poll_seconds)


def launch(args):
    require(1 <= args.max_workers <= 4, 'Maximum simultaneous workers must be 1-4')
    require(args.gpus and len(set(args.gpus)) == len(args.gpus) and all(0 <= g <= 7 for g in args.gpus),
            'Candidate GPUs must be distinct physical indices 0-7')
    require(args.poll_seconds > 0, 'Poll interval must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, data = prepare(args.output)
        stopped = []

        def drain(signum, frame):
            stopped.append(signum)
            print('Draining our active workers; no new GPU admissions.', flush=True)

        signal.signal(signal.SIGINT, drain)
        signal.signal(signal.SIGTERM, drain)
        queue('gate', args, plan, data, stopped)
        require(not stopped, 'Launch stopped after acceptance gate')
        queue('full', args, plan, data, stopped)
        require(not stopped, 'Launch stopped before scoring')
        subprocess.run([sys.executable, '-u', str(HERE / 'address_eval.py'), 'merge',
                        '--output', str(args.output)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--gpus', nargs='+', type=int, default=list(range(8)))
    p.add_argument('--max-workers', type=int, default=4)
    p.add_argument('--poll-seconds', type=float, default=15)
    args = p.parse_args()
    args.output = args.output.resolve()
    launch(args)


if __name__ == '__main__':
    main()
