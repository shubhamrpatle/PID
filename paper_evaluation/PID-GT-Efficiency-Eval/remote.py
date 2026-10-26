#!/usr/bin/env python3
"""At most four paired workers, admitted only on idle matching GPUs."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from pid_common import HERE, DATASETS, load_plan, finished, atomic, require
from gpu_pool import GpuPool, query_smi


def eligible_devices(inventory, hardware, candidates):
    return {uuid for index, uuid, name, memory in inventory
            if int(index.strip()) in candidates
            and dict(name=name.strip(), memory_mb=int(memory.strip())) == hardware}


def launch(args):
    require(1 <= args.max_gpus <= 4, 'Use 1-4 simultaneous workers')
    require(args.gpus and len(set(args.gpus)) == len(args.gpus) and all(0 <= g <= 7 for g in args.gpus),
            'Candidate GPUs must be distinct physical indices from 0 to 7')
    require(args.poll_seconds > 0, 'Poll interval must be positive')
    plan, data = load_plan(args.output)
    count = 1 if args.smoke else plan['shards']
    with (args.output / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = [(ds, i) for ds in DATASETS for i in range(count)
                   if not finished(args.output, ds, i, count, args.smoke, plan, data)]
        pool = GpuPool(max_gpus=args.max_gpus, min_free_mb=args.min_free_mb,
                       max_used_mb=500, max_utilization=5)
        active, failures, stopped = [], [], []

        def request_stop(signum, frame):
            stopped.append(signum)
            print('Draining active workers; no new admissions and no unrelated processes touched.', flush=True)

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
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
                ds, shard = item['job']
                if rc or not finished(args.output, ds, shard, count, args.smoke, plan, data):
                    failures.append(dict(job=item['job'], returncode=rc, log=str(item['path'])))
                    print('FAILED', failures[-1], flush=True)
                else:
                    print('DONE', item['job'], flush=True)
            if pending and not stopped and not failures:
                inventory = query_smi('--query-gpu=index,uuid,name,memory.total')
                eligible = eligible_devices(inventory, plan['hardware'], args.gpus)
                used = {i['gpu'] for i in active}
                for device in pool.available(used, eligible_uuids=eligible):
                    if not pending:
                        break
                    lease = pool.claim(device)
                    if lease is None:
                        continue
                    ds, shard = pending.pop(0)
                    tag = f'{"smoke_" if args.smoke else ""}paired_{ds}_shard{shard:02d}of{count:02d}'
                    path = logs / (tag + '.log')
                    log = path.open('a')
                    command = [sys.executable, '-u', str(HERE / 'paired_eval.py'), 'worker',
                               '--output', str(args.output), '--dataset', ds, '--shard', str(shard)]
                    if args.smoke:
                        command.append('--smoke')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device['uuid'],
                               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
                    try:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True, pass_fds=(lease.fileno(),))
                    except BaseException:
                        log.close()
                        lease.close()
                        raise
                    active.append(dict(job=(ds, shard), process=process, log=log, lease=lease,
                                       path=path, gpu=device['uuid']))
                    print('START', tag, 'gpu', device['uuid'], 'pid', process.pid, flush=True)
            atomic(args.output / 'queue_status.json', dict(pending=pending, active=[
                dict(job=i['job'], pid=i['process'].pid, gpu=i['gpu']) for i in active],
                failures=failures, draining=bool(stopped), smoke=args.smoke,
                maximum_workers=args.max_gpus, hardware=plan['hardware']))
            if (stopped or failures) and not active:
                raise SystemExit(1 if failures else 130)
            if pending or active:
                print(f'GPU QUEUE: {len(active)} running; {len(pending)} pending paired jobs; maximum {args.max_gpus}', flush=True)
                time.sleep(args.poll_seconds)
        command = [sys.executable, '-u', str(HERE / 'paired_eval.py'), 'merge', '--output', str(args.output)]
        if args.smoke:
            command.append('--smoke')
        subprocess.run(command, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE / 'results')
    p.add_argument('--gpus', nargs='+', type=int, default=list(range(8)))
    p.add_argument('--max-gpus', type=int, default=int(os.environ.get('MAX_WORKERS', '4')))
    p.add_argument('--min-free-mb', type=int, default=30000)
    p.add_argument('--poll-seconds', type=float, default=15)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    args.output = args.output.resolve()
    launch(args)


if __name__ == '__main__':
    main()
