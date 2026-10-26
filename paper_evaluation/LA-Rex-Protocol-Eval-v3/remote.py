#!/usr/bin/env python3
"""Queue the frozen RS protocol on at most four idle allocated GPUs."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import HERE, read, atomic, freeze, jobs as make_jobs
from evaluate import load_plan, paths, finished
from gpu_pool import GpuPool, query_smi


def jobs(plan, smoke):
    count = 1 if smoke else plan['shards']
    return make_jobs(plan['run_models'], count)


def worker_command(output, model, kind, dataset, shard, smoke, plan):
    python = plan.get('laedino_python') if model == 'laedino' else sys.executable
    if not python or not Path(python).is_file():
        raise ValueError('Set --laedino-python when freezing the plan; use its separate MMDetection environment.')
    command = [python, '-u', str(HERE/'evaluate.py'), 'worker',
               '--output', str(output), '--model', model, '--kind', kind, '--dataset', dataset,
               '--shard', str(shard)]
    if smoke:
        command.append('--smoke')
    return command


def launch(args):
    with (args.output/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return launch_locked(args)


def launch_locked(args):
    if not 1 <= args.max_gpus <= 4:
        raise ValueError('Use one to four simultaneous GPUs.')
    plan, examples = load_plan(args.output)
    candidates = jobs(plan, args.smoke)
    for model,kind,dataset,shard,count in candidates:
        worker_command(args.output,model,kind,dataset,shard,args.smoke,plan)
    pending = [j for j in candidates if not finished(args.output, *j, args.smoke, plan, examples)]
    pool = GpuPool(max_gpus=args.max_gpus, min_free_mb=args.min_free_mb,
                   max_used_mb=500, max_utilization=5)
    active, failures, stopped = [], [], []
    def request_stop(signum, frame):
        stopped.append(signum)
        print('Stopping new admissions; allowing current workers to finish. No unrelated jobs touched.', flush=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    pin = args.output/'hardware.json'
    hardware = read(pin) if pin.exists() else None
    logs = args.output/'logs'
    logs.mkdir(exist_ok=True)
    while pending or active:
        for item in list(active):
            rc = item['process'].poll()
            if rc is None:
                continue
            item['log'].close()
            item['lease'].close()
            active.remove(item)
            if rc or not finished(args.output, *item['job'], args.smoke, plan, examples):
                failures.append(dict(job=item['job'], returncode=rc, log=str(item['log_path'])))
                print('FAILED', failures[-1], flush=True)
            else:
                print('DONE', item['job'], flush=True)
        if pending and not stopped and not failures:
            used = {i['gpu'] for i in active}
            names = {uuid: dict(name=name.strip(), memory_mb=int(memory.strip()))
                     for uuid, name, memory in query_smi('--query-gpu=uuid,name,memory.total')}
            eligible = {uuid for uuid, identity in names.items() if identity == hardware} if hardware else None
            for device in pool.available(used, eligible_uuids=eligible):
                if not pending:
                    break
                identity = names[device['uuid']]
                if hardware and identity != hardware:
                    continue
                lease = pool.claim(device)
                if lease is None:
                    continue
                if hardware is None:
                    hardware = identity
                    freeze(pin, hardware)
                model, kind, dataset, shard, count = pending.pop(0)
                label = f'{"smoke_" if args.smoke else ""}{model}_{kind}_{dataset}_shard{shard:02d}of{count:02d}'
                path = logs / (label + '.log')
                log = path.open('a')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device['uuid'],
                           TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
                try:
                    process = subprocess.Popen(worker_command(args.output, model, kind, dataset, shard, args.smoke, plan),
                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        pass_fds=(lease.fileno(),))
                except BaseException:
                    log.close()
                    lease.close()
                    raise
                active.append(dict(job=(model, kind, dataset, shard, count), process=process, log=log,
                                   lease=lease, gpu=device['uuid'], log_path=path))
                print('START', label, 'gpu', device['uuid'], 'pid', process.pid, flush=True)
        atomic(args.output/'queue_status.json', dict(pending=pending, active=[
            dict(job=i['job'], pid=i['process'].pid, gpu=i['gpu']) for i in active],
            failures=failures, draining=bool(stopped), smoke=args.smoke))
        if (stopped or failures) and not active:
            raise SystemExit(1 if failures else 130)
        if pending or active:
            print(f'GPU QUEUE: {len(active)} running; {len(pending)} pending; maximum {args.max_gpus}', flush=True)
            time.sleep(args.poll_seconds)
    command = [sys.executable, str(HERE/'evaluate.py'), 'merge', '--output', str(args.output)]
    if args.smoke:
        command.append('--smoke')
    subprocess.run(command, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE/'results')
    p.add_argument('--max-gpus', type=int, default=4)
    p.add_argument('--min-free-mb', type=int, default=30000)
    p.add_argument('--poll-seconds', type=float, default=15)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.poll_seconds <= 0:
        p.error('poll-seconds must be positive.')
    args.output = args.output.resolve()
    launch(args)


if __name__ == '__main__':
    main()
