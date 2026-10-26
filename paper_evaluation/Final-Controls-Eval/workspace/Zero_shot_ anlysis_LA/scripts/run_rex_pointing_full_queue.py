#!/usr/bin/env python3
"""Rex boxes first, then five native-point models; idle GPUs 2/3 only."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import evaluate_rex_pointing_full as evaluation

PYTHON = evaluation.ROOT / '.venv/bin/python'
SCRIPT = Path(evaluation.__file__).resolve()
MODEL_ORDER = ('qwen3_4b', 'qwen3_8b', 'rexomni', 'molmo_7b_d', 'base_la')


def environment(model, gpu=None):
    env = dict(os.environ, OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    paths = [str(evaluation.ROOT/'Pixel-PIVR/src'), str(evaluation.ROOT/'RefWave/src'),
             str(evaluation.ROOT), '/data1/shubham/paper_comparator_deps']
    if model == 'molmo_7b_d':
        paths.insert(0, '/data2/shubham/models/qualitative_molmo_legacy')
    env['PYTHONPATH'] = ':'.join(paths)
    env['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else str(gpu)
    return env


def idle(gpu):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if result.returncode or int(result.stdout.strip()) >= 500:
        return False
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                    '--format=csv,noheader'], text=True).splitlines()
    return uuid not in {s.strip() for s in apps}


def job_name(job):
    return f"{job['kind']}_{job['model']}_{job['suite']}_shard{job['shard_index']}of{job['shard_count']}"


def command(args, job, mode):
    batch_size = getattr(args, 'batch_size', 1)
    script = SCRIPT if batch_size == 1 else SCRIPT.with_name('evaluate_rex_pointing_batched.py')
    cmd = [str(PYTHON), '-u', str(script), mode, '--output', str(args.output),
        '--model', job['model'], '--kind', job['kind'], '--suite', job['suite'],
        '--shard-index', str(job['shard_index']), '--shard-count', str(job['shard_count']),
        '--limit', str(args.limit)]
    if batch_size != 1:
        cmd += ['--batch-size', str(batch_size)]
    return cmd


def merged(args, kind, model, suite):
    path = args.output/('smoke' if args.limit else 'results')/kind/model/suite/'metrics.json'
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    if args.batch_size != 1 and data.get('complete') is True:
        if data.get('contract', {}).get('requested_batch_size') != args.batch_size:
            raise ValueError('Completed result has a different batch contract: ' + str(path))
    return data.get('complete') is True


def run_phase(args, kind, models, active):
    jobs = [dict(kind=kind, model=model, suite=suite, shard_index=i, shard_count=args.shards)
            for model in models for suite in evaluation.SUITES for i in range(args.shards)
            if not merged(args, kind, model, suite)]
    failed = []
    status_path = args.output / ('queue_smoke_status.json' if args.limit else 'queue_status.json')
    while jobs or active:
        for gpu in args.gpus:
            if not jobs or any(t['gpu'] == gpu for t in active) or not idle(gpu):
                continue
            job = jobs.pop(0)
            name = job_name(job)
            log = (args.output/'logs'/((name+'_smoke' if args.limit else name)+'.log')).open('a')
            cmd = command(args, job, 'infer')
            log.write('\nCOMMAND '+json.dumps(cmd)+'\n'); log.flush()
            process = subprocess.Popen(cmd, cwd=evaluation.PROJECT, env=environment(job['model'],gpu),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active.append(dict(job=job, gpu=gpu, process=process, log=log))
            print(json.dumps(dict(event='start', phase=kind, job=name, gpu=gpu, pid=process.pid)), flush=True)
        for task in active[:]:
            rc = task['process'].poll()
            if rc is None:
                continue
            task['log'].close()
            active.remove(task)
            job = task['job']
            if rc:
                failed.append(dict(job=job, exitcode=rc))
                # Do not burn more compute on another shard of a failing model/task.
                jobs = [j for j in jobs if (j['model'],j['kind'],j['suite']) !=
                        (job['model'],job['kind'],job['suite'])]
                print('FAILED', job_name(job), rc, flush=True)
            else:
                print('INFERENCE COMPLETE', job_name(job), flush=True)
                folder = args.output/('smoke' if args.limit else 'results')/kind/job['model']/job['suite']
                if all((folder/f'shard{i:02d}of{args.shards:02d}.done.json').exists() for i in range(args.shards)):
                    result = subprocess.run(command(args, dict(job, shard_index=0), 'merge'),
                        cwd=evaluation.PROJECT, env=environment(job['model']), capture_output=True, text=True)
                    with (folder/'merge.log').open('a') as log:
                        log.write(result.stdout+result.stderr)
                    if result.returncode:
                        failed.append(dict(job=job, merge_exitcode=result.returncode))
                    else:
                        print('METRICS COMPLETE', kind, job['model'], job['suite'], flush=True)
                        if not args.limit:
                            if args.batch_size == 1:
                                from report_rex_pointing_full import report
                            else:
                                from evaluate_rex_pointing_batched import report
                            report(args.output)
        evaluation.common.write(status_path, dict(phase=kind, state='running', remaining=len(jobs),
            active=[dict(job=t['job'], gpu=t['gpu'], pid=t['process'].pid) for t in active], failures=failed,
            updated=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        if jobs or active:
            time.sleep(5)
    if failed:
        evaluation.common.write(status_path, dict(phase=kind, state='failed', failures=failed))
        raise RuntimeError('Evaluation failures; inspect per-job logs before continuing')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--batch-size', type=int, choices=[1,8,32], default=1)
    p.add_argument('--gpus', type=int, nargs='+', choices=[2,3], default=[2,3])
    p.add_argument('--shards', type=int, default=2)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--phase', choices=['boxes','pointing','both'], default='both')
    args = p.parse_args()
    if args.output is None:
        if args.batch_size == 1:
            args.output = evaluation.OUTPUT
        else:
            from evaluate_rex_pointing_batched import default_output
            args.output = default_output(args.batch_size)
    if args.shards < 1 or args.limit < 0 or (args.limit and args.shards != 1):
        p.error('Smoke uses one shard; shards must be positive')
    (args.output/'logs').mkdir(parents=True, exist_ok=True)
    lock = (args.output/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = []

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for kind, models in [('boxes', ('rexomni',)), ('pointing', MODEL_ORDER)]:
            if args.phase not in ('both',kind):
                continue
            for model in models:
                if not (args.output/'assets'/(model+'.json')).exists():
                    raise RuntimeError(f'Prepare and verify model first: {model}')
            run_phase(args,kind,models,active)
        name = ('all' if args.phase == 'both' else args.phase) + ('.smoke' if args.limit else '') + '.done.json'
        evaluation.common.write(args.output/name, dict(complete=True, phase=args.phase))
    except KeyboardInterrupt:
        for task in active:
            if task['process'].poll() is None:
                os.killpg(task['process'].pid, signal.SIGINT)
        for task in active:
            try:
                task['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(task['process'].pid, signal.SIGTERM)
                task['process'].wait(timeout=30)
            task['log'].close()
        evaluation.common.write(args.output/'queue_stopped.json', dict(reason='user_interrupt',
            partial_predictions_preserved=True, time=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        raise SystemExit(130)


if __name__ == '__main__':
    main()
