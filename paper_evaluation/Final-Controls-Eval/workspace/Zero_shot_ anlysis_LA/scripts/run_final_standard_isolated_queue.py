#!/usr/bin/env python3
"""Queue the two final models after existing evaluations, on idle GPUs 2/3 only."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import evaluate_final_standard_isolated as evaluation
import run_rex_pointing_full_queue as scheduling


def process_identity(pid):
    path=Path(f'/proc/{pid}')
    try:
        command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        fields=(path/'stat').read_text().rsplit(') ',1)[1].split()
        return dict(pid=pid,start_ticks=fields[19],command=command)
    except FileNotFoundError:
        return None


def same_process(identity):
    return identity is not None and process_identity(identity['pid'])==identity


def jobs(smoke=False):
    if smoke:
        return [dict(model=model,kind=kind,suite=suite,shard_index=0,shard_count=1,limit=2)
                for model in evaluation.METHODS
                for kind,suite in [('boxes','DIOR'),('boxes','DIOR-RSVG'),('pointing','DIOR')]]
    return [dict(model=model,kind=kind,suite=suite,shard_index=i,shard_count=2,limit=0)
            for kind in ('boxes','pointing') for model in evaluation.METHODS
            for suite in evaluation.base.SUITES for i in range(2)]


def folder(output,job):
    return output/('smoke' if job['limit'] else 'results')/job['kind']/job['model']/job['suite']


def command(output,job,mode):
    cmd=[str(scheduling.PYTHON),'-u',str(Path(evaluation.__file__).resolve()),mode,'--output',str(output)]
    for name in ('model','kind','suite','shard_index','shard_count','limit'):
        cmd.extend(['--'+name.replace('_','-'),str(job[name])])
    return cmd


def job_name(job):
    return ('smoke_' if job['limit'] else '')+scheduling.job_name(job)


def write_status(output,state,**kwargs):
    evaluation.base.common.write(output/'queue_status.json',dict(state=state,
        updated=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs))


def run_phase(args,smoke,active):
    pending=jobs(smoke)
    failures=[]
    while pending or active:
        for gpu in args.gpus:
            if failures or not pending or any(t['gpu']==gpu for t in active) or not scheduling.idle(gpu):
                continue
            job=pending.pop(0)
            log=(args.output/'logs'/(job_name(job)+'.log')).open('a')
            cmd=command(args.output,job,'infer')
            log.write('\nCOMMAND '+json.dumps(cmd)+'\n');log.flush()
            process=subprocess.Popen(cmd,cwd=evaluation.base.PROJECT,
                env=scheduling.environment('base_la',gpu),stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True)
            active.append(dict(job=job,gpu=gpu,process=process,log=log))
            print(json.dumps(dict(event='start',job=job_name(job),gpu=gpu,pid=process.pid)),flush=True)
        for task in active[:]:
            rc=task['process'].poll()
            if rc is None:
                continue
            active.remove(task);task['log'].close()
            job=task['job'];directory=folder(args.output,job)
            if rc:
                failures.append(dict(job=job,exitcode=rc))
                print('FAILED',job_name(job),rc,flush=True)
                continue
            print('INFERENCE COMPLETE',job_name(job),flush=True)
            if all((directory/f"shard{i:02d}of{job['shard_count']:02d}.done.json").exists()
                   for i in range(job['shard_count'])):
                with (directory/'merge.log').open('a') as stream:
                    rc=subprocess.run(command(args.output,dict(job,shard_index=0),'merge'),
                        cwd=evaluation.base.PROJECT,env=scheduling.environment('base_la'),
                        stdout=stream,stderr=subprocess.STDOUT).returncode
                if rc:
                    failures.append(dict(job=job,merge_exitcode=rc))
                else:
                    print('METRICS COMPLETE',job['model'],job['kind'],job['suite'],flush=True)
        write_status(args.output,'running' if active else 'waiting_for_idle_gpu',
            phase='smoke' if smoke else 'full',remaining_shards=len(pending),failures=failures,
            active=[dict(job=t['job'],gpu=t['gpu'],pid=t['process'].pid) for t in active])
        if failures and not active:
            write_status(args.output,'failed',failures=failures,remaining_shards=len(pending))
            raise RuntimeError('Evaluation failed; no silent fallback, inspect logs before resuming')
        if pending or active:
            time.sleep(5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=evaluation.OUTPUT)
    p.add_argument('--gpus',type=int,nargs='+',choices=[2,3],default=[2,3])
    p.add_argument('--after-pid',type=int)
    p.add_argument('--download-pid',type=int)
    args=p.parse_args()
    if len(args.gpus)!=len(set(args.gpus)):
        p.error('Repeated GPU ID')
    (args.output/'logs').mkdir(parents=True,exist_ok=True)
    lock=(args.output/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dependency=process_identity(args.after_pid) if args.after_pid else None
    downloader=process_identity(args.download_pid) if args.download_pid else None
    evaluation.base.common.write(args.output/'queue_dependencies.json',
        dict(previous_evaluations=dependency,download=downloader,gpus=args.gpus))
    evaluation.base.common.write(args.output/'scheduled_jobs.json',
        dict(smoke=jobs(True),full=jobs(False),expected_cells=16))
    active=[]

    def stop(signum,frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        while not (args.output/'checkpoints_ready.json').exists() or same_process(dependency):
            ready=(args.output/'checkpoints_ready.json').exists()
            if not ready and args.download_pid and not same_process(downloader):
                write_status(args.output,'blocked_download',log=str(args.output/'logs/download.log'))
                raise RuntimeError('Downloader exited without verified checkpoints')
            write_status(args.output,'waiting_existing_evaluations' if ready else 'waiting_checkpoint_download',
                checkpoints_ready=ready,previous_queue_active=same_process(dependency),
                active=[],scheduled_cells=16)
            time.sleep(10)
        if not evaluation.release.read(args.output/'checkpoints_ready.json')['complete']:
            raise RuntimeError('Checkpoint verification incomplete')
        evaluation.check_sources(evaluation.release.read(args.output/'plan.json'))
        run_phase(args,True,active)
        run_phase(args,False,active)
        evaluation.report(args.output)
        if evaluation.release.read(args.output/'comparison.json')['completed_cells']!=16:
            raise RuntimeError('Expected all 16 model/benchmark cells')
        evaluation.base.common.write(args.output/'all.done.json',dict(complete=True,cells=16))
        write_status(args.output,'complete',completed_cells=16,active=[])
    except KeyboardInterrupt:
        for t in active:
            if t['process'].poll() is None:
                os.killpg(t['process'].pid,signal.SIGINT)
        for t in active:
            try:
                t['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(t['process'].pid,signal.SIGTERM)
                t['process'].wait(timeout=30)
            t['log'].close()
        write_status(args.output,'stopped',partial_outputs_preserved=True,active=[])
        raise SystemExit(130)


if __name__=='__main__':
    main()
