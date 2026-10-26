#!/usr/bin/env python3
"""Supplement frozen cluster evaluations with matched Base LA and pointing."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import evaluate_refwave_cluster_full_stage2 as full
import evaluate_refwave_cluster_stage2 as reference
from run_locany_pointing_balanced100 import parse_locateanything_points, match_points_to_gt, prf

OUTPUT = reference.PROJECT / 'runs/refwave_all_task_comparison_20260910'
MODELS = ('standard', 'refwave', 'refwave_full', 'base')


def point_branches(row):
    from pixel_pivr.class_wave import ClassPointBranch
    return [ClassPointBranch(s['canonical'], f"Point to: {s['prompt']}.",
            reference.stable_seed(0, 'DOTAv2-Pointing100', row['uid'], s['canonical']))
            for s in row['class_specs']]


def parse_points(answers, width, height):
    points, fallbacks = [], 0
    for answer in answers:
        parsed = parse_locateanything_points(answer['answer'], width, height, answer['label'])
        fallbacks += sum(p['source'] != 'point' for p in parsed)
        points.extend(p for p in parsed if p['source'] == 'point')
    return points, fallbacks


def score_points(rows, predictions):
    import numpy as np
    index = {p['uid']: p for p in predictions}
    if len(index) != len(predictions) or set(index) != {r['uid'] for r in rows}:
        raise ValueError('Pointing prediction IDs do not exactly match the frozen manifest')
    totals, per_class, per_image = Counter(), {}, []
    absent, absent_null = 0, 0
    for row in rows:
        pred = index[row['uid']]
        counts = Counter()
        for spec in row['class_specs']:
            label = spec['canonical']
            gt = [g['hbox'] for g in row['gt'] if g['label'] == label]
            points = [p for p in pred['points'] if p['label'] == label]
            match = match_points_to_gt(gt, points)
            values = {k: match[k] for k in ('tp', 'fp', 'fn')}
            counts.update(values)
            per_class.setdefault(label, Counter()).update(values)
            if not gt:
                absent += 1
                # Malformed/box-shaped answers are not credited as correct nulls.
                answer = next(a['answer'] for a in pred['answers'] if a['label'] == label)
                absent_null += int(not points and '<box>None</box>' in answer)
        totals.update(counts)
        per_image.append(dict(uid=row['uid'], **counts, **prf(**counts)))
    times = [p['latency_seconds'] for p in predictions]
    metrics = dict(records=len(rows), class_queries=sum(len(r['class_specs']) for r in rows),
        targets=sum(len(r['gt']) for r in rows), **totals, **prf(**totals),
        mean_seconds=float(np.mean(times)), p50_seconds=float(np.median(times)),
        p95_seconds=float(np.percentile(times, 95)), peak_gpu_mb=max(p['peak_gpu_mb'] for p in predictions),
        absent_queries=absent, absent_correct_explicit_null=absent_null,
        box_fallbacks_excluded=sum(p['box_fallbacks_excluded'] for p in predictions),
        per_class={k: dict(v, **prf(**v)) for k, v in per_class.items()})
    return metrics, per_image


def load(method):
    if method == 'refwave_full':
        return full.load_full_worker()
    if method != 'base':
        return reference.load_worker(method)
    from pixel_pivr.worker import LocateAnythingPixelPIVRWorker
    worker = LocateAnythingPixelPIVRWorker(reference.BASE_MODEL, device='cuda:0')
    worker.processor.image_processor.in_token_limit = 6000
    worker.model.eval().requires_grad_(False)
    return worker


def select_identity(method):
    if method == 'base':
        return dict(model_path=str(reference.BASE_MODEL), tuning='none',
                    model_index_sha256=reference.sha256_file(reference.BASE_MODEL/'model.safetensors.index.json'))
    if method == 'refwave_full':
        selected = json.loads((full.OUTPUT/'checkpoint.json').read_text())
    else:
        selected = json.loads((reference.OUTPUT/'checkpoints.json').read_text())['selections'][method]
    return {k: selected[k] for k in ('checkpoint', 'sha256', 'step')}


def infer(args):
    import torch
    from PIL import Image
    from refwave.io import read_jsonl_log
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    suites = ('DOTAv2-Pointing100',) if args.kind == 'pointing' else reference.SUITES
    worker = load(args.method)
    point_decoder = reference.make_decoders(worker, optimized=True)[0] if args.kind == 'pointing' else None
    for suite in suites:
        manifest_suite = 'DOTAv2-Balanced100' if args.kind == 'pointing' else suite
        manifest = reference.OUTPUT/'assets'/f'{manifest_suite}.jsonl'
        rows = reference.read_jsonl(manifest)[args.shard_index::args.shard_count]
        if args.limit:
            rows = rows[:args.limit]
        directory = args.output/('smoke' if args.limit else 'results')/suite/args.method
        directory.mkdir(parents=True, exist_ok=True)
        tag = f'shard{args.shard_index:02d}of{args.shard_count:02d}'
        path = directory/f'{tag}.jsonl'
        budget = 8192 if args.kind == 'pointing' else (2048 if '100' in suite else 8192)
        contract = dict(method=args.method, **select_identity(args.method),
            suite=suite, manifest_sha256=reference.sha256_file(manifest), image_token_limit=6000,
            max_new_tokens=budget, temperature=.7, top_p=.9, repetition_penalty=1.1,
            shard_index=args.shard_index, shard_count=args.shard_count, limit=args.limit,
            gpu=torch.cuda.get_device_name(0), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            source_sha256=reference.sha256_file(Path(__file__)), warmup_images=2,
            scoring=('Strict two-coordinate points, one-to-one point-in-GT-HBB matching; '
                     'all 18 classes, duplicates retained, no box-center fallback'
                     if args.kind == 'pointing' else
                     'Uncapped class-aware HBB IoU, raw/NMS separate; grounding first predicted box'),
            timing='CUDA synchronized preprocessing + decoding + parsing; excludes image IO, weights and scoring',
            pointing_prompt='Point to: {class}.' if args.kind == 'pointing' else None,
            class_wave=18 if args.kind == 'pointing' else None)
        cp = path.with_suffix('.contract.json')
        if cp.exists() and json.loads(cp.read_text()) != contract:
            raise ValueError(f'Output contract changed: {cp}')
        reference.atomic_json(cp, contract)
        existing = read_jsonl_log(path, repair_tail=True) if path.exists() else []
        done = {p['uid'] for p in existing}
        if len(done) != len(existing) or not done <= {r['uid'] for r in rows}:
            raise ValueError('Duplicate or foreign prediction IDs')

        def one(row):
            with Image.open(row['image_path']) as opened:
                image = opened.convert('RGB')
            if args.kind != 'pointing':
                result = reference.infer_one(worker, None, reference.inference_query(row), image, 'standard', budget)
                result.update(method='base', image_id=row['image_id'], dataset=row['dataset'])
                if row['task'] == 'grounding':
                    result.update(sample_id=row['sample_id'], query=row['query'])
                return result
            branches = point_branches(row)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            answers, stats = point_decoder.generate(image, branches, max_new_tokens=budget,
                temperature=.7, top_p=.9, repetition_penalty=1.1, generation_mode='hybrid')
            points, fallbacks = parse_points(answers, *image.size)
            torch.cuda.synchronize()
            return dict(uid=row['uid'], method=args.method, points=points, answers=answers,
                box_fallbacks_excluded=fallbacks, latency_seconds=time.perf_counter()-started,
                peak_gpu_mb=max(torch.cuda.max_memory_allocated()/1024**2, stats.get('peak_gpu_mb', 0)),
                stats=stats)

        if len(done) != len(rows):
            for row in rows[:2]:
                one(row)
            with path.open('a') as handle:
                for i, row in enumerate(rows, 1):
                    if row['uid'] in done:
                        continue
                    result = one(row)
                    handle.write(json.dumps(result, sort_keys=True)+'\n')
                    handle.flush()
                    print(f"{args.method} {suite} {tag} {i}/{len(rows)} seconds={result['latency_seconds']:.3f}", flush=True)
        reference.atomic_json(path.with_suffix('.done.json'), dict(records=len(rows)))
    del worker
    gc.collect()


def merge(args):
    suites = ('DOTAv2-Pointing100',) if args.kind == 'pointing' else reference.SUITES
    for suite in suites:
        manifest_suite = 'DOTAv2-Balanced100' if args.kind == 'pointing' else suite
        rows = reference.read_jsonl(reference.OUTPUT/'assets'/f'{manifest_suite}.jsonl')
        directory = args.output/'results'/suite/args.method
        paths = [directory/f'shard{i:02d}of{args.shard_count:02d}.jsonl' for i in range(args.shard_count)]
        if not all(p.with_suffix('.done.json').exists() for p in paths):
            raise ValueError(f'Incomplete {suite}/{args.method}')
        predictions = [r for p in paths for r in reference.read_jsonl(p)]
        contracts = [json.loads(p.with_suffix('.contract.json').read_text()) for p in paths]
        omit = {'shard_index', 'visible_devices'}
        if any({k:v for k,v in c.items() if k not in omit} !=
               {k:v for k,v in contracts[0].items() if k not in omit} for c in contracts):
            raise ValueError('Inconsistent shard contracts')
        metrics, per = (score_points(rows, predictions) if args.kind == 'pointing' else full.score(rows, predictions))
        metrics['contract'] = contracts[0]
        reference.jsonl(directory/'predictions.jsonl', predictions)
        reference.jsonl(directory/'per_record_metrics.jsonl', per)
        reference.atomic_json(directory/'metrics.json', metrics)
        print(json.dumps(dict(suite=suite, method=args.method, records=metrics['records'],
                             pointing_f1=metrics.get('f1'), raw=metrics.get('raw'), grounding=metrics.get('grounding'))), flush=True)


def queue(args):
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logdir = args.output/'logs'
    logdir.mkdir(exist_ok=True)
    # One whole pointing model per GPU; then sharded Base LA benchmark inference.
    jobs = [(m, 'pointing', 0, 1) for m in MODELS]
    active = []
    failure = False
    def available(gpu):
        result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True)
        return result.returncode == 0 and int(result.stdout.strip()) < 500
    def launch(job, gpu):
        method, kind, shard, count = job
        name = f'{method}_{kind}_{shard}'
        log = (logdir/f'{name}.log').open('a')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), 'infer', '--output', str(args.output),
                   '--method', method, '--kind', kind, '--shard-index', str(shard), '--shard-count', str(count)]
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=reference.ROOT)
        print(json.dumps(dict(event='start', job=name, gpu=gpu, pid=process.pid)), flush=True)
        return dict(job=job, gpu=gpu, process=process, log=log)
    for phase in range(2):
        reference.atomic_json(args.output/'status.json', dict(state='running',
            phase='pointing' if phase == 0 else 'base_benchmarks', gpus=args.gpus))
        if phase == 1:
            jobs = [('base', 'benchmarks', i, len(args.gpus)) for i in range(len(args.gpus))]
        while jobs or active:
            for gpu in args.gpus:
                if jobs and not any(j['gpu'] == gpu for j in active) and available(gpu):
                    active.append(launch(jobs.pop(0), gpu))
            for task in active[:]:
                rc = task['process'].poll()
                if rc is None:
                    continue
                task['log'].close()
                active.remove(task)
                print(json.dumps(dict(event='exit', job=task['job'], returncode=rc)), flush=True)
                if rc:
                    failure = True
                elif phase == 0:
                    merge(argparse.Namespace(output=args.output, kind='pointing', method=task['job'][0], shard_count=1))
            if failure:
                jobs.clear()
            if jobs or active:
                time.sleep(5)
        if failure:
            reference.atomic_json(args.output/'status.json', dict(state='failed', phase=phase))
            raise RuntimeError('Evaluation failed; see per-job logs. Existing jobs were not killed.')
    merge(argparse.Namespace(output=args.output, kind='benchmarks', method='base', shard_count=len(args.gpus)))
    reference.atomic_json(args.output/'all.done.json', dict(completed=True, pointing_models=list(MODELS), base_suites=list(reference.SUITES)))
    reference.atomic_json(args.output/'status.json', dict(state='completed'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('infer', 'merge', 'queue'))
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--kind', choices=('pointing', 'benchmarks'), default='pointing')
    p.add_argument('--method', choices=MODELS, default='base')
    p.add_argument('--gpus', nargs='+', type=int, default=[2, 3])
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    if args.kind == 'benchmarks' and args.method != 'base':
        p.error('Existing trained-model benchmarks must be reused, not rerun here')
    if len(set(args.gpus)) != len(args.gpus) or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        p.error('Invalid GPU/shard configuration')
    {'infer': infer, 'merge': merge, 'queue': queue}[args.mode](args)


if __name__ == '__main__':
    main()
