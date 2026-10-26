#!/usr/bin/env python3
"""Fresh full-FT comparison: direct LA, fused sequential and fused wave RefWave."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import evaluate_refwave_cluster_stage2 as ref
import evaluate_refwave_cluster_full_stage2 as full
import evaluate_refwave_remaining_tasks as pointing
import run_refwave_inference_ablations as ablation
from refwave.io import append_jsonl, read_jsonl_log


OUTPUT = ref.PROJECT / 'runs/refwave_full_onepass_benchmarks_20260912'
DIRECT_MODEL = Path('/data1/tmp_shubham/locany_models/type2_stage2_packed_bestloss3000')
SUITES = ('DOTAv2-Balanced100', 'DIOR-RSVG-Unseen100', 'DOTAv2-Pointing100',
          'DIOR', 'DOTAv2', 'DIOR-RSVG', 'VRSBench-VG', 'LAE-80C')
LABELS = dict(la_full_direct='LA full FT Type-2 v8 best3000',
              refwave_full_seq='RefWave full FT one-pass sequential',
              refwave_full_wave='RefWave full FT one-pass Wave-200')


def read(path):
    return json.loads(Path(path).read_text())


def arms(family):
    return ['la_full_direct'] if family == 'direct' else ['refwave_full_seq', 'refwave_full_wave']


def file_identity(path):
    return dict(sha256=ref.sha256_file(path), bytes=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns)


def prepare(args):
    import torch
    from PIL import Image
    if (args.output / 'plan.json').exists():
        old = read(args.output / 'plan.json')
        if (old['suites'] != args.suites or old['smoke_records'] != args.smoke_records
                or old['shards'] != args.shards):
            raise ValueError('Existing plan differs; use another output directory')
        return old
    args.output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output).free < 10 * 1024**3:
        raise RuntimeError('Need 10 GiB free for new evaluation outputs; no files removed')
    selected = read(full.OUTPUT / 'checkpoint.json')
    print('Verifying full-FT checkpoint identity (no download or export).', flush=True)
    full_id = file_identity(full.CHECKPOINT)
    if full_id['sha256'] != selected['sha256'] or full_id['sha256'] != full.CHECKPOINT_SHA:
        raise ValueError('Full RefWave checkpoint differs from the selected model')
    saved = torch.load(full.CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
    full.validate_checkpoint(saved, read(full.CHECKPOINT.with_name('config.json')),
        ref.read_jsonl(full.CHECKPOINT.with_name('validation_curve.jsonl')),
        read(full.CHECKPOINT.with_name('done.json')),
        tied_output_head=read(ref.BASE_MODEL/'config.json')['text_config']['tie_word_embeddings'])
    del saved
    gc.collect()
    index = read(DIRECT_MODEL / 'model.safetensors.index.json')
    export = read(DIRECT_MODEL / 'inference_export_manifest.json')
    if read(DIRECT_MODEL / 'trainer_state.json')['global_step'] != 3000:
        raise ValueError('Wrong historical direct full-FT checkpoint')
    if 'best_training_loss/checkpoint-3000' not in export['source_training_checkpoint']:
        raise ValueError('Direct model export does not identify the intended checkpoint')
    direct_files = sorted(set(index['weight_map'].values()))
    if set(direct_files) != set(export['model_shards']):
        raise ValueError('Incomplete direct full-FT export')
    print('Verifying direct full-FT weight shards.', flush=True)
    weights = {str(full.CHECKPOINT): full_id}
    weights.update({str(DIRECT_MODEL/name): file_identity(DIRECT_MODEL/name) for name in direct_files})
    sources = [Path(__file__), Path(ref.__file__), Path(full.__file__), Path(pointing.__file__),
               Path(ablation.__file__), Path(ref.detection.__file__), Path(ref.grounding.__file__),
               ref.PROJECT/'scripts/run_locany_pointing_balanced100.py',
               ref.PROJECT/'scripts/evaluate_pixel_pivr_sequential_wave_balanced100.py']
    for directory in (ref.ROOT/'Pixel-PIVR/src/pixel_pivr', ref.ROOT/'RefWave/src/refwave',
                      ref.ROOT/'zero_shot_analysis', ref.BASE_MODEL, DIRECT_MODEL):
        sources.extend(directory.glob('*.py'))
    sources.extend(DIRECT_MODEL/name for name in ('config.json', 'tokenizer_config.json',
                   'model.safetensors.index.json', 'preprocessor_config.json'))
    source_hashes = {str(p.resolve()): ref.sha256_file(p) for p in sources}
    manifests, tasks, dimensions = {}, [], {}
    previous = read(ref.OUTPUT/'assets/manifest_summary.json')
    for suite in args.suites:
        original_suite = 'DOTAv2-Balanced100' if suite == 'DOTAv2-Pointing100' else suite
        source = ref.OUTPUT/'assets'/f'{original_suite}.jsonl'
        if ref.sha256_file(source) != previous[original_suite]['sha256']:
            raise ValueError('Frozen benchmark manifest changed')
        rows = ref.read_jsonl(source)
        if args.smoke_records:
            rows = rows[:args.smoke_records]
        if suite == 'DOTAv2-Pointing100':
            rows = [dict(r, suite=suite, task='pointing') for r in rows]
        if not rows or len({r['uid'] for r in rows}) != len(rows):
            raise ValueError('Empty or duplicate evaluation IDs')
        for row in rows:
            path = row['image_path']
            if path not in dimensions:
                with Image.open(path) as image:
                    dimensions[path] = image.size
            if dimensions[path] != (row['width'], row['height']):
                raise ValueError(f'Wrong image dimensions: {path}')
        target = args.output/'assets'/f'{suite}.jsonl'
        ref.jsonl(target, rows)
        budget = 8192 if suite == 'DOTAv2-Pointing100' or '100' not in suite else 2048
        manifests[suite] = dict(path=str(target), sha256=ref.sha256_file(target),
            source=str(source), source_sha256=previous[original_suite]['sha256'],
            records=len(rows), images=len({r['image_path'] for r in rows}),
            targets=sum(len(r['gt']) if r['task'] != 'grounding' else 1 for r in rows),
            task=rows[0]['task'], budget=budget,
            protocol='Existing frozen protocol; detection DIOR/DOTAv2 use the LAE evaluation manifests')
        count = min(args.shards, len(rows))
        for shard in range(count):
            # Small diagnostic suites first, with both models interleaved.
            for family in ('direct', 'refwave'):
                tasks.append(dict(id=f'{suite}__{family}__{shard:02d}of{count:02d}',
                    suite=suite, family=family, shard_index=shard, shard_count=count,
                    records=len(rows[shard::count])))
    plan = dict(schema='refwave-full-onepass-benchmarks-v1', suites=args.suites, shards=args.shards,
        smoke_records=args.smoke_records, manifests=manifests, tasks=tasks,
        models=dict(refwave_full=dict(checkpoint=str(full.CHECKPOINT), step=selected['step'],
                criterion=selected['criterion'], sha256=selected['sha256']),
            la_full_direct=dict(model=str(DIRECT_MODEL), step=3000,
                criterion='historical Type-2 v8 best training loss; export provenance verified',
                source_checkpoint=export['source_training_checkpoint'])),
        weights=weights, sources=source_hashes, image_token_limit=6000,
        round2_execution='fused_one_pass', class_wave=18, box_waves=[1, 200],
        generation=dict(temperature=.7, top_p=.9, repetition_penalty=1.1, mode='hybrid', seed=0),
        raw_policy='All parsed predictions, duplicate addresses retained; no containment, NMS or maxDets in headline.',
        pointing_policy='Point to: {class}. All 18 classes; strict two-coordinate points, no box-center fallback; one-to-one point-in-HBB.',
        pointing_execution='RefWave seq/wave rows compare Round-1 class waves 1/18 only; no Round 2. Direct full-FT uses the same class-wave-18 pointing protocol.',
        budget_policy='Direct per-query budget versus RefWave per-class discovery budget plus one PBD6 per address.',
        timing='CUDA-synchronized preprocessing + all decoding + parsing; excludes model loading, warmup, image file I/O and scoring.',
        caveats=['Different full-FT training recipes and checkpoint criteria; not a matched training ablation.',
                 'Single checkpoint/seed; no across-training-seed claim.',
                 'Pointing100 is a derived diagnostic, not an official independent pointing benchmark.',
                 'Other users may run on other GPUs; timings are not from an isolated node.',
                 'Sequential/wave BF16 equality is measured, not assumed.'], training=False)
    for path in sources:
        try:
            relative = path.resolve().relative_to(ref.ROOT)
        except ValueError:
            relative = Path('external_model')/path.name
        dest = args.output/'source_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    ref.atomic_json(args.output/'plan.json', plan)
    print(json.dumps({s: {k: v for k, v in m.items() if k in ('records','images','targets','budget')}
                      for s,m in manifests.items()}, indent=2), flush=True)
    return plan


def output_path(output, job, arm):
    return output/'results'/job['suite']/arm/f"shard{job['shard_index']:02d}of{job['shard_count']:02d}.jsonl"


def assert_fused(result):
    stats = result['round2']
    if (stats['round2_execution'] != 'fused_one_pass'
            or stats['qwen_branch_suffix_forwards'] != 0 or stats['qwen_geometry_forwards'] != 0
            or stats['qwen_total_forwards'] != stats['qwen_fused_prompt_geometry_forwards']):
        raise ValueError('Round 2 did not execute the fused one-pass route')
    if result['image_encodes'] != 1 or result['prefix_forwards'] != 1:
        raise ValueError('Unexpected repeated image/prefix encoding')


def point_one(worker, decoders, row, image, arm, budget):
    import torch
    branches = pointing.point_branches(row)
    point_decoder = decoders[0]
    point_decoder.max_wave_size = 1 if arm == 'refwave_full_seq' else 18
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    answers, stats = point_decoder.generate(image, branches, max_new_tokens=budget,
        temperature=.7, top_p=.9, repetition_penalty=1.1, generation_mode='hybrid')
    points, fallbacks = pointing.parse_points(answers, *image.size)
    torch.cuda.synchronize()
    return dict(uid=row['uid'], suite=row['suite'], task='pointing', points=points, answers=answers,
        box_fallbacks_excluded=fallbacks, latency_seconds=time.perf_counter()-started,
        peak_gpu_mb=max(torch.cuda.max_memory_allocated()/1024**2, stats.get('peak_gpu_mb', 0)),
        stats=stats, round2_execution=None, class_wave=point_decoder.max_wave_size)


def worker(args):
    import torch
    from PIL import Image
    from pixel_pivr.worker import LocateAnythingPixelPIVRWorker
    plan = read(args.output/'plan.json')
    job = next(j for j in plan['tasks'] if j['id'] == args.job)
    for path, sha in plan['sources'].items():
        if ref.sha256_file(Path(path)) != sha:
            raise ValueError(f'Code changed after preparation: {path}')
    for path, item in plan['weights'].items():
        stat = Path(path).stat()
        if stat.st_size != item['bytes'] or stat.st_mtime_ns != item['mtime_ns']:
            raise ValueError('Checkpoint changed after hash verification')
    info = plan['manifests'][job['suite']]
    if ref.sha256_file(Path(info['path'])) != info['sha256']:
        raise ValueError('Manifest changed after preparation')
    rows = ref.read_jsonl(Path(info['path']))[job['shard_index']::job['shard_count']]
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    previous, contracts = {}, {}
    for arm in arms(job['family']):
        path = output_path(args.output, job, arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        contract = dict(plan_sha256=ref.sha256_file(args.output/'plan.json'), **job, arm=arm,
            manifest_sha256=info['sha256'], round2_execution=None if info['task']=='pointing' or job['family']=='direct' else 'fused_one_pass',
            class_wave=(1 if arm=='refwave_full_seq' else 18) if info['task']=='pointing' else 18,
            box_wave=None if info['task']=='pointing' or job['family']=='direct' else (1 if arm=='refwave_full_seq' else 200),
            max_new_tokens=info['budget'], gpu=torch.cuda.get_device_name(0),
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=torch.__version__)
        cp = path.with_suffix('.contract.json')
        if cp.exists() and read(cp) != contract:
            # A fresh process may resume on another identical GPU; each row also records its GPU.
            ignored = {'visible_devices'}
            if {k:v for k,v in read(cp).items() if k not in ignored} != {k:v for k,v in contract.items() if k not in ignored}:
                raise ValueError('Existing prediction contract differs')
        else:
            ref.atomic_json(cp, contract)
        records = read_jsonl_log(path, repair_tail=True) if path.exists() else []
        done = {r['uid'] for r in records}
        if len(done) != len(records) or not done <= {r['uid'] for r in rows}:
            raise ValueError('Duplicate or foreign predictions')
        previous[arm], contracts[arm] = done, contract
    if all(len(d)==len(rows) for d in previous.values()):
        for arm in previous:
            ref.atomic_json(output_path(args.output,job,arm).with_suffix('.done.json'),
                            dict(records=len(rows), plan_sha256=contracts[arm]['plan_sha256']))
        return
    print(f"Loading {job['family']} for {job['id']} on {torch.cuda.get_device_name(0)}", flush=True)
    if job['family'] == 'refwave':
        model = full.load_full_worker()
    else:
        model = LocateAnythingPixelPIVRWorker(DIRECT_MODEL, device='cuda:0')
        model.model.eval().requires_grad_(False)
        model.processor.image_processor.in_token_limit = 6000
    decoders = ref.make_decoders(model, optimized=True, round2_execution='fused_one_pass') \
        if job['family']=='refwave' or info['task']=='pointing' else None

    def one(row, image, arm):
        if info['task'] == 'pointing':
            result = point_one(model, decoders, row, image, arm, info['budget'])
        else:
            config = dict(budget=info['budget'], class_wave=18,
                          box_wave=1 if arm=='refwave_full_seq' else 200)
            result = ablation.infer_one(model, decoders, ref.inference_query(row), image,
                'standard' if job['family']=='direct' else 'refwave_full', config)
            if job['family']=='refwave':
                assert_fused(result)
        result.update(method=arm, image_id=row['image_id'], dataset=row['dataset'],
                      visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        if info['task']=='grounding':
            result.update(sample_id=row['sample_id'], query=row['query'])
        return result

    for row in rows[:min(2, len(rows))]:
        with Image.open(row['image_path']) as opened:
            image = opened.convert('RGB')
        for arm in previous:
            one(row, image, arm)
    for index, row in enumerate(rows):
        with Image.open(row['image_path']) as opened:
            image = opened.convert('RGB')
        if image.size != (row['width'], row['height']):
            raise ValueError('Image dimensions changed')
        order = list(previous)
        if (index * job['shard_count'] + job['shard_index']) % 2:
            order.reverse()
        for arm in order:
            if row['uid'] in previous[arm]:
                continue
            result = one(row, image, arm)
            append_jsonl(output_path(args.output, job, arm), result)
            previous[arm].add(row['uid'])
            ref.atomic_json(args.output/'progress'/f"{job['id']}.json",
                dict(job=job['id'], completed=sum(map(len,previous.values())),
                     expected=len(rows)*len(previous), uid=row['uid'], arm=arm,
                     last_seconds=result['latency_seconds'], time=time.time()))
            print(f"{job['id']} {arm} {index+1}/{len(rows)} seconds={result['latency_seconds']:.3f}", flush=True)
    for arm in previous:
        ref.atomic_json(output_path(args.output,job,arm).with_suffix('.done.json'),
                        dict(records=len(rows), plan_sha256=contracts[arm]['plan_sha256']))
    del decoders, model
    gc.collect()
    torch.cuda.empty_cache()


def summarize_suite(output, plan, suite):
    import numpy as np
    info = plan['manifests'][suite]
    rows = ref.read_jsonl(Path(info['path']))
    results, metrics = {}, {}
    for arm in LABELS:
        family = 'direct' if arm=='la_full_direct' else 'refwave'
        jobs = [j for j in plan['tasks'] if j['suite']==suite and j['family']==family]
        paths = [output_path(output,j,arm) for j in jobs]
        if not all(p.with_suffix('.done.json').exists() for p in paths):
            return False
        predictions = []
        for path, job in zip(paths, jobs):
            contract = read(path.with_suffix('.contract.json'))
            if contract['plan_sha256'] != ref.sha256_file(output/'plan.json') or contract['arm'] != arm:
                raise ValueError('Cannot merge different evaluation contracts')
            records = ref.read_jsonl(path)
            expected = {r['uid'] for r in rows[job['shard_index']::job['shard_count']]}
            if len(records)!=len(expected) or {p['uid'] for p in records}!=expected:
                raise ValueError('Shard contains missing or duplicate records')
            predictions.extend(records)
        indexed = {p['uid']:p for p in predictions}
        if len(predictions)!=len(indexed) or set(indexed)!={r['uid'] for r in rows}:
            raise ValueError('Merged predictions differ from frozen manifest')
        predictions = [indexed[r['uid']] for r in rows]
        if info['task']=='pointing':
            value, per = pointing.score_points(rows,predictions)
        else:
            value, per = full.score(rows,predictions)
            elapsed = sum(p['latency_seconds'] for p in predictions)
            value['bps'] = sum(len(p['predictions']) for p in predictions)/elapsed
            if info['task']=='detection':
                value['cbps_iou0p5'] = value['raw']['0.5']['tp']/elapsed
            if arm!='la_full_direct':
                value['round1_mean_seconds'] = float(np.mean([p['round1_seconds'] for p in predictions]))
                value['round2_mean_seconds'] = float(np.mean([p['round2_seconds'] for p in predictions]))
                value['point_diagnostics'] = ablation.point_diagnostics(rows,predictions)
                value['oom_retries'] = sum(p['round2'].get('oom_retries',0) for p in predictions)
            value['budget_saturated_branches'] = sum(p['budget_saturated_branches'] for p in predictions)
            value['generated_branches'] = sum(p['generated_branches'] for p in predictions)
        value.update(plan_sha256=ref.sha256_file(output/'plan.json'), label=LABELS[arm],
            round2_execution='fused_one_pass' if arm!='la_full_direct' and info['task']!='pointing' else None)
        directory = output/'results'/suite/arm
        ref.atomic_json(directory/'metrics.json',value)
        ref.jsonl(directory/'per_record_metrics.jsonl',per)
        ref.jsonl(directory/'predictions.jsonl',predictions)
        metrics[arm], results[arm] = value, predictions
    seq, wave = results['refwave_full_seq'], results['refwave_full_wave']
    if info['task']=='pointing':
        equivalence = dict(records=len(rows), exact_answers=sum(a['answers']==b['answers'] for a,b in zip(seq,wave)),
            exact_points=sum(a['points']==b['points'] for a,b in zip(seq,wave)))
    else:
        equivalence = dict(records=len(rows), exact_round1=sum(a['point_answers']==b['point_answers'] for a,b in zip(seq,wave)),
            exact_geometry=sum([p['block_tokens'] for p in a['refinements']]==[p['block_tokens'] for p in b['refinements']]
                               for a,b in zip(seq,wave)),
            exact_prediction_multisets=sum(ablation.signature(a)==ablation.signature(b) for a,b in zip(seq,wave)))
    ref.atomic_json(output/'results'/suite/'comparison.json',dict(metrics=metrics,equivalence=equivalence))
    return True


def summary(output, plan):
    comparisons = {s:read(output/'results'/s/'comparison.json') for s in plan['suites']
                   if (output/'results'/s/'comparison.json').exists()}
    ref.atomic_json(output/'comparison.json',dict(plan_sha256=ref.sha256_file(output/'plan.json'),
        complete=len(comparisons)==len(plan['suites']), smoke_records=plan['smoke_records'],
        suites=comparisons, caveats=plan['caveats']))
    lines = ['# Full-FT one-pass evaluation comparison', '',
        'Fresh local measurements. RefWave full FT best validation step 2400 vs historical Type-2 v8 full FT best training-loss step 3000.',
        'Different training recipes: this is not a matched full-FT training ablation.',
        'Detection: uncapped raw class-aware IoU; NMS0.5 is separate in metrics.json. Grounding: first predicted box.',
        'Pointing: derived all-class diagnostic, strict point-in-HBB, no Round 2 or box-center fallback.',
        'Native-6000 patch budget; same prompts/settings within each protocol. Detection Round 1 fixed at class-wave 18.',
        'TPI is end-to-end inference, excluding loading, warmup, image disk I/O and scoring; other users may occupy other GPUs.',
        'One-pass is one Round-2 prompt+PBD6 forward per wave, not one forward per image.',
        f"Completed suites: {len(comparisons)}/{len(plan['suites'])}. Smoke subset: {plan['smoke_records']}.", '',
        '| Set | Model / execution | Records | P / Acc@0.5 | R / Acc@0.7 | F1 / mIoU | F1@0.75 | TPI s | p95 s | Peak MiB | BPS | CBPS |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for suite, report in comparisons.items():
        for arm,m in report['metrics'].items():
            task = plan['manifests'][suite]['task']
            values = m['raw']['0.5'] if task=='detection' else m['grounding'] if task=='grounding' else m
            keys = ('first_acc_0_5','first_acc_0_7','first_mean_iou') if task=='grounding' else ('precision','recall','f1')
            a,b,c = [100*values[k] for k in keys]
            f75 = f"{100*m['raw']['0.75']['f1']:.2f}" if task=='detection' else '-'
            bps = f"{m['bps']:.2f}" if 'bps' in m else '-'
            cbps = f"{m['cbps_iou0p5']:.2f}" if 'cbps_iou0p5' in m else '-'
            label = LABELS[arm]
            if task=='pointing' and arm!='la_full_direct':
                label = 'RefWave full FT Round-1 ' + ('sequential' if arm.endswith('seq') else 'Wave-18')
            lines.append(f"| {suite} | {label} | {m['records']} | {a:.2f} | {b:.2f} | {c:.2f} | {f75} | "
                         f"{m['mean_seconds']:.3f} | {m['p95_seconds']:.3f} | {m['peak_gpu_mb']:.0f} | {bps} | {cbps} |")
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


def free_gpus(allowed):
    inventory = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used',
        '--format=csv,noheader,nounits'],text=True)
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True)
    occupied = {line.split(',')[0].strip() for line in processes.splitlines() if ',' in line}
    result = []
    for line in inventory.splitlines():
        index,uuid,memory = [s.strip() for s in line.split(',')]
        if int(index) in allowed and uuid not in occupied and int(memory)<700:
            result.append(int(index))
    return result


def queue(args):
    plan = prepare(args)
    lock = (args.output/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    running, failures, completed = {}, {}, set()
    summarized = set()
    (args.output/'logs').mkdir(exist_ok=True)
    for job in plan['tasks']:
        paths = [output_path(args.output,job,a) for a in arms(job['family'])]
        if all(p.with_suffix('.done.json').exists() for p in paths):
            completed.add(job['id'])
    while len(completed)+len(failures)<len(plan['tasks']) or running:
        for gpu, item in list(running.items()):
            code = item['process'].poll()
            if code is None:
                continue
            job = item['job']
            if code:
                failures[job['id']] = dict(exitcode=code,log=item['log'])
            else:
                completed.add(job['id'])
            del running[gpu]
        active = {r['job']['id'] for r in running.values()}
        pending = [j for j in plan['tasks'] if j['id'] not in completed|active|set(failures)]
        for gpu in ([] if failures else free_gpus(args.gpus)):
            if gpu in running or not pending:
                continue
            job = pending.pop(0)
            path = args.output/'logs'/f"{job['id']}_gpu{gpu}.log"
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',
                       TOKENIZERS_PARALLELISM='false',ROUND2_EXECUTION='fused_one_pass')
            command = [sys.executable,'-u',str(Path(__file__).resolve()),'worker',
                       '--output',str(args.output),'--job',job['id']]
            with path.open('a') as log:
                process = subprocess.Popen(command,env=env,cwd=ref.ROOT,stdout=log,stderr=subprocess.STDOUT)
            running[gpu] = dict(process=process,job=job,log=str(path))
            print(f"START GPU {gpu}: {job['id']} pid={process.pid}",flush=True)
        for suite in plan['suites']:
            if suite not in summarized and all(j['id'] in completed for j in plan['tasks'] if j['suite']==suite):
                if summarize_suite(args.output,plan,suite):
                    summarized.add(suite)
                    summary(args.output,plan)
                    print(f'COMPLETE {suite}: scores and comparison written',flush=True)
        progress = [read(p) for p in (args.output/'progress').glob('*.json')]
        state = dict(state='running' if running else 'waiting' if pending else 'completed' if not failures else 'failed',
            completed_jobs=len(completed), total_jobs=len(plan['tasks']), failures=failures,
            active=[dict(gpu=g, pid=i['process'].pid, job=i['job']['id'],log=i['log']) for g,i in running.items()],
            completed_record_executions=sum(p['completed'] for p in progress),
            expected_record_executions=sum(j['records']*len(arms(j['family'])) for j in plan['tasks']),
            elapsed_seconds=time.time()-started, completed_suites=sorted(summarized), allowed_gpus=args.gpus)
        ref.atomic_json(args.output/'status.json',state)
        if failures and not running:
            break
        if running or pending:
            time.sleep(5)
    for suite in plan['suites']:
        if suite not in summarized and all(j['id'] in completed for j in plan['tasks'] if j['suite']==suite):
            if summarize_suite(args.output,plan,suite):
                summarized.add(suite)
    summary(args.output,plan)
    if failures:
        raise RuntimeError(f'Evaluation jobs failed; see status.json: {failures}')
    ref.atomic_json(args.output/'all.done.json',dict(completed=True,suites=list(summarized),
        plan_sha256=ref.sha256_file(args.output/'plan.json'),smoke_records=plan['smoke_records']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','worker','queue','merge'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--suites',nargs='+',choices=SUITES,default=list(SUITES))
    parser.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3])
    parser.add_argument('--shards',type=int,default=8)
    parser.add_argument('--smoke-records',type=int,default=0)
    parser.add_argument('--job')
    args = parser.parse_args()
    if args.shards<1 or args.smoke_records<0 or len(set(args.suites))!=len(args.suites):
        parser.error('Invalid shard, smoke or suite settings')
    if args.action=='prepare':
        prepare(args)
    elif args.action=='worker':
        worker(args)
    elif args.action=='queue':
        queue(args)
    else:
        plan=read(args.output/'plan.json')
        for suite in plan['suites']:
            summarize_suite(args.output,plan,suite)
        summary(args.output,plan)


if __name__=='__main__':
    main()
