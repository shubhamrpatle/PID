#!/usr/bin/env python3
"""Evaluate only the uploaded full-FT Stage-2 best using the prior LoRA protocol."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import evaluate_refwave_cluster_stage2 as reference

PROJECT, ROOT = reference.PROJECT, reference.ROOT
atomic_json, read_jsonl, sha256_file = reference.atomic_json, reference.read_jsonl, reference.sha256_file
REPO = 'CaseStudyRef/RefWave-Cluster-Runs'
REVISION = 'f8328254e667f8def4433c05fe179fda513c0a77'
RUN = 'refwave_full_4gpu_20260908T184627Z_4djaFo'
DOWNLOAD = Path('/data2/shubham/refwave_cluster_full_stage2_best_20260910')
CHECKPOINT = DOWNLOAD / 'runs' / RUN / 'stage2/best.pt'
CHECKPOINT_SHA = '2f39607b1f96d3a7d7662d8dc09b59d01251e9298cd7609f33b797f8c86368e7'
OUTPUT = PROJECT / 'runs/refwave_cluster_full_stage2_best_20260910'


def validate_checkpoint(saved, config, curve, completion, *, tied_output_head=False):
    from refwave.train import SCHEMA, cursor_after_step
    best = min(curve, key=lambda row: row['detection_grounding_macro_loss'])
    if saved['schema'] != SCHEMA or saved['config'] != config:
        raise ValueError('Checkpoint/config mismatch')
    if config['tuning'] != 'full' or config['method'] != 'refwave' or config['stage'] != 'stage2':
        raise ValueError('Expected full-parameter RefWave Stage 2, not a LoRA adapter')
    if saved['step'] != best['step'] or saved['best_loss'] != best['detection_grounding_macro_loss']:
        raise ValueError('best.pt does not match minimum validation macro loss')
    if saved['cursor'] != cursor_after_step(saved['step'], config['records'], config['global_batch']):
        raise ValueError('Best checkpoint cursor does not match its step')
    if completion['steps'] != config['steps'] or completion['records'] != config['records'] * config['epochs']:
        raise ValueError('Stage 2 has no matching completion record')
    if completion['epochs'] != config['epochs']:
        raise ValueError('Completion epoch count differs')
    count = sum(t.numel() for t in saved['trainable_state'].values())
    if count != config['trainable']['trainable_parameters']:
        raise ValueError('Incomplete full-model parameter state')
    components = Counter()
    for name, tensor in saved['trainable_state'].items():
        components[name.split('.')[0]] += tensor.numel()
        if 'lora_' in name:
            raise ValueError('Unexpected LoRA tensors in full checkpoint')
    if not all(components[name] for name in ('vision_model', 'mlp1', 'language_model')):
        raise ValueError('Missing vision, projector or language-model weights')
    if ('language_model.lm_head.weight' not in saved['trainable_state']
            and not (tied_output_head and 'language_model.model.embed_tokens.weight' in saved['trainable_state'])):
        raise ValueError('Missing complete output head')
    return best, dict(components)


def prepare(args):
    import torch
    from huggingface_hub import snapshot_download
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [f'runs/{RUN}/stage2/{name}' for name in
                ('best.pt', 'config.json', 'done.json', 'validation_curve.jsonl')]
    selected += [f'runs/{RUN}/stage1/{name}' for name in ('config.json', 'done.json')]
    snapshot_download(REPO, repo_type='dataset', revision=REVISION, local_dir=DOWNLOAD,
                      allow_patterns=selected, max_workers=2)
    digest = sha256_file(CHECKPOINT)
    if digest != CHECKPOINT_SHA:
        raise ValueError('Downloaded checkpoint SHA256 differs from pinned Hugging Face LFS hash')
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
    config = json.loads(CHECKPOINT.with_name('config.json').read_text())
    curve = read_jsonl(CHECKPOINT.with_name('validation_curve.jsonl'))
    completion = json.loads(CHECKPOINT.with_name('done.json').read_text())
    base_config = json.loads((reference.BASE_MODEL/'config.json').read_text())
    tied_head = base_config['text_config'].get('tie_word_embeddings', False)
    best, components = validate_checkpoint(saved, config, curve, completion, tied_output_head=tied_head)
    selection = dict(repo=REPO, revision=REVISION, run=RUN, checkpoint=str(CHECKPOINT), sha256=digest,
        bytes=CHECKPOINT.stat().st_size, step=saved['step'], criterion='minimum validation detection_grounding_macro_loss',
        validation=best, config=config, completion=completion, component_parameters=components,
        output_head_storage='shared input/output embedding' if tied_head else 'separate LM head',
        note='Only Stage-2 best weights downloaded; file also contains optimizer state, unused during evaluation.')
    del saved
    path = args.output / 'checkpoint.json'
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError('Existing output selects a different checkpoint')
    atomic_json(path, selection)
    assets = args.output / 'assets'
    assets.mkdir(exist_ok=True)
    manifest_summary = json.loads((reference.OUTPUT / 'assets/manifest_summary.json').read_text())
    for suite in reference.SUITES:
        source, destination = reference.OUTPUT / 'assets' / f'{suite}.jsonl', assets / f'{suite}.jsonl'
        if sha256_file(source) != manifest_summary[suite]['sha256']:
            raise ValueError(f'Prior manifest changed: {suite}')
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise ValueError(f'Output manifest changed: {suite}')
        if not destination.exists():
            shutil.copy2(source, destination)
        rows = read_jsonl(destination)
        assert len(rows) == len({r['uid'] for r in rows}) == manifest_summary[suite]['records']
        for row in rows:
            if not Path(row['image_path']).is_file():
                raise FileNotFoundError(row['image_path'])
        for method in ('standard', 'refwave'):
            prior = json.loads((reference.OUTPUT / suite / method / 'metrics.json').read_text())
            assert prior['contract']['manifest_sha256'] == manifest_summary[suite]['sha256']
    atomic_json(assets / 'manifest_summary.json', manifest_summary)
    files = [Path(__file__), Path(reference.__file__), Path(reference.detection.__file__), Path(reference.grounding.__file__)]
    for folder in (ROOT/'RefWave/src/refwave', ROOT/'Pixel-PIVR/src/pixel_pivr', ROOT/'zero_shot_analysis'):
        files.extend(folder.glob('*.py'))
    files += [PROJECT/'scripts/evaluate_self_conditioned_point_witness_balanced100.py',
              PROJECT/'scripts/evaluate_pixel_pivr_sequential_wave_balanced100.py',
              ROOT/'Pixel-PIVR/tools/evaluate_class_addressed_wave.py']
    files += list(reference.BASE_MODEL.glob('*.py')) + list(reference.BASE_MODEL.glob('*.json'))
    signatures = {str(path): sha256_file(path) for path in files}
    source_manifest = args.output / 'source_sha256.json'
    if source_manifest.exists() and json.loads(source_manifest.read_text()) != signatures:
        raise ValueError('Evaluation source changed; use a fresh output directory')
    for path in files:
        destination = args.output / 'source_snapshot' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(path, destination)
    atomic_json(source_manifest, signatures)
    print(json.dumps(dict(step=selection['step'], validation=best, component_parameters=components,
                          suites=manifest_summary), indent=2), flush=True)


def load_full_worker():
    import torch
    from pixel_pivr.worker import LocateAnythingPixelPIVRWorker
    from refwave.model import configure_full
    from refwave.train import restore
    worker = LocateAnythingPixelPIVRWorker(reference.BASE_MODEL, device='cuda:0')
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
    config = saved['config']
    if config['tuning'] != 'full' or config['method'] != 'refwave':
        raise ValueError('This loader accepts full RefWave parameters only')
    configure_full(worker.model)
    # Exact name/shape checks restore every component, including MoonViT and the LM head.
    restore(worker.model, saved['trainable_state'])
    worker.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    worker.processor.image_processor.in_token_limit = config['image_token_limit']
    del saved
    gc.collect()
    torch.cuda.empty_cache()
    return worker


def infer(args):
    import torch
    from PIL import Image
    from refwave.io import read_jsonl_log
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    signatures = json.loads((args.output / 'source_sha256.json').read_text())
    if any(sha256_file(Path(path)) != sha for path, sha in signatures.items()):
        raise ValueError('Source changed after preparation')
    manifest = args.output / 'assets' / f'{args.suite}.jsonl'
    rows = read_jsonl(manifest)[args.shard_index::args.shard_count]
    if args.limit:
        rows = rows[:args.limit]
    directory = args.output / ('smoke' if args.limit else 'benchmarks') / args.suite
    directory.mkdir(parents=True, exist_ok=True)
    tag = f'shard{args.shard_index:02d}of{args.shard_count:02d}'
    path = directory / f'{tag}.jsonl'
    selection = json.loads((args.output / 'checkpoint.json').read_text())
    budget = 2048 if '100' in args.suite else 8192
    contract = dict(checkpoint_sha256=selection['sha256'], step=selection['step'], method='refwave_full',
        tuning='full', manifest_sha256=sha256_file(manifest), shard_index=args.shard_index,
        shard_count=args.shard_count, limit=args.limit, image_token_limit=6000, max_new_tokens=budget,
        class_wave=18, box_wave=200, temperature=.7, top_p=.9, repetition_penalty=1.1,
        image_prefix_shared_across_rounds=True, optimized=True, full_head=True,
        round2_execution=reference.round2_execution_mode(),
        adapters='none; all trained parameters restored, BF16 inference',
        scoring='uncapped class-aware greedy HBB IoU; raw and NMS0.5 separate; grounding first box',
        timing='CUDA-synchronized; preprocessing + decode + parsing/containment; excludes image disk IO, weights, metrics',
        warmup_queries=2, cpu_threads=4, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu=torch.cuda.get_device_name(0), source_manifest_sha256=sha256_file(args.output/'source_sha256.json'))
    contract_path = path.with_suffix('.contract.json')
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Existing inference contract differs; do not append')
    atomic_json(contract_path, contract)
    existing = read_jsonl_log(path, repair_tail=True) if path.exists() else []
    done = {r['uid'] for r in existing}
    assert len(done) == len(existing) and done <= {r['uid'] for r in rows}
    if len(done) == len(rows):
        return
    worker = load_full_worker()
    decoders = reference.make_decoders(worker, optimized=True)
    for row in rows[:2]:
        with Image.open(row['image_path']) as image:
            reference.infer_one(worker, decoders, reference.inference_query(row), image.convert('RGB'), 'refwave', budget)
    with path.open('a') as handle:
        for index, row in enumerate(rows):
            if row['uid'] in done:
                continue
            with Image.open(row['image_path']) as opened:
                image = opened.convert('RGB')
            assert image.size == (int(row['width']), int(row['height']))
            result = reference.infer_one(worker, decoders, reference.inference_query(row), image, 'refwave', budget)
            result['method'] = 'refwave_full'
            result.update(image_id=row['image_id'], dataset=row['dataset'])
            if row['task'] == 'grounding':
                result.update(sample_id=row['sample_id'], query=row['query'])
            handle.write(json.dumps(result, sort_keys=True)+'\n')
            handle.flush()
            print(f"{args.suite} full {tag} {index+1}/{len(rows)} points={len(result['points'])} "
                  f"boxes={len(result['predictions'])} seconds={result['latency_seconds']:.3f}", flush=True)
    atomic_json(directory / f'{tag}.done.json', dict(records=len(rows), smoke=bool(args.limit)))


def score(rows, predictions):
    import numpy as np
    indexed = {p['uid']: p for p in predictions}
    assert len(indexed) == len(predictions) == len(rows) and set(indexed) == {r['uid'] for r in rows}
    predictions = [indexed[r['uid']] for r in rows]
    times = [r['latency_seconds'] for r in predictions]
    metrics = dict(records=len(rows), mean_seconds=float(np.mean(times)), p50_seconds=float(np.median(times)),
        p95_seconds=float(np.percentile(times,95)), peak_gpu_mb=max(r['peak_gpu_mb'] for r in predictions),
        raw_boxes=sum(len(r['predictions']) for r in predictions))
    if rows[0]['task'] == 'grounding':
        result = reference.grounding.evaluate_predictions(rows, predictions)
        assert result['missing_predictions'] == result['missing_images'] == 0
        metrics.update(grounding=result['overall'], by_size=result['by_size'], by_class=result['by_class'])
        per_record = result['rows']
    else:
        classes = [s['canonical'] for s in rows[0]['class_specs']]
        raw_index = {(p['dataset'],p['image_id']):p for p in predictions}
        nms = [dict(p, predictions=reference.detection.nms_predictions(p['predictions'], .5)) for p in predictions]
        for name, items in [('raw',predictions), ('nms0p5',nms)]:
            index = {(p['dataset'],p['image_id']):p for p in items}
            metrics[name] = {str(t):reference.detection.f1_for_threshold(rows,index,classes,t,0) for t in (.5,.75,.95)}
        metrics['per_class'] = {c:reference.detection.f1_for_threshold(rows,raw_index,[c],.5,0) for c in classes}
        per_record = [dict(uid=r['uid'], **reference.detection.f1_for_threshold(
            [r], {(r['dataset'],r['image_id']):p}, classes, .5, 0)) for r,p in zip(rows,predictions)]
    return metrics, per_record


def merge(args):
    directory = args.output / 'benchmarks' / args.suite
    rows = read_jsonl(args.output / 'assets' / f'{args.suite}.jsonl')
    paths = [directory / f'shard{i:02d}of{args.shard_count:02d}.jsonl' for i in range(args.shard_count)]
    contracts = [json.loads(p.with_suffix('.contract.json').read_text()) for p in paths]
    ignore = {'shard_index','visible_devices'}
    assert all({k:v for k,v in c.items() if k not in ignore} ==
               {k:v for k,v in contracts[0].items() if k not in ignore} for c in contracts)
    assert contracts[0]['limit'] == 0
    predictions = [r for p in paths for r in read_jsonl(p)]
    metrics, per_record = score(rows, predictions)
    metrics['contract'] = contracts[0]
    indexed = {p['uid']:p for p in predictions}
    reference.jsonl(directory/'predictions.jsonl', [indexed[r['uid']] for r in rows])
    reference.jsonl(directory/'per_record_metrics.jsonl', per_record)
    atomic_json(directory/'metrics.json', metrics)
    comparison = {'refwave_full': metrics}
    for method in ('standard','refwave'):
        prior = reference.OUTPUT / args.suite / method
        old = json.loads((prior/'metrics.json').read_text())
        for key in ('manifest_sha256','image_token_limit','max_new_tokens','temperature','top_p',
                    'repetition_penalty','class_wave','box_wave','gpu','shard_count'):
            assert old['contract'][key] == contracts[0][key], key
        rebuilt, _ = score(rows, read_jsonl(prior/'predictions.jsonl'))
        for key in ('raw','nms0p5','grounding'):
            if key in old:
                assert old[key] == rebuilt[key], f'{method} {key} prior metric differs from raw predictions'
        comparison[method] = old
    atomic_json(directory/'comparison.json', dict(models=comparison,
        note='Identical evaluation protocol, not a matched training-budget ablation: full FT and LoRA differ in update count, batch size and trainable parameters.'))
    summary(args)
    print(json.dumps({k:metrics[k] for k in ('records','mean_seconds','raw','grounding') if k in metrics},indent=2),flush=True)


def summary(args):
    lines = ['# Full-Parameter RefWave Stage-2 Best', '',
        'Full FT: step 2400, minimum validation macro loss; completed Stage 2 at step 2495 (one epoch).',
        'Same frozen benchmark manifests, native-6000 input budget, class waves 18, geometry waves 200 and shared global Qwen prefix.',
        'Detection headline: uncapped raw class-aware IoU@0.5; NMS0.5 remains separate. Grounding: first predicted box.',
        'Prior Standard/RefWave LoRA results are reused and rescored from raw predictions, not rerun.',
        'This compares tuning regimes, not matched optimizer-step budgets. Full FT global batch 16; prior LoRA batch/update settings differ.', '',
        '| Suite | Model | Records | TP / FP / FN | Precision or Acc@0.5 | Recall or Acc@0.7 | F1 or mIoU | Seconds/query |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    combined = {}
    for suite in reference.SUITES:
        path = args.output / 'benchmarks' / suite / 'comparison.json'
        if not path.exists():
            lines.append(f'| {suite} | Pending | | | | | | |')
            continue
        combined[suite] = json.loads(path.read_text())
        for method in ('standard','refwave','refwave_full'):
            m = combined[suite]['models'][method]
            if 'raw' in m:
                v = m['raw']['0.5']
                counts = f"{v['tp']} / {v['fp']} / {v['fn']}"
                a,b,c = (100*v[k] for k in ('precision','recall','f1'))
            else:
                v, counts = m['grounding'], '-'
                a,b,c = (100*v[k] for k in ('first_acc_0_5','first_acc_0_7','first_mean_iou'))
            lines.append(f"| {suite} | {method} | {m['records']} | {counts} | {a:.2f} | {b:.2f} | {c:.2f} | {m['mean_seconds']:.3f} |")
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    atomic_json(args.output/'all_benchmark_metrics.json',combined)


def wait_idle(gpus):
    def available():
        devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                           '--format=csv,noheader,nounits'], text=True)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'], text=True)
        occupied = {line.split(',')[0].strip() for line in apps.splitlines() if ',' in line}
        free = set()
        for line in devices.splitlines():
            index, uuid, memory = (v.strip() for v in line.split(','))
            if uuid not in occupied and int(memory) >= 30000:
                free.add(int(index))
        return set(gpus) <= free

    while True:
        if available():
            time.sleep(5)
            if available():
                return
        print(f'Waiting for selected GPUs {gpus} to be idle; no processes will be stopped.', flush=True)
        time.sleep(30)


def launch_shards(args, suite, limit=0):
    logs = args.output/'logs'
    logs.mkdir(exist_ok=True)
    wait_idle(args.gpus)
    children, handles = [], []
    for shard,gpu in enumerate(args.gpus):
        path = logs/f'{suite}_{"smoke" if limit else "full"}_shard{shard}.log'
        handle = path.open('a')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), 'infer', '--output', str(args.output),
                   '--suite',suite,'--shard-index',str(shard),'--shard-count',str(len(args.gpus)),'--limit',str(limit)]
        child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        children.append(child); handles.append(handle)
        print(json.dumps(dict(event='START',suite=suite,smoke=bool(limit),gpu=gpu,pid=child.pid,log=str(path))),flush=True)
    while any(child.poll() is None for child in children):
        time.sleep(5)
    for handle in handles:
        handle.close()
    failed = [{'pid':p.pid,'returncode':p.returncode} for p in children if p.returncode]
    if failed:
        raise RuntimeError(f'Inference failed, no merged results: {failed}')


def queue(args):
    prepare(args)
    summary(args)
    atomic_json(args.output/'status.json',dict(state='smoke_pending',gpus=args.gpus))
    for suite in reference.SUITES[:2]:
        launch_shards(args,suite,limit=1)
    atomic_json(args.output/'smoke.done.json',dict(suites=list(reference.SUITES[:2]),records_per_suite=len(args.gpus)))
    for suite in args.suites:
        atomic_json(args.output/'status.json',dict(state='running',suite=suite,gpus=args.gpus))
        launch_shards(args,suite)
        merge(argparse.Namespace(output=args.output,suite=suite,shard_count=len(args.gpus)))
    atomic_json(args.output/'all.done.json',dict(completed=True,suites=args.suites))
    atomic_json(args.output/'status.json',dict(state='complete',suites=args.suites))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','infer','merge','queue','summary'))
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--suite',choices=reference.SUITES)
    p.add_argument('--suites',nargs='+',choices=reference.SUITES,default=list(reference.SUITES))
    p.add_argument('--gpus',nargs='+',type=int,default=[2,3])
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--shard-count',type=int,default=2)
    p.add_argument('--limit',type=int,default=0)
    args = p.parse_args()
    if len(args.gpus) != len(set(args.gpus)):
        p.error('Duplicate GPU assignment')
    try:
        {'prepare':prepare,'infer':infer,'merge':merge,'queue':queue,'summary':summary}[args.mode](args)
    except Exception as error:
        if args.mode == 'queue':
            atomic_json(args.output/'status.json',dict(state='failed',error=str(error)))
        raise


if __name__ == '__main__':
    main()
