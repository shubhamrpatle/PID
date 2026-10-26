#!/usr/bin/env python3
"""Frozen, resumable benchmark comparison of the uploaded Stage-2 best models."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path[:0] = [str(ROOT / 'RefWave/src'), str(ROOT / 'Pixel-PIVR/src'),
                str(ROOT / 'Pixel-PIVR'), str(ROOT), str(PROJECT), str(PROJECT / 'scripts')]
from pixel_pivr.io import atomic_json, read_jsonl, sha256_file
import evaluate_locateanything_lae_style as detection
import evaluate_locateanything_grounding_benchmarks as grounding
from tools.evaluate_class_addressed_wave import stable_seed, parse_class_points
from zero_shot_analysis.geometry import hbox_iou

REVISION = 'd88ef88d98a8df34fea03bb22346d8411a0f33eb'
DOWNLOAD = Path('/data2/shubham/refwave_cluster_stage2_best_20260909')
OUTPUT = PROJECT / 'runs/refwave_cluster_stage2_best_20260909'
BASE_MODEL = PROJECT / 'model/LocateAnything-3B/checkpoint'
RUNS = dict(refwave='refwave_lora_4gpu_20260908T144740Z_G4OT9L',
            standard='standard_lora_4gpu_20260908T185018Z_lJdQkO')
SUITES = ('DOTAv2-Balanced100', 'DIOR-RSVG-Unseen100', 'DIOR', 'DOTAv2',
          'DIOR-RSVG', 'VRSBench-VG', 'LAE-80C')


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + '\n')
    os.replace(temp, path)


def checkpoint(method):
    return DOWNLOAD / 'runs' / RUNS[method] / 'stage2/best.pt'


def build_manifests(output):
    from benchmark_refwave_shared_image import protocol as balanced_protocol
    output.mkdir(parents=True, exist_ok=True)
    config = detection.read_json(PROJECT / 'config/lae_benchmarks.json')
    report = {}
    for suite in SUITES:
        if suite in ('DIOR', 'DOTAv2', 'LAE-80C', 'DOTAv2-Balanced100'):
            name = 'DOTAv2' if suite == 'DOTAv2-Balanced100' else suite
            bc = config['benchmarks'][name]
            if suite == 'DOTAv2-Balanced100':
                rows, specs, _, _ = balanced_protocol()
            else:
                specs = detection.benchmark_class_specs(name, bc, 'lae-metainfo', 'raw_class_name')
                rows = detection.load_lae_records(name, bc, specs, 'raw_class_name')
            for row in rows:
                row['image_path'] = str(detection.resolve_image_path(row, PROJECT / bc['image_root']).resolve())
                row.update(task='detection', uid=str(row['image_id']), suite=suite,
                           direct_question=detection.detection_prompt(specs), class_specs=specs,
                           references=[dict(label=s['canonical'], reference=s['prompt'],
                               question=f"Point to all instances that match the following categories: {s['prompt']}.")
                               for s in specs])
        else:
            if suite == 'DIOR-RSVG-Unseen100':
                rows = read_jsonl(PROJECT / 'dataset/eval_grounding100/DIOR-RSVG/records_joint_exact_unseen100.jsonl')
            else:
                args = SimpleNamespace(dior_rsvg_root=str(PROJECT / 'dataset/grounding/DIOR-RSVG'),
                    vrsbench_root=str(PROJECT / 'dataset/grounding/VRSBench'), split='test',
                    image_roots=[str(PROJECT / 'dataset/grounding/DIOR-RSVG/JPEGImages'),
                                 str(PROJECT / 'dataset/grounding/VRSBench/Images_val')])
                rows = (grounding.parse_dior_rsvg(args) if suite == 'DIOR-RSVG'
                        else grounding.parse_vrsbench_grounding(args))
            for row in rows:
                phrase = row['query']
                row.update(task='grounding', uid=grounding.prediction_key(row), suite=suite,
                    direct_question=f'Locate a single instance that matches the following description: {phrase}.',
                    references=[dict(label=phrase, reference=phrase, question=f'Point to: {phrase}.')])
        assert len({r['uid'] for r in rows}) == len(rows)
        for row in rows:
            if not row['image_path'] or not Path(row['image_path']).is_file():
                raise FileNotFoundError(row['image_path'])
        path = output / 'assets' / f'{suite}.jsonl'
        if path.exists() and read_jsonl(path) != rows:
            raise RuntimeError(f'Frozen manifest changed: {path}')
        if not path.exists():
            jsonl(path, rows)
        report[suite] = dict(records=len(rows), images=len({r['image_path'] for r in rows}),
            targets=sum(len(r['gt']) if r['task'] == 'detection' else 1 for r in rows),
            sha256=sha256_file(path), max_new_tokens=2048 if '100' in suite else 8192)
    atomic_json(output / 'assets/manifest_summary.json', report)
    return report


def prepare(args):
    import torch
    from huggingface_hub import HfApi
    args.output.mkdir(parents=True, exist_ok=True)
    files = {f.path: f for f in HfApi().list_repo_tree('CaseStudyRef/RefWave-Cluster-Runs',
        repo_type='dataset', revision=REVISION, recursive=True) if hasattr(f, 'size')}
    selections = {}
    for method in RUNS:
        path = checkpoint(method)
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        config = json.loads(path.with_name('config.json').read_text())
        curve = read_jsonl(path.with_name('validation_curve.jsonl'))
        best = min(curve, key=lambda r:r['detection_grounding_macro_loss'])
        assert saved['config'] == config and saved['step'] == best['step']
        assert saved['best_loss'] == best['detection_grounding_macro_loss']
        assert saved['config']['method'] == method and saved['config']['stage'] == 'stage2'
        sha = sha256_file(path)
        remote = files[str(path.relative_to(DOWNLOAD))]
        assert remote.size == path.stat().st_size and remote.lfs.sha256 == sha
        selections[method] = dict(checkpoint=str(path), sha256=sha, step=saved['step'],
            criterion='minimum validation detection_grounding_macro_loss within this method',
            validation=best, config=config, bytes=path.stat().st_size,
            tensors=len(saved['trainable_state']),
            trainable_parameters=sum(v.numel() for v in saved['trainable_state'].values()),
            completion=json.loads(path.with_name('done.json').read_text()))
        del saved
    for k in ('train_sha256', 'validation_sha256', 'records', 'steps', 'lora_rank',
              'learning_rate', 'head_lr', 'global_batch', 'image_token_limit'):
        assert selections['standard']['config'][k] == selections['refwave']['config'][k], k
    atomic_json(args.output / 'checkpoints.json', dict(repo='CaseStudyRef/RefWave-Cluster-Runs',
        revision=REVISION, selections=selections, downloaded_weights_bytes=sum(r['bytes'] for r in selections.values()),
        note='Only the two Stage-2 best.pt files. Each also contains optimizer state; no extra checkpoint downloaded.'))
    print(json.dumps(build_manifests(args.output), indent=2), flush=True)
    source_files = [Path(__file__)]
    for folder in (ROOT/'RefWave/src/refwave', ROOT/'Pixel-PIVR/src/pixel_pivr', ROOT/'zero_shot_analysis'):
        source_files.extend(folder.glob('*.py'))
    source_files.extend([Path(detection.__file__), Path(grounding.__file__),
        PROJECT/'scripts/evaluate_self_conditioned_point_witness_balanced100.py',
        PROJECT/'scripts/evaluate_pixel_pivr_sequential_wave_balanced100.py',
        ROOT/'Pixel-PIVR/tools/evaluate_class_addressed_wave.py'])
    source_files.extend(BASE_MODEL.glob('*.py'))
    source_files.extend(BASE_MODEL.glob('*.json'))
    signatures = {}
    for source in source_files:
        destination = args.output / 'source_snapshot' / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        signatures[str(source)] = sha256_file(source)
    atomic_json(args.output/'source_sha256.json', signatures)


def load_worker(method):
    import torch
    from pixel_pivr.worker import LocateAnythingPixelPIVRWorker
    from refwave.model import configure_lora_head
    from refwave.train import restore
    worker = LocateAnythingPixelPIVRWorker(BASE_MODEL, device='cuda:0')
    saved = torch.load(checkpoint(method), map_location='cpu', weights_only=False, mmap=True)
    configure_lora_head(worker.model, saved['config']['lora_rank'])
    restore(worker.model, saved['trainable_state'])
    # Merge in memory, including the full adapted output head; no 7-GB export copy.
    worker.model.language_model = worker.model.language_model.merge_and_unload()
    worker.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    worker.processor.image_processor.in_token_limit = 6000
    del saved
    gc.collect()
    torch.cuda.empty_cache()
    return worker


def round2_execution_mode(value=None):
    mode = value or os.environ.get('ROUND2_EXECUTION', 'fused_one_pass')
    if mode not in ('two_pass', 'fused_one_pass'):
        raise ValueError('ROUND2_EXECUTION must be two_pass or fused_one_pass')
    return mode


def make_decoders(worker, optimized, round2_execution=None):
    from pixel_pivr.hybrid_class_wave import CachedHybridClassWaveDecoder
    from pixel_pivr.fast_class_wave import FastCachedHybridClassWaveDecoder
    from zero_shot_analysis.pixel_pivr_wave import PixelPIVRWaveDecoder
    from zero_shot_analysis.fused_pivr_geometry import FusedPIVRGeometryDecoder
    Point = FastCachedHybridClassWaveDecoder if optimized else CachedHybridClassWaveDecoder
    Box = FusedPIVRGeometryDecoder if optimized else PixelPIVRWaveDecoder
    if not optimized and round2_execution not in (None, 'two_pass'):
        raise ValueError('The historical reference decoder supports two_pass only')
    options = {'round2_execution': round2_execution_mode(round2_execution)} if optimized else {}
    return (Point(worker, image_token_limit=6000, max_wave_size=18, bucket_suffix_lengths=False),
            Box(worker, image_token_limit=6000, visual_context='global_only',
                prefix_cache_mode='shared', geometry_prefix_mode='box_only', **options))


def infer_one(worker, decoders, query, image, method, budget):
    import torch
    from pixel_pivr.class_wave import ClassPointBranch
    from zero_shot_analysis.locate_anything import parse_locateanything_labeled_boxes
    from zero_shot_analysis.pixel_pivr_wave import AddressedCrop
    from evaluate_pixel_pivr_sequential_wave_balanced100 import compact_round2_prompt, point_to_norm
    from evaluate_self_conditioned_point_witness_balanced100 import choose_witness_box
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    seed = stable_seed(0, query['suite'], query['uid'], 'generation')
    grounding.set_generation_seed(seed)
    details = {}
    if method == 'standard':
        answer = worker.predict_points(image, query['direct_question'], max_new_tokens=budget,
            temperature=.7, top_p=.9, repetition_penalty=1.1)
        raw = parse_locateanything_labeled_boxes(answer, *image.size)
        predictions = []
        for pred in raw:
            box = detection.clip_hbox_to_image(pred['hbox'], *image.size)
            if box is None:
                continue
            if query['task'] == 'detection':
                label = detection.canonicalize_label(pred['label'], query['aliases'], set(query['classes']))
                if label is None:
                    continue
            else:
                label = query['references'][0]['label']
            predictions.append(dict(label=label, hbox=box, score=1.0))
        details['answer'] = answer
    else:
        point_decoder, box_decoder = decoders
        branches = [ClassPointBranch(b['label'], b['question'],
            stable_seed(0, query['suite'], query['uid'], b['label'], 'isolated-point'))
            for b in query['references']]
        t = time.perf_counter()
        prepared = point_decoder.prepare_image(image, branches)
        torch.cuda.synchronize()
        prepare_seconds = time.perf_counter()-t
        answers, r1 = point_decoder.generate(image, branches, max_new_tokens=budget, temperature=.7,
            top_p=.9, repetition_penalty=1.1, generation_mode='hybrid', prepared_image=prepared)
        points = parse_class_points(answers, *image.size, 'geometry_only')
        addresses = [AddressedCrop(str(i), p['label'], None,
            compact_round2_prompt(p['label'], p['point'], *image.size),
            point_normalized=point_to_norm(p['point'], *image.size)) for i,p in enumerate(points)]
        boxes, r2 = box_decoder.decode_prepared_image(prepared, addresses, requested_wave_size=200, allow_none=True)
        assert len(boxes) == len(points)
        predictions = []
        for p,b in zip(points, boxes):
            aliases = query['aliases'] if query['task']=='detection' else {detection.normalize_class_name(p['label']):p['label']}
            classes = set(query['classes']) if query['task']=='detection' else {p['label']}
            pred, diagnostic = choose_witness_box(b['answer'], p['label'], p['point'], aliases, classes, *image.size)
            b['containment_diagnostics'] = diagnostic
            if pred is not None:
                predictions.append(pred)
        details.update(points=points, point_answers=answers, refinements=boxes, round1=r1, round2=r2,
            preparation_seconds=prepare_seconds,
            image_encodes=1+r1['global_moonvit_encodes']+r2.get('global_moonvit_encodes',0),
            prefix_forwards=1+r1['qwen_global_prefix_forwards']+r2['qwen_global_prefix_forwards'])
        assert details['image_encodes']==1 and details['prefix_forwards']==1
        del prepared
    torch.cuda.synchronize()
    peak=max(torch.cuda.max_memory_allocated()/1024**2,
             details.get('round1',{}).get('peak_gpu_mb',0),details.get('round2',{}).get('peak_gpu_mb',0))
    return dict(uid=query['uid'], suite=query['suite'], task=query['task'], predictions=predictions,
        latency_seconds=time.perf_counter()-started, peak_gpu_mb=peak,
        generation_seed=seed, **details)


def inference_query(row):
    # Deliberately exclude GT, target class, size, and annotation metadata.
    query = {k:row[k] for k in ('uid','suite','task','direct_question','references')}
    if row['task']=='detection':
        query.update(aliases=detection.build_alias_lookup(row['class_specs']),
                     classes=[s['canonical'] for s in row['class_specs']])
    return query


def infer(args):
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    manifest = args.output/'assets'/f'{args.suite}.jsonl'
    rows = read_jsonl(manifest)[args.shard_index::args.shard_count]
    if args.limit:
        rows = rows[:args.limit]
    directory = args.output/args.suite/args.method
    directory.mkdir(parents=True, exist_ok=True)
    tag = f'shard{args.shard_index:02d}of{args.shard_count:02d}'
    path = directory/f'{tag}.jsonl'
    selected = json.loads((args.output/'checkpoints.json').read_text())['selections'][args.method]
    budget = 2048 if '100' in args.suite else 8192
    execution = ('two_pass' if args.reference else round2_execution_mode(
        getattr(args, 'round2_execution', None))) if args.method == 'refwave' else None
    contract = dict(checkpoint_sha256=selected['sha256'], step=selected['step'], method=args.method,
        manifest_sha256=sha256_file(manifest), shard_index=args.shard_index, shard_count=args.shard_count,
        limit=args.limit, image_token_limit=6000, max_new_tokens=budget, class_wave=18, box_wave=200,
        temperature=.7, top_p=.9, repetition_penalty=1.1, image_prefix_shared_across_rounds=True,
        optimized=not args.reference, round2_execution=execution,
        full_head=True, adapters='merged BF16 in memory',
        scoring='uncapped class-aware greedy HBB IoU; raw and NMS0.5 separate; grounding first box',
        timing='CUDA-synchronized; preprocessing + decode + parsing/containment; excludes image disk IO, weights, metrics',
        warmup_queries=2, cpu_threads=4, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu=torch.cuda.get_device_name(0), source_sha256=sha256_file(Path(__file__)))
    cpath = directory/f'{tag}.contract.json'
    if cpath.exists() and json.loads(cpath.read_text())!=contract:
        raise RuntimeError('Changed inference contract: use a new output directory')
    atomic_json(cpath, contract)
    existing = read_jsonl(path) if path.exists() else []
    done = {r['uid'] for r in existing}
    expected = {r['uid'] for r in rows}
    assert len(done)==len(existing) and done<=expected
    if done==expected:
        return
    worker = load_worker(args.method)
    decoders = make_decoders(worker, not args.reference, execution) if args.method=='refwave' else None
    for row in rows[:2]:
        with Image.open(row['image_path']) as f:
            image=f.convert('RGB')
        infer_one(worker,decoders,inference_query(row),image,args.method,budget)
    with path.open('a') as f:
        for index,row in enumerate(rows):
            if row['uid'] in done:
                continue
            with Image.open(row['image_path']) as opened:
                image=opened.convert('RGB')
            assert image.size==(int(row['width']),int(row['height']))
            result=infer_one(worker,decoders,inference_query(row),image,args.method,budget)
            result.update(image_id=row['image_id'],dataset=row['dataset'])
            if row['task']=='grounding':
                result.update(sample_id=row['sample_id'],query=row['query'])
            f.write(json.dumps(result,sort_keys=True)+'\n')
            f.flush()
            print(f"{args.suite} {args.method} {tag} {index+1}/{len(rows)} boxes={len(result['predictions'])} seconds={result['latency_seconds']:.3f}",flush=True)
    atomic_json(directory/f'{tag}.done.json',dict(records=len(rows)))


def merge(args):
    import numpy as np
    rows=read_jsonl(args.output/'assets'/f'{args.suite}.jsonl')
    expected={r['uid'] for r in rows}
    metrics={}
    paired={}
    for method in RUNS:
        directory=args.output/args.suite/method
        paths=[directory/f'shard{i:02d}of{args.shard_count:02d}.jsonl' for i in range(args.shard_count)]
        predictions=[r for p in paths for r in read_jsonl(p)]
        indexed={r['uid']:r for r in predictions}
        assert len(predictions)==len(indexed) and set(indexed)==expected
        predictions=[indexed[r['uid']] for r in rows]
        contracts=[json.loads(p.with_suffix('.contract.json').read_text()) for p in paths]
        ignore={'shard_index','visible_devices'}
        assert all({k:v for k,v in c.items() if k not in ignore}=={k:v for k,v in contracts[0].items() if k not in ignore} for c in contracts)
        timings=[r['latency_seconds'] for r in predictions]
        m=dict(records=len(rows),mean_seconds=float(np.mean(timings)),p50_seconds=float(np.median(timings)),
               p95_seconds=float(np.percentile(timings,95)),peak_gpu_mb=max(r['peak_gpu_mb'] for r in predictions),
               contract=contracts[0],raw_boxes=sum(len(r['predictions']) for r in predictions))
        if rows[0]['task']=='detection':
            classes=[s['canonical'] for s in rows[0]['class_specs']]
            for label,items in [('raw',predictions),('nms0p5',[dict(r,predictions=detection.nms_predictions(r['predictions'],.5)) for r in predictions])]:
                ix={(r['dataset'],r['image_id']):r for r in items}
                m[label]={str(t):detection.f1_for_threshold(rows,ix,classes,t,0) for t in (.5,.75,.95)}
            per=[]
            for row,pred in zip(rows,predictions):
                count=detection.f1_for_threshold([row],{(row['dataset'],row['image_id']):pred},classes,.5,0)
                per.append(dict(uid=row['uid'],**count))
            paired[method]=np.array([[v[k] for k in ('tp','fp','fn')] for v in per])
            m['per_class']={c:detection.f1_for_threshold(rows,{(r['dataset'],r['image_id']):r for r in predictions},[c],.5,0) for c in classes}
            jsonl(directory/'per_image_metrics.jsonl',per)
        else:
            result=grounding.evaluate_predictions(rows,predictions)
            assert result['missing_predictions']==0 and result['missing_images']==0
            m['grounding']=result['overall']
            m['by_size']=result['by_size']
            paired[method]=np.array([r['first_acc_0_5'] for r in result['rows']],dtype=float)
            jsonl(directory/'per_query_metrics.jsonl',result['rows'])
        jsonl(directory/'predictions.jsonl',predictions)
        atomic_json(directory/'metrics.json',m)
        metrics[method]=m
    assert all(metrics['standard']['contract'][k]==metrics['refwave']['contract'][k]
               for k in ('manifest_sha256','image_token_limit','max_new_tokens','shard_count'))
    rng=np.random.default_rng(0)
    differences=[]
    for _ in range(2000):
        ids=rng.integers(0,len(rows),len(rows))
        if rows[0]['task']=='detection':
            scores=[]
            for method in ('standard','refwave'):
                tp,fp,fn=paired[method][ids].sum(axis=0)
                scores.append(2*tp/max(1,2*tp+fp+fn))
            differences.append(scores[1]-scores[0])
        else:
            differences.append(float((paired['refwave'][ids]-paired['standard'][ids]).mean()))
    metrics['paired_bootstrap']=dict(replicates=2000,unit='image' if rows[0]['task']=='detection' else 'query',
        difference='RefWave minus Standard; F1@0.5 or first-box Acc@0.5',
        ci95_percentage_points=(100*np.percentile(differences,[2.5,97.5])).tolist(),
        note='Single training seed; sampling uncertainty, not training-seed reliability.')
    atomic_json(args.output/args.suite/'comparison.json',metrics)
    update_summary(args.output)
    print(json.dumps({k:{a:b for a,b in v.items() if a in ('raw','grounding','mean_seconds')} for k,v in metrics.items()},indent=2),flush=True)


def update_summary(output):
    lines=['# Cluster Stage-2 Best Checkpoint Evaluation','',
        'Both models: Stage-2 step 1200, selected independently by validation macro loss. Rank-32 Qwen LoRA + trained LM head.',
        'Native 6000 image-patch limit. Standard: direct boxes. RefWave: class waves 18, address waves 200, shared image-only Qwen prefix across rounds.',
        'Detection headline is uncapped RAW class-aware HBB IoU>=0.5; no NMS. NMS0.5 is stored separately. Grounding uses first predicted box, not best-of-boxes.',
        'Latencies are fresh local A6000 measurements with the same GPU shard assignment for both methods. Not comparable to old tables without matching their protocols.',
        'Full detection sets are the existing LAE-style releases (DIOR 586, DOTAv2 874, LAE-80C 3592), not the entire original DIOR/DOTA image collections.','',
        '| Evaluation | Method | Records | TP / FP / FN | Precision or Acc@0.5 | Recall or Acc@0.7 | F1 or mIoU | Seconds/query |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    combined={}
    for suite in SUITES:
        path=output/suite/'comparison.json'
        if not path.exists():
            lines.append(f'| {suite} | Pending | | | | | | |')
            continue
        r=json.loads(path.read_text()); combined[suite]=r
        for method in ('standard','refwave'):
            m=r[method]
            if 'raw' in m:
                v=m['raw']['0.5']; counts=f"{v['tp']} / {v['fp']} / {v['fn']}"
                a,b,c=[100*v[k] for k in ('precision','recall','f1')]
            else:
                v=m['grounding']; counts='-'
                a,b,c=[100*v[k] for k in ('first_acc_0_5','first_acc_0_7','first_mean_iou')]
            lines.append(f"| {suite} | {method} | {m['records']} | {counts} | {a:.2f} | {b:.2f} | {c:.2f} | {m['mean_seconds']:.3f} |")
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    atomic_json(output/'all_benchmark_metrics.json',combined)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','infer','merge','summary'))
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--method',choices=tuple(RUNS))
    p.add_argument('--suite',choices=SUITES)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--shard-count',type=int,default=2)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--reference',action='store_true')
    p.add_argument('--round2-execution', choices=('two_pass', 'fused_one_pass'),
                   help='Default: ROUND2_EXECUTION or fused_one_pass; --reference uses two_pass.')
    args=p.parse_args()
    {'prepare':prepare,'infer':infer,'merge':merge,'summary':lambda a:update_summary(a.output)}[args.mode](args)


if __name__=='__main__':
    main()
