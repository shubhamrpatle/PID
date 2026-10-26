#!/usr/bin/env python3
"""Final full-FT standard and class-isolated direct controls on frozen benchmarks."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import prepare_final_standard_isolated as release
import evaluate_rex_pointing_full as base
import evaluate_refwave_cluster_stage2 as prior

OUTPUT = release.OUTPUT
METHODS = tuple(release.SPECS)
BUDGET = 8192


def prepare(output):
    selected = release.read(output/'selection.json')
    manifests = {s:release.read(output/'assets'/f'{s}.json') for s in base.SUITES}
    for suite, summary in manifests.items():
        if base.common.sha(output/'assets'/f'{suite}.jsonl') != summary['sha256']:
            raise ValueError('Benchmark manifest changed')
    sources = {Path(__file__).resolve(), Path(release.__file__).resolve(),
               Path(base.__file__).resolve(), Path(prior.__file__).resolve(),
               Path(base.common.__file__).resolve()}
    names = ['evaluate_locateanything_lae_style.py', 'evaluate_locateanything_grounding_benchmarks.py',
             'evaluate_molmo_pointing.py', 'run_locany_pointing_balanced100.py']
    sources.update(base.PROJECT/'scripts'/name for name in names)
    for directory in (base.ROOT/'Pixel-PIVR/src/pixel_pivr', base.ROOT/'RefWave/src/refwave',
                      base.ROOT/'zero_shot_analysis', base.BASE):
        sources.update(directory.glob('*.py'))
    signatures = {str(p):base.common.sha(p) for p in sorted(sources)}
    plan = dict(protocol='final_std_ciso_direct_v1', repo=release.REPO, revision=release.REVISION,
        models={k:dict(run=v['run'],sha256=v['sha256'],step=v['step']) for k,v in selected.items()},
        manifests=manifests, sources=signatures, generation=dict(max_new_tokens=BUDGET,
            temperature=.7,top_p=.9,repetition_penalty=1.1,generation_mode='hybrid'),
        image_token_limit=6000, added_nms=False, max_detections=None,
        standard_detection='one all-class direct-box query',
        isolated_detection='all vocabulary classes; concurrent class-conditioned direct boxes; shared prefix',
        grounding='one full expression; direct boxes; first-box metrics',
        pointing='native Point to: {reference}. query; no box-center conversion; same protocol for both models',
        point_matching='one-to-one point inside HBB; all fixed classes, including absent classes',
        timing='synchronized preprocessing + generation + parsing, excluding disk IO, load, warmup and scoring',
        comparison_limits=['Training and validation record counts/hashes differ between these supplied runs.',
            'Best selection uses each run\'s own validation set, never the benchmark.',
            'Class-isolated direct decoding is not the two-round RefWave/PID method.',
            'Pointing is evaluated natively, regardless of the available pointing training supervision.'])
    base.freeze(output/'plan.json', plan)
    for source in sources:
        dest = output/'source_snapshot'/source.relative_to(base.ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(source, dest)
    report(output)
    return plan


def check_sources(plan):
    for path, digest in plan['sources'].items():
        if base.common.sha(path) != digest:
            raise ValueError('Evaluation source changed: '+path)


def load_worker(method, output):
    import torch
    from pixel_pivr.worker import LocateAnythingPixelPIVRWorker
    from refwave.model import configure_full
    from refwave.train import restore
    info = release.read(output/'checkpoints_ready.json')['models'][method]
    path = Path(info['path'])
    if (path.stat().st_size,path.stat().st_mtime_ns) != (info['bytes'],info['mtime_ns']):
        raise ValueError('Checkpoint changed after hash verification')
    selected = release.read(output/'selection.json')[method]
    saved = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    tied = release.read(base.BASE/'config.json')['text_config'].get('tie_word_embeddings',False)
    release.validate_saved(saved,method,selected['configs'],selected['completions'],selected['curves'],tied)
    worker = LocateAnythingPixelPIVRWorker(base.BASE,device='cuda:0')
    configure_full(worker.model)
    restore(worker.model,saved['trainable_state'])
    worker.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    worker.processor.image_processor.in_token_limit=6000
    del saved
    gc.collect()
    torch.cuda.empty_cache()
    return worker


def isolated_branches(row):
    from pixel_pivr.class_wave import ClassPointBranch
    from refwave.direct_control import detection_question
    return [ClassPointBranch(s['canonical'],detection_question(s['prompt']),
        prior.stable_seed(0,row['suite'],row['uid'],s['canonical'],'isolated-direct-box'))
        for s in row['class_specs']]


def parse_isolated_answers(answers, branches, specs, size):
    from zero_shot_analysis.locate_anything import parse_locateanything_labeled_boxes
    if len(answers)!=len(branches):
        raise ValueError('Missing class-isolated answer')
    aliases=prior.detection.build_alias_lookup(specs)
    classes={s['canonical'] for s in specs}
    predictions=[]
    for answer, branch in zip(answers,branches):
        for item in parse_locateanything_labeled_boxes(answer['answer'],*size,default_label=branch.label):
            label=prior.detection.canonicalize_label(item['label'],aliases,classes)
            box=prior.detection.clip_hbox_to_image(item['hbox'],*size)
            if label is not None and box is not None:
                predictions.append(dict(label=label,hbox=box,score=1.0))
    return predictions


def boxes(worker, decoder, row, image, method):
    import torch
    if method=='standard' or row['task']=='grounding':
        return prior.infer_one(worker,None,prior.inference_query(row),image,'standard',BUDGET)
    # Vocabulary-only query construction. GT and target-class fields are never used here.
    branches=isolated_branches(row)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    answers,stats=decoder.generate(image,branches,max_new_tokens=BUDGET,temperature=.7,
        top_p=.9,repetition_penalty=1.1,generation_mode='hybrid')
    predictions=parse_isolated_answers(answers,branches,row['class_specs'],image.size)
    torch.cuda.synchronize()
    return dict(predictions=predictions,answers=answers,decode_stats=stats,
        latency_seconds=time.perf_counter()-started,peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2)


def points(worker,row,image):
    import torch
    answers=[]
    for ref in base.references(row):
        seed_text=json.dumps([0,row['suite'],row['uid'],ref['label']])
        seed=int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4],'big')
        prior.grounding.set_generation_seed(seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        answer=base.native_point(worker,image,ref,SimpleNamespace(model='base_la',budget=BUDGET))
        torch.cuda.synchronize()
        answer.update(latency_seconds=time.perf_counter()-started,
                      peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2,generation_seed=seed)
        answers.append(answer)
    return dict(answers=answers,points=[p for a in answers for p in a['points']],
        latency_seconds=sum(a['latency_seconds'] for a in answers),
        peak_gpu_mb=max(a['peak_gpu_mb'] for a in answers))


def infer(args):
    import torch
    from PIL import Image
    from pixel_pivr.fast_class_wave import FastCachedHybridClassWaveDecoder
    from refwave.io import append_jsonl,read_jsonl_log
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    plan=release.read(args.output/'plan.json')
    check_sources(plan)
    selected=release.read(args.output/'selection.json')[args.model]
    manifest=args.output/'assets'/f'{args.suite}.jsonl'
    if base.common.sha(manifest)!=plan['manifests'][args.suite]['sha256']:
        raise ValueError('Manifest changed')
    rows=base.common.read(manifest)
    if args.limit:
        rows=rows[:args.limit]
    rows=rows[args.shard_index::args.shard_count]
    directory=base.folder(args)
    directory.mkdir(parents=True,exist_ok=True)
    tag=f'shard{args.shard_index:02d}of{args.shard_count:02d}'
    contract=dict(protocol=plan['protocol'],plan_sha256=base.common.sha(args.output/'plan.json'),
        checkpoint_sha256=selected['sha256'],step=selected['step'],method=args.model,kind=args.kind,
        suite=args.suite,manifest_sha256=plan['manifests'][args.suite]['sha256'],
        shard_index=args.shard_index,shard_count=args.shard_count,limit=args.limit,
        warmup_records=min(2,len(rows)),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
        generation=plan['generation'],image_token_limit=6000,added_nms=False,max_detections=None,
        native_query_batch=1,class_isolated_max_concurrent_references=32,
        timing=plan['timing'])
    base.freeze(directory/(tag+'.contract.json'),contract)
    path=directory/(tag+'.jsonl')
    previous=read_jsonl_log(path,repair_tail=True) if path.exists() else []
    done={r['uid'] for r in previous}
    if len(done)!=len(previous) or not done<={r['uid'] for r in rows}:
        raise ValueError('Duplicate/foreign saved predictions')
    if len(done)<len(rows):
        worker=load_worker(args.model,args.output)
        decoder=(FastCachedHybridClassWaveDecoder(worker,image_token_limit=6000,
                    max_wave_size=32,bucket_suffix_lengths=False)
                 if args.model=='class_isolated' and args.kind=='boxes' else None)

        def one(row):
            with Image.open(row['image_path']) as im:
                image=im.convert('RGB')
            if image.size!=(row['width'],row['height']):
                raise ValueError('Image dimensions differ from manifest')
            with torch.inference_mode():
                result=(points(worker,row,image) if args.kind=='pointing'
                        else boxes(worker,decoder,row,image,args.model))
            result.update({k:row[k] for k in ('uid','dataset','image_id','task','suite')})
            if row['task']=='grounding':
                result.update(sample_id=row['sample_id'],query=row['query'])
            return result

        for row in rows[:2]:
            one(row)
        for row in rows:
            if row['uid'] in done:
                continue
            result=one(row)
            append_jsonl(path,result)
            done.add(row['uid'])
            base.common.write(directory/(tag+'.status.json'),dict(records=len(done),expected=len(rows),
                latest_seconds=result['latency_seconds'],updated=time.time()))
            print(args.model,args.kind,args.suite,f'{len(done)}/{len(rows)}',
                  f"{result['latency_seconds']:.3f}s",flush=True)
    base.common.write(directory/(tag+'.done.json'),dict(complete=True,records=len(rows)))


def report(output):
    lines=['# Final Standard / Class-Isolated Stage-2 Models','',
        'Validation-selected step 1200 for each supplied full-FT run. No benchmark checkpoint selection.',
        'Training and validation record counts differ; these are not claimed to be data-identical runs.','',
        '| Model | Task | Benchmark | Records | Scores (%) | Mean s/record |',
        '| --- | --- | --- | ---: | --- | ---: |']
    completed=[]
    for model in METHODS:
        for kind in ('boxes','pointing'):
            for suite in base.SUITES:
                path=output/'results'/kind/model/suite/'metrics.json'
                count=release.read(output/'assets'/f'{suite}.json')['records']
                score=timing='-'
                if path.exists():
                    m=release.read(path)
                    if not m['complete'] or m['records']!=count or m['contract']['limit']:
                        raise ValueError('Incomplete or diagnostic result in full report')
                    if 'pointing' in m:
                        values=[m['pointing']['f1']]
                    elif 'raw' in m:
                        values=[m['raw']['0.5']['f1'],m['raw']['0.75']['f1'],m['mean_f1_iou_0p50_to_0p95']]
                    else:
                        values=[m['grounding'][k] for k in ('first_acc_0_5','first_acc_0_7','first_mean_iou')]
                    score=' / '.join(f'{100*v:.2f}' for v in values)
                    timing=f"{m['mean_seconds']:.3f}"
                    completed.append(dict(model=model,kind=kind,suite=suite,metrics=m))
                lines.append(f'| {model} | {kind} | {suite} | {count} | {score} | {timing} |')
    lines+=['','Detection: F1@0.50 / F1@0.75 / mean F1 across all ten IoUs 0.50:0.05:0.95.',
        'Grounding: first-box Acc@0.5 / Acc@0.7 / mIoU. Pointing: one-to-one point-inside-HBB F1.',
        'Detection-pointing seconds cover all class queries per image; grounding-pointing seconds cover one expression.',
        'Native points only; no conversion of predicted boxes into points. Raw boxes without added NMS/maxDets100.','']
    (output/'comparison.md').write_text('\n'.join(lines))
    base.common.write(output/'comparison.json',dict(expected_cells=16,completed_cells=len(completed),cells=completed))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','infer','merge','report'])
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--model',choices=METHODS,default='standard')
    p.add_argument('--kind',choices=['boxes','pointing'],default='boxes')
    p.add_argument('--suite',choices=base.SUITES,default='DIOR')
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--shard-count',type=int,default=2)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    if args.limit<0 or not 0<=args.shard_index<args.shard_count:
        p.error('Invalid limit/shard')
    if args.mode=='prepare': prepare(args.output)
    elif args.mode=='report': report(args.output)
    elif args.mode=='merge':
        base.merge(args)
        if not args.limit: report(args.output)
    else: infer(args)


if __name__=='__main__':
    main()
