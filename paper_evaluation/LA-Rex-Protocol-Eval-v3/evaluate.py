#!/usr/bin/env python3
"""Frozen plans, resumable workers and shared scoring; never trains a model."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
from common import (HERE, PROTOCOL, DEFAULT_MODELS, VLM, DETECTORS, DATASETS, SCORER_SHA,
                    read, rows, sha, digest, atomic, freeze, query_only, source_hashes, jobs)


def plan(args):
    data=read(args.output/'data.json')
    identities=read(args.output/'models.json')
    missing=set(args.models)-set(identities)
    if missing: raise ValueError('Prepare model identities first: '+str(missing))
    if len(args.models)!=len(set(args.models)): raise ValueError('Duplicate model names in plan')
    if not 1<=args.shards<=8 or args.budget<1: raise ValueError('Invalid shards/budget')
    sources=source_hashes()
    sources[str(HERE/'evidence/other_metric.py')]=SCORER_SHA
    from native_parser import SOURCE, SOURCE_SHA
    sources[str(SOURCE)]=SOURCE_SHA
    if set(args.models)&set(VLM)-{'qwen3_4b','qwen3_8b','rex_omni'}:
        for p in (args.controls/'workspace').rglob('*.py'):
            if '__pycache__' not in p.parts: sources[str(p.resolve())]=sha(p)
    selected={m:identities[m] for m in args.models}
    if 'laedino' in selected: selected['bert']=identities['bert']
    for m,i in selected.items():
        if 'code' in i:
            for p in Path(i['code']).rglob('*.py'):
                if '.git' not in p.parts: sources[str(p.resolve())]=sha(p)
    p=dict(protocol=PROTOCOL,models=selected,run_models=args.models,datasets=data,
        controls=str(args.controls.resolve()),sources=sources,shards=args.shards,budget=args.budget,
        detector_threshold=.25,detector_text_threshold=.25,additional_nms=False,
        detector_reporting='native LAE top300 primary + all-pairs sensitivity; GDINO native queries',
        geometry_policy='LA-native primary for LA-family outputs; strict no-repair sensitivity on the same raw answers; other native interfaces unchanged',
        generation=dict(la=dict(temperature=.7,top_p=.9,repetition_penalty=1.1,mode='hybrid'),
            qwen=dict(temperature=.7,top_p=.9,top_k=0,repetition_penalty=1.1),
            rex=dict(temperature=0.,top_p=.05,top_k=1,repetition_penalty=1.05)),
        scope='GT-present categories only, vocabulary order; no GT counts, boxes, points or crops in prompts',
        detection_metric='LA dense image-average P/R harmonic F1; class-aware primary, released compatibility',
        pointing_metric='released LA class-aware GT-first HBB containment; image-average P/R harmonic F1 + micro',
        detector_pointing=False,smoke_records=3,warmup_records=1,
        timing='synchronized preprocessing + all requested query generation + parsing; excludes disk image loading, scoring, warmup and queue',
        laedino_python=os.path.abspath(args.laedino_python) if args.laedino_python else None)
    freeze(args.output/'plan.json',p)
    print('Frozen',len(jobs(args.models,args.shards)),'full jobs; smoke separate; protocol',PROTOCOL)


def load_plan(output,check_sources=True):
    p=read(output/'plan.json')
    if p['protocol']!=PROTOCOL: raise ValueError('Wrong output protocol')
    if check_sources:
        for file,expected in p['sources'].items():
            if sha(file)!=expected: raise ValueError('Code changed after plan: '+file)
    data={}
    for ds,info in p['datasets'].items():
        if sha(info['path'])!=info['sha256']: raise ValueError('Manifest changed')
        data[ds]=rows(info['path'])
    return p,data


def paths(output,model,kind,ds,shard,count,smoke):
    folder=output/('smoke' if smoke else 'predictions')/model/kind/ds
    return folder/f'shard{shard:02d}of{count:02d}.jsonl'


def selected_rows(data,ds,shard,count,smoke):
    if smoke:
        # Fixed sparse, multi-class and maximum-density cases; never use model performance.
        ordered=sorted(data[ds],key=lambda r:(len(r['gt']),r['uid']))
        first=ordered[0]
        mixed=next((r for r in ordered if len({g['label'] for g in r['gt']})>=3),None)
        second=mixed or next((r for r in ordered if r['uid']!=first['uid']),None)
        dense=sorted(data[ds],key=lambda r:(-len(r['gt']),r['uid']))[0]
        chosen={r['uid']:r for r in (first,second,dense) if r is not None}
        return list(chosen.values())
    return data[ds][shard::count]


def finished(output,model,kind,ds,shard,count,smoke,p,data):
    path=paths(output,model,kind,ds,shard,count,smoke)
    done=path.with_suffix('.done.json')
    if not done.exists(): return False
    j=read(done)
    selected=selected_rows(data,ds,shard,count,smoke)
    if j.get('plan_sha256')!=digest(p) or j.get('uids')!=[r['uid'] for r in selected]:
        raise ValueError('Completion contract differs: '+str(done))
    if not path.exists() or sha(path)!=j['predictions_sha256']:
        raise ValueError('Completed predictions changed: '+str(path))
    return True


def worker(args):
    from common import safe_cache
    safe_cache()
    import torch
    from PIL import Image
    from models import Backend,software
    p,data=load_plan(args.output)
    count=1 if args.smoke else p['shards']
    if args.model not in p['run_models'] or (args.model in DETECTORS and args.kind!='boxes'):
        raise ValueError('Unplanned model/task')
    if not 0<=args.shard<count: raise ValueError('Invalid shard')
    path=paths(args.output,args.model,args.kind,args.dataset,args.shard,count,args.smoke)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if finished(args.output,args.model,args.kind,args.dataset,args.shard,count,args.smoke,p,data): return
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        env=software()
        freeze(args.output/('software_'+args.model+'.json'),env)
        if args.model!='laedino':
            freeze(args.output/'software_shared_vlm.json',env)
        expected=selected_rows(data,args.dataset,args.shard,count,args.smoke)
        plan_hash=digest(p)
        saved=rows(path) if path.exists() else []
        if [r['uid'] for r in saved]!=[r['uid'] for r in expected[:len(saved)]]:
            raise ValueError('Invalid partial record coverage')
        if any(r.get('plan_sha256')!=digest(p) for r in saved): raise ValueError('Mixed prediction contracts')
        for info in (p['models'][args.model],p['models'].get('bert',{})):
            for file,identity in info.get('files',{}).items():
                stat=Path(file).stat()
                if (stat.st_size,stat.st_mtime_ns)!=(identity['bytes'],identity['mtime_ns']):
                    raise ValueError('Model changed since hash verification: '+file)
        model=Backend(args.model,p)
        def one(row):
            file=Path(row['image_path'])
            if sha(file)!=p['datasets'][args.dataset]['images'][str(file)]['sha256']:
                raise ValueError('Image hash changed')
            with Image.open(file) as im: image=im.convert('RGB')
            if image.size!=(row['width'],row['height']): raise ValueError('Image dimensions changed')
            query=query_only(row)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            with torch.inference_mode(): result=model.infer(query,image,args.kind)
            torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            result.update(uid=row['uid'],suite=args.dataset,query=query,plan_sha256=plan_hash,
                latency_seconds=seconds,peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2)
            if not math.isfinite(result['latency_seconds']): raise ValueError('Invalid timer')
            return result
        for row in expected[:p['warmup_records']]: one(row)
        with path.open('a') as f:
            for row in expected[len(saved):]:
                result=one(row)
                f.write(json.dumps(result,sort_keys=True,allow_nan=False)+'\n')
                f.flush()
                saved.append(result)
                atomic(path.with_suffix('.progress.json'),dict(records=len(saved),expected=len(expected)))
                print(args.model,args.kind,args.dataset,len(saved),'/',len(expected),
                      'outputs',len(result['predictions']),'seconds',round(result['latency_seconds'],3),flush=True)
        atomic(path.with_suffix('.done.json'),dict(plan_sha256=digest(p),uids=[r['uid'] for r in expected],
               predictions_sha256=sha(path),complete=True))


def merge(args):
    import numpy as np
    from scoring import score,load_upstream,select_predictions
    from native_parser import reparse
    p,data=load_plan(args.output)
    scorer=load_upstream()
    summary={}
    for model in p['run_models']:
        for kind in (('boxes',) if model in DETECTORS else ('boxes','pointing')):
            for ds in DATASETS:
                count=1 if args.smoke else p['shards']
                predictions,hashes=[],{}
                for shard in range(count):
                    if not finished(args.output,model,kind,ds,shard,count,args.smoke,p,data):
                        raise ValueError(f'Incomplete: {model}/{kind}/{ds}/{shard}')
                    path=paths(args.output,model,kind,ds,shard,count,args.smoke)
                    predictions.extend(rows(path)); hashes[str(path)]=sha(path)
                index={r['uid']:r for r in predictions}
                expected=selected_rows(data,ds,0,1,args.smoke)
                if len(index)!=len(predictions) or set(index)!={r['uid'] for r in expected}:
                    raise ValueError('Missing, extra or duplicate predictions')
                predictions=[index[r['uid']] for r in expected]
                if any(r.get('plan_sha256')!=digest(p) for r in predictions): raise ValueError('Mixed plans')
                replay=[reparse(model,r,o,kind,True) for r,o in zip(expected,predictions)]
                if any(a['predictions']!=b['predictions'] for a,b in zip(replay,predictions)):
                    raise ValueError('Saved predictions differ from native-parser replay')
                detector=model in DETECTORS
                threshold=.25 if detector else None
                cap=300 if model=='laedino' else None
                result,per=score(expected,predictions,kind,threshold,cap,scorer)
                strict=[reparse(model,r,o,kind,False) for r,o in zip(expected,predictions)]
                result['strict_sensitivity'],_=score(expected,strict,kind,threshold,cap,scorer)
                result['parsing_policy']=p['geometry_policy']
                times=[r['latency_seconds'] for r in predictions]
                outputs=sum(len(select_predictions(r['predictions'],threshold,cap)) for r in predictions)
                metric=result['scores']['class_aware' if kind=='boxes' else 'released'][str(.5) if kind=='boxes' else 'point']
                result.update(complete=True,model=model,task=kind,dataset=ds,smoke=args.smoke,
                    prediction_hashes=hashes,plan_sha256=digest(p),seconds_per_complete_image=sum(times)/len(times),
                    p50_seconds=float(np.percentile(times,50)),p95_seconds=float(np.percentile(times,95)),
                    outputs=outputs,outputs_per_second=outputs/sum(times),
                    valid_geometry_outputs=outputs-result['invalid_geometry_outputs'],
                    valid_geometry_outputs_per_second=(outputs-result['invalid_geometry_outputs'])/sum(times),
                    correct_outputs_per_second=metric['micro']['tp']/sum(times),
                    peak_gpu_mb=max(r['peak_gpu_mb'] for r in predictions),
                    invalid_items=sum(r.get('invalid_items',0) for r in predictions),
                    saturated_queries=sum(r.get('saturated_queries',0) for r in predictions),
                    generated_tokens=sum(r['generated_tokens'] for r in predictions) if not detector else None)
                if model=='laedino':
                    result['uncapped_same_forward'],_=score(expected,predictions,kind,.25,None,scorer)
                    result['author_threshold_0p30_native'],_=score(expected,predictions,kind,.30,300,scorer)
                folder=path.parent
                atomic(folder/'metrics.json',result)
                atomic(folder/'per_image.json',per)
                summary['/'.join((model,kind,ds))]=result
                print('SCORED',model,kind,ds,'images',len(expected),flush=True)
    atomic(args.output/('smoke_summary.json' if args.smoke else 'summary.json'),summary)
    from report import render
    (args.output/('SMOKE.md' if args.smoke else 'RESULTS.md')).write_text(render(summary))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['plan','worker','merge'])
    p.add_argument('--output',type=Path,default=HERE/'results')
    p.add_argument('--controls',type=Path,default=HERE.parent/'Final-Controls-Eval')
    p.add_argument('--models',nargs='+',choices=list(VLM+DETECTORS),default=list(DEFAULT_MODELS))
    p.add_argument('--budget',type=int,default=8192)
    p.add_argument('--shards',type=int,default=4)
    p.add_argument('--laedino-python')
    p.add_argument('--model',choices=list(VLM+DETECTORS))
    p.add_argument('--kind',choices=['boxes','pointing'])
    p.add_argument('--dataset',choices=list(DATASETS))
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args(); args.output=args.output.resolve(); args.controls=args.controls.resolve()
    globals()[args.mode](args)


if __name__=='__main__': main()
