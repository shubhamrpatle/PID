#!/usr/bin/env python3
"""Paired opt-in Round-2 fusion tests on the current checkpoint and frozen points."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

import evaluate_refwave_cluster_lora_twoepoch as run
from pixel_pivr.class_wave import ClassPointBranch
from zero_shot_analysis.pixel_pivr_wave import AddressedCrop
from zero_shot_analysis.fused_pivr_geometry import FusedPIVRGeometryDecoder
from evaluate_pixel_pivr_sequential_wave_balanced100 import compact_round2_prompt, point_to_norm

OUTPUT = run.ref.PROJECT/'runs/refwave_fused_round2_20260911'
SOURCE = run.OUTPUT/'resume_2gpu'
SUITE = 'DOTAv2-Balanced100'
ARMS = [('two_pass', 1), ('fused_one_pass', 1), ('two_pass', 200), ('fused_one_pass', 200)]


def name(mode, wave):
    return f'{mode}_wave{wave}'


def addresses(points, size):
    return [AddressedCrop(str(i), p['label'], None, compact_round2_prompt(p['label'],p['point'],*size),
                          point_normalized=point_to_norm(p['point'],*size)) for i,p in enumerate(points)]


def point_branches(row):
    query = run.ref.inference_query(row)
    return [ClassPointBranch(b['label'], b['question'],
        run.ref.stable_seed(0,query['suite'],query['uid'],b['label'],'isolated-point')) for b in query['references']]


def inputs():
    manifest = SOURCE/'assets'/f'{SUITE}.jsonl'
    rows = run.ref.read_jsonl(manifest)
    paths = sorted((SOURCE/'benchmarks'/SUITE).glob('shard*.jsonl'))
    saved = [p for path in paths for p in run.ref.read_jsonl(path)]
    indexed = {p['uid']:p for p in saved}
    if len(saved) != len(rows) or set(indexed) != {r['uid'] for r in rows}:
        raise ValueError('Frozen Round-1 predictions are incomplete or duplicated')
    for path in paths:
        contract = json.loads(path.with_suffix('.contract.json').read_text())
        if contract['checkpoint_sha256'] != run.SHA or contract['manifest_sha256'] != run.ref.sha256_file(manifest):
            raise ValueError('Frozen points/checkpoint/manifest mismatch')
    return rows, indexed, paths, manifest


def prepare(args):
    import torch
    import transformers
    rows, saved, paths, manifest = inputs()
    base_plan = json.loads((SOURCE/'plan.json').read_text())
    sources = dict(base_plan['sources'])
    sources.update({str(p):run.ref.sha256_file(p) for p in [Path(__file__).resolve(),
        run.ref.ROOT/'zero_shot_analysis/fused_pivr_geometry.py']})
    if any(run.ref.sha256_file(Path(p)) != sha for p,sha in sources.items()):
        raise ValueError('The baseline inference source changed')
    contract = dict(checkpoint=str(run.CHECKPOINT), checkpoint_sha256=run.SHA, step=2100,
        manifest=str(manifest), manifest_sha256=run.ref.sha256_file(manifest),
        points={str(p):run.ref.sha256_file(p) for p in paths}, source_sha256=sources,
        torch=torch.__version__, transformers=transformers.__version__, arms=ARMS,
        repeats=args.repeats, image_token_limit=6000, round1_wave=18, max_new_tokens=2048,
        temperature=.7,top_p=.9,repetition_penalty=1.1, compact_head=False,
        raw_policy=base_plan['raw_policy'], cpu_threads=4, warmup_images=2,
        frozen_total='Archived Round-1 seconds + measured Round-2 seconds; derived, not an end-to-end timing.',
        end_to_end_total='Measured preprocessing + Round 1 + Round 2 + parsing, no disk IO/loading/metrics.',
        gpu=torch.cuda.get_device_name(0), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        ordering='Alternating arm order by image/repeat; same GPU; no claim of isolated-node timing.',
        adoption='Require exact outputs AND positive repeated latency benefit; default remains two_pass.')
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'contract.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(contract)):
        raise ValueError('Contract differs; use another output directory')
    run.ref.atomic_json(path,contract)
    for p,sha in sources.items():
        dest=args.output/'source_snapshot'/Path(p).relative_to(run.ref.ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            shutil.copy2(p,dest)
    return rows,saved


def make_decoders(worker):
    point = run.ref.make_decoders(worker,optimized=True)[0]
    box = FusedPIVRGeometryDecoder(worker,image_token_limit=6000,visual_context='global_only',
        prefix_cache_mode='shared',geometry_prefix_mode='box_only',compact_head=False)
    return point,box


def geometry_signature(results):
    return [(p['address_id'],p['label'],p['block_tokens']) for p in results]


def capture_decode(worker, box, prepared, owned, mode, wave):
    import torch
    from zero_shot_analysis.pointrelay_pbd import base_causal_lm
    chunks=[]
    original=box._geometry_from_hidden
    ids=worker.model.token_ids
    def capture(hidden,branches,*,allow_none):
        logits=torch.nn.functional.linear(hidden,base_causal_lm(worker.model.language_model).lm_head.weight)
        relevant=torch.cat((logits[:,:,ids['coord_start_token_id']:ids['coord_end_token_id']+1],
                            logits[:,:,ids['none_token_id']:ids['none_token_id']+1]),dim=-1)
        chunks.append(relevant.float().cpu())
        del logits,relevant
        return original(hidden,branches,allow_none=allow_none)
    forwards=[]
    hook=base_causal_lm(worker.model.language_model).model.register_forward_hook(
        lambda module,args,output: forwards.append(1))
    box._geometry_from_hidden=capture
    box.round2_execution=mode
    try:
        results,stats=box.decode_prepared_image(prepared,owned,requested_wave_size=wave,allow_none=True)
    finally:
        box._geometry_from_hidden=original
        hook.remove()
    if len(forwards) != stats['qwen_total_forwards']:
        raise ValueError('Reported decoder call count differs from the actual forwards')
    return results,stats,torch.cat(chunks) if chunks else torch.empty(0)


def gate(worker, decoders, rows, frozen, output):
    import torch
    from PIL import Image
    # Capacity probes use predicted addresses. Repeating them is labelled synthetic.
    row=max(rows,key=lambda r:len(frozen[r['uid']]['points']))
    with Image.open(row['image_path']) as f:
        image=f.convert('RGB')
    prepared=decoders[0].prepare_image(image,point_branches(row))
    owned=addresses(frozen[row['uid']]['points'],image.size)
    snapshot=[(k.clone(),v.clone()) for k,v in prepared.key_values]
    records=[]
    for count,wave in [(0,200),(1,1),(18,18),(200,200),(203,200)]:
        probe=[owned[i%len(owned)] for i in range(count)]
        a,sa,la=capture_decode(worker,decoders[1],prepared,probe,'two_pass',wave)
        b,sb,lb=capture_decode(worker,decoders[1],prepared,probe,'fused_one_pass',wave)
        equal=[x['block_tokens']==y['block_tokens'] for x,y in zip(a,b)]
        different=[dict(address=probe[i].address_id, label=probe[i].label,
            two_pass=x['block_tokens'],fused_one_pass=y['block_tokens'])
            for i,(x,y) in enumerate(zip(a,b)) if x['block_tokens']!=y['block_tokens']]
        result=dict(addresses=count,wave=wave,source_uid=row['uid'],synthetic_repetition=count>len(owned),
            exact_blocks=sum(equal),none_equal=sum(x['block_type']==y['block_type'] for x,y in zip(a,b)),
            two_pass_forwards=sa['qwen_total_forwards'],fused_forwards=sb['qwen_total_forwards'],
            max_logit_abs_diff=float((la-lb).abs().max()) if count else 0,
            mean_logit_abs_diff=float((la-lb).abs().mean()) if count else 0,
            differences=different[:20],reference_none=sum(x['block_type']=='empty_box' for x in a),
            fused_none=sum(x['block_type']=='empty_box' for x in b))
        if len(a)!=count or len(b)!=count:
            raise ValueError('Address lost during wave continuation')
        if not all(torch.equal(x,y) for old,new in zip(snapshot,prepared.key_values) for x,y in zip(old,new)):
            raise ValueError('Shared prefix was modified')
        records.append(result)
        print('GATE',json.dumps(result),flush=True)
    report=dict(cases=records,unchanged_decoded_outputs=all(r['exact_blocks']==r['addresses'] for r in records),
        shared_cache_unchanged=True,default_changed=False,checkpoint_sha256=run.SHA,
        note='BF16 checkpoint capacity probes, not benchmark metrics; CPU tests verify native mask semantics.')
    run.ref.atomic_json(output/'correctness_gate.json',report)
    del prepared,snapshot
    gc.collect();torch.cuda.empty_cache()
    return report


def frozen_one(worker, decoders, row, saved, image, prepared, mode, wave):
    import torch
    owned=addresses(saved['points'],image.size)
    box=decoders[1];box.round2_execution=mode
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    refined,stats=box.decode_prepared_image(prepared,owned,requested_wave_size=wave,allow_none=True)
    query=run.ref.inference_query(row)
    predictions=[p for r in refined for p in run.ablation.parsed_boxes(r['answer'],query,image.size)]
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    return dict(uid=row['uid'],dataset=row['dataset'],image_id=row['image_id'],predictions=predictions,
        refinements=refined,round2=stats,round2_seconds=elapsed,
        latency_seconds=saved['round1_seconds']+elapsed,round1_seconds=saved['round1_seconds'],
        peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2,
        timing='R2 measured; total uses frozen historical R1, not end-to-end timing.')


def bench(worker,decoders,rows,frozen,args,kind):
    from PIL import Image
    from refwave.io import read_jsonl_log
    output=args.output/f'{kind}.jsonl'
    previous=read_jsonl_log(output,repair_tail=True)
    done={(r['uid'],r['repeat']) for r in previous}
    if len(done)!=len(previous):
        raise ValueError('Duplicate completed pairs')
    selected=rows[:args.limit] if args.limit else rows
    config=dict(budget=2048,class_wave=18,box_wave=200)

    def one(row,image,prepared,mode,wave):
        decoders[1].round2_execution=mode
        if kind=='frozen':
            return frozen_one(worker,decoders,row,frozen[row['uid']],image,prepared,mode,wave)
        config['box_wave']=wave
        return run.ablation.infer_one(worker,decoders,run.ref.inference_query(row),image,'refwave',config)

    if len(done)<len(selected)*args.repeats:
        for row in selected[:2]:
            with Image.open(row['image_path']) as f:
                image=f.convert('RGB')
            prepared=decoders[0].prepare_image(image,point_branches(row)) if kind=='frozen' else None
            for mode,wave in ARMS:
                one(row,image,prepared,mode,wave)
            del prepared
    for index,row in enumerate(selected):
        with Image.open(row['image_path']) as f:
            image=f.convert('RGB')
        prepared=decoders[0].prepare_image(image,point_branches(row)) if kind=='frozen' else None
        for repeat in range(args.repeats):
            if (row['uid'],repeat) in done:
                continue
            order=ARMS if (index+repeat)%2==0 else list(reversed(ARMS))
            results={name(mode,wave):one(row,image,prepared,mode,wave) for mode,wave in order}
            equality={str(wave):geometry_signature(results[name('two_pass',wave)]['refinements'])==
                geometry_signature(results[name('fused_one_pass',wave)]['refinements']) for wave in (1,200)}
            if kind=='end_to_end':
                equality['round1']=all(results[name('two_pass',1)]['point_answers']==r['point_answers']
                    for r in results.values())
            run.append_jsonl(output,dict(uid=row['uid'],repeat=repeat,arms=results,equality=equality))
            print(kind,index+1,len(selected),'repeat',repeat,'equal',equality,
                  'R2', {k:round(v['round2_seconds'],3) for k,v in results.items()},flush=True)
        del prepared
    summarize(args,selected,kind)


def summarize(args,rows,kind):
    import numpy as np
    pairs=run.ref.read_jsonl(args.output/f'{kind}.jsonl')
    expected={(r['uid'],repeat) for r in rows for repeat in range(args.repeats)}
    if len(pairs)!=len(expected) or {(r['uid'],r['repeat']) for r in pairs}!=expected:
        raise ValueError('Incomplete paired benchmark')
    first={r['uid']:r for r in pairs if r['repeat']==0}
    report=dict(images=len(rows),repeats=args.repeats,kind=kind,arms={},equivalence={},default_changed=False)
    for mode,wave in ARMS:
        tag=name(mode,wave)
        predictions=[first[r['uid']]['arms'][tag] for r in rows]
        for row,pred in zip(rows,predictions):
            pred.update(image_id=row['image_id'],dataset=row['dataset'])
        metrics,per=run.full.score(rows,predictions)
        measures=[p['arms'][tag] for p in pairs]
        for field in ('round2_seconds','latency_seconds'):
            times=[p[field] for p in measures]
            metrics[field+'_mean']=float(np.mean(times))
            metrics[field+'_p50']=float(np.median(times))
            metrics[field+'_p95']=float(np.percentile(times,95))
        metrics['peak_gpu_mb']=max(p['peak_gpu_mb'] for p in measures)
        metrics['qwen_round2_forwards']=sum(p['round2']['qwen_total_forwards'] for p in predictions)
        report['arms'][tag]=metrics
    for wave in (1,200):
        a,b=name('two_pass',wave),name('fused_one_pass',wave)
        times=np.array([[p['arms'][a]['round2_seconds'],p['arms'][b]['round2_seconds']] for p in pairs])
        image_times=np.array([[np.mean([p['arms'][tag]['round2_seconds'] for p in pairs if p['uid']==r['uid']])
                              for tag in (a,b)] for r in rows])
        rng=np.random.default_rng(20260911)
        indices=rng.integers(0,len(rows),(5000,len(rows)))
        boot=image_times[indices].mean(axis=1)
        reduction=1-boot[:,1]/boot[:,0]
        report['equivalence'][str(wave)]=dict(exact_pairs=sum(p['equality'][str(wave)] for p in pairs),
            total_pairs=len(pairs),round2_speedup=float(times[:,0].mean()/times[:,1].mean()),
            latency_reduction_paired_image_ci95=np.quantile(reduction,[.025,.975]).tolist(),
            exact_blocks=sum(x['block_tokens']==y['block_tokens'] for p in pairs for x,y in zip(
                p['arms'][a]['refinements'],p['arms'][b]['refinements'])),
            none_decisions_equal=sum(x['block_type']==y['block_type'] for p in pairs for x,y in zip(
                p['arms'][a]['refinements'],p['arms'][b]['refinements'])),
            blocks=sum(len(p['arms'][a]['refinements']) for p in pairs))
    report['repeat_exact_pairs']={tag:sum(
        geometry_signature(p['arms'][tag]['refinements']) ==
        geometry_signature(first[p['uid']]['arms'][tag]['refinements'])
        for p in pairs) for tag in report['arms']}
    report['round1_exact_pairs']=sum(p['equality'].get('round1',True) for p in pairs)
    gate_path=args.output/'correctness_gate.json'
    capacity_passed=gate_path.exists() and json.loads(gate_path.read_text())['unchanged_decoded_outputs']
    report['adoption_gate_passed']=capacity_passed and len(rows)==100 and args.repeats>=2 and all(
        v['exact_pairs']==v['total_pairs'] and v['latency_reduction_paired_image_ci95'][0]>0
        for v in report['equivalence'].values()) and report['round1_exact_pairs']==len(pairs)
    report['timing_caveat']='Total is derived using archived R1' if kind=='frozen' else 'Total measured end to end'
    run.ref.atomic_json(args.output/f'{kind}_comparison.json',report)
    write_summary(args.output)
    print('SUMMARY',kind,json.dumps(report['equivalence']),flush=True)
    return report


def write_summary(output):
    lines=['# RefWave one-forward Round-2 test', '',
        'The existing two-pass implementation remains the default. No training or standalone port.',
        'Same checkpoint, images, prompts, predicted points, output head, and raw metric policy.',
        'First-repeat accuracy; timing averages include all repeats. NMS scores remain separate in JSON.', '']
    reports=[]
    for kind in ('frozen','end_to_end'):
        path=output/f'{kind}_comparison.json'
        if not path.exists():
            lines.extend([f'## {kind}', '', 'Pending.', ''])
            continue
        report=json.loads(path.read_text());reports.append(report)
        lines.extend([f'## {kind}', '',
            f"Images: {report['images']}; timing repeats: {report['repeats']}.",
            report['timing_caveat']+'.',
            '**Not a full Balanced-100 result.**' if report['images'] != 100 else 'Full frozen Balanced-100 coverage.',
            '', '| Execution | TP | FP | FN | Precision % | Recall % | F1 % |',
            '|---|---:|---:|---:|---:|---:|---:|'])
        for mode,wave in ARMS:
            tag=name(mode,wave);m=report['arms'][tag]['raw']['0.5']
            lines.append(f"| {tag} | {m['tp']} | {m['fp']} | {m['fn']} | "
                f"{100*m['precision']:.2f} | {100*m['recall']:.2f} | {100*m['f1']:.2f} |")
        lines.extend(['', '| Execution | R2 mean s | R2 p50 s | R2 p95 s | Total mean s | Total p50 s | Total p95 s | Peak MiB |',
            '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for mode,wave in ARMS:
            tag=name(mode,wave);m=report['arms'][tag]
            values=[m[f'{field}_{stat}'] for field in ('round2_seconds','latency_seconds')
                    for stat in ('mean','p50','p95')]
            lines.append(f'| {tag} | '+' | '.join(f'{v:.3f}' for v in values)+f" | {m['peak_gpu_mb']:.0f} |")
        lines.extend(['', '| Wave | Identical image/repeat pairs | Identical blocks | R2 speedup |',
            '|---:|---:|---:|---:|'])
        for wave,e in report['equivalence'].items():
            lines.append(f"| {wave} | {e['exact_pairs']}/{e['total_pairs']} | "
                f"{e['exact_blocks']}/{e['blocks']} | {e['round2_speedup']:.3f}x |")
        if kind == 'end_to_end':
            lines.extend(['',f"Identical Round-1 paired outputs: {report['round1_exact_pairs']}."])
        lines.extend(['',f"This phase's adoption gate: {report['adoption_gate_passed']}.", ''])
    passed=len(reports)==2 and all(r['adoption_gate_passed'] for r in reports)
    lines.extend(['## Adoption', '', f'Combined gate passed: {passed}.',
        'A completed timing test does not imply exact-output equivalence. Do not enable fusion by default if any geometry or None decision changes.',
        'Latency confidence intervals in the JSON are paired image bootstraps, not training-seed evidence or isolated-node timing.', ''])
    (output/'summary.md').write_text('\n'.join(lines))


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['gate','frozen','end_to_end','all'])
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    rows,frozen=prepare(args)
    worker=run.load_worker();decoders=make_decoders(worker)
    if args.mode in ('gate','all'):
        gate(worker,decoders,rows,frozen,args.output)
    for kind in ('frozen','end_to_end'):
        if args.mode in (kind,'all'):
            bench(worker,decoders,rows,frozen,args,kind)
    run.ref.atomic_json(args.output/f'{args.mode}.done.json',dict(completed=True,default_changed=False))


if __name__=='__main__':
    main()
