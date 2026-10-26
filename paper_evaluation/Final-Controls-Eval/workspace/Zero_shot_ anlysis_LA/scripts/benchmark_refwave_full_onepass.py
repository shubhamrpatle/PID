#!/usr/bin/env python3
"""Measure full-FT fused Round 2 without changing the reference evaluator."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import benchmark_refwave_fused_round2 as fused
from refwave.io import append_jsonl, read_jsonl_log

ref = fused.run.ref
full = fused.run.full
ablation = fused.run.ablation
SUITE = 'DOTAv2-Balanced100'
OUTPUT = ref.PROJECT / 'runs/refwave_full_onepass_balanced100_20260912'


def read(path):
    return json.loads(path.read_text())


def archived(config):
    directory = ablation.OUTPUT / SUITE / 'refwave_full' / config
    paths = sorted(directory.glob('shard*.jsonl'))
    assert paths and all(p.with_suffix('.done.json').exists() for p in paths)
    rows = [r for p in paths for r in ref.read_jsonl(p)]
    assert len(rows) == len({r['uid'] for r in rows}) == 100
    return {r['uid']: r for r in rows}


def summarize(output, rows, repeats, old):
    import numpy as np
    saved = ref.read_jsonl(output / 'predictions.jsonl')
    expected = {(r['uid'], w, rep) for r in rows for w in (1, 200) for rep in range(repeats)}
    assert len(saved) == len(expected)
    assert {(r['uid'], r['wave'], r['repeat']) for r in saved} == expected
    report = dict(records=len(rows), repeats=repeats, tuning='full', execution='fused_one_pass',
                  accuracy_policy='First repeat; all parsed boxes; no dedup, containment, NMS or maxDets.',
                  timing_policy='Aggregate boxes or TP divided by summed end-to-end inference seconds.',
                  default_changed=False, arms={}, historical={})
    for wave in (1, 200):
        measures = [r for r in saved if r['wave'] == wave]
        first = [r for r in measures if r['repeat'] == 0]
        metrics, per = full.score(rows, first)
        by_uid = {r['uid']: r for r in first}
        total_time = sum(r['latency_seconds'] for r in measures)
        times = [r['latency_seconds'] for r in measures]
        metrics.update(mean_seconds=float(np.mean(times)), p50_seconds=float(np.median(times)),
            p95_seconds=float(np.percentile(times, 95)),
            round1_mean_seconds=float(np.mean([r['round1_seconds'] for r in measures])),
            round2_mean_seconds=float(np.mean([r['round2_seconds'] for r in measures])),
            peak_gpu_mb=max(r['peak_gpu_mb'] for r in measures),
            bps=sum(len(r['predictions']) for r in measures)/total_time,
            repeat_prediction_equal=sum(r['predictions'] == by_uid[r['uid']]['predictions'] for r in measures),
            repeat_geometry_equal=sum(fused.geometry_signature(r['refinements']) ==
                fused.geometry_signature(by_uid[r['uid']]['refinements']) for r in measures),
            historical_round1_equal=sum(r['point_answers'] == old[wave][r['uid']]['point_answers'] for r in first),
            historical_predictions_equal=sum(r['predictions'] == old[wave][r['uid']]['predictions'] for r in first),
            historical_geometry_equal=sum(fused.geometry_signature(r['refinements']) ==
                fused.geometry_signature(old[wave][r['uid']]['refinements']) for r in first))
        tp_sum = sum(full.score(rows, [r for r in measures if r['repeat'] == rep])[0]['raw']['0.5']['tp']
                     for rep in range(repeats))
        metrics['cbps_iou0p5'] = tp_sum / total_time
        report['arms'][str(wave)] = metrics
        ref.jsonl(output / f'wave{wave}_per_image.jsonl', per)
    for method, cfg in [('standard', 'direct_b2048'), ('refwave_full', 'c18_p1_b2048'),
                        ('refwave_full', 'c18_p200_b2048')]:
        path = ablation.OUTPUT / SUITE / method / cfg / 'metrics.json'
        m = read(path)
        report['historical'][method + '/' + cfg] = dict(source=str(path), **m,
            bps=m['raw_boxes']/(m['records']*m['mean_seconds']),
            cbps_iou0p5=m['raw']['0.5']['tp']/(m['records']*m['mean_seconds']))
    ref.atomic_json(output / 'comparison.json', report)
    lines = ['# Full-FT one-pass Round-2 evaluation', '',
        'DOTAv2 Balanced-100, 9045 GT, native-6000; class-wave 18; discovery budget 2048 per class.',
        'Raw metrics without NMS, containment, deduplication or maxDets; same frozen checkpoint.',
        'TPI includes preprocessing, Round 1, Round 2 and parsing; excludes loading, disk I/O and scoring.',
        'BPS = total parsed boxes / total seconds. CBPS = total TP at IoU 0.5 / total seconds.',
        'The baseline and two-pass rows are archived timings, not a same-session controlled speed test.', '',
        '| Model | F1@0.5 | F1@0.75 | TPI s | BPS | CBPS |',
        '|---|---:|---:|---:|---:|---:|']
    for label, m in [*report['historical'].items(),
                     *[(f'RefWave full FT fused one-pass wave{w}', m) for w, m in report['arms'].items()]]:
        lines.append(f"| {label} | {100*m['raw']['0.5']['f1']:.2f} | {100*m['raw']['0.75']['f1']:.2f} | "
                     f"{m['mean_seconds']:.3f} | {m['bps']:.2f} | {m['cbps_iou0p5']:.2f} |")
    lines += ['', '## Checks', '']
    for wave, m in report['arms'].items():
        lines.append(f"Wave {wave}: Round-1 matches archive {m['historical_round1_equal']}/100; "
                     f"prediction matches {m['historical_predictions_equal']}/100; "
                     f"repeat predictions identical {m['repeat_prediction_equal']}/{len(rows)*repeats}.")
    lines += ['', 'No default change. Fused BF16 execution may change geometry or None decisions.',
              'Standard is LoRA, not a matched full-parameter-fine-tuned baseline.',
              'One training seed and a diagnostic subset; not full-benchmark or multi-seed evidence.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main():
    import torch
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    assert args.repeats > 0
    plan = read(ablation.OUTPUT / 'plan.json')
    manifest = Path(plan['manifests'][SUITE]['path'])
    assert ref.sha256_file(manifest) == plan['manifests'][SUITE]['sha256']
    rows = ref.read_jsonl(manifest)
    assert len(rows) == len({r['uid'] for r in rows}) == 100
    old = {w: archived(f'c18_p{w}_b2048') for w in (1, 200)}
    if args.summarize_only:
        summarize(args.output, rows, args.repeats, old)
        return
    identity = ablation.selected('refwave_full')
    assert identity == plan['models']['refwave_full']
    assert Path(identity['checkpoint']) == full.CHECKPOINT
    assert ref.sha256_file(full.CHECKPOINT) == identity['sha256']
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(fused.__file__), Path(ablation.__file__), Path(full.__file__),
               Path(ref.__file__), ref.ROOT/'zero_shot_analysis/fused_pivr_geometry.py',
               ref.ROOT/'zero_shot_analysis/pixel_pivr_wave.py']
    contract = dict(identity=identity, manifest=str(manifest), manifest_sha256=ref.sha256_file(manifest),
        records=100, repeats=args.repeats, image_token_limit=6000, class_wave=18, box_waves=[1, 200],
        discovery_budget=2048, execution='fused_one_pass', raw=True, nms=False, containment=False,
        deduplicate=False, default_changed=False, gpu=torch.cuda.get_device_name(0),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=torch.__version__,
        sources={str(p.resolve()):ref.sha256_file(p) for p in sources},
        timing='CUDA-synchronized end-to-end; excludes loading, warmup, disk I/O and scoring.',
        historical_ablation_directory=str(ablation.OUTPUT))
    cp = args.output / 'contract.json'
    if cp.exists() and read(cp) != contract:
        raise ValueError('Existing run contract differs; choose another output directory')
    ref.atomic_json(cp, contract)
    previous = read_jsonl_log(args.output / 'predictions.jsonl', repair_tail=True)
    done = {(p['uid'], p['wave'], p['repeat']) for p in previous}
    expected = {(r['uid'], w, rep) for r in rows for w in (1, 200) for rep in range(args.repeats)}
    assert len(done) == len(previous) and done <= expected
    worker = full.load_full_worker()
    decoders = fused.make_decoders(worker)
    decoders[1].round2_execution = 'fused_one_pass'
    for row in rows[:2]:
        with Image.open(row['image_path']) as f:
            image = f.convert('RGB')
        for wave in (1, 200):
            ablation.infer_one(worker, decoders, ref.inference_query(row), image, 'refwave_full',
                               dict(budget=2048, class_wave=18, box_wave=wave))
    started = time.time()
    for index, row in enumerate(rows):
        with Image.open(row['image_path']) as f:
            image = f.convert('RGB')
        assert image.size == (row['width'], row['height'])
        for rep in range(args.repeats):
            for wave in ((1, 200) if (index+rep)%2 == 0 else (200, 1)):
                if (row['uid'], wave, rep) in done:
                    continue
                result = ablation.infer_one(worker, decoders, ref.inference_query(row), image,
                    'refwave_full', dict(budget=2048, class_wave=18, box_wave=wave))
                r2 = result['round2']
                assert r2['round2_execution'] == 'fused_one_pass'
                assert r2['qwen_branch_suffix_forwards'] == r2['qwen_geometry_forwards'] == 0
                assert r2['qwen_total_forwards'] == r2['qwen_fused_prompt_geometry_forwards']
                result.update(wave=wave, repeat=rep, image_id=row['image_id'], dataset=row['dataset'])
                append_jsonl(args.output / 'predictions.jsonl', result)
                done.add((row['uid'], wave, rep))
                ref.atomic_json(args.output / 'status.json', dict(completed=len(done), expected=len(expected),
                    elapsed_seconds=time.time()-started, current_uid=row['uid'], all_done=False))
                print(f"image {index+1}/100 repeat {rep+1} wave {wave}: "
                      f"points={len(result['points'])} boxes={len(result['predictions'])} "
                      f"seconds={result['latency_seconds']:.3f} "
                      f"R1_archive_equal={result['point_answers']==old[wave][row['uid']]['point_answers']}", flush=True)
    del worker, decoders
    gc.collect()
    torch.cuda.empty_cache()
    summarize(args.output, rows, args.repeats, old)
    ref.atomic_json(args.output / 'all.done.json', dict(completed=True, records=100, repeats=args.repeats))
    ref.atomic_json(args.output / 'status.json', dict(completed=len(done), expected=len(expected), all_done=True))


if __name__ == '__main__':
    main()
