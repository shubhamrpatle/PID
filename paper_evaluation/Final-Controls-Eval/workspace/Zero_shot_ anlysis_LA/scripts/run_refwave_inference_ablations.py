#!/usr/bin/env python3
"""Inference-only ablations on frozen full-scale checkpoints and held-out manifests."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import evaluate_refwave_cluster_stage2 as ref
import evaluate_refwave_cluster_full_stage2 as full
from refwave.io import append_jsonl, read_jsonl_log
from run_locany_pointing_balanced100 import parse_locateanything_points, match_points_to_gt

OUTPUT = ref.PROJECT / 'runs/refwave_inference_ablations_20260911'
SUITES = ('DOTAv2-Balanced100', 'DIOR-RSVG-Unseen100')
METHODS = ('refwave_full', 'standard', 'refwave')
BUDGETS = (2048, 8192, 16384)


def configurations(method):
    if method == 'standard':
        return [dict(name=f'direct_b{b}', budget=b, class_wave=1, box_wave=1) for b in BUDGETS]
    result = [dict(name=f'c{c}_p{p}_b2048', budget=2048, class_wave=c, box_wave=p)
              for c, p in ((1, 1), (18, 1), (1, 200), (18, 200))]
    return result + [dict(name=f'c18_p200_b{b}', budget=b, class_wave=18, box_wave=200)
                     for b in BUDGETS[1:]]


def selected(method):
    if method == 'refwave_full':
        row = json.loads((full.OUTPUT / 'checkpoint.json').read_text())
    else:
        row = json.loads((ref.OUTPUT / 'checkpoints.json').read_text())['selections'][method]
    return {k: row[k] for k in ('checkpoint', 'sha256', 'step')}


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    identities = {m: selected(m) for m in METHODS}
    for ident in identities.values():
        if ref.sha256_file(Path(ident['checkpoint'])) != ident['sha256']:
            raise ValueError('Checkpoint hash differs from completed full-scale selection')
    sources = [Path(__file__), Path(ref.__file__), Path(full.__file__),
               Path(ref.detection.__file__), Path(ref.grounding.__file__),
               ref.PROJECT / 'scripts/run_locany_pointing_balanced100.py']
    for directory in (ref.ROOT / 'Pixel-PIVR/src/pixel_pivr', ref.ROOT / 'RefWave/src/refwave',
                      ref.ROOT / 'zero_shot_analysis', ref.BASE_MODEL):
        sources.extend(directory.glob('*.py'))
    contract = dict(models=identities, manifests={}, configurations={m: configurations(m) for m in METHODS},
        sources={str(p.resolve()): ref.sha256_file(p) for p in sources},
        image_token_limit=6000, training=False, nms_headline=False, warmup_images=2,
        raw_policy='Retain duplicate points and all parsed boxes; no containment or NMS in raw.',
        budget_policy='Per class discovery branch for RefWave; per query for direct LA. R2 adds one PBD6 per point.',
        caveats=['Single trained seed; no claim of across-seed reliability.',
                 'Standard is full-scale LoRA, not a matched full-FT baseline.',
                 'Other GPUs may be occupied; not an isolated-node latency measurement.',
                 'Outputs are diagnostic held-out subsets, not full benchmark scores.'])
    for suite in SUITES:
        source = ref.OUTPUT / 'assets' / f'{suite}.jsonl'
        contract['manifests'][suite] = dict(path=str(source), sha256=ref.sha256_file(source),
                                           records=len(ref.read_jsonl(source)))
    path = output / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Existing ablation plan differs; use a new output directory')
    for p in sources:
        destination = output / 'source_snapshot' / p.resolve().relative_to(ref.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(p, destination)
    ref.atomic_json(path, contract)
    return contract


def parsed_boxes(answer, query, size, expected=None):
    from zero_shot_analysis.locate_anything import parse_locateanything_labeled_boxes
    result = []
    for item in parse_locateanything_labeled_boxes(answer, *size):
        box = ref.detection.clip_hbox_to_image(item['hbox'], *size)
        if box is None:
            continue
        if query['task'] == 'detection':
            label = ref.detection.canonicalize_label(item['label'], query['aliases'], set(query['classes']))
            if label is None:
                continue
        else:
            label = query['references'][0]['label']
        result.append(dict(label=label, hbox=box, score=1.0))
    return result


def grammar_valid(answer, dimensions):
    """Strict whole-answer geometry grammar, allowing reference tags and EOS only."""
    text = re.sub(r'<ref>.*?</ref>|<\|im_end\|>|<\|endoftext\|>', '', answer, flags=re.DOTALL).strip()
    blocks = re.findall(r'<box>(.*?)</box>', text, re.DOTALL)
    if not blocks or re.sub(r'<box>.*?</box>', '', text, flags=re.DOTALL).strip():
        return False
    for body in blocks:
        if body in ('None', 'none'):
            continue
        coords = re.findall(r'<(\d+)>', body)
        if len(coords) != dimensions or any(int(v) > 1000 for v in coords):
            return False
        if re.sub(r'<\d+>', '', body).strip():
            return False
    return True


def infer_one(worker, decoders, query, image, method, config):
    import torch
    from pixel_pivr.class_wave import ClassPointBranch
    from zero_shot_analysis.pixel_pivr_wave import AddressedCrop
    from evaluate_pixel_pivr_sequential_wave_balanced100 import compact_round2_prompt, point_to_norm
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    seed = ref.stable_seed(0, query['suite'], query['uid'], 'generation')
    ref.grounding.set_generation_seed(seed)
    if method == 'standard':
        result = ref.infer_one(worker, None, query, image, 'standard', config['budget'])
        # The direct native generator returns text, not its stop-reason metadata.
        answer = result['answer']
        tokens = len(worker.tokenizer.encode(answer, add_special_tokens=False))
        result.update(grammar_valid_branches=int(grammar_valid(answer, 4)), generated_branches=1,
            generated_text_tokens=tokens,
            eos_branches=int(any(e in answer for e in ('<|im_end|>', '<|endoftext|>'))),
            budget_saturated_branches=int(tokens >= config['budget']),
            budget_measurement='Direct saturation inferred from returned-text token length; native stop reason unavailable.',
            point_duplicates=0, points=[])
        return result
    point_decoder, box_decoder = decoders
    point_decoder.max_wave_size = config['class_wave']
    branches = [ClassPointBranch(b['label'], b['question'],
        ref.stable_seed(0, query['suite'], query['uid'], b['label'], 'isolated-point'))
        for b in query['references']]
    prepared = point_decoder.prepare_image(image, branches)
    torch.cuda.synchronize()
    prep_seconds = time.perf_counter() - start
    prep_peak = torch.cuda.max_memory_allocated() / 1024**2
    answers, r1 = point_decoder.generate(image, branches, max_new_tokens=config['budget'],
        temperature=.7, top_p=.9, repetition_penalty=1.1, generation_mode='hybrid', prepared_image=prepared)
    # No box-center fallback or point deduplication: expose all generated addresses.
    points = [p for a in answers for p in parse_locateanything_points(a['answer'], *image.size, a['label'])
              if p['source'] == 'point']
    addresses = [AddressedCrop(str(i), p['label'], None,
        compact_round2_prompt(p['label'], p['point'], *image.size),
        point_normalized=point_to_norm(p['point'], *image.size)) for i, p in enumerate(points)]
    torch.cuda.synchronize()
    round1_seconds = time.perf_counter() - start
    round1_peak = torch.cuda.max_memory_allocated() / 1024**2
    refinements, r2 = box_decoder.decode_prepared_image(prepared, addresses,
        requested_wave_size=config['box_wave'], allow_none=True)
    assert len(refinements) == len(points)
    predictions, contained, dedup_contained, seen = [], [], [], set()
    for point, result in zip(points, refinements):
        key = (point['label'], *point['point'])
        candidates = parsed_boxes(result['answer'], query, image.size)
        predictions.extend(candidates)
        for candidate in candidates:
            x1, y1, x2, y2 = candidate['hbox']
            x, y = point['point']
            if candidate['label'] == point['label'] and x1 <= x <= x2 and y1 <= y <= y2:
                contained.append(candidate)
                if key not in seen:
                    dedup_contained.append(candidate)
        seen.add(key)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak = max(prep_peak, round1_peak, torch.cuda.max_memory_allocated() / 1024**2,
               r1.get('peak_gpu_mb', 0), r2['peak_gpu_mb'])
    assert r1['global_moonvit_encodes'] == r2['global_moonvit_encodes'] == 0
    assert r1['qwen_global_prefix_forwards'] == r2['qwen_global_prefix_forwards'] == 0
    result = dict(uid=query['uid'], suite=query['suite'], task=query['task'], predictions=predictions,
        contained_predictions=contained, dedup_contained_predictions=dedup_contained,
        points=points, point_answers=answers, refinements=refinements,
        round1=r1, round2=r2, preparation_seconds=prep_seconds,
        round1_seconds=round1_seconds, round2_seconds=seconds-round1_seconds,
        latency_seconds=seconds, peak_gpu_mb=peak, image_encodes=1, prefix_forwards=1,
        point_duplicates=len(points)-len(seen),
        generated_text_tokens=sum(a['tokens_generated'] for a in answers),
        geometry_tokens=6*len(refinements), generated_branches=len(answers),
        grammar_valid_branches=sum(grammar_valid(a['answer'], 2) for a in answers),
        eos_branches=sum(any(e in a['answer'] for e in ('<|im_end|>', '<|endoftext|>')) for a in answers),
        budget_saturated_branches=sum(a.get('fallback_reason') == 'max_new_tokens' for a in answers),
        generation_seed=seed)
    del prepared
    return result


def path_for(output, suite, method, config, shard, world):
    return output / suite / method / config['name'] / f'shard{shard:02d}of{world:02d}.jsonl'


def worker(args):
    import torch
    from PIL import Image
    plan = json.loads((args.output / 'plan.json').read_text())
    for source, sha in plan['sources'].items():
        if ref.sha256_file(Path(source)) != sha:
            raise ValueError('Code changed after preparation: ' + source)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    for method in args.methods:
        model = None
        for suite in SUITES:
            manifest = plan['manifests'][suite]
            if ref.sha256_file(Path(manifest['path'])) != manifest['sha256']:
                raise ValueError('Frozen evaluation manifest changed')
            rows = ref.read_jsonl(Path(manifest['path']))[args.shard_index::args.shard_count]
            if args.limit:
                rows = rows[:args.limit]
            configs = configurations(method)
            done = {}
            for cfg in configs:
                path = path_for(args.output, suite, method, cfg, args.shard_index, args.shard_count)
                path.parent.mkdir(parents=True, exist_ok=True)
                contract = dict(plan_sha256=ref.sha256_file(args.output / 'plan.json'), config=cfg,
                    method=method, suite=suite, identity=plan['models'][method], limit=args.limit,
                    shard_index=args.shard_index, shard_count=args.shard_count,
                    gpu=torch.cuda.get_device_name(0), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    timing='CUDA synchronized preprocessing + decoding + parsing; excludes disk IO/loading/scoring.')
                cp = path.with_suffix('.contract.json')
                if cp.exists() and json.loads(cp.read_text()) != contract:
                    raise ValueError('Output contract changed')
                ref.atomic_json(cp, contract)
                previous = read_jsonl_log(path, repair_tail=True) if path.exists() else []
                done[cfg['name']] = {p['uid'] for p in previous}
                if len(done[cfg['name']]) != len(previous) or not done[cfg['name']] <= {r['uid'] for r in rows}:
                    raise ValueError('Duplicate/foreign saved predictions')
            if all(len(d) == len(rows) for d in done.values()):
                continue
            if model is None:
                model = full.load_full_worker() if method == 'refwave_full' else ref.load_worker(method)
                # This historical ablation measures the original two-forward route.
                decoders = None if method == 'standard' else ref.make_decoders(
                    model, optimized=True, round2_execution='two_pass')
            for row in rows[:2]:
                with Image.open(row['image_path']) as im:
                    image = im.convert('RGB')
                for cfg in configs:
                    infer_one(model, decoders, ref.inference_query(row), image, method, cfg)
            for i, row in enumerate(rows):
                with Image.open(row['image_path']) as im:
                    image = im.convert('RGB')
                if image.size != (int(row['width']), int(row['height'])):
                    raise ValueError('Image dimensions differ from frozen annotations')
                # Rotate execution order, while keeping the same per-branch seeds.
                order = configs[i % len(configs):] + configs[:i % len(configs)]
                for cfg in order:
                    if row['uid'] in done[cfg['name']]:
                        continue
                    result = infer_one(model, decoders, ref.inference_query(row), image, method, cfg)
                    result.update(method=method, image_id=row['image_id'], dataset=row['dataset'])
                    if row['task'] == 'grounding':
                        result.update(sample_id=row['sample_id'], query=row['query'])
                    append_jsonl(path_for(args.output, suite, method, cfg, args.shard_index, args.shard_count), result)
                    print(f'{method} {suite} {cfg["name"]} {i+1}/{len(rows)} '
                          f'boxes={len(result["predictions"])} seconds={result["latency_seconds"]:.3f}', flush=True)
            for cfg in configs:
                path = path_for(args.output, suite, method, cfg, args.shard_index, args.shard_count)
                ref.atomic_json(path.with_suffix('.done.json'), dict(records=len(rows)))
        if model is not None:
            del model, decoders
            gc.collect()
            torch.cuda.empty_cache()
    ref.atomic_json(args.output / f'worker{args.shard_index}.done.json', dict(methods=args.methods, limit=args.limit))


def signature(row):
    return Counter((p['label'], *p['hbox']) for p in row['predictions'])


def point_diagnostics(rows, predictions):
    counts = Counter()
    for row, pred in zip(rows, predictions):
        if row['task'] != 'detection' or not pred.get('point_answers'):
            continue
        for spec in row['class_specs']:
            label = spec['canonical']
            gt = [g['hbox'] for g in row['gt'] if g['label'] == label]
            points = [p for p in pred['points'] if p['label'] == label]
            match = match_points_to_gt(gt, points)
            counts.update({k: match[k] for k in ('tp', 'fp', 'fn')})
            counts['present_classes'] += bool(gt)
            counts['missing_present_classes'] += bool(gt) and not points
    return dict(counts, recall=counts['tp']/max(1, counts['tp']+counts['fn']))


def merge(args, partial=False):
    import numpy as np
    all_metrics = {}
    cache = {}
    for suite in SUITES:
        rows = ref.read_jsonl(ref.OUTPUT / 'assets' / f'{suite}.jsonl')
        if args.limit:
            ids = {r['uid'] for i in range(args.shard_count) for r in rows[i::args.shard_count][:args.limit]}
            rows = [r for r in rows if r['uid'] in ids]
        for method in args.methods:
            for cfg in configurations(method):
                paths = [path_for(args.output, suite, method, cfg, i, args.shard_count) for i in range(args.shard_count)]
                if not all(p.with_suffix('.done.json').exists() for p in paths):
                    if partial:
                        continue
                    raise RuntimeError(f'Incomplete {suite}/{method}/{cfg["name"]}')
                records = [r for p in paths for r in ref.read_jsonl(p)]
                by_id = {r['uid']: r for r in records}
                if len(records) != len(by_id) or set(by_id) != {r['uid'] for r in rows}:
                    raise ValueError('Missing/duplicated evaluation records')
                records = [by_id[r['uid']] for r in rows]
                metrics, per = full.score(rows, records)
                if method != 'standard':
                    for tag, field in [('containment', 'contained_predictions'), ('dedup_containment', 'dedup_contained_predictions')]:
                        filtered = [dict(r, predictions=r[field]) for r in records]
                        metrics[tag], _ = full.score(rows, filtered)
                metrics['diagnostics'] = dict(
                    generated_branches=sum(r['generated_branches'] for r in records),
                    grammar_valid_branches=sum(r['grammar_valid_branches'] for r in records),
                    budget_saturated_branches=sum(r['budget_saturated_branches'] for r in records),
                    eos_branches=sum(r['eos_branches'] for r in records),
                    generated_text_tokens=sum(r['generated_text_tokens'] for r in records),
                    geometry_tokens=sum(r.get('geometry_tokens', 0) for r in records),
                    point_duplicates=sum(r['point_duplicates'] for r in records),
                    point_counts=point_diagnostics(rows, records),
                    oom_retries=sum(r.get('round2', {}).get('oom_retries', 0) for r in records))
                if method != 'standard':
                    metrics['round1_mean_seconds'] = float(np.mean([r['round1_seconds'] for r in records]))
                    metrics['round2_mean_seconds'] = float(np.mean([r['round2_seconds'] for r in records]))
                metrics['predicted_boxes_per_second'] = sum(len(r['predictions']) for r in records)/sum(r['latency_seconds'] for r in records)
                if suite == 'DOTAv2-Balanced100':
                    metrics['by_gt_density'] = {}
                    for low, high, name in ((0,19,'0-19'), (20,99,'20-99'), (100,499,'100-499'), (500,float('inf'),'500+')):
                        ids = [i for i, row in enumerate(rows) if low <= len(row['gt']) <= high]
                        if not ids:
                            continue
                        totals = {k: sum(per[i][k] for i in ids) for k in ('tp','fp','fn')}
                        elapsed = sum(records[i]['latency_seconds'] for i in ids)
                        metrics['by_gt_density'][name] = dict(records=len(ids), **totals,
                            f1=2*totals['tp']/max(1,2*totals['tp']+totals['fp']+totals['fn']),
                            mean_seconds=elapsed/len(ids),
                            predicted_boxes_per_second=sum(len(records[i]['predictions']) for i in ids)/elapsed)
                directory = paths[0].parent
                ref.atomic_json(directory / 'metrics.json', metrics)
                ref.jsonl(directory / 'per_record_metrics.jsonl', per)
                key = f'{suite}/{method}/{cfg["name"]}'
                all_metrics[key] = metrics
                cache[(suite, method, cfg['name'])] = records
    equivalence = {}
    for suite in SUITES:
        for method in args.methods:
            reference = cache.get((suite, method, 'c1_p1_b2048'))
            if reference is None:
                continue
            for name in ('c18_p1_b2048', 'c1_p200_b2048', 'c18_p200_b2048'):
                comparison = cache.get((suite, method, name))
                if comparison is None:
                    continue
                equivalence[f'{suite}/{method}/{name}'] = dict(records=len(reference),
                    exact_point_answers=sum([a['answer'] for a in x['point_answers']] == [a['answer'] for a in y['point_answers']]
                                            for x, y in zip(reference, comparison)),
                    exact_geometry_blocks=sum([a['block_tokens'] for a in x['refinements']] == [a['block_tokens'] for a in y['refinements']]
                                              for x, y in zip(reference, comparison)),
                    exact_prediction_multisets=sum(signature(x) == signature(y) for x, y in zip(reference, comparison)))
    ref.atomic_json(args.output / 'comparison.json', dict(metrics=all_metrics, equivalence=equivalence, partial=partial))
    lines = ['# Inference-only full-scale checkpoint ablations', '',
        'No 10K pilots. Standard = cluster LoRA; refwave = cluster LoRA; refwave_full = cluster full FT.',
        'Raw retains duplicate addresses and boxes. NMS and containment results are separate in comparison.json.',
        'Same-number token limits differ in scope: one direct query versus each class point branch plus PBD6 refinement.',
        'Single training seed. Subset diagnostics; no training-seed reliability or full-FT matched baseline claim.', '',
        '| Set / method / execution | Records | Raw P or Acc@0.5 | Raw R or Acc@0.7 | Raw F1 or mIoU | Mean s | p95 s | Peak MiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, m in all_metrics.items():
        values = m['raw']['0.5'] if 'raw' in m else m['grounding']
        names = ('precision','recall','f1') if 'raw' in m else ('first_acc_0_5','first_acc_0_7','first_mean_iou')
        a,b,c = [100*values[n] for n in names]
        lines.append(f'| {key} | {m["records"]} | {a:.2f} | {b:.2f} | {c:.2f} | '
                     f'{m["mean_seconds"]:.3f} | {m["p95_seconds"]:.3f} | {m["peak_gpu_mb"]:.0f} |')
    lines += ['', 'Exact sequential/wave answer, geometry-block and prediction equivalence is recorded in comparison.json.',
              'Differences must be reported; numerical/sampling equivalence is not assumed.']
    (args.output / 'summary.md').write_text('\n'.join(lines)+'\n')


def queue(args):
    import fcntl
    if not (args.output / 'plan.json').exists():
        prepare(args.output)
    lock = (args.output / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    workers = []
    logs = args.output / 'logs'
    logs.mkdir(exist_ok=True)
    for i, gpu in enumerate(args.gpus):
        # No other user's compute process is stopped or shared deliberately.
        while True:
            used = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
            if int(used.strip()) < 500:
                break
            print(f'Waiting for GPU {gpu}', flush=True)
            time.sleep(5)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--output', str(args.output),
                   '--shard-index', str(i), '--shard-count', str(len(args.gpus)), '--methods', *args.methods]
        if args.limit:
            command += ['--limit', str(args.limit)]
        log = (logs / f'gpu{gpu}.log').open('a')
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        workers.append(dict(gpu=gpu, pid=process.pid, process=process))
    ref.atomic_json(args.output / 'running.json', dict(workers=[{k:v for k,v in w.items() if k!='process'} for w in workers]))
    while any(w['process'].poll() is None for w in workers):
        time.sleep(15)
    failures = {w['gpu']: w['process'].returncode for w in workers if w['process'].returncode}
    args.shard_count = len(args.gpus)
    merge(args, partial=bool(failures))
    ref.atomic_json(args.output / 'status.json', dict(completed=not failures, failures=failures))
    if failures:
        raise RuntimeError(f'Workers failed: {failures}; completed outputs preserved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'worker', 'queue', 'merge'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--gpus', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output)
    else:
        globals()[args.action](args)


if __name__ == '__main__':
    main()
