#!/usr/bin/env python3
"""Paired PID detection inference and offline metrics for two extent schedules."""
import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import time

from pid_common import (HERE, MODES, DATASETS, load_plan, selected_rows, shard_folder,
                        record_path, order_for, check_pair, check_software, finished,
                        atomic, freeze, digest, sha, read, require, query_only)


class PairedBackend:
    def __init__(self, plan, backend=None):
        from models import Backend
        from sequential_extent import SequentialExtentRoute
        self.backend = backend or Backend('pid', plan)
        point, packed = self.backend.decoders
        self.routes = {'packed': (point, packed),
                       'sequential': (point, SequentialExtentRoute(packed.decoder))}

    def infer(self, mode, query, image):
        self.backend.decoders = self.routes[mode]
        result = self.backend.infer(query, image, 'boxes')
        result.update(execution_mode=mode, protocol='pid_packed_all_v1' if mode == 'packed'
                      else 'pid_sequential_extent_v1')
        return result


def timed_one(model, mode, query, image, plan_hash):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        result = model.infer(mode, query, image)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    inner = result['latency_seconds']
    components = dict(image_and_prefix=result['preparation_seconds'],
                      discovery_and_address_construction=result['round1_seconds'] - result['preparation_seconds'],
                      extent_and_legacy_parsing=result['round2_seconds'],
                      native_parser_and_outer_overhead=elapsed - inner)
    require(math.isfinite(elapsed) and elapsed > 0, 'Invalid outer inference timer')
    require(all(math.isfinite(v) and v >= -1e-6 for v in components.values()), 'Invalid component timer')
    require(abs(sum(components.values()) - elapsed) < 1e-6, 'Component times double count work')
    result.update(uid=query['uid'], suite=query['suite'], query=query, plan_sha256=plan_hash,
                  latency_seconds=elapsed, inner_pipeline_seconds=inner, component_seconds=components,
                  peak_gpu_mb=max(result['peak_gpu_mb'], torch.cuda.max_memory_allocated() / 1024**2))
    return result


def pair_one(model, row, image, index, plan_hash, timed=timed_one):
    query = query_only(row)
    order = order_for(index)
    results = {mode: timed(model, mode, query, image, plan_hash) for mode in order}
    pair = dict(uid=row['uid'], suite=row['suite'], plan_sha256=plan_hash,
                order=list(order), modes=results)
    check_pair(pair, row, plan_hash)
    return pair


def worker(args):
    from common import safe_cache
    safe_cache()
    import torch
    from PIL import Image
    from models import software
    plan, data = load_plan(args.output)
    count = 1 if args.smoke else plan['shards']
    require(0 <= args.shard < count, 'Invalid shard index')
    folder = shard_folder(args.output, args.dataset, args.shard, count, args.smoke)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if finished(args.output, args.dataset, args.shard, count, args.smoke, plan, data):
            return
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        actual = software()
        check_software(actual, plan['expected_software'])
        freeze(args.output / 'software.json', actual)
        for filename, identity in plan['models']['pid']['files'].items():
            stat = Path(filename).stat()
            require((stat.st_size, stat.st_mtime_ns) == (identity['bytes'], identity['mtime_ns']),
                    'Model files changed after one-time preparation: ' + filename)
        expected = selected_rows(data, args.dataset, args.shard, count, args.smoke)
        contract = dict(plan_sha256=digest(plan), dataset=args.dataset, shard=args.shard,
                        shards=count, smoke=args.smoke, uids=[r['uid'] for r in expected])
        freeze(folder / 'contract.json', contract)
        allowed = {record_path(folder, i).name for i in range(len(expected))}
        require({p.name for p in folder.glob('record*.json')} <= allowed, 'Foreign pair records in shard')
        missing = []
        for index, row in enumerate(expected):
            path = record_path(folder, index)
            if path.exists():
                check_pair(read(path), row, digest(plan))
            else:
                missing.append((index, row))
        if missing:
            model = PairedBackend(plan)

            def image_for(row):
                path = Path(row['image_path'])
                require(sha(path) == plan['datasets'][args.dataset]['images'][str(path)]['sha256'], 'Image changed')
                with Image.open(path) as im:
                    image = im.convert('RGB')
                require(image.size == (row['width'], row['height']), 'Image resolution changed')
                return image

            for index, row in enumerate(expected[:plan['warmup_records']]):
                pair_one(model, row, image_for(row), index, digest(plan))
            for index, row in missing:
                pair = pair_one(model, row, image_for(row), index, digest(plan))
                pair['gpu_uuid'] = os.environ.get('CUDA_VISIBLE_DEVICES')
                atomic(record_path(folder, index), pair)
                done_count = len(list(folder.glob('record*.json')))
                atomic(folder / 'progress.json', dict(records=done_count, expected=len(expected),
                    uid=row['uid'], times={m: pair['modes'][m]['latency_seconds'] for m in MODES}))
                print(args.dataset, f'{done_count}/{len(expected)}', 'uid', row['uid'],
                      'order', pair['order'], 'points', len(pair['modes']['packed']['points']),
                      'seconds', {m: round(pair['modes'][m]['latency_seconds'], 3) for m in MODES}, flush=True)
        files = {record_path(folder, i).name: sha(record_path(folder, i)) for i in range(len(expected))}
        atomic(folder / 'done.json', dict(complete=True, plan_sha256=digest(plan),
               uids=[r['uid'] for r in expected], files=files))


def metric_record(result, per_image, predictions, mode, dataset, plan_hash, smoke):
    import numpy as np
    times = [p['latency_seconds'] for p in predictions]
    require(all(math.isfinite(t) and t > 0 for t in times), 'Invalid saved inference time')
    total = sum(times)
    outputs = sum(len(p['predictions']) for p in predictions)
    valid = outputs - result['invalid_geometry_outputs']
    tp = sum(p['scores']['0.5']['class_aware']['tp'] for p in per_image)
    require(tp == result['scores']['class_aware']['0.5']['micro']['tp'], 'TP total mismatch')
    result.update(complete=True, model='pid_' + mode, task='boxes', dataset=dataset,
        smoke=smoke, plan_sha256=plan_hash, query_scope='GT-present categories',
        records=len(predictions), total_inference_seconds=total,
        seconds_per_complete_image=total / len(predictions),
        p50_seconds=float(np.percentile(times, 50)), p95_seconds=float(np.percentile(times, 95)),
        outputs=outputs, valid_geometry_outputs=valid, outputs_per_second=outputs / total,
        valid_geometry_outputs_per_second=valid / total, correct_outputs_per_second=tp / total,
        images_per_second=len(predictions) / total,
        generated_tokens=sum(p['generated_tokens'] for p in predictions),
        generated_discovery_tokens=sum(p['generated_text_tokens'] for p in predictions),
        generated_geometry_tokens=sum(p['geometry_tokens'] for p in predictions),
        saturated_queries=sum(p['saturated_queries'] for p in predictions),
        peak_gpu_mb=max(p['peak_gpu_mb'] for p in predictions),
        mean_component_seconds={k: statistics.mean(p['component_seconds'][k] for p in predictions)
                                for k in predictions[0]['component_seconds']},
        total_extent_forwards=sum(p['round2']['actual_round2_forwards'] for p in predictions))
    return result


def merge(args):
    from native_parser import reparse
    from scoring import score, load_upstream
    plan, data = load_plan(args.output)
    scorer = load_upstream()
    count = 1 if args.smoke else plan['shards']
    summary, comparisons = {}, {}
    for ds in DATASETS:
        pairs, hashes = {}, {}
        for shard in range(count):
            require(finished(args.output, ds, shard, count, args.smoke, plan, data), f'Incomplete {ds} shard {shard}')
            folder = shard_folder(args.output, ds, shard, count, args.smoke)
            for i, row in enumerate(selected_rows(data, ds, shard, count, args.smoke)):
                path = record_path(folder, i)
                pair = read(path)
                check_pair(pair, row, digest(plan))
                require(row['uid'] not in pairs, 'Duplicate image in paired results')
                pairs[row['uid']] = pair
                hashes[str(path.relative_to(args.output))] = sha(path)
        expected = selected_rows(data, ds, 0, 1, args.smoke)
        require(set(pairs) == {r['uid'] for r in expected}, 'Missing/extra benchmark image')
        ordered = [pairs[r['uid']] for r in expected]
        comparisons[ds] = dict(records=len(ordered), discovery_answers_equal=True,
            discovery_points_equal=True, first_mode_counts=dict(Counter(p['order'][0] for p in ordered)),
            equal_extent_answers=sum([a['answer'] for a in p['modes']['packed']['refinements']] ==
                                     [a['answer'] for a in p['modes']['sequential']['refinements']] for p in ordered),
            equal_parsed_predictions=sum(p['modes']['packed']['predictions'] == p['modes']['sequential']['predictions']
                                         for p in ordered))
        for mode in MODES:
            predictions = [p['modes'][mode] for p in ordered]
            replay = [reparse('pid', r, o, 'boxes', True) for r, o in zip(expected, predictions)]
            require(all(a['predictions'] == b['predictions'] for a, b in zip(replay, predictions)), 'Native-parser replay differs')
            result, per = score(expected, predictions, 'boxes', upstream=scorer)
            strict = [reparse('pid', r, o, 'boxes', False) for r, o in zip(expected, predictions)]
            result['strict_sensitivity'], _ = score(expected, strict, 'boxes', upstream=scorer)
            result = metric_record(result, per, predictions, mode, ds, digest(plan), args.smoke)
            result['pair_file_sha256'] = hashes
            out = args.output / ('smoke_metrics' if args.smoke else 'metrics') / ('pid_' + mode) / ds
            atomic(out / 'metrics.json', result)
            atomic(out / 'per_image.json', per)
            summary[f'pid_{mode}/boxes/{ds}'] = result
            print('SCORED', mode, ds, len(expected), 'images', flush=True)
    atomic(args.output / ('smoke_summary.json' if args.smoke else 'summary.json'), summary)
    atomic(args.output / ('smoke_pair_agreement.json' if args.smoke else 'pair_agreement.json'), comparisons)
    render(args.output, summary, comparisons, args.smoke)


def render(output, summary, comparisons, smoke):
    lines = ['# PID GT-Present-Category Detection: Paired Extent Execution', '',
             'Smoke check only; not benchmark results.' if smoke else 'Full frozen manifests. Same checkpoint, GT-present category names and scoring as the baseline protocol.', '',
             'TPI: mean seconds/image. BPS: valid returned boxes / summed inference seconds. '
             'CBPS: class-aware TP@0.50 / summed inference seconds. These are per-worker rates, not four-GPU wall-clock throughput.', '']
    for ds in DATASETS:
        lines += ['## ' + ds, '', '| Mode | F1@.50 | F1@.75 | mF1 | TPI | p50 | p95 | BPS | CBPS | Peak GB |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for mode in MODES:
            r = summary[f'pid_{mode}/boxes/{ds}']
            s = r['scores']['class_aware']
            lines.append(f"| {mode} | {100*s['0.5']['f1']:.2f} | {100*s['0.75']['f1']:.2f} | "
                f"{100*s['mean_f1_50_95']:.2f} | {r['seconds_per_complete_image']:.3f} | "
                f"{r['p50_seconds']:.3f} | {r['p95_seconds']:.3f} | {r['valid_geometry_outputs_per_second']:.2f} | "
                f"{r['correct_outputs_per_second']:.2f} | {r['peak_gpu_mb']/1024:.2f} |")
        lines += ['', f"Verified identical discovery on {comparisons[ds]['records']} images; "
                  f"identical parsed extent outputs on {comparisons[ds]['equal_parsed_predictions']}. "
                  'Packed and sequential BF16 geometry need not be bitwise identical.', '']
    lines += ['No NMS, oracle points, count caps or silent OOM fallback. Discovery is concurrent in both modes. '
              'Every mode independently performs image preparation, shared-prefix construction, discovery and extent localization.', '',
              'Timers include preprocessing and parsing, excluding checkpoint loading, disk image loading, warmup, scoring and queue wait. '
              'Round-1 timers include preparation; component breakdown subtracts preparation once.', '',
              'Same hardware/software does not eliminate shared-server contention. Earlier baselines were recorded in separate runs.']
    (output / ('SMOKE.md' if smoke else 'RESULTS.md')).write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('worker', 'merge'))
    p.add_argument('--output', type=Path, default=HERE / 'results')
    p.add_argument('--dataset', choices=list(DATASETS))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    args.output = args.output.resolve()
    globals()[args.mode](args)


if __name__ == '__main__':
    main()
