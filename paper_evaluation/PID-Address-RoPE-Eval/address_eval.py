#!/usr/bin/env python3
"""Fixed-checkpoint predicted/annotation-address evaluation, with a stable RoPE cache."""
import argparse
from collections import Counter
import fcntl
import hashlib
import math
import os
from pathlib import Path
import time

from rerun_common import (HERE, DEFAULT_OUTPUT, MODES, DATASETS, paired, require, load_plan,
    atomic, freeze, read, sha, digest, query_only, folder, record_path, check_pair, finished, gate_done)
from models import Backend, software
from native_parser import la_native, reparse
from scoring import score, load_upstream
from paired_eval import metric_record


def annotation_points(row, normalize):
    """Preserve source extents; quantize each visible-intersection center exactly once."""
    width, height = row['width'], row['height']
    require(width > 0 and height > 0, 'Invalid image resolution')
    points, annotations = [], []
    for spec in query_only(row)['class_specs']:
        for index, obj in enumerate(row['gt']):
            if obj['label'] != spec['canonical']:
                continue
            box = obj['hbox']
            require(len(box) == 4 and all(math.isfinite(v) for v in box), 'Invalid GT coordinates')
            x1, y1, x2, y2 = box
            require(x1 < x2 and y1 < y2, 'Nonpositive GT extent')
            visible = [max(0., x1), max(0., y1), min(width, x2), min(height, y2)]
            a, b, c, d = visible
            require(a < c and b < d, 'GT has no visible intersection')
            u, v = normalize(((a + c) / 2, (b + d) / 2), width, height)
            xy = [u * width / 1000., v * height / 1000.]
            require(x1 <= xy[0] <= x2 and y1 <= xy[1] <= y2, 'Quantized point leaves source GT')
            points.append(dict(label=obj['label'], point=xy, source='point'))
            annotations.append(dict(gt_index=index, label=obj['label'], normalized_point=[u, v],
                visible_extent_clipped=visible != box, inside_source_gt=True))
    require(len(points) == len(row['gt']), 'Oracle failed to retain every annotation')
    counts = Counter((p['label'], *p['point']) for p in points)
    return points, dict(addresses=len(points), duplicate_addresses=sum(v - 1 for v in counts.values()),
        clipped_visible_centres=sum(a['visible_extent_clipped'] for a in annotations), annotations=annotations)


class AddressBackend:
    def __init__(self, plan):
        from stable_rope import install_fp32_rope_cache
        self.backend = Backend('pid', plan)
        # The original loader has already restored weights and converted dtype.
        self.fixed_modules = install_fp32_rope_cache(self.backend.model.model)
        print('FP32 ROTARY CACHE INSTALLED', self.fixed_modules, 'modules before any prefill', flush=True)

    def predicted(self, query, image):
        return self.backend.infer(query, image, 'boxes')

    def oracle_sequential(self, query, image, points):
        from sequential_extent import SequentialExtentRoute
        original = self.backend.decoders
        try:
            self.backend.decoders = (original[0], SequentialExtentRoute(original[1].decoder))
            return self.oracle(query, image, points)
        finally:
            self.backend.decoders = original

    def oracle(self, query, image, points):
        import torch
        from pixel_pivr.class_wave import ClassPointBranch
        from zero_shot_analysis.pixel_pivr_wave import AddressedCrop
        from evaluate_pixel_pivr_sequential_wave_balanced100 import compact_round2_prompt, point_to_norm
        b = self.backend
        reference = b.pid.reference
        branches = [ClassPointBranch(s['canonical'],
            f"Point to all instances that match the following categories: {s['prompt']}.",
            reference.stable_seed(0, query['suite'], query['uid'], s['canonical'], 'isolated-point'))
            for s in query['class_specs']]
        reference.grounding.set_generation_seed(reference.stable_seed(0, query['suite'], query['uid'], 'generation'))
        point_decoder, box_decoder = b.decoders
        point_decoder.max_wave_size = len(branches)
        start = time.perf_counter()
        prepared = point_decoder.prepare_image(image, branches)
        torch.cuda.synchronize()
        preparation = time.perf_counter() - start
        prep_peak = torch.cuda.max_memory_allocated() / 1024**2
        requests = [AddressedCrop(str(i), p['label'], None,
            compact_round2_prompt(p['label'], p['point'], *image.size),
            point_normalized=point_to_norm(p['point'], *image.size)) for i, p in enumerate(points)]
        answers, stats = box_decoder.decode_prepared_image(prepared, requests, requested_wave_size=1, allow_none=True)
        parsed = [la_native(a['answer'], query['class_specs'], image.size, 'boxes', a['label']) for a in answers]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        require(stats['global_moonvit_encodes'] == stats['qwen_global_prefix_forwards'] == 0,
                'Oracle extent rebuilt the shared image prefix')
        return dict(points=points, refinements=answers, predictions=[p for group, _ in parsed for p in group],
            invalid_items=sum(len(e) for _, e in parsed), round2=stats,
            preparation_seconds=preparation, round1_seconds=preparation, round2_seconds=elapsed - preparation,
            latency_seconds=elapsed, peak_gpu_mb=max(prep_peak, stats['peak_gpu_mb']),
            image_encodes=1, prefix_forwards=1, round1_skipped=True,
            generated_text_tokens=0, geometry_tokens=6 * len(points), generated_tokens=6 * len(points),
            saturated_queries=0, shared_prefix_tokens=len(prepared.prefix))


def timed_one(model, mode, row, image):
    import torch
    from evaluate_pixel_pivr_sequential_wave_balanced100 import point_to_norm
    query = query_only(row)
    points, oracle_audit = annotation_points(row, point_to_norm) if mode == 'gt_address' else (None, None)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        result = model.predicted(query, image) if mode == 'predicted' else model.oracle(query, image, points)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    components = dict(image_and_prefix=result['preparation_seconds'],
        discovery_and_address_construction=result['round1_seconds'] - result['preparation_seconds'],
        extent_and_legacy_parsing=result['round2_seconds'],
        native_parser_and_outer_overhead=elapsed - result['latency_seconds'])
    require(elapsed > 0 and all(math.isfinite(v) and v >= -1e-6 for v in components.values()), 'Invalid timing')
    require(abs(sum(components.values()) - elapsed) < 1e-6, 'Overlapping component timing')
    result.update(uid=row['uid'], suite=row['suite'], query=query, address_source=mode,
        rope_fix_modules=model.fixed_modules, latency_seconds=elapsed, component_seconds=components,
        peak_gpu_mb=max(result['peak_gpu_mb'], torch.cuda.max_memory_allocated() / 1024**2))
    if oracle_audit is not None:
        result['oracle_audit'] = oracle_audit
    return result


def pair_one(model, row, image, index, plan, timed=timed_one):
    order = MODES if index % 2 == 0 else tuple(reversed(MODES))
    result = dict(uid=row['uid'], suite=row['suite'], order=list(order), plan_sha256=digest(plan),
        modes={mode: timed(model, mode, row, image) for mode in order})
    check_pair(result, row, plan)
    return result


def image_for(row, plan):
    from PIL import Image
    path = Path(row['image_path'])
    require(sha(path) == plan['datasets'][row['suite']]['images'][str(path)]['sha256'], 'Image bytes changed')
    with Image.open(path) as source:
        image = source.convert('RGB')
    require(image.size == (row['width'], row['height']), 'Image resolution changed')
    return image


def setup_worker(output, plan):
    from common import safe_cache
    safe_cache()
    import torch
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    actual = software()
    paired.check_software(actual, plan['expected_software'])
    freeze(output / 'software.json', actual)
    for file, identity in plan['models']['pid']['files'].items():
        stat = Path(file).stat()
        require((stat.st_size, stat.st_mtime_ns) == (identity['bytes'], identity['mtime_ns']), 'Model file changed: ' + file)
    return AddressBackend(plan)


def semantic_response(result):
    return dict(points=result['points'], predictions=result['predictions'],
        extent_answers=[a['answer'] for a in result['refinements']],
        point_answers=[{k: a.get(k) for k in ('label', 'question', 'answer', 'seed', 'tokens_generated')}
                       for a in result.get('point_answers', [])])


def rotary_snapshot(model):
    import torch
    result = []
    for module in model.backend.model.model.modules():
        if type(module).__name__ != 'Qwen2RotaryEmbedding':
            continue
        h = hashlib.sha256()
        for name in ('cos_cached', 'sin_cached'):
            h.update(getattr(module, name)[:2048].contiguous().view(torch.int16).cpu().numpy().tobytes())
        result.append(dict(sha256=h.hexdigest(), capacity=module.max_seq_len_cached))
    require(len(result) == model.fixed_modules, 'Rotary module count changed')
    return result


def extent_comparison(packed, sequential):
    require(packed['points'] == sequential['points'], 'Sequential probe changed the supplied addresses')
    count = len(packed['points'])
    require(len(packed['refinements']) == len(sequential['refinements']) == count, 'Sequential probe lost targets')
    require(sequential['round2']['actual_round2_forwards'] == sequential['round2']['qwen_total_forwards'] == count,
            'Sequential probe did not use one forward per address')
    pairs = list(zip(packed['refinements'], sequential['refinements']))
    require(all((a['address_id'], a['label']) == (b['address_id'], b['label']) for a, b in pairs),
            'Sequential probe changed address ownership')
    return dict(addresses=count, identical_answers=sum(a['answer'] == b['answer'] for a, b in pairs),
        changed_answers=sum(a['answer'] != b['answer'] for a, b in pairs),
        note='Numerical coordinate/None differences are reported, not silently treated as exact equivalence.')


def gate(args):
    import torch
    plan, data = load_plan(args.output)
    if gate_done(args.output, plan):
        return
    with (args.output / 'gate.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = setup_worker(args.output, plan)
        records = data['DOTAv2']
        dense = max(records, key=lambda r: len(r['gt']))
        smaller = next((r for r in records if str(r['uid']) == '195'), min(records, key=lambda r: len(r['gt'])))
        image = image_for(smaller, plan)
        before = {m: timed_one(model, m, smaller, image) for m in MODES}
        cache_before = rotary_snapshot(model)
        large = timed_one(model, 'gt_address', dense, image_for(dense, plan))
        cache_after = rotary_snapshot(model)
        after = {m: timed_one(model, m, smaller, image) for m in MODES}
        equal = {m: semantic_response(before[m]) == semantic_response(after[m]) for m in MODES}
        unchanged = [v['sha256'] for v in cache_before] == [v['sha256'] for v in cache_after]
        grew = all(b['capacity'] > a['capacity'] for a, b in zip(cache_before, cache_after))
        with torch.inference_mode():
            sequential = model.oracle_sequential(query_only(smaller), image, after['gt_address']['points'])
        extent_probe = extent_comparison(after['gt_address'], sequential)
        for row, results in ((smaller, before), (smaller, after)):
            check_pair(dict(uid=row['uid'], suite=row['suite'], order=list(MODES),
                modes=results, plan_sha256=digest(plan)), row, plan)
        require(len(large['refinements']) == len(dense['gt']) and large['round2']['actual_round2_forwards'] == 1,
                'Dense oracle lost or split addresses')
        evidence = dict(plan_sha256=digest(plan), passed=unchanged and grew and all(equal.values()),
            small_uid=smaller['uid'], dense_uid=dense['uid'], dense_addresses=len(dense['gt']),
            cache_before=cache_before, cache_after=cache_after, cache_grew=grew, cache_prefix_unchanged=unchanged,
            same_outputs_after_growth=equal, before=before, after=after, dense_result=large,
            same_address_extent_comparison=extent_probe, sequential_probe=sequential)
        atomic(args.output / 'gate.json', evidence)
        require(evidence['passed'], 'GPU cache-stability gate failed; see gate.json. Full evaluation has not started.')
        atomic(args.output / 'gate.done.json', dict(complete=True, plan_sha256=digest(plan),
            gate_sha256=sha(args.output / 'gate.json')))
        print('GATE PASSED', evidence['dense_addresses'], 'addresses; cache and both small-image outputs unchanged', flush=True)


def worker(args):
    plan, data = load_plan(args.output)
    require(gate_done(args.output, plan), 'Run the GPU acceptance gate before full evaluation')
    require(0 <= args.shard < plan['shards'], 'Invalid shard')
    directory = folder(args.output, args.dataset, args.shard, plan['shards'])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if finished(args.output, args.dataset, args.shard, plan, data):
            return
        selected = data[args.dataset][args.shard::plan['shards']]
        expected_files = {record_path(directory, i).name for i in range(len(selected))}
        require({p.name for p in directory.glob('record*.json')} <= expected_files, 'Foreign records in output')
        freeze(directory / 'contract.json', dict(plan_sha256=digest(plan), dataset=args.dataset,
            shard=args.shard, shards=plan['shards'], uids=[r['uid'] for r in selected]))
        missing = []
        for i, row in enumerate(selected):
            file = record_path(directory, i)
            if file.exists():
                check_pair(read(file), row, plan)
            else:
                missing.append((i, row))
        if missing:
            model = setup_worker(args.output, plan)
            for i, row in enumerate(selected[:plan['warmup_records']]):
                pair_one(model, row, image_for(row, plan), i, plan)
            for i, row in missing:
                pair = pair_one(model, row, image_for(row, plan), i, plan)
                pair['gpu_uuid'] = os.environ.get('CUDA_VISIBLE_DEVICES')
                atomic(record_path(directory, i), pair)
                atomic(directory / 'progress.json', dict(records=len(list(directory.glob('record*.json'))),
                    expected=len(selected), uid=row['uid']))
                print(args.dataset, f"{len(list(directory.glob('record*.json')))}/{len(selected)}",
                    'uid', row['uid'], 'addresses', {m: len(v['points']) for m, v in pair['modes'].items()}, flush=True)
        files = {record_path(directory, i).name: sha(record_path(directory, i)) for i in range(len(selected))}
        atomic(directory / 'done.json', dict(complete=True, plan_sha256=digest(plan),
            uids=[r['uid'] for r in selected], files=files))


def merge(args):
    plan, data = load_plan(args.output)
    require(gate_done(args.output, plan), 'Acceptance gate is not complete')
    results, gaps, hashes = {}, {}, {}
    scorer = load_upstream()
    lines = ['# PID Predicted vs GT Addresses: FP32 Rotary-Cache Rerun', '',
        'Frozen GT-present-category manifests. Same final PID checkpoint and packed-all extents in both modes.', '',
        '| Dataset | Address source | F1@.50 | F1@.75 | mF1 |', '|---|---|---:|---:|---:|']
    for ds, records in data.items():
        pairs = {}
        for shard in range(plan['shards']):
            require(finished(args.output, ds, shard, plan, data), f'Incomplete {ds} shard {shard}')
            directory = folder(args.output, ds, shard, plan['shards'])
            for i, row in enumerate(records[shard::plan['shards']]):
                file = record_path(directory, i)
                value = read(file)
                check_pair(value, row, plan)
                require(row['uid'] not in pairs, 'Duplicate image')
                pairs[row['uid']] = value
                hashes[str(file.relative_to(args.output))] = sha(file)
        require(set(pairs) == {r['uid'] for r in records}, 'Missing benchmark images')
        cells = {}
        for mode in MODES:
            outputs = [pairs[r['uid']]['modes'][mode] for r in records]
            replay = [reparse('pid', dict(r, width=int(r['width']), height=int(r['height'])), o, 'boxes', True)
                      for r, o in zip(records, outputs)]
            require(all(a['predictions'] == b['predictions'] and a['invalid_items'] == b['invalid_items']
                        for a, b in zip(replay, outputs)), 'Native parser replay differs')
            result, per = score(records, outputs, 'boxes', upstream=scorer)
            strict = [reparse('pid', dict(r, width=int(r['width']), height=int(r['height'])), o, 'boxes', False)
                      for r, o in zip(records, outputs)]
            result['strict_sensitivity'], _ = score(records, strict, 'boxes', upstream=scorer)
            result = metric_record(result, per, outputs, mode, ds, digest(plan), False)
            result.update(oracle=mode == 'gt_address', deployment_latency=mode == 'predicted',
                null_answers=sum('<box>None</box>' in a['answer'] for o in outputs for a in o['refinements']),
                source_addresses=sum(len(o['points']) for o in outputs))
            out = args.output / 'metrics' / mode / ds
            atomic(out / 'metrics.json', result)
            atomic(out / 'per_image.json', per)
            results[f'{mode}/{ds}'] = result
            scores = result['scores']['class_aware']
            cells[mode] = [scores['0.5']['f1'] * 100, scores['0.75']['f1'] * 100, scores['mean_f1_50_95'] * 100]
            lines.append('| ' + ds + ' | ' + mode + ' | ' + ' | '.join(f'{v:.2f}' for v in cells[mode]) + ' |')
        gaps[ds] = [a - b for a, b in zip(cells['gt_address'], cells['predicted'])]
        lines.append('| ' + ds + ' | GT minus predicted (pp) | ' + ' | '.join(f'{v:+.2f}' for v in gaps[ds]) + ' |')
    lines += ['', 'GT addresses are privileged diagnostic inputs, not deployable predictions. '
        'The gap is not an additive decomposition of discovery error: cardinality, location and false proposals all change.', '',
        'GT timing skips discovery and must not be used for a deployment speedup claim. '
        'Predicted-mode timing and both-mode null/coordinate outputs are retained in metrics and raw pairs.', '',
        'No target or output is capped or removed; no extra NMS. mF1 averages all ten IoUs .50:.05:.95 '
        'using the same class-aware, image-averaged P/R harmonic F1 as the current main tables.']
    atomic(args.output / 'summary.json', dict(complete=True, plan_sha256=digest(plan), results=results,
        gt_minus_predicted_pp=gaps, prediction_sha256=hashes))
    (args.output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    atomic(args.output / 'all.done.json', dict(complete=True, plan_sha256=digest(plan),
        records=sum(len(v) for v in data.values()), modes=list(MODES), summary_sha256=sha(args.output / 'summary.json')))
    print('\n'.join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('gate', 'worker', 'merge'))
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--dataset', choices=tuple(DATASETS))
    p.add_argument('--shard', type=int, default=0)
    args = p.parse_args()
    args.output = args.output.resolve()
    globals()[args.mode](args)


if __name__ == '__main__':
    main()
