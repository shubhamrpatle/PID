#!/usr/bin/env python3
"""Parallel class discovery and optional all-address one-forward box refinement."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import benchmark_refwave_full_onepass as prior
from refwave.io import append_jsonl, read_jsonl_log

ref, full = prior.ref, prior.full
OUTPUT = ref.PROJECT / 'runs/refwave_full_parallel_classes_allboxes_10_20260913'
AFTER = ref.PROJECT / 'runs/refwave_cluster_full_8gpu_onepass_20260912/all.done.json'
BLOCKS = re.compile(r'<ref>(.*?)</ref>|<box>(.*?)</box>', re.DOTALL)
POINT = re.compile(r'<(\d+)><(\d+)>')


def read(path):
    return json.loads(path.read_text())


def all_class_prompt(query):
    names = [b['reference'] for b in query['references']]
    if query['task'] != 'detection' or len(names) != len(set(names)) or not names:
        raise ValueError('Expected all unique benchmark categories, not GT-present classes')
    return 'Point to all instances that match the following categories: ' + '</c>'.join(names) + '.'


def parse_all_class_points(answer, query, size):
    """Require explicit class ownership; never infer classes or box-center points."""
    points, counts, seen = [], Counter(), set()
    label = None
    for match in BLOCKS.finditer(answer):
        reference, body = match.groups()
        if reference is not None:
            label = ref.detection.canonicalize_label(reference, query['aliases'], set(query['classes']))
            if label is None:
                counts['unknown_reference_groups'] += 1
            else:
                seen.add(label)
            continue
        if body.strip() == 'None':
            counts['none_groups'] += 1
            continue
        coords = POINT.fullmatch(body.strip())
        if coords is None:
            counts['rejected_nonpoint_blocks'] += 1
            continue
        x, y = map(int, coords.groups())
        counts['point_blocks'] += 1
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            counts['rejected_out_of_range'] += 1
            continue
        if label is None:
            counts['rejected_unlabelled_or_unknown_class'] += 1
            continue
        points.append(dict(label=label, point=[x * size[0] / 1000, y * size[1] / 1000],
                           source='point', score=1.0))
    counts['retained_points'] = len(points)
    counts['duplicate_points'] = len(points) - len({(p['label'], *p['point']) for p in points})
    counts['classes_with_reference'] = len(seen)
    counts['classes_without_reference'] = len(query['classes']) - len(seen)
    remainder = BLOCKS.sub('', answer)
    remainder = remainder.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
    valid = (not remainder and bool(seen) and prior.ablation.grammar_valid(answer, 2)
             and not any(v for k, v in counts.items() if k.startswith(('rejected_', 'unknown_'))))
    return points, dict(counts, grammar_valid=bool(valid))


def decode_all_addresses(decoder, prepared, branches):
    """One fused decoder call, with no width cap, truncation, or OOM fallback."""
    import torch
    if decoder.round2_execution != 'fused_one_pass':
        raise ValueError('All-address execution requires fused_one_pass')
    if decoder.visual_context != 'global_only' or decoder.prefix_cache_mode != 'shared':
        raise ValueError('All-address execution requires the existing shared global-only prefix')
    if prepared.model is not decoder.model or prepared.processor is not decoder.processor:
        raise ValueError('Prepared prefix belongs to another model or processor')
    device = torch.device(decoder.device)
    cuda = device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if branches:
        prefix, suffixes, visuals = decoder._prepared_global_suffixes(prepared, branches)
        prepared_at = time.perf_counter()
        with torch.no_grad():
            outputs = decoder._decode_shared_atomic_blocks(prefix, suffixes, visuals,
                prepared.key_values, branches, temperature=0.0, top_p=1.0,
                repetition_penalty=1.0, allow_none=True)
        if len(outputs) != len(branches):
            raise ValueError('All-address decoder lost an instance')
    else:
        outputs, prepared_at = [], time.perf_counter()
    if cuda:
        torch.cuda.synchronize(device)
    ended = time.perf_counter()
    calls = int(bool(branches))
    return outputs, dict(round2_execution='fused_one_pass', parallel_width='all_predicted_addresses',
        requested_wave_size=len(branches), effective_wave_sizes=[len(branches)] if branches else [],
        waves=calls, addresses=len(branches), qwen_fused_prompt_geometry_forwards=calls,
        qwen_total_forwards=calls, qwen_branch_suffix_forwards=0, qwen_geometry_forwards=0,
        qwen_global_prefix_forwards=0, global_moonvit_encodes=0, local_moonvit_encodes=0,
        cross_round_cache_reuse=True, global_image_preprocessing_calls=0,
        preparation_seconds=prepared_at-started, model_seconds=ended-prepared_at,
        end_to_end_seconds=ended-started, oom_retries=0, oom_policy='raise_without_splitting',
        peak_gpu_mb=torch.cuda.max_memory_allocated(device)/1024**2 if cuda else 0.0)


def prepare(args):
    source = read(prior.OUTPUT / 'contract.json')
    if not read(prior.OUTPUT / 'all.done.json')['completed']:
        raise ValueError('Historical comparison is incomplete')
    if source['identity'] != dict(checkpoint=str(full.CHECKPOINT), sha256=full.CHECKPOINT_SHA, step=2400):
        raise ValueError('Historical comparison uses a different model')
    manifest = Path(source['manifest'])
    if ref.sha256_file(manifest) != source['manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    all_rows = ref.read_jsonl(manifest)
    if not 1 <= args.limit <= len(all_rows):
        raise ValueError('Requested sample count is outside the frozen manifest')
    rows = all_rows[:args.limit]
    expected = [(s['canonical'], s['prompt']) for s in all_rows[0]['class_specs']]
    if len(expected) != 18:
        raise ValueError('Expected the full 18-class DOTAv2 vocabulary')
    for row in rows:
        if [(b['label'], b['reference']) for b in row['references']] != expected:
            raise ValueError('Query vocabulary is not the full benchmark vocabulary')
    contract = dict(identity=source['identity'], manifest=str(manifest),
        manifest_sha256=source['manifest_sha256'], uids=[r['uid'] for r in rows], records=len(rows),
        selection='First N frozen manifest records, no selection by predictions or GT',
        query_mode=args.query_mode,
        round1=('18 isolated class queries processed in parallel; each class generates a variable-length point list'
                if args.query_mode == 'class_parallel' else
                'One all-class query; variable-length autoregressive/native-hybrid point generation'),
        prompts=([b['question'] for b in rows[0]['references']] if args.query_mode == 'class_parallel'
                 else [all_class_prompt(ref.inference_query(rows[0]))]),
        max_new_tokens=args.max_new_tokens, image_token_limit=6000,
        round2_execution='fused_one_pass', box_wave=args.box_width, temperature=.7, top_p=.9,
        repetition_penalty=1.1, trained_round1='class-isolated geometry-only',
        predictions='Model-generated labelled two-coordinate points only; no box-center fallback, '
                    'GT labels, point deduplication, containment filtering, or maxDets',
        limitation=('Inference-only execution diagnostic; unchanged class prompts and per-class seeds. '
                    if args.query_mode == 'class_parallel' else
                    'Inference-only format-transfer diagnostic, not matched retraining. ') +
                   'Historical discovery budget is 2048 per class. Historical latency is not a '
                   'controlled same-session speed comparison. One Round-1 wave is NOT one forward. '
                   'All-address Round 2 fails on OOM instead of silently splitting instances.',
        reference=str(prior.OUTPUT / 'predictions.jsonl'),
        reference_sha256=ref.sha256_file(prior.OUTPUT / 'predictions.jsonl'))
    sources = [Path(__file__).resolve(), Path(prior.fused.__file__), Path(prior.ablation.__file__),
               Path(full.__file__), Path(ref.__file__),
               ref.ROOT / 'zero_shot_analysis/fused_pivr_geometry.py',
               ref.ROOT / 'zero_shot_analysis/pixel_pivr_wave.py',
               ref.ROOT / 'Pixel-PIVR/src/pixel_pivr/fast_class_wave.py',
               ref.ROOT / 'Pixel-PIVR/src/pixel_pivr/hybrid_class_wave.py',
               ref.ROOT / 'Pixel-PIVR/src/pixel_pivr/class_wave.py']
    contract['sources'] = {str(p): ref.sha256_file(p) for p in sources}
    path = args.output / 'contract.json'
    if path.exists() and read(path) != contract:
        raise ValueError('Diagnostic settings/code changed; use another output directory')
    ref.atomic_json(path, contract)
    for p in sources:
        dest = args.output / 'source_snapshot' / p.relative_to(ref.ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(p, dest)
    return rows, contract


def infer_one(worker, decoders, query, image, budget, query_mode, box_width):
    import torch
    from pixel_pivr.class_wave import ClassPointBranch
    from zero_shot_analysis.pointrelay_pbd import base_causal_lm
    point, box = decoders
    if query_mode == 'class_parallel':
        branches = [ClassPointBranch(b['label'], b['question'],
            ref.stable_seed(0, query['suite'], query['uid'], b['label'], 'isolated-point'))
            for b in query['references']]
        point.max_wave_size = len(branches)
    else:
        branches = [ClassPointBranch('__all_classes__', all_class_prompt(query),
            ref.stable_seed(0, query['suite'], query['uid'], 'all-class-point'))]
    ref.grounding.set_generation_seed(ref.stable_seed(0, query['suite'], query['uid'], 'generation'))
    forwards = Counter()
    phase = ['preparation']
    hook = base_causal_lm(worker.model.language_model).model.register_forward_hook(
        lambda module, inputs, output: forwards.update([phase[0]]))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        prepared = point.prepare_image(image, branches)
        preparation_peak = torch.cuda.max_memory_allocated() / 1024**2
        phase[0] = 'round1'
        answers, r1 = point.generate(image, branches, max_new_tokens=budget,
            temperature=.7, top_p=.9, repetition_penalty=1.1, generation_mode='hybrid',
            prepared_image=prepared)
        if len(answers) != len(branches) or r1['waves'] != 1:
            raise ValueError('Round 1 must process every requested query in one parallel wave')
        if query_mode == 'class_parallel':
            points = [p for a in answers for p in prior.ablation.parse_locateanything_points(
                a['answer'], *image.size, a['label']) if p['source'] == 'point']
            parsing = dict(retained_points=len(points),
                duplicate_points=len(points)-len({(p['label'], *p['point']) for p in points}),
                grammar_valid_branches=sum(prior.ablation.grammar_valid(a['answer'], 2) for a in answers))
        else:
            points, parsing = parse_all_class_points(answers[0]['answer'], query, image.size)
        owned = prior.fused.addresses(points, image.size)
        torch.cuda.synchronize()
        round1_seconds = time.perf_counter() - start
        round1_peak = torch.cuda.max_memory_allocated() / 1024**2
        phase[0] = 'round2'
        try:
            results, r2 = (decode_all_addresses(box, prepared, owned) if box_width == 'all' else
                box.decode_prepared_image(prepared, owned, requested_wave_size=200, allow_none=True))
        except torch.cuda.OutOfMemoryError as exc:
            exc.diagnostic = dict(uid=query['uid'], predicted_points=len(points), box_width=box_width,
                query_mode=query_mode, point_answers=answers, points=points,
                reason='All-address forward exceeded VRAM; no points were truncated and no fallback was run')
            raise
        if len(results) != len(points) or r2['round2_execution'] != 'fused_one_pass':
            raise ValueError('Round 2 lost an address or changed execution mode')
        if (forwards['round2'] != r2['qwen_fused_prompt_geometry_forwards']
                or r2['qwen_branch_suffix_forwards'] or r2['qwen_geometry_forwards']):
            raise ValueError('Actual Round-2 call count differs from fused execution')
        if box_width == 'all' and forwards['round2'] != int(bool(points)):
            raise ValueError('All-address Round 2 used more than one forward')
        if forwards['preparation'] != 1 or r1['global_moonvit_encodes'] or r2['global_moonvit_encodes']:
            raise ValueError('Global preparation was repeated')
        boxes = [p for item in results for p in prior.ablation.parsed_boxes(item['answer'], query, image.size)]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        return dict(uid=query['uid'], suite=query['suite'], task='detection', predictions=boxes,
            points=points, point_answers=answers, refinements=results, round1=r1, round2=r2,
            parser=parsing, actual_decoder_forwards=dict(forwards), round1_generate_calls=1,
            round1_query_branches=len(branches), query_mode=query_mode, box_width=box_width,
            round1_seconds=round1_seconds, round2_seconds=seconds-round1_seconds,
            latency_seconds=seconds, peak_gpu_mb=max(preparation_peak, round1_peak,
                r1['peak_gpu_mb'], r2['peak_gpu_mb']),
            budget_saturated=any(a.get('fallback_reason') == 'max_new_tokens' for a in answers))
    finally:
        hook.remove()


def summarize(args, rows, contract):
    saved = ref.read_jsonl(args.output / 'predictions.jsonl')
    indexed = {p['uid']: p for p in saved}
    if len(saved) != len(indexed) or set(indexed) != set(contract['uids']):
        raise ValueError('Incomplete or duplicate diagnostic outputs')
    saved = [indexed[r['uid']] for r in rows]
    if ref.sha256_file(Path(contract['reference'])) != contract['reference_sha256']:
        raise ValueError('Historical reference changed')
    archived = {p['uid']: p for p in ref.read_jsonl(Path(contract['reference']))
                if p['wave'] == 200 and p['repeat'] == 0}
    historical = [archived[r['uid']] for r in rows]
    report = dict(contract=contract, arms={})
    lines = ['# Parallel class discovery and all-address box decoding', '', contract['limitation'], '',
        f"Same {len(rows)} images; full-FT Stage-2 checkpoint 2400; native-6000; fused parallel Round 2.",
        f"Round-1 mode: {args.query_mode}. Round-2 parallel width: {args.box_width}.",
        'No sequential refinement was run.', '',
        '| Query mode | Points | TP | FP | FN | P@0.5 | R@0.5 | F1@0.5 | TPI s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    new_name = args.query_mode + ' / box-width-' + args.box_width + ' (new)'
    for name, predictions in [('class-wave18 / box-wave200 (archive)', historical), (new_name, saved)]:
        metrics, per = full.score(rows, predictions)
        metrics['points'] = sum(len(p['points']) for p in predictions)
        metrics['point_diagnostics'] = prior.ablation.point_diagnostics(rows, predictions)
        report['arms'][name] = metrics
        ref.jsonl(args.output / ('new_per_image.jsonl' if name == new_name else 'archive_per_image.jsonl'), per)
        m = metrics['raw']['0.5']
        lines.append(f"| {name} | {metrics['points']} | {m['tp']} | {m['fp']} | {m['fn']} | "
            f"{100*m['precision']:.2f} | {100*m['recall']:.2f} | {100*m['f1']:.2f} | {metrics['mean_seconds']:.3f} |")
    totals = Counter()
    for p in saved:
        totals.update(p['parser'])
    report['parsing'] = dict(totals)
    report['budget_saturated_images'] = sum(p['budget_saturated'] for p in saved)
    report['reference_equivalence'] = dict(
        point_answers_equal=sum([a['answer'] for a in p['point_answers']] ==
                                [a['answer'] for a in archived[p['uid']]['point_answers']] for p in saved),
        point_lists_equal=sum(p['points'] == archived[p['uid']]['points'] for p in saved),
        geometry_equal=sum(prior.fused.geometry_signature(p['refinements']) ==
                           prior.fused.geometry_signature(archived[p['uid']]['refinements']) for p in saved),
        predictions_equal=sum(p['predictions'] == archived[p['uid']]['predictions'] for p in saved),
        records=len(saved))
    report['actual_round2_forward_counts'] = [p['actual_decoder_forwards'].get('round2', 0) for p in saved]
    lines += ['', f"Parsing totals: {dict(totals)}", f"Budget-saturated images: {report['budget_saturated_images']}",
        f"Archived equivalence: {report['reference_equivalence']}",
        f"Actual Round-2 forwards per image: {report['actual_round2_forward_counts']}",
        'Raw results above. NMS0.5, per-class results, and point recall are separate in comparison.json.',
        'Class-specific query identity supplies point ownership; no GT classes are consulted.',
        'All-address mode attempts exactly one Round-2 forward for nonempty points, zero for empty points.',
        'Historical timings are not a controlled speed comparison. BF16 batch widths can change predictions.']
    ref.atomic_json(args.output / 'comparison.json', report)
    (args.output / 'summary.md').write_text('\n'.join(lines) + '\n')
    ref.atomic_json(args.output / 'all.done.json', dict(completed=True, records=len(saved)))
    ref.atomic_json(args.output / 'status.json', dict(state='completed', completed=len(saved), expected=len(rows)))
    print('\n'.join(lines), flush=True)


def run(args, rows, contract):
    import torch
    from PIL import Image
    if ref.sha256_file(full.CHECKPOINT) != full.CHECKPOINT_SHA:
        raise ValueError('Checkpoint hash mismatch')
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    previous = read_jsonl_log(args.output / 'predictions.jsonl', repair_tail=True)
    done = {p['uid'] for p in previous}
    if len(done) != len(previous) or not done <= set(contract['uids']):
        raise ValueError('Saved outputs do not belong to this diagnostic')
    worker = full.load_full_worker()
    decoders = prior.fused.make_decoders(worker)
    append_jsonl(args.output / 'sessions.jsonl', dict(started=time.time(),
        gpu=torch.cuda.get_device_name(0), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        torch=torch.__version__, timing='CUDA synchronized, image loading/model loading/scoring excluded'))
    with Image.open(rows[0]['image_path']) as opened:
        infer_one(worker, decoders, ref.inference_query(rows[0]), opened.convert('RGB'), args.max_new_tokens,
                  args.query_mode, args.box_width)
    for row in rows:
        if row['uid'] in done:
            continue
        with Image.open(row['image_path']) as opened:
            image = opened.convert('RGB')
        if image.size != (row['width'], row['height']):
            raise ValueError('Image dimensions differ from the frozen manifest')
        result = infer_one(worker, decoders, ref.inference_query(row), image, args.max_new_tokens,
                           args.query_mode, args.box_width)
        result.update(dataset=row['dataset'], image_id=row['image_id'])
        append_jsonl(args.output / 'predictions.jsonl', result)
        done.add(row['uid'])
        ref.atomic_json(args.output / 'status.json', dict(state='running', completed=len(done), expected=len(rows)))
        print(f"{len(done)}/{len(rows)} uid={row['uid']} points={len(result['points'])} "
              f"boxes={len(result['predictions'])} forwards={result['actual_decoder_forwards']} "
              f"parsing={result['parser']}", flush=True)
    summarize(args, rows, contract)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'queue', 'summarize'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--query-mode', choices=('class_parallel', 'all_class'), default='class_parallel')
    parser.add_argument('--box-width', choices=('all', '200'), default='all')
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--after', type=Path, default=AFTER)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 16384:
        parser.error('Use a positive discovery budget no larger than 16384')
    args.output.mkdir(parents=True, exist_ok=True)
    kind = 'queue' if args.action == 'queue' else 'worker'
    with (args.output / f'{kind}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            rows, contract = prepare(args)
            if args.action == 'prepare':
                print(json.dumps({k: v for k, v in contract.items() if k != 'sources'}, indent=2))
            elif args.action == 'summarize' or (args.output / 'all.done.json').exists():
                summarize(args, rows, contract)
            elif args.action == 'queue':
                while not args.after.exists() or not read(args.after).get('completed'):
                    ref.atomic_json(args.output / 'status.json', dict(state='waiting_for_prior_evaluation',
                        after=str(args.after), gpu=args.gpu, completed=0, expected=len(rows)))
                    print(f'Waiting for {args.after}; no running processes will be stopped.', flush=True)
                    time.sleep(30)
                ref.atomic_json(args.output / 'status.json', dict(state='waiting_for_gpu', gpu=args.gpu))
                full.wait_idle([args.gpu])
                command = [sys.executable, '-u', str(Path(__file__).resolve()), 'run', '--output', str(args.output),
                           '--limit', str(args.limit), '--max-new-tokens', str(args.max_new_tokens),
                           '--query-mode', args.query_mode, '--box-width', args.box_width]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4',
                           TOKENIZERS_PARALLELISM='false')
                subprocess.run(command, cwd=ref.ROOT, env=env, check=True)
            else:
                run(args, rows, contract)
        except Exception as exc:
            if hasattr(exc, 'diagnostic'):
                ref.atomic_json(args.output / 'oom_diagnostic.json', exc.diagnostic)
            ref.atomic_json(args.output / 'status.json', dict(state='failed', error=str(exc)))
            raise


if __name__ == '__main__':
    main()
