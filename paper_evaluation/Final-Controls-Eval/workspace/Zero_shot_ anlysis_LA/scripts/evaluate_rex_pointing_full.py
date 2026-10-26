#!/usr/bin/env python3
"""Pinned Rex box evaluation and native-point baselines on full RS manifests."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

import evaluate_paper_comparators as common
import evaluate_rex_omni_paper as rex

ROOT, PROJECT = common.ROOT, common.PROJECT
sys.path[:0] = [str(ROOT / 'RefWave/src'), str(ROOT / 'Pixel-PIVR/src')]
OUTPUT = PROJECT / 'runs/paper_rex_pointing_full_20260915_v1'
SUITES = ('DIOR', 'DOTAv2', 'DIOR-RSVG', 'VRSBench-VG')
BASE = PROJECT / 'model/LocateAnything-3B/checkpoint'
MODELS = {
    'rexomni': dict(repo=rex.REPO, revision=rex.REVISION, cache=str(rex.CACHE)),
    **{k: dict(repo=common.MODELS[k]['repo'], revision=common.MODELS[k]['revision'],
              cache=str(common.CACHE)) for k in ('qwen3_4b', 'qwen3_8b')},
    'molmo_7b_d': dict(repo='allenai/Molmo-7B-D-0924',
        revision='cab33fb7f1a40091911f81165f8481920621948f',
        cache='/data2/shubham/models/qualitative_hf_cache'),
    'base_la': dict(repo='nvidia/LocateAnything-3B', local_path=str(BASE)),
}
THRESHOLDS = tuple((50 + 5 * i) / 100 for i in range(10))


def freeze(path, value):
    value = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Changed contract: {path}; use a new release directory')
    common.write(path, value)


def prepare(args):
    from huggingface_hub import HfApi, snapshot_download
    from PIL import Image
    assets = args.output / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    spec = MODELS[args.model]
    weight_hashes = {}
    if args.model == 'base_la':
        path = BASE
        index = json.loads((path / 'model.safetensors.index.json').read_text())
        for name in sorted(set(index['weight_map'].values())):
            weight_hashes[name] = common.sha(path / name)
        remote_verified = False
    else:
        info = HfApi().model_info(spec['repo'], revision=spec['revision'], files_metadata=True)
        if info.sha != spec['revision']:
            raise ValueError('Resolved model revision differs')
        cache = Path(spec['cache'])
        cache.mkdir(parents=True, exist_ok=True)
        expected = cache / ('models--' + spec['repo'].replace('/', '--')) / 'snapshots' / info.sha
        missing = sum(f.size or 0 for f in info.siblings
                      if f.rfilename.endswith('.safetensors') and not (expected / f.rfilename).exists())
        if shutil.disk_usage(cache).free < missing + 10 * 1024**3:
            raise RuntimeError('Insufficient model-cache space plus 10 GiB reserve')
        path = Path(snapshot_download(spec['repo'], revision=info.sha, cache_dir=cache,
            allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', '*.py'], max_workers=2))
        for f in info.siblings:
            p = path / f.rfilename
            if not p.exists() or p.suffix not in ('.safetensors', '.json', '.py', '.txt', '.model', '.jinja'):
                continue
            if p.stat().st_size != f.size:
                raise ValueError(f'Download size mismatch: {p}')
            digest = common.sha(p)
            if f.lfs:
                valid = digest == f.lfs.sha256
            else:
                git_digest = hashlib.sha1(b'blob ' + str(f.size).encode() + b'\0' + p.read_bytes()).hexdigest()
                valid = git_digest == f.blob_id
            if not valid:
                raise ValueError(f'Download hash mismatch: {p}')
            weight_hashes[f.rfilename] = digest
            print('VERIFIED', args.model, f.rfilename, flush=True)
        for f in info.siblings:
            if f.rfilename.endswith('.safetensors') and f.rfilename not in weight_hashes:
                raise ValueError('Missing model weights: ' + f.rfilename)
        remote_verified = True
    freeze(assets / (args.model + '.json'), dict(**spec, path=str(path),
        files_sha256=weight_hashes, hub_hashes_verified=remote_verified))
    for suite in SUITES:
        source = common.ASSETS / (suite + '.jsonl')
        dest = assets / source.name
        if dest.exists() and common.sha(dest) != common.sha(source):
            raise ValueError('Frozen manifest changed')
        if not dest.exists():
            shutil.copy2(source, dest)
        rows = common.read(dest)
        if len({r['uid'] for r in rows}) != len(rows):
            raise ValueError('Duplicate manifest IDs')
        dimensions = {}
        for row in rows:
            image_path = row['image_path']
            if image_path not in dimensions:
                with Image.open(image_path) as im:
                    dimensions[image_path] = im.size
            if dimensions[image_path] != (row['width'], row['height']):
                raise ValueError('Image/manifest dimension mismatch')
        freeze(assets / (suite + '.json'), dict(records=len(rows), images=len(dimensions),
            pointing_queries=sum(len(references(r)) for r in rows),
            targets=sum(len(r['gt']) if r['task'] == 'detection' else 1 for r in rows),
            sha256=common.sha(dest)))
    print('PREPARED', args.model, path, flush=True)


def references(row):
    if row['task'] == 'detection':
        return [dict(label=s['canonical'], text=s['prompt']) for s in row['class_specs']]
    return [dict(label=row['query'], text=row['query'])]


def qwen_point_prompt(text):
    return (f'Point to all instances matching the following description: {text}. '
            'Return a JSON array of objects with "point_2d": [x,y], normalized to 0-1000. '
            'Return [] if absent.')


def parse_qwen_points(raw, width, height):
    decoder = json.JSONDecoder()
    points, errors, cursor = [], [], 0
    while cursor < len(raw):
        start = raw.find('{', cursor)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            errors.append('invalid_or_truncated_json')
            cursor = start + 1
            continue
        cursor = start + end
        p = obj.get('point_2d') if isinstance(obj, dict) else None
        if (not isinstance(p, list) or len(p) != 2 or
                not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1000 for v in p)):
            errors.append('non_point_or_invalid_coordinates')
            continue
        points.append([p[0] * width / 1000, p[1] * height / 1000])
    if not points and not errors and raw.strip().strip('`').strip() not in ('[]', 'json\n[]'):
        errors.append('no_valid_point_or_explicit_empty_array')
    return points, errors


def parse_la_points(raw, width, height):
    points, errors = [], []
    for block in re.findall(r'<box>(.*?)</box>', raw, re.DOTALL):
        if block.strip() == 'None':
            continue
        match = re.fullmatch(r'\s*<(\d+)>\s*<(\d+)>\s*', block)
        if not match or any(int(v) > 1000 for v in match.groups()):
            errors.append('non_point_or_invalid_coordinates')
            continue
        x, y = map(int, match.groups())
        points.append([x * width / 1000, y * height / 1000])
    if raw.count('<box>') != raw.count('</box>'):
        errors.append('incomplete_geometry_block')
    if not points and not errors and '<box>None</box>' not in raw:
        errors.append('no_valid_point_or_explicit_null')
    return points, errors


def load_model(args, identity):
    import torch
    from transformers import AutoProcessor
    path = identity['path']
    if args.model == 'rexomni':
        import subprocess
        if subprocess.check_output(['git', '-C', str(rex.CODE), 'rev-parse', 'HEAD'], text=True).strip() != rex.CODE_REVISION:
            raise ValueError('Official Rex code revision changed')
        sys.path.insert(0, str(rex.CODE))
        from rex_omni import RexOmniWrapper
        return RexOmniWrapper(path, backend='transformers', max_tokens=args.budget,
            temperature=0, top_p=.05, top_k=1, repetition_penalty=1.05,
            attn_implementation='sdpa', device_map='cuda:0', torch_dtype=torch.bfloat16)
    if args.model.startswith('qwen3'):
        return common.load_backend(args)
    if args.model == 'base_la':
        from pixel_pivr.worker import LocateAnythingPixelPIVRWorker
        worker = LocateAnythingPixelPIVRWorker(path, device='cuda:0')
        worker.processor.image_processor.in_token_limit = 6000
        return worker
    from transformers import AutoModelForCausalLM
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map='cuda:0', local_files_only=True).eval()
    return dict(model=model, processor=processor)


def native_point(state, image, ref, args):
    import torch
    text, label = ref['text'], ref['label']
    tokens = None
    effective_budget = args.budget
    if args.model == 'rexomni':
        result = state.inference(images=image, task='pointing', categories=[text])[0]
        if not result['success']:
            raise RuntimeError(result.get('error', 'Rex inference failed'))
        raw, prompt, tokens = result['raw_output'], result['prompt'], result['num_output_tokens']
        xy, errors = [], []
        for items in result['extracted_predictions'].values():
            for item in items:
                if item['type'] == 'point':
                    xy.append(item['coords'])
                else:
                    errors.append('non_point_geometry')
    elif args.model.startswith('qwen3'):
        processor, model = state['processor'], state['model']
        prompt = qwen_point_prompt(text)
        messages = [dict(role='user', content=[dict(type='image', image=image), dict(type='text', text=prompt)])]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=image, text=[rendered], return_tensors='pt',
                           max_pixels=6000 * 14 * 14, min_pixels=4096).to('cuda')
        ids = model.generate(**inputs, max_new_tokens=args.budget, do_sample=True,
            temperature=.7, top_p=.9, top_k=0, repetition_penalty=1.1, use_cache=True)
        generated = ids[0, inputs.input_ids.shape[1]:]
        raw = processor.decode(generated, skip_special_tokens=True)
        tokens = len(generated)
        xy, errors = parse_qwen_points(raw, *image.size)
    elif args.model == 'base_la':
        prompt = f'Point to: {text}.'
        raw = state.predict_points(image, prompt, max_new_tokens=args.budget,
                                   temperature=.7, top_p=.9, repetition_penalty=1.1)
        tokens = len(state.tokenizer.encode(raw, add_special_tokens=False))
        xy, errors = parse_la_points(raw, *image.size)
    else:
        from transformers import GenerationConfig
        from evaluate_molmo_pointing import parse_molmo2
        prompt = f'Point to {text}.'
        processor, model = state['processor'], state['model']
        inputs = processor.process(images=[image], text=prompt)
        inputs = {k: v.to('cuda').unsqueeze(0) for k, v in inputs.items()}
        inputs['images'] = inputs['images'].to(torch.bfloat16)
        effective_budget = min(args.budget, model.config.max_position_embeddings - inputs['input_ids'].shape[1])
        if effective_budget < 1:
            raise ValueError('Molmo prompt exhausts its native positional context')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ids = model.generate_from_batch(inputs, GenerationConfig(max_new_tokens=effective_budget,
                do_sample=False, use_cache=True, stop_strings='<|endoftext|>'), tokenizer=processor.tokenizer)
        generated = ids[0, inputs['input_ids'].shape[1]:]
        raw = processor.tokenizer.decode(generated, skip_special_tokens=False)
        tokens = len(generated)
        xy, errors = parse_molmo2(raw, *image.size)
    from evaluate_molmo_pointing import validate_xy
    xy, invalid = validate_xy(xy, *image.size)
    errors.extend(['out_of_bounds_coordinate'] * invalid)
    return dict(label=label, points=[dict(label=label, point=p, source='point') for p in xy],
        raw_response=raw, prompt=prompt, parse_errors=errors, generated_tokens=tokens,
        token_count_kind='retokenized_response' if args.model == 'base_la' else 'generated_ids',
        effective_budget=effective_budget, budget_saturated=tokens >= effective_budget)


def rex_boxes(state, row, image):
    query = common.query_only(row)
    categories = ([s['prompt'] for s in query['class_specs']]
                  if query['task'] == 'detection' else [query['query']])
    result = state.inference(images=image, task='detection', categories=categories)[0]
    if not result['success']:
        raise RuntimeError(result.get('error', 'Rex inference failed'))
    if query['task'] == 'detection':
        boxes, rejected = rex.parse_boxes(result['extracted_predictions'], query['class_specs'], *image.size)
    else:
        import evaluate_locateanything_lae_style as det
        boxes, rejected = [], []
        # A single full expression is the only semantic input; labels are not GT classes.
        for items in result['extracted_predictions'].values():
            for item in items:
                b = det.clip_hbox_to_image(item['coords'], *image.size) if item['type'] == 'box' else None
                if b is None:
                    rejected.append(item)
                else:
                    boxes.append(dict(label=query['query'], hbox=b, score=1.0))
    return dict(predictions=boxes, rejected_labels=rejected, raw_response=result['raw_output'],
                prompt=result['prompt'], generated_tokens=result['num_output_tokens'])


def folder(args):
    return args.output / ('smoke' if args.limit else 'results') / args.kind / args.model / args.suite


def infer(args):
    import torch
    import transformers
    from PIL import Image
    from refwave.io import read_jsonl_log
    from evaluate_locateanything_grounding_benchmarks import set_generation_seed
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    manifest = args.output / 'assets' / (args.suite + '.jsonl')
    all_rows = common.read(manifest)
    if args.limit:
        all_rows = all_rows[:args.limit]
    rows = all_rows[args.shard_index::args.shard_count]
    identity = json.loads((args.output / 'assets' / (args.model + '.json')).read_text())
    directory = folder(args)
    directory.mkdir(parents=True, exist_ok=True)
    tag = f'shard{args.shard_index:02d}of{args.shard_count:02d}'
    path = directory / (tag + '.jsonl')
    contract = dict(protocol='rs_full_native_points_and_rex_boxes_v1', model=identity,
        kind=args.kind, suite=args.suite, manifest_sha256=common.sha(manifest), limit=args.limit,
        shard_index=args.shard_index, shard_count=args.shard_count, budget=args.budget,
        seed=0, source_sha256=common.sha(__file__),
        dependency_sources={str(p):common.sha(p) for p in [Path(common.__file__), Path(rex.__file__),
            PROJECT/'scripts/evaluate_molmo_pointing.py', ROOT/'Pixel-PIVR/src/pixel_pivr/worker.py']},
        rex_official_sources={str(p):common.sha(p) for p in sorted((rex.CODE/'rex_omni').glob('*.py'))},
        prompting='all requested classes, one native point query per class; full expression for grounding',
        point_parser='native points only; no box-center conversion; duplicate points retained',
        generation='Rex greedy repetition1.05; Qwen/LA stable seed0 sample .7/top_p.9/repetition1.1; Molmo greedy',
        native_context_limit='Molmo: min(requested generation budget, 4096 minus prompt length), logged per query',
        images='original RGB; Rex native2560*28^2; Qwen6000*14^2; LA6000 premerge patches; Molmo native crops',
        dtype='bfloat16', attention='native/SDPA; no custom shared-prefix modification',
        timing='CUDA synchronized preprocessing+all requested query generation+parsing; excludes disk/weights/warmup/scoring',
        query_schedule='serial per reference; no GT-present filtering or cross-query feature cache',
        added_nms=False, max_detections=None, warmup_queries=2,
        torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0))
    freeze(directory / (tag + '.contract.json'), contract)
    expected = {r['uid'] for r in rows}
    old = read_jsonl_log(path, repair_tail=True) if path.exists() else []
    done = {r['uid'] for r in old}
    if len(old) != len(done) or not done <= expected:
        raise ValueError('Duplicate or foreign saved prediction IDs')
    if done != expected:
        state = load_model(args, identity)

        def timed(row, image, ref=None):
            seed_text = json.dumps([0, args.suite, row['uid'], ref['label'] if ref else 'boxes'])
            set_generation_seed(int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4], 'big'))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                result = native_point(state, image, ref, args) if ref else rex_boxes(state, row, image)
            torch.cuda.synchronize()
            result.update(latency_seconds=time.perf_counter()-started,
                          peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2)
            result['budget_saturated'] = result['generated_tokens'] >= result.get('effective_budget', args.budget)
            return result

        row = rows[0]
        with Image.open(row['image_path']) as opened:
            image = opened.convert('RGB')
        for ref in (references(row)[:2] if args.kind == 'pointing' else [None, None]):
            timed(row, image, ref)
        with path.open('a') as stream:
            for row in rows:
                if row['uid'] in done:
                    continue
                with Image.open(row['image_path']) as opened:
                    image = opened.convert('RGB')
                if image.size != (row['width'], row['height']):
                    raise ValueError('Image/manifest dimensions differ')
                if args.kind == 'pointing':
                    answers = [timed(row, image, ref) for ref in references(row)]
                    result = dict(answers=answers, points=[p for a in answers for p in a['points']],
                        latency_seconds=sum(a['latency_seconds'] for a in answers),
                        peak_gpu_mb=max(a['peak_gpu_mb'] for a in answers))
                else:
                    result = timed(row, image)
                result.update({k:row[k] for k in ('uid', 'dataset', 'image_id', 'task', 'suite')})
                if row['task'] == 'grounding':
                    result.update(sample_id=row['sample_id'], query=row['query'])
                stream.write(json.dumps(result, sort_keys=True)+'\n')
                stream.flush()
                done.add(row['uid'])
                common.write(directory / (tag + '.status.json'), dict(records=len(done), expected=len(rows)))
                print(args.model, args.kind, args.suite, tag, f'{len(done)}/{len(rows)}',
                      'seconds', round(result['latency_seconds'], 3), flush=True)
    common.write(directory / (tag + '.done.json'), dict(records=len(rows), complete=True))


def point_metrics(rows, preds):
    from run_locany_pointing_balanced100 import match_points_to_gt, prf
    totals, per_class, per_query = Counter(tp=0, fp=0, fn=0), {}, []
    for row, pred in zip(rows, preds):
        expected = {r['label'] for r in references(row)}
        answers = {a['label']: a for a in pred['answers']}
        if len(answers) != len(pred['answers']) or set(answers) != expected:
            raise ValueError('Missing, duplicated or foreign class/expression answers')
        for ref in references(row):
            label = ref['label']
            answer = answers[label]
            gt = ([g['hbox'] for g in row['gt'] if g['label'] == label]
                  if row['task'] == 'detection' else [row['gt_hbox']])
            match = match_points_to_gt(gt, answer['points'])
            count = {k:match[k] for k in ('tp', 'fp', 'fn')}
            totals.update(count)
            if row['task'] == 'detection':
                per_class.setdefault(label, Counter()).update(count)
            per_query.append(dict(uid=row['uid'], label=label, targets=len(gt), **count,
                **prf(**count), seconds=answer['latency_seconds'], parse_errors=answer['parse_errors'],
                budget_saturated=answer['budget_saturated']))
    return dict(**totals, **prf(**totals), queries=len(per_query),
        targets=totals['tp']+totals['fn'],
        parse_error_queries=sum(bool(q['parse_errors']) for q in per_query),
        budget_saturated_queries=sum(q['budget_saturated'] for q in per_query),
        per_class={k:dict(v, **prf(**v)) for k,v in per_class.items()}), per_query


def merge(args):
    import numpy as np
    rows = common.read(args.output / 'assets' / (args.suite + '.jsonl'))
    if args.limit:
        rows = rows[:args.limit]
    directory = folder(args)
    preds, contracts = [], []
    for i in range(args.shard_count):
        tag = f'shard{i:02d}of{args.shard_count:02d}'
        if not (directory / (tag + '.done.json')).exists():
            raise ValueError('Incomplete shard: ' + tag)
        items = common.read(directory / (tag + '.jsonl'))
        if {p['uid'] for p in items} != {r['uid'] for r in rows[i::args.shard_count]}:
            raise ValueError('Shard membership differs from manifest')
        preds.extend(items)
        contracts.append(json.loads((directory / (tag + '.contract.json')).read_text()))
    if any({k:v for k,v in c.items() if k != 'shard_index'} !=
           {k:v for k,v in contracts[0].items() if k != 'shard_index'} for c in contracts):
        raise ValueError('Mixed inference contracts')
    if contracts[0]['manifest_sha256'] != common.sha(args.output/'assets'/(args.suite+'.jsonl')):
        raise ValueError('Manifest changed since inference')
    indexed = {p['uid']: p for p in preds}
    if len(indexed) != len(preds) or set(indexed) != {r['uid'] for r in rows}:
        raise ValueError('Incomplete or duplicated prediction IDs')
    preds = [indexed[r['uid']] for r in rows]
    times = [p['latency_seconds'] for p in preds]
    metrics = dict(complete=True, records=len(rows), contract=contracts[0],
        mean_seconds=float(np.mean(times)), p50_seconds=float(np.median(times)),
        p95_seconds=float(np.percentile(times, 95)), peak_gpu_mb=max(p['peak_gpu_mb'] for p in preds),
        prediction_files={p.name:common.sha(p) for p in sorted(directory.glob('shard*of*.jsonl'))})
    if args.kind == 'pointing':
        metrics['pointing'], per = point_metrics(rows, preds)
        common.write(directory/'per_query_metrics.json', per)
        metrics.update(points_per_second=(metrics['pointing']['tp']+metrics['pointing']['fp'])/sum(times),
                       correct_points_per_second=metrics['pointing']['tp']/sum(times))
    elif rows[0]['task'] == 'detection':
        import evaluate_locateanything_lae_style as det
        index = {(p['dataset'],p['image_id']):p for p in preds}
        classes = [s['canonical'] for s in rows[0]['class_specs']]
        metrics['raw'] = {f'{t:g}':det.f1_for_threshold(rows,index,classes,t,0) for t in THRESHOLDS}
        metrics['mean_f1_iou_0p50_to_0p95'] = float(np.mean([v['f1'] for v in metrics['raw'].values()]))
        count = sum(len(p['predictions']) for p in preds)
        assert metrics['raw']['0.5']['tp'] + metrics['raw']['0.5']['fp'] == count
        metrics.update(bps=count/sum(times), cbps_iou0p5=metrics['raw']['0.5']['tp']/sum(times))
    else:
        import evaluate_locateanything_grounding_benchmarks as ground
        result = ground.evaluate_predictions(rows, preds)
        assert result['missing_predictions'] == 0
        metrics['grounding'], metrics['by_size'] = result['overall'], result['by_size']
        common.write(directory/'per_query_metrics.json', result['rows'])
    common.write(directory/'metrics.json', metrics)
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('contract','by_size')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare','infer','merge'])
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--kind', choices=['boxes','pointing'], default='boxes')
    p.add_argument('--suite', choices=SUITES, default='DIOR')
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--budget', type=int, default=8192)
    a = p.parse_args()
    if not 0 <= a.shard_index < a.shard_count or a.limit < 0 or a.budget < 1:
        p.error('Invalid shard, limit or budget')
    if a.mode != 'prepare' and a.kind == 'boxes' and a.model != 'rexomni':
        p.error('This box evaluator is for Rex-Omni only')
    globals()[a.mode](a)


if __name__ == '__main__':
    main()
