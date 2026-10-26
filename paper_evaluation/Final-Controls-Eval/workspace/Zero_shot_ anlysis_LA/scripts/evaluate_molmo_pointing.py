#!/usr/bin/env python3
"""Pinned native Molmo image pointing on the frozen all-category DOTAv2 diagnostic."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET

from evaluate_paper_comparators import PROJECT, ROOT, ASSETS, read, sha, write

OUTPUT = PROJECT / 'runs/molmo_pointing_comparators_20260914'
CACHE = Path('/data2/shubham/models/molmo_pointing_hf')
MANIFEST = ASSETS / 'DOTAv2-Balanced100.jsonl'
MODELS = {
    'molmo2_4b': dict(repo='allenai/Molmo2-4B',
                      revision='042abfa7a38879a376cec03d949eff0aefaa0600'),
    'molmopoint_8b': dict(repo='allenai/MolmoPoint-8B',
                         revision='188130f961c8e0888a34e11121a1423c461a01ba'),
}
OFFICIAL_CODE_REVISION = 'f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a'


def model_path(name):
    spec = MODELS[name]
    return CACHE / ('models--' + spec['repo'].replace('/', '--')) / 'snapshots' / spec['revision']


def query_specs(row):
    # Only the complete requested vocabulary reaches inference, never GT labels/counts.
    return [dict(label=s['canonical'], prompt=f"Point to: {s['prompt']}.")
            for s in row['class_specs']]


def messages(image, prompt):
    return [dict(role='user', content=[dict(type='text', text=prompt),
                                     dict(type='image', image=image)])]


def parse_molmo2(text, width, height):
    """Native html-v1/v2 (0..1000), with legacy XML (0..100) fallback.

    Semantics follow allenai/molmo2 olmo/preprocessing/point_formatter.py.
    XML parsing prevents labels or unrelated response numbers becoming coordinates.
    """
    candidates, errors = [], []
    for match in re.finditer(r'<(?:points|point)\b[^>]*>', text, flags=re.DOTALL):
        start = match.group()
        try:
            node = ET.fromstring(start if start.endswith('/>') else start[:-1] + '/>')
        except ET.ParseError:
            errors.append('malformed_point_tag')
            continue
        if 'coords' in node.attrib:
            groups = re.split(r'[\t:;,]', node.attrib['coords'])
            for group in groups:
                values = group.split()
                if not values:
                    continue
                if values[0] not in ('1', '1.0'):
                    errors.append('unexpected_image_id')
                    continue
                values = values[1:]
                if len(values) % 3:
                    errors.append('incomplete_coordinate_triple')
                for i in range(0, len(values) - 2, 3):
                    ident, x, y = values[i:i+3]
                    if not (ident.isdigit() and re.fullmatch(r'[0-9]{3,4}', x)
                            and re.fullmatch(r'[0-9]{3,4}', y)):
                        errors.append('invalid_coordinate_triple')
                        continue
                    candidates.append([float(x) * width / 1000, float(y) * height / 1000])
        else:
            indices = [''] if 'x' in node.attrib else sorted(
                [k[1:] for k in node.attrib if re.fullmatch(r'x\d+', k)], key=int)
            for index in indices:
                try:
                    x, y = float(node.attrib['x'+index]), float(node.attrib['y'+index])
                    candidates.append([x * width / 100, y * height / 100])
                except (ValueError, KeyError):
                    errors.append('invalid_legacy_coordinate')
    valid, invalid = validate_xy(candidates, width, height)
    if invalid:
        errors.extend(['out_of_bounds_coordinate'] * invalid)
    # Incomplete tags must not be silently credited as correct absence.
    if '<point' in text and not re.search(r'<points?\b[^>]*>', text, re.DOTALL):
        errors.append('incomplete_point_tag')
    return valid, errors


def validate_xy(points, width, height):
    valid, rejected = [], 0
    for p in points:
        if (len(p) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)
                and 0 <= p[0] <= width and 0 <= p[1] <= height):
            valid.append(list(p))
        else:
            rejected += 1
    return valid, rejected


def parse_molmopoint(extracted, width, height):
    # Official extract_image_points returns (object_id, zero-based image_num, x, y).
    xy, errors = [], []
    for item in extracted:
        if len(item) != 4 or item[1] != 0:
            errors.append('unexpected_image_or_tuple')
            continue
        xy.append([float(item[2]), float(item[3])])
    valid, invalid = validate_xy(xy, width, height)
    errors.extend(['out_of_bounds_coordinate'] * invalid)
    return valid, errors


def freeze(path, value):
    normalized = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text()) != normalized:
        raise ValueError(f'Existing contract differs: {path}; use a new output directory')
    write(path, value)


def prepare(args):
    from PIL import Image
    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    import torch
    import transformers
    torch.set_num_threads(4)
    rows = read(MANIFEST)
    assert len(rows) == 100 and len({r['uid'] for r in rows}) == 100
    assert sum(len(r['gt']) for r in rows) == 9045
    assert all(len(query_specs(r)) == 18 for r in rows)
    for row in rows:
        with Image.open(row['image_path']) as image:
            if (image.width, image.height) != (row['width'], row['height']):
                raise ValueError(f"Image dimension mismatch: {row['uid']}")
    for name, spec in MODELS.items():
        info = HfApi().model_info(spec['repo'], revision=spec['revision'], files_metadata=True)
        files = [dict(name=f.rfilename, size=f.size, sha256=f.lfs.sha256)
                 for f in info.siblings if f.rfilename.endswith('.safetensors')]
        path = model_path(name)
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, padding_side='left')
        extra = dict(return_pointing_metadata=True) if name == 'molmopoint_8b' else {}
        inputs = processor.apply_chat_template(messages(Image.new('RGB', (1024, 1024)), 'Point to: ship.'),
            tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True, **extra)
        report = dict(**spec, weights=files, model_path=str(path),
            downloaded_weight_bytes=sum((path / f['name']).stat().st_size for f in files if (path / f['name']).exists()),
            required_weight_bytes=sum(f['size'] for f in files),
            processor_test={k:list(v.shape) for k,v in inputs.items() if hasattr(v, 'shape')},
            max_crops=processor.image_processor.max_crops,
            torch=torch.__version__, transformers=transformers.__version__)
        freeze(args.output / 'assets' / f'{name}.json', {k:v for k,v in report.items() if k != 'downloaded_weight_bytes'})
        print(json.dumps(report), flush=True)
    frozen = args.output / 'assets/DOTAv2-Balanced100.jsonl'
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if frozen.exists() and sha(frozen) != sha(MANIFEST):
        raise ValueError('Frozen evaluation manifest changed')
    if not frozen.exists():
        shutil.copy2(MANIFEST, frozen)
    freeze(args.output / 'assets/protocol.json', dict(images=100, class_queries=1800, targets=9045,
        manifest_sha256=sha(MANIFEST), prompt='Point to: {class}.', all_classes=True,
        diagnostic='Class-aware one-to-one point-in-HBB, not mask-based public pointing benchmark',
        matching='Existing LA scorer: prediction-order greedy, nearest GT center among unmatched containing HBBs',
        no_nms=True, duplicates_retained=True, no_gt_in_prompts=True,
        source_revision=OFFICIAL_CODE_REVISION,
        caveat='Historical LA checkpoints predate corrected v6; this is not proof of clean unseen-image generalization.'))


def download(args):
    from huggingface_hub import snapshot_download
    identity = json.loads((args.output / 'assets' / f'{args.model}.json').read_text())
    path = model_path(args.model)
    missing = sum(f['size'] for f in identity['weights'] if not (path / f['name']).exists())
    # Preserve a reserve for existing research jobs and downloaded-file verification.
    if shutil.disk_usage(CACHE).free < missing + 15 * 1024**3:
        raise RuntimeError('Insufficient /data2 space for model download plus 15 GiB reserve')
    spec = MODELS[args.model]
    snapshot_download(spec['repo'], revision=spec['revision'], cache_dir=CACHE,
        allow_patterns=['*.json', '*.py', '*.safetensors', '*.txt', '*.jinja', '*.md'], max_workers=2)
    checked = []
    for f in identity['weights']:
        local = path / f['name']
        if local.stat().st_size != f['size'] or sha(local) != f['sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {local}')
        checked.append(f)
    write(args.output / 'assets' / f'{args.model}.download_verified.json',
          dict(**spec, checked=checked, verified=True))
    print(f'{args.model}: all checkpoint shard hashes verified', flush=True)


def inference(args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from refwave.io import read_jsonl_log
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    model_dir = model_path(args.model)
    verification = args.output / 'assets' / f'{args.model}.download_verified.json'
    if not verification.exists():
        raise RuntimeError('Download has not been verified')
    directory = args.output / ('smoke' if args.limit else 'results') / args.model
    directory.mkdir(parents=True, exist_ok=True)
    frozen = args.output / 'assets/DOTAv2-Balanced100.jsonl'
    rows = read(frozen)
    if args.limit:
        rows = rows[:args.limit]
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, padding_side='left')
    contract = dict(**MODELS[args.model], manifest_sha256=sha(frozen), images=len(rows),
        source_sha256=sha(Path(__file__)), budget=args.budget, seed=args.seed,
        prompt='Point to: {class}.', class_batch_size=1, do_sample=False,
        model_dtype='bfloat16', attention='sdpa', max_crops=processor.image_processor.max_crops,
        gpu=torch.cuda.get_device_name(0), official_point_extraction=args.model == 'molmopoint_8b',
        timing='Synchronized processor + image encoder + all generation + parsing, excludes disk IO/loading/scoring',
        latency_caveat='Native serial class queries re-encode the image; not a matched wave/cache implementation ablation',
        warmup_queries=2, native_preprocessing=True)
    freeze(directory / 'contract.json', contract)
    out = directory / 'queries.jsonl'
    expected = {(r['uid'], s['label']) for r in rows for s in query_specs(r)}
    previous = read_jsonl_log(out, repair_tail=True) if out.exists() else []
    keys = [(p['uid'], p['label']) for p in previous]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError('Duplicate or unexpected prediction keys')
    if set(keys) == expected:
        write(directory / 'done.json', dict(complete=True, images=len(rows), queries=len(expected)))
        return
    model = AutoModelForImageTextToText.from_pretrained(model_dir, trust_remote_code=True,
        dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa').eval()
    is_point = args.model == 'molmopoint_8b'
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])

    def one(image, spec):
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        extra = dict(return_pointing_metadata=True) if is_point else {}
        inputs = processor.apply_chat_template(messages(image, spec['prompt']), tokenize=True,
            add_generation_prompt=True, return_tensors='pt', return_dict=True, **extra)
        metadata = inputs.pop('metadata', None)
        input_shapes = {k:list(v.shape) for k,v in inputs.items()}
        inputs = inputs.to('cuda:0')
        options = dict(logits_processor=model.build_logit_processor_from_inputs(inputs)) if is_point else {}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            ids = model.generate(**inputs, max_new_tokens=args.budget, do_sample=False, use_cache=True, **options)
        generated = ids[:, inputs['input_ids'].shape[1]:]
        token_ids = generated[0].tolist()
        raw = processor.post_process_image_text_to_text(generated, skip_special_tokens=False,
                                                        clean_up_tokenization_spaces=False)[0]
        if is_point:
            extracted = model.extract_image_points(raw, metadata['token_pooling'],
                                                   metadata['subpatch_mapping'], metadata['image_sizes'])
            xy, errors = parse_molmopoint(extracted, *image.size)
        else:
            xy, errors = parse_molmo2(raw, *image.size)
        torch.cuda.synchronize()
        latency = time.perf_counter() - start
        return dict(**spec, points=[dict(label=spec['label'], point=p, source='point') for p in xy],
            raw_response=raw, parse_errors=errors, output_token_ids=token_ids,
            generated_tokens=len(token_ids), eos=bool(token_ids and token_ids[-1] in eos),
            budget_saturated=len(token_ids) >= args.budget, input_shapes=input_shapes,
            latency_seconds=latency, peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2)

    with Image.open(rows[0]['image_path']) as opened:
        image = opened.convert('RGB')
    for spec in query_specs(rows[0])[:2]:
        one(image, spec)
    done = set(keys)
    with out.open('a') as handle:
        for row in rows:
            if all((row['uid'], s['label']) in done for s in query_specs(row)):
                continue
            with Image.open(row['image_path']) as opened:
                image = opened.convert('RGB')
            for spec in query_specs(row):
                key = (row['uid'], spec['label'])
                if key in done:
                    continue
                result = dict(uid=row['uid'], **one(image, spec))
                handle.write(json.dumps(result, sort_keys=True) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
                done.add(key)
                write(directory / 'status.json', dict(complete=False, queries=len(done), expected=len(expected),
                    last_uid=row['uid'], last_class=spec['label']))
                print(f"{args.model} {len(done)}/{len(expected)} uid={row['uid']} class={spec['label']} "
                      f"points={len(result['points'])} seconds={result['latency_seconds']:.3f}", flush=True)
    write(directory / 'done.json', dict(complete=True, images=len(rows), queries=len(done)))


def aggregate(rows, queries):
    expected = {(r['uid'], s['label']) for r in rows for s in query_specs(r)}
    indexed = {(q['uid'], q['label']):q for q in queries}
    if len(indexed) != len(queries) or set(indexed) != expected:
        raise ValueError('Incomplete, duplicate or unexpected query outputs')
    predictions = []
    for row in rows:
        answers = [indexed[row['uid'], s['label']] for s in query_specs(row)]
        predictions.append(dict(uid=row['uid'], points=[p for a in answers for p in a['points']],
            answers=[dict(label=a['label'], answer=a['raw_response']) for a in answers],
            box_fallbacks_excluded=0, latency_seconds=sum(a['latency_seconds'] for a in answers),
            peak_gpu_mb=max(a['peak_gpu_mb'] for a in answers)))
    return predictions


def merge(args):
    import numpy as np
    from evaluate_refwave_remaining_tasks import score_points
    from run_locany_pointing_balanced100 import match_points_to_gt, prf
    directory = args.output / ('smoke' if args.limit else 'results') / args.model
    if not (directory / 'done.json').exists():
        raise ValueError('Inference is incomplete')
    rows = read(args.output / 'assets/DOTAv2-Balanced100.jsonl')
    if args.limit:
        rows = rows[:args.limit]
    queries = read(directory / 'queries.jsonl')
    predictions = aggregate(rows, queries)
    metrics, per_image = score_points(rows, predictions)
    # Explicit LA <box>None</box> is not Molmo's absence grammar.
    metrics.pop('absent_correct_explicit_null')
    per_query, density = [], {}
    indexed_rows = {r['uid']:r for r in rows}
    for q in queries:
        row = indexed_rows[q['uid']]
        gt = [g['hbox'] for g in row['gt'] if g['label'] == q['label']]
        match = match_points_to_gt(gt, q['points'])
        counts = {k:match[k] for k in ('tp', 'fp', 'fn')}
        per_query.append(dict(uid=q['uid'], label=q['label'], gt=len(gt), **counts,
            **prf(**counts), latency_seconds=q['latency_seconds'],
            budget_saturated=q['budget_saturated'], parse_errors=q['parse_errors']))
    for row, counts in zip(rows, per_image):
        n = len(row['gt'])
        bucket = '0-19' if n < 20 else ('20-99' if n < 100 else ('100-499' if n < 500 else '500+'))
        b = density.setdefault(bucket, Counter())
        b.update({k:counts[k] for k in ('tp', 'fp', 'fn')})
        p = next(p for p in predictions if p['uid'] == row['uid'])
        b.update(images=1, seconds=p['latency_seconds'], points=len(p['points']))
    query_times = [q['latency_seconds'] for q in queries]
    total_time = sum(query_times)
    metrics.update(contract=json.loads((directory / 'contract.json').read_text()),
        predictions_sha256=sha(directory / 'queries.jsonl'),
        mean_seconds_per_query=float(np.mean(query_times)), p50_seconds_per_query=float(np.median(query_times)),
        p95_seconds_per_query=float(np.percentile(query_times, 95)),
        points_per_second=(metrics['tp']+metrics['fp'])/total_time,
        correct_points_per_second=metrics['tp']/total_time,
        budget_saturated_queries=sum(q['budget_saturated'] for q in queries),
        eos_queries=sum(q['eos'] for q in queries),
        parse_error_queries=sum(bool(q['parse_errors']) for q in queries),
        absent_zero_valid_points=sum(q['gt'] == 0 and q['fp'] == 0 for q in per_query),
        absent_false_points=sum(q['fp'] for q in per_query if q['gt'] == 0),
        zero_points_caveat='Zero parsed points alone is not proof of an explicit valid absence response',
        by_image_density={k:dict(v, **prf(v['tp'],v['fp'],v['fn']),
            points_per_second=v['points']/v['seconds'], correct_points_per_second=v['tp']/v['seconds'])
            for k,v in density.items()})
    write(directory / 'metrics.json', metrics)
    write(directory / 'per_image_metrics.json', per_image)
    write(directory / 'per_query_metrics.json', per_query)
    write(directory / 'predictions.json', predictions)
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('per_class','contract','by_image_density')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'download', 'infer', 'merge'])
    p.add_argument('--model', choices=MODELS, default='molmo2_4b')
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--budget', type=int, default=8192)
    p.add_argument('--seed', type=int, default=20260914)
    args = p.parse_args()
    if args.limit < 0 or args.budget < 1:
        p.error('limit must be nonnegative and budget positive')
    {'prepare':prepare, 'download':download, 'infer':inference, 'merge':merge}[args.mode](args)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'RefWave/src'))
    main()
