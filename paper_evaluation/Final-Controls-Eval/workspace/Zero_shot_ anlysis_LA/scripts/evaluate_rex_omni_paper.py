#!/usr/bin/env python3
"""Pinned official Rex-Omni detection on the existing paper manifests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import evaluate_paper_comparators as common

CODE = common.ROOT / 'external/Rex-Omni'
CACHE = Path('/data1/shubham/rex_omni_hf')
REPO = 'IDEA-Research/Rex-Omni'
REVISION = '0e5693d24657f6c0e091008dd6809bb1bd28988c'
CODE_REVISION = '6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e'


def download():
    from huggingface_hub import snapshot_download
    return snapshot_download(REPO, revision=REVISION, cache_dir=CACHE,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'], max_workers=2)


def parse_boxes(result, specs, width, height):
    import evaluate_locateanything_lae_style as det
    aliases = det.build_alias_lookup(specs)
    classes = {s['canonical'] for s in specs}
    predictions, rejected = [], []
    for label, items in result.items():
        canonical = det.canonicalize_label(label, aliases, classes)
        for item in items:
            if canonical is None or item['type'] != 'box':
                rejected.append(dict(label=label, **item))
                continue
            box = det.clip_hbox_to_image(item['coords'], width, height)
            if box is None:
                rejected.append(dict(label=label, **item))
                continue
            predictions.append(dict(label=canonical, hbox=box, score=1.0))
    return predictions, rejected


def infer(args):
    import torch
    import transformers
    from PIL import Image
    from refwave.io import read_jsonl_log
    sys.path.insert(0, str(CODE))
    from rex_omni import RexOmniWrapper
    commit = subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != CODE_REVISION:
        raise ValueError('Official Rex-Omni code revision changed')
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    manifest = common.ASSETS / (args.suite + '.jsonl')
    rows = common.read(manifest)
    if args.limit:
        rows = rows[:args.limit]
    folder = args.output / args.model / args.suite
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'shard00of01.jsonl'
    contract = dict(model=dict(repo=REPO, revision=REVISION, official_code_revision=commit),
        source_sha256=common.sha(__file__), official_sources={str(p):common.sha(p) for p in sorted((CODE/'rex_omni').glob('*.py'))},
        suite=args.suite, manifest_sha256=common.sha(manifest), limit=args.limit,
        shard_index=0, shard_count=1, budget=args.budget,
        generation='official greedy: temperature0/top_p.05/top_k1/repetition1.05',
        image_protocol='official processor: min_pixels=16*28^2, max_pixels=2560*28^2; original image',
        prompt_protocol='official detection template; all requested classes; no GT presence input',
        parser='official bins divided by 999; same HBB clipping/label aliases as other detectors',
        timing='CUDA synchronized preprocessing+generation+parsing; excludes disk/loading/warmup/scoring',
        torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0),
        added_nms=False, evaluator_max_detections=None, warmup_queries=2)
    cp = folder / 'shard00of01.contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Rex-Omni contract changed; use a new output directory')
    common.write(cp, contract)
    previous = read_jsonl_log(path, repair_tail=True) if path.exists() else []
    done = {r['uid'] for r in previous}
    expected = {r['uid'] for r in rows}
    if len(done) != len(previous) or not done <= expected:
        raise ValueError('Unexpected or duplicate saved IDs')
    if done != expected:
        model = RexOmniWrapper(download(), backend='transformers', max_tokens=args.budget,
            temperature=0, top_p=.05, top_k=1, repetition_penalty=1.05,
            attn_implementation='sdpa', device_map='cuda:0', torch_dtype=torch.bfloat16)

        def one(row, image):
            specs = common.query_only(row)['class_specs']
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = model.inference(images=image, task='detection', categories=[s['prompt'] for s in specs])[0]
            if not result['success']:
                raise RuntimeError(result.get('error', 'Rex-Omni inference failed'))
            predictions, rejected = parse_boxes(result['extracted_predictions'], specs, *image.size)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            return dict(predictions=predictions, rejected_labels=rejected,
                raw_response=result['raw_output'], prompt=result['prompt'],
                generated_tokens=result['num_output_tokens'], budget_saturated=result['num_output_tokens'] >= args.budget,
                latency_seconds=elapsed, peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2)

        for row in rows[:2]:
            with Image.open(row['image_path']) as image:
                one(row, image.convert('RGB'))
        with path.open('a') as stream:
            for row in rows:
                if row['uid'] in done:
                    continue
                with Image.open(row['image_path']) as opened:
                    image = opened.convert('RGB')
                if image.size != (row['width'], row['height']):
                    raise ValueError('Image/manifest dimensions differ')
                result = one(row, image)
                result.update({k:row[k] for k in ('uid','dataset','image_id','task','suite')})
                stream.write(json.dumps(result)+'\n')
                stream.flush()
                done.add(row['uid'])
                common.write(folder/'shard00of01.status.json', dict(records=len(done), expected=len(rows)))
                print(args.suite, len(done), '/', len(rows), 'boxes',len(result['predictions']), flush=True)
    common.write(folder/'shard00of01.done.json', dict(records=len(rows)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['download','infer','merge'])
    p.add_argument('--suite', choices=['DIOR','DOTAv2','LAE-80C'], default='DIOR')
    p.add_argument('--output', type=Path, default=common.OUTPUT)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--budget', type=int, default=8192)
    args = p.parse_args()
    args.model, args.shard_count = 'rexomni', 1
    if args.limit < 0 or args.budget < 1:
        p.error('Invalid limit/budget')
    if args.mode == 'download':
        print(download())
    elif args.mode == 'infer':
        infer(args)
    else:
        common.merge(args)


if __name__ == '__main__':
    sys.path.insert(0, str(common.ROOT/'RefWave/src'))
    main()
