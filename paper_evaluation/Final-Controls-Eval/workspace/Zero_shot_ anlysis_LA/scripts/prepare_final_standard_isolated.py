#!/usr/bin/env python3
"""Download and verify only the pinned final standard/class-isolated Stage-2 bests."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import shutil

import evaluate_rex_pointing_full as base

REPO = 'CaseStudyRef/RefWave-Cluster-Runs'
REVISION = '6947859bf4e2d18ad28e389756aa1cfc253a465e'
DOWNLOAD = Path('/data1/shubham/final_std_ciso_stage2_20260916')
OUTPUT = base.PROJECT / 'runs/final_standard_isolated_stage2_20260916'
SPECS = {
    'standard': dict(
        run='standard_full_8gpu_s1e2_s2e2_both_std_ep2_v3_20260914T080021Z_49TbTl',
        sha256='b0c4c248b16de59f0339c2d4658e36de86cdbcd8c81c4b408bd2f9d199fadb83',
        bytes=42216772310, step=1200,
        parent_sha256='d97dc16f4b725da3da638f8450e5df1c775f6f7f529d2aa0ce9aa83eb48ba716'),
    'class_isolated': dict(
        run='class_isolated_full_8gpu_s1e2_s2e2_both_both_ft_ciso_ep2_v3_20260914T125647Z_1zsyt7',
        sha256='a4655c4e1138920162f1416b5197c73a32e399afdcaa50be3a171b97d8d43c8b',
        bytes=42216772310, step=1200,
        parent_sha256='640527fe06312f66ce7b7dfdaf1e7ad90d05e5dc0e35871f6dd051ab7c2f3dd7'),
}


def checkpoint(method):
    return DOWNLOAD / 'runs' / SPECS[method]['run'] / 'stage2/best.pt'


def read(path):
    return json.loads(Path(path).read_text())


def validate_metadata(method, configs, completions, curves):
    spec = SPECS[method]
    for stage in ('stage1', 'stage2'):
        c, done = configs[stage], completions[stage]
        if (c['method'], c['tuning'], c['stage'], c['epochs'], c['world_size']) != (
                method, 'full', stage, 2, 8):
            raise ValueError('Unexpected method, tuning, stage, epochs or world size')
        expected = c['epochs'] * math.ceil(c['records'] / c['global_batch'])
        if c['steps'] != expected or done['steps'] != expected:
            raise ValueError('Incomplete epoch schedule')
        if done['records'] != c['records'] * c['epochs'] or done['epochs'] != c['epochs']:
            raise ValueError('Incomplete record passes')
        if not curves[stage] or any(not math.isfinite(r['detection_grounding_macro_loss'])
                                    for r in curves[stage]):
            raise ValueError('Missing/nonfinite validation losses')
    if configs['stage2']['init_checkpoint_sha256'] != spec['parent_sha256']:
        raise ValueError('Stage-2 parent is not the pinned Stage-1 best')
    if configs['stage2']['image_token_limit'] != 6000:
        raise ValueError('Unexpected training image-token limit')
    best = min(curves['stage2'], key=lambda row: row['detection_grounding_macro_loss'])
    if best['step'] != spec['step']:
        raise ValueError('Pinned best does not match validation selection')
    return best


def validate_saved(saved, method, configs, completions, curves, tied_head):
    from refwave.train import SCHEMA, cursor_after_step
    best = validate_metadata(method, configs, completions, curves)
    c = configs['stage2']
    if saved['schema'] != SCHEMA or saved['config'] != c:
        raise ValueError('Checkpoint/config mismatch')
    if saved['step'] != best['step'] or saved['best_loss'] != best['detection_grounding_macro_loss']:
        raise ValueError('Checkpoint is not the best validation model')
    if saved['cursor'] != cursor_after_step(saved['step'], c['records'], c['global_batch']):
        raise ValueError('Checkpoint cursor mismatch')
    state = saved['trainable_state']
    components = Counter()
    for name, value in state.items():
        if 'lora_' in name:
            raise ValueError('LoRA weights in full checkpoint')
        components[name.split('.')[0]] += value.numel()
    if sum(components.values()) != c['trainable']['trainable_parameters']:
        raise ValueError('Incomplete full-model state')
    if any(not components[k] for k in ('vision_model', 'mlp1', 'language_model')):
        raise ValueError('Missing trained model component')
    if ('language_model.lm_head.weight' not in state and not
            (tied_head and 'language_model.model.embed_tokens.weight' in state)):
        raise ValueError('Missing trained output head')
    return best, dict(components)


def metadata(output):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    selections = {}
    for method, spec in SPECS.items():
        remote = f"runs/{spec['run']}/stage2/best.pt"
        info = api.get_paths_info(REPO, paths=[remote], repo_type='dataset', revision=REVISION)[0]
        if info.size != spec['bytes'] or info.lfs.sha256 != spec['sha256']:
            raise ValueError('Pinned remote checkpoint identity changed')
        configs, completions, curves = {}, {}, {}
        for stage in ('stage1', 'stage2'):
            for name in ('config.json', 'done.json', 'validation_curve.jsonl'):
                hf_hub_download(REPO, f"runs/{spec['run']}/{stage}/{name}",
                    repo_type='dataset', revision=REVISION, local_dir=DOWNLOAD)
            directory = DOWNLOAD / 'runs' / spec['run'] / stage
            configs[stage] = read(directory/'config.json')
            completions[stage] = read(directory/'done.json')
            curves[stage] = base.common.read(directory/'validation_curve.jsonl')
        best = validate_metadata(method, configs, completions, curves)
        selections[method] = dict(spec, method=method, repo=REPO, revision=REVISION,
            checkpoint=str(checkpoint(method)), criterion='minimum validation detection_grounding_macro_loss',
            validation=best, configs=configs, completions=completions, curves=curves)
    base.freeze(output/'selection.json', selections)
    assets = output/'assets'
    assets.mkdir(parents=True, exist_ok=True)
    for suite in base.SUITES:
        src = base.OUTPUT/'assets'/f'{suite}.jsonl'
        summary = read(src.with_suffix('.json'))
        if base.common.sha(src) != summary['sha256']:
            raise ValueError('Frozen benchmark manifest changed')
        dst = assets/src.name
        if dst.exists() and base.common.sha(dst) != summary['sha256']:
            raise ValueError('Output benchmark manifest differs')
        if not dst.exists():
            shutil.copy2(src, dst)
        base.freeze(dst.with_suffix('.json'), summary)
    return selections


def download(output):
    import torch
    from huggingface_hub import hf_hub_download
    selections = metadata(output)
    required = sum(s['bytes'] for k,s in SPECS.items() if not checkpoint(k).exists())
    if shutil.disk_usage(DOWNLOAD).free < required + 8*1024**3:
        raise RuntimeError('Insufficient download space; no existing files were deleted')
    tied = read(base.BASE/'config.json')['text_config'].get('tie_word_embeddings', False)
    verified = {}
    for method, selected in selections.items():
        print('DOWNLOADING', method, 'Stage-2 best only', selected['bytes'], flush=True)
        path = Path(hf_hub_download(REPO, f"runs/{selected['run']}/stage2/best.pt",
            repo_type='dataset', revision=REVISION, local_dir=DOWNLOAD))
        print('HASHING', method, flush=True)
        if path.stat().st_size != selected['bytes'] or base.common.sha(path) != selected['sha256']:
            raise ValueError('Downloaded checkpoint failed SHA256/size check')
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        best, components = validate_saved(saved, method, selected['configs'], selected['completions'],
                                          selected['curves'], tied)
        verified[method] = dict(path=str(path), sha256=selected['sha256'], bytes=path.stat().st_size,
            mtime_ns=path.stat().st_mtime_ns, step=best['step'], components=components,
            verified=True, optimizer_state='embedded in upstream file; unused for inference')
        del saved
        gc.collect()
        base.common.write(output/f'{method}.checkpoint_verified.json', verified[method])
        print('VERIFIED', method, 'step', best['step'], flush=True)
    base.common.write(output/'checkpoints_ready.json', dict(complete=True, models=verified))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--metadata-only', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    DOWNLOAD.mkdir(parents=True, exist_ok=True)
    if args.metadata_only:
        selected = metadata(args.output)
        print(json.dumps({k:dict(step=v['step'], loss=v['validation']['detection_grounding_macro_loss'])
                          for k,v in selected.items()}, indent=2))
    else:
        download(args.output)


if __name__ == '__main__':
    main()
