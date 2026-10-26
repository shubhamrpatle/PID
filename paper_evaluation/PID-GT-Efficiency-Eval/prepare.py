#!/usr/bin/env python3
"""Reuse the completed remote baseline manifests and the existing final PID weights."""
import argparse
from pathlib import Path

from pid_common import (HERE, BASELINES, PROTOCOL, SOFTWARE_KEYS, DATASETS, CHECKPOINTS,
                        require, read, rows, sha, digest, freeze, query_only)


def prepare(args):
    from common import semantic_rows
    from prepare import SEMANTIC
    from evaluate import finished as baseline_finished
    prior = read(args.baseline_results / 'plan.json')
    require(prior['protocol'] == 'rs_gt_categories_la_native_v3', 'Use the completed LA/Rex v3 baseline results')
    controls = (args.controls or Path(prior['controls'])).resolve()
    require((controls / 'workspace').is_dir(), f'Missing existing controls runtime: {controls}')
    sources = {str(p.resolve()): sha(p) for p in HERE.glob('*.py')}
    # Reuse the exact interfaces and scorer used for the existing baseline tables.
    for name in ('common.py', 'models.py', 'parsing.py', 'native_parser.py', 'scoring.py',
                 'gpu_pool.py', 'evaluate.py', 'prepare.py',
                 'evidence/other_metric.py', 'evidence/inference_grounding_ddp.py'):
        file = BASELINES / name
        matches = [h for p, h in prior['sources'].items() if p.endswith('/LA-Rex-Protocol-Eval-v3/' + name)]
        require(matches and len(set(matches)) == 1 and sha(file) == matches[0], f'Baseline code differs: {file}')
        sources[str(file)] = matches[0]
    count = 0
    for old, expected in prior['sources'].items():
        marker = '/Final-Controls-Eval/workspace/'
        if marker not in old:
            continue
        file = controls / 'workspace' / old.split(marker, 1)[1]
        require(sha(file) == expected, f'Controls code differs from baseline: {file}')
        sources[str(file)] = expected
        count += 1
    require(count > 0, 'Baseline plan has no frozen controls source files')
    hardware = read(args.baseline_results / 'hardware.json')
    software = read(args.baseline_results / 'software_shared_vlm.json')
    datasets = {}
    baseline_completions = {}
    for ds, (n, targets, _) in DATASETS.items():
        info = dict(prior['datasets'][ds])
        file = Path(info['path'])
        require(sha(file) == info['sha256'], 'Baseline manifest changed: ' + ds)
        values = rows(file)
        require(len(values) == len({r['uid'] for r in values}) == n, 'Wrong or duplicate manifest records')
        require(sum(len(r['gt']) for r in values) == targets, 'GT count changed')
        require(digest(semantic_rows(values)) == SEMANTIC[ds], 'Frozen query/annotation contents differ')
        require(sum(len(query_only(r)['class_specs']) for r in values) == info['gt_present_queries'], 'Query count differs')
        # A global summary can be absent when unrelated pointing jobs are pending.
        # Verify the completed detection shards themselves, not the presence of a report.
        for model in ('base_la', 'standard', 'rex_omni'):
            for shard in range(prior['shards']):
                require(baseline_finished(args.baseline_results, model, 'boxes', ds, shard,
                    prior['shards'], False, prior, {ds: values}), f'Incomplete baseline: {model}/{ds}/{shard}')
                done = args.baseline_results / 'predictions' / model / 'boxes' / ds / f'shard{shard:02d}of{prior["shards"]:02d}.done.json'
                baseline_completions[str(done)] = sha(done)
        datasets[ds] = info
        print(f'{ds}: {n} images, {targets} GT boxes, {info["gt_present_queries"]} category queries', flush=True)
    ready = read(args.pid_results / 'checkpoints_ready.json')['models']['refwave']
    selected = read(args.pid_results / 'selection.json')['refwave']
    _, expected_hash, expected_step = CHECKPOINTS['pid']
    checkpoint = (args.checkpoint or Path(ready['path'])).resolve()
    require(selected['sha256'] == expected_hash and selected['step'] == expected_step, 'Wrong selected PID checkpoint')
    print('Checking existing final PID checkpoint (one-time hash; no download):', checkpoint, flush=True)
    require(sha(checkpoint) == expected_hash, 'Checkpoint does not match final PID Stage-2 best step 2400')
    base = controls / 'workspace/Zero_shot_ anlysis_LA/model/LocateAnything-3B/checkpoint'
    inventory = read(controls / 'transfer_inventory.json')['files']
    files = {}
    prefix = str(base.relative_to(controls)) + '/'
    for entry in inventory:
        if entry['path'].startswith(prefix):
            file = controls / entry['path']
            require(sha(file) == entry['sha256'], 'Base model file changed: ' + str(file))
            files[str(file)] = dict(sha256=entry['sha256'], bytes=file.stat().st_size, mtime_ns=file.stat().st_mtime_ns)
    require(files, 'Missing base model inventory')
    files[str(checkpoint)] = dict(sha256=expected_hash, bytes=checkpoint.stat().st_size,
                                  mtime_ns=checkpoint.stat().st_mtime_ns)
    plan = dict(protocol=PROTOCOL, baseline_protocol=prior['protocol'],
                baseline_results=str(args.baseline_results), baseline_plan_sha256=sha(args.baseline_results / 'plan.json'),
                baseline_detection_completions=baseline_completions,
                controls=str(controls), datasets=datasets, sources=sources, hardware=hardware,
                expected_software={k: software.get(k) for k in SOFTWARE_KEYS},
                models={'pid': dict(base=str(base), path=str(checkpoint), sha256=expected_hash,
                                    step=expected_step, files=files)},
                budget=prior['budget'], generation=prior['generation']['la'],
                shards=args.shards, warmup_records=prior['warmup_records'], image_token_limit=6000,
                additional_nms=False, evaluator_max_detections=None, max_addresses=None,
                scope='GT-present category names only; no GT counts, geometry or oracle addresses enter inference',
                pairing='Same image/GPU/checkpoint, independent full pipeline calls; alternate packed-first/sequential-first',
                discovery='All requested categories concurrent in both modes; autoregressive within each category',
                extent='Same atomic kernel: all addresses packed versus one address per forward',
                timing=prior['timing'], primary_parsing='LA native', strict_sensitivity=True,
                metric='Class-aware matching; image-mean P/R harmonic F1; mF1 over 0.50:0.05:0.95')
    require(1 <= args.shards <= 8, 'Use 1-8 fixed shards')
    require(plan['generation'] == dict(temperature=.7, top_p=.9, repetition_penalty=1.1, mode='hybrid'),
            'Baseline sampling settings changed')
    args.output.mkdir(parents=True, exist_ok=True)
    freeze(args.output / 'plan.json', plan)
    print(f'Prepared {2 * args.shards} paired jobs (two extent modes each). Budget inherited: {plan["budget"]}.', flush=True)
    print('Output:', args.output, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-results', type=Path, default=BASELINES / 'results')
    p.add_argument('--pid-results', type=Path, default=HERE.parent / 'Final-RefWave-Eval/results')
    p.add_argument('--controls', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, default=HERE / 'results')
    p.add_argument('--shards', type=int, default=4)
    args = p.parse_args()
    args.baseline_results, args.pid_results, args.output = (p.resolve() for p in
        (args.baseline_results, args.pid_results, args.output))
    prepare(args)


if __name__ == '__main__':
    main()
