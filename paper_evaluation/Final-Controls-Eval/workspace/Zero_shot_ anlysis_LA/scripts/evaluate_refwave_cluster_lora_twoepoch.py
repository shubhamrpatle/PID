#!/usr/bin/env python3
"""Evaluate the pinned two-epoch cluster LoRA best, without training or exports."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import evaluate_refwave_cluster_stage2 as ref
import evaluate_refwave_cluster_full_stage2 as full
import evaluate_refwave_remaining_tasks as pointing
import run_refwave_inference_ablations as ablation
from refwave.io import append_jsonl, read_jsonl_log

REPO = 'CaseStudyRef/RefWave-Cluster-Runs'
REVISION = '13d603da48209a666f4f0801395df65e447b427c'
RUN = 'refwave_lora_4gpu_20260910T013749Z_tYtPZN'
DOWNLOAD = Path('/data2/shubham/refwave_cluster_lora_2epoch_stage2_best_20260911')
CHECKPOINT = DOWNLOAD / 'runs' / RUN / 'stage2/best.pt'
SHA = '0facbb368eae5f721bcbe175a56e69f70633e6b9ec19d2c8f2b05b7c2e8299d1'
PARENT_SHA = '732b9ab5770d4c449a81c4c010af285d686a216e3e5418fc85c976224e8356d9'
OUTPUT = ref.PROJECT / 'runs/refwave_cluster_lora_2epoch_stage2_best_20260911'
SUITES = ('DOTAv2-Balanced100', 'DIOR-RSVG-Unseen100', 'DOTAv2-Pointing100',
          'DIOR', 'DOTAv2', 'DIOR-RSVG', 'VRSBench-VG', 'LAE-80C')


def validate_selection(saved, configs, completions, curves):
    from refwave.train import SCHEMA, cursor_after_step
    for stage in ('stage1', 'stage2'):
        config, done = configs[stage], completions[stage]
        if (config['epochs'] != 2 or config['method'] != 'refwave'
                or config['tuning'] != 'lora' or config['stage'] != stage):
            raise ValueError('Expected a two-epoch RefWave LoRA stage')
        steps = config['epochs'] * math.ceil(config['records'] / config['global_batch'])
        if config['steps'] != steps or done['steps'] != steps:
            raise ValueError('Stage completion step count differs')
        if done['records'] != config['records'] * 2 or done['epochs'] != 2:
            raise ValueError('Stage did not complete both record passes')
    config = configs['stage2']
    best = min(curves['stage2'], key=lambda r: r['detection_grounding_macro_loss'])
    if saved['schema'] != SCHEMA or saved['config'] != config:
        raise ValueError('Checkpoint/config mismatch')
    if saved['step'] != best['step'] or saved['best_loss'] != best['detection_grounding_macro_loss']:
        raise ValueError('Checkpoint is not the minimum validation-loss checkpoint')
    if saved['cursor'] != cursor_after_step(saved['step'], config['records'], config['global_batch']):
        raise ValueError('Checkpoint record cursor mismatch')
    if config['init_checkpoint_sha256'] != PARENT_SHA:
        raise ValueError('Stage-1 parent does not match the pinned uploaded best')
    return best


def prepare(output):
    import torch
    from huggingface_hub import snapshot_download
    files = [f'runs/{RUN}/{s}/{f}' for s in ('stage1', 'stage2')
             for f in ('config.json', 'done.json', 'validation_curve.jsonl')]
    files.append(f'runs/{RUN}/stage2/best.pt')
    snapshot_download(REPO, repo_type='dataset', revision=REVISION, local_dir=DOWNLOAD,
                      allow_patterns=files, max_workers=2)
    if ref.sha256_file(CHECKPOINT) != SHA:
        raise ValueError('Checkpoint differs from pinned remote LFS SHA256')
    configs, completions, curves = {}, {}, {}
    for stage in ('stage1', 'stage2'):
        directory = CHECKPOINT.parent.parent / stage
        configs[stage] = json.loads((directory / 'config.json').read_text())
        completions[stage] = json.loads((directory / 'done.json').read_text())
        curves[stage] = ref.read_jsonl(directory / 'validation_curve.jsonl')
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
    best = validate_selection(saved, configs, completions, curves)
    state = saved['trainable_state']
    if sum(t.numel() for t in state.values()) != configs['stage2']['trainable']['trainable_parameters']:
        raise ValueError('Incomplete trainable parameter state')
    if not any('lm_head' in name for name in state) or not any('lora_' in name for name in state):
        raise ValueError('Missing trained LoRA weights or complete output head')
    if configs['stage2']['image_token_limit'] != 6000:
        raise ValueError('Unexpected training image budget')
    selection = dict(repo=REPO, revision=REVISION, run=RUN, checkpoint=str(CHECKPOINT), sha256=SHA,
        bytes=CHECKPOINT.stat().st_size, step=saved['step'], cursor=saved['cursor'],
        selected_stage2_epoch=saved['cursor'] / configs['stage2']['records'],
        criterion='minimum validation detection_grounding_macro_loss', validation=best,
        configs=configs, completions=completions,
        stage1_best=min(curves['stage1'], key=lambda r: r['detection_grounding_macro_loss']))
    del saved, state
    output.mkdir(parents=True, exist_ok=True)
    old = json.loads((ref.OUTPUT / 'checkpoints.json').read_text())['selections']['refwave']['config']
    selection['prior_stage2_config_differences'] = {
        k: dict(previous=old.get(k), current=v) for k, v in configs['stage2'].items() if old.get(k) != v}
    sources = [Path(__file__), Path(ref.__file__), Path(full.__file__), Path(ablation.__file__),
               Path(pointing.__file__), Path(ref.detection.__file__), Path(ref.grounding.__file__),
               ref.PROJECT / 'scripts/run_locany_pointing_balanced100.py',
               ref.PROJECT / 'scripts/evaluate_pixel_pivr_sequential_wave_balanced100.py',
               ref.ROOT / 'Pixel-PIVR/tools/evaluate_class_addressed_wave.py']
    for directory in (ref.ROOT/'Pixel-PIVR/src/pixel_pivr', ref.ROOT/'RefWave/src/refwave',
                      ref.ROOT/'zero_shot_analysis', ref.BASE_MODEL):
        sources.extend(directory.glob('*.py'))
    sources.extend(ref.BASE_MODEL.glob('*.json'))
    plan = dict(selection=selection, sources={str(p.resolve()): ref.sha256_file(p) for p in sources},
        manifests={}, image_token_limit=6000, class_wave=18, box_wave=200,
        temperature=.7, top_p=.9, repetition_penalty=1.1, training=False,
        protocol='Same frozen manifests and decoding as the prior inference ablations; no GT in inference.',
        raw_policy='All parsed HBBs; duplicate points retained; no containment or NMS.',
        filtered_policy='Containment, deduplication+containment, and class-aware NMS0.5 reported separately.',
        scoring='Uncapped class-aware greedy HBB matching; grounding first box; pointing point-in-GT-HBB.',
        timing='CUDA synchronized preprocessing, decoding and parsing; excludes disk IO, loading, warmup and scoring.',
        caveats=['Pointing is a derived diagnostic, not an official standalone benchmark.',
                 'DIOR and DOTAv2 use existing LAE-style evaluation releases, not the entire original datasets.',
                 'GPU sharding measures per-worker inference, not isolated-node latency.',
                 'Best validation checkpoint precedes the final step; only the training run completed two epochs.',
                 'Single training seed; no new claim of seed reliability or OBB performance.'])
    original = json.loads((ref.OUTPUT / 'assets/manifest_summary.json').read_text())
    for suite in SUITES:
        manifest_suite = 'DOTAv2-Balanced100' if suite == 'DOTAv2-Pointing100' else suite
        source = ref.OUTPUT / 'assets' / f'{manifest_suite}.jsonl'
        digest = ref.sha256_file(source)
        if digest != original[manifest_suite]['sha256']:
            raise ValueError(f'Frozen manifest changed: {suite}')
        rows = ref.read_jsonl(source)
        if len(rows) != len({r['uid'] for r in rows}) or len(rows) != original[manifest_suite]['records']:
            raise ValueError('Manifest count or uniqueness mismatch')
        for row in rows:
            if not Path(row['image_path']).is_file():
                raise FileNotFoundError(row['image_path'])
        destination = output / 'assets' / f'{suite}.jsonl'
        destination.parent.mkdir(exist_ok=True)
        if destination.exists() and ref.sha256_file(destination) != digest:
            raise ValueError('Existing output manifest differs')
        if not destination.exists():
            shutil.copy2(source, destination)
        plan['manifests'][suite] = dict(records=len(rows), sha256=digest,
                                       budget=8192 if suite == 'DOTAv2-Pointing100' or '100' not in suite else 2048)
    path = output / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Output plan differs; use a fresh output directory')
    for source in sources:
        dest = output / 'source_snapshot' / source.resolve().relative_to(ref.ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(source, dest)
    ref.atomic_json(path, plan)
    ref.atomic_json(output / 'checkpoint.json', selection)
    print(json.dumps(dict(step=selection['step'], validation=best, manifests=plan['manifests']), indent=2), flush=True)


def load_worker():
    import torch
    from pixel_pivr.worker import LocateAnythingPixelPIVRWorker
    from refwave.model import configure_lora_head
    from refwave.train import restore
    worker = LocateAnythingPixelPIVRWorker(ref.BASE_MODEL, device='cuda:0')
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
    configure_lora_head(worker.model, saved['config']['lora_rank'])
    restore(worker.model, saved['trainable_state'])
    worker.model.language_model = worker.model.language_model.merge_and_unload()
    worker.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    worker.processor.image_processor.in_token_limit = 6000
    del saved
    gc.collect()
    torch.cuda.empty_cache()
    return worker


def import_predictions(output, source, gpus):
    """Repartition completed rows by UID; keep their original timing provenance."""
    source, output = source.resolve(), output.resolve()
    if source == output or len(set(gpus)) != len(gpus):
        raise ValueError('Migration requires a new output directory and distinct GPUs')
    previous = json.loads((source/'plan.json').read_text())
    current = json.loads((output/'plan.json').read_text())
    for key in ('selection', 'manifests', 'image_token_limit', 'class_wave', 'box_wave',
                'temperature', 'top_p', 'repetition_penalty', 'raw_policy', 'scoring'):
        if previous[key] != current[key]:
            raise ValueError(f'Cannot import predictions with a different {key}')
    # Inference and parsers must be byte-identical; only this orchestration file changed.
    for path, sha in previous['sources'].items():
        if Path(path).resolve() != Path(__file__).resolve() and current['sources'].get(path) != sha:
            raise ValueError(f'Inference source differs: {path}')
    previous_plan_sha = ref.sha256_file(source/'plan.json')
    current_plan_sha = ref.sha256_file(output/'plan.json')
    provenance = dict(source=str(source), source_plan_sha256=previous_plan_sha,
                      target_plan_sha256=current_plan_sha, target_gpus=gpus, suites={}, files={})
    if (output/'prediction_import.json').exists():
        raise ValueError('Predictions already imported; resume the queue directly')
    for suite in SUITES:
        rows = ref.read_jsonl(output/'assets'/f'{suite}.jsonl')
        indexed = {}
        for path in sorted((source/'benchmarks'/suite).glob('shard*.jsonl')):
            contract = json.loads(path.with_suffix('.contract.json').read_text())
            if (contract['plan_sha256'] != previous_plan_sha or contract['checkpoint_sha256'] != SHA
                    or contract['manifest_sha256'] != current['manifests'][suite]['sha256']
                    or contract['suite'] != suite or contract['limit'] != 0):
                raise ValueError(f'Invalid source shard contract: {path}')
            allowed = {r['uid'] for r in rows[contract['shard_index']::contract['shard_count']]}
            for prediction in read_jsonl_log(path):
                uid = prediction['uid']
                if uid not in allowed or uid in indexed:
                    raise ValueError('Duplicate or foreign source prediction')
                indexed[uid] = dict(prediction, resumed_from=dict(path=str(path),
                    visible_devices=contract['visible_devices'], gpu=contract['gpu']))
            provenance['files'][str(path)] = ref.sha256_file(path)
        provenance['suites'][suite] = len(indexed)
        for rank, gpu in enumerate(gpus):
            shard_rows = rows[rank::len(gpus)]
            imported = [indexed[r['uid']] for r in shard_rows if r['uid'] in indexed]
            if not imported:
                continue
            directory = output/'benchmarks'/suite
            directory.mkdir(parents=True, exist_ok=True)
            path = directory/f'shard{rank:02d}of{len(gpus):02d}.jsonl'
            if path.exists():
                raise ValueError(f'Refusing to overwrite predictions: {path}')
            models = {r['resumed_from']['gpu'] for r in imported}
            target = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=name',
                                               '--format=csv,noheader'], text=True).strip()
            if models != {target}:
                raise ValueError('Migration would mix different GPU models in latency measurements')
            contract = dict(plan_sha256=current_plan_sha, suite=suite, shard_index=rank,
                shard_count=len(gpus), limit=0, gpu=target, visible_devices=str(gpu),
                checkpoint_sha256=SHA, manifest_sha256=current['manifests'][suite]['sha256'], warmup=2)
            ref.atomic_json(path.with_suffix('.contract.json'), contract)
            for row in imported:
                append_jsonl(path, row)
            if len(imported) == len(shard_rows):
                ref.atomic_json(path.with_suffix('.done.json'), dict(records=len(imported), smoke=False))
    ref.atomic_json(output/'prediction_import.json', provenance)
    print('PRESERVED PREDICTIONS', json.dumps(provenance['suites']), flush=True)


def infer_pointing(worker, decoders, row, image):
    import torch
    # These branches use only the frozen vocabulary and stable IDs, never GT.
    branches = pointing.point_branches(row)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    answers, stats = decoders[0].generate(image, branches, max_new_tokens=8192,
        temperature=.7, top_p=.9, repetition_penalty=1.1, generation_mode='hybrid')
    points, fallbacks = pointing.parse_points(answers, *image.size)
    torch.cuda.synchronize()
    return dict(uid=row['uid'], points=points, answers=answers, box_fallbacks_excluded=fallbacks,
        latency_seconds=time.perf_counter()-start,
        peak_gpu_mb=max(torch.cuda.max_memory_allocated()/1024**2, stats.get('peak_gpu_mb', 0)), stats=stats)


def worker(args):
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    plan = json.loads((args.output / 'plan.json').read_text())
    if any(ref.sha256_file(Path(p)) != sha for p, sha in plan['sources'].items()):
        raise ValueError('Evaluation source changed after preparation')
    model = load_worker()
    decoders = ref.make_decoders(model, optimized=True)
    for suite in args.suites:
        manifest = args.output / 'assets' / f'{suite}.jsonl'
        if ref.sha256_file(manifest) != plan['manifests'][suite]['sha256']:
            raise ValueError('Evaluation manifest changed')
        rows = ref.read_jsonl(manifest)[args.shard_index::args.shard_count]
        if args.limit:
            rows = rows[:args.limit]
        directory = args.output / ('smoke' if args.limit else 'benchmarks') / suite
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'shard{args.shard_index:02d}of{args.shard_count:02d}.jsonl'
        contract = dict(plan_sha256=ref.sha256_file(args.output/'plan.json'), suite=suite,
            shard_index=args.shard_index, shard_count=args.shard_count, limit=args.limit,
            gpu=torch.cuda.get_device_name(0), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            checkpoint_sha256=SHA, manifest_sha256=plan['manifests'][suite]['sha256'], warmup=2,
            round2_execution=ref.round2_execution_mode() if suite != 'DOTAv2-Pointing100' else None)
        cp = path.with_suffix('.contract.json')
        if cp.exists() and json.loads(cp.read_text()) != contract:
            raise ValueError('Cannot append predictions with a different contract')
        ref.atomic_json(cp, contract)
        existing = read_jsonl_log(path, repair_tail=True) if path.exists() else []
        done = {p['uid'] for p in existing}
        if len(done) != len(existing) or not done <= {r['uid'] for r in rows}:
            raise ValueError('Duplicate or foreign prediction IDs')
        config = dict(budget=plan['manifests'][suite]['budget'], class_wave=18, box_wave=200)

        def one(row):
            with Image.open(row['image_path']) as opened:
                image = opened.convert('RGB')
            if image.size != (int(row['width']), int(row['height'])):
                raise ValueError('Image dimensions differ from frozen manifest')
            if suite == 'DOTAv2-Pointing100':
                return infer_pointing(model, decoders, row, image)
            result = ablation.infer_one(model, decoders, ref.inference_query(row), image, 'refwave', config)
            result.update(image_id=row['image_id'], dataset=row['dataset'], method='refwave_lora_2epoch')
            if row['task'] == 'grounding':
                result.update(sample_id=row['sample_id'], query=row['query'])
            return result

        if len(done) != len(rows):
            for row in rows[:2]:
                one(row)
            for i, row in enumerate(rows, 1):
                if row['uid'] in done:
                    continue
                result = one(row)
                append_jsonl(path, result)
                print(f"{suite} shard={args.shard_index} {i}/{len(rows)} "
                      f"seconds={result['latency_seconds']:.3f}", flush=True)
        ref.atomic_json(path.with_suffix('.done.json'), dict(records=len(rows), smoke=bool(args.limit)))
    print('WORKER COMPLETE', args.shard_index, flush=True)


def merge_ready(args):
    completed = {}
    for suite in SUITES:
        directory = args.output / 'benchmarks' / suite
        paths = [directory / f'shard{i:02d}of{args.shard_count:02d}.jsonl' for i in range(args.shard_count)]
        if not all(p.with_suffix('.done.json').is_file() for p in paths):
            continue
        if not all((directory/name).is_file() for name in ('metrics.json', 'metrics_per_record.json')):
            rows = ref.read_jsonl(args.output / 'assets' / f'{suite}.jsonl')
            contracts = [json.loads(p.with_suffix('.contract.json').read_text()) for p in paths]
            for i, contract in enumerate(contracts):
                if (contract['limit'] != 0 or contract['shard_index'] != i
                        or contract['shard_count'] != args.shard_count
                        or contract['plan_sha256'] != ref.sha256_file(args.output/'plan.json')):
                    raise ValueError('Shard contract mismatch')
            predictions = [r for path in paths for r in ref.read_jsonl(path)]
            indexed = {p['uid']: p for p in predictions}
            if len(predictions) != len(indexed) or set(indexed) != {r['uid'] for r in rows}:
                raise ValueError('Incomplete or duplicate merged prediction coverage')
            predictions = [indexed[r['uid']] for r in rows]
            if suite == 'DOTAv2-Pointing100':
                metrics, per_record = pointing.score_points(rows, predictions)
            else:
                metrics, per_record = full.score(rows, predictions)
                if rows[0]['task'] == 'detection':
                    classes = [s['canonical'] for s in rows[0]['class_specs']]
                    for policy, field in [('containment', 'contained_predictions'),
                                          ('dedup_containment', 'dedup_contained_predictions')]:
                        index = {(p['dataset'], p['image_id']): dict(p, predictions=p[field]) for p in predictions}
                        metrics[policy] = ref.detection.f1_for_threshold(rows, index, classes, .5, 0)
                    metrics['point_diagnostics'] = ablation.point_diagnostics(rows, predictions)
                metrics['round1_mean_seconds'] = sum(p['round1_seconds'] for p in predictions)/len(predictions)
                metrics['round2_mean_seconds'] = sum(p['round2_seconds'] for p in predictions)/len(predictions)
                metrics['budget_saturated_branches'] = sum(p['budget_saturated_branches'] for p in predictions)
                metrics['generated_branches'] = sum(p['generated_branches'] for p in predictions)
                metrics['grammar_valid_branches'] = sum(p['grammar_valid_branches'] for p in predictions)
            metrics['checkpoint_sha256'] = SHA
            metrics['manifest_sha256'] = ref.sha256_file(args.output/'assets'/f'{suite}.jsonl')
            # atomic_json accepts a mapping, not a bare list of per-query rows.
            ref.atomic_json(directory/'metrics_per_record.json', dict(rows=per_record))
            ref.atomic_json(directory/'metrics.json', metrics)
            print('SCORED', suite, json.dumps(metrics.get('raw', metrics.get('grounding', {}))), flush=True)
        completed[suite] = json.loads((directory/'metrics.json').read_text())
    ref.atomic_json(args.output/'results.json', completed)
    lines = ['# Two-epoch RefWave LoRA: Stage-2 best-validation step 2100', '',
             'Predictions are uncapped. Raw, containment, and NMS scores are separate.',
             'Grounding uses the first predicted box. Pointing is a derived diagnostic.', '',
             '| Suite | Records | Raw P / R / F1 (%) or grounding Acc50 / Acc70 / mIoU (%) | Mean seconds |',
             '|---|---:|---|---:|']
    for suite, metrics in completed.items():
        if 'raw' in metrics:
            value = metrics['raw']['0.5']; keys = ('precision', 'recall', 'f1')
        elif 'grounding' in metrics:
            value = metrics['grounding']; keys = ('first_acc_0_5', 'first_acc_0_7', 'first_mean_iou')
        else:
            value = metrics; keys = ('precision', 'recall', 'f1')
        score = ' / '.join(f'{100*value[k]:.2f}' for k in keys)
        lines.append(f"| {suite} | {metrics['records']} | {score} | {metrics['mean_seconds']:.3f} |")
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    return completed


def gpu_free(gpu):
    pids = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True).strip()
    memory = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                                         '--format=csv,noheader,nounits'], text=True).strip())
    return not pids and memory < 500


def queue(args):
    args.shard_count = len(args.gpus)
    if len(set(args.gpus)) != args.shard_count:
        raise ValueError('GPU IDs must be unique')
    queue_contract = dict(gpus=args.gpus, shards=args.shard_count, plan_sha256=ref.sha256_file(args.output/'plan.json'))
    path = args.output/'queue.json'
    if path.exists() and json.loads(path.read_text()) != queue_contract:
        raise ValueError('Resume must keep the same GPU/shard assignment')
    ref.atomic_json(path, queue_contract)
    (args.output/'logs').mkdir(exist_ok=True)
    children, handles, finished = {}, {}, {}
    completed, merge_error = {}, None
    while len(finished) < args.shard_count:
        for rank, gpu in enumerate(args.gpus):
            if rank in finished:
                continue
            if rank not in children and rank not in finished and gpu_free(gpu):
                handle = (args.output/'logs'/f'gpu{gpu}.log').open('a')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                           TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
                children[rank] = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker',
                    '--output', str(args.output), '--shard-index', str(rank), '--shard-count', str(args.shard_count)],
                    env=env, stdout=handle, stderr=subprocess.STDOUT)
                handles[rank] = handle
                print('START', rank, gpu, children[rank].pid, flush=True)
            if rank in children and children[rank].poll() is not None:
                finished[rank] = children[rank].returncode
                handles[rank].close()
                print('EXIT', rank, finished[rank], flush=True)
        try:
            completed = merge_ready(args)
            merge_error = None
        except Exception:
            error = traceback.format_exc()
            if error != merge_error:
                print(error, flush=True)
            merge_error = error
        ref.atomic_json(args.output/'status.json', dict(completed_suites=list(completed),
            expected_suites=list(SUITES), worker_exitcodes=finished,
            pids={r:p.pid for r,p in children.items() if p.poll() is None},
            merge_error=merge_error,
            complete=not merge_error and len(completed)==len(SUITES)
            and len(finished)==args.shard_count and not any(finished.values())))
        if len(finished) < args.shard_count:
            time.sleep(15)
    if any(finished.values()) or len(merge_ready(args)) != len(SUITES):
        raise RuntimeError('Evaluation incomplete; inspect worker logs and resume the same queue')
    print('ALL EVALUATIONS COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'worker', 'queue', 'merge', 'import'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--suites', nargs='+', choices=SUITES, default=list(SUITES))
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output)
    elif args.mode == 'worker':
        worker(args)
    elif args.mode == 'merge':
        merge_ready(args)
    elif args.mode == 'import':
        if not args.resume_from:
            parser.error('import requires --resume-from')
        import_predictions(args.output, args.resume_from, args.gpus)
    else:
        queue(args)


if __name__ == '__main__':
    main()
