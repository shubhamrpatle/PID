#!/usr/bin/env python3
"""Evaluate the final RefWave model beside an existing Final-Controls-Eval install."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CONTROLS = Path(os.environ.get('CONTROLS_DIR', HERE.parent / 'Final-Controls-Eval')).resolve()
OUTPUT = Path(os.environ.get('REFWAVE_OUTPUT', HERE / 'results')).resolve()
DOWNLOAD = Path(os.environ.get('REFWAVE_CHECKPOINTS', HERE / 'checkpoints')).resolve()
REPO = 'CaseStudyRef/RefWave-Cluster-Runs'
REVISION = '24d7f19e120601aada5d4b50d8ed113e6e741c43'
SPEC = dict(
    run='refwave_full_8gpu_s1e2_s2e2_both_ft_gpu8_ep2_v3_20260914T075320Z_9mgG6V',
    sha256='a1f8da46db0b3447b59900a848b035a11aeefa5b465f7a5faf6e2e64f52aedf3',
    bytes=42216772310, step=2400,
    parent_sha256='f88180dbafc214e63f0623b28f4a7674afbefd86812458b1fc9c9f843d29f16f')
BUDGET = 8192
EXPECTED = {'DIOR': 586, 'DOTAv2': 874, 'DIOR-RSVG': 7500, 'VRSBench-VG': 16159}

# Set caches before importing HF or Transformers; ignore the inaccessible shared token.
HF_CACHE = Path(os.environ.get('EVAL_HF_HOME', HERE / '.hf-cache')).expanduser().resolve()
HF_CACHE.mkdir(parents=True, exist_ok=True, mode=0o700)
for key, value in {'HF_HOME': HF_CACHE, 'HF_TOKEN_PATH': HF_CACHE/'token',
                  'HF_HUB_CACHE': HF_CACHE/'hub', 'HF_MODULES_CACHE': HF_CACHE/'modules',
                  'HF_ASSETS_CACHE': HF_CACHE/'assets', 'HF_XET_CACHE': HF_CACHE/'xet'}.items():
    os.environ[key] = str(value)
for key in ('TRANSFORMERS_CACHE', 'PYTORCH_TRANSFORMERS_CACHE',
            'PYTORCH_PRETRAINED_BERT_CACHE', 'HUGGINGFACE_HUB_CACHE', 'HUGGINGFACE_ASSETS_CACHE'):
    os.environ.pop(key, None)
ROOT = CONTROLS / 'workspace'
PROJECT = ROOT / 'Zero_shot_ anlysis_LA'
PATHS = [ROOT/'RefWave/src', ROOT/'Pixel-PIVR/src', ROOT/'Pixel-PIVR', ROOT, PROJECT/'scripts']
if not (PROJECT/'scripts/evaluate_final_standard_isolated.py').is_file():
    raise FileNotFoundError('Set CONTROLS_DIR to the completed Final-Controls-Eval directory')
sys.path[:0] = list(map(str, PATHS))
import evaluate_final_standard_isolated as evaluation
import run_final_standard_isolated_queue as queue
base, release = evaluation.base, evaluation.release
import pid_packed_protocol as pid

release.REPO, release.REVISION = REPO, REVISION
release.DOWNLOAD, release.SPECS = DOWNLOAD, {'refwave': SPEC}
evaluation.METHODS = ('refwave',)


def read(path):
    return json.loads(Path(path).read_text())


def verify_runtime():
    expected = read(HERE / 'runtime_hashes.json')
    for relative, digest in expected.items():
        path = CONTROLS / relative
        if not path.is_file() or base.common.sha(path) != digest:
            raise ValueError(f'Controls runtime differs from the tested version: {path}')
    return {str(CONTROLS / p): digest for p, digest in expected.items()}


def metadata():
    from huggingface_hub import HfApi, hf_hub_download
    DOWNLOAD.mkdir(parents=True, exist_ok=True)
    remote = f"runs/{SPEC['run']}/stage2/best.pt"
    info = HfApi(token=False).get_paths_info(REPO, paths=[remote], repo_type='dataset', revision=REVISION)[0]
    if (info.size, info.lfs.sha256) != (SPEC['bytes'], SPEC['sha256']):
        raise ValueError('Remote checkpoint identity differs from the pinned release')
    configs, completions, curves = {}, {}, {}
    for stage in ('stage1', 'stage2'):
        for name in ('config.json', 'done.json', 'validation_curve.jsonl'):
            hf_hub_download(REPO, f"runs/{SPEC['run']}/{stage}/{name}", repo_type='dataset',
                            revision=REVISION, local_dir=DOWNLOAD, token=False)
        folder = DOWNLOAD / 'runs' / SPEC['run'] / stage
        configs[stage] = read(folder / 'config.json')
        completions[stage] = read(folder / 'done.json')
        curves[stage] = base.common.read(folder / 'validation_curve.jsonl')
    best = release.validate_metadata('refwave', configs, completions, curves)
    return dict(SPEC, method='refwave', repo=REPO, revision=REVISION,
                checkpoint=str(release.checkpoint('refwave')),
                criterion='minimum validation detection_grounding_macro_loss', validation=best,
                configs=configs, completions=completions, curves=curves)


def copy_manifests():
    control_plan = read(CONTROLS / 'results/plan.json')
    evaluation.check_sources(control_plan)
    assets = OUTPUT / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for suite, count in EXPECTED.items():
        source = CONTROLS / 'results/assets' / (suite + '.jsonl')
        summary = read(source.with_suffix('.json'))
        digest = base.common.sha(source)
        if digest != summary['sha256'] or digest != control_plan['manifests'][suite]['sha256']:
            raise ValueError('Controls benchmark manifest changed: ' + suite)
        rows = base.common.read(source)
        if len(rows) != count or summary['records'] != count or len({r['uid'] for r in rows}) != count:
            raise ValueError('Unexpected benchmark membership: ' + suite)
        for row in rows:
            if not Path(row['image_path']).is_file():
                raise FileNotFoundError(row['image_path'])
            if row['task'] == 'detection':
                expected_refs = [(s['canonical'], s['prompt']) for s in row['class_specs']]
                actual_refs = [(r['label'], r['reference']) for r in row['references']]
                if actual_refs != expected_refs or len(expected_refs) != (20 if suite == 'DIOR' else 18):
                    raise ValueError('Discovery does not cover the complete fixed class vocabulary')
            elif len(row['references']) != 1 or row['references'][0]['label'] != row['query']:
                raise ValueError('Grounding reference differs from the full query')
        dest = assets / source.name
        if dest.exists() and base.common.sha(dest) != digest:
            raise ValueError('Existing RefWave manifest differs; use a new REFWAVE_OUTPUT')
        if not dest.exists():
            shutil.copy2(source, dest)
        base.freeze(dest.with_suffix('.json'), summary)
        manifests[suite] = summary
    return manifests


def prepare():
    from huggingface_hub import hf_hub_download
    import torch
    signatures = verify_runtime()
    manifests = copy_manifests()
    selected = metadata()
    base.freeze(OUTPUT / 'selection.json', {'refwave': selected})
    signatures.update({str(p): base.common.sha(p) for p in
                       (Path(__file__).resolve(), HERE / 'runtime_hashes.json')})
    plan = dict(protocol=pid.PROTOCOL_ID, policy=pid.POLICY, repo=REPO, revision=REVISION,
        models={'refwave': SPEC}, manifests=manifests, sources=signatures,
        controls_plan_sha256=base.common.sha(CONTROLS / 'results/plan.json'),
        generation=dict(max_new_tokens=BUDGET, temperature=.7, top_p=.9,
                        repetition_penalty=1.1, generation_mode='hybrid'),
        image_token_limit=6000, warmup_records_per_shard=2, shards=2,
        timing=pid.POLICY['timing'], added_nms=False, max_detections=None,
        pointing='Same native Point to: {reference}. prompt/parser as controls; all references concurrent; no Round 2',
        comparison_limits=[
            'Use the same frozen test records and scorer; supplied training/validation recipes may differ.',
            'RefWave per-reference discovery budget versus standard per-image direct-box budget.',
            'Controls use sequential native pointing queries; PID uses concurrent references.',
            'Concurrent pointing is timed as one group; individual reference latencies are not measured.',
            'Timings on a shared node can depend on contention; compare recorded hardware/software.'])
    base.freeze(OUTPUT / 'plan.json', plan)
    path = release.checkpoint('refwave')
    ready_path = OUTPUT / 'checkpoints_ready.json'
    if ready_path.is_file() and path.is_file():
        old = read(ready_path)['models']['refwave']
        if (old['sha256'], old['bytes'], old['mtime_ns']) == (
                SPEC['sha256'], path.stat().st_size, path.stat().st_mtime_ns):
            print('Reusing previously SHA256-verified RefWave checkpoint.', flush=True)
            return
    required = (0 if path.is_file() else SPEC['bytes']) + 8 * 1024**3
    if shutil.disk_usage(DOWNLOAD).free < required:
        raise RuntimeError('Need approximately 50 GB free for best.pt plus output/headroom; nothing deleted')
    print('Downloading only RefWave Stage-2 best.pt (42.2 GB).', flush=True)
    path = Path(hf_hub_download(REPO, f"runs/{SPEC['run']}/stage2/best.pt", repo_type='dataset',
                               revision=REVISION, local_dir=DOWNLOAD, token=False))
    print('Checking checkpoint SHA256 and saved validation selection.', flush=True)
    if path.stat().st_size != SPEC['bytes'] or base.common.sha(path) != SPEC['sha256']:
        raise ValueError('Downloaded checkpoint failed SHA256/size verification')
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    tied = read(base.BASE / 'config.json')['text_config'].get('tie_word_embeddings', False)
    best, components = release.validate_saved(saved, 'refwave', selected['configs'],
        selected['completions'], selected['curves'], tied)
    verified = dict(path=str(path), sha256=SPEC['sha256'], bytes=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns, step=best['step'], components=components, verified=True)
    del saved
    gc.collect()
    base.common.write(ready_path, dict(complete=True, models={'refwave': verified}))
    print('Verified validation-best step 2400; ready for remote GPU smoke tests.', flush=True)


def point_branches(row):
    from pixel_pivr.class_wave import ClassPointBranch
    result = []
    for ref in base.references(row):
        # Exactly the control's per-reference seed, independent of sharding/order.
        seed_text = json.dumps([0, row['suite'], row['uid'], ref['label']])
        seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4], 'big')
        result.append(ClassPointBranch(ref['label'], f"Point to: {ref['text']}.", seed))
    return result


def pointing(worker, decoder, row, image):
    import torch
    from evaluate_molmo_pointing import validate_xy
    branches = point_branches(row)
    decoder.max_wave_size = len(branches)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    answers, stats = decoder.generate(image, branches, max_new_tokens=BUDGET,
        temperature=.7, top_p=.9, repetition_penalty=1.1, generation_mode='hybrid')
    pid.validate_discovery(stats, len(branches))
    if len(answers) != len(branches) or [a['label'] for a in answers] != [b.label for b in branches]:
        raise ValueError('Missing/reordered native pointing answer')
    parsed = []
    for answer, branch in zip(answers, branches):
        xy, errors = base.parse_la_points(answer['answer'], *image.size)
        xy, invalid = validate_xy(xy, *image.size)
        errors.extend(['out_of_bounds_coordinate'] * invalid)
        tokens = len(worker.tokenizer.encode(answer['answer'], add_special_tokens=False))
        parsed.append(dict(label=branch.label,
            points=[dict(label=branch.label, point=p, source='point') for p in xy],
            raw_response=answer['answer'], prompt=branch.question, parse_errors=errors,
            generated_tokens=tokens, token_count_kind='retokenized_response',
            effective_budget=BUDGET, budget_saturated=tokens >= BUDGET,
            latency_seconds=None, timing_scope='concurrent group; see record latency_seconds'))
    torch.cuda.synchronize()
    return dict(answers=parsed, points=[p for a in parsed for p in a['points']], stats=stats,
        latency_seconds=time.perf_counter()-started,
        peak_gpu_mb=max(torch.cuda.max_memory_allocated()/1024**2, stats.get('peak_gpu_mb', 0)),
        protocol=pid.PROTOCOL_ID, round2_execution=None)


def infer(args):
    import torch
    from PIL import Image
    from refwave.io import append_jsonl, read_jsonl_log
    from pixel_pivr.fast_class_wave import FastCachedHybridClassWaveDecoder
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    plan = read(OUTPUT / 'plan.json')
    evaluation.check_sources(plan)
    manifest = OUTPUT / 'assets' / (args.suite + '.jsonl')
    if base.common.sha(manifest) != plan['manifests'][args.suite]['sha256']:
        raise ValueError('Frozen evaluation manifest changed')
    rows = base.common.read(manifest)
    if args.limit:
        rows = rows[:args.limit]
    rows = rows[args.shard_index::args.shard_count]
    directory = base.folder(args)
    directory.mkdir(parents=True, exist_ok=True)
    tag = f'shard{args.shard_index:02d}of{args.shard_count:02d}'
    contract = dict(protocol=pid.PROTOCOL_ID, plan_sha256=base.common.sha(OUTPUT / 'plan.json'),
        checkpoint_sha256=SPEC['sha256'], step=SPEC['step'], method='refwave', kind=args.kind,
        suite=args.suite, manifest_sha256=plan['manifests'][args.suite]['sha256'],
        shard_index=args.shard_index, shard_count=args.shard_count, limit=args.limit,
        warmup_records=min(2,len(rows)), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        generation=plan['generation'], image_token_limit=6000, added_nms=False, max_detections=None,
        round1='all_requested_references_concurrent',
        round2=None if args.kind == 'pointing' else 'packed_all_one_forward', timing=plan['timing'])
    base.freeze(directory / (tag+'.contract.json'), contract)
    path = directory / (tag+'.jsonl')
    previous = read_jsonl_log(path, repair_tail=True) if path.exists() else []
    done = {r['uid'] for r in previous}
    if len(done) != len(previous) or not done <= {r['uid'] for r in rows}:
        raise ValueError('Duplicate/foreign saved predictions')
    if len(done) < len(rows):
        worker = evaluation.load_worker('refwave', OUTPUT)
        decoders = (FastCachedHybridClassWaveDecoder(worker, image_token_limit=6000,
            max_wave_size=32, bucket_suffix_lengths=False) if args.kind == 'pointing'
            else pid.make_decoders(worker))

        def one(row):
            with Image.open(row['image_path']) as source:
                image = source.convert('RGB')
            if image.size != (row['width'], row['height']):
                raise ValueError('Image dimensions differ from the controls manifest')
            with torch.inference_mode():
                result = (pointing(worker, decoders, row, image) if args.kind == 'pointing'
                          else pid.infer_one(worker, decoders, row, image, BUDGET))
            result.update({k:row[k] for k in ('uid','dataset','image_id','task','suite')})
            if row['task'] == 'grounding':
                result.update(sample_id=row['sample_id'], query=row['query'])
            return result

        for row in rows[:2]:
            one(row)
        for row in rows:
            if row['uid'] in done:
                continue
            result = one(row)
            append_jsonl(path, result)
            done.add(row['uid'])
            base.common.write(directory / (tag+'.status.json'), dict(records=len(done), expected=len(rows),
                latest_seconds=result['latency_seconds'], updated=time.time()))
            print(args.kind, args.suite, f'{len(done)}/{len(rows)}',
                  f"{result['latency_seconds']:.3f}s", flush=True)
    base.common.write(directory / (tag+'.done.json'), dict(complete=True, records=len(rows)))


def report():
    plan = read(OUTPUT / 'plan.json')
    lines = ['# Final full-FT comparison', '',
        'Detection: F1@0.50 / F1@0.75 / mean F1 over ten IoUs 0.50:0.05:0.95.',
        'Grounding: first-box Acc@0.5 / Acc@0.7 / mIoU. Pointing: point-inside-HBB F1.',
        'Scores are percentages; time is measured mean seconds per image or expression.', '',
        '| Model | Task | Benchmark | Records | Scores (%) | Seconds/record |',
        '| --- | --- | --- | ---: | --- | ---: |']
    cells = []
    for model, root in [('standard', CONTROLS/'results'), ('class_isolated', CONTROLS/'results'),
                        ('refwave', OUTPUT)]:
        for kind in ('boxes', 'pointing'):
            for suite, count in EXPECTED.items():
                path = root/'results'/kind/model/suite/'metrics.json'
                scores = seconds = '-'
                if path.is_file():
                    m = read(path)
                    c = m['contract']
                    if (not m['complete'] or m['records'] != count or c['limit'] != 0
                            or c['manifest_sha256'] != plan['manifests'][suite]['sha256']):
                        raise ValueError('Incomplete or incomparable result: ' + str(path))
                    selected = read(root/'selection.json')[model]
                    if c['checkpoint_sha256'] != selected['sha256'] or c['step'] != selected['step']:
                        raise ValueError('Result uses a different checkpoint: ' + str(path))
                    if 'pointing' in m:
                        values = [m['pointing']['f1']]
                    elif 'raw' in m:
                        values = [m['raw']['0.5']['f1'], m['raw']['0.75']['f1'], m['mean_f1_iou_0p50_to_0p95']]
                    else:
                        values = [m['grounding'][k] for k in ('first_acc_0_5','first_acc_0_7','first_mean_iou')]
                    scores = ' / '.join(f'{v*100:.2f}' for v in values)
                    seconds = f"{m['mean_seconds']:.3f}"
                    cells.append(dict(model=model, kind=kind, suite=suite, metrics=m))
                task = 'pointing' if kind == 'pointing' else ('detection' if suite in ('DIOR','DOTAv2') else 'grounding')
                lines.append(f'| {model} | {task} | {suite} | {count} | {scores} | {seconds} |')
    lines += ['', 'RefWave: validation-best step 2400, PID packed-all. Controls: their frozen validation-selected checkpoints.',
        'Identical evaluation data/scoring does not imply identical training data or compute.',
        'Control pointing is sequential per reference; RefWave pointing is concurrent. No per-reference latency is invented.',
        'Generation budgets are 8192 per reference for RefWave, versus per all-class query for standard detection.',
        'No added NMS or maxDets=100. GPU/software are recorded per cell; check them before latency claims.', '']
    (OUTPUT/'comparison.md').write_text('\n'.join(lines))
    result = dict(expected_cells=24, completed_cells=len(cells),
                  refwave_completed_cells=sum(c['model']=='refwave' for c in cells), cells=cells)
    base.common.write(OUTPUT/'comparison.json', result)
    return result


def command(output, job, mode):
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', mode]
    for name in ('model', 'kind', 'suite', 'shard_index', 'shard_count', 'limit'):
        cmd += ['--'+name.replace('_','-'), str(job[name])]
    return cmd


def environment(model, gpu=None):
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    env['PYTHONPATH'] = os.pathsep.join(map(str, PATHS))
    env['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else str(gpu)
    env.update(CONTROLS_DIR=str(CONTROLS), REFWAVE_OUTPUT=str(OUTPUT), REFWAVE_CHECKPOINTS=str(DOWNLOAD))
    return env


def validate_gpus(gpus):
    if not 1 <= len(gpus) <= 8 or len(set(gpus)) != len(gpus):
        raise ValueError('Select 1-8 distinct allocated GPU IDs or GPU UUIDs')
    if any(not (g.isdecimal() or g.startswith('GPU-')) for g in gpus):
        raise ValueError('Use physical nvidia-smi GPU indices or full GPU UUIDs')
    allocated = os.environ.get('CUDA_VISIBLE_DEVICES')
    if allocated is not None and not set(gpus) <= set(allocated.split(',')):
        raise ValueError('Requested GPUs are outside CUDA_VISIBLE_DEVICES')
    return gpus


def jobs(smoke=False):
    return [dict(model='refwave', kind=kind, suite=suite,
                 shard_index=i, shard_count=1 if smoke else 2, limit=2 if smoke else 0)
            for kind in ('boxes', 'pointing') for suite in EXPECTED
            for i in range(1 if smoke else 2)]


def run(gpus, smoke_only=False):
    gpus = validate_gpus(gpus)
    evaluation.check_sources(read(OUTPUT/'plan.json'))
    if not read(OUTPUT/'checkpoints_ready.json')['complete']:
        raise ValueError('Run prepare before inference')
    queue.command, queue.jobs = command, jobs
    queue.scheduling.PYTHON = Path(sys.executable)
    queue.scheduling.environment = environment
    (OUTPUT/'logs').mkdir(exist_ok=True)
    base.common.write(OUTPUT/'scheduled_jobs.json', dict(smoke=jobs(True), full=jobs(False), expected_cells=8, gpus=gpus))
    active = []

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        args = argparse.Namespace(output=OUTPUT, gpus=gpus)
        queue.run_phase(args, True, active)
        if smoke_only:
            queue.write_status(OUTPUT, 'smoke_complete', active=[])
            return
        queue.run_phase(args, False, active)
        result = report()
        if result['refwave_completed_cells'] != 8:
            raise RuntimeError('Not all eight RefWave result cells completed')
        base.common.write(OUTPUT/'all.done.json', dict(complete=True, cells=8))
        queue.write_status(OUTPUT, 'complete', active=[], completed_cells=8)
    except KeyboardInterrupt:
        for task in active:
            if task['process'].poll() is None:
                os.killpg(task['process'].pid, signal.SIGINT)
        for task in active:
            try:
                task['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(task['process'].pid, signal.SIGTERM)
                task['process'].wait(timeout=30)
            task['log'].close()
        queue.write_status(OUTPUT, 'stopped', active=[], partial_outputs_preserved=True)
        raise SystemExit(130)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        p = argparse.ArgumentParser()
        p.add_argument('mode', choices=['infer','merge'])
        p.add_argument('--model', choices=['refwave'], required=True)
        p.add_argument('--kind', choices=['boxes','pointing'], required=True)
        p.add_argument('--suite', choices=EXPECTED, required=True)
        p.add_argument('--shard-index', type=int, required=True)
        p.add_argument('--shard-count', type=int, required=True)
        p.add_argument('--limit', type=int, default=0)
        args = p.parse_args(sys.argv[2:])
        args.output = OUTPUT
        if not 0 <= args.shard_index < args.shard_count or args.limit < 0:
            p.error('Invalid shard or limit')
        evaluation.check_sources(read(OUTPUT/'plan.json'))
        (infer if args.mode == 'infer' else base.merge)(args)
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['launch','prepare','metadata','run','smoke','report','check-imports'])
    p.add_argument('--gpus', nargs='+', default=['0','1','2','3'])
    args = p.parse_args()
    if args.mode in ('launch', 'run', 'smoke'):
        validate_gpus(args.gpus)
    if args.mode == 'metadata':
        s = metadata()
        print(json.dumps({k:s[k] for k in ('run','step','sha256','criterion','validation')}, indent=2))
        return
    if args.mode == 'check-imports':
        import importlib
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        verify_runtime()
        for name in ('refwave.train', 'pixel_pivr.worker', 'pixel_pivr.fast_class_wave',
                     'zero_shot_analysis.fused_pivr_geometry', 'refwave_packed_geometry'):
            importlib.import_module(name)
        for reference in read(base.BASE/'config.json')['auto_map'].values():
            if isinstance(reference, str):
                get_class_from_dynamic_module(reference, str(base.BASE), local_files_only=True)
        print('PID packed-all imports passed; GPU inference has not been tested here.')
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.mode == 'report':
        report()
        print((OUTPUT/'comparison.md').read_text())
        return
    with (OUTPUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode in ('launch','prepare'):
            prepare()
        if args.mode in ('launch','run','smoke'):
            run(args.gpus, args.mode == 'smoke')


if __name__ == '__main__':
    main()
