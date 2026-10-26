"""Shared contracts for the paired PID detection efficiency evaluation."""
import importlib
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASELINES = Path(os.environ.get('PID_BASELINES', HERE.parent / 'LA-Rex-Protocol-Eval-v3')).resolve()
if not (BASELINES / 'common.py').is_file():
    raise FileNotFoundError(f'Expected the completed baseline package at {BASELINES}; set PID_BASELINES if moved.')
sys.path.insert(0, str(BASELINES))
core = importlib.import_module('common')
if Path(core.__file__).resolve() != BASELINES / 'common.py':
    raise RuntimeError('An unrelated common.py was imported')
from common import atomic, digest, freeze, read, rows, sha, query_only, DATASETS, CHECKPOINTS

PROTOCOL = 'pid_gt_categories_paired_extents_v1'
MODES = ('packed', 'sequential')
SOFTWARE_KEYS = ('python', 'torch', 'cuda', 'gpu', 'transformers', 'torchvision',
                 'numpy', 'Pillow', 'tokenizers', 'safetensors', 'accelerate')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_plan(output, check_sources=True):
    plan = read(output / 'plan.json')
    require(plan['protocol'] == PROTOCOL, 'Wrong PID efficiency protocol')
    if check_sources:
        for file, expected in plan['sources'].items():
            require(sha(file) == expected, 'Code changed after preparation: ' + file)
    data = {}
    for ds, info in plan['datasets'].items():
        require(sha(info['path']) == info['sha256'], 'Frozen manifest changed: ' + ds)
        data[ds] = rows(info['path'])
    return plan, data


def selected_rows(data, dataset, shard, count, smoke):
    from evaluate import selected_rows as original_selection
    return original_selection(data, dataset, shard, count, smoke)


def shard_folder(output, dataset, shard, count, smoke):
    return output / ('smoke_pairs' if smoke else 'pairs') / dataset / f'shard{shard:02d}of{count:02d}'


def record_path(folder, index):
    return folder / f'record{index:05d}.json'


def order_for(index):
    return MODES if index % 2 == 0 else tuple(reversed(MODES))


def finished(output, dataset, shard, count, smoke, plan, data):
    folder = shard_folder(output, dataset, shard, count, smoke)
    done = folder / 'done.json'
    if not done.exists():
        return False
    record = read(done)
    selected = selected_rows(data, dataset, shard, count, smoke)
    expected_files = {record_path(folder, i).name for i in range(len(selected))}
    require(record.get('complete') and record['plan_sha256'] == digest(plan), 'Completion plan mismatch')
    require(record['uids'] == [r['uid'] for r in selected], 'Completion membership mismatch')
    require(set(record['files']) == expected_files, 'Incomplete completion file list')
    require({p.name for p in folder.glob('record*.json')} == expected_files, 'Unexpected saved pair files')
    for name, expected in record['files'].items():
        require(sha(folder / name) == expected, f'Completed predictions changed: {folder / name}')
    return True


def check_software(actual, expected):
    differences = {k: (expected.get(k), actual.get(k)) for k in SOFTWARE_KEYS
                   if actual.get(k) != expected.get(k)}
    require(not differences, f'Environment differs from the baseline evaluation: {differences}. '
            'Activate the same remote environment; do not upgrade packages for this run.')


def check_pair(pair, row, plan_hash):
    require(pair['uid'] == row['uid'] and pair['suite'] == row['suite'], 'Pair identity differs')
    require(pair['plan_sha256'] == plan_hash, 'Mixed paired evaluation plans')
    require(set(pair['modes']) == set(MODES), 'Both modes must be saved together')
    require(sorted(pair['order']) == sorted(MODES), 'Invalid execution order')
    packed, sequential = (pair['modes'][m] for m in MODES)
    query = query_only(row)
    for mode, result in pair['modes'].items():
        require(result['query'] == query, 'Inference query differs from GT-present category scope')
        require(result['uid'] == row['uid'] and result['suite'] == row['suite'], 'Mode identity differs')
        require(result['execution_mode'] == mode, 'Mode label differs')
        require(result['plan_sha256'] == plan_hash, 'Mode plan mismatch')
        require(result['round1']['branches'] == len(query['class_specs'])
                and result['round1']['waves'] == 1, 'Discovery was not concurrent')
        require(result['image_encodes'] == result['prefix_forwards'] == 1, 'Image/prefix was not computed once')
        require(result['round2'].get('cross_round_cache_reuse') is True, 'Cross-round prefix not reused')
        n = len(result['points'])
        require(len(result['refinements']) == n, 'An address was dropped')
        for i, (point, extent) in enumerate(zip(result['points'], result['refinements'])):
            require(extent['address_id'] == str(i) and extent['label'] == point['label'], 'Address ownership changed')
        calls = int(n > 0) if mode == 'packed' else n
        require(result['round2']['actual_round2_forwards'] == calls
                and result['round2']['qwen_total_forwards'] == calls, 'Unexpected extent forward count')
    require(packed['points'] == sequential['points'], 'Round-1 points differ; not a matched extent ablation')
    for result in (packed, sequential):
        require([a['label'] for a in result['point_answers']] == [s['canonical'] for s in query['class_specs']],
                'Reference order changed')
    fields = ('label', 'question', 'answer', 'seed', 'tokens_generated')
    p = [[a.get(k) for k in fields] for a in packed['point_answers']]
    s = [[a.get(k) for k in fields] for a in sequential['point_answers']]
    require(p == s, 'Round-1 answers/RNG differ between extent modes')

