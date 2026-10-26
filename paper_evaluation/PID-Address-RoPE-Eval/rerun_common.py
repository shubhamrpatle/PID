"""Versioned contracts; read previous evaluation assets without modifying them."""
from copy import deepcopy
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PAIR_DIR = Path(os.environ.get('PID_PAIRED_DIR', HERE.parent / 'PID-GT-Efficiency-Eval')).resolve()
if not (PAIR_DIR / 'pid_common.py').is_file():
    raise FileNotFoundError(f'Missing completed paired-evaluation installation: {PAIR_DIR}')
sys.path.insert(0, str(PAIR_DIR))
import pid_common as paired
from common import atomic, freeze, read, rows, sha, digest, query_only, DATASETS, CHECKPOINTS

PROTOCOL = 'pid_gt_address_fp32_rope_v1'
MODES = ('predicted', 'gt_address')
DEFAULT_OUTPUT = HERE / 'results'
require = paired.require


def prepare(output):
    output = Path(output).resolve()
    source = Path(os.environ.get('PID_PAIRED_RESULTS', PAIR_DIR / 'results')).resolve()
    require(output != source and not output.is_relative_to(source), 'Use a new output directory')
    original, data = paired.load_plan(source)
    require(original['models']['pid']['sha256'] == CHECKPOINTS['pid'][1]
            and original['models']['pid']['step'] == CHECKPOINTS['pid'][2], 'Wrong final PID checkpoint')
    require(original['budget'] == 8192 and original['image_token_limit'] == 6000,
            'Existing final generation/image budgets changed')
    require(original['generation'] == dict(temperature=.7, top_p=.9, repetition_penalty=1.1, mode='hybrid'),
            'Existing final sampling settings changed')
    require(original['shards'] == 4 and original['warmup_records'] == 1,
            'Expected the completed four-shard, one-warmup paired evaluation plan')
    for ds, (n, boxes, _) in DATASETS.items():
        require(len(data[ds]) == len({r['uid'] for r in data[ds]}) == n, 'Incomplete/duplicate manifest')
        require(sum(len(r['gt']) for r in data[ds]) == boxes, 'Ground truth changed')
    for file, identity in original['models']['pid']['files'].items():
        stat = Path(file).stat()
        require((stat.st_size, stat.st_mtime_ns) == (identity['bytes'], identity['mtime_ns']),
                'Previously verified model file changed; re-prepare the original paired plan: ' + file)
    plan = deepcopy(original)
    plan.update(protocol=PROTOCOL, source_plan=str(source / 'plan.json'),
        source_plan_sha256=sha(source / 'plan.json'), modes=list(MODES),
        scope='Both modes request GT-present category names; only gt_address receives annotation-derived points.',
        pairing='Same checkpoint/image/GPU; alternate predicted-first and GT-first; no equality of address sets assumed.',
        extent='Packed-all for both modes; no splitting, address cap or OOM fallback.',
        rope_fix='Reconstruct frequencies and positions in FP32, outside autocast; cast only final cosine/sine caches.',
        gate='Small image before/after the maximum-GT DOTAv2 image; same outputs and unchanged low-position rotary cache.',
        oracle_timing='Skips discovery; not deployment latency or a speedup comparator.')
    for file in sorted(HERE.glob('*.py')) + [HERE / 'run.sh']:
        plan['sources'][str(file)] = sha(file)
    output.mkdir(parents=True, exist_ok=True)
    freeze(output / 'plan.json', plan)
    print('Prepared', 2 * plan['shards'], 'paired full jobs + one GPU acceptance gate. No downloads.', flush=True)
    return plan, data


def load_plan(output, check_sources=True):
    plan = read(output / 'plan.json')
    require(plan['protocol'] == PROTOCOL, 'Wrong output protocol; do not reuse old result folders')
    if check_sources:
        for file, expected in plan['sources'].items():
            require(sha(file) == expected, 'Source changed after preparation: ' + file)
    data = {}
    for ds, info in plan['datasets'].items():
        require(sha(info['path']) == info['sha256'], 'Manifest changed: ' + ds)
        data[ds] = rows(info['path'])
    return plan, data


def folder(output, ds, shard, count):
    return output / 'pairs' / ds / f'shard{shard:02d}of{count:02d}'


def record_path(directory, index):
    return directory / f'record{index:05d}.json'


def check_pair(value, row, plan):
    require(value['uid'] == row['uid'] and value['suite'] == row['suite'], 'Pair identity differs')
    require(value['plan_sha256'] == digest(plan), 'Mixed evaluation plans')
    require(set(value['modes']) == set(MODES) and sorted(value['order']) == sorted(MODES), 'Missing address mode')
    query = query_only(row)
    for mode, r in value['modes'].items():
        require(r['query'] == query and r['address_source'] == mode, 'Incorrect mode or query scope')
        require(r['uid'] == row['uid'] and r['suite'] == row['suite'], 'Result identity differs')
        require(r['image_encodes'] == r['prefix_forwards'] == 1, 'Repeated image/prefix encoding')
        require(r['rope_fix_modules'] > 0, 'FP32 rotary fix was not applied')
        n = len(r['points'])
        require(len(r['refinements']) == n, 'Dropped extent requests')
        require(r['round2']['layout'] == 'packed_all' and r['round2']['cross_round_cache_reuse'], 'Extent path changed')
        require(r['round2']['actual_round2_forwards'] == r['round2']['qwen_total_forwards'] == int(n > 0),
                'Extent requests were split or not decoded')
        for i, (point, extent) in enumerate(zip(r['points'], r['refinements'])):
            require(extent['address_id'] == str(i) and extent['label'] == point['label'], 'Address ownership changed')
        if mode == 'gt_address':
            require(n == len(row['gt']) and r['round1_skipped'] is True, 'Incomplete or non-oracle addresses')
            annotations = r['oracle_audit']['annotations']
            require(sorted(a['gt_index'] for a in annotations) == list(range(len(row['gt']))), 'Missing/repeated GT address')
            for point, annotation in zip(r['points'], annotations):
                gt = row['gt'][annotation['gt_index']]
                x, y = point['point']
                require(point['label'] == gt['label'] and gt['hbox'][0] <= x <= gt['hbox'][2]
                        and gt['hbox'][1] <= y <= gt['hbox'][3], 'Oracle point is outside its own source target')
        else:
            require(not r.get('round1_skipped', False), 'Normal discovery was bypassed')
            require(r['round1']['branches'] == len(query['class_specs']) and r['round1']['waves'] == 1,
                    'Discovery is not concurrent for all requested references')


def finished(output, ds, shard, plan, data):
    directory = folder(output, ds, shard, plan['shards'])
    path = directory / 'done.json'
    if not path.exists():
        return False
    done = read(path)
    selected = data[ds][shard::plan['shards']]
    expected = {record_path(directory, i).name for i in range(len(selected))}
    require(done['complete'] and done['plan_sha256'] == digest(plan), 'Completion plan mismatch')
    require(done['uids'] == [r['uid'] for r in selected], 'Incomplete/wrong shard membership')
    require(set(done['files']) == expected == {p.name for p in directory.glob('record*.json')}, 'Foreign/missing records')
    for name, expected_hash in done['files'].items():
        require(sha(directory / name) == expected_hash, 'Completed record changed: ' + name)
    return True


def gate_done(output, plan):
    path = output / 'gate.done.json'
    if not path.exists():
        return False
    done = read(path)
    require(done['complete'] and done['plan_sha256'] == digest(plan), 'Gate plan mismatch')
    require(sha(output / 'gate.json') == done['gate_sha256'], 'Gate evidence changed')
    return True
