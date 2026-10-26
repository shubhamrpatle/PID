#!/usr/bin/env python3
"""Verify and score completed paired outputs without modifying their frozen run."""
import argparse
from collections import Counter
import math
from pathlib import Path

from pid_common import (HERE, MODES, DATASETS, PROTOCOL, read, rows, sha, digest, require,
                        check_pair, check_software, finished, atomic, record_path, shard_folder)
from paired_eval import metric_record, render
from native_parser import reparse
from scoring import score, load_upstream


def image_dimensions(record):
    """Match PIL.Image.size types; do not truncate non-integral manifest dimensions."""
    dimensions = {}
    for name in ('width', 'height'):
        value = record[name]
        require(type(value) in (int, float) and math.isfinite(value)
                and value > 0 and value == int(value), 'Invalid image dimension: ' + name)
        dimensions[name] = int(value)
    return dict(record, **dimensions)


def run(args):
    source, output = args.results.resolve(), args.output.resolve()
    require(source != output and source not in output.parents,
            'Write the recovered report outside the original results directory')
    plan = read(source / 'plan.json')
    require(plan['protocol'] == PROTOCOL, 'Wrong paired protocol')
    ph = digest(plan)
    verified_sources = {}
    for remote, expected in plan['sources'].items():
        require('/LA_DET/' in remote, 'Unknown source path mapping: ' + remote)
        local = args.workspace / remote.split('/LA_DET/', 1)[1]
        require(sha(local) == expected, 'Frozen inference/scoring code changed: ' + str(local))
        verified_sources[str(local.resolve())] = expected
    check_software(read(source / 'software.json'), plan['expected_software'])
    data = {}
    for ds in DATASETS:
        manifest = args.manifests / (ds + '.jsonl')
        require(sha(manifest) == plan['datasets'][ds]['sha256'], 'Frozen manifest differs: ' + ds)
        data[ds] = rows(manifest)
        require(len(data[ds]) == plan['datasets'][ds]['records'], 'Manifest count differs')
    output.mkdir(parents=True, exist_ok=True)
    summary, comparisons, diagnostic_differences = {}, {}, []
    scorer = load_upstream()
    for ds in DATASETS:
        pairs, hashes = {}, {}
        for shard in range(plan['shards']):
            require(finished(source, ds, shard, plan['shards'], False, plan, data),
                    f'Incomplete shard {ds}/{shard}')
            folder = shard_folder(source, ds, shard, plan['shards'], False)
            for index, record in enumerate(data[ds][shard::plan['shards']]):
                path = record_path(folder, index)
                pair = read(path)
                check_pair(pair, record, ph)
                require(record['uid'] not in pairs, 'Duplicate saved image')
                pairs[record['uid']] = pair
                hashes[str(path.relative_to(source))] = sha(path)
        records = data[ds]
        require(set(pairs) == {r['uid'] for r in records}, 'Saved image coverage differs')
        ordered = [pairs[r['uid']] for r in records]
        comparisons[ds] = dict(records=len(ordered), discovery_answers_equal=True,
            discovery_points_equal=True, first_mode_counts=dict(Counter(p['order'][0] for p in ordered)),
            equal_extent_answers=sum([a['answer'] for a in p['modes']['packed']['refinements']] ==
                                     [a['answer'] for a in p['modes']['sequential']['refinements']] for p in ordered),
            equal_parsed_predictions=sum(p['modes']['packed']['predictions'] == p['modes']['sequential']['predictions']
                                         for p in ordered))
        for mode in MODES:
            predictions = [p['modes'][mode] for p in ordered]
            for record, prediction in zip(records, predictions):
                legacy = reparse('pid', record, prediction, 'boxes', True)
                replay = reparse('pid', image_dimensions(record), prediction, 'boxes', True)
                require(replay['predictions'] == prediction['predictions'],
                        f'Native replay still differs: {ds}/{record["uid"]}/{mode}')
                if legacy['predictions'] != prediction['predictions']:
                    require(len(legacy['predictions']) == len(prediction['predictions']), 'Prediction count differs')
                    changed = []
                    for index, (saved, previous) in enumerate(zip(prediction['predictions'], legacy['predictions'])):
                        if saved != previous:
                            require(saved.get('invalid_geometry') and previous.get('invalid_geometry'),
                                    'Replay changed a valid prediction')
                            require({k:v for k,v in saved.items() if k != 'raw_geometry'} ==
                                    {k:v for k,v in previous.items() if k != 'raw_geometry'},
                                    'Replay changed more than invalid diagnostic text')
                            changed.append(dict(index=index, saved=saved, prior_replay=previous))
                    diagnostic_differences.append(dict(dataset=ds, uid=record['uid'], mode=mode, changes=changed))
            # Score the original saved predictions, including every invalid candidate as FP.
            result, per = score(records, predictions, 'boxes', upstream=scorer)
            strict = [reparse('pid', image_dimensions(r), p, 'boxes', False) for r,p in zip(records, predictions)]
            result['strict_sensitivity'], _ = score(records, strict, 'boxes', upstream=scorer)
            result = metric_record(result, per, predictions, mode, ds, ph, False)
            result['pair_file_sha256'] = hashes
            folder = output / 'metrics' / ('pid_' + mode) / ds
            atomic(folder / 'metrics.json', result)
            atomic(folder / 'per_image.json', per)
            summary[f'pid_{mode}/boxes/{ds}'] = result
            print('VERIFIED AND SCORED', ds, mode, len(records), 'images', flush=True)
        for relative, expected in hashes.items():
            require(sha(source / relative) == expected, 'Input changed while scoring')
    audit = dict(source_results=str(source), source_plan_sha256=sha(source / 'plan.json'),
                 postprocessor_sha256=sha(__file__), verified_inference_sources=verified_sources,
                 full_pairs=sum(len(v) for v in data.values()),
                 replay_rule='Use integral image dimensions exactly as PIL.Image.size during inference',
                 diagnostic_text_differences=diagnostic_differences,
                 predictions_modified=False, timings_modified=False, gpu_inference_rerun=False)
    atomic(output / 'replay_audit.json', audit)
    atomic(output / 'summary.json', summary)
    atomic(output / 'pair_agreement.json', comparisons)
    render(output, summary, comparisons, False)
    print('Report:', output / 'RESULTS.md', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--manifests', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workspace', type=Path, default=HERE.parent)
    run(p.parse_args())


if __name__ == '__main__':
    main()
