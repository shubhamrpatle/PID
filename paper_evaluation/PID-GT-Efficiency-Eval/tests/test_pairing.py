"""CPU regression checks; these are not part of the remote launch path."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pid_common as common
import paired_eval as evaluation
from sequential_extent import sequential_atomic
from native_parser import reparse
from scoring import score


def local_module(name):
    spec = importlib.util.spec_from_file_location('paired_' + name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(uid='r0', ds='DIOR'):
    return dict(uid=uid, suite=ds, width=100, height=100,
                class_specs=[dict(canonical='ship', prompt='ship'),
                             dict(canonical='vehicle', prompt='vehicle')],
                gt=[dict(label='ship', hbox=[10, 10, 30, 30])])


def fake_result(mode, record, ph='plan', n=2):
    result = dict(uid=record['uid'], suite=record['suite'], query=common.query_only(record),
        execution_mode=mode, plan_sha256=ph, round1=dict(branches=1, waves=1),
        image_encodes=1, prefix_forwards=1,
        round2=dict(cross_round_cache_reuse=True, actual_round2_forwards=int(n > 0) if mode == 'packed' else n,
                    qwen_total_forwards=int(n > 0) if mode == 'packed' else n),
        points=[dict(label='ship', point=[20, 20]) for _ in range(n)],
        refinements=[dict(address_id=str(i), label='ship', answer='<box><100><100><300><300></box>') for i in range(n)],
        point_answers=[dict(label='ship', question='Point to ships.', answer='<box><200><200></box>',
                            seed=123, tokens_generated=4)],
        latency_seconds=2., generated_tokens=4 + 6*n, generated_text_tokens=4, geometry_tokens=6*n,
        saturated_queries=0, peak_gpu_mb=1024,
        component_seconds=dict(image_and_prefix=.5, discovery_and_address_construction=.5,
                               extent_and_legacy_parsing=.75, native_parser_and_outer_overhead=.25))
    result['predictions'] = reparse('pid', record, result, 'boxes', True)['predictions']
    return result


def pair(record, ph='plan', n=2):
    return dict(uid=record['uid'], suite=record['suite'], plan_sha256=ph,
                order=list(common.MODES), modes={m: fake_result(m, record, ph, n) for m in common.MODES})


class Contracts(unittest.TestCase):
    def test_query_excludes_geometry_and_counts(self):
        query = common.query_only(row())
        self.assertEqual(set(query), {'uid', 'suite', 'class_specs'})
        self.assertEqual([s['canonical'] for s in query['class_specs']], ['ship'])

    def test_valid_pair(self):
        common.check_pair(pair(row()), row(), 'plan')

    def test_empty_predictions(self):
        common.check_pair(pair(row(), n=0), row(), 'plan')

    def test_extent_numerical_differences_allowed(self):
        p = pair(row())
        p['modes']['sequential']['refinements'][0]['answer'] = '<box><101><100><300><300></box>'
        common.check_pair(p, row(), 'plan')

    def test_rejects_changed_discovery(self):
        p = pair(row())
        p['modes']['sequential']['points'][0]['point'] = [21, 20]
        with self.assertRaisesRegex(ValueError, 'Round-1 points differ'):
            common.check_pair(p, row(), 'plan')

    def test_rejects_changed_random_seed(self):
        p = pair(row())
        p['modes']['sequential']['point_answers'][0]['seed'] = 124
        with self.assertRaisesRegex(ValueError, 'answers/RNG differ'):
            common.check_pair(p, row(), 'plan')

    def test_rejects_ground_truth_in_query(self):
        p = pair(row())
        p['modes']['packed']['query']['gt'] = row()['gt']
        with self.assertRaisesRegex(ValueError, 'Inference query differs'):
            common.check_pair(p, row(), 'plan')

    def test_rejects_missing_or_misowned_extent(self):
        for change in ('drop', 'wrong_label', 'wrong_address'):
            p = pair(row())
            extents = p['modes']['packed']['refinements']
            if change == 'drop':
                extents.pop()
            elif change == 'wrong_label':
                extents[0]['label'] = 'vehicle'
            else:
                extents[0]['address_id'] = '7'
            with self.subTest(change=change), self.assertRaises(ValueError):
                common.check_pair(p, row(), 'plan')

    def test_rejects_extra_prefix_and_split_packing(self):
        for field in ('prefix_forwards', 'image_encodes', 'round2'):
            p = pair(row())
            if field == 'round2':
                p['modes']['packed'][field]['actual_round2_forwards'] = 2
            else:
                p['modes']['packed'][field] = 2
            with self.subTest(field=field), self.assertRaises(ValueError):
                common.check_pair(p, row(), 'plan')

    def test_alternating_order(self):
        self.assertEqual(common.order_for(0), ('packed', 'sequential'))
        self.assertEqual(common.order_for(1), ('sequential', 'packed'))

    def test_pair_calls_both_modes_with_query_only(self):
        calls = []
        r = row()
        def timed(model, mode, query, image, ph):
            self.assertEqual(query, common.query_only(r))
            calls.append(mode)
            return fake_result(mode, r, ph)
        evaluation.pair_one(None, r, None, 1, 'plan', timed=timed)
        self.assertEqual(calls, ['sequential', 'packed'])

    def test_sequential_kernel_preserves_prefix_and_order(self):
        prefix, kv = object(), object()
        calls = []
        def kernel(decoder, p, suffix, visuals, cache, branches, **kwargs):
            self.assertIs(p, prefix)
            self.assertIs(cache, kv)
            self.assertEqual(len(suffix), 1)
            self.assertEqual(kwargs, dict(temperature=0.))
            calls.append((suffix, visuals, branches))
            return branches
        with patch.dict(sys.modules, {'refwave_packed_geometry': SimpleNamespace(packed_atomic=kernel)}):
            result = sequential_atomic(None, prefix, ['s0', 's1'], ['v0', 'v1'], kv,
                                       ['b0', 'b1'], temperature=0.)
            self.assertEqual(result, ['b0', 'b1'])
            self.assertEqual(len(calls), 2)
            self.assertEqual(sequential_atomic(None, prefix, [], [], kv, []), [])
            with self.assertRaisesRegex(ValueError, 'Inconsistent'):
                sequential_atomic(None, prefix, ['s0'], [], kv, [])

    def test_outer_timing_does_not_double_count_preparation(self):
        result = fake_result('packed', row())
        result.update(preparation_seconds=.5, round1_seconds=1., round2_seconds=.75, latency_seconds=1.75)
        class Context:
            def __enter__(self): return self
            def __exit__(self, *args): return False
        torch = SimpleNamespace(inference_mode=Context, cuda=SimpleNamespace(
            synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 1024**3))
        model = SimpleNamespace(infer=lambda *args: result)
        with patch.dict(sys.modules, {'torch': torch}), patch.object(evaluation.time, 'perf_counter', side_effect=[0., 2.]):
            out = evaluation.timed_one(model, 'packed', common.query_only(row()), None, 'plan')
        self.assertEqual(out['latency_seconds'], 2.)
        self.assertEqual(out['component_seconds']['discovery_and_address_construction'], .5)
        self.assertEqual(sum(out['component_seconds'].values()), 2.)

    def test_metric_throughput_and_all_iou_thresholds(self):
        r = row()
        output = fake_result('packed', r)
        metrics, per = score([r], [output], 'boxes')
        out = evaluation.metric_record(metrics, per, [output], 'packed', 'DIOR', 'plan', False)
        self.assertEqual(out['valid_geometry_outputs_per_second'], 1.)
        self.assertEqual(out['correct_outputs_per_second'], .5)
        self.assertEqual(out['seconds_per_complete_image'], 2.)
        self.assertEqual(len(out['scores']['class_aware']), 11)
        self.assertEqual(out['scores']['class_aware']['0.5']['micro']['tp'], 1)

    def test_software_mismatch_rejected(self):
        expected = {key: 'same' for key in common.SOFTWARE_KEYS}
        common.check_software(expected, expected)
        with self.assertRaisesRegex(ValueError, 'Environment differs'):
            common.check_software(dict(expected, torch='different'), expected)

    def test_matching_hardware_only(self):
        queue = local_module('remote')
        devices = [('0', 'a', 'NVIDIA A100-PCIE-40GB', '40960'),
                   ('1', 'b', 'NVIDIA A100-PCIE-40GB', '40960'),
                   ('2', 'c', 'NVIDIA A100-SXM4-80GB', '81920')]
        found = queue.eligible_devices(devices, dict(name='NVIDIA A100-PCIE-40GB', memory_mb=40960), [1, 2])
        self.assertEqual(found, {'b'})

    def test_gpu_pool_waits_for_idle_and_caps_workers(self):
        import gpu_pool
        devices = [dict(index=str(i), uuid=f'GPU-{i}', free_mb=40000, used_mb=0,
                        utilization=0, compute_pids=[]) for i in range(8)]
        devices[0]['compute_pids'] = [99999]
        devices[1]['used_mb'] = 12000
        with tempfile.TemporaryDirectory() as temp, patch.object(gpu_pool, 'probe', return_value=devices), \
                patch.dict('os.environ', {}, clear=True):
            pool = gpu_pool.GpuPool(max_gpus=4, lock_dir=temp, max_used_mb=500)
            self.assertEqual(pool.available(set()), [])
            selected = pool.available(set())
            self.assertEqual([d['index'] for d in selected], ['2', '3', '4', '5'])
            self.assertEqual(len(pool.available({'GPU-2', 'GPU-3', 'GPU-4'})), 1)
            self.assertEqual(pool.available({'GPU-2', 'GPU-3', 'GPU-4', 'GPU-5'}), [])

    def test_gpu_lease_prevents_two_workers(self):
        import gpu_pool
        device = dict(index='2', uuid='GPU-test', free_mb=40000, used_mb=0, utilization=0, compute_pids=[])
        with tempfile.TemporaryDirectory() as temp, patch.object(gpu_pool, 'probe', return_value=[device]), \
                patch.dict('os.environ', {}, clear=True):
            pool = gpu_pool.GpuPool(lock_dir=temp)
            first = pool.claim(device)
            self.assertIsNotNone(first)
            try:
                self.assertIsNone(pool.claim(device))
            finally:
                first.close()
            next_lease = pool.claim(device)
            self.assertIsNotNone(next_lease)
            next_lease.close()

    def test_gpu_allocation_is_respected(self):
        import gpu_pool
        devices = [dict(index=str(i), uuid=f'GPU-{i}') for i in range(8)]
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '2,3'}, clear=True):
            self.assertEqual([d['index'] for d in gpu_pool.allocated(devices)], ['2', '3'])

    def test_queue_enforces_four_worker_maximum(self):
        queue = local_module('remote')
        with self.assertRaisesRegex(ValueError, '1-4'):
            queue.launch(SimpleNamespace(max_gpus=5))

    def test_prepare_import(self):
        self.assertTrue(callable(local_module('prepare').prepare))

    def test_merge_and_resume_completeness(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            data = {ds: [row(ds=ds)] for ds in common.DATASETS}
            plan = dict(protocol=common.PROTOCOL, sources={}, shards=1, datasets={})
            for ds, records in data.items():
                file = output / (ds + '.jsonl')
                import json
                file.write_text(json.dumps(records[0]) + '\n')
                plan['datasets'][ds] = dict(path=str(file), sha256=common.sha(file))
            common.atomic(output / 'plan.json', plan)
            ph = common.digest(plan)
            for ds, records in data.items():
                folder = common.shard_folder(output, ds, 0, 1, False)
                file = common.record_path(folder, 0)
                common.atomic(file, pair(records[0], ph))
                common.atomic(folder / 'done.json', dict(complete=True, plan_sha256=ph,
                    uids=['r0'], files={file.name: common.sha(file)}))
                self.assertTrue(common.finished(output, ds, 0, 1, False, plan, data))
            evaluation.merge(SimpleNamespace(output=output, smoke=False))
            self.assertEqual(len(common.read(output / 'summary.json')), 4)
            self.assertTrue((output / 'RESULTS.md').exists())
            agreement = common.read(output / 'pair_agreement.json')
            self.assertEqual(agreement['DIOR']['equal_parsed_predictions'], 1)
            file = common.record_path(common.shard_folder(output, 'DIOR', 0, 1, False), 0)
            file.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'predictions changed'):
                common.finished(output, 'DIOR', 0, 1, False, plan, data)


if __name__ == '__main__':
    unittest.main()
