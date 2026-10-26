"""Local CPU tests; no downloads, remote modifications, or GPU inference."""
import ast
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
WORKSPACE = HERE.parent
sys.path.insert(0, str(HERE))
import rerun_common as common
import address_eval as evaluation
import launch
from stable_rope import install_fp32_rope_cache
from native_parser import reparse
from scoring import score
import gpu_pool


def normalize(point, width, height):
    return tuple(max(0, min(1000, int(round(max(0., min(float(size), float(v))) / max(1., float(size)) * 1000))))
                 for v, size in zip(point, (width, height)))


def row(uid='r0', ds='DIOR'):
    return dict(uid=uid, suite=ds, width=100, height=100,
        class_specs=[dict(canonical='ship', prompt='ship'), dict(canonical='vehicle', prompt='vehicle')],
        gt=[dict(label='ship', hbox=[10, 10, 30, 30])])


def output(mode, record):
    points, audit = evaluation.annotation_points(record, normalize)
    result = dict(uid=record['uid'], suite=record['suite'], query=common.query_only(record),
        address_source=mode, rope_fix_modules=2, image_encodes=1, prefix_forwards=1,
        round1=dict(branches=1, waves=1), round1_skipped=mode == 'gt_address',
        points=points, oracle_audit=audit,
        refinements=[dict(address_id='0', label='ship', answer='<ref>ship</ref><box><100><100><300><300></box>')],
        round2=dict(layout='packed_all', cross_round_cache_reuse=True, actual_round2_forwards=1, qwen_total_forwards=1),
        latency_seconds=1., generated_text_tokens=4 if mode == 'predicted' else 0,
        geometry_tokens=6, generated_tokens=10 if mode == 'predicted' else 6, saturated_queries=0,
        peak_gpu_mb=1000, component_seconds=dict(image_and_prefix=.5,
            discovery_and_address_construction=0., extent_and_legacy_parsing=.4, native_parser_and_outer_overhead=.1))
    result.update(reparse('pid', record, result, 'boxes', True))
    return result


def pair(record, plan):
    return dict(uid=record['uid'], suite=record['suite'], order=list(common.MODES),
        plan_sha256=common.digest(plan), modes={m: output(m, record) for m in common.MODES})


class Addresses(unittest.TestCase):
    def test_normal_query_excludes_geometry_count_and_absent_classes(self):
        self.assertEqual(common.query_only(row()), dict(uid='r0', suite='DIOR',
            class_specs=[dict(canonical='ship', prompt='ship')]))

    def test_exact_center(self):
        points, audit = evaluation.annotation_points(row(), normalize)
        self.assertEqual(points, [dict(label='ship', point=[20., 20.], source='point')])
        self.assertEqual(audit['annotations'][0]['normalized_point'], [200, 200])

    def test_visible_center_preserves_original_gt(self):
        r = row()
        r['gt'][0]['hbox'] = [-10, 10, 30, 30]
        original = deepcopy(r)
        points, audit = evaluation.annotation_points(r, normalize)
        self.assertEqual(points[0]['point'], [15., 20.])
        self.assertEqual(audit['clipped_visible_centres'], 1)
        self.assertEqual(r, original)

    def test_duplicates_not_dropped(self):
        r = row()
        r['gt'] *= 2
        points, audit = evaluation.annotation_points(r, normalize)
        self.assertEqual(len(points), 2)
        self.assertEqual(audit['duplicate_addresses'], 1)

    def test_invalid_gt_rejected(self):
        for box in ([1, 2, 1, 3], [110, 2, 120, 3], [float('nan'), 2, 5, 6]):
            r = row()
            r['gt'][0]['hbox'] = box
            with self.subTest(box=box), self.assertRaises(ValueError):
                evaluation.annotation_points(r, normalize)

    def test_quantization_must_stay_in_original_box(self):
        r = row()
        r['gt'][0]['hbox'] = [10.04, 20.04, 10.049, 20.049]
        with self.assertRaisesRegex(ValueError, 'Quantized'):
            evaluation.annotation_points(r, normalize)

    def test_reference_order_and_annotation_order(self):
        r = row()
        r['gt'].insert(0, dict(label='vehicle', hbox=[40, 40, 50, 50]))
        points, audit = evaluation.annotation_points(r, normalize)
        self.assertEqual([p['label'] for p in points], ['ship', 'vehicle'])
        self.assertEqual([a['gt_index'] for a in audit['annotations']], [1, 0])

    def test_all_frozen_gt_addresses_match_previous_oracle_exactly(self):
        total = 0
        for ds in common.DATASETS:
            records = common.rows(WORKSPACE / 'remote_results/LA-Rex-Protocol-Eval-v3/assets' / (ds + '.jsonl'))
            folder = WORKSPACE / 'remote_results/PID-Discovery-Ablations-Eval/results/results/boxes/pid_gt' / ds
            saved = {r['uid']: r for p in folder.glob('shard*.jsonl') for r in common.rows(p)}
            self.assertEqual(len(saved), len(records))
            for record in records:
                points, audit = evaluation.annotation_points(record, normalize)
                self.assertEqual(points, saved[record['uid']]['points'])
                self.assertEqual(audit, saved[record['uid']]['oracle_audit'])
                total += len(points)
        self.assertEqual(total, 32708)


class Contracts(unittest.TestCase):
    def test_new_plan_preserves_previous_settings_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, dest = root / 'old', root / 'new'
            original = dict(protocol=common.paired.PROTOCOL,
                models={'pid': dict(sha256=common.CHECKPOINTS['pid'][1],
                    step=common.CHECKPOINTS['pid'][2], files={})}, shards=4, warmup_records=1,
                generation=dict(temperature=.7, top_p=.9, repetition_penalty=1.1, mode='hybrid'),
                budget=8192, image_token_limit=6000, sources={}, datasets={'DIOR': {'sha256': 'frozen'}})
            common.atomic(source / 'plan.json', original)
            original_hash = common.sha(source / 'plan.json')
            with patch.dict('os.environ', PID_PAIRED_RESULTS=str(source)), \
                 patch.object(common.paired, 'load_plan', return_value=(original, {'DIOR': [row()]})), \
                 patch.object(common, 'DATASETS', {'DIOR': (1, 1, 1)}), patch('builtins.print'):
                first, _ = common.prepare(dest)
                second, _ = common.prepare(dest)
                self.assertEqual(first, second)
                self.assertEqual(first['models'], original['models'])
                self.assertEqual(first['datasets'], original['datasets'])
                self.assertEqual(first['source_plan_sha256'], original_hash)
                self.assertEqual(common.sha(source / 'plan.json'), original_hash)
                self.assertEqual(original['sources'], {})
                for forbidden in (source, source / 'child'):
                    with self.assertRaisesRegex(ValueError, 'new output'):
                        common.prepare(forbidden)

    def test_valid_pair(self):
        common.check_pair(pair(row(), {}), row(), {})

    def test_different_predicted_addresses_are_allowed(self):
        p = pair(row(), {})
        p['modes']['predicted']['points'][0]['point'] = [21, 21]
        common.check_pair(p, row(), {})

    def test_invalid_contracts_rejected(self):
        for change in ('geometry_in_query', 'no_fix', 'two_forwards', 'wrong_owner', 'missing_gt'):
            p = pair(row(), {})
            r = p['modes']['gt_address']
            if change == 'geometry_in_query': r['query']['gt'] = row()['gt']
            if change == 'no_fix': r['rope_fix_modules'] = 0
            if change == 'two_forwards': r['round2']['actual_round2_forwards'] = 2
            if change == 'wrong_owner': r['refinements'][0]['label'] = 'vehicle'
            if change == 'missing_gt': r['oracle_audit']['annotations'] = []
            with self.subTest(change=change), self.assertRaises(ValueError):
                common.check_pair(p, row(), {})

    def test_alternating_order(self):
        calls = []
        def timed(model, mode, r, image):
            calls.append(mode)
            return output(mode, r)
        evaluation.pair_one(None, row(), None, 1, {}, timed)
        self.assertEqual(calls, ['gt_address', 'predicted'])

    def test_complete_records_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, plan = Path(tmp), dict(shards=1)
            r = row()
            data = {'DIOR': [r]}
            directory = common.folder(root, 'DIOR', 0, 1)
            path = common.record_path(directory, 0)
            common.atomic(path, pair(r, plan))
            common.atomic(directory / 'done.json', dict(complete=True, plan_sha256=common.digest(plan),
                uids=[r['uid']], files={path.name: common.sha(path)}))
            self.assertTrue(common.finished(root, 'DIOR', 0, plan, data))
            common.atomic(path, {})
            with self.assertRaisesRegex(ValueError, 'changed'):
                common.finished(root, 'DIOR', 0, plan, data)

    def test_full_launch_requires_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'gate'):
                launch.jobs_for_phase('full', Path(tmp), dict(shards=4), {})

    def test_completed_shards_are_not_requeued(self):
        with patch.object(launch, 'gate_done', return_value=True), \
             patch.object(launch, 'finished', side_effect=lambda output, ds, shard, plan, data: ds == 'DIOR'):
            self.assertEqual(launch.jobs_for_phase('full', Path('/unused'), dict(shards=4), {}),
                             [('worker', 'DOTAv2', i) for i in range(4)])

    def test_gpu_pool_caps_concurrency_and_respects_occupancy_and_allocation(self):
        devices = [dict(index=str(i), uuid=f'GPU-{i}', free_mb=40500, used_mb=100,
                        utilization=0, compute_pids=[]) for i in range(8)]
        devices[7]['compute_pids'] = [999]
        with tempfile.TemporaryDirectory() as tmp, patch.object(gpu_pool, 'probe', return_value=devices), \
             patch.dict('os.environ', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'):
            pool = gpu_pool.GpuPool(max_gpus=4, min_free_mb=30000, max_used_mb=500, lock_dir=tmp)
            self.assertEqual(pool.available(set()), [])
            self.assertEqual(len(pool.available(set())), 4)
            self.assertEqual(len(pool.available({'GPU-0', 'GPU-1', 'GPU-2'})), 1)
            self.assertEqual(pool.available({'GPU-0', 'GPU-1', 'GPU-2', 'GPU-3'}), [])
            with patch.dict('os.environ', CUDA_VISIBLE_DEVICES='5,7'):
                self.assertEqual([d['uuid'] for d in pool.available(set())], ['GPU-5'])

    def test_merge_uses_native_and_strict_scorer_and_signed_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, plan = Path(tmp), dict(shards=1)
            r = row()
            directory = common.folder(root, 'DIOR', 0, 1)
            path = common.record_path(directory, 0)
            common.atomic(path, pair(r, plan))
            common.atomic(directory / 'done.json', dict(complete=True, plan_sha256=common.digest(plan),
                uids=[r['uid']], files={path.name: common.sha(path)}))
            with patch.object(evaluation, 'load_plan', return_value=(plan, {'DIOR': [r]})), \
                 patch.object(evaluation, 'gate_done', return_value=True), patch('builtins.print'):
                evaluation.merge(SimpleNamespace(output=root))
            summary = common.read(root / 'summary.json')
            self.assertEqual(summary['gt_minus_predicted_pp']['DIOR'], [0., 0., 0.])
            result = summary['results']['predicted/DIOR']
            self.assertEqual(result['scores']['class_aware']['0.5']['f1'], 1.)
            self.assertEqual(result['correct_outputs_per_second'], 1.)
            self.assertIn('strict_sensitivity', result)
            self.assertFalse(summary['results']['gt_address/DIOR']['deployment_latency'])
            self.assertTrue(common.read(root / 'all.done.json')['complete'])

    def test_model_loading_patches_before_first_inference(self):
        fake = SimpleNamespace(model=SimpleNamespace(model=object()))
        with patch.object(evaluation, 'Backend', return_value=fake) as loader, \
             patch('stable_rope.install_fp32_rope_cache', return_value=36) as install:
            backend = evaluation.AddressBackend({})
            loader.assert_called_once_with('pid', {})
            install.assert_called_once_with(fake.model.model)
            self.assertEqual(backend.fixed_modules, 36)

    def test_sequential_probe_restores_packed_route_on_failure(self):
        backend = object.__new__(evaluation.AddressBackend)
        original = (object(), SimpleNamespace(decoder=object()))
        backend.backend = SimpleNamespace(decoders=original)
        with patch.object(backend, 'oracle', side_effect=RuntimeError('simulated failure')):
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                backend.oracle_sequential({}, None, [])
        self.assertIs(backend.backend.decoders, original)

    def test_sequential_probe_reports_different_answers(self):
        packed = output('gt_address', row())
        sequential = deepcopy(packed)
        sequential['refinements'][0]['answer'] = '<box>None</box>'
        comparison = evaluation.extent_comparison(packed, sequential)
        self.assertEqual(comparison['changed_answers'], 1)
        self.assertEqual(comparison['identical_answers'], 0)

    def test_gpu_inventory_excludes_wrong_hardware(self):
        actual = [('0', 'GPU-a', 'A100', '40000'), ('1', 'GPU-b', 'A100', '80000'),
                  ('2', 'GPU-c', 'A100', '40000')]
        self.assertEqual(launch.eligible_devices(actual, dict(name='A100', memory_mb=40000), [0, 1]), {'GPU-a'})

    def test_child_command_uses_selected_interpreter_and_only_job_args(self):
        command = launch.command_for(('worker', 'DOTAv2', 3), Path('/tmp/output'))
        self.assertEqual(command[0], sys.executable)
        self.assertIn('worker', command)
        self.assertEqual(command[-4:], ['--dataset', 'DOTAv2', '--shard', '3'])

    def test_shell_help_needs_no_old_python_env_helper(self):
        result = subprocess.run(['bash', str(HERE / 'run.sh'), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('python_env.sh', (HERE / 'run.sh').read_text())


class Rotary(unittest.TestCase):
    def test_real_tiny_qwen_bf16_packed_forward_is_stable_after_growth(self):
        import torch
        sys.path.insert(0, str(WORKSPACE / 'Zero_shot_ anlysis_LA/scripts'))
        from test_refwave_fused_round2 import model as make_model
        from test_refwave_packed_geometry import packed
        from zero_shot_analysis.pixel_pivr_wave import _legacy_key_values
        differences = {}
        for fixed in (False, True):
            decoder, qwen, _, _ = make_model.__wrapped__()
            qwen.to(torch.bfloat16)
            if fixed:
                install_fp32_rope_cache(qwen)
            prefix = torch.tensor([1, 2] + [3] * 298)
            with torch.inference_mode():
                cache = _legacy_key_values(qwen(input_ids=prefix[None],
                    visual_features=torch.ones(1, 32, dtype=torch.bfloat16), image_token_index=2,
                    attention_mask=torch.ones(1, len(prefix)), position_ids=torch.arange(len(prefix))[None],
                    use_cache=True).past_key_values)
                state = (decoder, qwen, prefix, cache)
                suffix = torch.tensor([11, 12])
                before = packed(state, [suffix])
                packed(state, [suffix] * 100)
                after = packed(state, [suffix])
            differences[fixed] = float((before.float() - after.float()).abs().max())
        self.assertGreater(differences[False], 0.)
        self.assertEqual(differences[True], 0.)

    def test_actual_rotary_class_preserves_cache_after_growth(self):
        import torch
        source = WORKSPACE / 'Zero_shot_ anlysis_LA/model/LocateAnything-3B/checkpoint/modeling_qwen2.py'
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'Qwen2RotaryEmbedding')
        namespace = dict(torch=torch, nn=torch.nn)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        module = namespace[node.name](128, max_position_embeddings=32768, base=1000000).to(torch.bfloat16)
        old = module.cos_cached[:2048].clone()
        self.assertEqual(install_fp32_rope_cache(module), 1)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            module(torch.empty(1, dtype=torch.bfloat16), seq_len=40000)
        self.assertTrue(torch.equal(old, module.cos_cached[:2048]))


if __name__ == '__main__':
    unittest.main()
