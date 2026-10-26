"""Final PID inference policy; historical decoders and runs remain unchanged."""
from functools import partial

import evaluate_refwave_cluster_stage2 as reference
import evaluate_refwave_allclass_points as all_addresses
import run_refwave_inference_ablations as ablation
import run_refwave_full_onepass_benchmarks as historical
from refwave_packed_geometry import packed_atomic

PROTOCOL_ID = 'pid_packed_all_v1'
POLICY = dict(
    protocol=PROTOCOL_ID, round1='all_requested_references_concurrent',
    round1_autoregressive=True, round2='packed_all',
    round2_execution='fused_one_pass', shared_image_prefix=True,
    geometry_prefix='box_only', compact_head=False, image_token_limit=6000,
    max_addresses=None, oom_fallback=False, raw_nms=False, raw_containment=False,
    evaluator_max_detections=None,
    timing='CUDA synchronized preprocessing + discovery + geometry + parsing; excludes loading, warmup, disk IO and scoring',
)


class PackedAllRoute:
    def __init__(self, decoder):
        self.decoder = decoder

    def decode_prepared_image(self, prepared, branches, *, requested_wave_size=None, allow_none=True):
        from zero_shot_analysis.pointrelay_pbd import base_causal_lm
        if not allow_none:
            raise ValueError('Final PID requires the trained box-or-None grammar')
        decoder = self.decoder
        original = decoder._decode_shared_atomic_blocks
        calls = []
        qwen = base_causal_lm(decoder.model.language_model).model
        handle = qwen.register_forward_hook(lambda *args: calls.append(1))
        decoder._decode_shared_atomic_blocks = partial(packed_atomic, decoder)
        try:
            results, stats = all_addresses.decode_all_addresses(decoder, prepared, branches)
        finally:
            decoder._decode_shared_atomic_blocks = original
            handle.remove()
        expected = int(bool(branches))
        if len(results) != len(branches) or len(calls) != expected or stats['qwen_total_forwards'] != expected:
            raise RuntimeError('Packed-all output/forward count violated the final protocol')
        stats.update(layout='packed_all', actual_round2_forwards=len(calls), protocol=PROTOCOL_ID)
        return results, stats


def make_decoders(worker):
    from zero_shot_analysis.fused_pivr_geometry import FusedPIVRGeometryDecoder
    point = reference.make_decoders(worker, optimized=True, round2_execution='fused_one_pass')[0]
    box = FusedPIVRGeometryDecoder(worker, image_token_limit=6000,
        visual_context='global_only', prefix_cache_mode='shared', geometry_prefix_mode='box_only',
        round2_execution='fused_one_pass', compact_head=False)
    return point, PackedAllRoute(box)


def validate_discovery(stats, references):
    if (stats['branches'] != references or stats['waves'] != 1
            or stats.get('fallback_qwen_full_prompt_forwards', 0) != 0):
        raise RuntimeError('Final PID requires one concurrent discovery group without prompt replay')


def infer_one(worker, decoders, row, image, budget):
    if row['task'] == 'pointing':
        # The frozen native pointing diagnostic has exactly 18 references and no Round 2.
        if len(row['class_specs']) != 18:
            raise ValueError('This pointing manifest requires all 18 benchmark classes')
        result = historical.point_one(worker, decoders, row, image, 'pid_packed_all', budget)
        validate_discovery(result['stats'], 18)
    else:
        query = reference.inference_query(row)
        if not query['references']:
            raise ValueError('A localization query requires at least one reference')
        result = ablation.infer_one(worker, decoders, query, image, 'refwave_full',
            dict(budget=budget, class_wave=len(query['references']), box_wave=1))
        validate_discovery(result['round1'], len(query['references']))
    result.update(protocol=PROTOCOL_ID, dataset=row['dataset'], image_id=row['image_id'])
    if row['task'] == 'grounding':
        result.update(sample_id=row['sample_id'], query=row['query'])
    return result
