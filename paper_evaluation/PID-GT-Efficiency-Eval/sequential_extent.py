"""One-address calls to the same frozen packed extent kernel."""
from functools import partial


def sequential_atomic(decoder, prefix_ids, suffix_rows, branch_visuals,
                      prefix_key_values, branches, **kwargs):
    from refwave_packed_geometry import packed_atomic
    if not len(suffix_rows) == len(branch_visuals) == len(branches):
        raise ValueError('Inconsistent address inputs')
    outputs = []
    for row, visual, branch in zip(suffix_rows, branch_visuals, branches):
        result = packed_atomic(decoder, prefix_ids, [row], [visual],
                               prefix_key_values, [branch], **kwargs)
        if len(result) != 1:
            raise RuntimeError('Each address must return exactly one box-or-None response')
        outputs.extend(result)
    return outputs


class SequentialExtentRoute:
    def __init__(self, decoder):
        self.decoder = decoder

    def decode_prepared_image(self, prepared, branches, *, requested_wave_size=None, allow_none=True):
        import pid_packed_protocol as packed
        from zero_shot_analysis.pointrelay_pbd import base_causal_lm
        if not allow_none:
            raise ValueError('Keep the trained box-or-None grammar')
        decoder = self.decoder
        original = decoder._decode_shared_atomic_blocks
        calls = []
        qwen = base_causal_lm(decoder.model.language_model).model
        handle = qwen.register_forward_hook(lambda *args: calls.append(1))
        decoder._decode_shared_atomic_blocks = partial(sequential_atomic, decoder)
        try:
            results, stats = packed.all_addresses.decode_all_addresses(decoder, prepared, branches)
        finally:
            decoder._decode_shared_atomic_blocks = original
            handle.remove()
        n = len(branches)
        if len(results) != n or len(calls) != n:
            raise RuntimeError('Expected exactly one Qwen extent forward per address')
        stats.update(protocol='pid_sequential_extent_v1', layout='sequential_extent', parallel_width=1,
                     requested_wave_size=1, effective_wave_sizes=[1] * n, waves=n,
                     qwen_fused_prompt_geometry_forwards=n, qwen_total_forwards=n,
                     actual_round2_forwards=n, oom_policy='raise_without_fallback')
        return results, stats

