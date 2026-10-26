import copy
from types import SimpleNamespace

import pytest

from refwave.direct_control import detection_question, isolated_conversations
from refwave.data import ClusterDataset
from refwave.episode import SCHEMA
from refwave.evaluate import Evaluator, inference_query


def episode():
    return dict(
        schema_version=SCHEMA, episode_id="test", task="detection", stage="stage1",
        dataset="test", image="image.jpg",
        direct=dict(conversations=[
            {"from": "human", "value": "<image>\nAll ships and bridges"},
            {"from": "gpt", "value": "<ref>ship</ref><box><1><2><3><4></box>"
             "<box><5><6><7><8></box><ref>bridge</ref><box>None</box>"},
        ]),
        round1=dict(branches=[dict(reference=r, polarity=p, branch_id=r,
            conversations=[{"from": "human", "value": "<image>\nPoint to " + r}])
            for r, p in [("ship", "positive"), ("bridge", "negative")]]),
        round2=dict(branches=[]),
    )


def test_exact_target_preservation_and_existing_negative():
    row = episode()
    before = copy.deepcopy(row)
    conv = isolated_conversations(row)
    assert len(conv) == 2
    assert ''.join(c[1]['value'] for c in conv) == row['direct']['conversations'][1]['value']
    assert conv[0][0]['value'] == '<image>\n' + detection_question('ship')
    assert conv[1][1]['value'] == '<ref>bridge</ref><box>None</box>'
    assert row == before
    row['round1']['branches'].reverse()
    with pytest.raises(ValueError, match='queried references'):
        isolated_conversations(row)


def test_dense_class_and_negative_are_not_capped_or_taken_from_round2():
    row = episode()
    answer = ('<ref>ship</ref>' + '<box><1><2><3><4></box>' * 1001
              + '<ref>bridge</ref><box>None</box>')
    row['direct']['conversations'][1]['value'] = answer
    row['round2']['branches'] = [{'conversations': [{'value': 'unused sampled target'}]}]
    conversations = isolated_conversations(row)
    assert ''.join(c[1]['value'] for c in conversations) == answer
    assert conversations[0][1]['value'].count('<box>') == 1001
    assert conversations[1][1]['value'] == '<ref>bridge</ref><box>None</box>'


def test_isolated_negative_only_queries_are_retained():
    row = episode()
    for branch in row['round1']['branches']:
        branch['polarity'] = 'negative'
    row['direct']['conversations'][1]['value'] = (
        '<ref>ship</ref><box>None</box><ref>bridge</ref><box>None</box>'
    )
    conversations = isolated_conversations(row)
    assert len(conversations) == 2
    assert all(c[1]['value'].endswith('<box>None</box>') for c in conversations)


@pytest.mark.parametrize('task', ['detection', 'grounding'])
def test_control_compiles_direct_only_and_leaves_grounding_unchanged(task, monkeypatch):
    import refwave.data as module
    row = episode()
    row['task'] = task
    ds = object.__new__(ClusterDataset)
    ds.rows, ds.method = [row], 'class_isolated'
    ds.max_sequence, ds.maximum_wave_size = 32768, 200
    ds._prepare_image = lambda *a: (None, None, 10)
    ds._compile = lambda image, conv, tokens, **kw: dict(conversations=conv, **kw)
    monkeypatch.setattr(module, 'split_waves', lambda branches, **kw: [branches])
    sample = ds[0]
    assert sample['round1_waves'] == sample['round2_waves'] == []
    assert len(sample['direct_waves'][0]) == (2 if task == 'detection' else 1)
    if task == 'grounding':
        assert sample['direct_waves'][0][0]['conversations'] == row['direct']['conversations']


def test_inference_queries_do_not_use_gt_and_control_never_refines(monkeypatch):
    import refwave.evaluate as module
    row = episode()
    query = inference_query(row)
    row['direct']['conversations'][1]['value'] = 'changed GT'
    row['round2'] = {'branches': ['hidden GT']}
    assert inference_query(row) == query
    for name in ('synchronize', 'reset_peak_memory_stats'):
        monkeypatch.setattr(module.torch.cuda, name, lambda *a: None)
    monkeypatch.setattr(module.torch.cuda, 'max_memory_allocated', lambda *a: 0)
    evaluator = object.__new__(Evaluator)
    evaluator.args = SimpleNamespace(method='class_isolated', max_new_tokens=100)
    evaluator.device = 'cpu'
    def generate(image, branches, **kw):
        assert [b.question for b in branches] == [detection_question(r) for r in ('ship', 'bridge')]
        return [dict(answer='<ref>ship</ref><box><1><2><3><4></box>'),
                dict(answer='<ref>bridge</ref><box>None</box>')], {}
    evaluator.points = SimpleNamespace(clear_image=lambda: None, generate=generate)
    out = evaluator.predict('image.jpg', query, 1)
    assert out['predictions'] == [dict(label='ship', hbox=[1, 2, 3, 4])]
    assert out['predicted_points'] == []
    assert out['round2_seconds'] is None
