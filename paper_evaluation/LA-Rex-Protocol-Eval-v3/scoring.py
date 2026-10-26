"""Released LA dense/point matchers, plus explicitly class-aware dense scores."""
import ast
from collections import defaultdict
from functools import lru_cache
from statistics import mean
from common import HERE, SCORER_SHA, sha

THRESHOLDS = tuple(n/100 for n in range(50, 100, 5))


def load_upstream(path=None):
    path = path or HERE/'evidence/other_metric.py'
    if sha(path) != SCORER_SHA:
        raise ValueError('Released scorer SHA mismatch')
    names = {'calculate_iou', 'calculate_detection_metrics', 'is_point_in_bbox', 'calculate_pointing_metrics'}
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(nodes) != len(names):
        raise ValueError('Missing upstream functions')
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    # Memoization changes no arithmetic or match order; clear between images.
    cached = lru_cache(maxsize=1000000)(namespace['calculate_iou'])
    namespace['calculate_iou'] = lambda a, b: cached(tuple(a), tuple(b))
    return namespace, cached


def prf(tp, fp, fn):
    p, r = tp/(tp+fp) if tp+fp else 0., tp/(tp+fn) if tp+fn else 0.
    return dict(precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.)


def aggregate(items):
    p, r = mean(x['precision'] for x in items), mean(x['recall'] for x in items)
    totals = {k: sum(x[k] for x in items) for k in ('tp', 'fp', 'fn')}
    return dict(precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.,
                micro=dict(**totals, **prf(**totals)))


def select_predictions(values, threshold=None, native_cap=None):
    if native_cap is not None:
        if values and all('native_rank' in p for p in values):
            values=[p for p in values if p['native_rank']<native_cap]
        else:
            values=sorted(values,key=lambda p:-p['score'])[:native_cap]
    if threshold is not None:
        values=[p for p in values if p['score']>threshold]
    return values


def score(records, outputs, kind, threshold=None, native_cap=None, upstream=None):
    if len(records) != len(outputs) or not records:
        raise ValueError('Incomplete scoring inputs')
    functions, cache = upstream or load_upstream()
    per = []
    for row, result in zip(records, outputs):
        if row['uid'] != result['uid']:
            raise ValueError('Prediction order/identity mismatch')
        gt, pred = defaultdict(list), defaultdict(list)
        for obj in row['gt']:
            gt[obj['label']].append(obj['hbox'])
        values = select_predictions(result['predictions'],threshold,native_cap)
        invalid_count=sum(bool(obj.get('invalid_geometry')) for obj in values)
        for obj in values:
            if obj.get('invalid_geometry'): continue
            pred[obj['label']].append(obj['hbox' if kind == 'boxes' else 'point'])
        item = dict(uid=row['uid'], invalid_geometry_outputs=invalid_count, scores={})
        for t in (THRESHOLDS if kind == 'boxes' else (None,)):
            if kind == 'boxes':
                r, p, ngt, npr = functions['calculate_detection_metrics'](gt, pred, t)
                npr+=invalid_count
                matches = sum(round(functions['calculate_detection_metrics']({c: b}, {c: pred[c]}, t)[0]*len(b))
                              for c, b in gt.items())
                released_tp = round(r*ngt)
                item['scores'][str(t)] = dict(
                    released=dict(tp=released_tp, fp=npr-released_tp, fn=ngt-released_tp, **prf(released_tp,npr-released_tp,ngt-released_tp)),
                    class_aware=dict(tp=matches, fp=npr-matches, fn=ngt-matches, **prf(matches, npr-matches, ngt-matches)))
            else:
                r, p, ngt, npr = functions['calculate_pointing_metrics'](gt, pred)
                npr+=invalid_count
                tp = round(r*ngt)
                item['scores']['point'] = dict(released=dict(tp=tp, fp=npr-tp, fn=ngt-tp, **prf(tp,npr-tp,ngt-tp)))
        per.append(item)
        cache.cache_clear()
    modes = ('released', 'class_aware') if kind == 'boxes' else ('released',)
    scores = {}
    for mode in modes:
        scores[mode] = {t: aggregate([x['scores'][t][mode] for x in per]) for t in per[0]['scores']}
        if kind == 'boxes':
            scores[mode]['mean_f1_50_95'] = mean(v['f1'] for v in scores[mode].values())
    return dict(records=len(records), threshold=threshold, native_cap=native_cap, scores=scores,
                invalid_geometry_outputs=sum(x['invalid_geometry_outputs'] for x in per)), per
