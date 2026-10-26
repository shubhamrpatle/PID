"""Pinned LA-native decoding and strict-parser sensitivity on the same answers."""
import ast
from functools import lru_cache
import re

from common import HERE, canonical, sha
import parsing

SOURCE = HERE / 'evidence/inference_grounding_ddp.py'
SOURCE_SHA = '0a08b029a2633b6293eb0428a806c08041a352e497a917fea84d32d6472c40ac'
LA_MODELS = {'base_la', 'standard', 'pid', 'class_isolated'}


@lru_cache(maxsize=1)
def released_parser():
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Pinned LA inference parser hash mismatch')
    names = {'parse_bbox_with_labels', 'convert_normalized_point_to_absolute',
             'convert_normalized_bbox_to_absolute', 'parse_prediction'}
    nodes = [n for n in ast.parse(SOURCE.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError('Missing released LA parser functions')
    namespace = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['parse_prediction']


def la_native(raw, specs, size, kind, address_label=None):
    # PID's extent identity is an input address, not an inferred or GT-assigned label.
    if address_label is not None and '<ref>' not in raw:
        raw = f'<ref>{address_label}</ref>' + raw
    grouped = released_parser()(raw, *size)
    dimension = 4 if kind == 'boxes' else 2
    key = 'hbox' if kind == 'boxes' else 'point'
    predictions, errors = [], []
    for label, values in grouped.items():
        name = canonical(label, specs) or '__unmapped__'
        if name == '__unmapped__':
            errors.append('unknown_or_wrong_label')
        for coords in values:
            # Do not turn point responses into boxes or report collapsed boxes as usable.
            if parsing.geometry(coords, size, dimension) is None:
                predictions.append(parsing.invalid_prediction(name, coords))
                errors.append('invalid_geometry_after_native_parsing')
            else:
                predictions.append(dict(label=name, **{key: coords}))
    if not predictions and '<box>None</box>' not in raw:
        errors.append('no_native_geometry_or_null')
    return predictions, errors


def reparse(model, record, result, kind, native):
    """CPU-only; image dimensions and query labels, never GT geometry, enter parsing."""
    if model not in LA_MODELS:
        return result
    specs = result['query']['class_specs']
    size = record['width'], record['height']
    answers = result['refinements'] if model == 'pid' and kind == 'boxes' else result['answers']
    predictions, errors = [], []
    for answer in answers:
        requested = answer.get('queried_categories')
        group = [s for s in specs if s['canonical'] in requested] if requested is not None else specs
        if not group:
            raise ValueError('No requested categories for recorded answer')
        raw = answer['raw_response'] if 'raw_response' in answer else answer['answer']
        address_label = answer.get('label') if model in {'pid', 'class_isolated'} else None
        if model == 'class_isolated' and address_label is None and len(group) == 1:
            address_label = group[0]['canonical']
        if native:
            values, failures = la_native(raw, group, size, kind, address_label)
        else:
            default = address_label or (group[0]['canonical'] if len(group) == 1 else None)
            values, failures = parsing.la(raw, group, size, kind, default)
        predictions.extend(values)
        errors.extend(failures)
    return dict(result, predictions=predictions, invalid_items=len(errors))
