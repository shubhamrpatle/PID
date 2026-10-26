"""Native-coordinate parsers; never infer boxes from points or vice versa."""
import json
import math
import re
from common import canonical


def gdino_label(phrase, specs):
    # Official GroundingDINO Model.phrases2classes: first requested name contained in the phrase.
    return next((s['canonical'] for s in specs if s['prompt'] in phrase),None)


def geometry(coords, size, dimension, normalized=False):
    if (not isinstance(coords, (list, tuple)) or len(coords) != dimension or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in coords)):
        return None
    if normalized:
        if any(x < 0 or x > 1000 for x in coords):
            return None
        coords = [v*size[i % 2]/1000 for i, v in enumerate(coords)]
    elif any(v < 0 or v > size[i % 2] for i, v in enumerate(coords)):
        return None
    if dimension == 4 and (coords[0] >= coords[2] or coords[1] >= coords[3]):
        return None
    return list(coords)


def invalid_prediction(label, raw_geometry, **extra):
    # Preserve an emitted candidate in the FP denominator without inventing usable coordinates.
    return dict(label=label, invalid_geometry=True, raw_geometry=str(raw_geometry), **extra)


def la(raw, specs, size, kind, default_label=None):
    dim = 4 if kind == 'boxes' else 2
    label, predictions, errors = default_label, [], []
    for m in re.finditer(r'<ref>(.*?)</ref>|<box>(.*?)</box>', raw, re.DOTALL):
        if m.group(1) is not None:
            label = m.group(1)
            continue
        body = m.group(2).strip()
        if body.lower() == 'none':
            continue
        tokens = re.findall(r'<(\d+)>', body)
        coords = geometry([int(v) for v in tokens], size, dim, True)
        name = canonical(label, specs) if label is not None else None
        if name is None:
            errors.append('unknown_or_missing_label')
            # Do not silently turn a wrong class into the queried class.
            name = '__unmapped__'
        if coords is None or re.sub(r'<\d+>', '', body).strip():
            errors.append('invalid_geometry')
            predictions.append(invalid_prediction(name, body))
            continue
        predictions.append(dict(label=name, **{('hbox' if dim == 4 else 'point'): coords}))
    if raw.count('<box>') != raw.count('</box>'):
        errors.append('incomplete_geometry_block')
    if not predictions and not errors and '<box>None</box>' not in raw:
        errors.append('no_geometry_or_null')
    return predictions, errors


def qwen(raw, spec, size, kind):
    decoder = json.JSONDecoder()
    key, dim = ('bbox_2d', 4) if kind == 'boxes' else ('point_2d', 2)
    predictions, errors, cursor = [], [], 0
    while cursor < len(raw):
        start = raw.find('{', cursor)
        if start < 0:
            break
        try:
            item, end = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            errors.append('incomplete_or_invalid_json')
            cursor = start + 1
            continue
        cursor = start + end
        coords = geometry(item.get(key), size, dim, True) if isinstance(item, dict) else None
        # A singleton query supplies an omitted label, not an explicit wrong label.
        labels = [item[k] for k in ('label', 'category') if k in item]
        name = spec['canonical']
        if any(canonical(label, [spec]) != name for label in labels):
            name = '__unmapped__'
            errors.append('unknown_or_wrong_label')
        if coords is None:
            errors.append('invalid_geometry')
            if any(k in item for k in ('bbox_2d', 'point_2d')):
                predictions.append(invalid_prediction(name, item))
        else:
            predictions.append(dict(label=name, **{('hbox' if dim == 4 else 'point'): coords}))
    if not predictions and not errors and '[]' not in raw:
        errors.append('no_geometry_or_empty_array')
    return predictions, errors
