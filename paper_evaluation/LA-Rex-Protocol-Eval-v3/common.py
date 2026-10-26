"""Frozen RS adaptation of the released LA/Rex dense evaluation protocol."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = 'rs_gt_categories_la_native_v3'
DATASETS = {'DIOR': (586, 3379, 20), 'DOTAv2': (874, 29329, 18)}
VLM = ('base_la', 'standard', 'rex_omni', 'qwen3_4b', 'qwen3_8b', 'pid', 'class_isolated')
DETECTORS = ('gdino_tiny', 'laedino')
DEFAULT_MODELS = VLM[:5] + DETECTORS
EXTERNAL = {
    'qwen3_4b': ('Qwen/Qwen3-VL-4B-Instruct', 'ebb281ec70b05090aa6165b016eac8ec08e71b17'),
    'qwen3_8b': ('Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'),
    'rex_omni': ('IDEA-Research/Rex-Omni', '0e5693d24657f6c0e091008dd6809bb1bd28988c'),
    'gdino_tiny': ('IDEA-Research/grounding-dino-tiny', 'a2bb814dd30d776dcf7e30523b00659f4f141c71'),
    'laedino': ('ML4Sustain/LAE-DINO', '3b3332e963371363bb9c30b4384ca8ae1d2b5857'),
    'bert': ('google-bert/bert-base-uncased', '86b5e0934494bd15c9632b12f734a8a67f723594'),
}
CODES = {
    'Rex-Omni': ('https://github.com/IDEA-Research/Rex-Omni.git', '6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e'),
    'LAE-DINO': ('https://github.com/jaychempan/LAE-DINO.git', '6b1519626e39d1f39f8ed1f38761c20f7e0e8c35'),
}
LA_PIN = '783f656d127ee498137b5ff52603ce36c292d317'
SCORER_SHA = '86999ceb31eafd64da5111c0545bee66a2fe9bca2eb2ca68d993d043d1858a36'
LAE_WEIGHT = 'checkpoints/lae_dino_swint_lae1m-28ca3a15.pth'
CHECKPOINTS = {
    'standard': ('standard', 'b0c4c248b16de59f0339c2d4658e36de86cdbcd8c81c4b408bd2f9d199fadb83', 1200),
    'pid': ('refwave', 'a1f8da46db0b3447b59900a848b035a11aeefa5b465f7a5faf6e2e64f52aedf3', 2400),
    'class_isolated': ('class_isolated', 'eb662db4078042c39eb55b8d8760472bb053fc4d372c87ec613426a9e8750185', 1246),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def freeze(path, value):
    if Path(path).exists() and read(path) != value:
        raise ValueError(f'Frozen settings changed: {path}. Use a new output directory.')
    atomic(path, value)


def safe_cache():
    home = Path(os.environ.get('RS_PROTOCOL_HF_HOME', HERE/'.hf-cache')).resolve()
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    for k, suffix in {'HF_HOME': '', 'HF_TOKEN_PATH': 'token', 'HF_HUB_CACHE': 'hub',
                      'HF_MODULES_CACHE': 'modules', 'HF_ASSETS_CACHE': 'assets', 'HF_XET_CACHE': 'xet'}.items():
        os.environ[k] = str(home/suffix)
    for k in ('TRANSFORMERS_CACHE', 'HUGGINGFACE_HUB_CACHE', 'HUGGINGFACE_ASSETS_CACHE',
              'PYTORCH_TRANSFORMERS_CACHE', 'PYTORCH_PRETRAINED_BERT_CACHE'):
        os.environ.pop(k, None)


def semantic_rows(values):
    return [{k: v for k, v in r.items() if k != 'image_path'} for r in values]


def query_only(row):
    # Category presence is the ONLY annotation-derived information allowed here.
    present = {g['label'] for g in row['gt']}
    specs = [s for s in row['class_specs'] if s['canonical'] in present]
    if not specs or {s['canonical'] for s in specs} != present:
        raise ValueError('Unknown category or empty GT; frozen benchmarks must be nonempty.')
    return dict(uid=row['uid'], suite=row['suite'], class_specs=specs)


def norm_label(s):
    return ''.join(c.lower() for c in str(s) if c.isalnum())


def canonical(label, specs):
    matches = {s['canonical'] for s in specs if norm_label(label) in
               {norm_label(s[k]) for k in ('canonical', 'prompt', 'lae_name') if k in s}}
    if len(matches) > 1:
        raise ValueError('Ambiguous category alias')
    return next(iter(matches), None)


def la_box_prompt(specs):
    return 'Locate all the instances that matches the following description: ' + '</c>'.join(s['prompt'] for s in specs) + '.'


def point_prompt(spec):
    return f"Point to: {spec['prompt']}."


def qwen_prompt(spec, kind):
    geometry = ('"bbox_2d": [x1,y1,x2,y2]' if kind == 'boxes' else '"point_2d": [x,y]')
    verb = 'Locate' if kind == 'boxes' else 'Point to'
    return (f"{verb} all instances matching the following description: {spec['prompt']}. "
            f'Return a JSON array of objects with {geometry}, normalized to 0-1000. Return [] if absent.')


def jobs(models, shards):
    return [(m, kind, ds, shard, shards) for m in models
            for kind in (('boxes',) if m in DETECTORS else ('boxes', 'pointing'))
            for ds in DATASETS for shard in range(shards)]


def source_hashes():
    return {str(p.resolve()): sha(p) for p in sorted(HERE.glob('*.py'))}
