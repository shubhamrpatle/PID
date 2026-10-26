#!/usr/bin/env python3
"""Developer-only: pin the existing handoff's small code/config files, not datasets."""
import hashlib
import json
from pathlib import Path
import sys

import remote_eval as runtime

here = Path(__file__).resolve().parent
controls = here.parent / 'Final-Controls-Eval'
workspace = controls / 'workspace'
files = set()
for module in list(sys.modules.values()):
    name = getattr(module, '__file__', None)
    if name:
        path = Path(name).resolve()
        if path.is_relative_to(workspace) and path.suffix == '.py':
            files.add(path)
for sub in ('RefWave/src/refwave', 'Pixel-PIVR/src/pixel_pivr', 'zero_shot_analysis',
            'Zero_shot_ anlysis_LA/model/LocateAnything-3B/checkpoint'):
    files.update((workspace / sub).glob('*.py'))
for name in ('evaluate_molmo_pointing.py', 'run_locany_pointing_balanced100.py'):
    files.add(workspace / 'Zero_shot_ anlysis_LA/scripts' / name)
model = workspace / 'Zero_shot_ anlysis_LA/model/LocateAnything-3B/checkpoint'
files.update(model.glob('*.json'))
files.update(model.glob('*.model'))
files.update(model.glob('*.txt'))
data = {str(p.relative_to(controls)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(here / 'runtime_hashes.json').write_text(json.dumps(data, indent=2) + '\n')
print(f'Pinned {len(data)} existing code/tokenizer/config files; no images or checkpoint weights read.')
