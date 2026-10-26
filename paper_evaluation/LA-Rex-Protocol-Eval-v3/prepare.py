#!/usr/bin/env python3
"""One-time data/model preparation, separate from inference and training."""
import argparse
import hashlib
import math
from pathlib import Path
import subprocess
import urllib.request
from common import (HERE, DATASETS, EXTERNAL, CODES, LA_PIN, SCORER_SHA, LAE_WEIGHT,
                    DEFAULT_MODELS, CHECKPOINTS, sha, digest, read, rows, atomic,
                    freeze, safe_cache, semantic_rows, query_only)

SEMANTIC = {'DIOR': '1775f2e28c4f3ec51b6dd37444fcdb75c227a78fe912c0c0025b2c41308da3ed',
            'DOTAv2': 'abb5851627abae2dc918399d95bce33ed8cc3c6912c78c2217d9b2b048341079'}


def evidence():
    target = HERE/'evidence/other_metric.py'
    target.parent.mkdir(exist_ok=True)
    url = f'https://raw.githubusercontent.com/NVlabs/Eagle/{LA_PIN}/Embodied/evaluation/metrics/other_metric.py'
    if not target.exists():
        content = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(content).hexdigest() != SCORER_SHA:
            raise ValueError('Upstream download hash mismatch')
        target.write_bytes(content)
    if sha(target) != SCORER_SHA:
        raise ValueError('Upstream source changed')
    freeze(HERE/'evidence/source.json', dict(url=url, revision=LA_PIN, sha256=SCORER_SHA))
    from native_parser import SOURCE, SOURCE_SHA
    native_url=f'https://raw.githubusercontent.com/NVlabs/Eagle/{LA_PIN}/Embodied/evaluation/inference_grounding_ddp.py'
    if not SOURCE.exists():
        content=urllib.request.urlopen(native_url,timeout=60).read()
        if hashlib.sha256(content).hexdigest()!=SOURCE_SHA:
            raise ValueError('Native LA parser download hash mismatch')
        SOURCE.write_bytes(content)
    if sha(SOURCE)!=SOURCE_SHA:
        raise ValueError('Native LA parser changed')
    freeze(HERE/'evidence/native_source.json',dict(url=native_url,revision=LA_PIN,sha256=SOURCE_SHA))


def data(args):
    from PIL import Image
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.manifests or args.controls/'workspace/Zero_shot_ anlysis_LA/runs/paper_rex_pointing_full_20260915_v1/assets'
    report = {}
    image_dirs = read(args.image_dirs) if args.image_dirs else {}
    inventory = {e['path']:e for e in read(args.controls/'transfer_inventory.json')['files']}
    for ds, (count, targets, classes) in DATASETS.items():
        original = rows(source/(ds+'.jsonl'))
        if digest(semantic_rows(original)) != SEMANTIC[ds]:
            raise ValueError('Frozen record contents/order differ: '+ds)
        if len(original) != count or len({r['uid'] for r in original}) != count:
            raise ValueError('Wrong manifest length/duplicate UID')
        images, out = {}, []
        for r in original:
            image = Path(r['image_path'])
            if args.image_root:
                # Preserve the full suffix, never search by a potentially ambiguous basename.
                parts = image.parts
                marker = parts.index('Zero_shot_ anlysis_LA')
                image = args.image_root.joinpath(*parts[marker:])
            if ds in image_dirs:
                image = Path(image_dirs[ds])/r['file_name']
            if not image.is_file():
                raise FileNotFoundError(image)
            with Image.open(image) as im:
                if im.size != (r['width'], r['height']):
                    raise ValueError('Image dimensions differ: '+str(image))
            if len(r['class_specs']) != classes:
                raise ValueError('Vocabulary size changed')
            query_only(r)
            for g in r['gt']:
                b = g['hbox']
                if len(b) != 4 or not all(math.isfinite(v) for v in b) or b[0] >= b[2] or b[1] >= b[3]:
                    raise ValueError('Invalid source geometry')
            image_sha = sha(image)
            key = f"workspace/Zero_shot_ anlysis_LA/dataset/benchmark/{ds}/images/{r['file_name']}"
            if key not in inventory or inventory[key]['sha256'] != image_sha:
                raise ValueError('Image content differs from original controls transfer: '+str(image))
            images[str(image.resolve())] = dict(sha256=image_sha, bytes=image.stat().st_size,
                                                 mtime_ns=image.stat().st_mtime_ns)
            out.append(dict(r, image_path=str(image.resolve())))
        if sum(len(r['gt']) for r in out) != targets:
            raise ValueError('GT count changed')
        destination = args.output/'assets'/(ds+'.jsonl')
        destination.parent.mkdir(exist_ok=True)
        if destination.exists() and rows(destination) != out:
            raise ValueError('Prepared data changed; choose a new output folder')
        import json
        destination.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in out))
        report[ds] = dict(records=count, targets=targets, vocabulary_classes=classes,
            gt_present_queries=sum(len(query_only(r)['class_specs']) for r in out),
            original_sha256=sha(source/(ds+'.jsonl')), semantic_sha256=SEMANTIC[ds],
            path=str(destination.resolve()), sha256=sha(destination), images=images,
            source_outside_image_boxes=sum(r.get('outside_image_gt_boxes', 0) for r in out))
        print(ds, count, 'images;', targets, 'targets;', report[ds]['gt_present_queries'], 'positive category queries', flush=True)
    freeze(args.output/'data.json', report)
    evidence()


def git_code(name, args, override=None):
    url, revision = CODES[name]
    path = Path(override).resolve() if override else args.models_dir/name
    if not path.exists():
        subprocess.run(['git', 'clone', '--filter=blob:none', url, str(path)], check=True)
        subprocess.run(['git', '-C', str(path), 'checkout', '--detach', revision], check=True)
    if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'], text=True).strip() != revision:
        raise ValueError('Wrong code revision: '+str(path))
    if subprocess.check_output(['git','-C',str(path),'diff','HEAD','--name-only'], text=True):
        raise ValueError('Modified upstream code: '+str(path))
    return path


def verify_hf(name, args, override):
    from huggingface_hub import HfApi, snapshot_download
    repo, revision = EXTERNAL[name]
    info = HfApi(token=False).model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError('Unexpected HF revision')
    patterns = [LAE_WEIGHT] if name == 'laedino' else ['*.json','*.safetensors','*.txt','*.model','*.jinja','*.py']
    path = Path(override).resolve() if override else args.models_dir/name
    if not override:
        snapshot_download(repo, revision=revision, local_dir=path, allow_patterns=patterns, token=False, max_workers=2)
    import fnmatch
    files = {}
    has_weights = False
    for f in info.siblings:
        if not any(fnmatch.fnmatch(f.rfilename, pat) for pat in patterns):
            continue
        p = path/f.rfilename
        if not p.is_file() or p.stat().st_size != f.size:
            raise ValueError('Missing/wrong model file: '+str(p))
        hashed = sha(p)
        valid = hashed == f.lfs.sha256 if f.lfs else hashlib.sha1(b'blob '+str(f.size).encode()+b'\0'+p.read_bytes()).hexdigest() == f.blob_id
        if not valid:
            raise ValueError('HF file hash mismatch: '+str(p))
        files[str(p.resolve())] = dict(sha256=hashed, bytes=f.size, mtime_ns=p.stat().st_mtime_ns)
        has_weights |= f.rfilename.endswith(('.safetensors','.pth'))
    if not files or not has_weights:
        raise ValueError('No model weights')
    return dict(repo=repo, revision=revision, path=str(path.resolve()), files=files)


def models(args):
    safe_cache()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    overrides = read(args.model_paths) if args.model_paths else {}
    existing = read(args.output/'models.json') if (args.output/'models.json').exists() else {}
    entries = dict(existing)
    names = list(args.models) + (['bert'] if 'laedino' in args.models else [])
    for name in names:
        if name in EXTERNAL:
            reuse=overrides.get(name)
            if not reuse and args.reuse_models_dir and (args.reuse_models_dir/name).is_dir():
                reuse=args.reuse_models_dir/name
            entries[name] = verify_hf(name, args, reuse)
        else:
            base = args.controls/'workspace/Zero_shot_ anlysis_LA/model/LocateAnything-3B/checkpoint'
            inventory = read(args.controls/'transfer_inventory.json')['files']
            prefix = str(base.relative_to(args.controls))+'/'
            files = {}
            for entry in inventory:
                if entry['path'].startswith(prefix):
                    p = args.controls/entry['path']
                    if sha(p) != entry['sha256']:
                        raise ValueError('Base LA identity mismatch')
                    files[str(p.resolve())] = dict(sha256=entry['sha256'], bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            if not files:
                raise ValueError('No LA model in controls inventory')
            entries[name] = dict(base=str(base.resolve()), files=files)
            if name in CHECKPOINTS:
                method, expected_sha, step = CHECKPOINTS[name]
                defaults = {'standard': args.controls/'results', 'pid': HERE.parent/'Final-RefWave-Eval/results',
                            'class_isolated': HERE.parent/'Final-Class-Isolated-Eval/results'}
                root = Path(overrides.get(name+'_results', defaults[name]))
                selected = read(root/'selection.json')[method]
                ready = read(root/'checkpoints_ready.json')['models'][method]
                p = Path(overrides.get(name, ready['path'])).resolve()
                if selected['sha256'] != expected_sha or selected['step'] != step or sha(p) != expected_sha:
                    raise ValueError('Wrong final Stage-2 best checkpoint: '+name)
                entries[name].update(path=str(p), sha256=expected_sha, step=step, selected=selected)
                entries[name]['files'][str(p)] = dict(sha256=expected_sha, bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
        if name in ('rex_omni','laedino'):
            code_name='Rex-Omni' if name=='rex_omni' else 'LAE-DINO'
            reuse=overrides.get(code_name+'_code')
            if not reuse and args.reuse_models_dir:
                candidate=args.reuse_models_dir/(code_name+'-code' if name=='rex_omni' else code_name)
                if candidate.is_dir(): reuse=candidate
            p=git_code(code_name,args,reuse)
            entries[name]['code']=str(p.resolve())
            entries[name]['code_revision']=CODES[code_name][1]
        for old in existing:
            if entries[old]!=existing[old]:
                raise ValueError('Prepared model identity changed: '+old+'; use a new output directory')
        args.output.mkdir(parents=True,exist_ok=True)
        atomic(args.output/'models.json',entries)
        print('VERIFIED MODEL',name,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['data','models','evidence'])
    p.add_argument('--output',type=Path,default=HERE/'results')
    p.add_argument('--controls',type=Path,default=HERE.parent/'Final-Controls-Eval')
    p.add_argument('--manifests',type=Path)
    p.add_argument('--image-root',type=Path)
    p.add_argument('--image-dirs',type=Path,help='Optional JSON dataset-to-directory map; image hashes must still match controls')
    p.add_argument('--models-dir',type=Path,default=HERE/'models')
    p.add_argument('--model-paths',type=Path,help='JSON overrides; verified against pinned HF/checkpoint hashes')
    p.add_argument('--reuse-models-dir',type=Path,help='Reuse already downloaded MAGRET model folders after hash verification')
    p.add_argument('--models',nargs='+',default=list(DEFAULT_MODELS),choices=list(DEFAULT_MODELS)+['pid','class_isolated'])
    args=p.parse_args()
    args.output,args.controls,args.models_dir = args.output.resolve(),args.controls.resolve(),args.models_dir.resolve()
    if args.mode=='evidence': evidence()
    else: globals()[args.mode](args)


if __name__=='__main__': main()
