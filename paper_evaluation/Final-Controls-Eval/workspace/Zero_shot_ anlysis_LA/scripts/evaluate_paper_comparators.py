#!/usr/bin/env python3
"""Released-model inference on the paper's frozen manifests; no training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path[:0] = [str(ROOT), str(PROJECT / 'scripts')]
ASSETS = PROJECT / 'runs/refwave_cluster_stage2_best_20260909/assets'
OUTPUT = PROJECT / 'runs/paper_comparators_20260913'
CACHE = Path('/data2/shubham/models/paper_comparators_hf')
LAE = ROOT / 'external/LAE-DINO'
MMDET = LAE / 'mmdetection_lae'
MODELS = {
    'qwen3_4b': dict(backend='qwen', repo='Qwen/Qwen3-VL-4B-Instruct', revision='ebb281ec70b05090aa6165b016eac8ec08e71b17'),
    'qwen3_8b': dict(backend='qwen', repo='Qwen/Qwen3-VL-8B-Instruct', revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'),
    'gdino_base': dict(backend='gdino', repo='IDEA-Research/grounding-dino-base', revision='12bdfa3120f3e7ec7b434d90674b3396eccf88eb'),
    'laedino_pretrain': dict(backend='laedino', checkpoint='lae_dino_swint_lae1m-28ca3a15.pth'),
    'laedino_ft': dict(backend='laedino', checkpoint='benchmark-specific released FT; NOT the +LAE-1M row'),
    'geoground': dict(backend='geoground', checkpoint='/data2/shubham/models/GeoGround/llava-v1.5-7b-geoground-merged'),
}
SUITES = ('DIOR', 'DOTAv2', 'LAE-80C', 'DIOR-RSVG', 'VRSBench-VG')


def read(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def query_only(row):
    # Pass only the requested vocabulary/expression, never GT or target_class.
    if row['task'] == 'detection':
        return dict(task='detection', class_specs=row['class_specs'])
    return dict(task='grounding', query=row['query'])


def qwen_prompt(query):
    if query['task'] == 'detection':
        names = [c['prompt'] for c in query['class_specs']]
        instruction = 'Locate all instances of these categories: ' + ', '.join(names) + '.'
        form = 'Return a JSON array of objects with "label" and "bbox_2d". Use the given category names as labels.'
    else:
        instruction = 'Locate a single instance that matches the following description: ' + query['query'] + '.'
        form = 'Return a JSON array containing one object with "bbox_2d".'
    return instruction + ' ' + form + ' Each bbox_2d is [x1,y1,x2,y2], normalized to 0-1000. Return [] if absent.'


def parse_qwen(text, query, width, height):
    import evaluate_locateanything_lae_style as det
    decoder = json.JSONDecoder()
    # Recover complete JSON objects even when the generation budget cuts an array.
    objects, cursor = [], 0
    while cursor < len(text):
        start = text.find('{', cursor)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(obj, dict) and 'bbox_2d' in obj:
            objects.append(obj)
        cursor = start + end
    preds, rejected = [], []
    specs = query.get('class_specs', [])
    aliases = det.build_alias_lookup(specs) if specs else {}
    classes = {c['canonical'] for c in specs}
    for obj in objects:
        b = obj.get('bbox_2d')
        if (not isinstance(b, list) or len(b) != 4 or
                not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1000 for v in b)
                or b[2] <= b[0] or b[3] <= b[1]):
            rejected.append(obj)
            continue
        label = (det.canonicalize_label(str(obj.get('label', '')), aliases, classes)
                 if specs else query['query'])
        if label is None:
            rejected.append(obj)
            continue
        preds.append(dict(label=label, hbox=[b[0]*width/1000, b[1]*height/1000, b[2]*width/1000, b[3]*height/1000], score=1.0))
    return preds, rejected


def checkpoint(model, suite):
    if model != 'laedino_ft':
        return LAE / 'weights/hf/checkpoints' / MODELS[model]['checkpoint']
    names = {'DIOR': 'lae_dino_swint_fintune_dior-e612b298.pth',
             'DOTAv2': 'lae_dino_swint_fintune_dota-9e1e8782.pth'}
    if suite not in names:
        raise ValueError('No verified benchmark-specific LAE-DINO FT checkpoint for ' + suite)
    return LAE / 'weights/hf/checkpoints' / names[suite]


def lae_config(model, suite):
    suffix = {'DIOR':'finetune_DIOR', 'DOTAv2':'finetune_DOTA'}[suite] if model == 'laedino_ft' else 'pretrain_LAE-1M'
    return MMDET / 'configs/lae_dino' / ('lae_dino_swin-t_' + suffix + '.py')


def split_vocabulary(specs, tokenizer, limit=230):
    chunks, current = [], []
    for spec in specs:
        proposed = current + [spec]
        prompt = ' . '.join(s['prompt'] for s in proposed) + ' .'
        if len(tokenizer(prompt)['input_ids']) > limit:
            if not current:
                raise ValueError('Single category exceeds native detector text budget')
            chunks.append(current)
            current = [spec]
        else:
            current = proposed
    if current:
        chunks.append(current)
    return chunks


def load_backend(args):
    import torch
    spec = MODELS[args.model]
    if spec['backend'] in ('qwen', 'gdino'):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, AutoModelForZeroShotObjectDetection
        from huggingface_hub import snapshot_download
        if spec['backend'] == 'gdino':
            cached = Path('/data2/shubham/models/qualitative_hf_cache/models--IDEA-Research--grounding-dino-base/snapshots') / spec['revision']
            path = cached if (cached / 'model.safetensors').is_file() else spec['repo']
        else:
            path = snapshot_download(spec['repo'], revision=spec['revision'], cache_dir=CACHE,
                allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'], max_workers=2)
        processor = AutoProcessor.from_pretrained(path, revision=spec['revision'], cache_dir=CACHE)
        if spec['backend'] == 'qwen':
            model = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16,
                attn_implementation='sdpa', device_map='cuda:0').eval()
        else:
            model = AutoModelForZeroShotObjectDetection.from_pretrained(path, revision=spec['revision'], cache_dir=CACHE).to('cuda').eval()
        return dict(model=model, processor=processor)
    if spec['backend'] == 'laedino':
        from mmengine.config import Config
        from mmcv.transforms import Compose
        from mmdet.apis import init_detector
        config = lae_config(args.model, args.suite)
        cfg = Config.fromfile(str(config))
        cfg.model.language_model.name = str(LAE / 'weights/bert-base-uncased')
        model = init_detector(cfg, str(checkpoint(args.model, args.suite)), device='cuda:0', palette='random').eval()
        pipeline = Compose([
            dict(type='mmdet.LoadImageFromNDArray'),
            dict(type='mmdet.FixScaleResize', scale=(800, 1333), keep_ratio=True),
            dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'ori_shape', 'img_shape', 'scale_factor', 'text', 'custom_entities'))])
        return dict(model=model, pipeline=pipeline, tokenizer=model.language_model.tokenizer,
                    native_max_per_image=int(cfg.model.test_cfg.max_per_img), config_sha256=sha(config))
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    path = spec['checkpoint']
    tokenizer, model, processor, context = load_pretrained_model(path, None, get_model_name_from_path(path))
    return dict(model=model.eval(), tokenizer=tokenizer, processor=processor, context=context)


def predict(state, query, image, args):
    import numpy as np
    import torch
    import evaluate_locateanything_lae_style as det
    model = state['model']
    backend = MODELS[args.model]['backend']
    if backend == 'qwen':
        p = state['processor']
        prompt = qwen_prompt(query)
        messages = [dict(role='user', content=[dict(type='image', image=image), dict(type='text', text=prompt)])]
        text = p.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = p(images=image, text=[text], return_tensors='pt',
                   max_pixels=6000*14*14, min_pixels=4096).to('cuda')
        ids = model.generate(**inputs, max_new_tokens=args.budget, do_sample=True,
            temperature=.7, top_p=.9, top_k=0, repetition_penalty=1.1, use_cache=True)
        output_ids = ids[0, inputs.input_ids.shape[1]:].tolist()
        raw = p.decode(output_ids, skip_special_tokens=True)
        boxes, rejected = parse_qwen(raw, query, image.width, image.height)
        return dict(predictions=boxes, raw_response=raw, rejected_json_objects=rejected,
            prompt=prompt, generated_tokens=len(output_ids), budget_saturated=len(output_ids) >= args.budget,
            image_grid_thw=inputs['image_grid_thw'].cpu().tolist())
    if backend in ('laedino', 'gdino'):
        specs = query['class_specs']
        tok = state['tokenizer'] if backend == 'laedino' else state['processor'].tokenizer
        groups = split_vocabulary(specs, tok)
        native, prompts, rejected = [], [], []
        for group in groups:
            text = ' . '.join(s['prompt'] for s in group) + ' .'
            prompts.append(text)
            if backend == 'laedino':
                from mmdet.apis import inference_detector
                pred = inference_detector(model, np.array(image)[:, :, ::-1].copy(),
                    test_pipeline=state['pipeline'], text_prompt=text, custom_entities=True).pred_instances
                boxes, scores, labels = pred.bboxes.cpu().tolist(), pred.scores.cpu().tolist(), pred.labels.cpu().tolist()
                for box, score, label in zip(boxes, scores, labels):
                    if not 0 <= int(label) < len(group):
                        raise ValueError('Detector category index out of range')
                    native.append(dict(hbox=box, label=group[int(label)]['canonical'], score=score))
            else:
                p = state['processor']
                inputs = p(images=image, text=text, return_tensors='pt').to('cuda')
                out = model(**inputs)
                result = p.post_process_grounded_object_detection(out, inputs.input_ids,
                    threshold=0.0, text_threshold=.25, target_sizes=[(image.height, image.width)])[0]
                aliases = det.build_alias_lookup(group)
                classes = {s['canonical'] for s in group}
                for b, score, label in zip(result['boxes'].cpu().tolist(), result['scores'].cpu().tolist(), result['text_labels']):
                    name = det.canonicalize_label(label, aliases, classes)
                    if name is None:
                        if score >= args.threshold:
                            rejected.append(dict(hbox=b, score=score, raw_label=label))
                        continue
                    native.append(dict(hbox=b, score=score, label=name))
        native = [dict(p, hbox=det.clip_hbox_to_image(p['hbox'], image.width, image.height)) for p in native]
        native = [p for p in native if p['hbox'] is not None]
        return dict(predictions=[p for p in native if p['score'] >= args.threshold],
            native_predictions=native, prompts=prompts, text_forwards=len(groups), rejected_labels=rejected)
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.mm_utils import tokenizer_image_token
    from evaluate_geoground_grounding_manifest import parse_geoground_hbox
    prompt = '[refer] output the bounding box of the <ref>' + query['query'] + '</ref> in the image.'
    image_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                   if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN)
    conv = conv_templates['llava_v1'].copy()
    conv.append_message(conv.roles[0], image_token + '\n' + prompt)
    conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), state['tokenizer'], IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    pixels = state['processor'].preprocess([image.resize((336, 336))], crop_size={'height':336,'width':336},
        size={'shortest_edge':336}, return_tensors='pt')['pixel_values'].half().cuda()
    out = model.generate(ids, images=pixels, do_sample=False, num_beams=1, max_new_tokens=256, length_penalty=2.0, use_cache=True)
    raw = state['tokenizer'].batch_decode(out, skip_special_tokens=True)[0].strip()
    box = parse_geoground_hbox(raw, image.width, image.height)
    return dict(predictions=[dict(hbox=box, label=query['query'], score=1.0)] if box else [], raw_response=raw, prompt=prompt)


def infer(args):
    import torch
    import transformers
    from PIL import Image
    from evaluate_locateanything_grounding_benchmarks import set_generation_seed
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    manifest = ASSETS / (args.suite + '.jsonl')
    rows = read(manifest)[args.shard_index::args.shard_count]
    if args.limit:
        rows = rows[:args.limit]
    backend = MODELS[args.model]['backend']
    if backend in ('laedino', 'gdino') and rows[0]['task'] != 'detection':
        raise ValueError('Detector-to-phrase-grounding requires a separately specified protocol')
    if backend == 'geoground' and rows[0]['task'] != 'grounding':
        raise ValueError('GeoGround backend implements only its released HBB grounding task')
    folder = args.output / args.model / args.suite
    folder.mkdir(parents=True, exist_ok=True)
    tag = f'shard{args.shard_index:02d}of{args.shard_count:02d}'
    path = folder / (tag + '.jsonl')
    contract = dict(model=MODELS[args.model], suite=args.suite, manifest_sha256=sha(manifest),
        source_sha256=sha(__file__), shard_index=args.shard_index, shard_count=args.shard_count,
        metric_sources={name:sha(PROJECT / 'scripts' / name) for name in
            ('evaluate_locateanything_lae_style.py', 'evaluate_locateanything_grounding_benchmarks.py')},
        geometry_sha256=sha(ROOT / 'zero_shot_analysis/geometry.py'),
        limit=args.limit, budget=args.budget, detector_threshold=args.threshold,
        prompt_protocol='full requested vocabulary or full expression; native model answer grammar; no GT input',
        generation='Qwen: sample .7/top_p .9/top_k0/repetition1.1/stable seed0; GeoGround: released greedy 256',
        image_protocol='Original image; Qwen max area 6000*14^2, native patch/merge; detectors native800/1333; GeoGround native336 square',
        detector_text='whole vocabulary, deterministic <=230-token chunks only if native256 text limit requires',
        detector_native_max=300 if backend == 'laedino' else None,
        scoring='shared class-aware greedy HBB IoU at .5/.75/.95, no added cap; raw and NMS.5 separate; grounding first-box',
        timing='CUDA synchronized preprocessing+inference+parsing; excludes image disk IO, weights, warmup and metric computation',
        warmup_queries=2, cpu_threads=4, gpu=torch.cuda.get_device_name(0),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=torch.__version__, transformers=transformers.__version__)
    if backend == 'laedino':
        contract['checkpoint_sha256'] = sha(checkpoint(args.model, args.suite))
        contract['config_sha256'] = sha(lae_config(args.model, args.suite))
    if backend == 'geoground':
        contract['checkpoint_files'] = {p.name:sha(p) for p in sorted(Path(MODELS[args.model]['checkpoint']).glob('*'))
                                        if p.suffix in ('.bin', '.safetensors', '.json')}
    cp = folder / (tag + '.contract.json')
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Changed run contract; use a new output directory')
    write(cp, contract)
    existing = read(path) if path.exists() else []
    done = {r['uid'] for r in existing}
    expected = {r['uid'] for r in rows}
    if len(done) != len(existing) or not done <= expected:
        raise ValueError('Unexpected/duplicate saved predictions')
    if done == expected:
        write(folder / (tag + '.done.json'), dict(records=len(rows)))
        return
    state = load_backend(args)
    def one(row, image):
        query = query_only(row)
        payload = '|'.join(['0', row['suite'], str(row['uid']), 'generation'])
        set_generation_seed(int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], 'big'))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            result = predict(state, query, image, args)
        torch.cuda.synchronize()
        result.update(latency_seconds=time.perf_counter()-start,
            peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2)
        return result
    for row in rows[:2]:
        with Image.open(row['image_path']) as im:
            one(row, im.convert('RGB'))
    with path.open('a') as f:
        for row in rows:
            if row['uid'] in done:
                continue
            with Image.open(row['image_path']) as im:
                image = im.convert('RGB')
            if image.size != (int(row['width']), int(row['height'])):
                raise ValueError('Manifest/image dimension mismatch')
            result = one(row, image)
            result.update({k:row[k] for k in ('uid','dataset','image_id','suite','task')})
            if row['task'] == 'grounding':
                result.update(sample_id=row['sample_id'], query=row['query'])
            f.write(json.dumps(result, sort_keys=True) + '\n')
            f.flush()
            done.add(row['uid'])
            write(folder / (tag + '.status.json'), dict(records=len(done), expected=len(rows), complete=len(done)==len(rows)))
            print(args.model, args.suite, f'{len(done)}/{len(rows)}', 'boxes=',len(result['predictions']),
                  'seconds=',round(result['latency_seconds'],3), flush=True)
    write(folder / (tag + '.done.json'), dict(records=len(rows)))


def merge(args):
    import numpy as np
    import evaluate_locateanything_lae_style as det
    import evaluate_locateanything_grounding_benchmarks as ground
    rows = read(ASSETS / (args.suite + '.jsonl'))
    if args.limit:
        rows = rows[:args.limit]
        if args.shard_count != 1:
            raise ValueError('Limited smoke merge requires one shard')
    folder = args.output / args.model / args.suite
    preds, contracts = [], []
    for shard in range(args.shard_count):
        tag = f'shard{shard:02d}of{args.shard_count:02d}'
        if not (folder / (tag + '.done.json')).is_file():
            raise ValueError('Incomplete shard: ' + tag)
        preds.extend(read(folder / (tag + '.jsonl')))
        contracts.append(json.loads((folder / (tag + '.contract.json')).read_text()))
    ignore = {'shard_index', 'visible_devices'}
    if not all({k:v for k,v in c.items() if k not in ignore} == {k:v for k,v in contracts[0].items() if k not in ignore} for c in contracts):
        raise ValueError('Mixed inference contracts')
    indexed = {r['uid']:r for r in preds}
    if len(indexed) != len(preds) or set(indexed) != {r['uid'] for r in rows}:
        raise ValueError('Prediction/manifest completeness mismatch')
    if contracts[0]['manifest_sha256'] != sha(ASSETS / (args.suite + '.jsonl')):
        raise ValueError('Manifest changed since inference')
    preds = [indexed[r['uid']] for r in rows]
    times = [r['latency_seconds'] for r in preds]
    m = dict(complete=True, records=len(rows), raw_boxes=sum(len(r['predictions']) for r in preds),
        mean_seconds=float(np.mean(times)), p50_seconds=float(np.median(times)), p95_seconds=float(np.percentile(times,95)),
        peak_gpu_mb=max(r['peak_gpu_mb'] for r in preds), contract=contracts[0],
        budget_saturated_records=sum(r.get('budget_saturated',False) for r in preds),
        rejected_output_items=sum(len(r.get('rejected_json_objects',[])) + len(r.get('rejected_labels',[])) for r in preds),
        prediction_files={p.name:sha(p) for p in sorted(folder.glob('shard*of*.jsonl'))},
        metric_sources={name:sha(PROJECT / 'scripts' / name) for name in
            ('evaluate_locateanything_lae_style.py', 'evaluate_locateanything_grounding_benchmarks.py')})
    if rows[0]['task'] == 'detection':
        classes = [s['canonical'] for s in rows[0]['class_specs']]
        for key, items in [('raw', preds), ('nms0p5', [dict(r, predictions=det.nms_predictions(r['predictions'],.5)) for r in preds])]:
            ix = {(r['dataset'],r['image_id']):r for r in items}
            m[key] = {str(t):det.f1_for_threshold(rows,ix,classes,t,0) for t in (.5,.75,.95)}
        tp = m['raw']['0.5']['tp']
        assert tp + m['raw']['0.5']['fp'] == m['raw_boxes']
    else:
        result = ground.evaluate_predictions(rows,preds)
        assert result['missing_predictions'] == 0
        m['grounding'], m['by_size'] = result['overall'], result['by_size']
        tp = sum(r['first_acc_0_5'] for r in result['rows'])
    m.update(bps=m['raw_boxes']/sum(times), cbps_iou0p5=tp/sum(times),
        definition='BPS all returned boxes per summed timed second; CBPS raw true positives / correct first grounding answers per summed second')
    write(folder / 'metrics.json', m)
    print(json.dumps(m, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['infer','merge'])
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--suite', choices=SUITES, required=True)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--budget', type=int, default=8192)
    p.add_argument('--threshold', type=float, default=.30)
    a = p.parse_args()
    if not 0 <= a.shard_index < a.shard_count or a.limit < 0 or a.budget < 1:
        p.error('Invalid shard, limit, or budget')
    globals()[a.mode](a)


if __name__ == '__main__':
    main()
