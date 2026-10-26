"""Native inference interfaces. Only category names, image and UID enter infer()."""
import gc
import importlib.metadata
import os
from pathlib import Path
import sys
from common import canonical, digest, la_box_prompt, point_prompt, qwen_prompt, safe_cache
import parsing
from native_parser import la_native


def runtime(controls):
    root = Path(controls)/'workspace'
    project = root/'Zero_shot_ anlysis_LA'
    paths = [root/'RefWave/src',root/'Pixel-PIVR/src',root/'Pixel-PIVR',root,project/'scripts']
    sys.path[:0] = list(map(str,paths))
    import evaluate_final_standard_isolated as evaluation
    import pid_packed_protocol as pid
    if Path(evaluation.__file__).resolve() != (project/'scripts/evaluate_final_standard_isolated.py').resolve():
        raise RuntimeError('Wrong controls implementation imported')
    return evaluation,pid


def software():
    import torch
    result=dict(python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0))
    for name in ('transformers','torchvision','mmcv','mmengine','mmdet','numpy','Pillow',
                 'qwen-vl-utils','huggingface-hub','accelerate','tokenizers','safetensors'):
        try: result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name]=None
    return result


def seed_for(query,label,kind):
    return int(digest([0,query['suite'],query['uid'],label,kind])[:8],16)


def seed(value):
    import random
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


class Backend:
    def __init__(self,name,plan):
        import torch
        safe_cache()
        self.name,self.plan,self.budget=name,plan,plan['budget']
        identity=plan['models'][name]
        if name.startswith('qwen3_'):
            from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
            self.processor=AutoProcessor.from_pretrained(identity['path'],local_files_only=True)
            self.model=Qwen3VLForConditionalGeneration.from_pretrained(identity['path'],local_files_only=True,
                torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
        elif name=='rex_omni':
            sys.path.insert(0,identity['code'])
            from rex_omni import RexOmniWrapper
            self.model=RexOmniWrapper(identity['path'],backend='transformers',max_tokens=self.budget,
                temperature=0.,top_p=.05,top_k=1,repetition_penalty=1.05,
                torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa')
        elif name=='gdino_tiny':
            from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
            self.processor=AutoProcessor.from_pretrained(identity['path'],local_files_only=True)
            self.model=AutoModelForZeroShotObjectDetection.from_pretrained(identity['path'],local_files_only=True).to('cuda:0').eval()
        elif name=='laedino':
            code=Path(identity['code'])/'mmdetection_lae'
            sys.path.insert(0,str(code))
            from mmengine.config import Config
            from mmcv.transforms import Compose
            from mmdet.apis import init_detector
            from common import LAE_WEIGHT
            config=Config.fromfile(str(code/'configs/lae_dino/lae_dino_swin-t_pretrain_LAE-1M.py'))
            config.model.language_model.name=plan['models']['bert']['path']
            self.model=init_detector(config,str(Path(identity['path'])/LAE_WEIGHT),device='cuda:0',palette='random').eval()
            self.native_cap=int(self.model.bbox_head.test_cfg['max_per_img'])
            self.pipeline=Compose([dict(type='mmdet.LoadImageFromNDArray'),
                dict(type='mmdet.FixScaleResize',scale=(800,1333),keep_ratio=True),
                dict(type='mmdet.PackDetInputs',meta_keys=('img_id','ori_shape','img_shape','scale_factor','text','custom_entities'))])
        else:
            self.evaluation,self.pid=runtime(plan['controls'])
            from pixel_pivr.worker import LocateAnythingPixelPIVRWorker
            self.model=LocateAnythingPixelPIVRWorker(identity['base'],device='cuda:0')
            if name!='base_la':
                from refwave.model import configure_full
                from refwave.train import restore
                saved=torch.load(identity['path'],map_location='cpu',mmap=True,weights_only=False)
                if saved['step']!=identity['step']:
                    raise ValueError('Loaded checkpoint step differs')
                configure_full(self.model.model)
                restore(self.model.model,saved['trainable_state'])
                del saved
                gc.collect()
            self.model.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
            self.model.processor.image_processor.in_token_limit=6000
            if name in ('pid','class_isolated'):
                self.decoders=self.pid.make_decoders(self.model)

    def infer(self,query,image,kind):
        if set(query)!={'uid','suite','class_specs'}:
            raise ValueError('Inference received forbidden fields (GT/count/addresses/etc.)')
        if self.name in ('gdino_tiny','laedino'):
            if kind!='boxes': raise ValueError('Detectors do not emit native points')
            return self.detector(query,image)
        if self.name=='pid': return self.pid_infer(query,image,kind)
        if self.name=='class_isolated' and kind=='boxes': return self.isolated(query,image)
        specs=query['class_specs']
        # Rex and LA use joint category detection; Qwen uses the paper's per-category strategy.
        groups=[[s] for s in specs] if kind=='pointing' or self.name.startswith('qwen3') else [specs]
        answers=[]
        for group in groups:
            value=seed_for(query,'|'.join(s['canonical'] for s in group),kind)
            seed(value)
            answers.append(self.direct(group,image,kind,value))
        return dict(predictions=[p for a in answers for p in a['predictions']],answers=answers,
            generated_tokens=sum(a['generated_tokens'] for a in answers),
            saturated_queries=sum(a['budget_saturated'] for a in answers),
            invalid_items=sum(len(a['parse_errors']) for a in answers),query_forwards=len(answers))

    def direct(self,specs,image,kind,value):
        errors=[]
        if self.name.startswith('qwen3'):
            prompt=qwen_prompt(specs[0],kind)
            message=[dict(role='user',content=[dict(type='image',image=image),dict(type='text',text=prompt)])]
            text=self.processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
            inputs=self.processor(images=image,text=[text],return_tensors='pt',min_pixels=4096,max_pixels=1176000).to('cuda:0')
            ids=self.model.generate(**inputs,max_new_tokens=self.budget,do_sample=True,temperature=.7,top_p=.9,
                                    top_k=0,repetition_penalty=1.1,use_cache=True)
            new=ids[0,inputs['input_ids'].shape[1]:]
            raw=self.processor.decode(new,skip_special_tokens=True)
            count=len(new)
            predictions,errors=parsing.qwen(raw,specs[0],image.size,kind)
        elif self.name=='rex_omni':
            answer=self.model.inference(images=image,task='detection' if kind=='boxes' else 'pointing',
                                        categories=[s['prompt'] for s in specs])[0]
            if not answer['success']: raise RuntimeError(answer.get('error','Rex inference failed'))
            raw,prompt,count=answer['raw_output'],answer['prompt'],answer['num_output_tokens']
            predictions=[]
            for label,objects in answer['extracted_predictions'].items():
                name=canonical(label,specs)
                for obj in objects:
                    dim=4 if kind=='boxes' else 2
                    coords=parsing.geometry(obj['coords'],image.size,dim)
                    if obj['type']!=('box' if kind=='boxes' else 'point') or coords is None:
                        errors.append('invalid_geometry')
                        predictions.append(parsing.invalid_prediction(name or '__unmapped__', obj))
                        continue
                    if name is None: errors.append('unknown_label')
                    predictions.append(dict(label=name or '__unmapped__',**{('hbox' if dim==4 else 'point'):coords}))
        else:
            prompt=la_box_prompt(specs) if kind=='boxes' else point_prompt(specs[0])
            raw=self.model.predict_points(image,prompt,max_new_tokens=self.budget,temperature=.7,top_p=.9,repetition_penalty=1.1)
            count=len(self.model.tokenizer.encode(raw,add_special_tokens=False))
            address_label=specs[0]['canonical'] if self.name=='class_isolated' and len(specs)==1 else None
            predictions,errors=la_native(raw,specs,image.size,kind,address_label)
        return dict(raw_response=raw,prompt=prompt,queried_categories=[s['canonical'] for s in specs],
                    predictions=predictions,parse_errors=errors,generated_tokens=count,
                    token_count_kind='retokenized_response' if self.name in ('base_la','standard','class_isolated') else 'generated_ids',
                    budget_saturated=count>=self.budget,generation_seed=value)

    def isolated(self,query,image):
        from pixel_pivr.class_wave import ClassPointBranch
        branches=[ClassPointBranch(s['canonical'],la_box_prompt([s]),seed_for(query,s['canonical'],'boxes'))
                  for s in query['class_specs']]
        decoder=self.decoders[0]
        decoder.max_wave_size=len(branches)
        answers,stats=decoder.generate(image,branches,max_new_tokens=self.budget,temperature=.7,top_p=.9,
                                      repetition_penalty=1.1,generation_mode='hybrid')
        parsed=[la_native(a['answer'],query['class_specs'],image.size,'boxes',a['label']) for a in answers]
        return dict(predictions=[p for ps,_ in parsed for p in ps],answers=answers,stats=stats,
            invalid_items=sum(len(e) for _,e in parsed),generated_tokens=sum(a['tokens_generated'] for a in answers),
            saturated_queries=sum(a.get('fallback_reason')=='max_new_tokens' for a in answers))

    def pid_infer(self,query,image,kind):
        from pixel_pivr.class_wave import ClassPointBranch
        specs=query['class_specs']
        refs=[dict(label=s['canonical'],reference=s['prompt'],
                   question=f"Point to all instances that match the following categories: {s['prompt']}.") for s in specs]
        if kind=='pointing':
            # No hard-coded 18-class entry point and no box-to-center conversion.
            branches=[ClassPointBranch(s['canonical'],point_prompt(s),seed_for(query,s['canonical'],kind)) for s in specs]
            decoder=self.decoders[0]
            decoder.max_wave_size=len(branches)
            answers,stats=decoder.generate(image,branches,max_new_tokens=self.budget,temperature=.7,top_p=.9,
                                          repetition_penalty=1.1,generation_mode='hybrid')
            self.pid.validate_discovery(stats,len(branches))
            parsed=[la_native(a['answer'],specs,image.size,kind,a['label']) for a in answers]
            return dict(predictions=[p for ps,_ in parsed for p in ps],answers=answers,stats=stats,
                generated_tokens=sum(a['tokens_generated'] for a in answers),invalid_items=sum(len(e) for _,e in parsed),
                saturated_queries=sum(a.get('fallback_reason')=='max_new_tokens' for a in answers))
        row=dict(query,dataset=query['suite'],image_id=query['uid'],task='detection',
                 direct_question=la_box_prompt(specs),references=refs)
        result=self.pid.infer_one(self.model,self.decoders,row,image,self.budget)
        parsed=[la_native(a['answer'],specs,image.size,kind,a['label']) for a in result['refinements']]
        result['predictions']=[p for ps,_ in parsed for p in ps]
        result['invalid_items']=sum(len(e) for _,e in parsed)
        result['generated_tokens']=result['generated_text_tokens']+result['geometry_tokens']
        result['saturated_queries']=result['budget_saturated_branches']
        return result

    def detector(self,query,image):
        specs=query['class_specs']
        text=' . '.join(s['prompt'] for s in specs)+' .'
        predictions,errors=[],[]
        if self.name=='gdino_tiny':
            inputs=self.processor(images=image,text=text,return_tensors='pt').to('cuda:0')
            if inputs['input_ids'].shape[1]>self.model.config.max_text_len:
                raise ValueError('Text exceeds native detector context; no silent truncation')
            outputs=self.model(**inputs)
            out=self.processor.post_process_grounded_object_detection(outputs,inputs.input_ids,threshold=0.,
                text_threshold=.25,target_sizes=[(image.height,image.width)])[0]
            triples=zip(out['boxes'].cpu().tolist(),out['scores'].cpu().tolist(),out['text_labels'])
            count=int(outputs.pred_boxes.shape[1])
            for box,score,label in triples:
                name=parsing.gdino_label(label,specs)
                predictions.append(dict(hbox=box,score=score,label=name or '__unmapped__',raw_label=label))
                if name is None: errors.append('unmapped_native_phrase')
            cap=None
        else:
            import numpy as np
            from mmdet.apis import inference_detector
            tokenizer=self.model.language_model.tokenizer
            if len(tokenizer(text,add_special_tokens=True)['input_ids'])>230:
                raise ValueError('Text too long; no silent vocabulary truncation')
            config=self.model.bbox_head.test_cfg
            count=int(self.model.num_queries)*len(specs)
            old=config['max_per_img']
            try:
                config['max_per_img']=count
                out=inference_detector(self.model,np.asarray(image)[:,:,::-1].copy(),test_pipeline=self.pipeline,
                                       text_prompt=text,custom_entities=True).pred_instances
            finally: config['max_per_img']=old
            if len(out)!=count: raise ValueError('LAE candidate pool incomplete')
            for rank,(b,s,c) in enumerate(zip(out.bboxes.cpu().tolist(),out.scores.cpu().tolist(),out.labels.cpu().tolist())):
                if not 0<=c<len(specs): raise ValueError('LAE invalid category index')
                predictions.append(dict(hbox=b,score=s,label=specs[c]['canonical'],native_rank=rank))
            cap=self.native_cap
        # Detector boxes can exceed the image by construction; clip all detector outputs consistently.
        valid=[]
        for p in predictions:
            b=[min(max(v,0.),image.size[i%2]) for i,v in enumerate(p['hbox'])]
            if parsing.geometry(b,image.size,4) is None:
                errors.append('invalid_geometry')
                valid.append(parsing.invalid_prediction(p['label'],p['hbox'],
                    **{k:v for k,v in p.items() if k not in ('label','hbox')}))
            else: valid.append(dict(p,hbox=b))
        return dict(predictions=valid,prompts=[text],native_candidate_count=count,native_cap=cap,
                    invalid_items=len(errors),parse_errors=errors,generated_tokens=None,saturated_queries=0)
