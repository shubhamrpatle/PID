"""CPU tests; run explicitly, never injected into a training launcher."""
import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import common as c
import evaluate as e
import parsing
import remote
import scoring
from gpu_pool import GpuPool,allocated
from models import seed_for


SPECS=[dict(canonical='ship',prompt='ship'),dict(canonical='storagetank',prompt='storage tank')]


def record(uid='a'):
    return dict(uid=uid,suite='DIOR',class_specs=SPECS,
                gt=[dict(label='ship',hbox=[0,0,10,10])],width=100,height=100)


class ProtocolTests(unittest.TestCase):
    def test_query_boundary(self):
        r=record(); r['gt_count']=100; r['oracle_point']=[1,2]
        q=c.query_only(r)
        self.assertEqual(set(q),{'uid','suite','class_specs'})
        self.assertEqual(q['class_specs'],SPECS[:1])
        r['gt'].append(dict(label='ship',hbox=[20,20,30,30]))
        self.assertEqual(q,c.query_only(r))

    def test_unknown_gt_and_empty_rejected(self):
        for gt in ([],[dict(label='invented',hbox=[0,0,1,1])]):
            r=record(); r['gt']=gt
            with self.assertRaises(ValueError): c.query_only(r)

    def test_la_template(self):
        self.assertEqual(c.la_box_prompt(SPECS),'Locate all the instances that matches the following description: ship</c>storage tank.')
        self.assertEqual(c.point_prompt(SPECS[0]),'Point to: ship.')

    def test_la_parsing(self):
        raw='<ref>storage tank</ref><box><0><0><500><1000></box><box>None</box>'
        p,errors=parsing.la(raw,SPECS,(200,100),'boxes')
        self.assertEqual(p,[dict(label='storagetank',hbox=[0,0,100,100])]); self.assertFalse(errors)
        self.assertEqual(parsing.la('<box><200><300></box>',SPECS,(200,100),'pointing','ship')[0][0]['point'],[40,30])

    def test_no_geometry_conversion(self):
        p,errors=parsing.la('<box><1><2><3><4></box>',SPECS,(100,100),'pointing','ship')
        self.assertEqual(len(p),1); self.assertTrue(p[0]['invalid_geometry']); self.assertTrue(errors)

    def test_wrong_label_not_relabelled(self):
        p,errors=parsing.la('<ref>car</ref><box><0><0><10><10></box>',SPECS,(100,100),'boxes','ship')
        self.assertEqual(p[0]['label'],'__unmapped__'); self.assertTrue(errors)

    def test_invalid_coordinates(self):
        for b in ([5,5,1,1],[0,0,1001,5],[0,0,float('nan'),5],[0,0,True,5]):
            self.assertIsNone(parsing.geometry(b,(100,100),4,True))

    def test_complete_invalid_boxes_are_not_dropped(self):
        r=record()
        raw='<ref>ship</ref><box><0><0><100><100></box>'+('<box><20><60><80><40></box>'*5)
        pred,errors=parsing.la(raw,SPECS,(100,100),'boxes')
        self.assertEqual(len(pred),6); self.assertEqual(len(errors),5)
        result,_=scoring.score([r],[dict(uid='a',predictions=pred)],'boxes')
        m=result['scores']['class_aware']['0.5']['micro']
        self.assertEqual((m['tp'],m['fp'],m['fn']),(1,5,0))
        self.assertEqual(result['invalid_geometry_outputs'],5)
        upstream,_=scoring.load_upstream()
        recall,precision,_,_=upstream['calculate_detection_metrics']({'ship':[[0,0,10,10]]},
            {'ship':[[0,0,10,10]]+[[2,6,8,4]]*5},.5)
        self.assertEqual((m['precision'],m['recall']),(precision,recall))

    def test_qwen_invalid_geometry_counts_fp_not_repaired(self):
        for kind,key,coords in (('boxes','bbox_2d',[20,60,80,40]),('boxes','bbox_2d',[0,0,0,0]),
                                ('pointing','point_2d',[1001,5])):
            pred,errors=parsing.qwen(json.dumps([{key:coords}]),SPECS[0],(100,100),kind)
            self.assertTrue(pred[0]['invalid_geometry']); self.assertTrue(errors)
            result,_=scoring.score([record()],[dict(uid='a',predictions=pred)],kind)
            metric=result['scores']['class_aware' if kind=='boxes' else 'released']['0.5' if kind=='boxes' else 'point']
            self.assertEqual((metric['micro']['tp'],metric['micro']['fp'],metric['micro']['fn']),(0,1,1))

    def test_invalid_detector_rank_and_threshold(self):
        pred=[dict(label='ship',hbox=[0,0,10,10],score=.9,native_rank=0),
              parsing.invalid_prediction('ship',[1,1,1,1],score=.8,native_rank=1),
              parsing.invalid_prediction('ship',[1,1,1,1],score=.2,native_rank=2),
              dict(label='ship',hbox=[0,0,10,10],score=.7,native_rank=300)]
        result,_=scoring.score([record()],[dict(uid='a',predictions=pred)],'boxes',.25,300)
        self.assertEqual(result['invalid_geometry_outputs'],1)
        self.assertEqual(result['scores']['class_aware']['0.5']['micro']['fp'],1)

    def test_detector_native_phrase_mapping(self):
        specs=[dict(canonical='basketballcourt',prompt='basketball court'),dict(canonical='tenniscourt',prompt='tennis court')]
        self.assertEqual(parsing.gdino_label('basketball court tennis court',specs),'basketballcourt')
        self.assertIsNone(parsing.gdino_label('court',specs))
        self.assertEqual(parsing.gdino_label('tennis court',specs),'tenniscourt')

    def test_qwen_partial_and_duplicates(self):
        raw='[{"bbox_2d":[0,0,100,100]},{"bbox_2d":[0,0,100,100]},{"bbox_2d":'
        p,errors=parsing.qwen(raw,SPECS[0],(100,100),'boxes')
        self.assertEqual(len(p),2); self.assertTrue(errors)
        self.assertEqual(parsing.qwen('[]',SPECS[0],(100,100),'pointing'),([],[]))

    def test_qwen_explicit_wrong_labels_are_false_positives(self):
        for kind,key,coords in (('boxes','bbox_2d',[0,0,100,100]),('pointing','point_2d',[50,50])):
            for field in ('label','category'):
                raw=json.dumps([{key:coords,field:'storage tank'}])
                pred,errors=parsing.qwen(raw,SPECS[0],(100,100),kind)
                self.assertEqual(pred[0]['label'],'__unmapped__')
                self.assertEqual(errors,['unknown_or_wrong_label'])
                result,_=scoring.score([record()],[dict(uid='a',predictions=pred)],kind)
                s=result['scores']['class_aware' if kind=='boxes' else 'released']['0.5' if kind=='boxes' else 'point']
                self.assertEqual(s['micro'],dict(tp=0,fp=1,fn=1,precision=0.,recall=0.,f1=0.))

    def test_qwen_omitted_or_correct_label(self):
        for extra in ({},{'label':'SHIP'},{'category':'ship'},{'label':'ship','category':'ship'}):
            pred,errors=parsing.qwen(json.dumps([dict(bbox_2d=[0,0,100,100],**extra)]),SPECS[0],(100,100),'boxes')
            self.assertEqual(pred[0]['label'],'ship'); self.assertFalse(errors)
        pred,errors=parsing.qwen('[{"bbox_2d":[0,0,100,100],"label":"ship","category":"car"}]',SPECS[0],(100,100),'boxes')
        self.assertEqual(pred[0]['label'],'__unmapped__'); self.assertTrue(errors)

    def test_scores_perfect_missing_duplicate(self):
        r=record()
        for predictions,f1 in ((r['gt'],1.),([],0.),(r['gt']*2,2/3)):
            out,_=scoring.score([r],[dict(uid='a',predictions=predictions)],'boxes')
            self.assertAlmostEqual(out['scores']['class_aware']['0.5']['f1'],f1)
            self.assertAlmostEqual(out['scores']['class_aware']['mean_f1_50_95'],f1)

    def test_cross_class_bug_is_not_primary(self):
        r=record(); r['gt'].append(dict(label='storagetank',hbox=[20,20,30,30]))
        pred=[dict(label='ship',hbox=[20,20,30,30]),dict(label='storagetank',hbox=[0,0,10,10])]
        out,_=scoring.score([r],[dict(uid='a',predictions=pred)],'boxes')
        self.assertEqual(out['scores']['released']['0.5']['f1'],1.)
        self.assertEqual(out['scores']['class_aware']['0.5']['f1'],0.)

    def test_point_containment_not_center(self):
        r=record(); preds=[dict(label='ship',point=[1,1])]*2
        out,_=scoring.score([r],[dict(uid='a',predictions=preds)],'pointing')
        self.assertEqual(out['scores']['released']['point']['micro']['tp'],1)
        self.assertAlmostEqual(out['scores']['released']['point']['f1'],2/3)

    def test_point_class_aware(self):
        r=record(); preds=[dict(label='storagetank',point=[1,1])]
        out,_=scoring.score([r],[dict(uid='a',predictions=preds)],'pointing')
        self.assertEqual(out['scores']['released']['point']['micro']['tp'],0)

    def test_point_boundary_and_overlap_are_one_to_one(self):
        r=record(); r['gt']*=2
        out,_=scoring.score([r],[dict(uid='a',predictions=[dict(label='ship',point=[10,10])])],'pointing')
        self.assertEqual(out['scores']['released']['point']['micro']['tp'],1)
        self.assertEqual(out['scores']['released']['point']['micro']['fn'],1)

    def test_iou_threshold_boundary_and_all_ten(self):
        pred=[dict(label='ship',hbox=[0,0,20,10])]
        out,_=scoring.score([record()],[dict(uid='a',predictions=pred)],'boxes')
        s=out['scores']['class_aware']
        self.assertEqual(s['0.5']['f1'],1.)
        self.assertEqual(s['0.55']['f1'],0.)
        self.assertAlmostEqual(s['mean_f1_50_95'],.1)
        self.assertEqual(len(scoring.THRESHOLDS),10)

    def test_randomized_independent_class_aware_matcher(self):
        rng=random.Random(983)
        def box():
            x,y=rng.randrange(40),rng.randrange(40)
            return [x,y,x+rng.randrange(1,30),y+rng.randrange(1,30)]
        def iou(a,b):
            area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            return area/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)
        for _ in range(80):
            r=record(); r['gt']=[dict(label=rng.choice(['ship','storagetank']),hbox=box()) for _ in range(rng.randrange(1,15))]
            pred=[dict(label=rng.choice(['ship','storagetank','__unmapped__']),hbox=box()) for _ in range(rng.randrange(20))]
            pred+=copy.deepcopy(r['gt'][:3])
            out,_=scoring.score([r],[dict(uid='a',predictions=pred)],'boxes')
            for threshold in scoring.THRESHOLDS:
                used=set()
                for gt in r['gt']:
                    eligible=[(iou(gt['hbox'],p['hbox']),j) for j,p in enumerate(pred)
                              if j not in used and p['label']==gt['label']]
                    best=max(eligible,key=lambda item:(item[0],-item[1]),default=(0.,-1))
                    if best[0]>=threshold: used.add(best[1])
                m=out['scores']['class_aware'][str(threshold)]['micro']
                self.assertEqual((m['tp'],m['fp'],m['fn']),(len(used),len(pred)-len(used),len(r['gt'])-len(used)))

    def test_threshold_and_no_added_cap(self):
        r=record(); pred=[dict(label='ship',hbox=[0,0,10,10],score=.3)]*301
        out,_=scoring.score([r],[dict(uid='a',predictions=pred)],'boxes',.25)
        self.assertEqual(out['scores']['class_aware']['0.5']['micro']['fp'],300)
        cap,_=scoring.score([r],[dict(uid='a',predictions=pred)],'boxes',.25,300)
        self.assertEqual(cap['scores']['class_aware']['0.5']['micro']['fp'],299)

    def test_macro_not_micro(self):
        a,b=record(),record('b'); b['gt']=b['gt']*9
        result,_=scoring.score([a,b],[dict(uid='a',predictions=a['gt']),dict(uid='b',predictions=[])],'boxes')
        s=result['scores']['class_aware']['0.5']
        self.assertEqual(s['f1'],.5); self.assertNotEqual(s['micro']['f1'],s['f1'])

    def test_native_rank_does_not_promote_invalid_topk(self):
        values=[dict(score=.9,native_rank=0),dict(score=.8,native_rank=300)]
        self.assertEqual(scoring.select_predictions(values,.25,300),values[:1])
        self.assertFalse(scoring.select_predictions([dict(score=.25)],.25))

    def test_smoke_includes_multiple_classes(self):
        a,b,d=record('a'),record('b'),record('d')
        d['gt']=[dict(label=c,hbox=[0,0,10,10]) for c in ('ship','storagetank','vehicle')]
        self.assertEqual([r['uid'] for r in e.selected_rows({'DIOR':[a,b,d]},'DIOR',0,1,True)],['a','d'])

    def test_smoke_includes_densest_and_does_not_repeat(self):
        a,m,d=record('a'),record('m'),record('d')
        m['gt']=[dict(label=c,hbox=[0,0,10,10]) for c in ('ship','storagetank','vehicle')]
        d['gt']*=50
        self.assertEqual([r['uid'] for r in e.selected_rows({'DIOR':[a,m,d]},'DIOR',0,1,True)],['a','m','d'])

    def test_per_query_seed_does_not_depend_on_other_classes(self):
        query=c.query_only(record())
        value=seed_for(query,'ship','boxes')
        query['class_specs']+=SPECS[1:]
        self.assertEqual(value,seed_for(query,'ship','boxes'))
        self.assertNotEqual(value,seed_for(query,'ship','pointing'))

    def test_jobs_cover_all_once_and_no_detector_points(self):
        tasks=c.jobs(list(c.DEFAULT_MODELS),4)
        self.assertEqual(len(tasks),96)
        self.assertEqual(len(set(tasks)),len(tasks))
        self.assertFalse(any(m in c.DETECTORS and k=='pointing' for m,k,*_ in tasks))
        data={'DIOR':[record(str(i)) for i in range(11)]}
        ids=[r['uid'] for shard in range(4) for r in e.selected_rows(data,'DIOR',shard,4,False)]
        self.assertEqual(sorted(ids),sorted(str(i) for i in range(11)))

    def test_identity_and_completeness(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); plan={'test':True}; data={'DIOR':[record()]}
            path=e.paths(root,'base_la','boxes','DIOR',0,1,False); path.parent.mkdir(parents=True)
            path.write_text('{"uid":"a"}\n')
            self.assertFalse(e.finished(root,'base_la','boxes','DIOR',0,1,False,plan,data))
            c.atomic(path.with_suffix('.done.json'),dict(plan_sha256=c.digest(plan),uids=['a'],predictions_sha256=c.sha(path)))
            self.assertTrue(e.finished(root,'base_la','boxes','DIOR',0,1,False,plan,data))
            path.write_text('{"uid":"other"}\n')
            with self.assertRaises(ValueError): e.finished(root,'base_la','boxes','DIOR',0,1,False,plan,data)

    def test_cpu_merge_integration(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'manifest.jsonl'
            source.write_text(json.dumps(record())+'\n')
            plan=dict(protocol=c.PROTOCOL,run_models=['base_la'],sources={},shards=1,
                geometry_policy='LA-native primary; strict sensitivity',
                datasets={d:dict(path=str(source),sha256=c.sha(source)) for d in c.DATASETS})
            c.atomic(root/'plan.json',plan)
            for ds in c.DATASETS:
                for kind in ('boxes','pointing'):
                    path=e.paths(root,'base_la',kind,ds,0,1,False); path.parent.mkdir(parents=True)
                    pred=record()['gt'] if kind=='boxes' else [dict(label='ship',point=[1,1])]
                    raw='<ref>ship</ref><box>'+('<0><0><100><100>' if kind=='boxes' else '<10><10>')+'</box>'
                    row=dict(uid='a',plan_sha256=c.digest(plan),query=c.query_only(record()),
                        answers=[dict(raw_response=raw,queried_categories=['ship'])],
                        predictions=pred,latency_seconds=.5,peak_gpu_mb=1,generated_tokens=6)
                    path.write_text(json.dumps(row)+'\n')
                    c.atomic(path.with_suffix('.done.json'),dict(plan_sha256=c.digest(plan),uids=['a'],predictions_sha256=c.sha(path)))
            from types import SimpleNamespace
            e.merge(SimpleNamespace(output=root,smoke=False))
            result=c.read(root/'summary.json')
            self.assertEqual(len(result),4)
            self.assertEqual(result['base_la/boxes/DIOR']['outputs_per_second'],2.)
            self.assertEqual(result['base_la/boxes/DIOR']['strict_sensitivity']['scores'],
                             result['base_la/boxes/DIOR']['scores'])

    def test_queue_respects_mask(self):
        devices=[dict(index=str(i),uuid='GPU-'+str(i)) for i in range(8)]
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='2,3'):
            self.assertEqual([d['index'] for d in allocated(devices)],['2','3'])

    def test_queue_limits(self):
        with self.assertRaises(ValueError): GpuPool(max_gpus=5)
        pool=GpuPool(max_gpus=4)
        self.assertFalse(pool.free(dict(compute_pids=[123],free_mb=50000,used_mb=0,utilization=0)))

    def test_queue_two_polls_limit_and_hardware_filter(self):
        devices=[dict(index=str(i),uuid='GPU-'+str(i),compute_pids=[],free_mb=40000,used_mb=0,utilization=0) for i in range(8)]
        with tempfile.TemporaryDirectory() as temp,patch('gpu_pool.probe',return_value=devices),patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'):
            pool=GpuPool(max_gpus=4,lock_dir=temp)
            self.assertEqual(pool.available(set()),[])
            self.assertEqual(len(pool.available(set())),4)
            self.assertEqual(len(pool.available({'GPU-0','GPU-1','GPU-2'})),1)
            self.assertEqual(pool.available(set(),eligible_uuids={'GPU-7'})[0]['uuid'],'GPU-7')
            devices[7]['compute_pids']=[42]
            self.assertEqual(pool.available(set(),eligible_uuids={'GPU-7'}),[])

    def test_queue_lease_rechecks_occupancy(self):
        d=dict(index='3',uuid='GPU-test',compute_pids=[],free_mb=40000,used_mb=0,utilization=0)
        with tempfile.TemporaryDirectory() as temp,patch('gpu_pool.probe',return_value=[d]),patch.dict(os.environ,CUDA_VISIBLE_DEVICES='3'):
            pool=GpuPool(lock_dir=temp)
            first=pool.claim(d)
            try:
                self.assertIsNotNone(first)
                self.assertIsNone(pool.claim(d))
            finally: first.close()
            d['compute_pids']=[42]
            self.assertIsNone(pool.claim(d))

    def test_freeze_rejects_environment_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'software.json'
            c.freeze(p,dict(torch='2.5.1'))
            c.freeze(p,dict(torch='2.5.1'))
            with self.assertRaises(ValueError): c.freeze(p,dict(torch='2.11.0'))
            self.assertEqual(c.read(p),dict(torch='2.5.1'))

    def test_queue_worker_python(self):
        cmd=remote.worker_command(Path('/tmp/out'),'base_la','pointing','DIOR',0,True,{})
        self.assertIn('--smoke',cmd); self.assertIn('DIOR',cmd)
        with self.assertRaises(ValueError): remote.worker_command(Path('/tmp/out'),'laedino','boxes','DIOR',0,False,{})

    def test_plan_keeps_virtualenv_interpreter_symlink(self):
        import sys
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); interpreter=root/'venv/bin/python'
            interpreter.parent.mkdir(parents=True)
            interpreter.symlink_to(sys.executable)
            c.atomic(root/'data.json',{})
            c.atomic(root/'models.json',{'gdino_tiny':{}})
            e.plan(SimpleNamespace(output=root,controls=root,models=['gdino_tiny'],shards=1,
                                   budget=8192,laedino_python=str(interpreter)))
            self.assertEqual(c.read(root/'plan.json')['laedino_python'],str(interpreter))

    def test_plan_rejects_duplicate_models(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            c.atomic(root/'data.json',{})
            c.atomic(root/'models.json',{'base_la':{}})
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                e.plan(SimpleNamespace(output=root,models=['base_la','base_la']))

    def test_setup_resolves_reused_detector_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/'reused/LAE-DINO/mmdetection_lae'
            source.mkdir(parents=True); (source/'setup.py').touch()
            c.atomic(root/'results/models.json',{'laedino':{'code':str(source.parent)}})
            interpreter=root/'env/bin/python'
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$TEST_SETUP_ARGS"\n')
            interpreter.chmod(0o700)
            env=dict(os.environ,CREATE_PYTHON=sys.executable,LAEDINO_ENV=str(root/'env'),
                RESULTS_DIR=str(root/'results'),TEST_SETUP_ARGS=str(root/'args'))
            env.pop('LAEDINO_CODE',None)
            subprocess.run(['bash',str(c.HERE/'setup_laedino.sh')],env=env,check=True,capture_output=True)
            self.assertIn(str(source),(root/'args').read_text().splitlines())


if __name__=='__main__': unittest.main()
