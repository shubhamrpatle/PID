import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import remote_eval as remote


def first(suite):
    with (remote.CONTROLS/'original_manifests'/f'{suite}.jsonl').open() as stream:
        return json.loads(next(stream))


def test_runtime_is_existing_frozen_package():
    signatures = remote.verify_runtime()
    assert signatures
    for module in (remote.evaluation, remote.base, remote.release, remote.queue, remote.pid):
        assert Path(module.__file__).resolve().is_relative_to(remote.CONTROLS/'workspace')
    assert remote.pid.POLICY['max_addresses'] is None
    assert remote.pid.POLICY['oom_fallback'] is False
    assert remote.pid.POLICY['round2'] == 'packed_all'


def test_queue_is_eight_cells_two_shards_and_all_suite_smokes():
    jobs = remote.jobs(False)
    assert len(jobs) == 16
    assert len({(j['kind'],j['suite']) for j in jobs}) == 8
    assert all(j['model']=='refwave' and j['shard_count']==2 and j['limit']==0 for j in jobs)
    smoke = remote.jobs(True)
    assert len(smoke)==8 and all(j['limit']==2 and j['shard_count']==1 for j in smoke)
    cmd = remote.command(remote.OUTPUT, jobs[0], 'infer')
    assert cmd[:5] == [sys.executable, '-u', str(remote.HERE/'remote_eval.py'), 'worker', 'infer']
    env = remote.environment('base_la','3')
    assert env['CUDA_VISIBLE_DEVICES']=='3'
    assert env['CONTROLS_DIR']==str(remote.CONTROLS)
    assert env['REFWAVE_OUTPUT']==str(remote.OUTPUT)


@pytest.mark.parametrize('suite,count',[('DIOR',20),('DOTAv2',18),('DIOR-RSVG',1),('VRSBench-VG',1)])
def test_reference_queries_are_gt_independent(suite,count):
    row = first(suite)
    changed = copy.deepcopy(row)
    changed.update(gt=[], gt_hbox=[0,0,0,0], target_class='invented')
    branches = remote.point_branches(row)
    assert branches == remote.point_branches(changed)
    assert len(branches)==count
    for b,ref in zip(branches,remote.base.references(row)):
        assert b.question==f"Point to: {ref['text']}."
        text=json.dumps([0,row['suite'],row['uid'],ref['label']])
        assert b.seed==int.from_bytes(hashlib.sha256(text.encode()).digest()[:4],'big')
    assert remote.evaluation.prior.inference_query(row)==remote.evaluation.prior.inference_query(changed)


def test_metadata_best_and_parent_identity():
    folder=remote.DOWNLOAD/'runs'/remote.SPEC['run']
    if not (folder/'stage2/config.json').exists():
        pytest.skip('Run metadata once to download the six small upstream files')
    configs={s:remote.read(folder/s/'config.json') for s in ('stage1','stage2')}
    completions={s:remote.read(folder/s/'done.json') for s in configs}
    curves={s:remote.base.common.read(folder/s/'validation_curve.jsonl') for s in configs}
    best=remote.release.validate_metadata('refwave',configs,completions,curves)
    assert best['step']==2400 and configs['stage2']['steps']==2492
    broken=copy.deepcopy(curves)
    broken['stage2'][0]['detection_grounding_macro_loss']=-1
    with pytest.raises(ValueError,match='validation selection'):
        remote.release.validate_metadata('refwave',configs,completions,broken)
    configs['stage2']['init_checkpoint_sha256']='wrong'
    with pytest.raises(ValueError,match='parent'):
        remote.release.validate_metadata('refwave',configs,completions,curves)


def test_pointing_concurrency_parser_and_timing_are_compatible(monkeypatch):
    row=first('DIOR')
    calls=[]
    class Decoder:
        def generate(self,image,branches,**kwargs):
            calls.append((branches,kwargs))
            # Four-coordinate boxes must not be converted to points.
            raw='<box><100><200></box><box><0><0><500><500></box>'
            return [dict(label=b.label,answer=raw) for b in branches], dict(branches=20,waves=1,peak_gpu_mb=7)
    decoder=Decoder()
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:1024**2)
    worker=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a,**k:[1,2,3]))
    result=remote.pointing(worker,decoder,row,SimpleNamespace(size=(100,100)))
    assert len(calls)==1 and decoder.max_wave_size==20
    assert len(result['points'])==20 and result['round2_execution'] is None
    assert all(a['latency_seconds'] is None for a in result['answers'])
    assert all(a['points'][0]['point']==[10,20] for a in result['answers'])
    assert all(a['parse_errors']==['non_point_or_invalid_coordinates'] for a in result['answers'])
    assert result['latency_seconds']>=0 and result['peak_gpu_mb']==7
    score,per=remote.base.point_metrics([row],[result])
    assert score['queries']==20 and all(p['seconds'] is None for p in per)


def test_copy_manifests_is_byte_identical(tmp_path,monkeypatch):
    old=tmp_path/'controls'
    output=tmp_path/'new'
    source=old/'results/assets/DIOR.jsonl'
    source.parent.mkdir(parents=True)
    row=first('DIOR')
    image=tmp_path/'image.png'
    image.write_bytes(b'image-existence-fixture')
    row['image_path']=str(image)
    source.write_text(json.dumps(row)+'\n')
    sha=remote.base.common.sha(source)
    summary=dict(sha256=sha,records=1)
    source.with_suffix('.json').write_text(json.dumps(summary))
    (old/'results/plan.json').write_text(json.dumps(dict(manifests={'DIOR':summary},sources={})))
    monkeypatch.setattr(remote,'CONTROLS',old)
    monkeypatch.setattr(remote,'OUTPUT',output)
    monkeypatch.setattr(remote,'EXPECTED',{'DIOR':1})
    remote.copy_manifests()
    assert (output/'assets/DIOR.jsonl').read_bytes()==source.read_bytes()
    before=source.read_bytes()
    remote.copy_manifests()
    assert source.read_bytes()==before
    (output/'assets/DIOR.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='differs'):
        remote.copy_manifests()


def test_hf_cache_does_not_read_shared_token(tmp_path):
    env=dict(os.environ,EVAL_HF_HOME=str(tmp_path/'hf'),HF_TOKEN_PATH='/shared/denied/token')
    result=subprocess.run([sys.executable,'-c',
        'import remote_eval; from huggingface_hub import constants; print(constants.HF_TOKEN_PATH)'],
        cwd=remote.HERE,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip().splitlines()[-1]==str(tmp_path/'hf/token')


def test_gpu_allocation_and_launcher_syntax(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2,3')
    assert remote.validate_gpus(['2','3'])==['2','3']
    with pytest.raises(ValueError):
        remote.validate_gpus(['0'])
    subprocess.run(['bash','-n',str(remote.HERE/'run.sh')],check=True)


@pytest.mark.parametrize('task,suite',[('detection','DIOR'),('grounding','DIOR-RSVG')])
def test_packed_box_dispatch_uses_frozen_queries_without_gt(monkeypatch,task,suite):
    row=first(suite)
    captured=[]
    def inference(worker,decoders,query,image,method,config):
        captured.append((query,config))
        return dict(round1=dict(branches=len(query['references']),waves=1))
    monkeypatch.setattr(remote.pid.ablation,'infer_one',inference)
    remote.pid.infer_one(None,None,row,None,8192)
    query,config=captured[0]
    assert not {'gt','gt_hbox','target_class','image_path'} & query.keys()
    assert config['class_wave']==len(row['references']) and config['budget']==8192


def test_report_leaves_missing_cells_blank_and_checks_identity(tmp_path,monkeypatch):
    output=tmp_path/'new'
    output.mkdir()
    manifest={'sha256':'frozen'}
    (output/'plan.json').write_text(json.dumps(dict(manifests={'DIOR':manifest})))
    (output/'selection.json').write_text(json.dumps({'refwave':{'sha256':'weights','step':2400}}))
    directory=output/'results/boxes/refwave/DIOR'
    directory.mkdir(parents=True)
    metric=dict(complete=True,records=586,mean_seconds=1.0,
        contract=dict(limit=0,manifest_sha256='frozen',checkpoint_sha256='weights',step=2400),
        raw={'0.5':{'f1':.4},'0.75':{'f1':.2}},mean_f1_iou_0p50_to_0p95=.1)
    (directory/'metrics.json').write_text(json.dumps(metric))
    monkeypatch.setattr(remote,'OUTPUT',output)
    monkeypatch.setattr(remote,'CONTROLS',tmp_path/'controls')
    monkeypatch.setattr(remote,'EXPECTED',{'DIOR':586})
    result=remote.report()
    assert result['refwave_completed_cells']==1
    assert '| 40.00 / 20.00 / 10.00 | 1.000 |' in (output/'comparison.md').read_text()
    metric['contract']['checkpoint_sha256']='different'
    (directory/'metrics.json').write_text(json.dumps(metric))
    with pytest.raises(ValueError,match='different checkpoint'):
        remote.report()


@pytest.mark.parametrize('kind,suite',[('boxes','DIOR'),('boxes','DIOR-RSVG'),('pointing','DIOR')])
def test_worker_resume_and_real_scorer_integration(tmp_path,monkeypatch,kind,suite):
    from PIL import Image
    output=tmp_path/'outputs'
    assets=output/'assets'
    assets.mkdir(parents=True)
    rows=[]
    for i in range(2):
        row=first(suite)
        row['uid']=f'fixture-{i}'
        row['image_id']=f'image-{i}'
        if row['task']=='grounding':
            row['sample_id']=f'query-{i}'
        path=tmp_path/f'image-{i}.png'
        Image.new('RGB',(int(row['width']),int(row['height']))).save(path)
        row['image_path']=str(path)
        rows.append(row)
    manifest=assets/f'{suite}.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    plan=dict(protocol=remote.pid.PROTOCOL_ID,sources={},
        manifests={suite:dict(sha256=remote.base.common.sha(manifest))},
        generation=dict(max_new_tokens=8192),timing='test')
    (output/'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(remote,'OUTPUT',output)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _: 'CPU mocked device')
    loads=[]
    monkeypatch.setattr(remote.evaluation,'load_worker',lambda *a:loads.append(1))
    monkeypatch.setattr(remote.pid,'make_decoders',lambda *a:None)
    import pixel_pivr.fast_class_wave as wave
    monkeypatch.setattr(wave,'FastCachedHybridClassWaveDecoder',lambda *a,**k:None)

    def mock_prediction(worker,decoder,row,image,*unused):
        common=dict(latency_seconds=.01,peak_gpu_mb=1,protocol=remote.pid.PROTOCOL_ID)
        if kind=='boxes':
            return dict(common,predictions=[])
        return dict(common,points=[],answers=[dict(label=r['label'],points=[],
            latency_seconds=None,parse_errors=[],budget_saturated=False) for r in remote.base.references(row)])

    monkeypatch.setattr(remote.pid,'infer_one',mock_prediction)
    monkeypatch.setattr(remote,'pointing',mock_prediction)
    args=SimpleNamespace(output=output,model='refwave',kind=kind,suite=suite,
                         shard_index=0,shard_count=2,limit=2)
    for index in range(2):
        args.shard_index=index
        remote.infer(args)
        remote.infer(args)  # Completed shard must not reload or duplicate predictions.
    assert len(loads)==2
    remote.base.merge(args)
    metric=remote.read(remote.base.folder(args)/'metrics.json')
    assert metric['complete'] and metric['records']==2
    assert metric['mean_seconds']==.01
    assert ('pointing' if kind=='pointing' else ('raw' if suite=='DIOR' else 'grounding')) in metric
