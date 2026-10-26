import json
from types import SimpleNamespace

import pytest

import dynamic_queue as scheduler

evaluation = scheduler.evaluation


def inventory(names=None):
    names = names or ['NVIDIA A100-SXM4-80GB']*8
    return {str(i):dict(index=str(i), uuid=f'GPU-{i}', name=name, memory_mb=81920)
            for i,name in enumerate(names)}


def policy():
    return dict(candidate_gpus=list(map(str,range(8))), eligible_gpus=list(map(str,range(8))),
                hardware=dict(gpu_model='NVIDIA A100-SXM4-80GB'))


def test_pool_skips_other_hardware_without_changing_gpu_identity(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    devices = inventory(['NVIDIA A100-SXM4-80GB']*4+['NVIDIA H100']*4)
    good, bad = scheduler.select_pool(list(devices), devices, policy()['hardware'])
    assert [d['requested_id'] for d in good] == ['0','1','2','3']
    assert [d['requested_id'] for d in bad] == ['4','5','6','7']
    with pytest.raises(ValueError,match='No requested GPU matches'):
        scheduler.select_pool(['4'],devices,policy()['hardware'])
    with pytest.raises(ValueError,match='same physical GPU'):
        scheduler.select_pool(['0','GPU-0'],devices,policy()['hardware'])
    with pytest.raises(ValueError,match='physical GPU'):
        scheduler.select_pool(['9'],devices,policy()['hardware'])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2,3')
    with pytest.raises(ValueError,match='outside'):
        scheduler.select_pool(['0'],devices,policy()['hardware'])
    good,_=scheduler.select_pool(['2','3'],devices,policy()['hardware'])
    assert len(good)==2


def test_controls_hardware_is_not_guessed(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluation,'CONTROLS',tmp_path)
    with pytest.raises(ValueError,match='completed controls'):
        scheduler.control_hardware()
    path=tmp_path/'results/results/boxes/standard/DIOR/metrics.json'
    path.parent.mkdir(parents=True)
    data=dict(complete=True,records=586,contract=dict(limit=0,gpu='A100',torch='2.11.0+cu128'))
    path.write_text(json.dumps(data))
    assert scheduler.control_hardware()['gpu_model']=='A100'
    second=tmp_path/'results/results/boxes/class_isolated/DIOR/metrics.json'
    second.parent.mkdir(parents=True)
    data['contract']['gpu']='H100'
    second.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='one GPU model'):
        scheduler.control_hardware()


@pytest.mark.parametrize('cap',[0,9,-1])
def test_invalid_worker_cap_rejected(cap):
    with pytest.raises(ValueError):
        scheduler.validate_max_workers(cap)


@pytest.mark.parametrize('cap,availability',[
    (4,'all'), (1,'all'), (4,'one_then_four'), (4,'all_busy_then_free')])
def test_real_queue_obeys_cap_and_uses_newly_free_gpus(tmp_path,monkeypatch,cap,availability):
    queue=evaluation.queue
    active=[]
    processes=[]
    ticks=[0]
    peaks=[]
    states=[]
    jobs=[dict(model='refwave',kind='boxes',suite=f'synthetic-{i}',
               shard_index=0,shard_count=1,limit=0) for i in range(9)]
    (tmp_path/'logs').mkdir()

    def idle(gpu):
        if availability=='one_then_four' and ticks[0]<2:
            return gpu=='6'
        if availability=='all_busy_then_free' and ticks[0]<2:
            return False
        return True

    class Process:
        def __init__(self,cmd,**kwargs):
            self.cmd=cmd
            self.gpu=kwargs['env']['CUDA_VISIBLE_DEVICES']
            self.pid=1000+len(processes)
            self.started=ticks[0]
            self.finished=False
            assert len(active)<cap
            assert not any(p.gpu==self.gpu and not p.finished for p in processes)
            processes.append(self)
            peaks.append(sum(not p.finished for p in processes))

        def poll(self):
            if ticks[0]-self.started<2:
                return None
            self.finished=True
            job=next(j for j in jobs if j['suite']==self.cmd[1])
            dest=queue.folder(tmp_path,job)
            dest.mkdir(parents=True,exist_ok=True)
            (dest/'shard00of01.done.json').write_text('{}')
            return 0

    def sleep(_):
        ticks[0]+=1
        assert ticks[0]<40,'Queue did not drain'

    monkeypatch.setattr(queue,'jobs',lambda _:list(jobs))
    monkeypatch.setattr(queue,'command',lambda output,job,mode:[mode,job['suite']])
    monkeypatch.setattr(queue.scheduling,'environment',lambda model,gpu=None:{'CUDA_VISIBLE_DEVICES':gpu})
    monkeypatch.setattr(queue.scheduling,'idle',idle)
    monkeypatch.setattr(scheduler,'low_utilization',lambda _:True)
    monkeypatch.setattr(queue.subprocess,'Popen',Process)
    monkeypatch.setattr(queue.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0))
    monkeypatch.setattr(queue.time,'sleep',sleep)
    monkeypatch.setattr(queue,'write_status',lambda output,state,**k:states.append(dict(state=state,**k)))
    before=evaluation.jobs(False)
    original_phase=queue.run_phase
    args=SimpleNamespace(output=tmp_path,gpus=list(map(str,range(8))))
    with scheduler.capped_scheduler(cap,policy()):
        queue.run_phase(args,False,active)
    assert queue.run_phase is original_phase and queue.scheduling.idle is idle
    assert len(processes)==len(jobs) and len({tuple(p.cmd) for p in processes})==len(jobs)
    assert max(peaks)<=cap and not active
    assert all(s['max_workers']==cap and len(s['active'])<=cap for s in states)
    assert evaluation.jobs(False)==before  # Shards and record membership are scheduling-independent.
    if availability=='one_then_four':
        assert processes[0].gpu=='6' and len({p.gpu for p in processes})>1
    if availability=='all_busy_then_free':
        assert states[0]['state']=='waiting_for_idle_gpu' and processes[0].started>=2


def test_scheduler_restores_functions_on_failure(monkeypatch):
    queue=evaluation.queue
    def fail(*unused):
        raise RuntimeError('worker failure')
    monkeypatch.setattr(queue,'run_phase',fail)
    idle,status=queue.scheduling.idle,queue.write_status
    with pytest.raises(RuntimeError,match='worker failure'):
        with scheduler.capped_scheduler(4,policy()):
            queue.run_phase(None,False,[])
    assert queue.run_phase is fail and queue.scheduling.idle is idle and queue.write_status is status


def test_unknown_utilization_is_not_treated_as_idle(monkeypatch):
    monkeypatch.setattr(scheduler.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='N/A'))
    assert scheduler.low_utilization('0') is False
    monkeypatch.setattr(scheduler.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='12'))
    assert scheduler.low_utilization('0') is False
    monkeypatch.setattr(scheduler.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='0'))
    assert scheduler.low_utilization('0') is True


def test_launcher_passes_cap_and_all_eight_candidates():
    import os
    import subprocess
    here=evaluation.HERE
    # The fake interpreter prints exec arguments without loading a model.
    env=dict(os.environ,PYTHON_BIN='/bin/echo',MAX_WORKERS='4')
    result=subprocess.run(['bash',str(here/'run.sh'),*map(str,range(8))],
                          env=env,text=True,capture_output=True,check=True)
    assert result.stdout.strip()=='-u dynamic_queue.py launch --max-workers 4 --gpus 0 1 2 3 4 5 6 7'


def test_gpu_probe_failure_waits_instead_of_claiming_idle(monkeypatch):
    queue=evaluation.queue
    def unavailable(gpu):
        raise ValueError('GPU occupancy unavailable')
    def phase(args,smoke,active):
        assert queue.scheduling.idle('0') is False
    monkeypatch.setattr(queue.scheduling,'idle',unavailable)
    monkeypatch.setattr(queue,'run_phase',phase)
    with scheduler.capped_scheduler(4,policy()):
        queue.run_phase(None,False,[])
