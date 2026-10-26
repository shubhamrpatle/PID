import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import audit
import evaluate as app


def first(suite):
    with (app.CONTROLS/"original_manifests"/(suite+".jsonl")).open() as stream:
        return json.loads(next(stream))


def test_complete_training_evidence_and_validation_selection():
    evidence, configs, done, curves = audit.inspect(audit.HERE/"metadata/runs"/audit.SPEC["run"])
    assert evidence["stages"]["stage1"]["training_rows"] == 7276
    assert evidence["stages"]["stage2"]["training_rows"] == 2492
    assert evidence["stages"]["stage2"]["selected_step"] == 1246
    best = app.release.validate_metadata(app.MODEL, configs, done, curves)
    assert best["step"] == 1246
    curves["stage2"][-1]["detection_grounding_macro_loss"] = 0
    with pytest.raises(ValueError, match="validation selection"):
        app.release.validate_metadata(app.MODEL, configs, done, curves)


@pytest.mark.parametrize("mutation", ["nonfinite", "missing_step", "objective", "data", "parent"])
def test_training_audit_rejects_inconsistent_evidence(monkeypatch, mutation):
    original_read, original_jsonl = audit.read, audit.jsonl
    def read(path):
        result = original_read(path)
        if str(path).endswith("stage2/config.json"):
            if mutation == "data":
                result["train_sha256"] = "wrong"
            if mutation == "parent":
                result["init_checkpoint_sha256"] = "wrong"
        return result
    def jsonl(path):
        rows = original_jsonl(path)
        if str(path).endswith("stage1/training_curve.jsonl"):
            if mutation == "nonfinite":
                rows[0]["gradient_norm"] = float("nan")
            if mutation == "missing_step":
                rows.pop()
            if mutation == "objective":
                rows[0]["round1_loss"] = 0.5
        return rows
    monkeypatch.setattr(audit, "read", read)
    monkeypatch.setattr(audit, "jsonl", jsonl)
    with pytest.raises(ValueError):
        audit.inspect(audit.HERE/"metadata/runs"/audit.SPEC["run"])


def test_recipe_comparison_checks_more_than_record_counts():
    directory = audit.HERE/"metadata/runs"/audit.SPEC["run"]
    _, configs, _, _ = audit.inspect(directory)
    refs = {"standard":dict(configs=copy.deepcopy(configs))}
    assert audit.inspect(directory, refs)[0]["matched_recipe_fields"]
    refs["standard"]["configs"]["stage2"]["head_lr"] = 999
    with pytest.raises(ValueError, match="recipe differs"):
        audit.inspect(directory, refs)


def test_frozen_runtime_and_queue_commands():
    assert app.previous.verify_runtime()
    assert app.release.SPECS == {app.MODEL: app.SPEC}
    assert len(app.jobs()) == 16 and len(app.jobs(True)) == 8
    assert len({(j["suite"],j["kind"]) for j in app.jobs()}) == 8
    assert all(j["model"] == "class_isolated" and j["shard_count"] == 2 for j in app.jobs())
    assert all(j["limit"] == 2 for j in app.jobs(True))
    assert app.command(app.OUTPUT, app.jobs()[0], "infer")[2:5] == [str(app.HERE/"evaluate.py"), "worker", "infer"]
    env = app.environment("base_la", "3")
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["CLASS_ISOLATED_OUTPUT"] == str(app.OUTPUT)
    assert env["REFWAVE_OUTPUT"] == str(app.PID_OUTPUT)
    assert env["CLASS_ISOLATED_CHECKPOINTS"] == str(app.DOWNLOAD)


@pytest.mark.parametrize("suite,count", [("DIOR",20),("DOTAv2",18)])
def test_class_queries_cover_absent_classes_and_ignore_gt(suite, count):
    row = first(suite)
    altered = copy.deepcopy(row)
    altered.update(gt=[], target_class="invented", gt_hbox=[0,0,1,1])
    branches = app.evaluation.isolated_branches(row)
    assert branches == app.evaluation.isolated_branches(altered)
    assert len(branches) == count
    assert [b.label for b in branches] == [s["canonical"] for s in row["class_specs"]]
    from refwave.direct_control import detection_question
    assert all(b.question == detection_question(s["prompt"]) for b,s in zip(branches,row["class_specs"]))


def test_training_class_partition_preserves_targets_and_negatives():
    from refwave.direct_control import isolated_conversations
    answer = "<ref>ship</ref><box><1><2><3><4></box><box><5><6><7><8></box><ref>vehicle</ref>None"
    row = dict(direct=dict(conversations=[{},dict(value=answer)]),
        round1=dict(branches=[dict(reference="ship"),dict(reference="vehicle")]))
    conversations = isolated_conversations(row)
    assert len(conversations) == 2
    assert "".join(c[1]["value"] for c in conversations) == answer
    row["round1"]["branches"].reverse()
    with pytest.raises(ValueError):
        isolated_conversations(row)


def test_detection_uses_concurrent_native_direct_generator(monkeypatch):
    row = first("DIOR")
    calls = []
    class Decoder:
        def generate(self, image, branches, **kwargs):
            calls.append((branches,kwargs))
            return [dict(answer="<box><100><200><300><400></box>") for _ in branches], dict(waves=1)
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",lambda:None)
    monkeypatch.setattr(torch.cuda,"max_memory_allocated",lambda:1024**2)
    result = app.evaluation.boxes(None, Decoder(), row, SimpleNamespace(size=(100,100)), app.MODEL)
    assert len(calls) == 1 and len(calls[0][0]) == 20
    assert calls[0][1] == app.GENERATION
    assert len(result["predictions"]) == 20
    assert result["predictions"][0]["hbox"] == [10,20,30,40]


@pytest.mark.parametrize("suite", ["DIOR-RSVG", "VRSBench-VG"])
def test_grounding_is_unchanged_direct_full_expression(monkeypatch, suite):
    row = first(suite)
    calls = []
    monkeypatch.setattr(app.evaluation.prior,"infer_one",lambda *args: calls.append(args) or {"ok": True})
    result = app.evaluation.boxes(None, None, row, "image", app.MODEL)
    assert result["ok"] and len(calls) == 1
    assert calls[0][-2:] == ("standard",8192)
    assert calls[0][2] == app.evaluation.prior.inference_query(row)


@pytest.mark.parametrize("kind,suite", [("boxes","DIOR"),("boxes","DOTAv2"),
    ("boxes","DIOR-RSVG"),("boxes","VRSBench-VG"),("pointing","DIOR"),("pointing","VRSBench-VG")])
def test_native_worker_resume_and_real_scorer(tmp_path, monkeypatch, kind, suite):
    from PIL import Image
    output = tmp_path/"results"
    assets = output/"assets"
    assets.mkdir(parents=True)
    rows = []
    for i in range(2):
        row = first(suite)
        row.update(uid=f"fixture-{i}", image_id=f"image-{i}")
        if row["task"] == "grounding":
            row["sample_id"] = f"sample-{i}"
        path = tmp_path/f"image-{i}.png"
        Image.new("RGB", (int(row["width"]), int(row["height"]))).save(path)
        row["image_path"] = str(path)
        rows.append(row)
    manifest = assets/(suite+".jsonl")
    manifest.write_text("".join(json.dumps(r)+"\n" for r in rows))
    plan = dict(protocol=app.PROTOCOL, models={app.MODEL:app.SPEC}, sources={},
        manifests={suite:dict(sha256=audit.sha(manifest))}, generation=app.GENERATION, timing="fixture",
        hardware=dict(gpu_model="mock GPU",torch=str(torch.__version__)), software={})
    app.write(output/"plan.json", plan)
    app.write(output/"selection.json",{app.MODEL:dict(app.SPEC,method=app.MODEL)})
    monkeypatch.setattr(app,"OUTPUT",output)
    monkeypatch.setattr(app,"software",lambda:{})
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda _:"mock GPU")
    loads=[]
    monkeypatch.setattr(app.evaluation,"load_worker",lambda *a:loads.append(1))
    import pixel_pivr.fast_class_wave as wave
    monkeypatch.setattr(wave,"FastCachedHybridClassWaveDecoder",lambda *a,**kw:None)
    def prediction(worker, decoder, row, image, method):
        return dict(latency_seconds=.01,peak_gpu_mb=1,predictions=[])
    def point_prediction(worker,row,image):
        return dict(latency_seconds=.01,peak_gpu_mb=1,points=[], answers=[dict(label=r["label"],
            points=[],latency_seconds=.01,parse_errors=[],budget_saturated=False) for r in app.base.references(row)])
    monkeypatch.setattr(app.evaluation,"boxes",prediction)
    monkeypatch.setattr(app.evaluation,"points",point_prediction)
    args=SimpleNamespace(output=output,model=app.MODEL,kind=kind,suite=suite,
        shard_index=0,shard_count=2,limit=2)
    for i in range(2):
        args.shard_index=i
        app.infer(args)
        app.infer(args)
    assert len(loads)==2
    app.merge(args)
    metric=app.read(app.base.folder(args)/"metrics.json")
    assert metric["complete"] and metric["records"] == 2 and metric["mean_seconds"] == .01
    if kind=="boxes" and suite in ("DIOR","DOTAv2"):
        assert len(metric["raw"]) == 10
    else:
        assert ("pointing" if kind=="pointing" else "grounding") in metric


def test_new_checkpoint_can_never_reuse_old_stage2_best(tmp_path,monkeypatch):
    download=tmp_path/"download"
    path=download/"runs"/app.SPEC["run"]/"stage2/best.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong-old-checkpoint")
    monkeypatch.setattr(app.release,"DOWNLOAD",download)
    monkeypatch.setattr(app,"OUTPUT",tmp_path/"output")
    with pytest.raises(ValueError,match="hash/size"):
        app.verify_checkpoint({})
    assert path.read_bytes() == b"wrong-old-checkpoint"


def test_completed_checkpoint_skips_network_and_hash_on_resume(tmp_path,monkeypatch):
    download=tmp_path/"download"
    path=download/"runs"/app.SPEC["run"]/"stage2/best.pt"
    path.parent.mkdir(parents=True)
    path.touch()
    output=tmp_path/"output"
    output.mkdir()
    spec=dict(app.SPEC,bytes=0)
    app.write(output/"checkpoints_ready.json",dict(complete=True,models={app.MODEL:dict(
        verified=True,path=str(path),sha256=spec["sha256"],step=spec["step"],bytes=0,mtime_ns=path.stat().st_mtime_ns)}))
    monkeypatch.setattr(app.release,"DOWNLOAD",download)
    monkeypatch.setattr(app,"OUTPUT",output)
    monkeypatch.setattr(app,"SPEC",spec)
    monkeypatch.setattr(audit,"sha",lambda *a:pytest.fail("Unnecessary full rehash"))
    app.verify_checkpoint({})


def test_cpu_handoff_imports_choose_private_hf_cache():
    from huggingface_hub import constants
    assert str(Path(constants.HF_TOKEN_PATH).parent) == str(app.HERE/".hf-cache")
    assert Path(app.evaluation.__file__).resolve().is_relative_to(app.CONTROLS/"workspace")


def metric_fixture():
    c=dict(limit=0,manifest_sha256="manifest",checkpoint_sha256=app.SPEC["sha256"],step=1246,
        method=app.MODEL,generation=app.GENERATION,image_token_limit=6000,added_nms=False,
        max_detections=None,shard_count=2,warmup_records=2,suite="DIOR",kind="boxes",
        gpu="mock GPU",torch="mock version",class_isolated_max_concurrent_references=32,native_query_batch=1)
    return dict(complete=True,records=586,contract=c,raw={f"{t:g}":dict(f1=.5) for t in app.base.THRESHOLDS},
        mean_f1_iou_0p50_to_0p95=.5,mean_seconds=2.0,bps=10,cbps_iou0p5=5,prediction_files={})


@pytest.mark.parametrize("key,value", [("manifest_sha256","wrong"),("checkpoint_sha256","old"),
    ("method","standard"),("gpu","other GPU"),("limit",2),("added_nms",True),
    ("step",2492),("class_isolated_max_concurrent_references",1)])
def test_report_rejects_unmatched_results(key,value):
    metric=metric_fixture()
    hardware=dict(gpu_model="mock GPU",torch="mock version")
    selected=dict(app.SPEC,method=app.MODEL)
    app.validate_metric(metric,"boxes","DIOR","manifest",selected,hardware)
    metric["contract"][key]=value
    with pytest.raises(ValueError):
        app.validate_metric(metric,"boxes","DIOR","manifest",selected,hardware)


def test_report_missing_results_and_percent_units(tmp_path,monkeypatch):
    output=tmp_path/"results"
    output.mkdir()
    selected=dict(app.SPEC,method=app.MODEL)
    plan=dict(protocol=app.PROTOCOL,models={app.MODEL:app.SPEC},sources={},
        manifests={s:dict(sha256="manifest") for s in app.EXPECTED},
        historical_selections={name:dict(selected,method=name) for name in ("standard","refwave")},
        hardware=dict(gpu_model="mock GPU",torch="mock version"),comparison_limits=[])
    app.write(output/"plan.json",plan)
    app.write(output/"selection.json",{app.MODEL:selected})
    monkeypatch.setattr(app,"OUTPUT",output)
    monkeypatch.setattr(app,"CONTROLS",tmp_path/"old-controls")
    monkeypatch.setattr(app,"PID_OUTPUT",tmp_path/"old-pid")
    assert app.report()==0
    assert "| - | - | - | - |" in (output/"comparison.md").read_text()
    metric=metric_fixture()
    metric["contract"].update(protocol=app.PROTOCOL,plan_sha256=audit.sha(output/"plan.json"))
    path=output/"results/boxes"/app.MODEL/"DIOR/metrics.json"
    app.write(path,metric)
    assert app.report()==1
    assert "50.00 / 50.00 / 50.00 | 2.000 | 10.000 | 5.000" in (output/"comparison.md").read_text()
    assert not app.read(output/"comparison.json")["complete"]
    assert not (output/"all.done.json").exists()


def test_worker_refuses_software_or_hardware_change(tmp_path,monkeypatch):
    plan=dict(protocol=app.PROTOCOL,models={app.MODEL:app.SPEC},sources={},software={"x":1},
        hardware=dict(gpu_model="original",torch=str(torch.__version__)))
    app.write(tmp_path/"plan.json",plan)
    monkeypatch.setattr(app,"OUTPUT",tmp_path)
    monkeypatch.setattr(app,"software",lambda:{})
    with pytest.raises(ValueError,match="environment"):
        app.infer(None)
    monkeypatch.setattr(app,"software",lambda:{"x":1})
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda _:"different")
    with pytest.raises(ValueError,match="GPU/Torch"):
        app.infer(None)


def test_frozen_manifest_copy_preserves_every_byte(tmp_path,monkeypatch):
    controls=tmp_path/"controls"
    manifest=controls/"results/assets/DIOR.jsonl"
    manifest.parent.mkdir(parents=True)
    row=first("DIOR")
    image=tmp_path/"image.png"
    image.touch()
    row["image_path"]=str(image)
    manifest.write_text(json.dumps(row)+"\n")
    summary=dict(sha256=audit.sha(manifest),records=1)
    app.write(manifest.with_suffix(".json"),summary)
    app.write(controls/"results/plan.json",dict(sources={},manifests={"DIOR":summary}))
    monkeypatch.setattr(app.previous,"CONTROLS",controls)
    monkeypatch.setattr(app.previous,"OUTPUT",tmp_path/"new")
    monkeypatch.setattr(app.previous,"EXPECTED",{"DIOR":1})
    before=manifest.read_bytes()
    app.previous.copy_manifests()
    app.previous.copy_manifests()
    assert manifest.read_bytes()==before==(tmp_path/"new/assets/DIOR.jsonl").read_bytes()


def test_prepare_new_release_without_touching_existing_results(tmp_path,monkeypatch):
    controls,pid_root,output=tmp_path/"controls",tmp_path/"pid",tmp_path/"new/results"
    source=controls/"results/assets/DIOR.jsonl"
    source.parent.mkdir(parents=True)
    row=first("DIOR")
    image=tmp_path/"image.png"
    image.touch()
    row["image_path"]=str(image)
    source.write_text(json.dumps(row)+"\n")
    manifest=dict(sha256=audit.sha(source),records=1)
    app.write(source.with_suffix(".json"),manifest)
    app.write(controls/"results/plan.json",dict(sources={},manifests={"DIOR":manifest}))
    evidence,configs,done,curves=audit.inspect(audit.HERE/"metadata/runs"/audit.SPEC["run"])
    selected=dict(app.SPEC,method=app.MODEL,configs=configs,completions=done,curves=curves)
    for model,folder in (("standard",controls/"results"),("refwave",pid_root/"results")):
        app.write(folder/"selection.json",{model:dict(selected,method=model)})
    signatures={str(app.HERE/"audit.py"):audit.sha(app.HERE/"audit.py")}
    before={str(p):audit.sha(p) for root in (controls,pid_root) for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(app,"OUTPUT",output)
    monkeypatch.setattr(app,"DOWNLOAD",tmp_path/"new/checkpoints")
    monkeypatch.setattr(app,"CONTROLS",controls)
    monkeypatch.setattr(app,"PID_OUTPUT",pid_root/"results")
    monkeypatch.setattr(app.previous,"CONTROLS",controls)
    monkeypatch.setattr(app.previous,"EXPECTED",{"DIOR":1})
    monkeypatch.setattr(app.previous,"verify_runtime",lambda:dict(signatures))
    monkeypatch.setattr(app.scheduler,"control_hardware",lambda:dict(gpu_model="test",torch="test"))
    monkeypatch.setattr(app,"software",lambda:dict(torch="test"))
    def metadata(references):
        assert set(references)=={"standard","refwave"}
        audit.inspect(audit.HERE/"metadata/runs"/audit.SPEC["run"],references)
        return selected,evidence
    monkeypatch.setattr(app,"metadata",metadata)
    calls=[]
    monkeypatch.setattr(app,"verify_checkpoint",lambda s:calls.append(s))
    app.prepare()
    app.prepare()
    assert len(calls)==2
    assert app.read(output/"selection.json")[app.MODEL]["step"]==1246
    assert app.read(output/"plan.json")["protocol"]==app.PROTOCOL
    assert (output/"assets/DIOR.jsonl").read_bytes()==source.read_bytes()
    after={str(p):audit.sha(p) for root in (controls,pid_root) for p in root.rglob("*") if p.is_file()}
    assert before==after
