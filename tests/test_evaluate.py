import json
from pathlib import Path
import sys
import time

import pytest
from PIL import Image

from viridian402.cli import main
from viridian402.evaluate import evaluate, public_request, _preview
from viridian402.report import compare_runs, load_run


@pytest.fixture
def manifest(tmp_path):
    Image.new("RGB",(20,100),(120,80,50)).save(tmp_path/"image.png")
    (tmp_path/"boxes.json").write_text(json.dumps({"width":20,"height":100,"bounding_boxes":{"interpreted_lithology_column":[0,0,20,100]},"layers":[{"answer":"secret"}],"truth":"secret.csv"}))
    (tmp_path/"truth.csv").write_text("top_y,bottom_y,lithology,cluster_id\n0,50,a,0\n50,100,b,1\n")
    record={"id":"test-case","image":"image.png","annotation":"boxes.json","truth":"truth.csv","truth_window":[0,100],"source":"human","split":"dev","group":"well-1","window":[0,100],"label_provenance":{"kind":"human"},"tags":["stress","scan"]}
    path=tmp_path/"manifest.json";path.write_text(json.dumps({"schema_version":1,"name":"test","records":[record]}))
    return path


@pytest.fixture
def plugins(tmp_path,monkeypatch):
    (tmp_path/"test_plugins.py").write_text('''
import os,time,random
def perfect(request,config):
    assert 'truth' not in request and 'layers' not in request['annotation']
    assert 'GITHUB_TOKEN' not in os.environ and 'EXPERIMENT_SECRET' not in os.environ
    return [dict(top_y=0,bottom_y=50,lithology='a',cluster_id=0),dict(top_y=50,bottom_y=100,lithology='b',cluster_id=1)]
def crash(request,config):
    raise RuntimeError('intentional failure')
def slow(request,config):
    time.sleep(20)
    return []
def invalid(request,config):
    return [dict(top_y=0,bottom_y=1000,lithology='a')]
def seeded(request,config):
    return [dict(top_y=0,bottom_y=100,lithology=str(random.random()),cluster_id=0)]
def memory(request,config):
    allocation=bytearray(120*1024*1024)
    time.sleep(3)
    return []
def depths(request,config):
    rows=perfect(request,config)
    if not config.get('omit_depths'):
        for row in rows:
            row.update(top_depth=row['top_y']+config.get('offset',0),bottom_depth=row['bottom_y']+config.get('offset',0))
    return rows
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN","must-not-reach-worker")
    monkeypatch.setenv("EXPERIMENT_SECRET","also-must-not-reach-worker")


def test_success_artifacts_hashes_truth_isolation_and_immutable_runs(manifest,plugins,tmp_path):
    a,summary=evaluate(manifest,model="test_plugins:perfect",output=tmp_path/"runs",timeout=10)
    assert summary["overall"]["score"]==1
    assert summary["by"]["tags"]["scan"]["cases"]==1
    assert summary["quality_gates"]["passed"]
    assert (a/"summary.md").is_file() and len(list((a/"previews").glob("*.png")))==1
    assert "base_depth" in next((a/"submission").glob("*.csv")).read_text()
    prov=json.loads((a/"provenance.json").read_text())
    assert any(v["role"]=="model_source" for v in prov["files"].values())
    assert all(len(v["sha256"])==64 for v in prov["files"].values())
    b,_=evaluate(manifest,model="test_plugins:perfect",output=tmp_path/"runs",timeout=10,previews=False)
    assert a!=b
    assert compare_runs(a,b)["overall_deltas"]["score"]==0
    _,cases=load_run(a)
    assert cases[0]["tags"]==["stress","scan"]
    assert cases[0]["metrics"]["boundary_tolerance_pixels"]==4


def test_crop_training_group_alias_cannot_bypass_interval_overlap(manifest,tmp_path):
    data=json.loads(manifest.read_text())
    data['records'][0]['group']='challenge:60811405120000'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='training groups overlap'):
        evaluate(manifest,config={'training_groups':['api:6081140512']},output=tmp_path/'runs')


def test_preview_full_window_overlay_and_bands_share_projection(tmp_path):
    image=Image.new("RGB",(20,6000),(120,80,50))
    from PIL import ImageDraw
    ImageDraw.Draw(image).rectangle((0,4000,19,5999),fill=(0,128,0))
    image.save(tmp_path/"long.png")
    annotation=tmp_path/"long.json"
    annotation.write_text(json.dumps({"width":20,"height":6000,"bounding_boxes":{"interpreted_lithology_column":[0,0,20,6000]}}))
    rows=[dict(top_y=0,bottom_y=3000,lithology="a",cluster_id=0,top_depth=100,bottom_depth=200),dict(top_y=3000,bottom_y=6000,lithology="b",cluster_id=1,top_depth=200,bottom_depth=300)]
    record={"id":"long-window","image":str(tmp_path/"long.png"),"annotation":str(annotation),"window":[0,6000]}
    target=tmp_path/"preview.png";_preview(record,rows,rows,target)
    with Image.open(target) as preview:
        mapping=json.loads(preview.info["vertical_mapping"])
        boundary=mapping["top"]+mapping["height"]//2
        assert json.loads(preview.info["evaluation_window"])==[0,6000]
        assert preview.getpixel((100,boundary))==(210,0,0)
        assert preview.getpixel((350,boundary))==(0,0,0)
        assert preview.getpixel((670,boundary))==(0,0,0)
        # The image includes the final 2000 rows, not the old first-4000 slice.
        assert preview.getpixel((305,mapping["top"]+int(.9*mapping["height"])))==(0,128,0)


@pytest.mark.parametrize("model,error",[("crash","intentional failure"),("invalid","outside window")])
def test_model_errors_are_saved_and_fail_gates(manifest,plugins,tmp_path,model,error):
    run,result=evaluate(manifest,model=f"test_plugins:{model}",output=tmp_path/"runs",timeout=10,previews=False)
    _,cases=load_run(run)
    assert result["overall"]["cases"]==1 and result["overall"]["errors"]==1
    assert result["overall"]["score"]==0
    assert error in cases[0]["error"]
    assert not result["quality_gates"]["passed"]


def test_hard_timeout_includes_import_and_returns_promptly(manifest,plugins,tmp_path):
    started=time.monotonic()
    _,result=evaluate(manifest,model="test_plugins:slow",output=tmp_path/"runs",timeout=.5,previews=False)
    assert time.monotonic()-started<6
    assert result["overall"]["errors"]==1
    assert result["overall"]["score"]==0


def test_seed_repeated_predictions_are_byte_identical(manifest,plugins,tmp_path):
    a,_=evaluate(manifest,model="test_plugins:seeded",output=tmp_path/"runs",config={"seed":17},previews=False)
    b,_=evaluate(manifest,model="test_plugins:seeded",output=tmp_path/"runs",config={"seed":17},previews=False)
    assert next((a/"predictions").glob("*.csv")).read_bytes()==next((b/"predictions").glob("*.csv")).read_bytes()


def test_negative_seed_is_supported_consistently(manifest,plugins,tmp_path):
    _,summary=evaluate(manifest,model="test_plugins:seeded",output=tmp_path/"runs",config={"seed":-17},previews=False)
    assert summary["overall"]["errors"]==0


def test_sampled_memory_limit_is_saved_as_case_failure(manifest,plugins,tmp_path):
    run,summary=evaluate(manifest,model="test_plugins:memory",output=tmp_path/"runs",timeout=10,max_worker_rss_mb=64,previews=False)
    _,cases=load_run(run)
    assert "sampled worker RSS" in cases[0]["error"]
    assert summary["overall"]["errors"]==1 and summary["overall"]["score"]==0
    assert not summary["quality_gates"]["passed"]


def test_model_import_is_inside_hard_timeout(manifest,tmp_path,monkeypatch):
    (tmp_path/"import_slow.py").write_text("import time\ntime.sleep(20)\ndef predict(request,config): return []\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    started=time.monotonic()
    _,summary=evaluate(manifest,model="import_slow:predict",output=tmp_path/"runs",timeout=.5,previews=False)
    assert time.monotonic()-started<6
    assert summary["overall"]["errors"]==1


def test_full_page_truth_subwindow(manifest,tmp_path):
    data=json.loads(manifest.read_text());data["records"][0]["window"]=[25,75];manifest.write_text(json.dumps(data))
    _,result=evaluate(manifest,model="builtin:empty",output=tmp_path/"runs",previews=False)
    assert result["overall"]["row_acc"]==0
    assert not result["quality_gates"]["passed"]
    assert "empty predictions" in result["quality_gates"]["failures"][0]


def test_unlabelled_never_gets_accuracy(manifest,tmp_path):
    data=json.loads(manifest.read_text());r=data["records"][0];r["truth"]=None;r["source"]="challenge_unlabelled";manifest.write_text(json.dumps(data))
    _,result=evaluate(manifest,model="builtin:empty",output=tmp_path/"runs",previews=False,min_score=.1)
    assert result["overall"]["score"] is None
    assert not result["quality_gates"]["passed"]
    assert result["overall"]["empty_predictions"]==1


def test_comparisons_reject_modes_and_tampering(manifest,tmp_path):
    a,_=evaluate(manifest,model="builtin:empty",output=tmp_path/"runs",previews=False)
    b,_=evaluate(manifest,model="builtin:empty",output=tmp_path/"runs",mode="supplied_legend",previews=False)
    with pytest.raises(ValueError,match="mode differs"): compare_runs(a,b)
    (a/"per_case.jsonl").write_text("")
    with pytest.raises(ValueError,match="case count"): load_run(a)


def test_model_training_group_overlap_rejected_before_run(manifest,tmp_path):
    with pytest.raises(ValueError,match="training groups overlap"):
        evaluate(manifest,output=tmp_path/"runs",config={"training_groups":["well-1"]})
    assert not (tmp_path/"runs").exists()


@pytest.mark.parametrize("training_group,record_group",[("api:60811405120000","api:6081140512"),("challenge:MC765_P011_ST00BP00","well:mc765_p011"),("challenge:0000000009_d-582_0000020435","well:d582")])
def test_training_aliases_are_normalized_in_both_directions(manifest,tmp_path,training_group,record_group):
    data=json.loads(manifest.read_text());data["records"][0]["group"]=record_group;manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="training groups overlap"):
        evaluate(manifest,output=tmp_path/"runs",config={"training_groups":[training_group]},previews=False)


def test_selection_unknown_ids_tags_and_finite_config_fail_before_run(manifest,tmp_path):
    for kwargs,match in (({"case_ids":["absent"]},"unknown case IDs"),({"tags":["absent"]},"unknown tags"),({"case_ids":"test-case"},"collection of strings"),({"config":{"threshold":float("nan")}},"Out of range float")):
        with pytest.raises(ValueError,match=match): evaluate(manifest,output=tmp_path/"runs",previews=False,**kwargs)
    assert not (tmp_path/"runs").exists()


def test_selection_metadata_and_separate_preview_timings(manifest,plugins,tmp_path):
    run,summary=evaluate(manifest,model="test_plugins:perfect",output=tmp_path/"runs",case_ids=["test-case"],tags=["scan"])
    assert summary["selection"]=={"case_ids":["test-case"],"tags":["scan"]}
    assert summary["overall"]["preview_seconds"]>0
    assert summary["timings"]["wall_seconds"]>=summary["timings"]["worker_seconds"]+summary["timings"]["preview_seconds"]
    assert load_run(run)[0]["overall"]["latency_p95_seconds"]>0


@pytest.mark.parametrize("datum,unit,expected",[("MD","ft",True),("unknown","ft",False),("MD","m",False),("unspecified","ft",False)])
def test_depth_units_and_known_datum_required(manifest,plugins,tmp_path,datum,unit,expected):
    data=json.loads(manifest.read_text());data["records"][0].update(depth_unit=unit,depth_datum=datum);manifest.write_text(json.dumps(data))
    (manifest.parent/"truth.csv").write_text("top_y,bottom_y,lithology,cluster_id,top_depth,bottom_depth\n0,50,a,0,0,50\n50,100,b,1,50,100\n")
    _,summary=evaluate(manifest,model="test_plugins:depths",config={"depth_unit":"ft","depth_datum":datum},output=tmp_path/"runs",previews=False)
    assert summary["overall"]["depth_mae_ft"]==(0 if expected else None)
    assert summary["overall"]["depth_coverage"]==(1 if expected else None)


def test_compare_does_not_claim_depth_improvement_by_omitting_depths(manifest,plugins,tmp_path):
    data=json.loads(manifest.read_text());data["records"][0].update(depth_unit="ft",depth_datum="MD");manifest.write_text(json.dumps(data))
    (manifest.parent/"truth.csv").write_text("top_y,bottom_y,lithology,cluster_id,top_depth,bottom_depth\n0,50,a,0,0,50\n50,100,b,1,50,100\n")
    config={"depth_unit":"ft","depth_datum":"MD","offset":100}
    a,left=evaluate(manifest,model="test_plugins:depths",config=config,output=tmp_path/"runs",previews=False)
    b,right=evaluate(manifest,model="test_plugins:depths",config={**config,"omit_depths":True},output=tmp_path/"runs",previews=False)
    assert left["overall"]["depth_mae_ft"]==100 and right["overall"]["depth_coverage"]==0
    assert compare_runs(a,b)["overall_deltas"]["depth_mae_ft"] is None


def test_provenance_only_walks_declared_asset_paths(manifest,plugins,tmp_path,monkeypatch):
    from viridian402.data import load_manifest
    from viridian402.evaluate import provenance
    monkeypatch.chdir(tmp_path)
    (tmp_path/"unrelated.bin").write_bytes(b"unrelated")
    (tmp_path/"checkpoint.bin").write_bytes(b"model checkpoint")
    details=provenance(manifest,load_manifest(manifest),"test_plugins:perfect",{"depth_datum":".","checkpoint":str(tmp_path/"checkpoint.bin"),"cache_dir":str(tmp_path/"new-cache")})
    assert str(tmp_path/"unrelated.bin") not in details["files"]
    assert details["files"][str(tmp_path/"checkpoint.bin")]["role"]=="model_config_asset"


def test_shared_asset_collector_nested_paths_dedup_and_budgets(tmp_path):
    from viridian402.evaluate import discover_config_assets
    a=tmp_path/"a.bin";a.write_bytes(b"abc")
    b=tmp_path/"b.bin";b.write_bytes(b"def")
    config={"checkpoint":{"path":str(a)},"provenance_assets":[str(a),str(b)]}
    assert discover_config_assets(config)==sorted([a,b])
    with pytest.raises(ValueError,match="exceed"): discover_config_assets(config,max_files=1)
    with pytest.raises(ValueError,match="exceed"): discover_config_assets(config,max_bytes=5)
    with pytest.raises(ValueError,match="does not exist"): discover_config_assets({"checkpoint_path":str(tmp_path/"absent")})


def test_windows_bom_manifest_annotation_and_saved_summary(manifest,plugins,tmp_path):
    for path in (manifest,manifest.parent/"boxes.json"):
        path.write_text(path.read_text(encoding="utf-8"),encoding="utf-8-sig")
    run,summary=evaluate(manifest,model="test_plugins:perfect",output=tmp_path/"runs",previews=False)
    assert summary["overall"]["score"]==1 and summary["metric_version"]=="interval-v2"
    path=run/"metrics.json";path.write_text(path.read_text(encoding="utf-8"),encoding="utf-8-sig")
    assert load_run(run)[0]["overall"]["score"]==1


@pytest.mark.parametrize("timeout",[0,-1,float("inf"),float("nan")])
def test_invalid_timeout_rejected(manifest,tmp_path,timeout):
    with pytest.raises(ValueError,match="finite and positive"): evaluate(manifest,output=tmp_path/"runs",timeout=timeout)


def test_cli_exit_codes_and_empty_quality_gate(manifest,tmp_path,capsys):
    assert main(["validate","--manifest",str(manifest)])==0
    assert main(["evaluate","--manifest",str(manifest),"--model","builtin:empty","--output",str(tmp_path/"runs"),"--min-score",".1","--no-previews"])==1
    assert main(["evaluate","--manifest",str(manifest),"--timeout","nan"])==2
