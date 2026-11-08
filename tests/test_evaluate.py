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


@pytest.mark.parametrize("timeout",[0,-1,float("inf"),float("nan")])
def test_invalid_timeout_rejected(manifest,tmp_path,timeout):
    with pytest.raises(ValueError,match="finite and positive"): evaluate(manifest,output=tmp_path/"runs",timeout=timeout)


def test_cli_exit_codes_and_empty_quality_gate(manifest,tmp_path,capsys):
    assert main(["validate","--manifest",str(manifest)])==0
    assert main(["evaluate","--manifest",str(manifest),"--model","builtin:empty","--output",str(tmp_path/"runs"),"--min-score",".1","--no-previews"])==1
    assert main(["evaluate","--manifest",str(manifest),"--timeout","nan"])==2
