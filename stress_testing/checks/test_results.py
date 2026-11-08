"""Result integrity/comparison stress using generated tiny saved-run artifacts."""
from copy import deepcopy
import csv
import hashlib
import json
import random

import pytest

from viridian402.metrics import METRIC_VERSION,aggregate,grouped_aggregates,quality_gate_failures,score
from viridian402.report import compare_runs,list_runs,load_run


def _write(path,value):
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=True)+"\n",encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_run(path,seed,*,quality=None,depths=False,omit_depths=False):
    """Pure file fixture: no imports of models, private datasets or subprocesses."""
    path.mkdir(parents=True,exist_ok=True)
    (path/"predictions").mkdir(exist_ok=True)
    rng=random.Random(seed)
    cases=[]
    for index in range(4):
        split=rng.randint(15,45)
        truth=[{"top_y":0,"bottom_y":split,"lithology":"Shale","cluster_id":0},{"top_y":split,"bottom_y":64,"lithology":"Sandstone","cluster_id":1}]
        pred=deepcopy(truth)
        correct=rng.choice([True,False]) if quality is None else quality
        if not correct:pred[0]["lithology"]="Unknown"
        if depths:
            for row in truth:row.update(top_depth=row["top_y"],bottom_depth=row["bottom_y"])
            if not omit_depths:
                for row in pred:row.update(top_depth=row["top_y"]+5,bottom_depth=row["bottom_y"]+5)
        metric=score(pred,truth,[0,64],depth_comparable=depths)
        metric["missing_depth_rate"]=1.0 if not depths or omit_depths else 0.0
        relative=f"predictions/case-{index}.csv"
        with (path/relative).open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=["top_y","bottom_y","lithology","cluster_id","top_depth","bottom_depth"])
            writer.writeheader();writer.writerows(pred)
        cases.append({"id":f"case-{index}","source":"human","group":f"well:{index//2}","split":"dev","tags":["generated","even" if index%2==0 else "odd"],"has_truth":True,"status":"ok","error":None,"seconds":(index+1)/100,"peak_rss_mb":20+index,"preview_seconds":0,"metrics":metric,"prediction":relative,"prediction_sha256":_sha(path/relative),"empty_prediction":False,"preview":None})
    case_path=path/"per_case.jsonl"
    case_path.write_text("".join(json.dumps(case,sort_keys=True)+"\n" for case in cases),encoding="utf-8")
    overall=aggregate(cases)
    failures=quality_gate_failures(cases,overall)
    summary={"schema_version":1,"run_id":path.name,"metric_version":METRIC_VERSION,"dataset_sha256":"d"*64,"scorer_sha256":"a"*64,"per_case_sha256":_sha(case_path),"model":"fixture:generated","config":{},"mode":"full_image","split":"dev","overall":overall,"by":grouped_aggregates(cases),"quality_gates":{"passed":not failures,"failures":failures,"min_score":None,"min_completion":1}}
    provenance={"dataset_sha256":summary["dataset_sha256"],"model":summary["model"],"config":summary["config"],"files":{}}
    _write(path/"provenance.json",provenance)
    summary["provenance_sha256"]=_sha(path/"provenance.json")
    _write(path/"metrics.json",summary)
    assert load_run(path)[0]["overall"]["cases"]==4
    return summary,cases


def test_seeded_summary_tampering_and_required_deletions(tmp_path,stress_seed,stress_iterations):
    rng=random.Random(stress_seed ^ 0xB001)
    run=tmp_path/"run";original,_=_make_run(run,stress_seed)
    attacks=["score","counts","runtime","gate","delete-score","delete-fingerprint","delete-group","group-score","model-provenance"]
    for iteration in range(stress_iterations):
        summary=deepcopy(original);attack=attacks[iteration%len(attacks)]
        if attack=="score":summary["overall"]["score"]=2
        elif attack=="counts":summary["overall"]["completed"]+=rng.randint(1,10)
        elif attack=="runtime":summary["overall"]["seconds"]+=rng.random()+1
        elif attack=="gate":summary["quality_gates"].update(passed=False,failures=["invented"])
        elif attack=="delete-score":summary["overall"].pop("score")
        elif attack=="delete-fingerprint":summary.pop(rng.choice(["scorer_sha256","dataset_sha256","per_case_sha256"]))
        elif attack=="delete-group":summary["by"]["group"].pop(rng.choice(list(summary["by"]["group"])))
        elif attack=="group-score":summary["by"]["source"]["human"]["score"]=2
        else:summary["model"]="fixture:forged"
        _write(run/"metrics.json",summary)
        with pytest.raises(ValueError):load_run(run)
    _write(run/"metrics.json",original)
    assert load_run(run)[0]["overall"]==original["overall"]


def test_seeded_case_mutation_rejected_after_checksum_rewritten(tmp_path,stress_seed,stress_iterations):
    rng=random.Random(stress_seed ^ 0xB002)
    run=tmp_path/"run";original,original_cases=_make_run(run,stress_seed)
    attacks=["nan","infinite-runtime","duplicate-id","bad-truth-flag","bad-status","negative-pixels","accuracy-counts","confusion-counts","depth-pairs","supervised-unlabelled"]
    for iteration in range(stress_iterations):
        cases=deepcopy(original_cases);case=cases[rng.randrange(len(cases))];attack=attacks[iteration%len(attacks)]
        if attack=="nan":case["metrics"]["row_acc"]=float("nan")
        elif attack=="infinite-runtime":case["seconds"]=float("inf")
        elif attack=="duplicate-id":cases[1]["id"]=cases[0]["id"]
        elif attack=="bad-truth-flag":case["has_truth"]="yes"
        elif attack=="bad-status":case["status"]="discarded"
        elif attack=="negative-pixels":case["metrics"]["labelled_pixels"]=-1
        elif attack=="accuracy-counts":case["metrics"]["correct_pixels"]=1000
        elif attack=="confusion-counts":case["metrics"]["confusion"]["fake\tfake"]=1000
        elif attack=="depth-pairs":case["metrics"]["depth_pairs"]=1000
        else:case["has_truth"]=False
        case_path=run/"per_case.jsonl";case_path.write_text("".join(json.dumps(c,allow_nan=True)+"\n" for c in cases),encoding="utf-8")
        summary=deepcopy(original);summary["per_case_sha256"]=_sha(case_path);_write(run/"metrics.json",summary)
        with pytest.raises(ValueError):load_run(run)


def test_seeded_comparison_rejects_incompatible_fingerprints_modes_and_splits(tmp_path,stress_seed,stress_iterations):
    rng=random.Random(stress_seed ^ 0xB003)
    left=tmp_path/"left";right=tmp_path/"right"
    _make_run(left,stress_seed);original,_=_make_run(right,stress_seed)
    for iteration in range(stress_iterations):
        summary=deepcopy(original)
        key=rng.choice(["dataset_sha256","scorer_sha256","metric_version","mode","split"])
        summary[key]="b"*64 if key.endswith("sha256") else "incompatible"
        _write(right/"metrics.json",summary)
        with pytest.raises(ValueError):compare_runs(left,right)


def test_paired_delta_direction_bootstrap_and_depth_coverage(tmp_path,stress_seed,stress_iterations):
    left=tmp_path/"left";right=tmp_path/"right"
    _make_run(left,stress_seed,quality=False,depths=True)
    _make_run(right,stress_seed,quality=True,depths=True)
    result=compare_runs(left,right)
    assert result["overall_deltas"]["row_acc"]>0
    assert result["overall_deltas"]["score"]>0
    assert result["overall_deltas"]["depth_mae_ft"]==0
    assert all(pair["deltas"]["row_acc"]>0 for pair in result["paired"])
    # One deterministic comparison resamples 2000 groups internally. Repeating
    # it thousands of times adds no independent cases, so this is intentionally fixed.
    assert compare_runs(left,right)==result
    assert result["paired_group_bootstrap_95ci"]["score"]["groups"]==2
    _make_run(right,stress_seed,quality=True,depths=True,omit_depths=True)
    missing=compare_runs(left,right)
    assert missing["overall_deltas"]["depth_mae_ft"] is None
    assert all(pair["deltas"]["depth_mae_ft"] is None for pair in missing["paired"])


def test_artifact_changes_and_nested_list_integrity(tmp_path,stress_seed,stress_iterations):
    run=tmp_path/"suite"/"model"/"run"
    _,cases=_make_run(run,stress_seed)
    listed=list_runs(tmp_path)
    assert len(listed)==1 and listed[0]["cases"]==4 and listed[0]["completed"]==4
    assert listed[0]["seconds"]>0 and listed[0]["peak_rss_mb"]>0
    (run/cases[0]["prediction"]).write_text("altered",encoding="utf-8")
    with pytest.raises(ValueError,match="checksum"):load_run(run)
    assert list_runs(tmp_path)==[]


def test_deleting_per_class_summary_is_detected(tmp_path,stress_seed):
    run=tmp_path/"run";summary,_=_make_run(run,stress_seed)
    assert summary["overall"]["per_class"]
    summary["overall"]["per_class"]={}
    _write(run/"metrics.json",summary)
    with pytest.raises(ValueError):load_run(run)


def test_prediction_artifact_path_cannot_escape_saved_run(tmp_path,stress_seed):
    run=tmp_path/"run";summary,cases=_make_run(run,stress_seed)
    outside=tmp_path/"outside.csv";outside.write_text("outside",encoding="utf-8")
    cases[0].update(prediction="../outside.csv",prediction_sha256=_sha(outside))
    case_path=run/"per_case.jsonl";case_path.write_text("".join(json.dumps(case)+"\n" for case in cases),encoding="utf-8")
    summary["per_case_sha256"]=_sha(case_path);_write(run/"metrics.json",summary)
    with pytest.raises(ValueError,match="escapes"):load_run(run)
