"""Saved results must agree with cases, artifacts, provenance and gates."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from viridian402.evaluate import evaluate
from viridian402.report import compare_runs,list_runs,load_run
from viridian402.synthetic import generate_demo


@pytest.fixture(scope="module")
def original_run(tmp_path_factory):
    root=tmp_path_factory.mktemp("report-source")
    manifest=generate_demo(root/"data",n=2,seed=89)
    run,_=evaluate(manifest,model="builtin:empty",output=root/"runs",previews=False)
    return run


@pytest.fixture
def run(original_run,tmp_path):
    return Path(shutil.copytree(original_run,tmp_path/"run"))


def edit_summary(run,edit):
    path=run/"metrics.json";summary=json.loads(path.read_text())
    edit(summary);path.write_text(json.dumps(summary))


@pytest.mark.parametrize("section,key,value",[("overall","score",.9),("overall","completed",0),("overall","seconds",999),("overall","row_acc_weighted",.8)])
def test_aggregate_tamper_detected(run,section,key,value):
    edit_summary(run,lambda summary:summary[section].update({key:value}))
    with pytest.raises(ValueError,match="recomputation"): load_run(run)


def test_group_aggregate_tamper_detected(run):
    edit_summary(run,lambda summary:summary["by"]["source"]["synthetic"].update(score=.7))
    with pytest.raises(ValueError,match="recomputation"): load_run(run)


def test_gate_and_model_metadata_tamper_detected(run):
    edit_summary(run,lambda summary:summary["quality_gates"].update(passed=True,failures=[]))
    with pytest.raises(ValueError,match="quality gates"): load_run(run)


def test_summary_model_must_match_provenance(run):
    edit_summary(run,lambda summary:summary.update(model="invented:perfect"))
    with pytest.raises(ValueError,match="differs from provenance"): load_run(run)


def test_artifact_tamper_detected(run):
    _,cases=load_run(run)
    (run/cases[0]["prediction"]).write_text("fabricated output")
    with pytest.raises(ValueError,match="prediction checksum"): load_run(run)


def test_malformed_nonfinite_case_rejected_even_when_checksum_recomputed(run):
    path=run/"per_case.jsonl"
    cases=[json.loads(line) for line in path.read_text().splitlines()]
    cases[0]["metrics"]["row_acc"]=float("nan")
    path.write_text("\n".join(json.dumps(case) for case in cases)+"\n")
    edit_summary(run,lambda summary:summary.update(per_case_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with pytest.raises(ValueError,match="metric row_acc is invalid"): load_run(run)


def test_nested_run_discovery_validates_and_exposes_runtime(original_run,tmp_path):
    nested=tmp_path/"runs"/"suite"/"model"/"run"
    shutil.copytree(original_run,nested)
    rows=list_runs(tmp_path/"runs")
    assert len(rows)==1 and rows[0]["cases"]==2 and rows[0]["completed"]==2
    assert rows[0]["seconds"]>0 and "peak_rss_mb" in rows[0]
    edit_summary(nested,lambda summary:summary["overall"].update(score=.8))
    assert list_runs(tmp_path/"runs")==[]


def test_comparison_rejects_aggregate_edit_before_emitting_deltas(run,original_run):
    edit_summary(run,lambda summary:summary["overall"].update(score=1))
    with pytest.raises(ValueError,match="recomputation"): compare_runs(original_run,run)


@pytest.mark.parametrize("key",["score","cases","row_acc_weighted"])
def test_deleted_aggregate_fields_cannot_hide_mismatch(run,key):
    edit_summary(run,lambda summary:summary["overall"].pop(key))
    with pytest.raises(ValueError,match="missing required aggregate"): load_run(run)


@pytest.mark.parametrize("key",["scorer_sha256","dataset_sha256","per_case_sha256"])
def test_missing_fingerprints_cannot_compare_as_equal(run,key):
    edit_summary(run,lambda summary:summary.pop(key))
    with pytest.raises(ValueError,match="fingerprint is missing"): compare_runs(run,run)


def test_deleting_group_entry_is_not_silent(run):
    edit_summary(run,lambda summary:summary["by"]["source"].pop("synthetic"))
    with pytest.raises(ValueError,match="group set differs"): load_run(run)
