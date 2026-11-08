"""Bind the report's local-proxy policy to its source and accepted CSV replay."""
import copy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pytest
from rescore_policy import enrich,B
from report_policy_gate import require_policy
from test_rescore_policy import fixture_data


def valid(tmp_path):
    data=enrich(fixture_data(tmp_path));meta=data['grading_policy']
    audit=dict(grading_policy_id=meta['id'],grading_policy_source_sha256=meta['source_sha256'],rescored=1,accepted_rescored=1,pixel_diagnostics_rescored=1,accepted_pixel_diagnostics=1)
    return data,audit


def test_policy_gate_accepts_matching_source_metadata_and_both_replay_counts(tmp_path):
    data,audit=valid(tmp_path)
    assert require_policy(data,audit,B)['accepted_rescored']==1


@pytest.mark.parametrize('key,value',[('owner_configured',False),('official_challenge_rule_verified',True),('source_sha256','stale'),('shared_scorer_sha256','stale'),('equivalent_labels',['Clay'])])
def test_policy_gate_rejects_wrong_configuration_or_source(key,value,tmp_path):
    data,audit=valid(tmp_path);data['grading_policy'][key]=value
    with pytest.raises(ValueError,match='metadata mismatch'):require_policy(data,audit,B)


@pytest.mark.parametrize('key,value',[('grading_policy_source_sha256','stale'),('accepted_rescored',0),('accepted_pixel_diagnostics',0)])
def test_policy_gate_requires_both_bound_metric_series_replayed(key,value,tmp_path):
    data,audit=valid(tmp_path);audit[key]=value
    with pytest.raises(ValueError,match='binding is stale|coverage differs'):require_policy(data,audit,B)


def test_policy_gate_rejects_missing_pixel_series(tmp_path):
    data,audit=valid(tmp_path);data['records'][0]['pixel_only'].pop('accepted_metrics')
    with pytest.raises(ValueError,match='Missing accepted pixel'):require_policy(data,audit,B)
