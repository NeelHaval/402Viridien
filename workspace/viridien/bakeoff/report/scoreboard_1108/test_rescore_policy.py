"""Verify metric-only enrichment with saved-CSV fixtures."""
import copy,importlib.util,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pytest
from rescore_policy import enrich,matched_quartets
from harness import io,score


def fixture_data(tmp_path):
    truth=[dict(top_y=0,bottom_y=50,lithology='Shale',cluster_id=0,top_depth=0,bottom_depth=50),dict(top_y=50,bottom_y=100,lithology='Sandstone',cluster_id=1,top_depth=50,bottom_depth=100)]
    pred=copy.deepcopy(truth);pred[0]['lithology']='Clay-Shale'
    gt=tmp_path/'truth.csv';pr=tmp_path/'pred.csv';io.save_rows(gt,truth);io.save_rows(pr,pred)
    strict=score.score(pred,truth,(0,100))
    record=dict(case_id='x',strategy='s21_v4dino',window=[0,100],truth_csv=str(gt),prediction_csv=str(pr),seconds=7,**strict)
    record['pixel_only']=dict(prediction_csv=str(pr),seconds=3,**strict)
    return dict(records=[record],complete=False)


def test_enrichment_preserves_strict_metrics_and_changes_both_copied_policy_series(tmp_path):
    original=fixture_data(tmp_path);saved=copy.deepcopy(original);out=enrich(original)
    assert original==saved
    r=out['records'][0];before=original['records'][0]
    for field in ('score','row_acc','boundary_f1','cluster_ari','depth_mae_ft','seconds'):assert r[field]==before[field]
    assert r['row_acc']==.5 and r['accepted_metrics']['row_acc']==1
    assert r['accepted_metrics']['score']==1 and r['pixel_only']['accepted_metrics']['score']==1
    assert r['accepted_metrics']['boundary_f1']==r['boundary_f1']
    assert r['accepted_metrics']['depth_mae_ft']==r['depth_mae_ft']
    assert out['grading_policy']['owner_configured'] is True
    assert out['grading_policy']['official_challenge_rule_verified'] is False


def test_primary_failure_stays_zero_and_pixel_diagnostic_is_scored(tmp_path):
    data=fixture_data(tmp_path);r=data['records'][0];empty=tmp_path/'empty.csv';io.save_rows(empty,[])
    r.update(prediction_csv=str(empty),error='depth failure',score=0,row_acc=0,boundary_f1=0,cluster_ari=0,depth_mae_ft=None)
    out=enrich(data)['records'][0]
    assert all(out['accepted_metrics'][k]==0 for k in ('score','row_acc','boundary_f1','cluster_ari'))
    assert out['accepted_metrics']['depth_mae_ft'] is None
    assert out['pixel_only']['accepted_metrics']['row_acc']==1


def test_grading_cannot_change_recorded_contacts_silently(tmp_path):
    data=fixture_data(tmp_path);data['records'][0]['boundary_f1']=.123
    with pytest.raises(ValueError,match='boundary_f1 changed'):enrich(data)


def test_progress_summaries_use_complete_matched_candidate_sets_only():
    manifest=dict(candidates=['a','b','c','d'],cases=[dict(case_id='one'),dict(case_id='two')])
    records=[dict(case_id='one',strategy=s) for s in manifest['candidates']]+[dict(case_id='two',strategy='a')]
    assert len(matched_quartets(records,manifest))==4
    assert all(r['case_id']=='one' for r in matched_quartets(records,manifest))


def test_progress_rejects_disjoint_windows_even_with_matching_case_ids():
    manifest=dict(candidates=['a','b','c','d'],cases=[dict(case_id='one',window=[0,100])])
    records=[dict(case_id='one',strategy=s,window=[0,100]) for s in manifest['candidates']]
    records[-1]['window']=[100,200]
    with pytest.raises(ValueError,match='preview case identity'):matched_quartets(records,manifest)

from rescore_policy import merge_preview_shard
from report_integrity import manifest_digest
from report_inputs import IDENTITY


def remote_fixture():
    case={k:'value_'+k for k in IDENTITY};case.update(case_id='remote',y0=0,y1=100,window=[0,100],column_box=[0,0,1,100],truth_sources=[])
    manifest=dict(cases=[case],candidates=['s20_v4'])
    row=dict(case,strategy='s20_v4',manifest_digest=manifest_digest(manifest),seconds=.7,runtime_scope='remote CPU scope',remote_source_sha256={'code':'pin'})
    return dict(records=[]),dict(records=[row],manifest_digest=manifest_digest(manifest)),manifest


def test_remote_preview_merge_retains_provenance_and_does_not_change_local_data():
    data,shard,manifest=remote_fixture();out=merge_preview_shard(data,shard,manifest)
    assert data['records']==[] and out['records'][0]==shard['records'][0]
    assert out['records'][0]['seconds']==.7 and out['preview_merge']['preview_only'] is True


@pytest.mark.parametrize('field,value',[('strategy','s21_v4dino'),('window',[100,200]),('truth_sha256','stale'),('manifest_digest','stale')])
def test_remote_preview_rejects_wrong_candidate_or_frozen_identity(field,value):
    data,shard,manifest=remote_fixture();shard['records'][0][field]=value
    with pytest.raises(ValueError,match='S20 only|identity mismatch|digest mismatch'):merge_preview_shard(data,shard,manifest)


def test_remote_preview_rejects_local_or_shard_duplicates():
    data,shard,manifest=remote_fixture();data['records']=copy.deepcopy(shard['records'])
    with pytest.raises(ValueError,match='Duplicate'):merge_preview_shard(data,shard,manifest)
    data['records']=[];shard['records'].append(copy.deepcopy(shard['records'][0]))
    with pytest.raises(ValueError,match='Duplicate'):merge_preview_shard(data,shard,manifest)


def test_remote_preview_rejects_a_stale_top_level_shard():
    data,shard,manifest=remote_fixture();shard['manifest_digest']='stale'
    with pytest.raises(ValueError,match='digest mismatch'):merge_preview_shard(data,shard,manifest)
