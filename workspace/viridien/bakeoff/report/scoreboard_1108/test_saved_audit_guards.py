"""The CSV replay must reject stale provenance even when every score is correct."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('saved_audit_under_test',Path(__file__).with_name('audit_saved_predictions.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixture_run(folder):
    source=folder/'source.bin';source.write_bytes(b'fixture pixels')
    ann=folder/'annotation.json';ann.write_text('{}',encoding='utf-8')
    truth=folder/'truth.csv';pred=folder/'pred.csv'
    rows=[dict(top_y=0,bottom_y=10,lithology='Shale',cluster_id=1)]
    audit.io.save_rows(truth,rows);audit.io.save_rows(pred,rows)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    case=dict(case_id='fixture',log='fixture',suite='fixture',window=[0,10],
              image=str(source),annotation=str(ann),truth_csv=str(truth),
              image_sha256=sha(source),annotation_sha256=sha(ann),truth_sha256=sha(truth),
              truth_sources=[dict(path=str(truth),sha256=sha(truth))])
    candidates=['s19_combine','s20_v4','s21_v4dino','s23_v4dino_identity']
    manifest=dict(cases=[case],candidates=candidates,source_hashes={},
                  freeze=dict(dino=dict(weights_sha256='weights',config_sha256='config')))
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()
    metrics=audit.score.score(rows,rows,(0,10))
    records=[dict(case,strategy=s,prediction_csv=str(pred),manifest_digest=digest,**metrics) for s in candidates]
    policy=audit.grading_policy
    policy_path=Path(policy.__file__).resolve()
    scorer_path=Path(audit.score.__file__).resolve()
    metadata=dict(id=policy.POLICY_ID,source_path=str(policy_path),source_sha256=sha(policy_path),
                  equivalent_labels=list(policy.EQUIVALENT_LABELS),canonical_label='Clay/Shale',
                  shared_scorer_path=str(scorer_path),shared_scorer_sha256=sha(scorer_path),
                  owner_configured=True,official_challenge_rule_verified=False)
    for rec in records:
        rec['accepted_metrics']=policy.score_equivalent(rows,rows,(0,10))
    summaries=[dict(suite='fixture',strategy=s,family='all',metrics=metrics) for s in candidates]
    data=dict(complete=True,manifest_digest=digest,records=records,summaries=[],
              grading_policy=metadata,accepted_summaries=summaries)
    return manifest,data


def run_audit(folder,monkeypatch,manifest,data):
    monkeypatch.setattr(audit,'R',folder)
    (folder/'manifest_1108.json').write_text(json.dumps(manifest),encoding='utf-8')
    (folder/'scores.json').write_text(json.dumps(data),encoding='utf-8')
    failed=audit.main()
    result=json.loads((folder/'audit_saved_predictions.json').read_text(encoding='utf-8'))
    return failed,result


def test_valid_scores_and_provenance_are_accepted(tmp_path,monkeypatch):
    manifest,data=fixture_run(tmp_path)
    failed,result=run_audit(tmp_path,monkeypatch,manifest,data)
    assert not failed and result['errors']==[] and result['rescored']==4
    pred=Path(data['records'][0]['prediction_csv']).resolve()
    assert result['prediction_csv_sha256']=={str(pred):hashlib.sha256(pred.read_bytes()).hexdigest()}


@pytest.mark.parametrize('scope',['record_digest','primary_model','pixel_model','pixel_missing_model',
                                'accepted_score','policy_source','accepted_pixel_score'])
def test_correct_scores_cannot_hide_wrong_provenance(tmp_path,monkeypatch,scope):
    manifest,data=fixture_run(tmp_path)
    rec=data['records'][-1]
    if scope=='record_digest':
        rec['manifest_digest']='stale'
    elif scope=='accepted_score':
        rec['accepted_metrics']['score']=0
    elif scope=='policy_source':
        data['grading_policy']['source_sha256']='stale'
    elif scope=='accepted_pixel_score':
        rec['pixel_only']=dict(prediction_csv=rec['prediction_csv'],
                               accepted_metrics=dict(rec['accepted_metrics'],score=0),
                               **{f:rec[f] for f in audit.FIELDS})
    else:
        diag=dict(dino_chunk_vectors=1,model=dict(weights_sha256='WRONG',config_sha256='config'))
        if scope=='primary_model':rec['strategy_diagnostics']=diag
        else:
            if scope=='pixel_missing_model':diag['model']=None
            rec['pixel_only']=dict(prediction_csv=rec['prediction_csv'],strategy_diagnostics=diag,
                                   accepted_metrics=dict(rec['accepted_metrics']),
                                   **{f:rec[f] for f in audit.FIELDS})
    failed,result=run_audit(tmp_path,monkeypatch,manifest,data)
    assert failed and result['errors']
