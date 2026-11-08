"""Regression checks for Astra R1. No production data or model is opened."""
import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pytest
from report_inputs import digest,records_digest
from report_integrity import manifest_digest,require_run_provenance,require_integrity_audit


def example():
    manifest={'freeze':{'dino':{'weights_sha256':'weight_pin','config_sha256':'config_pin'}}}
    model=dict(manifest['freeze']['dino'])
    record=dict(case_id='case',strategy='s23_v4dino_identity',manifest_digest=manifest_digest(manifest),strategy_diagnostics=dict(dino_chunk_vectors=5,model=model),pixel_only=dict(strategy_diagnostics=dict(dino_chunk_vectors=3,model=dict(model))))
    raw=dict(records=[record],manifest_digest=manifest_digest(manifest))
    return manifest,raw


def test_current_primary_and_pixel_model_provenance_is_accepted():
    manifest,raw=example();proof=require_run_provenance(raw,manifest)
    assert proof['primary_embedding_records']==1 and proof['pixel_embedding_records']==1


@pytest.mark.parametrize('scope',['primary','pixel'])
def test_wrong_primary_or_pixel_model_hash_is_rejected(scope):
    manifest,raw=example();record=raw['records'][0]
    diag=record['strategy_diagnostics'] if scope=='primary' else record['pixel_only']['strategy_diagnostics']
    diag['model']['weights_sha256']='wrong'
    with pytest.raises(ValueError,match=scope+' model pin'):require_run_provenance(raw,manifest)


@pytest.mark.parametrize('scope',['primary','pixel'])
def test_embedding_chunks_require_model_evidence(scope):
    manifest,raw=example();record=raw['records'][0]
    diag=record['strategy_diagnostics'] if scope=='primary' else record['pixel_only']['strategy_diagnostics']
    diag['model']=None
    with pytest.raises(ValueError,match='lack model evidence'):require_run_provenance(raw,manifest)


def test_no_embedding_allows_no_model_evidence():
    manifest,raw=example()
    for diag in [raw['records'][0]['strategy_diagnostics'],raw['records'][0]['pixel_only']['strategy_diagnostics']]:diag.update(dino_chunk_vectors=0,model=None)
    assert require_run_provenance(raw,manifest)['primary_embedding_records']==0


@pytest.mark.parametrize('scope',['record','top'])
def test_stale_manifest_digest_is_rejected(scope):
    manifest,raw=example()
    if scope=='record':raw['records'][0]['manifest_digest']='stale'
    else:raw['manifest_digest']='stale'
    with pytest.raises(ValueError,match='manifest_digest is stale'):require_run_provenance(raw,manifest)


def test_integrity_audit_requires_bound_complete_exact_keys_and_proofs(tmp_path):
    manifest,raw=example();path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest),encoding='utf-8')
    audit=dict(errors=[],complete=True,records=1,expected=1,manifest_sha256=digest(path),records_sha256=records_digest(raw['records']),manifest_digest=manifest_digest(manifest),exact_key_proof=True,primary_model_evidence_checks=1,pixel_model_evidence_checks=1,input_hash_checks=1,source_hash_checks=1)
    assert require_integrity_audit(audit,raw,manifest,path)['status']=='valid'
    for key,value,pattern in [('complete',False,'incomplete'),('exact_key_proof',False,'exact-key'),('manifest_sha256','stale','stale manifest'),('records_sha256','stale','stale result')]:
        bad=dict(audit);bad[key]=value
        with pytest.raises(ValueError,match=pattern):require_integrity_audit(bad,raw,manifest,path)
    bad=dict(audit);bad.pop('pixel_model_evidence_checks')
    with pytest.raises(ValueError,match='missing proof'):require_integrity_audit(bad,raw,manifest,path)
