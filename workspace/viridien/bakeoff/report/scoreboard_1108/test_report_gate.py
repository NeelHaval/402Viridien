"""Check the independent-audit binding without opening input data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pytest
from report_inputs import validate_audit,records_digest,digest

def test_audit_is_bound_to_current_manifest_and_records(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}',encoding='utf-8')
    records=[dict(case_id='case',strategy='s19_combine')]
    audit=dict(errors=[],records=1,expected=1,manifest_sha256=digest(manifest),records_sha256=records_digest(records))
    assert validate_audit(audit,records,manifest)['status']=='valid'
    with pytest.raises(ValueError,match='stale result'):validate_audit(audit,[dict(records[0],score=.5)],manifest)
    manifest.write_text('{"changed": true}',encoding='utf-8')
    with pytest.raises(ValueError,match='stale manifest'):validate_audit(audit,records,manifest)

def test_failed_or_partial_independent_audit_is_rejected(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}',encoding='utf-8');records=[]
    with pytest.raises(ValueError,match='did not pass'):validate_audit(dict(errors=['CSV mismatch']),records,manifest)
    with pytest.raises(ValueError,match='coverage'):validate_audit(dict(errors=[],records=3,expected=204),records,manifest)
