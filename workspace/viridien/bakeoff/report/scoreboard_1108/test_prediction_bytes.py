"""Reject saved prediction changes after an independent numerical replay."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pytest
from report_inputs import digest,validate_prediction_bytes,validate_records
from test_report import example_run


def test_changed_prediction_bytes_invalidate_a_successful_numeric_audit(tmp_path):
    primary=tmp_path/'primary.csv';pixel=tmp_path/'pixel.csv'
    primary.write_text('original primary',encoding='utf-8');pixel.write_text('original pixel',encoding='utf-8')
    records=[dict(prediction_csv=str(primary),pixel_only=dict(prediction_csv=str(pixel)))]
    audit=dict(prediction_csv_sha256={str(primary):digest(primary),str(pixel):digest(pixel)})
    assert validate_prediction_bytes(audit,records)['prediction_csv_count']==2
    pixel.write_text('changed after audit',encoding='utf-8')
    with pytest.raises(ValueError,match='bytes changed'):validate_prediction_bytes(audit,records)


def test_missing_or_extra_prediction_byte_pins_are_rejected(tmp_path):
    path=tmp_path/'prediction.csv';path.write_text('rows',encoding='utf-8');records=[dict(prediction_csv=str(path))]
    with pytest.raises(ValueError,match='path set'):validate_prediction_bytes({},records)
    with pytest.raises(ValueError,match='path set'):validate_prediction_bytes(dict(prediction_csv_sha256={str(path):digest(path),'extra':'hash'}),records)


def test_initial_operator_stop_cannot_pass_final_completion_gate():
    manifest,run=example_run();run['records'][0].update(operational_timeout=True,timeout_scope='Initial manual stop had no START receipt')
    with pytest.raises(ValueError,match='formal capped attempt'):validate_records(run,manifest,expected_case_count=1)
