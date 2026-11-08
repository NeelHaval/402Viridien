"""Check that missing truth cannot turn into a reported perfect or zero metric."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pytest
from report_contract import validate_metric,validate_denominators


def unavailable(reason='no-filled-truth'):
    return dict(value=None,available=False,eligible=False,reasons=[reason])


def test_empty_truth_is_unavailable_and_cannot_earn_perfect_accuracy():
    assert validate_metric(unavailable(),'row_acc',filled_rows=0)['value'] is None
    with pytest.raises(ValueError,match='Zero-filled'):validate_metric(dict(value=1,available=True,eligible=True,reasons=[]),'row_acc',filled_rows=0)
    with pytest.raises(ValueError,match='Zero-filled'):validate_metric(dict(value=1,available=True,eligible=True,reasons=[]),'score',filled_rows=0)


def test_missing_depth_is_null_with_reason_and_cannot_be_zero():
    m=unavailable('depth-truth-missing');assert validate_metric(m,'depth_mae_ft')['value'] is None
    m['value']=0
    with pytest.raises(ValueError,match='must be null'):validate_metric(m,'depth_mae_ft')


def test_ineligible_view_cannot_display_a_reference_only_value():
    reference=dict(value=.8,available=True,eligible=True,reasons=[])
    assert validate_metric(reference,'row_acc',filled_rows=100,series='reference_diagnostic')['value']==.8
    reference['eligible']=False
    with pytest.raises(ValueError,match='must be eligible'):validate_metric(reference,'row_acc',filled_rows=100,series='qualified_development')


def test_denominators_count_exclusions_and_no_observation_mean_is_null():
    m=dict(mean=.7,total_windows=13,available_windows=10,excluded_windows=3,eligible_windows=10,total_logs=13,available_logs=10,excluded_logs=3,eligible_logs=10)
    assert validate_denominators(m)['excluded_windows']==3
    m['excluded_windows']=2
    with pytest.raises(ValueError,match='Inconsistent'):validate_denominators(m)
    empty=dict(mean=1,total_windows=13,available_windows=0,excluded_windows=13,eligible_windows=0,total_logs=13,available_logs=0,excluded_logs=13,eligible_logs=0)
    with pytest.raises(ValueError,match='No available'):validate_denominators(empty)
    empty['mean']=None;assert validate_denominators(empty)['mean'] is None


def test_available_denominator_cannot_exceed_eligible_denominator():
    m=dict(mean=.7,total_windows=13,available_windows=10,excluded_windows=3,eligible_windows=9,total_logs=13,available_logs=10,excluded_logs=3,eligible_logs=10)
    with pytest.raises(ValueError,match='Inconsistent'):validate_denominators(m)


def test_empty_truth_cannot_provide_contact_or_depth_metric():
    m=dict(value=0,available=True,eligible=True,reasons=[])
    for metric in ('boundary_f1','depth_mae_ft'):
        with pytest.raises(ValueError,match='Zero-filled'):validate_metric(m,metric,filled_rows=0)
