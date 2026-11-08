"""Check reporting distinctions that a pooled timing mean can hide."""
import copy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from report_diagnostics import runtime_summary,native_legend_summary,native_legend_status


def test_local_and_remote_times_are_never_pooled_in_regime_rows():
    rows=[dict(suite='eval',strategy='s20_v4',log='a',seconds=10),dict(suite='eval',strategy='s20_v4',log='b',seconds=20),dict(suite='eval',strategy='s20_v4',log='c',seconds=.5,remote=True,runtime_scope='VM CPU crop'),dict(suite='eval',strategy='s20_v4',log='d',seconds=None,remote=True,runtime_scope='VM CPU crop')]
    before=copy.deepcopy(rows);out=runtime_summary(rows)
    assert len(out)==2 and rows==before
    local=next(r for r in out if r['regime'].startswith('local'))
    remote=next(r for r in out if r['regime'].startswith('remote'))
    assert local['mean_seconds']==15 and local['median_seconds']==15 and local['timed']==2
    assert remote['mean_seconds']==.5 and remote['timed']==1 and remote['missing']==1


def test_runtime_scope_can_identify_remote_without_a_boolean_flag():
    rows=[dict(suite='eval',strategy='s20_v4',log='x',seconds=None,runtime_scope='VM CPU inference on decoded lossless window')]
    r=runtime_summary(rows)[0]
    assert r['regime'].startswith('remote') and r['timed']==0 and r['mean_seconds'] is None and r['missing']==1


def test_missing_legend_counts_accept_both_diagnostic_names_and_pixel_fallback():
    missing=dict(native_cache_present=False,native_entries=0,legend_entries=0)
    rows=[dict(suite='ext_dev',strategy='s20_v4',log='a',strategy_diagnostics=missing),dict(suite='ext_dev',strategy='s20_v4',log='b',strategy_diagnostics={},diagnostics=missing),dict(suite='ext_dev',strategy='s20_v4',log='b',pixel_only=dict(diagnostics=missing)),dict(suite='ext_dev',strategy='s20_v4',log='c')]
    before=copy.deepcopy(rows);r=native_legend_summary(rows)[0]
    assert r['missing_native_legend']==3 and r['missing_native_logs']==2
    assert r['unknown']==1 and r['pixel_only_evidence_cases']==1 and rows==before


def test_s19_is_not_misreported_as_a_missing_native_cache():
    assert native_legend_status(dict(strategy='s19_combine',diagnostics=dict(native_entries=0)))[0]=='not_applicable'


def test_failure_explanation_reads_portable_missing_native_evidence():
    from test_report import m
    record=dict(strategy='s20_v4',row_acc=0,boundary_f1=0,diagnostics=dict(native_cache_present=False,native_entries=0))
    assert 'native legend has no entries' in m.failure_reason(record)
