import copy
import pytest
from strategies import grading_availability_v2 as a

def structure(**extra):
    return dict(missing_truth=False,filled_rows=10,intervals=1,finite_depth_rows=0,finite_depth_boundaries=0,**extra)
def raw():return dict(score=1.,row_acc=1.,boundary_f1=1.,cluster_ari=1.,depth_mae_ft=None)
def view(s=None,**extra):return a.metric_view(raw(),s or structure(),series='reference_diagnostic',truth_status='QA-human',**extra)

def test_empty_truth_is_unavailable_not_perfect():
    s=a.truth_structure([], (0,10));r=view(s)
    assert all(not c['available'] and c['value'] is None for c in r.values())
    assert 'zero-supervision' in r['row_acc']['reasons']

def test_vocab_block_keeps_contact_diagnostic_and_raw_intact():
    values=raw();before=copy.deepcopy(values)
    r=a.metric_view(values,structure(),series='reference_diagnostic',truth_status=a.SILVER_STATUS,legend_invalid=True)
    assert all(r[k]['value'] is None for k in a.LITHOLOGY)
    assert r['boundary_f1']['value']==1.
    assert values==before

def test_missing_and_depth_are_distinct_unavailable_reasons():
    missing=view(a.truth_structure([], (0,10),missing=True))
    assert 'missing-truth' in missing['row_acc']['reasons']
    assert 'no-depth-truth' in view()['depth_mae_ft']['reasons']
    finite=a.truth_structure([dict(top_y=0,bottom_y=5,lithology='x',top_depth=100,bottom_depth=105)],(0,10))
    assert finite['finite_depth_boundaries']==1
    depth=view(finite)['depth_mae_ft']
    assert depth['eligible'] and not depth['available'] and depth['value'] is None
    assert depth['reasons']==['prediction-depth-unavailable']

def test_failed_prediction_on_valid_supervision_stays_zero():
    r=view(prediction_failed=True)
    assert all(r[k]['value']==0 and r[k]['available'] and r[k]['eligible'] for k in ('score','row_acc','boundary_f1','cluster_ari'))

def row(log,case,s,values,**extra):
    return dict(log=log,case_id=case,strategy='x',suite='human_qa',family='f',structural_supervision=s,legend_vocabulary_invalid=False,prediction_failed={'primary':extra.pop('failed',False)},views={'primary':{'reference_diagnostic':{'strict':values}}},**extra)

def test_denominators_include_failure_and_exclude_zero_supervision():
    s=structure();empty=a.truth_structure([], (0,10))
    rows=[row('a','1',s,view(prediction_failed=True),failed=True),row('b','2',empty,view(empty)),row('c','3',s,view())]
    r=a.denominator(rows,'score','reference_diagnostic','strict','primary');d=r['denominator']
    assert r['mean']==.5
    assert (d['total_windows'],d['available_windows'],d['excluded_windows'],d['eligible_windows'])==(3,2,1,2)
    assert (d['total_logs'],d['available_logs'],d['excluded_logs'],d['eligible_logs'])==(3,2,1,2)
    assert d['zero_supervision_windows']==1 and d['supervised_failure_windows']==1
    depth=a.denominator(rows,'depth_mae_ft','reference_diagnostic','strict','primary')
    assert depth['mean'] is None and depth['denominator']['no_depth_truth_windows']==3

def test_silver_not_qualified_for_lithology_but_contacts_stay_reference():
    r=a.metric_view(raw(),structure(),series='qualified_development',truth_status=a.SILVER_STATUS)
    assert all(r[k]['value'] is None for k in a.LITHOLOGY)
    assert r['boundary_f1']['available']
    q=a.metric_view(raw(),structure(),series='qualified_development',truth_status='human-unqa')
    assert all(c['value'] is None for c in q.values())

def test_missing_truth_is_a_separate_denominator_exclusion():
    present=structure();missing=a.truth_structure([], (0,10),missing=True)
    rows=[row('present','1',present,view(present)),row('missing','2',missing,view(missing))]
    d=a.denominator(rows,'row_acc','reference_diagnostic','strict','primary')['denominator']
    assert (d['total_windows'],d['available_windows'],d['eligible_windows'],d['excluded_windows'])==(2,1,1,1)
    assert d['missing_truth_windows']==1 and d['zero_supervision_windows']==0
    assert d['reason_windows']['missing-truth']==1 and d['reason_logs']['missing-truth']==1

def test_approved_equivalence_is_not_widened():
    assert set(a.grading_policy.EQUIVALENT_LABELS)=={'Clay','Shale','Clay/Shale','Clay-Shale'}
    for name in ('ClayShale','Claystone','Clay-S','Limestone-L'):
        assert a.grading_policy.canonical_grading_label(name)==name

def test_unknown_series_or_truth_status_cannot_bypass_masks():
    with pytest.raises(ValueError,match='Unknown availability series'):
        a.metric_view(raw(),structure(),series='qualified_developmnt',truth_status=a.SILVER_STATUS)
    with pytest.raises(ValueError,match='Unknown truth status'):
        a.metric_view(raw(),structure(),series='qualified_development',truth_status='QA-humn')

def test_depth_only_on_blank_or_nonpositive_rows_is_not_supervision():
    rows=[dict(top_y=0,bottom_y=2,lithology='x'),dict(top_y=2,bottom_y=5,lithology='',top_depth=12,bottom_depth=15),dict(top_y=7,bottom_y=7,lithology='x',top_depth=17,bottom_depth=17)]
    s=a.truth_structure(rows,(0,10))
    assert s['filled_rows']==2 and s['finite_depth_rows']==0 and s['finite_depth_boundaries']==0
    assert view(s)['depth_mae_ft']['value'] is None

def test_overall_uses_image_identity_and_truth_priority_across_aliases():
    rows=[dict(strategy='x',log='eval-alias',image_sha256='same',suite='viridien_eval',case_id='a'),dict(strategy='x',log='qa-alias',image_sha256='same',suite='human_qa',case_id='b'),dict(strategy='x',log='other',image_sha256='other',suite='ext_dev',case_id='c')]
    selected=a.select_overall(rows)
    assert {r['case_id'] for r in selected}=={'b','c'}
    assert rows[0]['log']=='eval-alias'
    same_priority=a.select_overall([dict(strategy='x',log=alias,image_sha256='same',suite='ext_dev',case_id=alias) for alias in ('aliasB','aliasA')])
    assert {r['log'] for r in same_priority}=={'aliasA'}
    assert len({r['overall_image_sha256'] for r in same_priority})==1

def test_pairing_resamples_only_exact_common_available_cases(monkeypatch):
    captured=[]
    def fake_ci(values,draws=10000):captured.append(values);return dict(n_logs=len(values))
    monkeypatch.setattr(a,'ci',fake_ci)
    records=[]
    for strategy in ('x','y'):
        for case,status in [('valid','QA-human'),('empty','QA-human')]:
            s=structure() if case=='valid' else a.truth_structure([], (0,10));v=a.metric_view(raw(),s,series='reference_diagnostic',truth_status=status)
            views={series:{policy:v for policy in a.POLICIES} for series in a.SERIES}
            r=row(case,case,s,v,window=[0,10],truth_sha256='t',image_sha256=case,annotation_sha256='a',truth_status=status)
            r['strategy']=strategy;r['views']['primary']=views;records.append(r)
    _,pairs=a.aggregate(records,['x','y'])
    for pair in pairs:
        assert pair['metrics']['score']['paired_case_ids']==['valid']
        assert pair['metrics']['score']['paired_logs']==['valid']
