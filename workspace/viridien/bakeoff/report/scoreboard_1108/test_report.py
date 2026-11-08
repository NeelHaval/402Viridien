"""Check aggregation rules; these tests do not read any dataset."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('scoreboard_report',Path(__file__).with_name('build_report.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_equal_log_mean_uses_products():
    rr=[dict(strategy='s',suite='dev',log='a',score=.2,row_acc=.4,boundary_f1=.5),dict(strategy='s',suite='dev',log='a',score=.4,row_acc=.8,boundary_f1=.5),dict(strategy='s',suite='dev',log='b',score=.9,row_acc=.9,boundary_f1=1)]
    out=m.summary(m.log_means(rr),['suite','strategy'])[0]
    assert abs(out['score']-.6)<1e-12
    assert abs(out['score']-out['row_acc']*out['boundary_f1'])>1e-3
    assert out['logs']==2 and out['windows']==3

def test_failed_case_stays_in_mean():
    rr=[dict(strategy='s',suite='dev',log='a',score=1,row_acc=1,boundary_f1=1),dict(strategy='s',suite='dev',log='b',score=0,row_acc=0,boundary_f1=0,error='failure')]
    out=m.summary(m.log_means(rr),['suite','strategy'])[0]
    assert out['score']==.5 and out['failure_count']==1

def test_paired_bootstrap_preserves_pairs_and_sign():
    rr=[]
    for log in ['one','two']:
        for s,value in [('a',.7),('b',.2)]:rr.append(dict(example_case(case_id=log),strategy=s,log=log,score=value,row_acc=value,boundary_f1=1))
    out=m.pairs(rr)[0]
    assert abs(out['mean_delta']-.5)<1e-12
    assert all(abs(x-.5)<1e-12 for x in out['ci95'])
    assert out['logs']==2 and out['wells']==1

def test_well_sensitivity_groups_continuation_pages():
    a=dict(log='ext_utah_well_004_p001',well='page_a')
    b=dict(log='ext_utah_well_004_p020',well='page_b')
    c=dict(log='ext_nl_well_004_p001',well='page_c')
    assert m.well_group(a)==m.well_group(b)
    assert m.well_group(a)!=m.well_group(c)
    d=dict(log='d582_log06',well='d582')
    assert m.well_group(d)=='d582'

import copy
import pytest
from report_inputs import STRATEGIES,validate_records,paired_case_rows,build_dev_map

def example_case(case_id='x',y0=0,y1=100):
    return dict(case_id=case_id,suite='dev',log='same_page',y0=y0,y1=y1,window=[y0,y1],family='f',well='w',dataset='d',image='i',annotation='a',column_box=[0,0,1,200],truth_csv='t',truth_sha256='truth',image_sha256='image',annotation_sha256='annotation',truth_sources=[dict(path='original',sha256='source')])

def example_run():
    c=example_case();return dict(cases=[c],candidates=list(STRATEGIES)),dict(complete=True,records=[dict(c,strategy=s,score=.5,row_acc=.5,boundary_f1=1,cluster_ari=.4) for s in STRATEGIES])

def test_complete_gate_accepts_only_exact_candidate_case_set():
    manifest,run=example_run()
    assert validate_records(run,manifest,expected_case_count=1)['records']==3
    with pytest.raises(ValueError,match='Incomplete'):validate_records(dict(run,records=run['records'][:-1]),manifest,expected_case_count=1)
    with pytest.raises(ValueError,match='Duplicate'):validate_records(dict(run,records=run['records']+[run['records'][0]]),manifest,expected_case_count=1)
    run['records'][0]['case_id']='not_declared'
    with pytest.raises(ValueError,match='Unmatched'):validate_records(run,manifest,expected_case_count=1)

@pytest.mark.parametrize('field,newvalue',[('truth_sha256','changed'),('image_sha256','changed'),('window',[100,200]),('well','different')])
def test_complete_gate_rejects_stale_or_mismatched_identity(field,newvalue):
    manifest,run=example_run();run['records'][0][field]=newvalue
    with pytest.raises(ValueError,match='identity'):validate_records(run,manifest,expected_case_count=1)

def test_complete_gate_requires_explicit_complete_flag():
    manifest,run=example_run();run['complete']=False
    with pytest.raises(ValueError,match='complete'):validate_records(run,manifest,expected_case_count=1)

def test_pairing_rejects_disjoint_windows_and_unmatched_cases():
    a=dict(example_case(),strategy='a');b=dict(example_case(y0=100,y1=200),strategy='b')
    with pytest.raises(ValueError,match='identity'):paired_case_rows([a,b],'a','b','dev')
    b['case_id']='different'
    with pytest.raises(ValueError,match='Unmatched'):paired_case_rows([a,b],'a','b','dev')

def test_pairing_rejects_duplicates():
    a=dict(example_case(),strategy='a');b=dict(example_case(),strategy='b')
    with pytest.raises(ValueError,match='Duplicate'):paired_case_rows([a,a,b],'a','b','dev')

def test_idaho_source_pages_group_together():
    assert m.well_group(dict(log='ext_idaho_barlow3-14_p001',well='page1'))==m.well_group(dict(log='ext_idaho_barlow3-14_p002',well='page2'))


def test_gate_accepts_an_explicit_fourth_candidate_and_requires_all_its_cases():
    manifest,run=example_run();manifest['candidates'].append('s23_v4dino_identity')
    with pytest.raises(ValueError,match='Incomplete'):validate_records(run,manifest,expected_case_count=1)
    run['records'].append(dict(run['records'][0],strategy='s23_v4dino_identity'))
    assert validate_records(run,manifest,expected_case_count=1)['strategies']==4

def test_gate_rejects_undeclared_or_missing_baseline_candidates():
    manifest,run=example_run();manifest.pop('candidates')
    with pytest.raises(ValueError,match='declare'):validate_records(run,manifest,expected_case_count=1)
    manifest['candidates']=['unapproved_a','unapproved_b','unapproved_c']
    with pytest.raises(ValueError,match='baseline'):validate_records(run,manifest,expected_case_count=1)


def test_gate_rejects_an_unapproved_fourth_candidate_name():
    manifest,run=example_run();manifest['candidates'].append('s23_v4dino_distinct')
    with pytest.raises(ValueError,match='baseline'):validate_records(run,manifest,expected_case_count=1)

def test_accepted_view_overrides_only_copied_metric_series():
    strict=dict(case_id='x',strategy='s21_v4dino',score=.1,row_acc=.2,boundary_f1=.5,cluster_ari=.3,depth_mae_ft=2,accepted_metrics=dict(score=.4,row_acc=.8,boundary_f1=.5,cluster_ari=.7,depth_mae_ft=2),pixel_only=dict(score=.1,accepted_metrics=dict(score=.4,row_acc=.8,boundary_f1=.5,cluster_ari=.7,depth_mae_ft=None)))
    view=m.accepted_records([strict])[0]
    assert view['score']==.4 and view['pixel_only']['score']==.4
    assert strict['score']==.1 and strict['pixel_only']['score']==.1
    assert view['_grading_id']=='clay_shale_equivalent_v1'


def test_accepted_view_requires_both_primary_and_pixel_metrics():
    with pytest.raises(ValueError,match='primary'):m.accepted_records([dict(score=.1)])
    accepted=dict(score=.4,row_acc=.8,boundary_f1=.5,cluster_ari=.7,depth_mae_ft=None)
    with pytest.raises(ValueError,match='pixel'):m.accepted_records([dict(accepted_metrics=accepted,pixel_only=dict(score=.1))])


def test_error_raster_uses_policy_without_changing_raw_interval_labels():
    import sys
    sys.path.insert(0,str(m.ROOT))
    pred=[dict(top_y=0,bottom_y=3,lithology='Clay / Shale',cluster_id=7)]
    truth=[dict(top_y=0,bottom_y=3,lithology='Shale',cluster_id=2)]
    assert list(m.graded_raster(pred,(0,3),'clay_shale_equivalent_v1'))==list(m.graded_raster(truth,(0,3),'clay_shale_equivalent_v1'))
    assert list(m.graded_raster(pred,(0,3),None))!=list(m.graded_raster(truth,(0,3),None))
    assert pred[0]['lithology']=='Clay / Shale' and pred[0]['cluster_id']==7
