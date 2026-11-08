import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).with_name('score_all.py');spec=importlib.util.spec_from_file_location('score_all',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def rec(case,log,strategy,suite,score,y0=0,y1=100,truth='t'):
    return dict(case_id=case,log=log,strategy=strategy,suite=suite,score=score,row_acc=score,boundary_f1=1.,cluster_ari=0.,family='x',well=log,y0=y0,y1=y1,truth_sha256=truth)

def test_equal_log_primary_not_window_mean():
    rows=[rec('1','a','s','ext_dev',1),rec('2','a','s','ext_dev',1),rec('3','b','s','ext_dev',0)]
    summary,_=m.summarize(rows);r=next(r for r in summary if r['suite']=='ext_dev' and r['family']=='all')
    assert r['metrics']['score']==.5
    assert r['window_metrics']['score']==pytest.approx(2/3)

def test_overall_truth_precedence_and_paired_difference():
    rows=[]
    for st,delta in [('a',0),('b',.1)]:
        rows.extend([rec('1','x',st,'viridien_eval',.1+delta),rec('2','x',st,'human_qa',.8+delta),rec('3','y',st,'ext_dev',.4+delta)])
    summary,pairs=m.summarize(rows)
    r=next(r for r in summary if r['suite']=='overall_unique_log' and r['family']=='all' and r['strategy']=='a')
    assert r['metrics']['score']==pytest.approx(.6)
    pair=next(r for r in pairs if r['suite']=='overall_unique_log')
    assert pair['metrics']['score']['log_ci']['mean']==pytest.approx(.1)
    assert pair['metrics']['score']['log_ci']['low']==pytest.approx(.1)

def test_pixel_clean_has_no_fake_depth():
    rows=m.pixel_clean([dict(top_y=0,bottom_y=30,lithology='A',top_depth=123,cluster_id=1)],(10,20))
    assert rows==[dict(top_y=10,bottom_y=20,lithology='A',cluster_id=1)]
    with pytest.raises(ValueError):m.pixel_clean([dict(top_y=10,bottom_y=20),dict(top_y=15,bottom_y=30)],(0,40))

def test_input_hash_guard_rejects_changed_bytes(tmp_path):
    p=tmp_path/'asset';p.write_text('original',encoding='utf-8');h=m.sha(p)
    case=dict(image=str(p),annotation=str(p),truth_csv=str(p),image_sha256=h,annotation_sha256=h,truth_sha256=h,truth_sources=[dict(path=str(p),sha256=h)])
    m.validate_inputs(case)
    p.write_text('changed',encoding='utf-8')
    with pytest.raises(RuntimeError,match='Input drift'):m.validate_inputs(case)

def test_resume_rejects_stale_identity_digest_and_duplicates():
    case=dict(case_id='x',log='log',suite='suite',window=[0,10],image='i',annotation='a',truth_csv='t',image_sha256='ih',annotation_sha256='ah',truth_sha256='th',truth_sources=[])
    manifest=dict(cases=[case]);record={**case,'strategy':'s19_combine','manifest_digest':m.manifest_digest(manifest)}
    # No real input is needed: identity errors must be rejected before input IO.
    with pytest.raises(RuntimeError,match='identity mismatch'):m.validate_records([{**record,'window':[1,10]}],manifest)
    with pytest.raises(RuntimeError,match='digest mismatch'):m.validate_records([{**record,'manifest_digest':'stale'}],manifest)
    with pytest.raises(RuntimeError,match='Duplicate'):m.validate_records([record,record],manifest)
