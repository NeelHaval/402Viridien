"""Offline checks: no external TEST pixels or truth are read."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from . import bench, depth, io, unlabelled_metrics as u


def row(a,b,name='Sand',cluster=None,**extra):
    r=dict(top_y=a,bottom_y=b,lithology=name,**extra)
    if cluster is not None:r['cluster_id']=cluster
    return r


def test_coverage_clips_unions_and_excludes_unknown_names():
    rows=[row(-10,30),row(20,50),row(70,130,'Unknown'),row(80,100,'Shale')]
    result=u.measure(rows,(0,100))
    assert result['assigned_share']==.7
    assert result['overlapping_intervals']==2
    assert result['failure_rate']==0
    assert result['legend_entries_found'] is None
    assert u.measure([], (0,100))['failure_rate']==1
    assert u.measure([row(0,100)],(0,100),failure=True)['failure_rate']==1


@pytest.mark.parametrize('name', ['Unknown','unlabelled','cluster_2','entry 3','',None,'123','n/a','blank'])
def test_unnamed_entries(name):
    assert not u.named(name)


def test_depth_checks_require_evidence_and_check_predicted_order():
    good=depth.Calib(.1,1000,4,2,'fixture')
    r=u.depth_sanity(good,[row(0,20,top_depth=1000,bottom_depth=1002),
                           row(20,40,top_depth=1001,bottom_depth=1004)])
    assert r['depth_fit_ok']
    assert not r['predicted_depth_monotonic']
    assert not u.depth_sanity(depth.Calib(.1,0,2,0,'two-anchor'))['depth_fit_ok']
    assert not u.depth_sanity(depth.Calib(-.1,0,10,0,'bad'))['depth_fit_ok']
    assert not u.depth_sanity(depth.Calib(.1,0,10,17,'bad'))['depth_fit_ok']
    assert not u.depth_sanity(depth.Calib(float('nan'),0,10,0,'bad'))['depth_fit_ok']
    assert u.depth_sanity()['depth_monotonic'] is None


def test_legend_counts_short_beds_and_redundant_boundaries():
    entries=[dict(label='Sand',is_lithology=True),dict(label='cluster_1'),
             dict(label='Gas',is_lithology=False)]
    r=u.measure([row(0,4),row(4,10),row(10,100,'Shale')],(0,100),legend_entries=entries)
    assert r['legend_entries_found']==2 and r['legend_entries_named']==1
    assert r['short_bed_rate']==pytest.approx(2/3)
    assert r['oversegmentation_rate']==.5
    assert r['beds_per_1000px']==30


def test_ari_is_partition_invariant_and_never_counts_common_blanks():
    a=[row(10,50,'Sand',0),row(50,90,'Shale',1)]
    b=[row(10,50,'Different name',7),row(50,90,'Other name',2)]
    result=u.agreement(a,b,(0,100))
    assert result['ari']==1 and result['common_assigned_share']==.8
    assert u.agreement([],[],(0,100))['ari'] is None
    assert u.agreement(a,b,(0,100),failure_a=True)['ari'] is None
    assert u.agreement([row(0,10)],[row(90,100)],(0,100))['ari'] is None
    fragmented=[row(10,30,'A'),row(30,50,'B'),row(50,90,'C')]
    assert u.agreement(a,fragmented,(0,100))['ari']<1


def test_report_pairs_by_window_and_keeps_failed_runs(tmp_path):
    records=[]
    for strategy in ('s19_combine','s17_synth_gulf','s22_boundary_v2'):
        path=f'{strategy}.csv'
        io.save_rows(tmp_path/path,[row(0,50,'Sand',0),row(50,100,'Shale',1)])
        records.append(dict(strategy=strategy,suite='unlab_test',log='fixture',source='offline',
                            well='well',family='test',y0=0,y1=100,csv=path,scored=False,
                            failure=strategy.startswith('s22')))
    payload=u.write_report(pd.DataFrame(records),tmp_path)
    assert len(payload['metrics'])==3 and len(payload['agreement'])==2
    assert payload['agreement'][0]['ari']==1
    assert payload['agreement'][1]['ari'] is None
    assert payload['by_family'][2]['failure_rate']==1
    json.loads((tmp_path/'unlabelled_report.json').read_text(encoding='utf-8'))


def test_scan_stability_uses_matching_page_and_failure_support(tmp_path):
    clean=tmp_path/'clean';scan=tmp_path/'scan';clean.mkdir();scan.mkdir()
    identity=dict(strategy='s19_combine',suite='unlab_test',log='page',source='offline',
                  well='well',family='test',y0=0,y1=100,csv='page.csv',scored=False,failure=False)
    io.save_rows(clean/'page.csv',[row(0,50,'Sand',0),row(50,100,'Shale',1)])
    io.save_rows(scan/'page.csv',[row(0,25,'Sand',0),row(25,100,'Shale',1)])
    (clean/'results.json').write_text(json.dumps(dict(results=[identity])),encoding='utf-8')
    report=u.write_report([identity],scan,clean_dir=clean)
    metric=report['metrics'][0]
    assert metric['scan_common_assigned_share']==1
    assert metric['scan_ari']<1 and metric['scan_eligible']
    failed=dict(identity,failure=True,unlabelled_metrics=u.measure([row(0,100)],(0,100)))
    report=u.write_report([failed],scan,clean_dir=clean)
    assert report['metrics'][0]['scan_ari'] is None
    assert report['metrics'][0]['failure_rate']==1


def test_registered_suite_has_no_truth(tmp_path,monkeypatch):
    root=tmp_path/'dataset';root.mkdir()
    items=root/'eval_set.json';items.write_text('[]',encoding='utf-8')
    registry=tmp_path/'suites.json';registry.write_text('{"suites":{}}',encoding='utf-8')
    monkeypatch.setenv('BENCH_UNLAB_TEST_DATASET',str(root))
    monkeypatch.setenv('BENCH_UNLAB_TEST_ITEMS',str(items))
    monkeypatch.setenv('BENCH_UNLAB_TEST_MANIFEST',str(root/'absent.csv'))
    suite=bench.load_suite('unlab_test',registry=registry)
    assert suite['truth_mode']=='none' and suite['truth'] is None
    assert bench._truth(suite,'fixture',(0,100)) is None


def test_missing_geometry_is_a_reported_failure(tmp_path):
    failed=dict(strategy='s19_combine',suite='unlab_test',log='missing',source='offline',
                well='well',family='test',y0=None,y1=None,csv=None,scored=False,failure=True)
    report=u.write_report([failed],tmp_path)
    assert report['metrics'][0]['failure_rate']==1
    assert report['metrics'][0]['assigned_share'] is None
    assert report['agreement']==[]


def test_bench_caches_diagnostics_without_truth(tmp_path,monkeypatch):
    from .test_bench import fixture_job
    job=fixture_job(tmp_path)
    job['config'].update(name='unlab_test',truth_mode='none',truth=None)
    monkeypatch.setattr(bench,'_thread_budget',lambda _:nullcontext())
    monkeypatch.setattr(bench.run,'_strategy',lambda _:SimpleNamespace(run=lambda *args:[row(0,100)]))
    monkeypatch.setattr(bench.legend,'parse',lambda _:[dict(label='Sand')])
    monkeypatch.setattr(bench.depth,'calibrate',lambda _:depth.Calib(.1,100,4,1,'fixture'))
    first=bench._execute(job)
    assert not first['failure'],first.get('traceback')
    assert first['score'] is None and not first['scored']
    assert first['unlabelled_metrics']['depth_fit_ok']
    second=bench._execute(job)
    assert second['cached']
    assert second['unlabelled_metrics']==first['unlabelled_metrics']
