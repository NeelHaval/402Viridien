"""BENCH checks use tiny fixtures; no test-well information enters generation."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from genlab.assemble import bench as b


def truth(layers=None,height=10):
    return dict(synth_id='x',height=height,layers=layers or [
        dict(top=0,bottom=5,class_id='a'),dict(top=5,bottom=10,class_id='b')])


def test_scorer_exact():
    t=truth();r=b.score_items([t],{'x':t['layers']})
    assert r['row_accuracy']==1
    assert r['boundary']['f1']==1
    assert r['per_class']['a']['precision']==1
    assert r['per_class']['b']['recall']==1


def test_scorer_shift_and_wrong_class():
    t=truth();p=[dict(top=0,bottom=6,class_id='a'),dict(top=6,bottom=10,class_id='b')]
    r=b.score_items([t],{'x':p},k=1)
    assert r['row_accuracy']==.9
    assert r['per_class']['a']['precision']==pytest.approx(5/6)
    assert r['per_class']['b']['recall']==.8
    assert r['boundary']['f1']==1
    assert b.score_items([t],{'x':p},k=0)['boundary']['f1']==0


def test_missing_rows_are_wrong_and_overlap_rejected():
    r=b.score_items([truth()],{'x':[dict(top=0,bottom=5,class_id='a')]})
    assert r['row_accuracy']==.5
    assert r['per_class']['b']['recall']==0
    assert r['boundary']['f1']==1  # coverage end still proposes the correct boundary
    assert b.score_items([truth()],{})['row_accuracy']==0
    with pytest.raises(ValueError,match='Overlapping'):
        b.score_items([truth()],{'x':[dict(top=0,bottom=7,class_id='a'),dict(top=5,bottom=10,class_id='b')]})
    with pytest.raises(ValueError,match='Unknown synth_id'):
        b.score_items([truth()],{'typo':[]})


def test_one_to_one_boundary_matching_and_no_boundary_case():
    assert b.boundary_counts([10,12],[11],2)==(1,2,1)
    assert b.boundary_counts([10],[9,11],2)==(1,1,2)
    t=truth([dict(top=0,bottom=10,class_id='a')])
    assert b.score_items([t],{'x':t['layers']})['boundary']['f1']==1


def test_gap_rows_and_labelled_only():
    t=truth([dict(top=2,bottom=8,class_id='a')])
    r=b.score_items([t],{'x':[dict(top=0,bottom=10,class_id='a')]})
    assert r['row_accuracy']==.6
    r=b.score_items([t],{'x':[dict(top=0,bottom=10,class_id='a')]},labelled_only=True)
    assert r['rows']==6 and r['row_accuracy']==1


def test_prediction_jsonl_formats():
    p=b.BENCH/'fixtures/predictions-test/pred.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(dict(synth_id='x',layers=truth()['layers']))+'\n',encoding='utf-8')
    assert b.load_predictions(p)=={'x':truth()['layers']}
    p.write_text('\n'.join(json.dumps(dict(synth_id='x',**l)) for l in truth()['layers']),encoding='utf-8')
    assert b.score_items([truth()],b.load_predictions(p))['row_accuracy']==1


def test_suite_plan_seed_determinism():
    for name in b.NAMES:
        assert b.suite_plan(name,42)==b.suite_plan(name,42)
        assert b.suite_plan(name,42)['seeds']!=b.suite_plan(name,43)['seeds']


def test_fixture_pixels_and_truth_deterministic():
    p=b.make_fixtures()
    def hashes():
        return {str(x.relative_to(p)):hashlib.sha256(x.read_bytes()).hexdigest()
                for x in p.rglob('*') if x.is_file()}
    first=hashes();b.make_fixtures();assert hashes()==first
    t=b.load_run(p)
    assert len(t)==2
    assert b.score_items(t,{x['synth_id']:x['layers'] for x in t})['row_accuracy']==1


def test_seeded_suite_orchestration_without_private_compose_imports(monkeypatch,tmp_path):
    # Simulate only the public CLI and its public manifest; all actual writes
    # remain in BENCH fixtures. No COMPOSE-owned out file is touched.
    monkeypatch.setattr(b,'BENCH',tmp_path)
    base=tmp_path/'suite-test'
    monkeypatch.setattr(b,'SUITES',base/'suites')
    stats={'schema':'asm-1','built_at':1,'classes':{'doc:1':dict(doc_id='doc',norm='shale',name='Shale')},
           'levels':{'global':dict(composition={'all':{'shale':1}})}}
    original=b.read;manifest={};calls=[]
    def read(path):
        p=Path(path)
        if p==b.ASSEMBLE/'stats.json':return stats
        if p==b.ROOT/'data/genlab/state.json':return {'cards':[{'key':'doc:1'}]}
        if p.name=='manifest.json' and p.parent.name.startswith('bench-typical-'):return manifest[p.parent.name]
        return original(p)
    # Avoid depending on a parallel agent having finished stats.json.
    original_exists=Path.exists
    monkeypatch.setattr(Path,'exists',lambda p:True if p==b.ASSEMBLE/'stats.json' else original_exists(p))
    monkeypatch.setattr(b,'read',read);monkeypatch.setattr(b,'report',lambda:None)
    monkeypatch.setattr(b,'training_document',lambda did,policy:True)
    # The fake CLI consumes these fixture inputs only. Parallel noise refreshes
    # and real source edits must not change this isolated orchestration cohort.
    monkeypatch.setattr(b,'suite_inputs',lambda *args:dict(schema='fixture',stats=stats,inventory=[],files={}))
    dependencies=[b.ASSEMBLE/'noise/measurements.json']+[b.ROOT/p for p in
        ('genlab/assemble/compose.py','genlab/assemble/noise.py','genlab/render.py')]
    original_bytes=Path.read_bytes
    frozen_bytes={p:original_bytes(p) for p in dependencies if p.exists()}
    monkeypatch.setattr(Path,'read_bytes',lambda p:frozen_bytes[p] if p in frozen_bytes else original_bytes(p))
    original_digest=b.digest_file
    monkeypatch.setattr(b,'digest_file',lambda p:b.digest_object(stats) if Path(p)==b.ASSEMBLE/'stats.json' else original_digest(p))
    def cli(command,log):
        assert command[3:6]==['-m','genlab.assemble.compose','column']
        calls.append(command)
        seed=int(command[command.index('--seed')+1]);run=command[command.index('--run')+1]
        source=base/'fake-cli'/str(seed);source.mkdir(parents=True,exist_ok=True)
        a=np.random.default_rng(seed).integers(0,256,(16,8,3),dtype=np.uint8)
        Image.fromarray(a).save(source/'column.png')
        t=dict(synth_id='compose-id',height=16,width=8,seed=seed,mode='column',base_doc='doc',
               layers=[dict(top=0,bottom=16,class_id='doc:1',norm='shale')])
        b.write_json(source/'truth.json',t)
        manifest[run]={'outputs':[{'path':str(source)}]}
    monkeypatch.setattr(b,'execute_compose',cli)
    existing=base/'suites/typical/suite.json'
    if original_exists(existing):
        stale=original(existing);stale['status']='building';b.write_json(existing,stale)
    first=b.build_suite('typical',seed=74917,n=2,height=16)
    fingerprints=dict(first['truth_sha256'])
    first['status']='building';b.write_json(base/'suites/typical/suite.json',first)
    second=b.build_suite('typical',seed=74917,n=2,height=16)
    assert fingerprints==second['truth_sha256']
    assert len({s['synth_id'] for s in second['samples']})==2
    assert second['counts']['rows']==32
    assert len(calls)==2  # Immutable version reused, never regenerated in place.


@pytest.mark.parametrize('destination',['stats.json','stats-report.md','stats/private.json','out/starter/test.json'])
def test_write_guard_forbids_generation_paths(destination):
    with pytest.raises(ValueError,match='outside BENCH'):
        b.write_json(b.ASSEMBLE/destination,{'TEST_SENTINEL':.12345})


def test_test_well_results_only_in_bench(monkeypatch):
    real=[dict(split='train',well='validation',base_doc='a'),dict(split='test',well='test-well',base_doc='b')]
    monkeypatch.setattr(b,'load_run',lambda run:[dict(base_doc='a',stats_level_used={})])
    snapshot=b.read(b.ROOT/'data/normalized/current.json')['snapshot']
    monkeypatch.setattr(b,'real_data',lambda:(real,{'document_groups':{'a':'validation','b':'test-well'}},snapshot))
    monkeypatch.setattr(b,'validation_well',lambda well:True)
    monkeypatch.setattr(b,'compare_fidelity',lambda rr,ss,stats=None:{'TEST_SENTINEL':987654321 if rr[0]['split']=='test' else 0,
                                                          'metrics':{'order1_js_bits':{}}})
    monkeypatch.setattr(b,'noisy_reference',lambda selected:selected)
    monkeypatch.setattr(b,'report',lambda:None)
    monkeypatch.setattr(b,'result_dir',lambda run:b.BENCH/'fixtures'/'leak-check')
    writes=[];original=b.write_json
    def record(path,obj):
        writes.append(Path(path).resolve());original(path,obj)
    monkeypatch.setattr(b,'write_json',record)
    out=b.fidelity('fixture-leak-check')
    assert out['test']['TEST_SENTINEL']==987654321
    assert writes and all(p.is_relative_to(b.BENCH.resolve()) for p in writes)
    assert '987654321' in (b.result_dir('fixture-leak-check')/'fidelity.json').read_text()


def test_validation_excludes_every_page_of_doc_stat_well():
    groups={'a':'same-well','b':'same-well','c':'other-well'}
    synth=[dict(base_doc='a',stats_level_used={'doc:a/thickness':3})]
    # Use production-sized IDs for the actual doc: parser.
    groups={'0'*16:'same-well','1'*16:'same-well','2'*16:'other-well'}
    synth=[dict(base_doc='0'*16,stats_level_used={'doc:'+('0'*16)+'/thickness':3})]
    assert b.used_doc_wells(synth,groups)=={'same-well'}


def test_transition_js_known_cases():
    from collections import Counter
    assert b.transition_js(Counter({('a','b'):10}),Counter({('a','b'):5}))==0
    assert b.transition_js(Counter({('a','b'):10}),Counter({('a','c'):5}))==pytest.approx(1)
    assert b.transition_js(Counter(),Counter({('a','b'):10})) is None


def test_native_page_bands_window():
    folder=b.BENCH/'fixtures'/'page';folder.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.full((4,8,3),40,np.uint8)).save(folder/'a.png')
    Image.fromarray(np.full((4,8,3),80,np.uint8)).save(folder/'b.png')
    b.write_json(folder/'page-bands.json',{'bands':[{'file':'a.png','top':0,'bottom':4},{'file':'b.png','top':4,'bottom':8}]})
    rgb=b.page_column_window(folder,[2,1,6,7],1,5)
    assert rgb.shape==(4,4,3)
    assert (rgb[:2]==40).all() and (rgb[2:]==80).all()


def test_large_lzw_strip_streams_exact_windows(monkeypatch):
    import model
    import tifffile
    if not b.ctypes.util.find_library('tiff'):
        pytest.skip('libtiff not installed')
    folder=b.BENCH/'fixtures'/'scanlines';(folder/'images').mkdir(parents=True,exist_ok=True)
    rgb=np.random.default_rng(9).integers(0,256,(5001,9,3),dtype=np.uint8)
    tifffile.imwrite(folder/'images/source.tif',rgb,compression='lzw',rowsperstrip=5001)
    monkeypatch.setattr(model,'DATASET',folder)
    doc=dict(id='stream-fixture',image_name='source.tif',width=9,height=5001)
    c=b.RealCrops()
    try:
        for a,z in ((1000,1128),(4000,4128),(32,160)):
            actual=c.get(doc,[2,a,7,z])
            assert np.array_equal(actual,rgb[a:z,2:7])
    finally:
        c.close()


def test_frozen_run_survives_parallel_compose_rewrite(monkeypatch):
    base=b.BENCH/'fixtures'/'snapshot-test'
    monkeypatch.setattr(b,'ASSEMBLE',base)
    source=base/'out'/'fixture-run';sample=source/'sample';sample.mkdir(parents=True,exist_ok=True)
    b.write_json(source/'manifest.json',{'outputs':[{'path':str(sample)}]})
    b.write_json(sample/'truth.json',truth())
    Image.fromarray(np.full((10,8,3),40,np.uint8)).save(sample/'column.png')
    first=b.freeze_run('fixture-run')
    fingerprint=hashlib.sha256((first/'sample/column.png').read_bytes()).hexdigest()
    Image.fromarray(np.full((10,8,3),80,np.uint8)).save(sample/'column.png')
    second=b.freeze_run('fixture-run')
    assert first!=second
    assert hashlib.sha256((first/'sample/column.png').read_bytes()).hexdigest()==fingerprint
    assert (first/'sample/column.png').read_bytes()!=(second/'sample/column.png').read_bytes()


def test_spatially_conflicting_boxes_never_become_row_truth():
    layers=[dict(top=0,bottom=5,norm='a'),dict(top=4,bottom=8,norm='b'),dict(top=8,bottom=10,norm='b')]
    clean,n=b.human_chain(layers)
    assert clean==[layers[-1]] and n==2
    clean,n=b.human_chain([dict(top=0,bottom=6,norm='a'),dict(top=4,bottom=8,norm='a')])
    assert clean==[dict(top=0,bottom=8,norm='a')] and n==0


def test_tiled_page_window_is_exact_across_tile_edges():
    import tifffile
    folder=b.BENCH/'fixtures'/'tiled-page';folder.mkdir(parents=True,exist_ok=True)
    rgb=np.random.default_rng(71).integers(0,256,(48,64,3),dtype=np.uint8)
    tifffile.imwrite(folder/'page.tif',rgb,tile=(16,16),compression='deflate')
    assert np.array_equal(b.tiled_tiff_window(folder/'page.tif',[11,9,41,37]),rgb[9:37,11:41])
    assert np.array_equal(b.page_column_window(folder,[11,5,41,45],4,32),rgb[9:37,11:41])
