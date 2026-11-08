"""Run: python -X utf8 -B -m pytest labeltool/test_labeltool.py -q -p no:cacheprovider."""
import csv
import json
from pathlib import Path
import shutil
import uuid

import pytest

from labeltool.common import HOME, ROOT, DATA, GOLD, load_rows, normalize, tier_for
from labeltool.build_queue import build, segment
from labeltool.store import Store
from labeltool.score_human import compare, score_sources, wilson


@pytest.fixture
def workspace():
    # pytest's own default tmp path may lie outside the authorized folders.
    parent=HOME/'cache'/'tests'
    parent.mkdir(parents=True,exist_ok=True)
    path=parent/('case_'+uuid.uuid4().hex)
    path.mkdir()
    yield path
    assert path.resolve().is_relative_to(parent.resolve())
    shutil.rmtree(path)


def test_queue_build_two_actual_logs(workspace):
    names=['0000000009_d-582_0000020435-0001_log02_fe_5in_md',GOLD]
    items,summary=build(names,workers=1,output=workspace)
    from harness import io
    assert summary['logs']==2
    assert set(r['log'] for r in items)==set(names)
    assert len({r['id'] for r in items})==len(items)
    for name in names:
        log=io.open_log(name)
        try:box=log.boxes['interpreted_lithology_column']
        finally:log.close()
        rows=[r for r in items if r['log']==name]
        assert rows[0]['top_y']==box[1]
        assert rows[-1]['bottom_y']==box[3]
        assert all(a['bottom_y']==b['top_y'] for a,b in zip(rows,rows[1:]))
        assert all(r['bottom_y']-r['top_y']>=12 for r in rows)
        assert all(r['truth_only']==(name==GOLD) for r in rows)
        assert all(r['suggestions'] for r in rows)
    assert any(s.get('truth') for r in items for s in r['suggestions'])
    assert not summary['errors']


def make_store(path):
    meta=path/'meta';meta.mkdir()
    (meta/'fake.json').write_text(json.dumps(dict(box=[0,0,50,1000],calibration=dict(a=.5,b=100))),encoding='utf-8')
    return Store(path/'human',meta)


def item(identity='i1',top=10,bottom=30):
    return dict(id=identity,log='fake',top_y=top,bottom_y=bottom,suggestion='Sand',tier='high',truth_only=False)


def test_save_rebuild_overlap_depth_and_reopen(workspace):
    store=make_store(workspace)
    first=store.save(item(),dict(label='Sand',mode='spot',request_id='request-1',ms_taken=123))
    store.save(item(),dict(label='Ignored duplicate',request_id='request-1'))
    assert len(store.events)==1
    assert first['suggestion_shown']=='Sand' and first['ms_taken']==123
    store.save(item('i2',20,40),dict(label='Clay',top_y=18,bottom_y=38,mode='walk'))
    rows=load_rows(store.directory/'fake.csv')
    assert [(r['top_y'],r['bottom_y'],r['lithology']) for r in rows]==[(10,18,'Sand'),(18,38,'Clay')]
    assert float(rows[0]['bottom_depth'])==109
    assert float(rows[1]['top_depth'])==109
    assert store.events[-1]['boundary_edited']
    assert list(csv.DictReader((store.directory/'fake.csv').open()))[0].keys()==dict.fromkeys(['top_depth','bottom_depth','lithology','top_y','bottom_y','cluster_id']).keys()
    reopened=Store(store.directory,store.metadata)
    assert reopened.completed()=={'i1','i2'}
    assert reopened.rows('fake')==store.rows('fake')


def test_undo_restores_overwritten_rows_and_survives_restart(workspace):
    store=make_store(workspace)
    store.save(item(),dict(label='Sand'))
    store.save(item('i2',15,25),dict(label='Clay'))
    assert len(store.rows('fake'))==3
    undone=store.undo()
    assert undone['item_id']=='i2'
    assert store.rows('fake')==[dict(top_y=10,bottom_y=30,lithology='Sand')]
    store=Store(store.directory,store.metadata)
    assert store.completed()=={'i1'}
    assert store.undo()['item_id']=='i1'
    assert load_rows(store.directory/'fake.csv')==[]
    assert store.undo() is None


def test_invalid_boundaries_and_gold_rejected(workspace):
    store=make_store(workspace)
    with pytest.raises(ValueError):store.save(item(),dict(label='Clay',top_y=40,bottom_y=20))
    with pytest.raises(ValueError):store.save(dict(item(),truth_only=True),dict(label='Clay'))
    assert not store.path.exists()


def test_tiny_score_only_human_rows():
    human=[dict(top_y=10,bottom_y=20,lithology='Sand'),dict(top_y=100,bottom_y=110,lithology='Clay')]
    pred=[dict(top_y=10,bottom_y=20,lithology='Sand'),dict(top_y=30,bottom_y=40,lithology='Wrong unreviewed gap'),
          dict(top_y=100,bottom_y=105,lithology='Clay'),dict(top_y=105,bottom_y=110,lithology='Sand')]
    result=score_sources({'fake':human},{'fake source':{'fake':pred},'missing source':{}})
    actual=result[0]
    assert actual['n_rows']==20 and actual['covered_rows']==20
    assert actual['correct_rows']==15 and actual['row_accuracy']==.75
    assert actual['boundary_hits']==4 and actual['predicted_boundaries']==5
    assert result[1]['row_accuracy']==0 and result[1]['covered_rows']==0
    assert compare([dict(top_y=0,bottom_y=200,lithology='Sand')],human)['correct_rows']==10


def test_confidence_and_wilson():
    agreeing=[dict(label='Sand',source='S11'),dict(label='sand',source='colour')]
    assert tier_for(agreeing,5,20,True)[0]=='high'
    assert tier_for(agreeing,5,2,True)[0]=='medium'
    assert tier_for(agreeing+[dict(label='Clay')],5,20,True)[0]=='low'
    assert tier_for(agreeing,None,None,True)[0]=='low'
    assert tier_for(agreeing,5,20,False)[0]=='low'
    low,high=wilson(5,10)
    assert .23<low<.24 and .76<high<.77


def test_generic_segmenter_covers_and_maps_rows():
    import numpy as np
    rgb=np.full((100,50,3),255,np.uint8)
    rgb[50:]=(240,140,30)
    intervals=segment(rgb,1000,1400,extra=[1001,1010,1200,1399])
    assert intervals[0][0]==1000 and intervals[-1][1]==1400
    assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
    assert all(b-a>=12 for a,b in intervals)
    assert any(abs(a-1200)<=12 for a,b in intervals)


def test_vectorized_fft_matches_original_s11():
    import numpy as np
    from strategies import s11_fft_legend as s11
    from labeltool.build_queue import fft_descriptor,fft_references,fft_distances
    rng=np.random.default_rng(17)
    patches=[rng.integers(0,256,(h,w,3),dtype=np.uint8) for h,w in [(1,1),(12,31),(64,64),(90,80)]]
    for mono in (False,True):
        originals=[s11.descriptor(patch,mono) for patch in patches]
        fast=[fft_descriptor(patch,mono) for patch in patches]
        for actual,expected in zip(fast,originals):
            for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)
        for original,actual in zip(originals,fast):
            expected=np.array([s11.distance(original,ref,mono) for ref in originals])
            np.testing.assert_allclose(fft_distances(actual,fft_references(fast),mono),expected,atol=1e-6)


def test_round_robin_does_not_restart_with_each_prefetch():
    from labeltool.serve import round_robin_order
    items=[dict(id=f'{name}{i}',log=name,priority=0,top_y=i*20) for name in ['a','b','c','d','e'] for i in range(3)]
    order=round_robin_order(items)
    sequence=sorted(items,key=lambda r:order[r['id']])
    assert [r['id'] for r in sequence[:6]]==['a0','b0','c0','d0','e0','a1']
    # First item saved; next three already buffered by the client.
    excluded={'a0','b0','c0','d0'}
    next_batch=sorted([r for r in items if r['id'] not in excluded],key=lambda r:order[r['id']])
    assert next_batch[0]['id']=='e0'
