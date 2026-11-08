"""Behavior checks. All image fixtures are made in memory."""
import json

import numpy as np
from PIL import Image
import pytest

from harness import io, depth, legend
from ml.combine import runtime
from ml.combine.experiment import allowed_items, synthetic_records, configs
from ml.integrate.core import decode


def test_split_has_only_current_dev():
    items,meta=allowed_items()
    assert len(items)==51
    assert len({r['well'] for r in meta.values()})==5
    assert all(r['well']!='ForsythB2-17' for r in meta.values())


def test_synthetic_sources_are_disjoint_and_not_eval():
    records,_=synthetic_records()
    train={r['source'] for r in records if r['split']=='train'}
    held={r['source'] for r in records if r['split']=='held'}
    assert train and held and not train & held
    assert len({r['family'] for r in records if r['split']=='held'})==7


def test_equal_support_depth_fit_uses_residual_before_span():
    # A longer false three-anchor line competes with a precise printed scale.
    # This fixture uses OCR anchors from the archived failure, without labels.
    points=[(564.0554,39.),(567.9284,3.),(928.5652,4000.),
            (1303.4896,4050.),(1679.1323,4100.)]
    old=depth._fit(points)
    a,b,n,rmse=runtime.fit_depth_points(points)
    assert old[0]>3
    assert a==pytest.approx(50/(1679.1323-1303.4896),rel=.005)
    assert n==3 and rmse<.4
    assert a*1303.4896+b==pytest.approx(4050,abs=.1)


@pytest.mark.parametrize('a,b',[(.05,24000),(.133333,3800),(.5,6000),(3.2,100)])
def test_depth_fit_supports_different_scales(a,b):
    ys=np.array([100,400,700,1000,1400],float)
    result=runtime.fit_depth_points(list(zip(ys,a*ys+b))+[(50,1),(1550,98000)])
    assert result[0]==pytest.approx(a)
    assert result[1]==pytest.approx(b)
    assert result[2]==len(ys)


def test_depth_fit_does_not_count_rotated_copies_as_anchors():
    with pytest.raises(ValueError):
        runtime.fit_depth_points([(100,500),(101,500),(102,500),(1000,620)])


def test_checked_depth_uses_full_image_rows_and_converts_metres(monkeypatch):
    log=io.Log(Image.new('RGB',(40,1500),'white'),dict(depth_column=[0,200,40,1400]),'fixture')
    monkeypatch.setattr(depth,'_read_points',lambda *args:[(400,100),(800,150),(1200,200)])
    old=depth.Calib(2,0,3,6,'rapidocr+consensus+m-to-ft')
    result=runtime.checked_depth(log,old)
    assert result.to_depth(800)==pytest.approx(150*depth.METRES_TO_FEET)
    assert result.a==pytest.approx(.125*depth.METRES_TO_FEET)
    assert result.method.startswith('s19')
    log.close()


def test_good_depth_fit_needs_no_new_ocr(monkeypatch):
    log=io.Log(Image.new('RGB',(40,1500),'white'),dict(depth_column=[0,200,40,1400]),'fixture')
    def forbidden(*args):
        raise AssertionError('Strong calibration needs no new OCR')
    monkeypatch.setattr(io,'read_rgb',forbidden)
    old=depth.Calib(.5,2000,12,.5,'rapidocr+consensus+ft')
    assert runtime.checked_depth(log,old) is old
    log.close()


def test_joint_decoder_keeps_blank_gap_and_real_separator():
    class Matcher:
        def predict_proba(self,x):
            return np.tile([.1,.9],(len(x),1))
    sw=np.full((10,10,3),100,np.uint8)
    entries=[legend.LegendEntry(7,'Shale',[0,0,10,10],sw,(100,100,100),True)]
    points=np.array([0,40,80,120,160])
    ev=dict(points=points,x=np.zeros((4,1,27)),ink=np.array([.5,0,.5,.5]))
    cfg=list(configs())[1]
    rows=decode(ev,entries,np.full(160,.5),{120},cfg,model=Matcher())
    assert [(r['top_y'],r['bottom_y']) for r in rows]==[(0,40),(80,120),(120,160)]
    assert {r['lithology'] for r in rows}=={'Shale'}


def test_runtime_returns_corrected_depths_and_absolute_rows(monkeypatch):
    class Net:
        pass
    log=io.Log(Image.new('RGB',(20,400),'white'),dict(interpreted_lithology_column=[0,100,20,400]),'fixture')
    monkeypatch.setattr(runtime,'load',lambda:(Net(),None,{}))
    monkeypatch.setattr(runtime,'parsed',lambda log:[])
    monkeypatch.setattr(runtime,'features',lambda *args:None)
    monkeypatch.setattr(runtime,'probabilities',lambda *args:np.zeros(200))
    monkeypatch.setattr(runtime,'proposals',lambda *args:(np.array([0,200]),set()))
    monkeypatch.setattr(runtime,'predict',lambda *args:[dict(top_y=10,bottom_y=80,lithology='Shale',cluster_id=2)])
    fixed=depth.Calib(.1,3000,3,.01,'s19')
    monkeypatch.setattr(runtime,'checked_depth',lambda *args:fixed)
    rows=runtime.run(log,(100,300),dict(calib=depth.Calib(5,0,3,6,'old')))
    assert rows[0]['top_y']==110 and rows[0]['bottom_y']==180
    assert rows[0]['top_depth']==3011 and rows[0]['bottom_depth']==3018
    log.close()


def test_all_folds_exclude_query_well_after_training():
    path=runtime.HERE/'training.json'
    if not path.exists():
        pytest.skip('Run clean training first')
    fits=json.loads(path.read_text())['fits']
    assert len(fits)==12
    for fit in fits:
        assert fit['held_well'] not in fit['training_wells']
        assert 'ForsythB2-17' not in fit['training_wells']


def test_matcher_fit_cannot_use_held_well_values():
    from ml.synth_gulf.experiment import fit_matcher
    records=[]
    for group in ('train_a','train_b','held'):
        for offset in (0.,.2):
            records.append(dict(group=group,weight=40,
                     x=np.array([[.1+offset,.2],[1.+offset,.9]],np.float32),
                     y=np.array([1,0])))
    first=fit_matcher(records,[],'held',0.)
    changed=[dict(r,x=r['x']*10000,y=1-r['y']) if r['group']=='held' else r for r in records]
    second=fit_matcher(changed,[],'held',0.)
    probes=np.array([[.2,.3],[.8,.7]],np.float32)
    np.testing.assert_array_equal(first.predict_proba(probes),second.predict_proba(probes))


def test_bad_new_depth_evidence_keeps_the_input_scale(monkeypatch):
    log=io.Log(Image.new('RGB',(40,1500),'white'),dict(depth_column=[0,200,40,1400]),'fixture')
    monkeypatch.setattr(depth,'_read_points',lambda *args:[(300,1),(301,5)])
    old=depth.Calib(.5,2000,3,6,'rapidocr+consensus+ft')
    assert runtime.checked_depth(log,old) is old
    log.close()


def test_parser_uses_explicit_defaults_and_restores_shared_state(monkeypatch):
    from harness import legend_v3
    old_home=legend_v3.HOME
    old_easy=legend_v3._easy
    log=io.Log(Image.new('RGB',(20,10),'white'),dict(lithology_legend=[0,0,20,10]),'fixture')
    def fail(log,config):
        assert config==dict(easy=True,cleanup=True,easy_weight=.9)
        assert legend_v3.HOME==runtime.HERE/'legend_cache'
        assert legend_v3._easy is runtime._easy_reader
        raise ValueError('Parser fixture failure')
    monkeypatch.setattr(legend_v3,'parse',fail)
    with pytest.raises(ValueError,match='fixture'):
        runtime.parsed(log)
    assert legend_v3.HOME==old_home
    assert legend_v3._easy is old_easy
    log.close()


def test_blended_boundary_and_matcher_outputs_use_the_same_weight(monkeypatch):
    class Model:
        def __init__(self,p):self.p=p
        def predict_proba(self,x):return np.tile([1-self.p,self.p],(len(x),1))
    mixed,real=Model(.9),Model(.2)
    scorer=runtime.MixedScorer(mixed,real,.25)
    np.testing.assert_allclose(scorer.predict_proba(np.zeros((3,27)))[:,1],.375)
    monkeypatch.setattr(runtime,'probabilities',lambda net,feat:np.full(4,net.p,np.float32))
    result=runtime.boundary_probabilities((mixed,real),None,dict(model_mix=.25))
    np.testing.assert_allclose(result,.375)


def test_selected_design_keeps_joint_decoding_and_synthetic_models():
    path=runtime.HERE/'selected.json'
    if not path.exists():
        pytest.skip('Run selection first')
    cfg=json.loads(path.read_text())
    assert cfg['mode']=='joint'
    assert cfg['boundary_fraction']==cfg['matcher_fraction']==.35
    assert 0<cfg.get('model_mix',1.)<=1
