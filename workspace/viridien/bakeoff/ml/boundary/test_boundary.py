"""Checks for spatial features, chunk stitching, and well-isolated folds."""
import json

import numpy as np
import pytest
import torch

from ml.boundary.model import BoundaryNet, HERE, features, probabilities
from ml.boundary.experiment import synth_column, targets


@pytest.mark.parametrize('kind,channels', [('six',6),('rich',32)])
def test_features_are_finite_for_blank_solid_and_pattern_columns(kind,channels):
    for rgb in (np.full((150,60,3),255,np.uint8),np.zeros((150,60,3),np.uint8),synth_column(7,150)[0]):
        f=features(rgb,kind)
        assert f.shape==(150,channels)
        assert np.isfinite(f).all()


def test_chunked_inference_matches_whole_sequence():
    torch.set_num_threads(3)
    torch.manual_seed(8)
    net=BoundaryNet(channels=6,hidden=8).eval()
    f=np.random.default_rng(8).normal(size=(17000,6)).astype(np.float32)
    actual=probabilities(net,f)
    with torch.inference_mode():
        expected=net(torch.from_numpy(f.T.copy()[None])).sigmoid().numpy()[0]
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)


def test_soft_targets_keep_each_true_peak_and_taper_with_distance():
    target=targets(100,np.array([1,50,98]))
    assert target[1]==target[50]==target[98]==1
    assert target[50]>target[54]>target[58]>target[62]
    assert target[30]==0


def test_all_cv_predictions_hold_out_entire_well():
    paths=sorted((HERE/'runs').glob('*/fold_*.pt'))
    if not paths:
        pytest.skip('CV checkpoints have not been generated')
    index=json.loads((HERE/'data'/'index.json').read_text())
    all_wells={c['well'] for c in index['cases']}
    assert len(all_wells)==6
    for path in paths:
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert payload['held_well'] not in payload['training_wells']
        assert set(payload['training_wells'])==all_wells-{payload['held_well']}


def test_training_manifest_excludes_reserved_in_domain_logs():
    index=json.loads((HERE/'data'/'index.json').read_text())
    reserved={v['log'] for v in json.loads((HERE.parents[1]/'eval_set.json').read_text())}
    assert not reserved.intersection(c['log'] for c in index['cases'])
    assert not reserved.intersection(index['extra_human_labels'])


def test_legend_matching_agrees_with_s11_given_same_intervals(monkeypatch):
    from PIL import Image
    from harness import io, legend
    from strategies import s11_fft_legend as s11
    from strategies.s13_boundary_net import name_intervals
    swatches=[np.full((64,64,3),c,np.uint8) for c in ((220,80,40),(40,150,210))]
    entries=[legend.LegendEntry(i,str(i),[0,0,64,64],s,tuple(s.mean((0,1))),True) for i,s in enumerate(swatches)]
    rgb=np.concatenate([swatches[0],swatches[1],swatches[0]],axis=0)
    log=io.Log(Image.fromarray(rgb),{'interpreted_lithology_column':[0,0,64,len(rgb)]},'virtual')
    monkeypatch.setattr(s11,'segment',lambda *_:[(0,64),(64,128),(128,192)])
    expected,_=s11.run(log,(0,192),{'legend':entries})
    actual=name_intervals(rgb,0,[0,64,128,192],entries)
    assert actual==expected
    log.close()
