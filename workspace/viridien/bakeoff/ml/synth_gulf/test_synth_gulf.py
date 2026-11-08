"""Check leakage guards, exact row geometry, and portable runtime equivalence."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pytest
import torch
from PIL import Image
from harness import io, legend
from ml.boundary.model import features
from ml.synth_gulf.generator import HERE, FAMILIES, excluded, make


def test_assets_exclude_eval_and_hold_out_sources_within_every_family():
    sources=json.loads((HERE/'assets.json').read_text())
    assert not ({s['log'] for s in sources} & excluded())
    for family in FAMILIES:
        train={s['log'] for s in sources if s['family']==family and s['split']=='train'}
        held={s['log'] for s in sources if s['family']==family and s['split']=='held'}
        assert train and held and train.isdisjoint(held)


@pytest.mark.parametrize('family',FAMILIES)
def test_generation_is_seeded_and_truth_has_valid_geometry(family):
    s=next(s for s in json.loads((HERE/'assets.json').read_text()) if s['family']==family)
    a,rows=make(s,1731,height=400)
    b,again=make(s,1731,height=400)
    assert rows==again
    np.testing.assert_array_equal(a,b)
    assert rows and a.shape[0]==400
    assert np.isfinite(features(a)).all()
    last=0
    for r in rows:
        assert last<=r['top_y']<r['bottom_y']<=400
        assert r['lithology']==s['labels'][r['cluster_id']]
        last=r['bottom_y']
    if family=='D-582 1-bit grid':assert set(np.unique(a)) <= {0,255}


def test_cv_wells_are_disjoint():
    paths=list(HERE.glob('cv_*/fold_*.pt'))
    if not paths:pytest.skip('CV not yet run')
    for p in paths:
        z=torch.load(p,map_location='cpu',weights_only=True)
        assert z['held_well'] not in z['training_wells']
        assert len(set(z['training_wells']))==5


def test_every_generated_source_matches_its_split():
    path=HERE/'synthetic_index.json'
    if not path.exists():pytest.skip('Generation not yet run')
    sources={s['log']:s for s in json.loads((HERE/'assets.json').read_text())}
    for r in json.loads(path.read_text()):
        assert r['split']==sources[r['source']]['split']


def test_runtime_and_selection_naming_agree():
    if not (HERE/'matcher.npz').exists():pytest.skip('Final fit not yet run')
    from ml.synth_gulf.runtime import load
    from ml.synth_gulf.experiment import interval_features,cached_rows
    from strategies.s14_learned_match import name_intervals
    _,matcher,cfg=load()
    source=json.loads((HERE/'assets.json').read_text())[0]
    rgb,_=make(source,1748,height=300)
    entries=[]
    for i,(label,p) in enumerate(zip(source['labels'],source['tiles'])):
        sw=np.array(Image.open(HERE/'assets'/p).convert('RGB'))
        entries.append(legend.LegendEntry(i,label,[0,0,sw.shape[1],len(sw)],sw,tuple(sw.mean((0,1))),True))
    points=[0,100,200,300]
    segments,entries=interval_features(rgb,points,entries)
    selected=cached_rows(segments,entries,matcher,500)
    actual=name_intervals(rgb,500,points,entries,matcher=(matcher,cfg['matcher']))
    assert actual==selected
