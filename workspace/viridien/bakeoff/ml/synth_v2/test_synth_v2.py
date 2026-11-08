"""Check data limits, source splits, exact geometry, and model export."""
import json
import pickle
import numpy as np
from PIL import Image
import pytest
import torch
from harness import io
from ml.synth_v2.generator import HERE, FAMILIES, excluded, make
from ml.synth_v2.experiment import COUNTS, dev_items, entries_for


def test_current_dev_set():
    items, manifest = dev_items()
    assert len(items) == 51
    assert len({manifest[r['log']]['well'] for r in items}) == 5


def test_sources_are_disjoint_and_outside_eval():
    sources = json.loads((HERE/'assets.json').read_text())
    assert not {s['log'] for s in sources} & excluded()
    for family in FAMILIES:
        groups = {sp:{s['log'] for s in sources if s['family']==family and s['split']==sp}
                  for sp in ('train','held')}
        assert groups['train'] and groups['held']
        assert not groups['train'] & groups['held']
    assert sum(COUNTS) == 10000


@pytest.mark.parametrize('family', FAMILIES)
def test_reproducible_geometry_in_each_family(family):
    sources = json.loads((HERE/'assets.json').read_text())
    source = next(s for s in sources if s['family']==family)
    a, rows = make(source, 21023, height=512)
    b, other = make(source, 21023, height=512)
    np.testing.assert_array_equal(a,b)
    assert rows == other
    assert a.dtype == np.uint8
    assert a.shape[0] == 512 and a.shape[2] == 3
    last = 0
    for r in rows:
        assert last <= r['top_y'] < r['bottom_y'] <= 512
        assert r['lithology'] in source['labels']
        last = r['bottom_y']
    if family == 'D-582 1-bit grid':
        assert set(np.unique(a)) <= {0,255}


def test_corpus_source_provenance():
    records = json.loads((HERE/'synthetic_index.json').read_text())
    sources = {s['log']:s for s in json.loads((HERE/'assets.json').read_text())}
    assert len(records)==10000
    assert sum(r['split']=='train' for r in records)==8000
    assert sum(r['split']=='held' for r in records)==2000
    assert sum(r['selection'] for r in records)==140
    for r in records:
        assert sources[r['source']]['split']==r['split']
    items,_ = dev_items()
    allowed = {(r['log'],r['y0'],r['y1']) for r in items}
    dev = json.loads((HERE/'dev_index.json').read_text())
    assert {(r['log'],*r['window']) for r in dev} == allowed


def test_folds_exclude_the_whole_well():
    paths = sorted(HERE.glob('cv_*/fold_*.pt'))
    assert len(paths)==10
    for path in paths:
        p = torch.load(path,map_location='cpu',weights_only=True)
        assert p['held_well'] not in p['training_wells']
        assert len(p['training_wells'])==4
        assert p['inherited_weights'] is False


@pytest.mark.parametrize('family', FAMILIES)
def test_strategy_runs_without_truth_or_shared_models(monkeypatch, family):
    from strategies import s21_synth_v2
    from ml.synth_v2.runtime import load
    sources = json.loads((HERE/'assets.json').read_text())
    source = next(s for s in sources if s['family']==family)
    rgb, _ = make(source, 88821, height=768)
    log = io.Log(Image.fromarray(rgb),
                 {'interpreted_lithology_column':[0,0,rgb.shape[1],len(rgb)]}, 'unit_synthetic')
    def forbidden(*args, **kwargs):
        raise AssertionError('Inference must not read truth or shared models')
    monkeypatch.setattr(io,'load_rows',forbidden)
    from strategies import s14_learned_match
    monkeypatch.setattr(s14_learned_match,'load_matcher',forbidden)
    load.cache_clear()
    rows = s21_synth_v2.run(log,(0,len(rgb)),{'legend':entries_for(source)})
    assert rows
    last = 0
    for r in rows:
        assert last <= r['top_y'] < r['bottom_y'] <= len(rgb)
        assert r['lithology'] in source['labels']
        last = r['bottom_y']
    log.close()
    assert sum((HERE/f).stat().st_size for f in ('boundary_net.pt','matcher.npz')) < 10_000_000


def test_selection_and_runtime_naming_agree():
    from ml.synth_v2.runtime import load
    from ml.synth_gulf.experiment import interval_features, cached_rows
    from strategies.s14_learned_match import name_intervals
    _,matcher,cfg = load()
    source = json.loads((HERE/'assets.json').read_text())[0]
    rgb,_ = make(source,1748,height=400)
    es = entries_for(source)
    points = [0,100,200,300,400]
    seg,es = interval_features(rgb,points,es)
    expected = cached_rows(seg,es,matcher,500)
    actual = name_intervals(rgb,500,points,es,matcher=(matcher,cfg['matcher']))
    assert actual == expected
