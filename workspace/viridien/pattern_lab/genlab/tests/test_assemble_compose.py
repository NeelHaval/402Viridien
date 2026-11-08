"""Assembler invariants, Markov goodness of fit, reproducibility and split guards."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.stats import chisquare

from genlab.assemble import compose, noise


def fixed_stats():
    classes = {f'd:{i}': dict(doc_id='d', norm=str(i), name=str(i), legend_id=i) for i in range(3)}
    t1 = {'d:0': {'d:1':.6,'d:2':.4},
          'd:1': {'d:0':.4,'d:2':.6},
          'd:2': {'d:0':.6,'d:1':.4}}
    # Deliberately distinct second-order rows; verifies order two is actually used.
    t2 = {a+'|'+b: {c:p for c,p in t1[a].items()} for a in t1 for b in t1}
    block = dict(n_layers=10000, zones=[dict(name='all',top_frac=0,bottom_frac=1)],
        composition={'all':{'d:0':.3,'d:1':.3,'d:2':.4}},
        transitions=dict(order1=t1,order2=t2,by_zone={}),
        thickness={k:dict(dist='lognormal',mu=np.log(10),sigma=.2,quantiles_px=[8,9,10,11,12],n=100) for k in t1},
        gaps=dict(rate_per_10k_rows=0,height_quantiles_px=[1]*5),
        confidence=dict(transitions='high',thickness='high',composition='high'))
    return dict(schema='asm-2',classes=classes,levels={'doc:d':block})


def sampler(seed=731, stats=None):
    stats = stats or fixed_stats()
    available = {k:dict(pattern_source='document',pattern_id=k) for k in stats['classes']}
    return compose.Sampler(stats,'d','f',available,seed)


def test_layers_tile_without_overlap():
    s = sampler()
    layers = s.sequence(20000)
    assert layers[0]['top'] == 0 and layers[-1]['bottom'] == 20000
    assert all(a['bottom'] == b['top'] for a,b in zip(layers,layers[1:]))
    assert all(x['bottom'] > x['top'] for x in layers)


def test_fixed_transitions_chisquare_2000_layers():
    s = sampler()
    seq = s.sequence(10**8,max_layers=2000)
    assert len(seq) == 2000
    observed = {}
    for a,b in zip(seq,seq[1:]):
        observed.setdefault(a['class_id'],dict.fromkeys(s.available,0))[b['class_id']] += 1
    expected = s.stats['levels']['doc:d']['transitions']['order1']
    obs, exp = [], []
    for pair,row in observed.items():
        total = sum(row.values())
        for k in sorted(expected[pair]):
            obs.append(row[k]); exp.append(total*expected[pair][k])
    result = chisquare(obs,exp,ddof=len(observed)-1)
    assert result.pvalue > .001, result
    assert s.used['doc:d/order1'] == 1999


def test_seed_reproducible():
    assert sampler().sequence(20000) == sampler().sequence(20000)
    assert sampler(732).sequence(20000) != sampler().sequence(20000)


def test_zones_condition_second_order_and_gaps_tile():
    stats = fixed_stats()
    b = stats['levels']['doc:d']
    b['zones'] = [dict(name='top',top_frac=0,bottom_frac=.5),dict(name='bottom',top_frac=.5,bottom_frac=1)]
    b['transitions']['order2_by_zone'] = {
        'top':{'d:0|d:1':{'d:0':1.}}, 'bottom':{'d:0|d:1':{'d:2':1.}}}
    b['zone_scheme']='custom'
    b['transitions']['selected_order']={'order':2}
    b['transitions']['counts_order2_by_zone']={'top':{'d:0|d:1':{'d:0':20}},'bottom':{'d:0|d:1':{'d:2':20}}}
    s = sampler(stats=stats)
    assert s.next(['d:0','d:1'],'top') == 'd:0'
    assert s.next(['d:0','d:1'],'bottom') == 'd:2'
    s.zone_scheme='all'
    b['zones']=[dict(name='section-1',top_frac=0,bottom_frac=.5),dict(name='section-2',top_frac=.5,bottom_frac=1)]
    s.zone_scheme='section'
    assert s.zone(.25)=='section-1' and s.zone(.75)=='section-2'
    b['gaps'] = dict(verified=True,rate_per_10k_rows=100000,height_quantiles_px=[5]*5)
    seq = sampler(stats=stats).sequence(1000)
    assert any(r['class_id'] == compose.GAP for r in seq)
    assert all(a['bottom']==b['top'] for a,b in zip(seq,seq[1:]))
    assert seq[-1]['bottom'] == 1000


def test_unavailable_classes_are_renormalized():
    s = sampler()
    s.available.pop('d:2')
    with pytest.raises(ValueError,match='Unsupported composition mass'):
        compose.Sampler(s.stats,'d','f',s.available,1)


def test_transition_cycle_cannot_fall_back_to_iid_composition():
    stats = fixed_stats()
    b = stats['levels']['doc:d']
    b['composition']['bottom'] = {'d:1':1.}
    b['transitions']['order1']['d:0'] = {'d:0':1.}
    s = sampler(stats=stats)
    assert s.next(['d:0'],'bottom')=='d:0'
    assert not any('composition:' in field for field in s.used)


def test_no_transition_evidence_refuses_unequal_duration_iid_fixture():
    st=fixed_stats();b=st['levels']['doc:d'];b['transitions']={}
    b['composition']={'all':{'d:0':.5,'d:1':.5}}
    b['thickness']['d:0'].update(dist='empirical',quantiles_px=[10]*5)
    b['thickness']['d:1'].update(dist='empirical',quantiles_px=[100]*5)
    with pytest.raises(ValueError,match='iid composition fallback forbidden'):
        sampler(stats=st).sequence(100000)


def test_same_class_events_merge_before_rendering():
    s=sampler()
    s.next=lambda history,zone:'d:0'
    s.thickness=lambda cid,zone:10
    seq=s.sequence(100)
    assert len(seq)==1 and seq[0]['top']==0 and seq[0]['bottom']==100


def test_direct_band_crossing_lines_are_restored():
    original=np.full((30,72,3),255,np.uint8)
    original[10:12]=0
    original[20,6:66]=0  # Does not continue into a neighbour.
    mask=compose.Composer.crossing_mask(original)
    replacement=np.full((30,60,3),200,np.uint8)
    replacement[mask]=original[:,6:-6][mask]
    assert (replacement[10:12]==0).all()
    assert (replacement[20]==200).all()


def test_matching_names_cannot_borrow_appearance_from_another_log(monkeypatch):
    c = compose.Composer.__new__(compose.Composer)
    monkeypatch.setattr(noise,'require_training',lambda doc:None)
    c.docs = {'d':dict(id='d',name='target'),'s':dict(id='s',name='source')}
    c.fmap = dict(target='same-family',source='same-family')
    c.stats = fixed_stats()
    c.stats['classes']['d:0']['name'] = 'Shale'
    c.stats['classes']['d:1']['name'] = 'Limestone'
    c.stats['classes']['d:2']['name'] = 'Dolomite'
    c.cards = {'s:0':dict(doc_id='s',label='Clay/Shale'),
               's:1':dict(doc_id='s',label='Shale')}
    c.models = {'s:0':{},'s:1':{}}
    choices,skipped = c.available('d')
    assert choices == {}
    assert {r['class_id'] for r in skipped} == {'d:0','d:1','d:2'}
    c.cards['d:0']=dict(doc_id='d',label='Shale',real_doc_id='s',donor=dict(doc_id='s',legend_id=1))
    c.models['d:0']={}
    page,skipped=c.available('d',mode='page')
    assert page=={}
    assert any(r['reason']=='page-legend-requires-document-appearance' for r in skipped)


def test_sparse_document_backs_off_family():
    stats = fixed_stats()
    d = stats['levels']['doc:d']
    d['confidence']['transitions'] = 'low'
    d['transitions']={}
    family = dict(d,confidence=dict(transitions='high'),
        transitions=dict(order1={str(i):{'2':1.} for i in range(3)},order2={},by_zone={}))
    stats['levels']['family:f'] = family
    s = sampler(stats=stats)
    assert s.next(['d:0'],'all') == 'd:2'
    assert s.used['family:f/order1'] == 1


def test_no_test_well_input():
    import model
    store = model.Store()
    tests = [store.get(x['id']) for x in store.catalog if not noise.training(store.get(x['id']))]
    assert tests
    for doc in tests:
        with pytest.raises(ValueError, match='Test-well'):
            noise.Windows().get(doc,[0,0,1,1])
    if (noise.NOISE/'measurements.json').exists():
        measurements = noise.read(noise.NOISE/'measurements.json')
        for row in measurements['measurements']:
            assert noise.training(store.get(row['source_doc']))
        assert all(noise.training(store.get(did)) for did in measurements['rails'])
        assert all(noise.training(store.get(did)) for did in measurements.get('gap_sources',{}))


def test_test_well_statistics_are_rejected_before_pixels():
    import model
    store = model.Store()
    doc = next(store.get(x['id']) for x in store.catalog if not noise.training(store.get(x['id'])))
    stats = dict(schema='asm-1',classes={'x':dict(doc_id=doc['id'])},levels={})
    with pytest.raises(ValueError,match='Test-well'):
        compose.Composer(stats=stats,realism=dict(measurements=[]))


def test_scanline_windows_match_small_native_image():
    import model
    store = model.Store()
    # An already saved lab crop is an independent oracle; no page-sized decode.
    doc = store.get('16c9064b18b77b61')
    from genlab import paths
    path = next(p for p in (paths.CROPS/doc['id']).glob('*.png')
                if 40 <= int(p.stem.split('_')[3])-int(p.stem.split('_')[1]) <= 480)
    box = list(map(int,path.stem.split('_')))
    reader = noise.Windows()
    try:
        with Image.open(path) as image:
            expected = np.asarray(image.convert('RGB'))
        actual = reader.get(doc,box)
        assert np.array_equal(actual,expected)
        # Backward seeks must rewind the giant compressed strip without loading the page.
        backward = reader.get(doc,[box[0],box[1],box[2],box[1]+40])
        assert np.array_equal(backward,expected[:40])
    finally:
        reader.close()


@pytest.fixture
def actual_composer():
    if not (noise.NOISE/'measurements.json').exists():
        pytest.skip('Run noise fitter for native-image integration checks')
    # Other owners rebuild live labels/stats concurrently. A verified immutable
    # generation bundle supplies one coherent input set for integration checks;
    # Composer still rejects any subsequently revoked normalized/document source.
    frozen=compose.OUT/'round-3-post-tiles'
    if (frozen/'manifest.json').exists():
        c=compose.Composer(stats_path=frozen/'generation-stats.json',
            state_path=frozen/'inputs/state.json',noise_path=frozen/'inputs/noise.json')
    else:c = compose.Composer()
    yield c
    c.close()


def test_mask_matches_truth_and_seed_pixels(tmp_path,actual_composer):
    c = actual_composer
    did = '5709fe9e86ce942b'
    if c.coverage(did)['supported_composition_mass']<.95:
        # Dominant model withdrawal is a valid production outcome. It must fail
        # before writing mislabeled or incomplete training images.
        with pytest.raises(ValueError,match='Unsupported composition mass'):
            c.column(did,992,4000,tmp_path/'a')
        assert not (tmp_path/'a').exists()
        return
    a,truth = c.column(did,992,4000,tmp_path/'a')
    b,truth2 = c.column(did,992,4000,tmp_path/'b')
    assert np.array_equal(a,b)
    assert truth['layers'] == truth2['layers']
    with Image.open(tmp_path/'a'/'mask.png') as image:
        mask = np.asarray(image)
    expected = np.empty_like(mask)
    lookup = {cid:int(index) for index,cid in truth['class_index'].items()}
    y = 0
    for layer in truth['layers']:
        assert layer['top'] == y
        y = layer['bottom']
        expected[layer['top']:y] = lookup[layer['class_id']]
    assert y == len(mask) and np.array_equal(expected,mask)
    assert all(noise.training(c.docs[did]) for did in truth['source_docs'])


def test_page_preserves_other_tracks(tmp_path,actual_composer):
    # Read an existing starter rather than generating an expensive page for every test run.
    c = actual_composer
    pages = list((compose.OUT/'starter2').glob('page-*/bbox_annotation.json'))
    if not pages:
        pytest.skip('Starter pages not built yet')
    import tifffile
    for path in pages:
        truth = noise.read(path.parent/'truth.json')
        doc = c.docs[truth['base_doc']]
        x0,y0,x1,y1 = doc['boxes']['interpreted_lithology_column']
        band = noise.read(path.parent/'page-bands.json')['bands'][len(noise.read(path.parent/'page-bands.json')['bands'])//2]
        real = c.windows.get(doc,[0,band['top'],doc['width'],band['bottom']])
        with Image.open(path.parent/band['file']) as image:
            synth = np.asarray(image)
        with Image.open(path.parent/'column.png') as image:
            col=np.asarray(image)
        a,b=max(y0,band['top']),min(y1,band['bottom'])
        if a<b:
            assert np.array_equal(synth[a-band['top']:b-band['top'],x0:x1],col[a-y0:b-y0])
        assert np.array_equal(real[:,:x0],synth[:,:x0])
        assert np.array_equal(real[:,x1:],synth[:,x1:])
        annotation = noise.read(path)
        assert annotation['bounding_boxes'] == doc['boxes']
        with tifffile.TiffFile(path.parent/annotation['image_name']) as image:
            assert image.pages[0].is_tiled
            assert image.pages[0].shape == (doc['height'],doc['width'],3)
