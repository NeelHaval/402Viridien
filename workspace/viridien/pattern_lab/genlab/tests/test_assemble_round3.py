"""Adversarial source revocation, viewport and truth regressions from round 3."""
import copy
import hashlib
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from genlab import paths
from genlab.assemble import compose,noise,stats


@pytest.mark.parametrize('member',['mask','colours','residual','valid','residual_dtype','valid_dtype'])
def test_identical_refit_never_recertifies_changed_content(tmp_path,monkeypatch,member):
    monkeypatch.setattr(noise,'source_binding',lambda *a:dict(kind='machine-only-unconfirmed',normalized_sources=[]))
    tile=np.full((12,20,3),220,np.uint8);rgb=np.tile(tile,(3,1,1))
    args=(tmp_path,dict(key='d:2',doc_id='d',signature='sig'),'d',[0,0,20,36],'f',{'d'},rgb,tile)
    first=noise.save_measurement(*args)
    if member in ('mask','colours'):
        file=tmp_path/first[member]
        with Image.open(file) as im:a=np.asarray(im).copy()
        a.flat[0]^=1;Image.fromarray(a).save(file)
    else:
        file=tmp_path/first['residual']
        with np.load(file) as z:field=z['residual'].copy();valid=z['valid'].copy()
        if member=='residual':field.flat[0]=23
        elif member=='residual_dtype':field=field.astype(np.int16)
        elif member=='valid_dtype':valid=valid.astype(np.float32)
        else:valid.flat[0]=not valid.flat[0]
        np.savez_compressed(file,residual=field,valid=valid)
    tampered=file.read_bytes()
    with pytest.raises(ValueError,match='immutable'):noise.save_measurement(*args)
    assert file.read_bytes()==tampered
    with pytest.raises(ValueError,match='changed appearance asset'):
        noise.asset_path(tmp_path,first,'residual' if member in ('valid','residual_dtype','valid_dtype') else member)


def test_sparse_identity_and_wrong_period_colour_controls():
    sparse=np.full((200,54,3),255,np.uint8);sparse[20,8:38]=0;sparse[70,8:38]=0
    assert compose.native_geometry(sparse,sparse)['eligible']
    real=np.full((208,54,3),[0,255,0],np.uint8);real[::32,8:30]=0
    wrong=np.full_like(real,[0,255,0]);wrong[::52,8:30]=0
    assert not compose.native_geometry(real,wrong)['eligible']
    recolour=real.copy();recolour[recolour.max(2)>=100]=[255,0,255]
    assert not compose.native_geometry(real,recolour)['eligible']
    varied=np.minimum(real.astype(int)+3,255).astype(np.uint8)
    assert compose.native_geometry(real,varied)['eligible']


def viewport_sampler(seed=441):
    names=['A','B','C'];comp={'A':.25,'B':.5,'C':.25}
    chain={'A':{'B':.1,'C':.9},'B':{'A':.5,'C':.5},'C':{'A':.9,'B':.1}}
    block=dict(n_layers=100,zones=[],composition={'all':comp},transitions={'order1':chain},
        thickness={k:dict(samples_px=[100 if k=='B' else 10],sample_weights=[1.],complete_n=20,effective_n=20) for k in names})
    return compose.Sampler(dict(classes={k:dict(norm=k) for k in names},levels={'doc:d':block}),
        'd','f',{k:dict(pattern_source='fixture') for k in names},seed)


def test_arbitrary_viewports_have_row_occupancy_and_uniform_residuals():
    s=viewport_sampler();starts=[];ages=[]
    for _ in range(10000):
        l=s.sequence(7,start_frac=.4,max_layers=1)[0]
        starts.append(l['class_id'])
        assert l['top_censored'] and not l['boundary_top_confirmed']
        if l['class_id']=='B':ages.append(l['sampled_age_rows'])
    assert starts.count('B')/len(starts)==pytest.approx(.5,abs=.02)
    assert np.mean(ages)==pytest.approx(49.5,abs=2)
    assert len(set(ages))==100
    # Confirmed boundary starts retain the bed-entry distribution.
    starts=[s.sequence(7,max_layers=1,start_at_boundary=True)[0]['class_id'] for _ in range(5000)]
    assert starts.count('B')/len(starts)==pytest.approx(1/11,abs=.02)


@pytest.fixture
def isolated_composer():
    import model
    did='5709fe9e86ce942b';doc=model.Store().get(did);cid=did+':1'
    index=noise.read(paths.DATA/'normalized/current.json')
    c=compose.Composer.__new__(compose.Composer)
    c.docs={did:doc};c.fmap={doc['name']:'fixture'}
    block=dict(n_layers=20,composition={'all':{cid:1.}},transitions={'order1':{cid:{cid:1.}}},zones=[],
        thickness={cid:dict(samples_px=[100],sample_weights=[1.],complete_n=10,effective_n=10)},
        depth_mapping={},zone_scheme='all',complete_human_rows=0)
    c.stats=dict(schema='asm-2',classes={cid:dict(doc_id=did,legend_id=1,norm='shale',name='Shale')},
        levels={'doc:'+did:block,'family:fixture':dict(document_ids=[did]),'global':dict(source_wells=[])},
        inputs=dict(normalized_snapshot=index['snapshot']))
    c.models={cid:dict(kind='tile',tile=np.full((1,1,3),180,np.uint8),bg=np.array([180,180,180]))}
    c.cards={cid:dict(key=cid,doc_id=did,signature='fixture',ancestor_docs=[did])}
    c.context=dict(normalized=index,documents={did:doc});c.context_hash='fixture'
    c.stats_hash='fixture';c.stats_path=Path('<fixture>');c.realism=dict(measurements=[],rails={},boundaries={})
    c.rejected_cards={};c.rejected_stale_measurements=[];c.windows=noise.Windows();c.noise_root=Path('.')
    c.assert_context_current=lambda:None
    c.available=lambda did,mode='column':({cid:dict(pattern_id=cid,pattern_source='document')},[])
    c.diagnostic_sheets=lambda root:[];c.family_sheets=lambda root:[]
    yield c
    c.close()


def test_cropped_and_context_edges_remain_censored(tmp_path,isolated_composer):
    c=isolated_composer;did=next(iter(c.docs))
    _,t=c.column(did,61,7,tmp_path/'cut',page_top=20000,iteration=0)
    assert t['layers'][0]['top_censored'] and t['layers'][0]['bottom_censored']
    assert not t['layers'][0]['boundary_bottom_confirmed']
    peers=[r for r in c.context['normalized']['records'] if r['document_id']==did and r.get('usable_for_generation') and not r.get('quality_issues')]
    row=next(r for r in peers if len(r.get('targets',[]))==1 and r['targets'][0]['legend_id']==1 and
        r['box_xyxy'][3]-r['box_xyxy'][1]>=20 and stats.infer_box_kind(r,peers)=='bed')
    _,t=c.column(did,62,7,tmp_path/'context',page_top=row['box_xyxy'][1]+5,context_consistent=True,iteration=0)
    assert t['layers'][0]['top_censored'] and t['layers'][0]['bottom_censored']
    assert not t['layers'][0]['boundary_top_confirmed'] and not t['layers'][0]['boundary_bottom_confirmed']


def test_ambiguous_blends_have_visible_ignore_and_latent_identity(tmp_path,isolated_composer,monkeypatch):
    c=isolated_composer;did=next(iter(c.docs));first=did+':1';second=did+':5'
    c.stats['classes'][second]=dict(doc_id=did,legend_id=5,norm='siltstone',name='Siltstone')
    c.models[first]['tile'][:]=[220,80,40]
    c.models[second]=dict(kind='tile',tile=np.array([[[40,80,220]]],np.uint8))
    c.cards[second]=dict(key=second,doc_id=did,ancestor_docs=[did])
    choices={k:dict(pattern_id=k,pattern_source='document') for k in [first,second]}
    c.available=lambda did,mode='column':(choices,[])
    c.realism['boundaries']={'fixture':[dict(doc=did,line=False,antialias=1.,antialias_alpha=.5)]}
    monkeypatch.setattr(noise,'nuisance_allowed',lambda *args:True)
    layers=[dict(top=0,bottom=10,class_id=first,norm='shale',zone='all'),dict(top=10,bottom=20,class_id=second,norm='siltstone',zone='all')]
    with patch.object(compose.Sampler,'sequence',return_value=layers):
        # Add the second duration solely for the isolated column setup.
        c.stats['levels']['doc:'+did]['thickness'][second]=copy.deepcopy(c.stats['levels']['doc:'+did]['thickness'][first])
        rgb,t=c.column(did,1,20,tmp_path/'blend',page_top=20000)
    with Image.open(tmp_path/'blend/mask.png') as im:visible=np.asarray(im)
    with Image.open(tmp_path/'blend/latent-mask.png') as im:latent=np.asarray(im)
    assert (rgb[10]==[130,80,130]).all()
    assert (visible[10]==255).all() and (latent[10]==2).all()
    assert t['supervised_rows']==19 and t['supervised_pixels']==int((visible!=255).sum())


@pytest.mark.parametrize('mode',['column','page'])
def test_direct_partial_status_and_immutable_run(tmp_path,isolated_composer,monkeypatch,mode):
    c=isolated_composer;did=next(iter(c.docs))
    if mode=='page':c.page=lambda did,seed,folder,**kw:c.column(did,seed,30,folder,mode='page')
    monkeypatch.setattr(compose,'OUT',tmp_path)
    monkeypatch.setattr(compose,'Composer',lambda **kw:c)
    argv=['compose',mode,'--doc',did,'--run','immutable','--height','30','--iteration','0']
    monkeypatch.setattr(sys,'argv',argv);compose.main()
    manifest=noise.read(tmp_path/'immutable/manifest.json')
    assert manifest['status']=='partial' and len(manifest['outputs'])==1
    before={p.relative_to(tmp_path):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(sys,'argv',['compose','starter','--run','immutable'])
    with pytest.raises(ValueError,match='immutable'):compose.main()
    assert {p.relative_to(tmp_path):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==before


def test_unvalidated_scale_transform_is_withheld(tmp_path):
    assert compose.Composer.scale_variants(tmp_path,dict(depth_mapping={'rows_per_ft':4}))==[]
    assert noise.read(tmp_path/'scale-availability.json')['status']=='unavailable'
    assert not list(tmp_path.glob('scale-*/truth.json'))


def test_nuisance_revocation_and_unbound_evidence(monkeypatch):
    did='d';doc=dict(id=did,name='fixture');box=[0,0,20,10]
    class Store:
        def get(self,key):assert key==did;return doc
    monkeypatch.setattr(noise.model,'Store',Store)
    monkeypatch.setattr(noise,'training',lambda doc:True)
    index=dict(records=[])
    monkeypatch.setattr(noise,'read',lambda path:index)
    record=noise.evidence_sources(did,[box],doc)
    assert noise.nuisance_allowed(record)
    index['records'].append(dict(document_id=did,box_xyxy=[0,5,20,10],usable_for_generation=False,quality_issues=['duplicate_of_test']))
    assert not noise.nuisance_allowed(record)
    assert not noise.nuisance_allowed(dict(source_doc=did))
    # Unrelated independently permitted evidence remains usable.
    other=noise.evidence_sources(did,[[0,20,20,30]],doc)
    assert noise.nuisance_allowed(other)


def test_rail_publication_does_not_rehash_tampered_asset(tmp_path):
    Image.fromarray(np.zeros((8,1,3),np.uint8)).save(tmp_path/'rail.png')
    rail=dict(file='rail.png',asset_sha256={'file':hashlib.sha256((tmp_path/'rail.png').read_bytes()).hexdigest()})
    Image.fromarray(np.ones((8,1,3),np.uint8)).save(tmp_path/'rail.png')
    with pytest.raises(ValueError,match='changed appearance asset'):
        noise.publish_library(tmp_path,dict(measurements=[],rails={'d':rail}))


def test_local_reviewed_rulers_never_extrapolate():
    frozen=noise.read(compose.OUT/'round-3/inputs/context.json')['documents']
    for did,rate in [('97adb9d2c69d860b',1.5),('9da74bcde9aee789',2.),('b1411b677e888159',2.),('28112eef4c3c645a',10.)]:
        mapping=stats.depth_mapping(frozen[did])
        assert mapping['rows_per_ft'] is None and mapping['local_rows_per_ft']==pytest.approx(rate)
        for y,depth in mapping['anchors']:assert stats.depth_at_page_row(mapping,y,0)==pytest.approx(depth,abs=1e-7)
        assert stats.depth_at_page_row(mapping,0,0) is None


def test_changed_calibration_source_withholds_mapping_without_blocking_other_documents():
    frozen=noise.read(compose.OUT/'round-3/inputs/context.json')['documents']
    doc=copy.deepcopy(frozen['28112eef4c3c645a']);doc['revision']+=1
    mapping=stats.depth_mapping(doc)
    assert mapping['rows_per_ft'] is None and mapping['depth_top_ft'] is None
    assert mapping['calibration_status']=='source-revision-changed; review required'
    assert stats.depth_at_page_row(mapping,39963.5,doc['boxes']['interpreted_lithology_column'][1]) is None


def test_two_dimensional_insets_preserve_exact_host_mask(tmp_path,isolated_composer):
    c=isolated_composer;did=next(iter(c.docs));host=did+':1';inset=did+':5'
    c.stats['classes'][inset]=dict(doc_id=did,legend_id=5,norm='sandstone',name='Sandstone')
    c.models[host]['tile'][:]=[0,255,255]
    c.models[inset]=dict(kind='tile',tile=np.array([[[255,255,0]]],np.uint8))
    c.cards[inset]=dict(key=inset,doc_id=did,ancestor_docs=[did])
    c.available=lambda did,mode='column':({k:dict(pattern_id=k,pattern_source='fixture') for k in [host,inset]},[])
    block=c.stats['levels']['doc:'+did];block['thickness'][inset]=copy.deepcopy(block['thickness'][host])
    regions=[dict(box_xyxy=[1+(i%3)*8,10+i*8,32+(i%3)*8,17+i*8],class_id=inset,confirmed=True,paint_order=i) for i in range(9)]
    rgb,t=c.column(did,1,100,tmp_path/'insets',page_top=20000,iteration=0,regions=regions)
    with Image.open(tmp_path/'insets/mask.png') as im:mask=np.asarray(im)
    expected=np.ones_like(mask)
    for r in regions:
        a,b,d,e=r['box_xyxy'];expected[b:e,a:d]=2
        assert (rgb[b:e,a:d]==[255,255,0]).all()
    assert np.array_equal(mask,expected) and len(t['regions'])==9
    assert all((rgb[y][expected[y]==1]==[0,255,255]).all() for y in range(100))


def test_page_crossing_visibility_and_neighbor_tracks(tmp_path,isolated_composer,monkeypatch):
    import model
    c=isolated_composer;did=next(iter(c.docs));doc=c.docs[did]
    doc.update(width=64,height=1024)
    doc['boxes']['interpreted_lithology_column']=[10,10,50,1000]
    source=np.full((1024,64,3),245,np.uint8);source[500:502]=0
    class Windows:
        def get(self,doc,box):
            a,b,d,e=box;return source[b:e,a:d].copy()
        def close(self):pass
    c.windows.close();c.windows=Windows()
    monkeypatch.setattr(model,'DATASET',tmp_path/'source')
    annotation=tmp_path/'source/bbox_annotation'/(doc['name']+'.json')
    noise.write(annotation,dict(image_name='source.tif'))
    rgb,t=c.page(did,4,tmp_path/'page')
    with Image.open(tmp_path/'page/mask.png') as im:mask=np.asarray(im)
    with Image.open(tmp_path/'page/latent-mask.png') as im:latent=np.asarray(im)
    assert (rgb[490:492]==0).all() and (mask[490:492]==255).all()
    assert (latent[490:492]!=255).all()
    assert t['supervised_rows']==988
    from genlab.assemble import bench
    band=bench.page_column_window(tmp_path/'page',[0,0,64,1024],480,520)
    assert np.array_equal(band[:,:10],source[480:520,:10])
    assert np.array_equal(band[:,50:],source[480:520,50:])


@pytest.mark.parametrize('height',[2399,2400,2401,4800])
def test_quilt_height_limit_through_column(tmp_path,isolated_composer,height):
    c=isolated_composer;did=next(iter(c.docs));cid=did+':1'
    c.models[cid]=dict(kind='quilt',strips=[np.full((2400,57,3),180,np.uint8)],
        strip_origins=[525],bg=np.array([180,180,180]),max_layer_height=2400)
    item=dict(top=0,bottom=height,class_id=cid,norm='shale',zone='all')
    with patch.object(compose.Sampler,'sequence',return_value=[item]):
        if height>2400:
            with pytest.raises(ValueError,match='supported maximum'):
                c.column(did,1,height,tmp_path/str(height),page_top=20000,iteration=0)
            assert not (tmp_path/str(height)).exists()
        else:
            rgb,t=c.column(did,1,height,tmp_path/str(height),page_top=20000,iteration=0)
            assert len(rgb)==height


def test_bench_thin_profile_caps_family_conversion_and_viewport_draws():
    s=viewport_sampler()
    s.stats['bench_duration_profile']=dict(schema='bench-duration-1',mode='thin-beds',
        caps_by_class={cid:{'all':3} for cid in s.available})
    for arbitrary in (False,True):
        for _ in range(100):
            seq=s.sequence(200,start_at_boundary=not arbitrary)
            assert all(l['bottom']-l['top']<=3 for l in seq)
    # The same hook applies after a sparse local/physical family mixture.
    s=compose.Sampler.__new__(compose.Sampler);s.doc_id='d';s.mapping=dict(rows_per_ft=2,coordinate='MD')
    s.stats=dict(classes={'d:1':dict(norm='sand')},bench_duration_profile=dict(schema='bench-duration-1',mode='thin-beds',caps_by_class={'d:1':{'all':7}}))
    local=dict(n=1,effective_n=1,samples_px=[309.],sample_weights=[1.])
    family=dict(n=6,effective_n=6,samples_ft_by_coordinate={'MD':[10.,20.,30.,40.,50.,60.]},sample_weights_by_coordinate={'MD':[1.]*6})
    s.levels=[('doc:d',dict(thickness={'d:1':local})),('family:f',dict(thickness={'sand':family}))]
    level,_,d=s.duration_model('d:1','all')
    assert np.max(s.duration_values(level,d))==7 and d['shrinkage_local_weight']==pytest.approx(1/6)


def test_bench_thin_profile_refuses_zero_cap_and_excess_same_class_merges():
    s=viewport_sampler();s.stats['bench_duration_profile']=dict(schema='bench-duration-1',mode='thin-beds',
        caps_by_class={cid:{'all':0} for cid in s.available})
    with pytest.raises(ValueError,match='positive integer cap'):s.sequence(20)
    s.stats['bench_duration_profile']['caps_by_class']={cid:{'all':3} for cid in s.available}
    s.next=lambda history,zone:'A'
    with pytest.raises(ValueError,match='after adjacent equal-class merge'):s.sequence(20)
