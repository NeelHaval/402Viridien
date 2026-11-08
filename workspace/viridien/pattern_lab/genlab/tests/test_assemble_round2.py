"""Regression coverage for round-two assembler interface and provenance failures."""
import copy
import hashlib
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from genlab import learn, paths, render
from genlab.assemble import compose, noise, stats


def test_coordinate_weights_survive_fit_and_sampling():
    rows=[dict(key='a',top=0,bottom=h,height=2000,weight=w,rows_per_ft=1.,depth_coordinate=c)
          for h,w,c in [(10,1.,'MD'),(100,.1,'MD'),(1000,.2,'TVD')]]
    fit=stats.thickness_fit(rows)['a']
    s=compose.Sampler.__new__(compose.Sampler)
    s.mapping=dict(coordinate='MD',rows_per_ft=1.);s.used=Counter();s.rng=np.random.default_rng(11)
    s.duration_model=lambda cid,zone:('family:f','thickness',fit)
    draws=np.array([s.draw_thickness('family:f','thickness',fit) for _ in range(20000)])
    assert np.mean(draws==100)==pytest.approx(1/11,abs=.01)
    assert draws.mean()==pytest.approx(200/11,abs=1.)
    assert s.expected_thickness('a','all')==pytest.approx(200/11)
    fit['sample_weights_by_coordinate']['MD'].pop()
    with pytest.raises(ValueError,match='weights mismatch'):s.expected_thickness('a','all')


@pytest.mark.parametrize('band_rows',[0,-256,1,300,513])
def test_nonaligned_page_rejected_before_any_reads(tmp_path,band_rows):
    c=compose.Composer.__new__(compose.Composer)
    with pytest.raises(ValueError,match='positive multiple of 256'):
        c.page('never-read',1,tmp_path/'output',band_rows=band_rows)
    assert not (tmp_path/'output').exists()


def test_beds_and_history_cross_depth_edges():
    s=compose.Sampler.__new__(compose.Sampler)
    s.stats=dict(classes={'a':dict(norm='a'),'b':dict(norm='b')})
    s.available={'a':dict(pattern_source='document'),'b':dict(pattern_source='document')}
    s.zone=lambda fraction:'one' if fraction<.5 else 'two'
    seen=[]
    def next_class(history,zone):
        seen.append((list(history),zone));return 'b' if history else 'a'
    s.next=next_class;s.thickness=lambda cid,zone:73;s.gap=lambda height:0
    out=s.sequence(140,clip_to_zone=True)
    assert [l['bottom'] for l in out]==[73,140]
    assert seen==[([], 'one'),(['a'],'two')]
    s.zone=lambda fraction:'one' if fraction<.6 else 'two'
    assert [(l['top'],l['bottom'],l['class_id']) for l in s.sequence(140,clip_to_zone=True)]==[(l['top'],l['bottom'],l['class_id']) for l in out]


def test_native_repeat_check_withholds_52_row_green():
    real=np.full((208,54,3),[0,255,0],np.uint8);real[::32,8:30]=0
    fake=np.full_like(real,[0,255,0]);fake[::52,8:30]=0
    assert compose.native_geometry(real,real)['eligible']
    assert not compose.native_geometry(real,fake)['eligible']


def test_unknown_and_transitive_ancestry_fail_closed():
    unknown=dict(key='d:1',doc_id='d',source='agent-made')
    with pytest.raises(ValueError,match='Unknown proposal ancestry'):noise.ancestors(unknown,{'d:1':unknown})
    card=dict(unknown,lineage=[dict(doc_id='donor',lineage=[dict(doc_id='held')])])
    assert noise.ancestors(card,{card['key']:card})=={'d','donor','held'}
    card['donor']=dict(doc_id='missing',legend_id=1)
    with pytest.raises(ValueError,match='Unresolved donor ancestry'):noise.ancestors(card,{card['key']:card})


def test_relabelled_source_rejected():
    row=dict(document_id='d',box_xyxy=[0,0,20,30],accepted=True,usable_for_generation=True,
             quality_issues=[],targets=[dict(legend_id=2)],sample_id='id',annotation_version='v1')
    assert noise.source_binding('d',[0,0,20,30],'d:2',[row])['kind']=='normalized-target'
    row['targets']=[dict(legend_id=0)]
    with pytest.raises(ValueError,match='relabelled'):noise.source_binding('d',[0,0,20,30],'d:2',[row])


def test_changed_fit_has_new_identity_and_old_assets_survive(tmp_path,monkeypatch):
    monkeypatch.setattr(noise,'source_binding',lambda *a:dict(kind='machine-only-unconfirmed',normalized_sources=[]))
    tile=np.full((12,20,3),220,np.uint8);rgb=np.tile(tile,(3,1,1))
    card=dict(key='d:2',doc_id='d',signature='sig')
    first=noise.save_measurement(tmp_path,card,'d',[0,0,20,36],'f',{'d'},rgb,tile)
    old=(tmp_path/first['residual']).read_bytes();measure=noise.measure
    def changed(*args):
        p,m,c,r=measure(*args);r[0,0,0]=7;return p,m,c,r
    monkeypatch.setattr(noise,'measure',changed)
    second=noise.save_measurement(tmp_path,card,'d',[0,0,20,36],'f',{'d'},rgb,tile)
    assert first['id']!=second['id']
    assert (tmp_path/first['residual']).read_bytes()==old
    (tmp_path/second['residual']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='changed appearance asset'):noise.asset_path(tmp_path,second,'residual')


def test_empty_residual_pool_still_applies_rails(tmp_path,monkeypatch):
    monkeypatch.setattr(noise,'nuisance_allowed',lambda *args:True)
    pixels=np.zeros((8,1,3),np.uint8);Image.fromarray(pixels).save(tmp_path/'rail.png')
    c=compose.Composer.__new__(compose.Composer);c.fmap={'doc':'f'};c.docs={'d':dict(name='doc')}
    c.noise_root=tmp_path
    c.realism=dict(measurements=[],rails={'d':dict(columns=[0],file='rail.png',
        asset_sha256={'file':hashlib.sha256((tmp_path/'rail.png').read_bytes()).hexdigest()})})
    rgb=np.full((30,20,3),255,np.uint8)
    out,params,events=c.appearance('d',rgb,[],np.random.default_rng(1))
    assert (out[:,0]==0).all() and (out[:,1:]==255).all()
    assert params['status']=='partial-no-compatible-residuals'
    assert not any(e['kind']=='real-residual' for e in events)


@pytest.mark.parametrize('kind',['tile','quilt'])
def test_complete_export_and_snapshot_roundtrip(tmp_path,kind):
    if not hasattr(learn,'export_model'):pytest.skip('Waiting for FIX-TILES render export')
    rng=np.random.default_rng(51)
    data=dict(kind=kind,bg=np.array([255,255,255]))
    if kind=='tile':
        data.update(tile=rng.integers(0,255,(6,20,3),dtype=np.uint8),anchored=False,anchored_x=True,
                    anchored_y=False,x_repeat=False,x_origin=100,phase=(0,0))
    else:
        data['strips']=[rng.integers(0,255,(h,20,3),dtype=np.uint8) for h in (2400,2700)]
        data['strip_origins']=[100,100]
    with patch.object(paths,'OUT',tmp_path):
        payload=learn.export_model(data,tmp_path,'signature')
    card=dict(key='d:1',doc_id='d',signature='signature',render_model=payload)
    with patch.object(paths,'OUT',tmp_path):loaded=noise.load_model(card)
    if kind=='quilt':assert [a.shape for a in loaded['strips']]==[(2400,20,3),(2700,20,3)]
    c=compose.Composer.__new__(compose.Composer)
    c.models={'d:1':loaded};c.cards={'d:1':card};c.context=dict(normalized=dict(records=[],snapshot='s'))
    c.realism=dict(measurements=[],rails={});c.noise_root=tmp_path
    with patch.object(paths,'OUT',tmp_path):
        state=c.snapshot_inputs(tmp_path/'run')
        restored=noise.load_model(noise.read(state)['cards'][0],Path(state).parent)
    for height in (960,2400):
        for seed in range(8):
            a=render.layer(data,100,140,0,height,np.random.default_rng(seed))
            b=render.layer(restored,100,140,0,height,np.random.default_rng(seed))
            assert np.array_equal(a,b)


def test_sparse_duration_never_borrows_global():
    s=compose.Sampler.__new__(compose.Sampler);s.doc_id='d';s.family='f';s.mapping=dict(rows_per_ft=1,coordinate='MD')
    s.stats=dict(classes={'d:1':dict(norm='clay')})
    d=dict(n=1,complete_n=1,effective_n=1,samples_px=[309.],sample_weights=[1.])
    g=dict(d,samples_ft_by_coordinate={'MD':[100.,200.]},sample_weights_by_coordinate={'MD':[1.,1.]})
    s.levels=[('doc:d',dict(thickness={'d:1':d})),('global',dict(thickness={'clay':g}))]
    with pytest.raises(ValueError,match='No compatible complete-bed'):s.duration_model('d:1','all')


def test_sparse_duration_shrinks_to_compatible_family():
    s=compose.Sampler.__new__(compose.Sampler);s.doc_id='d';s.family='f';s.mapping=dict(rows_per_ft=2,coordinate='MD')
    s.stats=dict(classes={'d:1':dict(norm='sand')})
    local=dict(n=1,effective_n=1,samples_px=[309.],sample_weights=[1.])
    family=dict(n=6,effective_n=6,samples_ft_by_coordinate={'MD':[10.,20.,30.,40.,50.,60.]},
                sample_weights_by_coordinate={'MD':[1.]*6})
    s.levels=[('doc:d',dict(thickness={'d:1':local})),('family:f',dict(thickness={'sand':family}))]
    level,field,d=s.duration_model('d:1','all')
    assert field=='thickness-family-shrinkage' and d['fallback_level']=='family:f'
    values,weights=s.duration_distribution(level,d)
    assert len(np.unique(values))==7 and weights[0]==pytest.approx(1/6)
    assert np.average(values,weights=weights)==pytest.approx(309/6+70*5/6)


def test_marl_residual_is_not_pasted_into_shale():
    c=compose.Composer.__new__(compose.Composer);c.family=lambda did:'f'
    m=dict(id='marl',doc_id='d',family='f',source_doc='d',blur=0.,drift=[0,0,0],noise_sigma=0.,
           overlay_usable=True,overlay_fraction=.2,jpeg_block_phase_energy=0)
    c.realism=dict(measurements=[m],rails={},boundaries={})
    rgb=np.full((200,54,3),180,np.uint8)
    out,params,events=c.appearance('d',rgb.copy(),[dict(class_id='d:shale',top=0,bottom=200)],np.random.default_rng(1),1)
    assert np.array_equal(out,rgb)
    assert not events and not params['unrestricted_overlays']
    crossing=np.full((50,66,3),255,np.uint8);crossing[25]=0
    assert c.crossing_mask(crossing)[25].all()


def test_frozen_context_rechecks_revocations(monkeypatch):
    row=dict(sample_id='s',document_id='d',usable_for_generation=True,targets=[dict(legend_id=1)])
    c=compose.Composer.__new__(compose.Composer);c.docs={'d':dict(id='d')};c.frozen_context=True
    c.context=dict(normalized=dict(snapshot='frozen',records=[row]))
    monkeypatch.setattr(noise,'training',lambda doc:True)
    monkeypatch.setattr(noise,'read',lambda path:dict(snapshot='frozen',records=[copy.deepcopy(row)]))
    c.assert_context_current()
    changed=copy.deepcopy(row);changed['targets'][0]['legend_id']=2
    monkeypatch.setattr(noise,'read',lambda path:dict(snapshot='new',records=[changed]))
    with pytest.raises(ValueError,match='Labels changed|revoked or relabelled'):c.assert_context_current()


def test_stale_statistics_snapshot_refused():
    s=noise.read(noise.ROOT/'stats.json');s['inputs']['normalized_snapshot']='old-snapshot-probe'
    with pytest.raises(ValueError,match='snapshot mismatch'):compose.Composer(stats=s,realism=dict(measurements=[]))


def test_stats_watcher_hashes_candidates_and_semantic_families(tmp_path,monkeypatch):
    root=tmp_path/'root';output=root/'data/genlab/assemble'
    monkeypatch.setattr(stats,'ROOT',root);monkeypatch.setattr(stats,'OUT',output)
    files=['data/normalized/current.json','data/normalized/split-policy.json','data/genlab/generated/manifest.json',
           '../bakeoff/labeltool/data/prefill.jsonl','data/genlab/candidates/d.json']
    for name in files:
        file=root/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('{}',encoding='utf-8')
    metadata=output/'stats/mix-metadata.json';noise.write(metadata,dict(mapping={'log':'one'},fetched_at=1))
    before=stats.watch_signature()
    (root/files[-1]).write_text('{"tier":"excluded"}',encoding='utf-8')
    after=stats.watch_signature();assert before!=after
    noise.write(metadata,dict(mapping={'log':'one'},fetched_at=2));assert stats.watch_signature()==after
    noise.write(metadata,dict(mapping={'log':'two'},fetched_at=2));assert stats.watch_signature()!=after


def test_quarantined_window_refused_before_decoder(monkeypatch):
    monkeypatch.setattr(noise,'require_training',lambda doc:None)
    monkeypatch.setattr(noise,'crop_allowed',lambda *a:False)
    with pytest.raises(ValueError,match='Quarantined'):noise.Windows().get(dict(id='d'),[0,0,10,10])


@pytest.mark.parametrize('context',[False,True])
def test_column_truth_routes_supervision_and_change(tmp_path,monkeypatch,context):
    from genlab.tests.test_assemble_compose import fixed_stats
    import model
    from types import SimpleNamespace
    did='5709fe9e86ce942b';doc=model.Store().get(did)
    s=fixed_stats();s['levels']['doc:'+did]=s['levels'].pop('doc:d')
    s['levels']['global']=dict(source_wells=[],thickness={})
    s['inputs']=dict(normalized_snapshot='fixture')
    for item in s['classes'].values():item['doc_id']=did
    choices={k:dict(pattern_source='document',pattern_id=k) for k in s['classes']}
    c=compose.Composer.__new__(compose.Composer)
    c.docs={did:doc};c.stats=s;c.stats_path=tmp_path/'stats.json';c.stats_hash='fixture'
    c.context=dict(normalized=dict(records=[]));c.context_hash='fixture'
    c.assert_context_current=lambda:None;c.available=lambda *a:(choices,[])
    c.coverage=lambda d:dict(supported_composition_mass=1.);c.family=lambda d:'fixture'
    c.models={k:dict(kind='tile',tile=np.full((10,57,3),50+i*50,np.uint8)) for i,k in enumerate(choices)}
    c.cards={k:dict(doc_id=did,ancestor_docs=[did],signature='fixture') for k in choices}
    c.realism=dict(measurements=[],rails={},boundaries={});c.rejected_stale_measurements=[]
    c.windows=SimpleNamespace(get=lambda doc,box:np.full((box[3]-box[1],box[2]-box[0],3),255,np.uint8))
    monkeypatch.setattr(noise,'require_training',lambda doc:None)
    rgb,t=c.column(did,12,1000,tmp_path/'column',context_consistent=context)
    assert t['supervised_rows']==(0 if context else 1000)
    assert t['changed_pixel_fraction']==(0 if context else 1.)
    assert t['localization_only']==context and t['lithology_augmentation']!=context
    mask=np.asarray(Image.open(tmp_path/'column/mask.png'))
    for layer in t['layers']:
        value=next(int(i) for i,k in t['class_index'].items() if k==layer['class_id'])
        assert (mask[layer['top']:layer['bottom']]==value).all()
