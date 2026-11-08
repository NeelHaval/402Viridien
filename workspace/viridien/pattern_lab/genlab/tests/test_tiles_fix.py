"""Regression checks and reproducible FIX-TILES artifact audit.

python -X utf8 -m genlab.tests.test_tiles_fix audit before|after
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from genlab import candidates, learn, paths, render, score


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def test_vote_never_invents_rgb():
    a = np.zeros((2,4,5,3),np.uint8);a[1]=192
    vote = learn.supported_vote(a)
    assert set(np.unique(vote)) <= {0,192}


def test_coloured_blank_controls():
    a = np.full((24,24,3),240,np.uint8);a[4:8,4:8]=(80,150,220)
    blank = dict(tile=np.full_like(a,240),bg=np.array([240]*3))
    assert candidates.pixel_match(a,blank)>.9
    assert candidates.symbol_match(a,blank)==0
    assert not score.fidelity(a,blank)['acceptable']


def test_antialiased_rail_and_genuine_stripes():
    from genlab.swatch import remove_frame
    a = np.full((60,40,3),255,np.uint8);a[5::10,4:35]=100;a[:,0]=119
    clean,trims = remove_frame(a)
    assert trims[0]==1
    assert not ((clean.max(2)<150).mean(0)>.95).any()
    stripes = np.full((60,40,3),230,np.uint8);stripes[::6]=0
    clean,trims = remove_frame(stripes)
    assert clean.shape==stripes.shape and not any(trims)


def test_reserve_geometry_before_pixels():
    boxes = [[0,0,20,100],[0,20,20,60],[0,1000,20,1100],[0,1020,20,1080]]
    exs = [dict(id=str(i),box=b) for i,b in enumerate(boxes)]
    train,held = learn.reserve_evidence(exs)
    assert len(train)==2 and len(held)==2
    assert all(not learn.boxes_overlap(a['box'],b['box'],512) for a in train for b in held)


def test_unmeasured_x_is_not_wrapped_or_randomized():
    tile=np.full((6,20,3),255,np.uint8);tile[:,0]=119
    data=dict(kind='tile',tile=tile,anchored=False,anchored_x=True,anchored_y=False,
              x_repeat=False,x_origin=100,phase=(0,0))
    for seed in range(8):
        a=render.layer(data,100,140,0,50,np.random.default_rng(seed))
        assert not (a[:,1:].max(2)<150).any()


def test_jitter_and_texture_keep_native_glyphs_and_vary():
    a=np.full((30,30,3),255,np.uint8)
    for y in (5,15,25):
        for x in (5,15,25):
            a[y:y+2,x:x+2]=0
    for kind in ('jitter','texture'):
        data=render.glyph_model(a,kind,spacing=(10,10))
        first=render.layer(data,0,100,0,240,np.random.default_rng(7))
        second=render.layer(data,0,100,0,240,np.random.default_rng(11))
        assert not np.array_equal(first,second)
        assert set(np.unique(first))=={0,255}
        assert not np.array_equal(first[:-30],first[30:])


def test_repeat_support_rejects_drifting_geometry():
    rng=np.random.default_rng(0)
    a=np.full((120,40,3),220,np.uint8);a[rng.random((120,40))<.2]=0
    assert not learn.repeat_supported(a,12)
    b=np.tile(a[:12],(10,1,1))
    assert learn.repeat_supported(b,12)


def test_short_regular_dots_do_not_become_jitter(monkeypatch):
    rgb=np.full((30,40,3),220,np.uint8)
    for y in (5,15,25):
        for x in (5,15,25,35):
            rgb[y:y+2,x:x+2]=0
    class Crops:
        def get(self,doc,box):
            return rgb
    ex=dict(id='a',doc_id='d',legend_id=1,label='Fine sandstone',box=[0,0,40,30],flagged=False)
    monkeypatch.setattr(learn,'estimate_period',lambda rgb,axis:dict(pixels=23))
    model,data=learn.learn(Crops(),{'d':dict(doc=dict(id='d'))},('d',1),[ex])
    assert model.get('renderer')!='jittered lattice'


def test_validation_donor_is_withheld():
    policy=read(paths.DATA/'normalized'/'split-policy.json')
    doc=dict(id='aeb414a4367bafd4',name='donor')
    assert learn.generation_split(doc,policy)[1]=='validation'
    assert not learn.lineage_allowed(dict(donor=dict(doc_id=doc['id'])),{doc['id']:dict(doc=doc)},policy)


def test_validation_report_cannot_enter_donor_pool(tmp_path,monkeypatch):
    monkeypatch.setattr(paths,'VERDICTS',tmp_path)
    monkeypatch.setattr(candidates,'reports',lambda:iter([dict(doc_id='952caeadf1d8467c',name='held donor',
        version=candidates.VERSION,training_allowed=True,entries=[dict(tier='human',legend_id=1,data_file='must-not-open.npz')])]))
    reads=[]
    monkeypatch.setattr(candidates,'load_data',lambda p:reads.append(p))
    assert candidates.donor_pool()==[] and reads==[]


def test_transfer_leaves_documents_out():
    pairs=[dict(doc_id=did,family='f',period_delta=-1,stroke_delta=0,colour_delta=[0,0,0],lineage={})
           for did in ('a','b','c') for _ in range(3)]
    result=learn.fit_transfer(pairs)['families']['f']
    assert result['usable'] and len(result['lodo'])==3
    assert result['lodo_before'][0]==1 and result['lodo_after'][0]==0
    json.dumps(learn.fit_transfer(pairs))
    pairs[0]['period_delta']=100
    json.dumps(learn.fit_transfer(pairs))


def test_lineage_revision_and_pixel_changes_revoke(tmp_path,monkeypatch):
    policy=dict(seed='example',document_groups={})
    doc=next(dict(id=str(i),name='well'+str(i),revision=2) for i in range(100)
             if learn.generation_split(dict(id=str(i),name='well'+str(i)),policy)[1]=='train')
    docs={doc['id']:dict(doc=doc)}
    monkeypatch.setattr(paths,'OUT',tmp_path)
    tile=tmp_path/'tile.png';Image.new('RGB',(4,4),'white').save(tile)
    source=dict(doc_id=doc['id'],source_revision=2,tile_path='tile.png',
                tile_sha256=hashlib.sha256(tile.read_bytes()).hexdigest(),boxes=[[0,0,4,4]],
                pixel_sha256=[hashlib.sha256(b'4x4:'+np.full((4,4,3),255,np.uint8).tobytes()).hexdigest()])
    proposal=dict(recovery=dict(lineage=[source]))
    assert learn.lineage_allowed(proposal,docs,policy)
    doc['revision']=3
    assert not learn.lineage_allowed(proposal,docs,policy)
    doc['revision']=2;Image.new('RGB',(4,4),'black').save(tile)
    assert not learn.lineage_allowed(proposal,docs,policy)


def test_model_roundtrip_preserves_bank_and_axes(tmp_path,monkeypatch):
    monkeypatch.setattr(paths,'OUT',tmp_path)
    tile=np.full((6,20,3),255,np.uint8);tile[:,0]=119
    tiledata=dict(kind='tile',tile=tile,bg=learn.background(tile),phase=(0,0),anchored=False,
                  anchored_x=True,anchored_y=False,x_repeat=False,x_origin=100)
    rng=np.random.default_rng(13)
    quilt=dict(kind='quilt',strips=[rng.integers(0,255,(2400,40,3),dtype=np.uint8),
                                  rng.integers(0,255,(1700,40,3),dtype=np.uint8)],bg=np.array([255]*3),strip_origins=[100,100])
    for data in (tiledata,quilt):
        payload=learn.export_model(data,tmp_path,'signature')
        loaded=learn.load_model(payload,tmp_path,'signature')
        for height in (240,960,2400):
            for seed in range(8):
                if data['kind']=='quilt' and height>1700:
                    import pytest
                    for d in (data,loaded):
                        with pytest.raises(ValueError,match='supported maximum'):render.layer(d,100,140,0,height,np.random.default_rng(seed))
                    continue
                a=render.layer(data,100,140,0,height,np.random.default_rng(seed))
                b=render.layer(loaded,100,140,0,height,np.random.default_rng(seed))
                assert np.array_equal(a,b)
                if data['kind']=='tile':assert not (b[:,1:].max(2)<150).any()
        if data['kind']=='quilt':assert [len(x) for x in loaded['strips']]==[2400,1700]
        import pytest
        with pytest.raises(ValueError,match='signature'):learn.load_model(payload,tmp_path,'wrong')
        asset=tmp_path/payload['asset']['path'];asset.write_bytes(asset.read_bytes()+b'changed')
        with pytest.raises(ValueError,match='hash'):learn.load_model(payload,tmp_path)


def test_lineage_unknown_explicit_held_and_cycle_fail_closed():
    import model
    store=model.Store();docs={d['id']:store.get(d['id']) for d in store.catalog}
    assert not learn.lineage_allowed({},docs)
    cycle={};cycle['lineage']=[cycle]
    assert not learn.lineage_allowed(cycle,docs)
    rows=read(paths.DATA/'normalized/current.json')['records']
    bad=next(r for r in rows if 'duplicate_of_test' in r.get('quality_issues',[]))
    source=dict(doc_id=bad['document_id'],source_revision=docs[bad['document_id']]['revision'],
                boxes=[bad['box_xyxy']],pixel_sha256=[bad['pixel_sha256']])
    assert not learn.lineage_allowed(dict(lineage=[source]),docs)
    good=next(r for r in rows if r.get('usable_for_generation') and learn.generation_split(docs[r['document_id']])[1]=='train')
    source=dict(doc_id=good['document_id'],source_revision=docs[good['document_id']]['revision'],
                boxes=[good['box_xyxy']],pixel_sha256=[good['pixel_sha256']],sample_id=good['sample_id'],annotation_version=good['annotation_version'])
    p=dict(lineage=[source])
    assert learn.lineage_allowed(p,docs)
    p['normalized_sources']=[dict(sample_id='accepted-1',annotation_version='v1')]
    assert not learn.lineage_allowed(p,docs)
    p.pop('normalized_sources')
    p['ancestor_docs']=[next(d for d in docs if learn.generation_split(docs[d])[1]=='test')]
    assert not learn.lineage_allowed(p,docs)


def test_proposal_roundtrip_preserves_recursive_ancestry_and_stochastic_kind(tmp_path,monkeypatch):
    import model
    store=model.Store();docs={d['id']:store.get(d['id']) for d in store.catalog}
    row=next(r for r in read(paths.DATA/'normalized/current.json')['records'] if r.get('usable_for_generation') and
             learn.generation_split(docs[r['document_id']])[1]=='train')
    did=row['document_id'];lid=row['targets'][0]['legend_id']
    source=dict(doc_id=did,legend_id=lid,source_revision=docs[did]['revision'],
        boxes=[row['box_xyxy']],pixel_sha256=[row['pixel_sha256']],
        sample_id=row['sample_id'],annotation_version=row['annotation_version'])
    glyph=np.zeros((2,2,3),np.uint8)
    data=dict(kind='texture',glyphs=[glyph],bg=np.array([255]*3),density=.1,large_density=0,small_density=.1)
    p=dict(doc_id=did,legend_id=lid,label='Roundtrip',round=1,lineage=[source],
           donor=dict(doc_id=did),recovery=dict(tile_is_preview_only=True,lineage=[source]),ancestor_docs=[did])
    monkeypatch.setattr(paths,'OUT',tmp_path)
    p['render_model']=learn.export_model(data,tmp_path,'proposal',p)
    m,loaded=learn.load_proposal(p,[],docs)
    assert loaded['kind']=='texture'
    assert m['lineage']==p['lineage'] and m['recovery']==p['recovery'] and m['donor']==p['donor']
    assert loaded['lineage']==p['lineage'] and loaded['ancestor_docs']==[did]
    a=render.layer(loaded,0,96,0,720,np.random.default_rng(7))
    b=render.layer(loaded,0,96,0,720,np.random.default_rng(11))
    assert not np.array_equal(a,b) and not np.array_equal(a[:-240],a[240:])
    import pytest
    p.pop('render_model')
    with pytest.raises(ValueError,match='Preview-only'):learn.load_proposal(p,[],docs)
    p['ancestor_docs']=[next(d for d in docs if learn.generation_split(docs[d])[1]=='test')]
    with pytest.raises(ValueError,match='ancestry'):learn.load_proposal(p,[],docs)


def test_texture_keeps_complete_large_motif_and_separate_ink_budgets():
    rgb=np.full((90,100,3),255,np.uint8)
    rgb[8:40,8:44]=128
    for y,x in ((60,10),(70,30),(65,70)):rgb[y:y+2,x:x+2]=0
    d=render.glyph_model(rgb)
    assert max(max(g.shape[:2]) for g in d['glyphs'])==36
    assert d['glyph_evidence']['morphology_supported']
    for seed in (7,11,41,42):
        out=render.layer(d,0,96,0,240,np.random.default_rng(seed))
        grey=float((out==128).all(2).mean())
        assert abs(grey-1152/9000)<.04
        assert not np.array_equal(out[:-60],out[60:])


def test_embedded_ancestry_refused_before_any_asset_load(tmp_path,monkeypatch):
    import copy,model,pytest
    store=model.Store();docs={e['id']:store.get(e['id']) for e in store.catalog}
    row=next(r for r in read(paths.DATA/'normalized/current.json')['records'] if r.get('usable_for_generation') and
             learn.generation_split(docs[r['document_id']])[1]=='train')
    did=row['document_id'];lid=row['targets'][0]['legend_id']
    source=dict(doc_id=did,legend_id=lid,source_revision=docs[did]['revision'],boxes=[row['box_xyxy']],pixel_sha256=[row['pixel_sha256']])
    p=dict(doc_id=did,legend_id=lid,label='fixture',round=1,lineage=[source],ancestor_docs=[did])
    monkeypatch.setattr(paths,'OUT',tmp_path)
    tile=np.full((9,13,3),240,np.uint8);tile[2:5,3:8]=0
    p['render_model']=learn.export_model(dict(kind='tile',tile=tile,bg=learn.background(tile)),tmp_path,'fixture',p)
    m,d=learn.load_proposal(p,[],docs)
    assert np.array_equal(d['tile'],tile)
    bad=copy.deepcopy(p)
    bad['render_model']['provenance']['ancestor_docs']=[next(d for d in docs if learn.generation_split(docs[d])[1]=='test')]
    opened=[];monkeypatch.setattr(learn,'load_model',lambda *a,**k:opened.append(1))
    with pytest.raises(ValueError,match='ancestry'):learn.load_proposal(bad,[],docs)
    assert not opened
    d['ancestor_docs']=bad['render_model']['provenance']['ancestor_docs']
    with pytest.raises(ValueError,match='ancestry'):learn.export_model(d,tmp_path,'launder',m)


def test_metadata_binding_covers_placement_and_ancestry(tmp_path,monkeypatch):
    import copy,pytest
    monkeypatch.setattr(paths,'OUT',tmp_path)
    p=learn.export_model(dict(kind='tile',tile=np.full((4,6,3),255,np.uint8),bg=np.array([255]*3)),tmp_path,'fixture')
    for field in ('placement','provenance'):
        q=copy.deepcopy(p)
        if field=='placement':q[field]['x_repeat']=False
        else:q[field]['ancestor_docs']=['new-source']
        with pytest.raises(ValueError,match='binding'):learn.load_model(q)


def test_quilt_native_coordinates_extent_and_finite_height(tmp_path,monkeypatch):
    import pytest
    monkeypatch.setattr(paths,'OUT',tmp_path)
    strip=np.full((2400,8,3),255,np.uint8);strip[:,:,0]=np.arange(8)*20
    data=dict(kind='quilt',strips=[strip],strip_origins=[102],native_extent=[102,110],bg=np.array([255]*3))
    payload=learn.export_model(data,tmp_path,'quilt');loaded=learn.load_model(payload)
    assert loaded['max_layer_height']==2400 and loaded['strip_origins']==[102]
    for h in (2399,2400):
        a=render.layer(loaded,100,112,0,h,np.random.default_rng(7))
        assert np.array_equal(a[:,2:10],strip[:h]) and (a[:,:2]==255).all() and (a[:,10:]==255).all()
        b=render.layer(loaded,101,113,0,h,np.random.default_rng(7))
        assert np.array_equal(a[:,1:],b[:,:-1])
    for h in (2401,4800):
        with pytest.raises(ValueError,match='supported maximum'):render.layer(loaded,100,112,0,h,np.random.default_rng(7))
    legacy=dict(kind='quilt',strips=[strip],bg=np.array([255]*3))
    restored=learn.load_model(learn.export_model(legacy,tmp_path,'legacy'))
    assert restored['quilt_local']
    assert np.array_equal(render.layer(legacy,100,108,0,240,np.random.default_rng(1)),
                          render.layer(restored,100,108,0,240,np.random.default_rng(1)))


def test_weighted_component_tail_rejects_damage_but_accepts_clean_dashes():
    rgb=np.full((240,64,3),220,np.uint8)
    for y in range(8,232,12):
        for x in (5,25,45):rgb[y:y+3,x:x+14]=0
    clean=rgb.copy();damaged=rgb.copy()
    for y in range(8,112,12):damaged[y-5:y,9:12]=0
    assert score.component_comparison(rgb,clean)['acceptable']
    result=score.component_comparison(rgb,damaged)
    assert not result['acceptable'] and result['upper_tail_ink_excess'][0]>.1


def test_source_isolation_does_not_confuse_connected_fills_with_crop_height():
    class Crops:
        def get(self,doc,box):
            a=np.full((box[3]-box[1],box[2]-box[0],3),230,np.uint8)
            a[:,18:21]=0
            return a
    exs=[dict(id=str(i),box=b,flagged=False,label='Connected fill') for i,b in enumerate([[0,0,40,48],[0,200,40,296]])]
    m,d=learn.learn(Crops(),{'d':dict(doc=dict(id='d'))},('d',1),exs)
    assert not any('Component ink distribution' in s['reason'] for s in m['suspects'])


def test_native_width_bank_validation_accepts_clean_and_finds_late_damage():
    rgb=np.full((240,64,3),220,np.uint8)
    for y in range(8,232,12):
        for x in (5,25,45):rgb[y:y+3,x:x+14]=0
    class Crops:
        def get(self,doc,box):return rgb.copy()
    examples=[dict(id=i,box=b) for i,b in [('fit-a',[0,0,64,240]),('fit-b',[0,800,64,1040]),('held',[0,2000,64,2240])]]
    model=dict(sources=['fit-a','fit-b'],training_boxes=[e['box'] for e in examples[:2]],heldout_ids=['held'],heldout_boxes=[examples[-1]['box']])
    strip=np.tile(rgb,(10,1,1));data=dict(kind='quilt',strips=[strip,strip.copy()],strip_origins=[0,0],bg=np.array([220]*3))
    good=score.realism(model,data,Crops(),dict(id='d'),examples)
    assert good['acceptable'] and good['scored_width']==good['emitted_width']==64
    assert len(good['bank_morphology'])==2 and all(len(b['windows'])==19 for b in good['bank_morphology'])
    for y in range(1808,1928,12):
        for x in (5,25,45):data['strips'][1][y-5:y,x+4:x+7]=0
    bad=score.realism(model,data,Crops(),dict(id='d'),examples)
    assert not bad['acceptable'] and not bad['bank_morphology'][1]['acceptable']


def test_recovery_translation_roundtrip_retains_nonrepeat_rails(tmp_path,monkeypatch):
    monkeypatch.setattr(paths,'OUT',tmp_path)
    tile=np.full((6,20,3),255,np.uint8);tile[:,0]=119
    data=dict(kind='tile',tile=tile,bg=learn.background(tile),phase=(0,0),anchored_x=True,anchored_y=False,x_repeat=False,x_origin=100)
    card=dict(signature='donor',render_model=learn.export_model(data,tmp_path,'donor'))
    translated=candidates.recovery_donor(card,300,100)
    loaded=learn.load_model(learn.export_model(translated,tmp_path,'recipient'))
    assert loaded['x_origin']==300 and not loaded['x_repeat']
    for seed in range(8):
        a=render.layer(data,100,140,0,50,np.random.default_rng(seed))
        b=render.layer(loaded,300,340,0,50,np.random.default_rng(seed))
        assert np.array_equal(a,b) and not (b[:,1:].max(2)<150).any()


def test_recovery_donor_evidence_survives_preview_pruning(tmp_path,monkeypatch):
    import model
    docs={e['id']:model.Store().get(e['id']) for e in model.Store().catalog}
    row=next(r for r in read(paths.DATA/'normalized/current.json')['records'] if r.get('usable_for_generation') and
             learn.generation_split(docs[r['document_id']])[1]=='train')
    did=row['document_id'];lid=row['targets'][0]['legend_id']
    source=dict(doc_id=did,legend_id=lid,source_revision=docs[did]['revision'],boxes=[row['box_xyxy']],pixel_sha256=[row['pixel_sha256']])
    monkeypatch.setattr(paths,'OUT',tmp_path)
    original=tmp_path/'generated/preview.png';original.parent.mkdir();Image.new('RGB',(4,4),'white').save(original)
    data=dict(kind='tile',tile=np.full((4,4,3),255,np.uint8),bg=np.array([255]*3))
    frozen=candidates.freeze_donor_tile(dict(key=f'{did}:{lid}',signature='donor'),data)
    original.unlink()
    evidence=dict(source,**frozen)
    assert learn.lineage_allowed(dict(lineage=[evidence]),docs)
    assert (tmp_path/frozen['tile_path']).is_file()


def test_cold_crop_never_uses_page_fallback(tmp_path,monkeypatch):
    import model,tifffile,pytest
    root=tmp_path/'dataset';(root/'images').mkdir(parents=True)
    pixels=np.arange(120*30*3,dtype=np.uint16).reshape(120,30,3).astype(np.uint8)
    tifffile.imwrite(root/'images'/'safe.tif',pixels,rowsperstrip=16,photometric='rgb')
    tifffile.imwrite(root/'images'/'unsafe.tif',pixels,rowsperstrip=120,photometric='rgb')
    # Make the unsafe layout exceed the production strip threshold via metadata
    # on a real single-strip page, without allocating a large source image.
    monkeypatch.setattr(paths,'CROPS',tmp_path/'crops');monkeypatch.setattr(model,'DATASET',root)
    class Forbidden:
        def crop(self,*a):raise AssertionError('whole-page fallback called')
    crops=learn.Crops(Forbidden());doc=dict(id='safe',image_name='safe.tif')
    box=[3,9,13,43]
    assert np.array_equal(crops.get(doc,box),pixels[9:43,3:13])
    original=tifffile.TiffFile
    class Unsafe:
        def __enter__(self):
            self.tf=original(root/'images'/'unsafe.tif');self.pages=self.tf.pages
            self.pages[0].rowsperstrip=5000
            return self
        def __exit__(self,*a):self.tf.close()
    monkeypatch.setattr(tifffile,'TiffFile',lambda *a,**k:Unsafe())
    with pytest.raises(ValueError,match='Unsafe TIFF'):crops.get(dict(id='unsafe',image_name='unsafe.tif'),box)


def test_normalized_containing_native_crop_is_verified_before_subcrop(tmp_path,monkeypatch):
    import pytest
    from normalized import version,sample_id
    monkeypatch.setattr(paths,'DATA',tmp_path);monkeypatch.setattr(paths,'CROPS',tmp_path/'crops')
    root=tmp_path/'normalized';(root/'objects').mkdir(parents=True)
    rgb=np.full((120,30,3),220,np.uint8);rgb[10:25,6:9]=0
    Image.fromarray(rgb).save(root/'objects/native.png')
    segment=dict(id='bed',box=[0,0,30,120],legend_ids=[1],status='approved')
    legend=dict(id=1,label_printed='fixture',box=[0,0,5,5])
    doc=dict(id='d',revision=2,image_name='must-not-open.tif',segments=[segment],legend=[legend])
    digest=hashlib.sha256(b'30x120:'+rgb.tobytes()).hexdigest()
    row=dict(document_id='d',box_xyxy=segment['box'],native_png='objects/native.png',
        annotation_id='bed',sample_id=sample_id(doc,segment),annotation_version=version(segment),source_revision=1,
        legend_version=hashlib.sha256(json.dumps([legend],sort_keys=True).encode()).hexdigest()[:16],
        pixel_sha256=digest,usable_for_generation=True,split='train')
    index=root/'current.json';index.write_text(json.dumps(dict(records=[row])),encoding='utf-8')
    crop=learn.Crops(object()).get(doc,[3,9,13,43])
    assert np.array_equal(crop,rgb[9:43,3:13])
    denied=dict(row,document_id='test',sample_id='test',split='test',usable_for_generation=False,
        pixel_sha256=hashlib.sha256(b'10x34:'+crop.tobytes()).hexdigest())
    index.write_text(json.dumps(dict(records=[row,denied])),encoding='utf-8')
    with pytest.raises(ValueError,match='Quarantined'):learn.Crops(object()).get(doc,[3,9,13,43])


def test_repeat_preview_matches_native_placement_outside_marker():
    tile=np.full((7,8,3),230,np.uint8);tile[2:5,2:4]=0
    data=dict(kind='tile',tile=tile,bg=learn.background(tile),phase=(2,0),anchored_x=True,anchored_y=True,x_repeat=False,x_origin=102)
    a=render.repeat_preview(data,16,False,x0=100)
    b=render.layer(dict(data,phase=(0,0)),100,116,0,len(a),np.random.default_rng(0))
    keep=np.any(a!=[195,45,45],2)
    assert np.array_equal(a[keep],b[keep])


def test_asm_column_forwards_finite_quilt_height(tmp_path,monkeypatch):
    import model,pytest
    from types import SimpleNamespace
    from genlab.assemble import compose
    doc=model.Store().get('1bac8dda58a39eaf');did=doc['id'];cid=did+':4'
    x0,y0,x1,y1=doc['boxes']['interpreted_lithology_column']
    d=dict(kind='quilt',strips=[np.full((2400,x1-x0,3),220,np.uint8)],strip_origins=[x0],bg=np.array([220]*3))
    c=compose.Composer.__new__(compose.Composer);c.docs={did:doc};c.models={cid:d}
    c.stats=dict(classes={cid:dict(legend_id=4,name='shale',norm='shale')})
    c.assert_context_current=lambda:None;c.family=lambda did:'fixture'
    c.available=lambda *a:({cid:dict(pattern_id=cid)},[])
    original=render.layer
    class Rendered(Exception):pass
    def stop_after_render(*a,**kw):
        original(*a,**kw)
        raise Rendered()
    monkeypatch.setattr(render,'layer',stop_after_render)
    for height in (2399,2400,2401,4800):
        sampler=SimpleNamespace(rng=np.random.default_rng(7),duration_model=lambda *a:('doc','field',{}),
            duration_distribution=lambda *a:([height],[1]),
            sequence=lambda *a,**kw:[dict(top=0,bottom=height,class_id=cid)])
        monkeypatch.setattr(compose,'Sampler',lambda *a,**kw:sampler)
        expected=Rendered if height<=2400 else ValueError
        with pytest.raises(expected):c.column(did,7,height,tmp_path,page_top=y0,iteration=0)
    assert not (tmp_path/'column.png').exists()


def test_reproduced_ledger_overlay_rechecks_live_bytes_and_revisions(tmp_path,monkeypatch):
    import model
    store=model.Store();docs={e['id']:store.get(e['id']) for e in store.catalog}
    row=next(r for r in read(paths.DATA/'normalized/current.json')['records'] if r.get('usable_for_generation') and
             learn.generation_split(docs[r['document_id']])[1]=='train')
    did=row['document_id'];lid=row['targets'][0]['legend_id']
    folder=tmp_path/'proposals'/did;folder.mkdir(parents=True)
    tile=folder/'tile.png';Image.new('RGB',(4,4),'white').save(tile)
    source=folder/f'legend-{lid}.json';source.write_text('{}',encoding='utf-8')
    p=dict(doc_id=did,legend_id=lid,round=1,label='fixture',tile_path=str(tile),reviewer=dict(verdict='confident'))
    lineage=dict(doc_id=did,legend_id=lid,source_revision=docs[did]['revision'],boxes=[row['box_xyxy']],pixel_sha256=[row['pixel_sha256']])
    record=dict(key=f'{did}:{lid}',doc_id=did,legend_id=lid,round=1,admitted=True,status='admitted',
        proposal_path=source.relative_to(tmp_path).as_posix(),tile_path=tile.relative_to(tmp_path).as_posix(),
        proposal_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),tile_sha256=hashlib.sha256(tile.read_bytes()).hexdigest(),
        lineage=[lineage],ancestor_docs=[did],normalized_sources=[])
    ledger=tmp_path/'reprov/ancestry.jsonl';ledger.parent.mkdir();ledger.write_text(json.dumps(record)+'\n',encoding='utf-8')
    monkeypatch.setattr(paths,'OUT',tmp_path);monkeypatch.setattr(paths,'LAB',tmp_path);monkeypatch.setattr(paths,'PROPOSALS',tmp_path/'proposals')
    monkeypatch.setattr(learn,'REPROV_SHA256',hashlib.sha256(ledger.read_bytes()).hexdigest())
    enriched=learn.reproduced_proposal(p,docs)
    assert enriched['lineage']==[lineage] and enriched['tile_path']==p['tile_path'] and enriched['reviewer']==p['reviewer']
    docs[did]['revision']+=1
    assert learn.reproduced_proposal(p,docs) is p
    docs[did]['revision']-=1
    source.write_text('{"changed":true}',encoding='utf-8')
    assert learn.reproduced_proposal(p,docs) is p
    source.write_text('{}',encoding='utf-8');Image.new('RGB',(4,4),'black').save(tile)
    assert learn.reproduced_proposal(p,docs) is p


def test_review_panes_use_common_integer_scale(tmp_path,monkeypatch):
    monkeypatch.setattr(paths,'CANDIDATES',tmp_path)
    tile=np.full((8,10,3),220,np.uint8);tile[2,3]=0
    data=dict(kind='tile',tile=tile,bg=learn.background(tile),anchored=False,phase=(0,0))
    class Crops:
        def get(self,doc,box):
            return np.full((60,47,3),220,np.uint8)
    seen=[];original=Image.Image.resize
    def resize(im,size,*args,**kwargs):
        seen.append((im.size,size))
        return original(im,size,*args,**kwargs)
    monkeypatch.setattr(Image.Image,'resize',resize)
    seam=dict(match=1,method='repeat-overlap')
    entry=dict(legend_id=1,label='Dot',tier='human',evidence=[dict(box=[0,0,47,60],match=1)],
               cycles=2,matches=1,depth_ranges=[[0,60]],reason='test',seam_x=seam,seam_y=seam,seam_risk=False)
    candidates.sheets(dict(id='d'),dict(entries=[entry]),{1:data},{1:tile},{1:data},Crops())
    assert len(seen)==4
    assert all(size==(native[0]*3,native[1]*3) for native,size in seen)


def audit(stage):
    out = paths.OUT / 'recover' / 'checks'
    out.mkdir(parents=True, exist_ok=True)
    state = read(paths.STATE)
    critic = paths.OUT / 'critique' / 's1-tiles'
    snapshot = read(critic / 'snapshot.json')
    result = dict(cards=len(state['cards']), realism=sum(c.get('realism') is not None for c in state['cards']),
                  tiles={}, blanks=[], rails=[], palette={})
    for c in state['cards']:
        if c.get('tile'):
            p = paths.OUT / c['tile']
            with Image.open(p) as im:
                a = np.asarray(im.convert('RGB'))
            result['tiles'][c['key']] = dict(hash=hashlib.sha256(a.tobytes()).hexdigest(),
                                            realism=c.get('realism'), shape=list(a.shape))
            if stage == 'before':
                dst = out / 'baseline' / (c['key'].replace(':', '-') + '.png')
                dst.parent.mkdir(exist_ok=True)
                Image.fromarray(a).save(dst)
    current = {c['key']: c for c in state['cards']}
    for i, old in enumerate(snapshot['cards']):
        c = current.get(old['key'], old)
        p = paths.OUT / (c.get('tile') or c['samples'][0])
        if not p.exists():
            continue
        with Image.open(p) as im:
            a = np.asarray(im.convert('RGB'))
        with Image.open(critic / f'{i+1:02d}-{c["doc_id"]}-{c["legend_id"]}-real0.png') as im:
            rgb, _ = learn.clean(np.asarray(im.convert('RGB')))
        blank = np.full_like(a, learn.background(a).astype(np.uint8))
        data = dict(kind='tile', tile=blank, bg=learn.background(blank))
        result['blanks'].append(dict(key=c['key'], pixel=candidates.pixel_match(rgb, data),
                                    foreground=candidates.symbol_match(rgb, data),
                                    accepted=(score.fidelity(rgb, data)['acceptable'] if hasattr(score, 'fidelity') else
                                              candidates.pixel_match(rgb, data) >= .85)))
        if i == 0:
            for seed in range(8):
                rgb = render.layer(dict(kind='tile',tile=a,anchored=False,phase=(0,0)),250,310,0,120,np.random.default_rng(seed))
                result['rails'].append(int(sum((rgb[:,x].max(1)<150).mean()>.95 for x in range(1,59))))
        if c['key'] == 'c524c8e761b99c48:0':
            result['palette'] = dict(gray96_fraction=float((a == 96).all(2).mean()),
                                     colours=np.unique(a.reshape(-1,3),axis=0).tolist())
    (out / (stage + '.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('tiles','blanks')}, indent=1))
    print('blank controls accepted:',sum(b['accepted'] for b in result['blanks']), '/',len(result['blanks']))


def recovery_audit():
    import model
    root=paths.OUT/'recover'
    q=read(root/'review-queue.json')
    result=dict(entries=q['total'],made=0,lineage_violations=[],textures=[],sand={})
    images=model.Images();crops=learn.Crops(images);store=model.Store()
    try:
        for row in q['entries']:
            p=read(root/row['proposal'])
            if p['status']!='made':
                continue
            result['made']+=1
            for source in p['recovery']['lineage']:
                if learn.generation_split(store.get(source['doc_id']))[1]!='train':
                    result['lineage_violations'].append(row['key'])
            if row['method']=='stochastic-glyph-bootstrap':
                with np.load(root/p['doc_id']/p['recovery']['texture_model']) as a:
                    data=dict(kind='texture',glyphs=[a[k] for k in a.files if k.startswith('glyph_')],bg=a['bg'],density=float(a['density']))
                outputs=[render.layer(data,0,100,0,240,np.random.default_rng(s)) for s in (7,11)]
                result['textures'].append(dict(key=row['key'],source_density=data['density'],
                    output_density=[float(score.foreground(a,data['bg']).mean()) for a in outputs],
                    seeds_distinct=not np.array_equal(*outputs),
                    exact_short_repeat=any(np.array_equal(outputs[0][:-lag],outputs[0][lag:]) for lag in range(1,80))))
            if row['key']=='5709fe9e86ce942b:6':
                old=read(paths.PROPOSALS/p['doc_id']/f'legend-{p["legend_id"]}.json')
                with Image.open(paths.PROPOSALS/p['doc_id']/old['tile_png']) as im:
                    before=np.asarray(im.convert('RGB'))
                with Image.open(root/p['doc_id']/p['tile_png']) as im:
                    after=np.asarray(im.convert('RGB'))
                result['sand']=dict(before=[],after=[],reserved_boxes=p['column_boxes'][1:],
                                    sizes=[list(before.shape[:2]),list(after.shape[:2])])
                for box in p['column_boxes']:
                    rgb,_=learn.clean(crops.get(store.get(p['doc_id']),box))
                    for name,tile in [('before',before),('after',after)]:
                        result['sand'][name].append(score.fidelity(rgb,dict(tile=tile,bg=learn.background(tile))))
        (root/'checks'/'recovery.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    finally:
        if images.current:
            images.current.close()
    print('PASS recovery audit:',json.dumps(dict(entries=result['entries'],made=result['made'],
        lineage_violations=len(result['lineage_violations']),textures=len(result['textures']),
        texture_repeats=sum(r['exact_short_repeat'] for r in result['textures']),
        sand_rgb_before=[v['pixel'] for v in result['sand']['before']],
        sand_rgb_after=[v['pixel'] for v in result['sand']['after']])))


def compare():
    """Report each changed model and explain every >.02 score reduction."""
    root=paths.OUT/'recover'/'checks'
    before=read(root/'before.json')
    after=read(root/'after.json')
    state=read(paths.STATE)
    oldstate=read(paths.OUT/'critique'/'s2-failed'/'state-snapshot.json')
    oldcards={c['key']:c for c in oldstate['cards']}
    rows=[];drops=[];changed=0;transitions=0
    for c in state['cards']:
        key=c['key'];old=before['tiles'].get(key)
        if not old:
            continue
        new=after['tiles'].get(key)
        different=new is None or old['hash']!=new['hash']
        changed+=different;transitions+=new is None
        a=(old.get('realism') or {}).get('median');b=(c.get('realism') or {}).get('median')
        row=dict(key=key,changed_tile=bool(different),kind=c['kind'],before=a,after=b,
                 foreground_after=c['realism'].get('foreground_dice'),independent=c['realism'].get('held_out',0))
        if a is not None and b is not None and b<a-.02:
            explanations=[]
            if c.get('heldout_boxes'):
                explanations.append('Evaluation now reserves the final depth component before fitting/selection (512-row guard); this is a harder, unpaired reference set.')
            if c['kind']=='quilt':
                explanations.append('Insufficient multi-cycle foreground agreement: native quilting replaces a forced short repeat. Score measures a fixed generated window, not the best source patch.')
            if key=='5fbeaf2a0650e013:1024':
                explanations.append('Removed neutral RGB119 rail; old edge/background agreement is intentionally lost. A differently coloured source crop still triggers the background gate; card stays provisional.')
            if key=='c524c8e761b99c48:0':
                explanations.append('Unsupported RGB96 interpolations removed (2.25% to zero); weak repeat now retains native motifs.')
            if not explanations:
                explanations.append('Source votes collapse overlapping rectangles and use observed RGB medoids; rail cleaning and reference cleaning changed. Lower similarity is retained and marked provisional where foreground gates fail.')
            row['explanation']=' '.join(explanations)
            drops.append(row)
        if different or row.get('explanation'):
            rows.append(row)
    report=dict(before_cards=before['cards'],after_cards=after['cards'],
        before_scored=before['realism'],after_scored=after['realism'],changed_tile_cards=changed,
        tile_to_native_quilt=transitions,
        original_scored_median=float(np.median([(v.get('realism') or {})['median'] for v in before['tiles'].values() if v.get('realism')])),
        after_all_median=float(np.median([c['realism']['median'] for c in state['cards'] if c['realism'].get('median') is not None])),
        blank_accepted_before=sum(b['accepted'] for b in before['blanks']),blank_accepted_after=sum(b['accepted'] for b in after['blanks']),
        rails_before=before['rails'],rails_after=after['rails'],
        gray96_before=before['palette']['gray96_fraction'],gray96_after=after['palette']['gray96_fraction'],
        score_drops_over_002=drops,changed_cards=rows,
        limitation='Before/after aggregate scores are not paired: depth reservation, cleaning, reference typing and quilt diagnostics changed. Every reduction >0.02 is enumerated with an explanation.')
    (root/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS comparison:',json.dumps({k:v for k,v in report.items() if k not in ('score_drops_over_002','changed_cards')}))
    print('EXPLAINED score drops >0.02:',len(drops))


def visual_audit():
    from PIL import ImageDraw
    root=paths.OUT/'recover'/'checks'
    state=read(paths.STATE);cards={c['key']:c for c in state['cards']}
    old=read(paths.OUT/'critique'/'s1-tiles'/'snapshot.json')['cards']
    panels=[]
    for n in (1,2,5):
        c=cards[old[n-1]['key']]
        realpath=paths.OUT/'critique'/'s1-tiles'/f'{n:02d}-{c["doc_id"]}-{c["legend_id"]}-real0.png'
        with Image.open(realpath) as im:
            real=np.asarray(im.convert('RGB'))[:100]
        with Image.open(root/'baseline'/(c['key'].replace(':','-')+'.png')) as im:
            tile=np.asarray(im.convert('RGB'))
        before=tile[np.arange(100)[:,None]%len(tile),np.arange(real.shape[1])[None,:]%tile.shape[1]]
        with Image.open(paths.OUT/c['samples'][0]) as im:
            after=np.asarray(im.convert('RGB'))[:100]
        panels.append((c['key'],[('Native real',real),('Before repeat',before),('After sample',after)]))
    sheet=Image.new('RGB',(1100,3*355),'#dddddd');draw=ImageDraw.Draw(sheet)
    for row,(key,panes) in enumerate(panels):
        draw.text((4,row*355+4),key,fill='black')
        for col,(label,rgb) in enumerate(panes):
            draw.text((col*360+4,row*355+23),label+' @3x',fill='black')
            im=Image.fromarray(rgb);sheet.paste(im.resize((im.width*3,im.height*3),Image.Resampling.NEAREST),(col*360,row*355+48))
    sheet.save(root/'human-defects-after.png')
    print('PASS visual sheet:',root/'human-defects-after.png')


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='recovery':
        recovery_audit()
    elif len(sys.argv)>1 and sys.argv[1]=='compare':
        compare()
    elif len(sys.argv)>1 and sys.argv[1]=='visual':
        visual_audit()
    else:
        audit(sys.argv[2] if len(sys.argv)>2 else 'before')
