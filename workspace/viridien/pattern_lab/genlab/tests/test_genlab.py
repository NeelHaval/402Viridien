import numpy as np
import json
import copy
import pytest

from genlab import candidates, learn, render, score


@pytest.fixture
def dataset_lab(tmp_path, monkeypatch):
    """Tiny real builder, renderer and disk artifacts; no TIFF pages or lab writes."""
    import threading
    from types import SimpleNamespace
    from genlab import dataset, paths
    policy = dict(seed='strata-well-split-v1', document_groups={})
    groups = {}
    for i in range(100):
        doc = dict(id=str(i), name='testwell' + str(i))
        well, split = dataset.well_split(doc, policy)
        groups.setdefault(split, doc)
    assert set(groups) == {'train', 'validation', 'test'}
    docs = {}
    for split, d in groups.items():
        d = dict(d, width=16, height=100, legend=[dict(id=0, label_printed='Sandstone', box=[0,60,16,80])],
                 boxes=dict(interpreted_lithology_column=[0, 0, 16, 100]),
                 segments=[dict(id='human', status='approved', approved=True, legend_ids=[0], box=[0, 0, 16, 40])])
        docs[d['id']] = d
    source = tmp_path / 'lab'
    source.mkdir()
    for name, value in dict(OUT=source, GEN=source/'generated', STATE=source/'state.json',
        DATA=source/'data', CANDIDATES=source/'candidates', RATINGS=source/'ratings.jsonl',
        PROPOSALS=source/'proposals').items():
        monkeypatch.setattr(paths, name, value)
    (paths.DATA / 'normalized').mkdir(parents=True)
    (paths.DATA / 'normalized/split-policy.json').write_text(json.dumps(policy))
    tile = np.full((4, 4, 3), (210, 190, 150), np.uint8)
    tile[0, ::2] = 0
    from PIL import Image
    Image.fromarray(tile).save(source/'tile.png')
    cards, manifest = [], []
    for d in docs.values():
        if dataset.well_split(d, policy)[1] == 'test':
            continue
        cid = dataset.class_id(d['id'], 0)
        cards.append(dict(key=cid, doc_id=d['id'], legend_id=0, label='Sandstone', source='human',
                          width=16, signature='same', tile='tile.png', samples=['tile.png'], verdict=None,tile_source='column',
                          rendering=dict(kind='tile',phase=[0,0],anchored_x=False,anchored_y=False,x_repeat=True,x_origin=0)))
        manifest.append(dict(pattern_id=cid, file='tile.png', source='human', machine_evidence=[],
            normalized_sources=[dict(sample_id=dataset.sample_id(d,d['segments'][0]),
                annotation_version=dataset.annotation_version(d['segments'][0]))]))
    (paths.GEN).mkdir()
    paths.STATE.write_text(json.dumps(dict(built_at=1, cards=cards)))
    (paths.GEN/'manifest.json').write_text(json.dumps(dict(patterns=manifest)))
    class Crops:
        def get(self, doc, box):
            yy, xx = np.indices((box[3]-box[1], box[2]-box[0]))
            rgb = tile[yy % 4, xx % 4].copy()
            if box==[0,0,16,40]:
                rgb[5:8, 4:7] = 45  # measured overlay, with known underlying class
            if doc['id']==groups['test']['id']:
                rgb[:]=(30,70,180)
            return rgb
    import hashlib
    records = []
    for d in docs.values():
        seg = d['segments'][0]
        records.append(dict(sample_id=dataset.sample_id(d,seg),document_id=d['id'],annotation_id=seg['id'],
            annotation_version=dataset.annotation_version(seg),
            legend_version=hashlib.sha256(json.dumps(d['legend'],sort_keys=True).encode()).hexdigest()[:16],
            box_xyxy=seg['box'],targets=[dict(legend_id=0)],accepted=True,quality='accepted',quality_issues=[],
            usable_for_generation=dataset.well_split(d,policy)[1]!='test',
            split='test' if dataset.well_split(d,policy)[1]=='test' else 'train',
            pixel_sha256=dataset.pixel_hash(Crops().get(d,seg['box']))))
    (paths.DATA/'normalized/current.json').write_text(json.dumps(dict(snapshot='fixture-v1',records=records)))
    monkeypatch.setattr(dataset.labstore, 'Store', lambda: SimpleNamespace(
        catalog=[dict(id=k) for k in docs], get=lambda did: docs[did]))
    monkeypatch.setattr(dataset.labstore, 'Images', lambda: SimpleNamespace(lock=threading.RLock()))
    monkeypatch.setattr(learn, 'Crops', lambda images: Crops())
    monkeypatch.setattr(score, 'load_prefill', lambda: {})
    return dataset.DatasetBuilder(tmp_path/'datasets'), groups, docs


def test_dataset_splits_and_stable_ids_across_builds(dataset_lab):
    from genlab import dataset
    builder, groups, docs = dataset_lab
    versions = [builder.build(), builder.build()]
    manifests = [[json.loads(s) for s in (p/'manifest.jsonl').read_text().splitlines()] for p in versions]
    assert sorted(r['sample_id'] for r in manifests[0]) == sorted(r['sample_id'] for r in manifests[1])
    test_id = groups['test']['id']
    for p, rows in zip(versions, manifests):
        assert rows and all(r['doc'] != test_id and r['split'] != 'test' for r in rows)
        assert all(r['doc'] != test_id for r in dataset.read(p/'real.json'))
        train_wells = {r['well'] for r in rows if r['split'] == 'train'}
        validation_wells = {r['well'] for r in rows if r['split'] == 'validation'}
        assert not train_wells & validation_wells
        for measurement in dataset.read(p/'realism.json')['measurements']:
            assert measurement['doc'] != test_id
        assert {r['split'] for r in rows} == {'train', 'validation'}


def test_dataset_augmentation_preserves_labels_and_column_masks(dataset_lab):
    from PIL import Image
    builder, _, _ = dataset_lab
    p = builder.build()
    rows = [json.loads(s) for s in (p/'manifest.jsonl').read_text().splitlines()]
    pairs = {}
    for r in rows:
        if r['format'] == 'strip':
            pairs.setdefault(r['class_id'], {})[r['variant']] = r
    for variants in pairs.values():
        clean, augmented = variants['clean'], variants['augmented']
        assert clean['class_id'] == augmented['class_id'] and clean['name'] == augmented['name']
        assert Image.open(p/clean['file']).size == Image.open(p/augmented['file']).size
        assert augmented['augmentation']['calibration'] == 'measured'
        if clean['split'] == 'validation':
            assert clean['tile_provenance']['real_crop_inputs'] == []
    columns = [r for r in rows if r['format'] == 'column']
    assert columns
    for r in columns:
        mask = np.asarray(Image.open(p/r['label_mask_png']))
        truth = json.loads((p/r['truth_json']).read_text())
        for layer in truth['layers']:
            assert np.all(mask[layer['top']:layer['bottom']] == layer['mask_value'])
            assert layer['class_id'] in r['classes']


def test_dataset_eval_refuses_machine_truth():
    from genlab.dataset_eval import classifier_run
    bad = dict(kind='real', evidence='machine', status='approved', split='validation')
    with pytest.raises(ValueError, match='machine evidence is not truth'):
        classifier_run(None, [], [bad], 'matcher')
    bad['evidence'] = 'human-accepted'
    bad['split'] = 'train'
    with pytest.raises(ValueError, match='validation'):
        classifier_run(None, [], [bad], 'matcher')


def test_dataset_validation_tiles_ignore_real_crop_changes(dataset_lab):
    from genlab import dataset
    from PIL import Image
    builder, groups, _ = dataset_lab
    val_doc = groups['validation']['id']
    first = builder.build()
    crop = builder.crop
    def changed(did, box):
        rgb = crop(did,box)
        if did==val_doc and box==[0,0,16,40]:
            return np.zeros_like(rgb)
        return rgb
    builder.crop = changed
    with pytest.raises(ValueError,match='Normalized pixels changed'):
        builder.build()
    assert all(m['split']=='train' for m in dataset.read(first/'realism.json')['measurements'])


def test_dataset_closed_set_ids_macro_and_chance():
    from genlab.dataset import closed_summary
    truth = [dict(class_id='a:0',name='shale'),dict(class_id='a:0',name='shale'),dict(class_id='b:1',name='shale')]
    r = closed_summary(truth,['a:0','b:1','b:1'],[2,2,4])
    assert r['correct']==2 and r['macro_classes']==2
    assert r['accuracy']==66.667 and r['macro_accuracy']==75.
    assert r['chance']==41.667 and r['macro_chance']==37.5
    with pytest.raises(ValueError,match='identical lengths'):
        closed_summary(truth,['a:0'],[2,2,4])
    with pytest.raises(ValueError,match='identical lengths'):
        closed_summary(truth,['a:0']*4,[2,2,4])
    from genlab.dataset_eval import summarize
    with pytest.raises(ValueError,match='identical lengths'):
        summarize(truth,['shale'],{'shale'})


def test_dataset_legend_white_margins_do_not_repeat_as_scanlines():
    from genlab.dataset import legend_model
    tile = np.full((8,16,3),(255,255,0),np.uint8)
    tile[2,::4]=0
    raw = np.full((28,32,3),255,np.uint8)
    raw[2:-2]=np.tile(tile,(3,2,1))
    _,model,detail=legend_model(raw)
    assert detail['white_edge_trim']==[2,2]
    assert not (model['tile'].min(2)>=240).any()
    white_pattern=np.full((28,32,3),255,np.uint8)
    white_pattern[4::8,::4]=0
    _,_,detail=legend_model(white_pattern)
    assert detail['white_edge_trim']==[0,0]


def test_dataset_quarantine_and_stale_baseline_fail_closed(dataset_lab):
    from genlab import dataset,paths
    builder,groups,_ = dataset_lab
    first = builder.build()
    index = dataset.read(paths.DATA/'normalized/current.json')
    train_doc = groups['train']['id']
    target = next(r for r in index['records'] if r['document_id']==train_doc)
    target.update(quality='review',quality_issues=['duplicate_of_test'],usable_for_generation=False)
    (paths.DATA/'normalized/current.json').write_text(json.dumps(index))
    builder.load_eligibility()
    assert all(r['doc']!=train_doc for r in builder.human_rows())
    builder.baseline = first
    assert all(r['doc']!=train_doc for r in builder.human_rows())
    target.update(quality='accepted',quality_issues=[],usable_for_generation=True,annotation_version='stale')
    (paths.DATA/'normalized/current.json').write_text(json.dumps(index))
    builder.load_eligibility()
    assert all(r['doc']!=train_doc for r in builder.human_rows())


def test_dataset_donor_ancestry_rejects_held_and_unknown(dataset_lab):
    builder,groups,_ = dataset_lab
    card = dict(key='train:0',doc_id=groups['train']['id'])
    assert not builder.safe_model(card,dict(donor=dict(doc_id=groups['test']['id'])),{'train:0':card})
    assert not builder.safe_model(card,dict(donor=dict(doc_id='missing',legend_id=1)),{'train:0':card})
    from genlab.dataset import model_ancestors
    did = groups['train']['id']
    parent = dict(key=did+':1',doc_id=did,donor=dict(doc_id=groups['test']['id']))
    ancestors = model_ancestors(card,dict(donor=dict(doc_id=did,legend_id=1)),{did+':1':parent})
    assert groups['test']['id'] in ancestors


def test_dataset_discriminator_inventory_is_exact_and_shared(dataset_lab):
    from genlab.dataset import read
    from genlab.dataset_eval import paired_inventory
    from PIL import Image
    builder,_,_ = dataset_lab
    folder = builder.build()
    inventory = paired_inventory(folder,read(folder/'real.json'))
    assert len(inventory['pairs'])==2
    for pair in inventory['pairs']:
        assert pair['real']['doc']==pair['clean']['doc']==pair['augmented']['doc']
        assert pair['real']['class_id']==pair['clean']['class_id']==pair['augmented']['class_id']
        assert all(list(Image.open(folder/pair[v]['file']).size)==pair['native_size'] for v in ('real','clean','augmented'))
        assert pair['augmented']['augmentation']['split']=='train'


def test_dataset_components_keep_topology_and_sparse_noise():
    from scipy.ndimage import label
    from genlab.realism import component_budget,measure,stroke
    mask = np.zeros((20,20),np.uint8)
    mask[2:12,2:12] = 255
    mask[14:16,14:16] = 255
    clean = np.full((20,20,3),220,np.uint8)
    kept = component_budget(mask,np.zeros_like(clean),clean,.05)
    assert kept.sum()/255==4 and label(kept)[1]==1
    tile = np.full((10,10,3),128,np.uint8)
    actual = tile.copy()
    actual[::4,::4] = 132
    assert measure(tile,actual)[0]['noise']>0
    dash = np.full((12,12,3),170,np.uint8)
    dash[4,3:8] = 0
    assert (stroke(dash,1).max(2)<80).sum()==10


def test_dataset_round3_overlay_destination_budget():
    from scipy.ndimage import label
    from genlab.dataset import dataset_augment
    mask = np.zeros((96,96),np.uint8)
    for y in range(0,90,6):
        for x in range(0,90,6):
            mask[y:y+2,x:x+2] = 255
    colours = np.zeros((96,96,3),np.uint8)
    for h,w in ((15,47),(10,10),(3,3),(160,200)):
        clean = np.full((h,w,3),220,np.uint8)
        for seed in range(4):
            rgb,params = dataset_augment(clean,None,(mask,colours),np.random.default_rng(seed))
            assert params['pasted_pixels'] <= int(h*w*.05)
            assert np.any(rgb!=clean,axis=2).sum() <= int(h*w*.05)
            assert all(c['pixels']==4 for c in params.get('overlay_components',[]))


def test_dataset_round3_current_policy_blocks_unlabelled_well(dataset_lab,monkeypatch):
    from genlab import dataset as d,dataset_eval as e,paths,server
    builder,groups,docs = dataset_lab
    folder = builder.build()
    # The training model remains archived after its annotation rows disappear.
    did = groups['train']['id']
    ix = d.read(paths.DATA/'normalized/current.json')
    ix['records'] = [r for r in ix['records'] if r['document_id']!=did]
    (paths.DATA/'normalized/current.json').write_text(json.dumps(ix))
    policy = d.read(paths.DATA/'normalized/split-policy.json')
    policy['document_groups'][did] = d.well_split(groups['test'],policy)[0]
    (paths.DATA/'normalized/split-policy.json').write_text(json.dumps(policy))
    (folder.parent/'latest.json').write_text(json.dumps(dict(version=folder.name)))
    monkeypatch.setattr(d,'ROOT',folder.parent)
    # Serving must notice the policy before a watcher updates the registry.
    assert server.Handler.quarantined_artifact(folder.parent/(folder.name+'.zip'))
    assert server.Handler.quarantined_artifact(folder/'real.json')
    audit = e.leak_audit(folder)
    assert not audit['passed'] and audit['current_held_synthetic_n']>0
    assert folder.name in e.revoke_unsafe_versions(folder.parent)
    assert d.read(folder.parent/'latest.json')['version'] is None


def test_dataset_round3_support_and_truth_pixel_integrity(dataset_lab):
    from genlab import dataset as d,dataset_eval as e
    from PIL import Image
    builder,_,_ = dataset_lab
    folder = builder.build()
    truth = [r for r in d.read(folder/'real.json') if r['split']=='validation']
    support = d.evaluation_candidates(folder,truth)
    target = next(r for r in support if r.get('format')=='strip')
    features = e.Features(folder)
    features.get(target)
    with Image.open(folder/target['file']) as im:
        size = im.size
    Image.new('RGB',size,'white').save(folder/target['file'])
    with pytest.raises(ValueError,match='pixel hash'):
        d.evaluation_candidates(folder,truth)
    with pytest.raises(ValueError,match='pixel hash'):
        features.get(target)
    assert not e.leak_audit(folder)['passed']
    target = truth[0]
    with Image.open(folder/target['file']) as im:
        size = im.size
    Image.new('RGB',size,'white').save(folder/target['file'])
    with pytest.raises(ValueError,match='pixel hash'):
        e.checked_rgb(folder,target)


def test_dataset_round3_pairs_reject_cohort_and_poison_before_pixels(dataset_lab,monkeypatch):
    from genlab import dataset as d,dataset_eval as e,paths
    from PIL import Image
    builder,groups,_ = dataset_lab
    folder = builder.build()
    real = d.read(folder/'real.json')
    inventory = e.paired_inventory(folder,real)
    assert e.paired_inventory(folder,list(reversed(real)))==inventory
    with pytest.raises(ValueError,match='cohort|identity'):
        e.paired_inventory(folder,[])
    poisoned = copy.deepcopy(inventory)
    target = next(p for p in poisoned['pairs'] if p['real']['split']=='train')
    target['real'].update(doc=groups['test']['id'],file='never-read.png')
    (folder/'discriminator-pairs.json').write_text(json.dumps(poisoned))
    original = Image.open
    def guard(path,*args,**kwargs):
        assert not str(path).endswith('never-read.png'), 'held pixels reached'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Image,'open',guard)
    with pytest.raises(ValueError):
        e.paired_inventory(folder,real)
    assert not e.leak_audit(folder)['passed']


def test_dataset_round3_pairs_bind_model_and_style_dependencies(dataset_lab,monkeypatch):
    from genlab import dataset as d,dataset_eval as e,paths
    builder,groups,_ = dataset_lab
    folder = builder.build()
    real = d.read(folder/'real.json')
    inventory = e.paired_inventory(folder,real)
    frozen_context = e.eligibility_context(folder)
    with monkeypatch.context() as decoder_patch:
        def incompatible_decoder(*args,**kwargs):
            raise AssertionError('Changed decoder reached model bytes')
        decoder_patch.setattr(learn,'load_model',incompatible_decoder)
        with pytest.raises(ValueError,match='stale dependencies'):
            e.paired_inventory(folder,real)
    # Fitting-only source edits do not change the loaded rendering decoder.
    unrelated = folder/'unrelated-fitting-source.py'
    unrelated.write_text('fit_revision = 2\n')
    with monkeypatch.context() as fitting_patch:
        fitting_patch.setattr(learn,'__file__',str(unrelated))
        assert e.paired_inventory(folder,real)==inventory
    with monkeypatch.context() as renderer_patch:
        renderer_patch.setattr(d,'local_model',lambda model:dict(model,x_origin=900))
        with pytest.raises(ValueError,match='stale dependencies'):
            e.paired_inventory(folder,real)
    ix = d.read(paths.DATA/'normalized/current.json')
    ix['snapshot'] = 'unrelated-annotation-growth'
    extra = dict(next(r for r in ix['records'] if r['document_id']==groups['train']['id']),
                 sample_id='unrelated-new-sample',annotation_id='unrelated-new-annotation')
    ix['records'].append(extra)
    (paths.DATA/'normalized/current.json').write_text(json.dumps(ix))
    e.checked_rgb(folder,real[0],frozen_context)
    assert e.paired_inventory(folder,real)==inventory
    params = d.read(folder/'realism.json')
    params['measurements'][0]['noise'] += 1
    (folder/'realism.json').write_text(json.dumps(params))
    with pytest.raises(ValueError,match='stale dependencies'):
        e.paired_inventory(folder,real)
    # Rejected model ancestry must also stop before hashing its pixel asset.
    stats = d.read(folder/'stats.json')
    target = next(r for r in stats['classes'] if r['split']=='train')
    target['ancestor_docs'].append(groups['test']['id'])
    (folder/'stats.json').write_text(json.dumps(stats))
    from pathlib import Path
    context = e.eligibility_context(folder)
    original_open = Path.open
    forbidden = folder/target['model_file']
    def model_guard(path,*args,**kwargs):
        assert path!=forbidden, 'held model bytes reached'
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',model_guard)
    with pytest.raises(ValueError,match='Unsafe cache dependency'):
        e.pair_dependency(folder,real,context)


def test_dataset_round3_complete_build_preserves_page_origin_model(dataset_lab):
    from genlab import dataset as d,dataset_eval as e,paths
    from PIL import Image
    builder,groups,_ = dataset_lab
    state = d.read(paths.STATE)
    card = next(c for c in state['cards'] if c['doc_id']==groups['train']['id'])
    tile = np.full((6,16,3),255,np.uint8)
    tile[:,0] = 119
    tile[::2,5:9] = 0
    model = dict(kind='tile',tile=tile,bg=np.array([255]*3),phase=[0,0],
        anchored_x=True,anchored_y=False,x_repeat=False,x_origin=100)
    card['render_model'] = learn.export_model(model,paths.OUT/'full-model',card['signature'])
    Image.fromarray(tile).save(paths.OUT/card['tile'])
    paths.STATE.write_text(json.dumps(state))
    folder = builder.build()
    rows = [json.loads(s) for s in (folder/'manifest.jsonl').read_text().splitlines()]
    clean = [r for r in rows if r['class_id']==card['key'] and r['format']=='strip' and r['variant']=='clean']
    assert len(clean)==4
    for row in clean:
        with Image.open(folder/row['file']) as im:
            rgb = np.asarray(im.convert('RGB'))
        expected = render.sample(model,dict(x0=100,x1=116,mask=np.zeros(16,bool),ink=np.empty((0,3),np.uint8)),
                                 len(rgb),np.random.default_rng(0),y0=0)
        # Independently fitted edge rails may replace the first column.
        assert any(np.array_equal(rgb[:,3:-3],np.roll(expected,shift,axis=0)[:,3:-3]) for shift in range(6))
        assert len(np.unique(rgb.reshape(-1,3),axis=0))==3
    pairs = e.paired_inventory(folder,d.read(folder/'real.json'))
    pair = next(p for p in pairs['pairs'] if p['real']['doc']==groups['train']['id'])
    with Image.open(folder/pair['clean']['file']) as im:
        assert len(np.unique(np.asarray(im).reshape(-1,3),axis=0))==3


def test_dataset_round3_long_crop_exports_effective_geometry(dataset_lab):
    from genlab import dataset as d,paths
    from PIL import Image
    builder,groups,docs = dataset_lab
    did = groups['train']['id']
    seg = docs[did]['segments'][0]
    seg['box'] = [0,0,16,1269]
    ix = d.read(paths.DATA/'normalized/current.json')
    record = next(r for r in ix['records'] if r['document_id']==did)
    record.update(box_xyxy=seg['box'],annotation_version=d.annotation_version(seg),
                  pixel_sha256=d.pixel_hash(builder.crops.get(docs[did],seg['box'])))
    (paths.DATA/'normalized/current.json').write_text(json.dumps(ix))
    builder.load_eligibility()
    folder = builder.build()
    row = next(r for r in d.read(folder/'real.json') if r['doc']==did)
    assert row['source_box']==[0,0,16,1269] and row['crop_box']==[0,0,16,480]
    with Image.open(folder/row['file']) as im:
        assert list(im.size)==row['image_size']==[16,480]


def test_dataset_round3_failed_build_can_retry_in_same_process(dataset_lab,monkeypatch):
    builder,_,_ = dataset_lab
    original = builder.snapshot
    calls = []
    def once(folder):
        calls.append(folder)
        if len(calls)==1:
            raise RuntimeError('transient snapshot failure')
        return original(folder)
    monkeypatch.setattr(builder,'snapshot',once)
    with pytest.raises(RuntimeError,match='transient snapshot'):
        builder.build()
    assert builder.build().is_dir()
    assert len(calls)==2 and calls[0]!=calls[1]


def test_dataset_round3_normalized_native_prefix_never_reads_tiff(dataset_lab,monkeypatch):
    from genlab import dataset as d,paths
    from PIL import Image
    builder,groups,docs = dataset_lab
    did = groups['train']['id']
    seg = docs[did]['segments'][0]
    seg['box'] = [0,0,16,1269]
    full = builder.crops.get(docs[did],seg['box'])
    native = paths.DATA/'normalized/native-prefix.png'
    Image.fromarray(full).save(native)
    ix = d.read(paths.DATA/'normalized/current.json')
    row = next(r for r in ix['records'] if r['document_id']==did)
    row.update(box_xyxy=seg['box'],annotation_version=d.annotation_version(seg),
               pixel_sha256=d.pixel_hash(full),native_png=native.name)
    (paths.DATA/'normalized/current.json').write_text(json.dumps(ix))
    builder.load_eligibility()
    def trap(*args,**kwargs):
        raise AssertionError('Native prefix reached TIFF reader')
    monkeypatch.setattr(builder.crops,'get',trap)
    assert np.array_equal(builder.crop(did,seg['box']),full[:480])
    Image.fromarray(np.full_like(full,255)).save(native)
    with pytest.raises(ValueError,match='pixels changed'):
        builder.crop(did,seg['box'])
    row['usable_for_generation'] = False
    (paths.DATA/'normalized/current.json').write_text(json.dumps(ix))
    with pytest.raises(ValueError,match='Revoked'):
        builder.crop(did,seg['box'])


def test_dataset_round3_optional_margin_cache_miss_stays_bounded(dataset_lab,monkeypatch):
    from genlab import dataset as d
    builder,groups,docs = dataset_lab
    docs[groups['train']['id']]['width'] = 128
    folder = builder.root/'optional-margin'
    folder.mkdir(parents=True)
    def missing(*args,**kwargs):
        raise ValueError('Unsafe TIFF crop-cache miss: bounded cached PNG required')
    monkeypatch.setattr(builder,'crop',missing)
    assert builder.context_library(folder,builder.human_rows())==[]
    assert len(d.read(folder/'nuisance.json')['unavailable'])==1
    def quarantined(*args,**kwargs):
        raise ValueError('Quarantined test-identical pixels')
    monkeypatch.setattr(builder,'crop',quarantined)
    with pytest.raises(ValueError,match='Quarantined'):
        builder.context_library(folder,builder.human_rows())


def test_dataset_round3_embedded_provenance_stops_before_model_bytes(dataset_lab,monkeypatch):
    from genlab import dataset as d,paths
    builder,groups,_ = dataset_lab
    state = d.read(paths.STATE)
    card = next(c for c in state['cards'] if c['doc_id']==groups['train']['id'])
    model = dict(kind='tile',tile=np.full((6,16,3),210,np.uint8),bg=np.array([210]*3))
    card['render_model'] = learn.export_model(model,paths.OUT/'embedded-model',card['signature'],
        provenance=dict(ancestor_docs=[groups['test']['id']]))
    paths.STATE.write_text(json.dumps(state))
    assert groups['test']['id'] in d.model_ancestors(card,{}, {card['key']:card})
    def trap(*args,**kwargs):
        raise AssertionError('Embedded held ancestry reached model bytes')
    monkeypatch.setattr(learn,'load_model',trap)
    folder = builder.root/'embedded-ancestry-probe'
    folder.mkdir(parents=True)
    patterns,_ = builder.snapshot(folder)
    assert all(p['meta']['tile_provenance']['kind']=='legend-swatch-only' for p in patterns)


def test_dataset_paired_ci_ignores_new_difficult_labels(tmp_path):
    from genlab.dataset_eval import compare_versions
    truths = [dict(doc=str(i),annotation_id='a',well=str(i),split='validation',box=[0,0,10,10],name='shale') for i in range(4)]
    pred = [dict(doc=str(i),annotation_id='a',actual='x',predicted='x',candidate_count=2) for i in range(4)]
    for name in ('v1','v2'):
        folder = tmp_path/name
        folder.mkdir()
        rows = truths if name=='v1' else truths+[dict(truths[0],annotation_id='new-hard')]
        (folder/'real.json').write_text(json.dumps(rows))
        predictions = pred if name=='v1' else pred+[dict(pred[0],annotation_id='new-hard',predicted=None)]
        (folder/'eval.json').write_text(json.dumps(dict(closed_set_within_document={alg:{v:dict(predictions=predictions) for v in ('clean','augmented')} for alg in ('matcher','logistic')})))
    result = compare_versions(tmp_path/'v2',tmp_path/'v1')
    assert result['changed_population'] and result['qualified']
    assert result['results']['matcher/augmented']['delta']['ci95']==[0.,0.]


def test_dataset_column_prior_preserves_dominant_class_share():
    from genlab.dataset import weighted_column
    models = {k:dict(kind='tile',tile=np.full((1,1,3),c,np.uint8),anchored=False,phase=[0,0]) for k,c in [(0,100),(1,200)]}
    rails = dict(x0=0,x1=4,mask=[False]*4,ink=[0,0,0])
    rgb,truth = weighted_column(models,rails,{0:[1],1:[1]},{},{0:90,1:10},10000,np.random.default_rng(5))
    assert .88<float((rgb[:,:,0]==100).mean())<.92
    assert all(a['legend_id']!=b['legend_id'] for a,b in zip(truth,truth[1:]))


def test_dataset_round2_finite_support_never_renders(dataset_lab,monkeypatch):
    from genlab import dataset
    from genlab.dataset_eval import Features,leak_audit
    builder,groups,docs = dataset_lab
    original = dataset.legend_model
    def unsafe(raw):
        a,b,detail = original(raw)
        return a,b,dict(detail,seam_risk=True)
    monkeypatch.setattr(dataset,'legend_model',unsafe)
    folder = builder.build()
    truth = [r for r in dataset.read(folder/'real.json') if r['split']=='validation']
    def trap(*args,**kwargs):
        raise AssertionError('Finite reference must never enter cyclic rendering')
    monkeypatch.setattr(render,'sample',trap)
    rows = dataset.evaluation_candidates(folder,truth)
    finite = [r for r in rows if r['format']=='finite-legend']
    assert len(finite)==2 and all(not r['renderable'] and not r['synthesis_eligible'] for r in finite)
    run = dataset.within_document(Features(folder,native=True),rows,truth,'matcher','augmented')
    assert run['correct']==1
    assert run['support_coverage']['generation_supported_boxes']==0
    assert run['support_coverage']['recognition_supported_boxes']==1
    assert leak_audit(folder)['passed']


def test_dataset_round2_ancestry_union_and_conflicts(dataset_lab):
    from genlab import dataset,paths
    builder,groups,_ = dataset_lab
    card = dataset.read(paths.STATE)['cards'][0]
    manifest = dataset.read(paths.GEN/'manifest.json')['patterns'][0]
    if builder.doc_info[card['doc_id']]['split']!='train':
        card = dataset.read(paths.STATE)['cards'][1]
        manifest = dataset.read(paths.GEN/'manifest.json')['patterns'][1]
    held = groups['test']['id']
    assert builder.safe_model(card,manifest,{card['key']:card})
    for c,m in [(dict(card,ancestor_docs=[held]),manifest),
                (card,dict(manifest,ancestor_docs=[held])),
                (dict(card,lineage=[dict(doc_id=held)]),manifest),
                (dict(card,recovery=dict(lineage=[dict(document_id=held)])),manifest),
                (dict(card,donor=dict(doc_id=held)),dict(manifest,donor=dict(doc_id=card['doc_id']))),
                (dict(card,ancestor_docs=['missing']),manifest),
                (dict(card,donor=dict(doc_id=card['doc_id'],legend_id=card['legend_id'])),manifest)]:
        assert not builder.safe_model(c,m,{card['key']:c})
    parent = dict(card,key=card['doc_id']+':999',legend_id=999)
    child = dict(card,donor=dict(doc_id=parent['doc_id'],legend_id=999))
    builder.model_manifests = {parent['key']:dict(manifest,ancestor_docs=[held])}
    assert not builder.safe_model(child,manifest,{child['key']:child,parent['key']:parent})


def test_dataset_round2_comparison_refuses_missing_duplicate_and_changed_predictions(tmp_path):
    from genlab.dataset_eval import compare_versions
    truths = [dict(doc=str(w),annotation_id=str(i),well=str(w),split='validation',box=[0,0,10,10],name='shale')
              for w in range(7) for i in range(10)]
    preds = [dict(doc=r['doc'],annotation_id=r['annotation_id'],actual='x',predicted='x',candidate_count=2) for r in truths]
    for case,rows in [('complete',preds),('missing',[preds[0],preds[10],preds[20]]),
                      ('duplicate',preds+[preds[0]]),('changed',[dict(p,candidate_count=3) for p in preds])]:
        folder = tmp_path/case
        folder.mkdir()
        (folder/'real.json').write_text(json.dumps(truths))
        (folder/'eval.json').write_text(json.dumps(dict(closed_set_within_document={alg:{v:dict(predictions=rows)
            for v in ('clean','augmented')} for alg in ('matcher','logistic')})))
    for case in ('missing','duplicate','changed'):
        assert not compare_versions(tmp_path/case,tmp_path/'complete')['qualified']
    assert compare_versions(tmp_path/'complete',tmp_path/'complete')['qualified']


def test_dataset_round2_rendering_roundtrip_refuses_legacy_axes():
    from genlab.dataset import rendering_payload
    tile = np.full((6,20,3),255,np.uint8)
    tile[:,0] = 119
    model = dict(kind='tile',tile=tile,anchored=False,anchored_x=True,anchored_y=False,
                 x_repeat=False,x_origin=100,phase=[0,0])
    saved = rendering_payload(dict(rendering={k:v for k,v in model.items() if k!='tile'}))
    for seed in range(8):
        native = render.layer(model,100,140,0,50,np.random.default_rng(seed))
        loaded = render.layer(dict(saved,tile=tile),100,140,0,50,np.random.default_rng(seed))
        assert np.array_equal(native,loaded)
        assert not (loaded[:,1:].max(2)<150).any()
    with pytest.raises(ValueError,match='placement'):
        rendering_payload(dict(kind='tile',phase=[0,0],anchored=False))


def test_dataset_round2_quarantine_walks_all_artifact_routes(tmp_path,monkeypatch):
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import urlopen
    from urllib.error import HTTPError
    from genlab import server,dataset,paths
    from types import SimpleNamespace
    root = tmp_path/'datasets'
    root.mkdir()
    monkeypatch.setattr(paths,'OUT',tmp_path)
    monkeypatch.setattr(dataset,'ROOT',root)
    monkeypatch.setattr(paths,'GEN',tmp_path/'generated')
    paths.GEN.mkdir()
    (paths.GEN/'ok.png').write_bytes(b'preview')
    monkeypatch.setattr(server,'LAB',SimpleNamespace(lock=threading.Lock()),raising=False)
    (root/'quarantine.json').write_text(json.dumps({'v5':{'reason':'test contamination'}}))
    for name in ('v5','v8'):
        (root/name/'models').mkdir(parents=True)
        (root/(name+'.zip')).write_bytes(b'PKfixture')
        (root/name/'models/tile.png').write_bytes(b'pixels')
        (root/name/'real.json').write_text('[]')
    service = ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
    worker = threading.Thread(target=service.serve_forever,daemon=True)
    worker.start()
    checked = []
    try:
        for prefix in ('/datasets/','/out/datasets/'):
            for suffix in ('v5.zip','v5/real.json','v5/models/tile.png','%76%35.zip',
                           'v8/../v5.zip','v5%5Cmodels%5Ctile.png','v5.zip?download=1'):
                try:
                    with urlopen(f'http://127.0.0.1:{service.server_port}'+prefix+suffix) as response:
                        status = response.status
                except HTTPError as error:
                    status = error.code
                assert status in (404,409), (prefix,suffix,status)
                checked.append(prefix+suffix)
            for suffix in ('v8.zip','v8/real.json','v8/models/tile.png'):
                with urlopen(f'http://127.0.0.1:{service.server_port}'+prefix+suffix) as response:
                    assert response.status==200
        with urlopen(f'http://127.0.0.1:{service.server_port}/dataset.zip') as response:
            assert response.status==200
        assert len(checked)==14
    finally:
        service.shutdown()
        service.server_close()
        worker.join(timeout=5)


def test_dataset_round2_full_model_asset_snapshot(tmp_path,monkeypatch):
    from genlab import dataset,paths
    monkeypatch.setattr(paths,'OUT',tmp_path)
    bank = [np.random.default_rng(k).integers(0,256,(h,20,3),dtype=np.uint8) for k,h in [(1,2400),(2,1200)]]
    models = [dict(kind='quilt',strips=bank,bg=np.array([255]*3),x_origin=100,strip_origins=[100,100]),
              dict(kind='tile',tile=bank[0][:6],bg=np.array([255]*3),phase=[2,0],
                   anchored_x=True,anchored_y=False,x_repeat=False,x_origin=100)]
    for i,model in enumerate(models):
        payload = learn.export_model(model,tmp_path/'source',str(i))
        card = dict(signature=str(i),render_model=payload)
        loaded,copied = dataset.snapshot_render_model(card,card,tmp_path/'snapshot',tmp_path)
        restored = learn.load_model(copied,root=tmp_path/'snapshot',signature=str(i))
        for h in ((50,960,1200) if model['kind']=='quilt' else (50,960,2400)):
            for seed in range(8):
                a = render.layer(model,100,140,0,h,np.random.default_rng(seed))
                b = render.layer(restored,100,140,0,h,np.random.default_rng(seed))
                assert np.array_equal(a,b)
        if model['kind']=='quilt':
            assert [list(s.shape) for s in restored['strips']]==[list(s.shape) for s in bank]
            shifted = dict(model,strip_origins=[100,105],native_extent=[100,125])
            local = dataset.local_model(shifted)
            assert local['strip_origins']==[0,5] and local['native_extent']==[0,25]
            for seed in range(8):
                assert np.array_equal(render.layer(shifted,100,140,0,50,np.random.default_rng(seed)),
                    render.layer(local,0,40,0,50,np.random.default_rng(seed)))
            for oversized in (1201,2400):
                with pytest.raises(ValueError,match='supported maximum'):
                    render.layer(restored,100,140,0,oversized,np.random.default_rng(0))


def test_dataset_round2_revocation_stops_before_feature_fitting(dataset_lab,monkeypatch):
    from genlab import dataset,dataset_eval,paths
    builder,groups,_ = dataset_lab
    folder = builder.build()
    index = dataset.read(paths.DATA/'normalized/current.json')
    source = next(r for r in index['records'] if r['document_id']==groups['train']['id'])
    source.update(usable_for_generation=False,quality_issues=['duplicate_of_test'])
    (paths.DATA/'normalized/current.json').write_text(json.dumps(index))
    def trap(*args,**kwargs):
        raise AssertionError('Revoked inputs reached model fitting')
    monkeypatch.setattr(dataset_eval,'Features',trap)
    with pytest.raises(ValueError,match='before model fitting'):
        dataset_eval.evaluate(folder)


def test_dataset_round2_held_ancestry_rejected_before_model_pixels(dataset_lab,monkeypatch):
    from genlab import dataset,paths
    from PIL import Image
    builder,groups,_ = dataset_lab
    state = dataset.read(paths.STATE)
    for card in state['cards']:
        if card['doc_id']==groups['train']['id']:
            card['ancestor_docs']=[groups['test']['id']]
    paths.STATE.write_text(json.dumps(state))
    original = Image.open
    def guarded(path,*args,**kwargs):
        if str(path)==str(paths.OUT/'tile.png'):
            raise AssertionError('Held ancestry reached model pixels')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Image,'open',guarded)
    folder = builder.root/'ancestry-probe'
    folder.mkdir(parents=True)
    patterns,_ = builder.snapshot(folder)
    assert all(p['meta']['tile_provenance']['kind']=='legend-swatch-only' for p in patterns)


def test_dataset_publication_uses_promoted_baseline_and_ci(tmp_path,monkeypatch):
    from genlab import dataset_eval,dataset_loop
    from genlab.dataset import read
    folder = tmp_path/'v3'
    folder.mkdir()
    (tmp_path/'latest.json').write_text(json.dumps(dict(version='v1')))
    (folder/'eval.json').write_text(json.dumps(dict(shortcut_gate=dict(passed=True))))
    monkeypatch.setattr(dataset_eval,'leak_audit',lambda _:dict(passed=True))
    monkeypatch.setattr(dataset_loop,'_dataset_original_publish',lambda *_:{},raising=False)
    calls = []
    comparison = dict(qualified=False,results={'matcher/augmented':dict(delta=dict(ci95=[-3.,1.]))})
    def compare(candidate,baseline):
        calls.append(Path(baseline).name)
        return comparison
    from pathlib import Path
    monkeypatch.setattr(dataset_eval,'compare_versions',compare)
    report = dataset_eval.publish(tmp_path,folder)
    assert calls==['v1'] and not report['eligible_latest'] and not report['regression']
    assert read(tmp_path/'latest.json')['version']=='v1'
    comparison.update(qualified=True,results={'matcher/augmented':dict(delta=dict(ci95=[-1.,2.]))})
    assert dataset_eval.publish(tmp_path,folder)['eligible_latest']
    assert read(tmp_path/'latest.json')['version']=='v3'


def test_dataset_borrowing_holds_out_the_whole_dependent_well(dataset_lab):
    from genlab import dataset, paths
    builder, groups, _ = dataset_lab
    state = json.loads(paths.STATE.read_text())
    manifest = json.loads((paths.GEN/'manifest.json').read_text())
    train_doc, val_doc = groups['train']['id'], groups['validation']['id']
    row = next(r for r in manifest['patterns'] if r['pattern_id'] == train_doc + ':0')
    row['donor'] = dict(doc_id=val_doc)
    card = dict(next(c for c in state['cards'] if c['doc_id'] == train_doc))
    card.update(key=train_doc + ':1', legend_id=1)
    state['cards'].append(card)
    manifest['patterns'].append(dict(pattern_id=card['key'], file='tile.png', source='human'))
    paths.STATE.write_text(json.dumps(state))
    (paths.GEN/'manifest.json').write_text(json.dumps(manifest))
    p = builder.build()
    rows = [json.loads(s) for s in (p/'manifest.jsonl').read_text().splitlines()]
    assert {r['split'] for r in rows if r['doc'] == train_doc} == {'validation'}
    assert {r['split'] for r in dataset.read(p/'real.json') if r['doc'] == train_doc} == {'validation'}


def test_dataset_reader_retries_windows_replacement():
    from genlab.reading import ReadOnlyStore
    class Replaced:
        calls = 0
        def get(self, ident):
            self.calls += 1
            if self.calls == 1:
                raise PermissionError('writer replacing file')
            return dict(id=ident)
    source = Replaced()
    assert ReadOnlyStore(source).get('d') == {'id': 'd'}
    assert source.calls == 2


def test_dataset_loop_ten_pattern_threshold_in_temp_folder(tmp_path):
    from genlab.dataset_loop import snapshot, trigger
    p = tmp_path/'state.json'
    p.write_text(json.dumps(dict(cards=[])))
    args = dict(state_path=p, reviews=tmp_path/'reviews', proposals=tmp_path/'proposals', ratings_path=tmp_path/'ratings.jsonl')
    baseline = snapshot(**args)
    for n in (9, 10):
        p.write_text(json.dumps(dict(cards=[dict(key=f'd:{i}', signature='a') for i in range(n)])))
        current = snapshot(**args)
        gate = trigger(baseline, current, last_build=100, now=200)
        assert gate['pattern_count'] == n
        assert gate['due'] == (n == 10)
    assert trigger(baseline, current, 100, 200)['reason'] == '10-pattern threshold'
    assert not trigger(current, current, 100, 2000)['due']
    nine = dict(current, patterns={k: v for k, v in list(current['patterns'].items())[:9]})
    assert trigger(baseline, nine, 100, 1900)['due']
    assert trigger(baseline, nine, 100, 1900)['reason'] == '30-minute change window'


def test_realism_phase_alignment_is_exact():
    from genlab.realism import aligned
    rng = np.random.default_rng(4)
    tile = rng.integers(0, 255, (7, 9, 3), dtype=np.uint8)
    yy, xx = np.indices((31, 14))
    real = tile[(yy+3) % 7, (xx+5) % 9]
    prediction, phase = aligned(tile, real)
    assert phase == [3, 5]
    assert np.array_equal(prediction, real)


def test_dataset_publication_regressions_retention_and_history(tmp_path, monkeypatch):
    from genlab.dataset_loop import publish, api_index
    from genlab.dataset import read
    monkeypatch.setattr(score, 'load_prefill', lambda: {})
    root = tmp_path/'datasets'
    for i in range(1, 8):
        folder = root/f'v{i}'
        folder.mkdir(parents=True)
        (folder/'stats.json').write_text(json.dumps(dict(version=folder.name, patterns=0, samples=0, classes=[])))
        # Every new version regresses: latest must stay v1 even when it is old.
        e = dict(status='passed', truth_n=3, truth_wells=['a'], truth_fingerprint='unchanged',
                 fake_to_real=dict(matcher=dict(macro_accuracy=80-3*i, per_class=[])),
                 discriminator=dict(augmented=dict(accuracy=50+6*i)))
        (folder/'eval.json').write_text(json.dumps(e))
        report = publish(root, folder)
        assert report['regression'] == (i > 1)
        assert read(root/'latest.json')['version'] == 'v1'
    assert (root/'v1').is_dir()  # protected latest
    assert not (root/'v2').exists()
    assert not (root/'v2.zip').exists()
    assert all((root/f'v{i}').is_dir() for i in range(3, 8))
    history = api_index(root)['versions']
    assert len(history) == 7 and not next(r for r in history if r['version'] == 'v2')['retained']
    assert 'REGRESSION' in (root/'CHANGELOG.md').read_text(encoding='utf-8')
    assert '## v2' in (root/'CHANGELOG.md').read_text(encoding='utf-8')


def write_proposal(root, doc_id='d', lid=1, round_=1, verdict='confident', unlisted=False, boxes=None):
    from PIL import Image
    folder = root / doc_id
    folder.mkdir(parents=True, exist_ok=True)
    prefix = 'unlisted' if unlisted else 'legend'
    tile = np.full((8, 10, 3), (180, 210, 140), np.uint8)
    tile[2:4, 2:6] = 0
    Image.fromarray(tile).save(folder / f'{prefix}-{lid}-tile.png')
    record = dict(doc_id=doc_id, status='made', method='mixed', round=round_,
                  tile_png=f'{prefix}-{lid}-tile.png', period=dict(x=10, y=8),
                  column_boxes=boxes if boxes is not None else [[2, 20, 62, 120]],
                  description='small marks', looks_like='Sandstone', share_of_column_rows=.2,
                  label=f'Pattern {lid}')
    record['unlisted_id' if unlisted else 'legend_id'] = f'u{lid}' if unlisted else lid
    path = folder / f'{prefix}-{lid}.json'
    path.write_text(json.dumps(record), encoding='utf-8')
    if verdict is not None:
        (folder / f'{prefix}-review-r{round_}.json').write_text(
            json.dumps({str(record.get('unlisted_id', lid)): dict(verdict=verdict, note='independent note')}),
            encoding='utf-8')
    return path


@pytest.fixture
def proposal_lab(tmp_path, monkeypatch):
    from genlab import build, paths
    doc, crops = candidate_page()
    other = copy.deepcopy(doc)
    other.update(id='other', name='Other log')
    docs = {d['id']: dict(id=d['id'], name=d['name'], doc=d, training_allowed=True,
                         column=d['boxes']['interpreted_lithology_column'],
                         legend={e['id']: e['label_printed'] for e in d['legend']}) for d in (doc, other)}
    for name, value in dict(OUT=tmp_path, GEN=tmp_path/'generated', STATE=tmp_path/'state.json',
                            CANDIDATES=tmp_path/'candidates', VERDICTS=tmp_path/'verdicts',
                            PROPOSALS=tmp_path/'proposals', FLEET=tmp_path/'fleet',
                            RATINGS=tmp_path/'ratings.jsonl').items():
        monkeypatch.setattr(paths, name, value)
    for d in (doc, other):
        candidates.refresh(d, crops, {}, [])
    humans = {}
    monkeypatch.setattr(build.labstore, 'Store', lambda: object())
    monkeypatch.setattr(build.labstore, 'Images', lambda: None)
    monkeypatch.setattr(learn, 'Crops', lambda images: crops)
    monkeypatch.setattr(learn, 'human_examples', lambda store: (docs, dict(humans)))
    monkeypatch.setattr(score, 'load_prefill', lambda: {})
    monkeypatch.setattr(score, 'queue_guesses', lambda: {})
    monkeypatch.setattr(score, 'reviewed_boxes', lambda *args: [])
    monkeypatch.setattr(build, 'COLUMNS', 2)
    return build.Builder(), docs, humans


def test_proposal_round_verdict_and_split(proposal_lab):
    from genlab import paths
    builder, docs, _ = proposal_lab
    # A missing candidate report is allowed: approved proposals stand alone.
    (paths.CANDIDATES/'d.json').unlink()
    write_proposal(paths.PROPOSALS)
    first = builder.build()
    card = next(c for c in first['cards'] if c['source'] == 'agent-made')
    assert card['verdict']['note'] == 'independent note' and card['real'] == [[2, 20, 62, 120]]
    write_proposal(paths.PROPOSALS, round_=2, verdict=None)
    assert builder.build()['agent_made_pattern_count'] == 0
    write_proposal(paths.PROPOSALS, round_=2, verdict='reject')
    assert builder.build()['agent_made_pattern_count'] == 0
    write_proposal(paths.PROPOSALS, round_=2)
    assert builder.build()['agent_made_pattern_count'] == 1
    path = paths.PROPOSALS/'d'/'legend-1.json'
    p = json.loads(path.read_text()); p['status'] = 'skip'; path.write_text(json.dumps(p))
    assert builder.build()['agent_made_pattern_count'] == 0
    write_proposal(paths.PROPOSALS, round_=2)
    docs['d']['training_allowed'] = False
    assert builder.build()['agent_made_pattern_count'] == 0


def test_agent_card_falls_back_to_real_legend_swatch(proposal_lab):
    from genlab import paths
    builder, docs, _ = proposal_lab
    (paths.CANDIDATES/'d.json').unlink()
    write_proposal(paths.PROPOSALS, boxes=[])
    card = next(c for c in builder.build()['cards'] if c['source'] == 'agent-made')
    assert card['evidence_kind'] == 'swatch' and card['examples'] == 0
    assert card['real'] == [docs['d']['doc']['legend'][0]['box']]


def test_proposal_priority_human_verified_agent_swatch(proposal_lab):
    from genlab import paths
    builder, docs, humans = proposal_lab
    write_proposal(paths.PROPOSALS)
    def source():
        return next(c['source'] for c in builder.build()['cards'] if c['key'] == 'd:1')
    assert source() == 'machine-verified'
    # Human boxes win even when an approved proposal is available.
    humans['d', 1] = [dict(id='accepted', doc_id='d', legend_id=1, label='Pattern 1',
                           box=[2, 20, 62, 120], provenance='human', flagged=False)]
    assert source() == 'human'
    humans.clear()
    # Keep the report signature current without rescanning this controlled tier.
    rp = paths.CANDIDATES/'d.json'
    report = json.loads(rp.read_text())
    report['signature'] = candidates.signature(docs['d']['doc'], {})
    entry = report['entries'][0]
    entry['tier'] = 'swatch'
    entry['model'] = candidates.swatch_model(docs['d']['doc'], entry, candidates.load_data(paths.CANDIDATES/entry['data_file']))
    rp.write_text(json.dumps(report))
    paths.VERDICTS.mkdir()
    (paths.VERDICTS/'d.json').write_text(json.dumps({'1': dict(verdict='confident')}))
    assert source() == 'agent-made'
    write_proposal(paths.PROPOSALS, verdict='reject')
    assert source() == 'swatch'
    # A reviewer reject for the machine candidate is still respected globally.
    (paths.VERDICTS/'d.json').write_text(json.dumps({'1': dict(verdict='reject')}))
    write_proposal(paths.PROPOSALS)
    assert not any(c['key'] == 'd:1' for c in builder.build()['cards'])


def test_unlisted_manifest_normalized_and_incremental(proposal_lab, monkeypatch):
    from genlab import paths
    from genlab.server import Lab
    from PIL import Image
    builder, _, _ = proposal_lab
    builder.build()
    before = {p: p.stat().st_mtime_ns for p in (paths.GEN/'other').glob('*')}
    rendered = []
    original = builder._render_doc
    def watch(doc_id, *args):
        rendered.append(doc_id)
        return original(doc_id, *args)
    monkeypatch.setattr(builder, '_render_doc', watch)
    # fingerprint includes names as well as time and size; use the real method.
    lab = Lab.__new__(Lab)
    old_fp = lab.fingerprint()
    write_proposal(paths.PROPOSALS, unlisted=True)
    assert lab.fingerprint() != old_fp
    state = builder.build()
    assert rendered == state['rebuilt_documents'] == ['d']
    assert before == {p: p.stat().st_mtime_ns for p in before}
    assert state['unlisted_pattern_count'] == 1
    manifest = json.loads((paths.GEN/'manifest.json').read_text())
    rows = [r for r in manifest['patterns'] if r['source'] == 'unlisted']
    assert len(rows) == 4
    for r in rows:
        assert r['legend_id'] == 'u1' and r['pattern_id'] == 'd:u1'
        assert r['label_ids'] == ['d:u1'] and r['looks_like'] == 'Sandstone'
        assert r['label'] == 'Unlisted: small marks' and r['targets'][0]['legend_id'] == 'u1'
        assert Image.open(paths.GEN/r['normalized_png']).size == (224, 224)
    assert not any(layer['legend_id'] == 'u1' for c in manifest['columns'] for layer in c['layers'])
    assert state['coverage']['generated_entries'] == 4
    assert state['coverage']['generated_patterns'] == 5
    assert state['coverage']['generated_unlisted_entries'] == 1
    rendered.clear()
    builder.build()
    assert not rendered


def test_reviewer_accuracy_latest_votes_and_historical_provenance():
    ratings = {'a': dict(verdict='good'), 'b': dict(verdict='bad'),
               'c': dict(verdict='clear', source='agent-made', method='column'),
               'old': dict(verdict='good', source='unlisted', method='reconstructed'),
               'human': dict(verdict='good')}
    manifest = [dict(file='a', source='agent-made', method='mixed'),
                dict(file='b', source='agent-made', method='mixed'),
                dict(file='human', source='human')]
    result = score.reviewer_accuracy(ratings, manifest)
    assert result['total']['total'] == 3
    mixed = next(r for r in result['by_method'] if r['method'] == 'mixed')
    assert (mixed['real'], mixed['wrong'], mixed['percent_real']) == (1, 1, 50)
    assert result['by_source'][1]['percent_real'] == 100


class FakeCrops:
    """A page made of one known repeat tile; crops are cut from it."""
    def __init__(self, page):
        self.page = page

    def get(self, doc, box):
        x0, y0, x1, y1 = box
        return self.page[y0:y1, x0:x1].copy()


def make_page(tile, h=2000, w=60):
    py, px = tile.shape[:2]
    return tile[np.arange(h)[:, None] % py, np.arange(w)[None, :] % px]


def test_tile_is_recovered_and_slip_is_flagged():
    rng = np.random.default_rng(1)
    tile = np.full((20, 10, 3), (255, 180, 110), np.uint8)
    tile[3:5, 2:8] = 0
    tile[13:15, 0:4] = 0
    page = make_page(tile)
    page[1500:1700] = (202, 202, 202)          # a grey layer someone labelled the same
    docs = {'d': dict(doc=dict(id='d'), legend={1: 'Siltstone'})}
    exs = [dict(id=str(i), doc_id='d', legend_id=1, label='Siltstone', box=[0, y, 60, y + 90],
                provenance='', flagged=False) for i, y in enumerate(range(0, 1400, 140))]
    exs.append(dict(id='slip', doc_id='d', legend_id=1, label='Siltstone', box=[0, 1500, 60, 1700],
                    provenance='', flagged=False))
    m, data = learn.learn(FakeCrops(page), docs, ('d', 1), exs)
    assert m['kind'] == 'tile' and m['tile_size'] == [10, 20]
    assert [s['id'] for s in m['suspects']] == ['slip']
    rails = dict(x0=0, x1=60, mask=[False] * 60, ink=[0, 0, 0])
    out = render.sample(data, rails, 200, rng, y0=0)
    shifted = [np.array_equal(out[:20], make_page(np.roll(np.roll(tile, dy, 0), dx, 1), 20, 60))
               for dy in range(20) for dx in range(10)]
    assert any(shifted)


def test_column_layers_cover_every_row_and_never_repeat():
    rng = np.random.default_rng(2)
    a = dict(kind='tile', tile=np.zeros((4, 4, 3), np.uint8), phase=(0, 0), anchored=False)
    b = dict(kind='tile', tile=np.full((4, 4, 3), 200, np.uint8), phase=(0, 0), anchored=False)
    rails = dict(x0=0, x1=8, mask=[True] + [False] * 7, ink=[9, 9, 9])
    rgb, truth = render.column({1: a, 2: b}, rails, [10, 30, 50], {}, 303, rng)
    assert truth[0]['top'] == 0 and truth[-1]['bottom'] == 303
    assert all(x['bottom'] == y['top'] for x, y in zip(truth, truth[1:]))
    assert all(x['legend_id'] != y['legend_id'] for x, y in zip(truth, truth[1:]))
    assert (rgb[:, 0] == 9).all()


def test_full_truth_scores_rows():
    doc = dict(id='d', name='n', height=100, legend=[dict(id=1, label_printed='Shale'), dict(id=2, label_printed='Sand')],
               segments=[dict(status='approved', box=[0, 0, 5, 50], legend_ids=[1]),
                         dict(status='approved', box=[0, 50, 5, 100], legend_ids=[2])])
    guesses = [dict(top_y=0, bottom_y=60, suggestion='shale', tier='low'),
               dict(top_y=60, bottom_y=100, suggestion='Sand', tier='medium')]
    r = score.full_truth_doc(doc, guesses)
    assert r['row_accuracy'] == 90.0
    assert r['confusion'][0] == dict(guessed='Shale', actual='Shale', rows=10) or r['confusion'][0]['rows'] == 10
    assert r['boundary']['median_px'] == 10


def candidate_page(twin=False):
    a = np.full((20, 10, 3), (255, 180, 110), np.uint8)
    a[3:5, 2:8] = 0
    a[13:15, 0:4] = 0
    b = np.full((12, 8, 3), (130, 210, 240), np.uint8)
    b[2:4, 1:6] = 0
    page = np.full((1800, 240, 3), 255, np.uint8)
    for y in range(0, 1800, 200):
        page[y:y + 200, :64] = make_page(a if (y // 200) % 2 == 0 else b, 200, 64)
    legends = []
    for lid, tile in enumerate((a, a if twin else b), 1):
        x, y = 100, 100 * lid
        page[y:y + 64, x:x + 54] = 0
        page[y + 2:y + 62, x + 2:x + 52] = make_page(tile, 60, 50)
        legends.append(dict(id=lid, label_printed=f'Pattern {lid}', box=[x, y, x + 54, y + 64]))
    doc = dict(id='d', name='Synthetic', legend=legends, segments=[], height=1800,
               boxes=dict(interpreted_lithology_column=[0, 0, 64, 1800]))
    return doc, FakeCrops(page)


def test_candidates_verify_two_swatches_and_column_patterns():
    doc, crops = candidate_page()
    # Deliberately wrong saved guesses are proposals, not accepted labels.
    prefill = [dict(top_y=y, bottom_y=y + 90, suggestion='Pattern 2') for y in (20, 420, 820)]
    report, data, _, _ = candidates.analyze(doc, crops, {}, prefill)
    assert [e['tier'] for e in report['entries']] == ['verified', 'verified']
    assert set(data) == {1, 2}
    for e in report['entries']:
        assert e['cycles'] >= 2
        assert e['matches'] >= 3 and len(e['depth_ranges']) >= 2
        assert all(v['match'] >= .90 and v['margin'] >= .08 for v in e['evidence'])
        assert all(ex['provenance'] == 'machine-verified' for ex in candidates.examples(doc, e))


def test_twin_swatch_is_ambiguous_but_human_overrides():
    doc, crops = candidate_page(twin=True)
    report, _, _, _ = candidates.analyze(doc, crops, {}, [])
    assert [e['tier'] for e in report['entries']] == ['ambiguous', 'ambiguous']
    assert report['entries'][0]['twins'][0]['legend_id'] == 2
    assert not any(e['evidence'] for e in report['entries'])
    human = dict(id='accepted', doc_id='d', legend_id=1, label='Pattern 1', box=[2, 20, 62, 120],
                 flagged=False, provenance='human')
    report, _, _, _ = candidates.analyze(doc, crops, {('d', 1): [human]}, [])
    assert [e['tier'] for e in report['entries']] == ['human', 'ambiguous']


def test_pixel_match_is_best_cyclic_fraction_and_checks_background():
    rng = np.random.default_rng(4)
    tile = np.full((5, 4, 3), 200, np.uint8)
    tile[rng.random((5, 4)) < .4] = 0
    data = dict(tile=tile, bg=learn.background(tile))
    crop = make_page(np.roll(np.roll(tile, 2, 0), 1, 1), 13, 9)
    crop[0, :3] = 110
    brute = max(float((np.linalg.norm(make_page(np.roll(np.roll(tile, dy, 0), dx, 1), 13, 9).astype(float) - crop, axis=2) < 45).mean())
                for dy in range(5) for dx in range(4))
    assert candidates.pixel_match(crop, data) == brute
    assert candidates.pixel_match(np.full_like(crop, (100, 170, 230)), data) == 0


def test_pixel_match_high_colour_fallback_is_exact():
    rng = np.random.default_rng(8)
    tile = rng.integers(190, 215, (9, 8, 3), dtype=np.uint8)
    tile[:2] = rng.integers(0, 30, (2, 8, 3), dtype=np.uint8)
    crop = make_page(np.roll(np.roll(tile, 3, 0), 4, 1), 19, 17)
    crop[:3, :4] = 100
    assert len(np.unique(crop.reshape(-1, 3), axis=0)) > 64
    brute = max(float((np.linalg.norm(make_page(np.roll(np.roll(tile, dy, 0), dx, 1), 19, 17).astype(float) - crop, axis=2) < 45).mean())
                for dy in range(9) for dx in range(8))
    assert candidates.pixel_match(crop, dict(tile=tile, bg=learn.background(tile))) == brute
    assert candidates.pixel_match(crop, dict(tile=tile, bg=learn.background(tile)), minimum=.95) == brute
    assert candidates.pixel_match(crop, dict(tile=tile, bg=learn.background(tile)), minimum=.99) == 0


def test_repeating_swatches_without_evidence_are_swatch_candidates():
    doc, crops = candidate_page()
    crops.page[:, :64] = 255
    report, _, _, _ = candidates.analyze(doc, crops, {}, [])
    assert [e['tier'] for e in report['entries']] == ['swatch', 'swatch']
    assert all(not e['evidence'] and e['reason'] for e in report['entries'])


def test_separate_ranges_and_cached_data(tmp_path):
    ev = [dict(box=[0, y, 20, y + 60]) for y in (0, 30, 200)]
    assert candidates.depth_ranges(ev) == [[0, 90], [200, 260]]
    data = dict(kind='tile', tile=np.full((5, 4, 3), 200, np.uint8),
                bg=np.array([200, 200, 200]), anchored=False, phase=(1, 2))
    path = tmp_path / 'tile.npz'
    candidates.save_data(path, data)
    loaded = candidates.load_data(path)
    assert np.array_equal(loaded['tile'], data['tile'])
    assert loaded['phase'] == (1, 2) and not loaded['anchored']


def test_build_uses_cache_verdicts_and_human_override(tmp_path, monkeypatch):
    import json
    from genlab import build, paths
    doc, crops = candidate_page()
    info = dict(id='d', name=doc['name'], doc=doc, training_allowed=True, column=doc['boxes']['interpreted_lithology_column'],
                legend={e['id']: e['label_printed'] for e in doc['legend']})
    for name, value in dict(OUT=tmp_path, GEN=tmp_path / 'generated', STATE=tmp_path / 'state.json',
                            CANDIDATES=tmp_path / 'candidates', VERDICTS=tmp_path / 'verdicts',
                            RATINGS=tmp_path / 'ratings.jsonl').items():
        monkeypatch.setattr(paths, name, value)
    candidates.refresh(doc, crops, {}, [])
    active_humans = {}
    monkeypatch.setattr(build.labstore, 'Store', lambda: object())
    monkeypatch.setattr(build.labstore, 'Images', lambda: None)
    monkeypatch.setattr(learn, 'Crops', lambda images: crops)
    monkeypatch.setattr(learn, 'human_examples', lambda store: ({'d': info}, dict(active_humans)))
    monkeypatch.setattr(score, 'load_prefill', lambda: {})
    monkeypatch.setattr(score, 'queue_guesses', lambda: {})
    monkeypatch.setattr(score, 'reviewed_boxes', lambda *args: [])
    refreshed = []
    refresh = candidates.refresh
    def watch_refresh(*args, **kwargs):
        refreshed.append(args[0]['id'])
        return refresh(*args, **kwargs)
    monkeypatch.setattr(candidates, 'refresh', watch_refresh)
    builder = build.Builder()
    first = builder.build()
    assert first['verified_pattern_count'] == 2 and first['dataset']['columns'] == 12
    # A second build reuses reports AND learned models (no hidden re-learning).
    original_learn = learn.learn
    monkeypatch.setattr(learn, 'learn', lambda *args: (_ for _ in ()).throw(AssertionError('unexpected relearn')))
    again = builder.build()
    assert not refreshed and again['pattern_count'] == 2
    assert [c['signature'] for c in first['cards']] == [c['signature'] for c in again['cards']]
    paths.VERDICTS.mkdir()
    vp = paths.VERDICTS / 'd.json'
    vp.write_text(json.dumps({'1': {'verdict': 'reject', 'note': 'wrong'},
                              '2': {'verdict': 'unsure', 'note': 'check'}}), encoding='utf-8')
    marked = builder.build()
    assert marked['pattern_count'] == 1 and marked['cards'][0]['legend_id'] == 2
    assert marked['cards'][0]['verdict']['verdict'] == 'unsure'
    assert not refreshed
    # Restore learning, then accept a human box: refresh only this document.
    monkeypatch.setattr(learn, 'learn', original_learn)
    vp.unlink()
    active_humans[('d', 1)] = [dict(id='human', doc_id='d', legend_id=1, label='Pattern 1',
                                   box=[2, 20, 62, 120], flagged=False, provenance='human',
                                   normalized_sample_id='accepted-1', annotation_version='v1')]
    accepted = builder.build()
    assert refreshed == ['d']
    assert accepted['human_pattern_count'] == 1 and accepted['verified_pattern_count'] == 1
    assert next(c for c in accepted['cards'] if c['legend_id'] == 1)['source'] == 'human'
    assert accepted['human_boxes'] == 1
    from normalized import letterbox
    from PIL import Image
    manifest=json.loads((paths.GEN/'manifest.json').read_text(encoding='utf-8'))
    for record in manifest['patterns']:
        if record['source'] == 'machine-verified':
            assert record['machine_evidence']
            assert all(e['provenance'] == 'machine-verified' for e in record['machine_evidence'])
        expected,mask,_=letterbox(Image.open(paths.GEN/record['native_png']).convert('RGB'))
        assert np.array_equal(np.asarray(expected),np.asarray(Image.open(paths.GEN/record['normalized_png'])))
        assert np.array_equal(np.asarray(mask),np.asarray(Image.open(paths.GEN/record['valid_mask_png'])))
    # Swatches are candidates until confident, and unsure revokes that approval.
    rp = paths.CANDIDATES / 'd.json'
    report = json.loads(rp.read_text(encoding='utf-8'))
    second = next(e for e in report['entries'] if e['legend_id'] == 2)
    second['tier'] = 'swatch'
    second['model'] = candidates.swatch_model(doc, second, candidates.load_data(paths.CANDIDATES / second['data_file']))
    rp.write_text(json.dumps(report), encoding='utf-8')
    assert builder.build()['pattern_count'] == 1
    vp.write_text(json.dumps({'2': {'verdict': 'confident', 'note': 'looks right'}}), encoding='utf-8')
    approved = builder.build()
    assert approved['swatch_pattern_count'] == 1 and approved['pattern_count'] == 2
    assert approved['dataset']['columns'] == 0  # no fabricated column evidence
    vp.write_text(json.dumps({'2': {'verdict': 'unsure', 'note': 'reconsider'}}), encoding='utf-8')
    assert builder.build()['pattern_count'] == 1
    info['training_allowed'] = False
    active_humans.clear()
    held_out = builder.build()
    assert held_out['pattern_count'] == 0 and held_out['dataset']['samples'] == 0


def test_column_trims_last_layer_to_less_than_four_remaining_rows():
    tile=np.zeros((4,4,3),np.uint8)
    data=dict(kind='tile',tile=tile,phase=(0,0),anchored=False)
    rails=dict(x0=0,x1=8,mask=[False]*8,ink=[0,0,0])
    for height in (1,2,3,9,10,11):
        rgb,truth=render.column({1:data},rails,[8],{},height,np.random.default_rng(0))
        assert rgb.shape==(height,8,3)
        assert truth[-1]['bottom']==height
        assert sum(r['bottom']-r['top'] for r in truth)==height


def test_tile_choice_uses_matching_swatch_for_quilt_and_keeps_colour():
    from genlab.selection import choose
    tile = np.full((12, 10, 3), (240, 200, 80), np.uint8)
    tile[3:5, 2:7] = 0
    real = make_page(tile, 96, 40)
    crops = FakeCrops(real)
    examples = [dict(id='one', box=[0, 0, 40, 96])]
    swatch = dict(kind='tile', tile=tile, bg=learn.background(tile), anchored=False, phase=(0, 0))
    model = dict(kind='quilt', sources=['one'])
    quilt = dict(kind='quilt', strips=[real], bg=learn.background(real))
    chosen, data = choose(crops, {}, examples, model, quilt, swatch, False)
    assert chosen['tile_source'] == 'legend swatch' and data['kind'] == 'tile'
    assert chosen['tile_scores']['legend_swatch']['median'] == 1
    different = dict(swatch, tile=tile.copy(), bg=np.array([100, 180, 240]))
    different['tile'][different['tile'].max(2) > 80] = different['bg']
    chosen, data = choose(crops, {}, examples, dict(kind='tile', voters=1), swatch, different, False)
    assert chosen['tile_source'] == 'column' and data is swatch
    assert chosen['tile_scores']['legend_swatch']['median'] == 0


def test_repeat_preview_uses_strip_native_width_and_bounded_height():
    for height, width in [(8, 1), (35, 1), (64, 35), (120, 48)]:
        tile = np.full((height, width, 3), 210, np.uint8)
        data = dict(kind='tile', tile=tile, bg=learn.background(tile), phase=(0, 0), anchored=False)
        rails = dict(x0=100, x1=137, mask=[False]*37, ink=[0, 0, 0])
        strip = render.sample(data, rails, 240, np.random.default_rng(0))
        preview = render.repeat_preview(data, strip.shape[1])
        assert preview.shape[1] == strip.shape[1] == 37
        assert 48 <= preview.shape[0] <= 240
        assert np.array_equal(data['tile'], tile)  # marker never enters training pixels


def test_swatches_reject_doubled_black_edges_and_interior_white_scanline():
    from genlab.swatch import edge_continuity
    tile = np.full((48, 40, 3), (128, 255, 128), np.uint8)
    tile[::12, 3:20] = 0
    tile[0] = tile[-1] = 0
    assert not edge_continuity(tile, 0)['ok']
    tile = np.full((70, 92, 3), (128, 255, 128), np.uint8)
    tile[2::23, 10:35] = 0
    tile[14::23, 50:75] = 0
    tile[44] = 255
    assert candidates.swatch(tile)[2]['seam_risk']


def test_borrowing_cannot_hide_wrong_symbols_in_matching_background():
    tile = np.full((20, 20, 3), (255, 255, 0), np.uint8)
    tile[3, 3] = 0
    crop = np.full((60, 60, 3), (255, 255, 0), np.uint8)
    crop[3:6, 3:8] = 0
    data = dict(tile=tile, bg=learn.background(tile))
    assert candidates.pixel_match(crop, data) > .92
    assert candidates.symbol_match(crop, data) < .80


def test_coverage_weights_rows_and_separates_training_documents():
    docs = {'a': dict(training_allowed=True, legend={1:'Clay-S', 2:'Sand'}),
            'b': dict(training_allowed=False, legend={1:'Clay-S'})}
    reports = [dict(doc_id='a', name='Log A'), dict(doc_id='b', name='Log B')]
    guesses = {'log a': [dict(top_y=0,bottom_y=80,suggestion=' clay-s '),
                         dict(top_y=80,bottom_y=100,suggestion='Sand')],
               'Log B': [dict(top_y=0,bottom_y=100,suggestion='Clay-S')]}
    c = score.coverage(docs, reports, guesses, {('a',1)})
    assert (c['generated_entries'],c['all_entries'],c['training_entries']) == (1,3,2)
    assert c['estimated_row_coverage'] == .4 and c['training_estimated_row_coverage'] == .8
    assert c['labels'][0]['label'] == 'Clay-S'


def test_column_period_can_exceed_half_lane_and_keeps_full_symbols():
    tile = np.full((18,42,3),(255,0,0),np.uint8)
    tile[3:10,4:7] = 0
    tile[8:11,4:15] = 0
    tile[11:17,25:28] = 0
    tile[15:17,25:35] = 0
    page = make_page(tile,200,61)
    assert learn.horizontal_period(page) == 42


def test_stripe_dropouts_are_outvoted_but_dashes_are_preserved():
    tile = np.full((26,43,3),(128,255,128),np.uint8)
    tile[::5] = 0
    tile[:,14] = (128,255,128)
    _,data,meta = candidates.swatch(tile)
    assert not meta['seam_risk']
    assert data['tile'].shape[1] == 1
    assert (data['tile'].max(2)<80).any()
    dash = np.full((12,28,3),(128,255,128),np.uint8)
    dash[2:4,2:14] = 0
    dash[8:10,16:26] = 0
    assert learn.horizontal_period(make_page(dash,96,64)) == 28


def test_evidence_trims_a_different_motif_with_same_background():
    tile = np.full((8,12,3),(255,255,0),np.uint8)
    tile[1:3,2:4] = 0
    crop = make_page(tile,64,48)
    crop[:6] = (255,255,0)
    crop[:6,8:12] = 0
    data = dict(tile=tile,bg=learn.background(tile))
    candidates.pixel_match(crop,data)
    pure,top = candidates.trim_evidence_symbols(crop,data)
    assert top == 6 and pure.shape[0] == 58


def test_partial_frames_removed_without_erasing_striped_fill():
    from genlab.swatch import remove_frame
    tile = np.full((12, 10, 3), (180, 210, 140), np.uint8)
    tile[3:5, 3:7] = 0
    interior = make_page(tile, 48, 40)
    for thickness in (1, 2, 3):
        # A frame on only left and bottom, with a 1px scanner margin at left.
        framed = np.full((48 + thickness, 41 + thickness, 3), 255, np.uint8)
        framed[:48, 1 + thickness:] = interior
        framed[:, 1:1 + thickness] = 0
        framed[48:] = 0
        cleaned, data, meta = candidates.swatch(framed)
        assert np.array_equal(cleaned, interior)
        assert not meta['seam_risk']
        output = render.layer(data, 0, 120, 0, 96, np.random.default_rng(0))
        assert not (output.max(2) < 80).all(0).any()
        assert candidates.pixel_match(output, dict(tile=tile, bg=learn.background(tile))) == 1
    stripe = np.full((40, 40, 3), 210, np.uint8)
    stripe[:, :3] = 0
    stripe[:, 12:15] = 0
    stripe[:, 24:27] = 0
    cleaned, trims = remove_frame(stripe)
    assert trims == [0, 0, 0, 0] and np.array_equal(cleaned, stripe)


def test_one_cycle_edges_and_nonmatching_wrap():
    from genlab.swatch import edge_continuity
    single = np.full((19, 23, 3), (110, 190, 230), np.uint8)
    single[3:7, 4:8] = 0
    single[11:15, 13:18] = 0
    assert edge_continuity(single, 0)['ok']
    assert edge_continuity(single, 1)['ok']
    _, data, repeat = candidates.swatch(single)
    assert repeat['observed_cycles'] == 1 and not repeat['seam_risk']
    assert np.array_equal(data['tile'], single)
    single[-1] = (240, 100, 110)
    assert not edge_continuity(single, 0)['ok']
    assert candidates.swatch(single)[2]['seam_risk']
    single[:, -1] = (240, 100, 110)
    assert not edge_continuity(single, 1)['ok']


def test_borrow_identical_synonym_and_reject_different_colour():
    tile = np.full((12, 10, 3), (240, 200, 80), np.uint8)
    tile[3:5, 2:7] = 0
    donor = dict(doc_id='donor', name='Donor log',
                 entry=dict(legend_id=2, label='Sandstone', tier='verified', evidence=[], data_file='donor.npz'),
                 data=dict(kind='tile', tile=tile, bg=learn.background(tile), phase=(0, 0), anchored=False))
    entry = dict(legend_id=1, label='Sand and Sandstone', tier='swatch', matches=0, reason='swatch')
    rgb = make_page(np.roll(tile, 3, 0), 36, 30)
    report, learned = dict(entries=[entry]), {}
    candidates.borrow(dict(id='target'), report, learned, {1: rgb}, [donor])
    assert entry['tier'] == 'borrowed' and entry['donor']['match'] == 1
    assert entry['donor']['doc_id'] == 'donor' and entry['donor']['legend_id'] == 2
    assert np.array_equal(learned[1]['tile'], tile)
    entry.update(tier='swatch')
    different = rgb.copy()
    different[(different.max(2) > 80)] = (100, 180, 240)
    assert candidates.find_donor('target', entry, different, [donor]) is None
    entry.update(label='Sand')
    assert candidates.find_donor('target', entry, rgb, [donor]) is None
    entry.update(label='Sandstone', tier='ambiguous')
    assert candidates.find_donor('target', entry, rgb, [donor]) is None
    assert candidates.name_key('Clay-shale') == candidates.name_key('Clay shale')


def test_accessory_exclusions_are_conservative():
    rgb = np.full((40, 40, 3), 255, np.uint8)
    rgb[18:21, 18:21] = 0
    for label in ('Oil_Good', 'Gas_Good', 'Fossils', 'Chert nodules', 'Pyrite'):
        assert candidates.not_lithology(label, rgb)
    fill = np.full_like(rgb, (180, 210, 230))
    for label in ('Chert', 'Bitumen', 'Coal', 'Dolomite'):
        assert candidates.not_lithology(label, fill) is None


def test_borrowed_build_uses_donor_tile_and_verdicts(tmp_path, monkeypatch):
    import copy
    import json
    from genlab import build, paths
    monkeypatch.setattr(paths, 'PROPOSALS', tmp_path/'proposals')
    donor, crops = candidate_page()
    donor['legend'][0]['label_printed'] = 'Sandstone'
    target = copy.deepcopy(donor)
    target.update(id='target', name='Target log')
    target['legend'] = [target['legend'][0]]
    target['legend'][0]['label_printed'] = 'Sand and Sandstone'
    target_crops = FakeCrops(crops.page.copy())
    target_crops.page[:, :64] = 255
    class ByDocument:
        def get(self, doc, box):
            return (target_crops if doc['id'] == 'target' else crops).get(doc, box)
    all_crops = ByDocument()
    docs = {doc['id']: dict(doc=doc, name=doc['name'], training_allowed=True,
                           column=doc['boxes']['interpreted_lithology_column'],
                           legend={e['id']: e['label_printed'] for e in doc['legend']})
            for doc in (donor, target)}
    for name, value in dict(OUT=tmp_path, GEN=tmp_path / 'generated', STATE=tmp_path / 'state.json',
                            CANDIDATES=tmp_path / 'candidates', VERDICTS=tmp_path / 'verdicts',
                            RATINGS=tmp_path / 'ratings.jsonl').items():
        monkeypatch.setattr(paths, name, value)
    candidates.refresh(donor, all_crops, {}, [])
    report = candidates.refresh(target, all_crops, {}, [], donors=candidates.donor_pool(), docs=docs)
    assert report['entries'][0]['tier'] == 'borrowed'
    monkeypatch.setattr(build.labstore, 'Store', lambda: object())
    monkeypatch.setattr(build.labstore, 'Images', lambda: None)
    monkeypatch.setattr(learn, 'Crops', lambda images: all_crops)
    monkeypatch.setattr(learn, 'human_examples', lambda store: (docs, {}))
    monkeypatch.setattr(score, 'load_prefill', lambda: {})
    monkeypatch.setattr(score, 'queue_guesses', lambda: {})
    monkeypatch.setattr(score, 'reviewed_boxes', lambda *args: [])
    builder = build.Builder()
    state = builder.build()
    assert state['borrowed_pattern_count'] == 1
    card = next(c for c in state['cards'] if c['source'] == 'borrowed')
    assert card['doc_id'] == 'target' and card['real_doc_id'] == donor['id']
    data = builder.cache[('target', 1)][2]
    donor_report = __import__('json').loads((paths.CANDIDATES/'d.json').read_text(encoding='utf-8'))
    donor_entry = next(e for e in donor_report['entries'] if e['legend_id']==card['donor']['legend_id'])
    assert card['tile_source'] == donor_entry['model']['tile_source']
    original = candidates.load_data(paths.CANDIDATES / card['donor']['data_file'])
    assert np.array_equal(data['tile'], original['tile'])
    # The same target prefers a current independently approved agent tile.
    write_proposal(paths.PROPOSALS, doc_id='target')
    preferred = builder.build()
    assert next(c['source'] for c in preferred['cards'] if c['doc_id'] == 'target') == 'agent-made'
    write_proposal(paths.PROPOSALS, doc_id='target', round_=2, verdict=None)
    assert builder.build()['borrowed_pattern_count'] == 1
    paths.VERDICTS.mkdir()
    (paths.VERDICTS / 'target.json').write_text(json.dumps({'1': {'verdict': 'reject'}}), encoding='utf-8')
    assert builder.build()['borrowed_pattern_count'] == 0
    (paths.VERDICTS / 'target.json').unlink()
    (paths.VERDICTS / 'd.json').write_text(json.dumps({'1': {'verdict': 'reject'}}), encoding='utf-8')
    assert builder.build()['borrowed_pattern_count'] == 0
    assert not any(d['entry']['legend_id'] == 1 for d in candidates.donor_pool())
    (paths.VERDICTS / 'd.json').unlink()
    docs['d']['training_allowed'] = False
    assert builder.build()['borrowed_pattern_count'] == 0
    docs['d']['training_allowed'] = True
    donor_path = paths.CANDIDATES / 'd.json'
    donor_report = json.loads(donor_path.read_text(encoding='utf-8'))
    donor_report['entries'][0]['tier'] = 'no'
    donor_path.write_text(json.dumps(donor_report), encoding='utf-8')
    assert builder.build()['borrowed_pattern_count'] == 0
