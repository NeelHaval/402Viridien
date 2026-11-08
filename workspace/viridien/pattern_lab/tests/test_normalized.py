import copy,json,sys,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import Store,write,read
from normalized import SplitPolicy,NormalizedDataset,letterbox,group_key
from review_queue import ReviewQueue
from review_media import ReviewMedia
from background import Background

class Pages:
    def __init__(self):self.same=False
    def crop(self,doc,box):
        page=Image.new('RGB',(60,300),'yellow' if doc['id']=='train' or self.same else 'lime');p=ImageDraw.Draw(page)
        for y in range(0,300,12):p.line((4,y,18,y),fill='black',width=2)
        return page.crop(box)

@pytest.fixture
def setup(tmp_path):
    catalog=[]
    for ident in ('train','test'):
        s=dict(id='human',box=[0,60,60,120],legend_ids=[1],approved=True,status='approved',example=True,comments=[dict(text='Human note')])
        doc=dict(id=ident,name=ident,width=60,height=300,revision=0,image_name=ident+'.tif',boxes={'interpreted_lithology_column':[0,0,60,300]},legend=[dict(id=1,label_printed='Sand and Sandstone',box=[0,0,60,24])],segments=[s])
        write(tmp_path/'seeds'/(ident+'.json'),doc);catalog.append(dict(id=ident,name=ident))
    write(tmp_path/'documents.json',catalog);q=ReviewQueue(Store(tmp_path),dict(groups=[],references=[]));pages=Pages();media=ReviewMedia(q,pages);policy=SplitPolicy(q.store);policy.split=lambda d:d['id'];dataset=NormalizedDataset(q,media,policy)
    return q,pages,media,policy,dataset

def test_related_well_scales_and_sidetracks_have_one_split_group():
    assert group_key('60811406050100_ardennes_5inch')==group_key('60811406050000_master_1inch')
    assert group_key('0000000009_d-582_foo_log01')==group_key('0000000009_d-582_other_log07')
    assert group_key('MC765_P011_ST00BP00_1inch')==group_key('MC765_P011_ST01BP00_5inch')

def test_letterbox_has_valid_mask_and_reversible_transform():
    image=Image.new('RGB',(40,100),'yellow');rgb,mask,t=letterbox(image)
    assert rgb.size==(224,224) and mask.mode=='L'
    x0,y0,x1,y1=t['content_xyxy'];assert y0==0 and y1==224 and x1-x0==90
    assert np.count_nonzero(np.asarray(mask))==90*224
    assert tuple(np.asarray(rgb)[112,112])==(255,255,0)
    assert t['scale_xy']==[90/40,224/100]

def test_export_separates_real_tests_from_generator_sources(setup):
    q,pages,media,policy,dataset=setup;m=dataset.build()
    assert m['stats']==dict(accepted=2,train=1,test=1,needs_review=0,test_groups=1)
    assert all(r['kind']=='real' and r['comments'][0]['text']=='Human note' for r in m['records'])
    path=dataset.archive(m)
    with zipfile.ZipFile(path) as z:
        train=[json.loads(l) for l in z.read('train.jsonl').decode().splitlines()]
        test=[json.loads(l) for l in z.read('test.jsonl').decode().splitlines()]
        sources=json.loads(z.read('generator-sources.json'))
        assert {r['document_id'] for r in train}=={'train'}
        assert {r['document_id'] for r in test}=={'test'}
        assert {r['sample_id'] for r in sources}=={train[0]['sample_id']}
        assert train[0]['targets'][0]['id']==test[0]['targets'][0]['id']
        assert all(r['native_png'] in z.namelist() and r['valid_mask_png'] in z.namelist() for r in test+train)
    assert policy.allowed(q.store.get('train'),q.store.get('train')['segments'][0])
    assert not policy.allowed(q.store.get('test'),q.store.get('test')['segments'][0])

def test_exact_test_duplicate_is_withheld_from_training(setup):
    q,pages,media,policy,dataset=setup;pages.same=True;m=dataset.build()
    assert m['stats']['train']==0
    row=next(r for r in m['records'] if r['document_id']=='train')
    assert 'duplicate_of_test' in row['quality_issues']

def test_wrong_legend_fill_is_saved_but_not_reused(setup):
    q,pages,media,policy,dataset=setup
    original=pages.crop
    def crop(doc,box):
        if doc['id']=='train' and box[1]>30:return Image.new('RGB',(box[2]-box[0],box[3]-box[1]),'lime')
        return original(doc,box)
    pages.crop=crop;m=dataset.build();row=next(r for r in m['records'] if r['document_id']=='train')
    assert 'legend_fill_mismatch' in row['quality_issues'] and not row['usable_for_generation']
    assert q.store.get('train')['segments'][0]['approved']

def test_correction_keeps_id_and_old_export_but_invalidates_old_source(setup):
    q,pages,media,policy,dataset=setup;first=dataset.build();old=next(r for r in first['records'] if r['document_id']=='train')
    doc=q.store.get('train');seg=copy.deepcopy(doc['segments'][0]);seg['comments']=[dict(text='Corrected note')];q.store.save('train',0,segment=seg)
    assert not policy.allowed(q.store.get('train'),seg)
    second=dataset.build();new=next(r for r in second['records'] if r['document_id']=='train')
    assert old['sample_id']==new['sample_id'] and old['annotation_version']!=new['annotation_version']
    assert first['snapshot']!=second['snapshot'] and dataset.archive(first).exists()

def test_split_policy_is_fixed_and_records_prior_exposure(setup):
    q,pages,media,policy,dataset=setup;restored=SplitPolicy(q.store)
    assert restored.config==policy.config
    assert group_key('train') in restored.config['previously_seen_groups']

def test_background_updates_do_not_modify_human_labels(setup):
    q,pages,media,policy,dataset=setup;before=[q.store.get(i) for i in ('train','test')]
    bg=Background(q,media,autostart=False,images=pages);bg.policy.split=lambda d:d['id'];bg.run_once()
    assert bg.status()['dataset']['accepted']==2 and bg.status()['error'] is None
    assert [q.store.get(i) for i in ('train','test')]==before
    assert len(media.feedback.refs)==1 and media.feedback.refs[0]['document_id']=='train'

def test_rejected_and_draft_samples_never_enter_real_test_data(setup):
    q,pages,media,policy,dataset=setup;doc=q.store.get('test');seg=copy.deepcopy(doc['segments'][0]);seg.update(approved=False,status='suggested',example=False);q.store.save('test',0,segment=seg)
    m=dataset.build();assert m['stats']['test']==0 and len(m['records'])==1

def test_draft_typing_does_not_create_new_dataset_archives(setup):
    q,pages,media,policy,dataset=setup;first=dataset.build();fingerprint=dataset.fingerprint()
    doc=q.store.get('train');draft=dict(id='draft',box=[0,180,60,220],legend_ids=[],approved=False,status='suggested',example=False,comments=[dict(text='Still typing')]);q.store.save('train',0,segment=draft)
    assert dataset.fingerprint()==fingerprint
    assert dataset.build()['snapshot']==first['snapshot']
    assert len(list((q.store.data/'exports').glob('normalized-*.zip')))==1

def test_generator_bridge_uses_only_train_sources_and_exports_same_label_ids(setup,monkeypatch,tmp_path):
    from genlab import build,paths,score
    q,pages,media,policy,dataset=setup;m=dataset.build()
    monkeypatch.setattr(SplitPolicy,'split',lambda self,d:d['id'])
    monkeypatch.setattr(build.labstore,'Store',lambda:q.store)
    monkeypatch.setattr(build.labstore,'Images',lambda:pages)
    monkeypatch.setattr(score,'load_prefill',lambda:{})
    monkeypatch.setattr(score,'queue_guesses',lambda:{})
    out=tmp_path/'genlab'
    for name,value in dict(OUT=out,GEN=out/'generated',CROPS=out/'crops',STATE=out/'state.json',RATINGS=out/'ratings.jsonl',CANDIDATES=out/'candidates',VERDICTS=out/'verdicts').items():monkeypatch.setattr(paths,name,value)
    write(paths.CANDIDATES/'test.json',dict(signature='wrong',entries=[dict(tier='verified',legend_id=1)]))
    builder=build.Builder();state=builder.build();manifest=read(paths.GEN/'manifest.json')
    assert state['human_boxes']==1 and manifest['kind']=='synthetic' and manifest['split']=='train'
    row=next(r for r in m['records'] if r['document_id']=='train')
    assert manifest['patterns']
    for sample in manifest['patterns']:
        assert sample['targets'][0]['id']==row['targets'][0]['id']
        assert sample['normalized_sources']==[dict(sample_id=row['sample_id'],annotation_version=row['annotation_version'])]
        assert sample['kind']=='synthetic' and sample['split']=='train'
        assert Image.open(paths.GEN/sample['normalized_png']).size==(224,224)
        assert Image.open(paths.GEN/sample['valid_mask_png']).mode=='L'


def test_missing_derived_image_is_rebuilt_from_source(setup):
    q,pages,media,policy,dataset=setup;first=dataset.build();row=first['records'][0]
    (dataset.root/row['valid_mask_png']).unlink()
    second=dataset.build()
    assert second['snapshot']==first['snapshot']
    assert Image.open(dataset.root/row['valid_mask_png']).size==(224,224)
