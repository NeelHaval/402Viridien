"""Checks for saved review data and approval gates."""
import copy
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import Store, Conflict, write, read, DATA
from server import CropSource


@pytest.fixture
def store(tmp_path):
    doc={'id':'test','width':100,'height':400,'legend':[{'id':2,'label_printed':'Sand','box':[0,0,20,20]}], 'revision':0,'segments':[]}
    write(tmp_path/'documents.json',[{'id':'test'}])
    write(tmp_path/'seeds/test.json',doc)
    return Store(tmp_path)


def box(**values):
    result=dict(id='sample',box=[10,20,50,100],legend_ids=[2],status='suggested',approved=False,example=False,comments=[])
    result.update(values)
    return result


def test_save_reload_and_source_is_unchanged(store):
    seed=read(store.data/'seeds/test.json')
    assert store.save('test',0,segment=box())==1
    loaded=Store(store.data).get('test')
    assert loaded['segments'][0]['box']==[10,20,50,100]
    assert read(store.data/'seeds/test.json')==seed
    assert len((store.data/'events.jsonl').read_text(encoding='utf-8').splitlines())==1


def test_stale_revision_does_not_replace_review(store):
    store.save('test',0,segment=box())
    with pytest.raises(Conflict):
        store.save('test',0,segment=box(box=[10,40,50,100]))
    assert store.get('test')['segments'][0]['box']==[10,20,50,100]


@pytest.mark.parametrize('values',[
    {'box':[-1,20,50,100]}, {'box':[10,20,50,500]}, {'box':[10,20,10,100]},
    {'box':[10.5,20,50,100]}, {'legend_ids':[99]}, {'legend_ids':[2,2]},
    {'approved':True}, {'status':'approved','approved':True,'legend_ids':[]},
    {'example':True}, {'comments':[{'text':42}]},
])
def test_invalid_edits_do_not_create_a_saved_file(store,values):
    with pytest.raises(ValueError):
        store.save('test',0,segment=box(**values))
    assert store.get('test')['revision']==0
    assert not (store.data/'reviews/test.json').exists()


def test_accepted_box_can_be_used_as_source_and_keep_comment(store):
    comment={'text':'Check the lower edge.'}
    store.save('test',0,segment=box(status='approved',approved=True,example=True,comments=[comment]))
    loaded=store.get('test')['segments'][0]
    assert loaded['example'] and loaded['approved']
    assert loaded['comments']==[comment]


def test_legend_must_keep_ids_used_by_segments(store):
    store.save('test',0,segment=box())
    with pytest.raises(ValueError):
        store.save('test',1,legend=[])
    assert store.get('test')['legend'][0]['id']==2


def test_crop_adapter_keeps_full_source_coordinates():
    from PIL import Image
    class Images:
        def crop(self,doc,coords):
            assert coords==[10,20,50,100]
            return Image.new('RGB',(40,80),'yellow')
    source=CropSource(Images(),{'width':100,'height':400})
    assert isinstance(source,Image.Image)
    assert source.size==(100,400)
    assert source.crop([10,20,50,100]).size==(40,80)


def test_prepared_source_keeps_supplied_label_ids():
    catalog=read(DATA/'documents.json')
    hadrian=next(d for d in catalog if d['supplied'])
    doc=read(DATA/'seeds'/(hadrian['id']+'.json'))
    original=read(DATA/'imported/hadrian.json')
    assert doc['legend']==original['legend']
    assert doc['source_layer_count']==len(original['layers'])==680
    assert all(s['approved'] for s in doc['segments'])
    assert all(s['legend_ids'][0] in {e['id'] for e in original['legend']} for s in doc['segments'])
    assert all(s['approved'] for s in doc['segments'] if s['example'])
