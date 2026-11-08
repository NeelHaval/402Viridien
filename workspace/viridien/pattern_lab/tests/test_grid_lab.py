import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grid_lab import grid_image, request_hash, validate_response
from grid_publish import prediction
import grid_lab


def fixture():
    batch=dict(batch_id='b',shape=[1,2],item_ids=['E1'],allowed_ids=[1,2])
    result=dict(batch_id='b',matrix=[[dict(item_id='E1',legend_ids=[1],status='match',confidence='high'),None]])
    return batch,result


def test_strict_row_major_matrix_maps_one_cell_and_padding():
    batch,result=fixture()
    assert validate_response(batch,result)==[result['matrix'][0][0]]


@pytest.mark.parametrize('kind',['foreign','duplicate','wrong_item','missing','padding','wrong_batch','empty_match','status','confidence'])
def test_bad_matrix_cannot_enter_proposals(kind):
    batch,result=fixture();cell=result['matrix'][0][0]
    if kind=='foreign':cell['legend_ids']=[99]
    if kind=='duplicate':cell['legend_ids']=[1,1]
    if kind=='wrong_item':cell['item_id']='E2'
    if kind=='missing':result['matrix'][0].pop()
    if kind=='padding':result['matrix'][0][1]=copy.deepcopy(cell)
    if kind=='wrong_batch':result['batch_id']='other'
    if kind=='empty_match':cell['legend_ids']=[]
    if kind=='status':cell['status']='approved'
    if kind=='confidence':cell['confidence']=.99
    with pytest.raises(ValueError):validate_response(batch,result)


def test_uncertain_can_abstain_or_name_twins():
    batch,result=fixture();cell=result['matrix'][0][0]
    cell.update(status='uncertain',legend_ids=[])
    assert validate_response(batch,result)[0]['legend_ids']==[]
    cell['legend_ids']=[1,2]
    assert validate_response(batch,result)[0]['legend_ids']==[1,2]


def test_cache_key_changes_with_pixels_ids_or_matrix_shape(tmp_path):
    image=tmp_path/'image.png';image.write_bytes(b'pixels')
    batch=dict(shape=[1,2],item_ids=['E1'],allowed_ids=[1,2],grid=str(image),legends=[str(image)])
    original=request_hash(batch)
    batch['allowed_ids']=[2,3];assert request_hash(batch)!=original
    batch['allowed_ids']=[1,2];batch['shape']=[2,1];assert request_hash(batch)!=original
    batch['shape']=[1,2];image.write_bytes(b'new pixels');assert request_hash(batch)!=original


def test_terra_proposal_preserves_crop_bounds_and_never_approves():
    item=dict(source_box=[10,20,40,60]);doc=dict(name='A',legend=[dict(id=1,label_printed='Shale'),dict(id=2,label_printed='Sand')])
    base=dict(box=[10,25,40,55],source_version='new',prediction=dict(choices=[dict(legend_ids=[1],label='Shale')],family='A'))
    cell=dict(legend_ids=[1],status='match',confidence='high',reason='Dashes')
    p=prediction(item,cell,base,doc,'hash')
    assert p['box']==item['source_box']
    assert p['prediction']['model_agreement']
    assert not p['prediction']['ambiguous']
    assert 'segment' not in p and 'approved' not in p
    cell.update(legend_ids=[],status='bad_crop',confidence='low')
    p=prediction(item,cell,base,doc,'hash')
    assert p['prediction']['ambiguous']
    assert p['prediction']['terra_status']=='bad_crop'
    assert base['box']==[10,25,40,55]


def test_collect_rejects_old_responses_after_source_changes(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(grid_lab,'OUT',tmp_path)
    batch,result=fixture();image=tmp_path/'image.png';image.write_bytes(b'original')
    batch.update(grid=str(image),legends=[str(image)])
    result['input_hash']=request_hash(batch)
    (tmp_path/'blind').mkdir()
    (tmp_path/'blind/packets.json').write_text(json.dumps(dict(batches=[batch])),encoding='utf-8')
    (tmp_path/'result-base.json').write_text(json.dumps(dict(batches=[result])),encoding='utf-8')
    grid_lab.collect()
    assert len(json.loads((tmp_path/'terra-results.json').read_text(encoding='utf-8'))['batches'])==1
    image.write_bytes(b'changed pixels')
    grid_lab.collect()
    assert json.loads((tmp_path/'terra-results.json').read_text(encoding='utf-8'))['batches']==[]
    assert len(json.loads((tmp_path/'retry-batches.json').read_text(encoding='utf-8'))['batches'])==1


@pytest.mark.parametrize('count',[32,64,128])
def test_large_grid_stays_within_image_limits_and_keeps_all_slots(tmp_path,count):
    from PIL import Image
    Image.new('RGB',(50,96),'yellow').save(tmp_path/'crop.png')
    items=[dict(item_id=f'N{i:03}',crop='crop.png') for i in range(count)]
    path=tmp_path/'grid.png';rows,cols=grid_image(items,tmp_path,path)
    with Image.open(path) as image:
        assert image.width==1024 and image.height<=1024
    assert rows*cols==count
