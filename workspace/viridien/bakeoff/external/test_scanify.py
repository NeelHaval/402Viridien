"""Geometry contracts: exact rows, enclosed boxes, unchanged physical depths."""
import numpy as np
import pytest
from bakeoff.external.scanify import geometry, transform_box, transform_rows, degrade

@pytest.mark.parametrize('scale',[.13,.4,1.,1.7])
@pytest.mark.parametrize('angle',[-.8,0,.8])
def test_boxes_and_truth_aligned(scale,angle):
    g=geometry((120,1000),scale,angle)
    box=[20,100,70,900]
    rows=[dict(top_y=100,bottom_y=500,lithology='Sand',top_depth=10.,bottom_depth=50.),
          dict(top_y=500,bottom_y=900,lithology='Shale',top_depth=50.,bottom_depth=90.)]
    transformed=transform_rows(rows,g);bounds=transform_box(box,g)
    assert bounds[1]==transformed[0]['top_y']
    assert bounds[3]==transformed[-1]['bottom_y']
    assert transformed[0]['bottom_y']==transformed[1]['top_y']
    assert transformed[0]['top_depth']==10.
    assert transformed[1]['bottom_depth']==90.
    for x in (20,70):
        for y in (bounds[1],bounds[3]):
            tx=x*g['sx']+g['shear']*y+g['offset']
            assert bounds[0]<=tx<=bounds[2]
    assert rows[0]['bottom_y']==500


def test_raster_boundary_alignment():
    import cv2
    g=geometry((80,400),.55,.6)
    rgb=np.zeros((400,80,3),np.uint8);rgb[200:]=255
    m=np.array([[g['sx'],g['shear']*g['sy'],g['offset']],[0,g['sy'],0]],np.float32)
    warped=cv2.warpAffine(rgb,m,(g['width'],g['height']))
    transformed=transform_rows([dict(top_y=0,bottom_y=200),dict(top_y=200,bottom_y=400)],g)
    boundary=transformed[0]['bottom_y']
    assert warped[boundary-1,20,0]<128
    assert warped[boundary,20,0]>128


def test_erased_interval_rejected():
    with pytest.raises(ValueError,match='erase'):
        transform_rows([dict(top_y=0,bottom_y=1)],geometry((20,100),.1,0))


def test_seed_and_one_bit():
    rgb=np.full((120,100,3),180,np.uint8)
    p=dict(blur_sigma=.5,saturation=.7,fade=.1,colour_shift=[1.,.97,.9],noise_sigma=6.,jpeg_quality=60,one_bit=False)
    assert np.array_equal(degrade(rgb,p,42),degrade(rgb,p,42))
    assert not np.array_equal(degrade(rgb,p,42),degrade(rgb,p,43))
    bits=degrade(rgb,dict(p,one_bit=True),42)
    assert set(np.unique(bits))=={0,255}


@pytest.mark.parametrize('target_ppf,expected_height',[(4.125,400),(12.75,680)])
def test_build_roundtrip(tmp_path,target_ppf,expected_height):
    import csv
    import json
    from PIL import Image
    from bakeoff.external.scanify import build, dump
    source=tmp_path/'source';out=tmp_path/'out'
    (source/'images').mkdir(parents=True)
    image=np.zeros((400,80,3),np.uint8);image[200:]=255
    Image.fromarray(image).save(source/'images'/'example.tif')
    dump(source/'bbox_annotation'/'example.json',dict(image_name='example.tif',width=80,height=400,bounding_boxes=dict(interpreted_lithology_column=[10,0,60,400])))
    from bakeoff.external.scanify import io
    io.save_rows(source/'truth'/'example.csv',[dict(top_y=0,bottom_y=200,lithology='A',top_depth=0,bottom_depth=20),dict(top_y=200,bottom_y=201,lithology='B',top_depth=20,bottom_depth=20.1),dict(top_y=201,bottom_y=400,lithology='C',top_depth=20.1,bottom_depth=40)])
    for split in ('dev','test'):dump(source/f'eval_set_{split}.json',[dict(log='example',y0=0,y1=400)])
    with (source/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','px_per_ft']);writer.writeheader();writer.writerow(dict(id='example',px_per_ft=7.5))
    p=dict(px_per_ft=target_ppf,skew_deg=.6,blur_sigma=.25,saturation=1.,fade=0.,colour_shift=[1.,1.,1.],noise_sigma=0.,jpeg_quality=95,one_bit=False)
    dump(out/'presets.json',dict(seed=42,levels=dict(light=p)))
    build(source,out)
    truth=io.load_rows(out/'light/truth/example.csv')
    annotation=json.loads((out/'light/bbox_annotation/example.json').read_text())
    with Image.open(out/'light/images/example.tif') as im:
        pixels=np.asarray(im)
        assert im.height==expected_height
        assert pixels[truth[0]['bottom_y']-3,20].mean()<40
        assert pixels[truth[1]['top_y']+3,20].mean()>220
    assert annotation['bounding_boxes']['interpreted_lithology_column'][3]==truth[-1]['bottom_y']
    assert truth[-1]['bottom_depth']==40
    assert len(truth)==3
    assert truth[1]['bottom_y']>truth[1]['top_y']
    assert json.loads((out/'light/eval_set_test.json').read_text())[0]['y1']==expected_height
    assert io.load_rows(source/'truth/example.csv')[0]['bottom_y']==200


def test_statistics_for_axis_lines():
    from bakeoff.external.scanify import stats
    rgb=np.full((180,160,3),255,np.uint8)
    rgb[::20]=0;rgb[:,::20]=0
    result=stats(rgb)
    assert result['laplacian_variance']>0
    assert abs(result['skew_deg'])<.1
    assert result['saturation']==0


def test_final_evaluation_is_blocked_before_io(tmp_path):
    from bakeoff.external.scanify import evaluate
    with pytest.raises(RuntimeError, match='orchestrator'):
        evaluate(tmp_path/'missing', tmp_path/'missing_out', 'test')
    assert not (tmp_path/'missing_out').exists()


def test_stale_dev_list_is_rejected(tmp_path):
    from bakeoff.external.scanify import checked_items, dump
    dump(tmp_path/'eval_set_dev.json', [dict(log='ext_kgs_ForsythB2-17_p015', y0=0, y1=100)])
    with pytest.raises(ValueError, match='verify'):
        checked_items(tmp_path, 'dev')
    with pytest.raises(ValueError, match='explicit'):
        checked_items(tmp_path, 'dev_full')


def test_refresh_removes_old_split_without_reading_truth(tmp_path):
    import csv
    import json
    from bakeoff.external.scanify import LEVELS, dump, refresh
    source=tmp_path/'source';out=tmp_path/'out'
    dump(source/'eval_set_dev.json', [dict(log='dev_page',y0=2,y1=10)])
    dump(source/'eval_set_test.json', [dict(log='test_page',y0=3,y1=12)])
    for level in LEVELS:
        root=out/level
        dump(root/'geometry.json', {name:dict(sy=2.) for name in ('dev_page','test_page','old_page')})
        # Invalid bytes must never be parsed by the refresh step.
        for folder,suffix in (('images','.tif'),('bbox_annotation','.json'),('truth','.csv')):
            (root/folder).mkdir()
            for name in ('dev_page','test_page','old_page'):
                (root/folder/(name+suffix)).write_bytes(b'not parsed')
        with (root/'manifest.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['id']);writer.writeheader()
            writer.writerows(dict(id=name) for name in ('dev_page','test_page','old_page'))
    refresh(source,out)
    for level in LEVELS:
        root=out/level
        assert not (root/'truth/old_page.csv').exists()
        assert (root/'truth/test_page.csv').read_bytes()==b'not parsed'
        assert json.loads((root/'eval_set_dev.json').read_text())[0]['y1']==20
        assert json.loads((root/'eval_set_test.json').read_text())[0]['y1']==24
        assert len(json.loads((root/'geometry.json').read_text()))==2
