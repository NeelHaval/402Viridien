"""Bounded notebook-rule adapter checks, without executing the notebook itself.

Contract checks do not imply thin-bed, gap, naming or held-out model accuracy.
All images and swatches here are procedural and at most a few thousand pixels.
"""
import warnings

import numpy as np
from PIL import Image
import pytest

pytest.importorskip('scipy',reason='Notebook adapter needs the optional SciPy dependency')
from scipy.ndimage import label

from viridian402 import notebook_ink as model
from viridian402.data import validate_intervals
from viridian402.evaluate import _predict


def _request(tmp_path,rgb,mode='RGB'):
    height,width=rgb.shape[:2];y0=17
    page=np.full((y0+height+4,width+4,3),255,np.uint8)
    page[y0:y0+height,2:2+width]=rgb
    path=tmp_path/('native-'+mode+'.png')
    image=Image.fromarray(page)
    if mode!='RGB':image=image.convert(mode)
    image.save(path)
    return {'id':'owned-procedural','image':str(path),'image_size':[width+4,y0+height+4],
            'window':[y0,y0+height],'annotation':{'interpreted_lithology_column':[2,y0,2+width,y0+height]}}


def test_one_component_center_cap_has_finite_features_and_no_empty_pair_mean():
    rgb=np.full((8,24,3),255,np.uint8);rgb[::2,::2]=0
    assert label(rgb.mean(2)<128,structure=np.ones((3,3)))[1]>1
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning)
        values=model.stripe_features(rgb,max_comp=1)
    assert values.shape==(7,) and np.isfinite(values).all()
    assert values[2]==0  # no pair exists after keeping one center


@pytest.mark.parametrize('cap',[1,2,256,2000])
def test_more_than_two_thousand_actual_components_exercise_sampling_and_seed_stability(cap):
    rgb=np.full((8,1200,3),255,np.uint8);rgb[::2,::2]=0
    assert label(rgb.mean(2)<128,structure=np.ones((3,3)))[1]==2400
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning)
        np.random.seed(1);a=model.stripe_features(rgb,max_comp=cap)
        np.random.seed(54321);b=model.stripe_features(rgb,max_comp=cap)
    np.testing.assert_array_equal(a,b)
    assert a[1]==2400 and a[3]==1 and np.isfinite(a).all()


@pytest.mark.parametrize('height,width',[(1,1),(1,2),(2,1),(7,3),(8,1),(9,2),(17,4)])
@pytest.mark.parametrize('mode',['RGB','L','P','1'])
def test_tiny_native_images_keep_valid_full_image_coordinates(tmp_path,height,width,mode):
    request=_request(tmp_path,np.zeros((height,width,3),np.uint8),mode)
    rows=validate_intervals(model.predict(request,{}),window=request['window'])
    assert rows and rows[0]['top_y']==17 and rows[-1]['bottom_y']==17+height
    assert all(row['lithology']=='Unknown' and 'top_depth' not in row for row in rows)


def test_white_image_and_disjoint_clipped_window_have_no_rows(tmp_path):
    request=_request(tmp_path,np.full((8,8,3),255,np.uint8))
    assert model.predict(request,{})==[]
    request['window']=[0,5]
    assert model.predict(request,{})==[]


@pytest.mark.parametrize('config',[
    {'max_component_centers':0},{'max_component_centers':True},{'max_component_centers':2001},
    {'stripe_height':257},{'boundary_window':0},{'min_gap':False},
    {'ink_k':-1},{'ink_k':np.inf},{'ink_k':np.nan},
    {'cluster_distance':0},{'cluster_distance':np.inf},{'cluster_distance':np.nan},
])
def test_invalid_thresholds_and_sampling_parameters_fail_closed(tmp_path,config):
    request=_request(tmp_path,np.zeros((16,16,3),np.uint8))
    with pytest.raises(ValueError):model.predict(request,config)


@pytest.mark.parametrize('window',[[-1,20],[17,17],[18,17],[17,1000],[False,20],[17.5,20]])
def test_bad_window_coordinates_are_rejected(tmp_path,window):
    request=_request(tmp_path,np.zeros((16,16,3),np.uint8));request['window']=window
    with pytest.raises(ValueError):model.predict(request,{})


def test_window_and_decode_limits_and_invalid_swatch_box_are_enforced(tmp_path):
    request=_request(tmp_path,np.zeros((16,16,3),np.uint8))
    with pytest.raises(ValueError,match='stripe budget'):model.predict(request,{'stripe_height':1,'max_feature_stripes':1})
    with pytest.raises(ValueError,match='max_window_pixels'):model.predict(request,{'max_window_pixels':1})
    with pytest.raises(MemoryError):model.predict(request,{'max_decode_bytes':1})
    request['legend']=[{'label':'Owned rock','swatch_box':[0,0,1000,1000]}]
    with pytest.raises(ValueError):model.predict(request,{})


def test_repeated_supplied_patterns_keep_page_local_ids_and_original_offsets(tmp_path):
    rgb=np.full((340,80,3),255,np.uint8)
    red=(240,140,140);blue=(140,140,240)
    rgb[20:120,10:50]=red;rgb[120:220,10:50]=blue;rgb[220:320,10:50]=red
    rgb[:12,:12]=red;rgb[:12,20:32]=blue
    path=tmp_path/'repeated.png';Image.fromarray(rgb).save(path)
    request={'image':str(path),'image_size':[80,340],'window':[20,320],
             'annotation':{'interpreted_lithology_column':[10,20,50,320]},
             'legend':[{'id':100,'label':'Red rock','swatch_box':[0,0,12,12]},
                       {'id':900,'label':'Blue rock','swatch_box':[20,0,32,12]}]}
    config={'stripe_height':2,'boundary_window':2,'min_gap':2,'ink_k':1,'max_component_centers':1}
    rows=model.predict(request,config)
    assert [(row['top_y'],row['bottom_y']) for row in rows]==[(20,120),(120,220),(220,320)]
    assert [row['cluster_id'] for row in rows]==[0,1,0]
    assert [row['lithology'] for row in rows]==['Red rock','Blue rock','Red rock']


def test_short_white_gap_and_large_rule_window_remain_valid_without_claiming_gap_recall(tmp_path):
    rgb=np.full((64,24,3),(240,140,140),np.uint8);rgb[28:36]=255
    request=_request(tmp_path,rgb)
    for config in ({},{'boundary_window':1000,'ink_k':1e100}):
        rows=model.predict(request,config)
        validate_intervals(rows,window=request['window'])
        assert rows and all(row['lithology']=='Unknown' for row in rows)
    # Short-gap recall is a measured model limitation documented separately;
    # this test does not require preserving an inaccurate prediction forever.


def test_worker_uses_pixels_not_target_paths_page_id_or_global_rng(tmp_path):
    rgb=np.full((64,32,3),255,np.uint8);rgb[::2,::2]=0
    request=_request(tmp_path,rgb)
    request.update(truth='MUST_NOT_OPEN',layers=['MUST_NOT_USE'],depth={'a':2,'b':100})
    config={'max_component_centers':1,'seed':42}
    rows,error,_,_=_predict('viridian402.notebook_ink:predict',request,config,timeout=10,max_worker_rss_mb=256)
    assert error is None
    validate_intervals(rows,window=request['window'])
    request['id']='another-page-name'
    np.random.seed(999)
    assert rows==model.predict(request,config)
    assert all(row['lithology']=='Unknown' and 'top_depth' not in row for row in rows)
