"""Bounded demo-pipeline contracts; no app startup, OCR, training or downloads."""
from pathlib import Path
import sys

import numpy as np
import pytest

pd=pytest.importorskip('pandas',reason='Demo pipeline checks need the backend environment')
pytest.importorskip('scipy')
pytest.importorskip('cv2')
BACKEND=Path(__file__).resolve().parents[2]/'demo/backend'
if str(BACKEND) not in sys.path:sys.path.insert(0,str(BACKEND))

from app.pipeline.schema import check_output_schema
from app.pipeline.stages import features, legend as legend_stage
from app.pipeline.stages.boundaries import find_beds
from app.pipeline.stages.depth import build_table
from app.pipeline.stages.load_column import load_column
from app.pipeline.stages.naming import describe, descriptor_distance, name_layers
from app.pipeline.types import ColumnImage, Legend, LegendEntry, OutputSchemaError, PipelineError
from app.tools.preprocess import Calibration


def _table():
    return pd.DataFrame({'top_y':[0],'bottom_y':[10],'cluster_id':[0],'lithology':['Shale'],
                         'top_depth':[100.0],'base_depth':[101.0]})


@pytest.mark.parametrize('column,value',[
    ('top_y',-1),('top_y',.5),('bottom_y',10.5),('cluster_id',.5),
    ('top_y',-np.inf),('bottom_y',np.inf),('cluster_id',np.inf),('cluster_id',np.nan),
    ('top_y',1+2j),('cluster_id',1+2j),('top_depth',100+2j),
    ('cluster_id',np.uint64(2**63)),('bottom_y',np.uint64(2**63)),
])
def test_schema_rejects_invalid_pixel_cluster_and_real_number_contract(column,value):
    table=_table()
    table[column]=pd.Series([value])
    with pytest.raises(OutputSchemaError):check_output_schema(table)


@pytest.mark.parametrize('value',['','   ',42,{},False])
def test_schema_requires_each_lithology_to_be_nonempty_text(value):
    table=_table();table['lithology']=pd.Series([value],dtype=object)
    with pytest.raises(OutputSchemaError):check_output_schema(table)


def test_schema_preserves_large_integer_precision_and_nullable_integer_columns():
    table=_table();table['top_y']=pd.Series([2**53+1],dtype='int64')
    table['bottom_y']=pd.Series([2**53+2],dtype='int64')
    table['cluster_id']=pd.Series([0],dtype='Int64')
    before=table.copy(deep=True)
    check_output_schema(table)
    pd.testing.assert_frame_equal(table,before)
    check_output_schema(_table().assign(cluster_id=-1))  # signed page-local IDs are allowed


@pytest.mark.parametrize('height,width',[(2,2),(3,2),(7,3),(8,1),(9,2),(17,4)])
def test_tiny_rgb_features_and_beds_are_finite_and_inside_full_image_rows(height,width):
    rgb=np.full((height,width,3),255,np.uint8);rgb[::2,::2]=0
    column=ColumnImage(rgb,137)
    feats=features.compute_features(column)
    assert feats.matrix.shape==((height+7)//8,7) and np.isfinite(feats.matrix).all()
    beds=find_beds(column,feats)
    assert beds.y_top.iloc[0]==137 and beds.y_bottom.iloc[-1]==137+height
    assert (beds.y_bottom>beds.y_top).all()
    for patch in (rgb,np.zeros_like(rgb),np.full_like(rgb,255)):
        d=describe(patch)
        assert np.isfinite(d['lab']).all() and np.isfinite(descriptor_distance(d,d))


def test_raw_component_subsampling_branch_really_has_more_than_two_thousand_components():
    rgb=np.full((8,1200,3),255,np.uint8);rgb[::2,::2]=0
    from scipy.ndimage import label
    assert label(rgb.mean(2)<128,structure=np.ones((3,3)))[1]==2400
    np.random.seed(1);a=features.stripe_features(rgb)
    np.random.seed(98765);b=features.stripe_features(rgb)
    assert a[1]==2400 and np.isfinite(a).all()
    np.testing.assert_array_equal(a,b)
    assert a[2]>0  # raw vectors, not a one-stripe zscore that is always zero


def test_exact_swatches_name_repeated_patterns_without_reading_other_metadata():
    red=np.full((32,24,3),(240,140,140),np.uint8)
    blue=np.full((32,24,3),(140,140,240),np.uint8)
    column=ColumnImage(np.concatenate([red,blue,red]),100)
    beds=pd.DataFrame({'y_top':[100,132,164],'y_bottom':[132,164,196]})
    entries=Legend([LegendEntry(7,'Red rock',(0,0,24,32),red),LegendEntry(3,'Blue rock',(0,0,24,32),blue)])
    result=name_layers(beds,column,entries)
    assert result.legend_id.tolist()==[7,3,7]
    assert result.lithology.tolist()==['Red rock','Blue rock','Red rock']
    assert name_layers(beds,column,Legend()).lithology.tolist()==['Unknown']*3


def test_depth_mapping_uses_full_image_coordinates_and_drops_only_blank_layers():
    layers=pd.DataFrame({'y_top':[100,110,120],'y_bottom':[110,120,130],
                         'legend_id':[2,-1,2],'lithology':['Shale','blank','Shale']})
    table=build_table(layers,Calibration(.25,1000,0,3,3))
    assert table.top_y.tolist()==[100,120] and table.top_depth.tolist()==[1025,1030]
    assert table.base_depth.tolist()==[1027.5,1032.5]
    check_output_schema(table)


def test_depth_precision_keeps_positive_one_pixel_bed_at_supported_small_slope():
    layers=pd.DataFrame({'y_top':[100],'y_bottom':[101],'legend_id':[0],'lithology':['Shale']})
    table=build_table(layers,Calibration(.02,0,0,3,3))
    assert table.base_depth.iloc[0]>table.top_depth.iloc[0], 'rounding must not turn a valid 0.02-depth bed into zero thickness'
    check_output_schema(table)


@pytest.mark.parametrize('column,value',[('y_top',.5),('y_bottom',np.inf),('legend_id',1.5)])
def test_depth_stage_rejects_invalid_geometry_or_ids_without_truncation(column,value):
    layers=pd.DataFrame({'y_top':[0],'y_bottom':[10],'legend_id':[0],'lithology':['Shale']})
    layers[column]=pd.Series([value])
    with pytest.raises(PipelineError):build_table(layers,Calibration(.25,1000,0,3,3))


def test_depth_stage_retains_signed64_pixels_on_windows():
    layers=pd.DataFrame({'y_top':[2**31+5],'y_bottom':[2**31+15],'legend_id':[2],'lithology':['Shale']})
    table=build_table(layers,Calibration(.25,1000,0,3,3))
    assert table.top_y.iloc[0]==2**31+5 and table.bottom_y.iloc[0]==2**31+15


@pytest.mark.parametrize('value',[None,42,''])
def test_depth_stage_does_not_turn_invalid_labels_into_strings(value):
    layers=pd.DataFrame({'y_top':[0],'y_bottom':[10],'legend_id':[2],'lithology':[value]})
    with pytest.raises(PipelineError):build_table(layers,Calibration(.25,1000,0,3,3))


def test_thin_bed_precision_fallback_keeps_shared_neighbour_boundaries_identical():
    layers=pd.DataFrame({'y_top':[0,100,101],'y_bottom':[100,101,200],
                         'legend_id':[0,1,0],'lithology':['Sand','Shale','Sand']})
    table=build_table(layers,Calibration(.02,0,0,3,3))
    assert table.base_depth.iloc[1]==pytest.approx(2.02)
    np.testing.assert_array_equal(table.base_depth.to_numpy()[:-1],table.top_depth.to_numpy()[1:])
    check_output_schema(table)


@pytest.mark.parametrize('slope,intercept',[(0,0),(-.25,0),(np.nan,0),(.25,np.inf)])
def test_depth_stage_does_not_fabricate_values_from_invalid_calibration(slope,intercept):
    layers=pd.DataFrame({'y_top':[0],'y_bottom':[10],'legend_id':[0],'lithology':['Shale']})
    with pytest.raises(PipelineError) as error:build_table(layers,Calibration(slope,intercept,0,3,3))
    assert error.value.code=='no_depth_calibration'
