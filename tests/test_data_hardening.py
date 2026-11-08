import json

import pytest

from viridian402.data import _int, validate_intervals, load_manifest
from viridian402.synthetic import generate_demo


def row(a,b,top,base):
    return {'top_y':a,'bottom_y':b,'lithology':'Shale','cluster_id':0,'top_depth':top,'bottom_depth':base}


def test_integer_coordinates_are_not_rounded_through_float():
    value=2**54+1
    assert _int(value,'coordinate')==value
    assert _int(str(value),'coordinate')==value
    with pytest.raises(ValueError):_int('1.000000000000000000001','coordinate')
    with pytest.raises(ValueError):_int('1e10000000','coordinate')


@pytest.mark.parametrize('rows', [
    [row(0,50,0,9999),row(50,100,50,100)],
    [row(0,50,100,50)],
    [row(0,50,True,50)],
    [row(0,50,0,50),row(60,100,40,100)],
])
def test_depths_cannot_hide_inconsistency_at_shared_boundary(rows):
    with pytest.raises(ValueError):validate_intervals(rows)


def test_shared_depth_accepts_numerical_roundoff():
    assert len(validate_intervals([row(0,50,0,50),row(50,100,50+1e-9,100)]))==2
    assert len(validate_intervals([row(0,1,0,0)]))==1
    assert len(validate_intervals([row(0,50,0,50),row(50,100,50.0016,100)]))==2


@pytest.mark.parametrize('tags', [[{}],['scan','scan'],[''],['scan',2]])
def test_invalid_tags_rejected_before_running_model(tmp_path,tags):
    path=generate_demo(tmp_path,n=1)
    manifest=json.loads(path.read_text());manifest['records'][0]['tags']=tags
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='tags'):load_manifest(path)
