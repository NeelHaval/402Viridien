"""Depth calibration regressions; fixtures contain no evaluation truth."""
import json

import numpy as np
import pytest
from PIL import Image

from . import depth, io


@pytest.mark.parametrize('text,value', [
    ('5', 5), ('55', 55), ('0', 0), ('12.5', 12.5), ('12,5', 12.5),
    ('1,250', 1250), ('25 400', 25400), ('4400-', 4400),
    ('-50', None), ('50 55', None), ('ROP', None), ('1e6', None),
])
def test_numeric_labels(text, value):
    assert depth._number(text) == value


@pytest.mark.parametrize('texts,unit', [
    (['Depth (m)'], 'm'), (['650.0 meters'], 'm'), (['650.0meters'], 'm'),
    (['ROP (min/m)'], 'm'), (['ROP (m/min)'], 'm'),
    (['Depth (ft)', 'ROP (m/min)'], 'ft'),
    (['244 mm', 'Gas (units)'], None), (['MD', 'Gamma API'], None),
])
def test_printed_units(texts, unit):
    assert depth._units(texts) == unit


def test_consensus_rejects_ocr_outliers_and_duplicate_votes():
    y=np.arange(100,2500,200,dtype=float)
    correct=np.column_stack([y,200+.04*y])
    wrong=np.array([[100,9999],[500,88],[1100,1],[1700,700],[2200,20]])
    # Many duplicate readings of a bad number cannot outweigh independent labels.
    points=np.vstack([correct,wrong,np.repeat(wrong[:1],30,axis=0)])
    a,b,n,rmse=depth._fit(points)
    assert a == pytest.approx(.04)
    assert b == pytest.approx(200)
    assert n == len(y)
    assert rmse < 1e-8
    assert depth._fit(points[::-1]) == pytest.approx((a,b,n,rmse))


def test_sparse_two_anchor_fit():
    assert depth._fit([(100,0),(600,50)]) == pytest.approx((.1,-10,2,0))


@pytest.mark.parametrize('points', [
    [], [(100,50)], [(100,50),(100,100)], [(100,50),(110,100)],
    [(100,100),(300,50),(500,0)],
    [(100,100),(300,200),(500,900)],
    [(0,float('nan')),(500,20)],
])
def test_rejects_unsupported_or_nonmonotonic_fits(points):
    with pytest.raises(ValueError): depth._fit(points)


def test_rotation_coordinates_and_padding(monkeypatch):
    rgb=np.full((200,30,3),255,np.uint8)
    call=iter([(8,50),(50,8),(149,8)])
    def tokens(im,engine):
        x,y=next(call)
        # Clean OCR adds a 12px border in the rotated coordinate system.
        return [(np.array([[x+12,y+12]]*4),'55',.99)]
    monkeypatch.setattr(depth,'_tokens',tokens)
    points=depth._read_points(rgb,1000,'rapidocr',clean=True)
    assert points == [(1050.,55.)]*3


@pytest.fixture
def log(tmp_path,monkeypatch):
    monkeypatch.setattr(io,'ROOT',tmp_path)
    result=io.Log(Image.new('RGB',(60,1500),'white'),
                  {'depth_column':[10,100,40,600]},'synthetic_axis')
    yield result
    result.close()


def test_retry_reads_scale_beyond_short_annotation(log,monkeypatch):
    reads=[]
    def points(rgb,top,engine,clean=False):
        reads.append((top,len(rgb),clean))
        if clean:
            return [(300,100),(800,150),(1300,200)]
        return [(300,100)]
    monkeypatch.setattr(depth,'_read_points',points)
    monkeypatch.setattr(depth,'_unit_context',lambda log:('m',['Depth (m)']))
    calib=depth.calibrate(log)
    assert calib.a == pytest.approx(.1/.3048)
    assert calib.to_depth(300) == pytest.approx(100/.3048)
    assert any(clean and top+height>600 for top,height,clean in reads)
    assert 'clean-retry' in calib.method
    assert 'm-to-ft' in calib.method
    assert all(height<=1800 for _,height,_ in reads)


def test_two_anchor_fallback_is_marked(log,monkeypatch):
    monkeypatch.setattr(depth,'_read_points',lambda *a,**kw:[(100,50),(600,100)])
    monkeypatch.setattr(depth,'_unit_context',lambda log:(None,[]))
    calib=depth.calibrate(log)
    assert calib.to_depth(350) == pytest.approx(75)
    assert 'two-anchor' in calib.method
    assert 'ft-assumed' in calib.method


def test_cache_invalidates_old_version_bad_json_and_changed_geometry(log,monkeypatch):
    calls=[]
    def points(*args,**kwargs):
        calls.append(1)
        return [(100,50),(300,70),(600,100)]
    monkeypatch.setattr(depth,'_read_points',points)
    monkeypatch.setattr(depth,'_unit_context',lambda log:('ft',['Depth (ft)']))
    cache=io.ROOT/'out/_cache/depth/synthetic_axis.rapidocr.json'
    cache.parent.mkdir(parents=True)
    cache.write_text('{broken')
    expected=depth.calibrate(log)
    count=len(calls)
    assert depth.calibrate(log) == expected
    assert len(calls)==count
    record=json.loads(cache.read_text())
    record['version']=0
    cache.write_text(json.dumps(record))
    assert depth.calibrate(log) == expected
    assert len(calls)>count
    count=len(calls)
    log.boxes['depth_column'][3]=700
    depth.calibrate(log)
    assert len(calls)>count


def test_no_fabricated_depth_when_retry_finds_no_anchors(log,monkeypatch):
    monkeypatch.setattr(depth,'_read_points',lambda *a,**kw:[])
    with pytest.raises(ValueError,match='two numeric depth anchors'):
        depth.calibrate(log)


def test_rejects_implausible_extrapolation_above_zero(log,monkeypatch):
    monkeypatch.setattr(depth,'_read_points',lambda *a,**kw:[(500,10),(700,30),(900,50)])
    with pytest.raises(ValueError,match='above zero'):
        depth.calibrate(log)
