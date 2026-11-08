"""Regressions for cell clipping, comparison semantics, and pixel cache use."""
import json
import numpy as np
from PIL import Image

from harness import io, legend_v3 as v3


def test_narrow_swatch_cell_keeps_descenders():
    boxes=[(0,10,90,24),(0,37,90,51)]
    regions=v3._regions(boxes[0],boxes,[],(80,300,3))
    right=dict(regions)['right']
    assert right[1] <= 10
    assert right[3] >= 32  # Text descenders below a 14 px swatch must survive.
    assert right[3] < boxes[1][1]


def test_symmetric_abbreviations_keep_colour_distinctions():
    assert v3.canonical_name('Ls dkgy') == v3.canonical_name('Limestone dark grey')
    assert v3.canonical_name('Ss') == v3.canonical_name('Sandstone')
    assert v3.canonical_name('Clyst ltgy') != v3.canonical_name('Clyst dkgy')
    assert v3.clean_name('Anhv') == 'Anhy'
    assert v3.clean_name('I mst') == 'Lmst'


def test_selection_uses_engine_toggle_and_cleans_punctuation():
    candidates=[dict(engine='rapid',confidence=.95,text=' Ss - ',direction='right'),
                dict(engine='easy',confidence=1.,text='Coal',direction='right')]
    assert v3.choose(candidates,dict(v3.DEFAULT,easy=False)) == 'Ss'
    assert v3.choose(candidates,dict(v3.DEFAULT,easy_weight=1.1)) == 'Coal'


def test_column_grid_does_not_steal_previous_columns_label():
    candidates=[dict(engine='rapid',confidence=.82,text='Ss',direction='right'),
                dict(engine='easy',confidence=.99,text='Salt',direction='left')]
    assert v3.choose(candidates,v3.DEFAULT) == 'Ss'


def test_punctuation_restored_only_when_letters_agree():
    candidates=[dict(engine='rapid',confidence=.98,text='Sh-green',direction='right'),
                dict(engine='page',confidence=.8,text='Sh--green',direction='right')]
    assert v3.choose(candidates,v3.DEFAULT) == 'Sh--green'
    candidates[1]['text']='Sh--brown'
    assert v3.choose(candidates,v3.DEFAULT) == 'Sh-green'


def test_pixel_cache_translates_coordinates_without_truth_or_ocr(tmp_path,monkeypatch):
    monkeypatch.setattr(v3,'HOME',tmp_path)
    rgb=np.full((40,130,3),255,np.uint8);rgb[4:20,5:55]=100
    log=io.Log(Image.fromarray(rgb),{'lithology_legend':[0,0,130,40]},'first')
    # Supply cached evidence for this exact image, then ensure neither OCR
    # backend can be called. Cache stores crop-relative coordinates.
    import hashlib
    digest=hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
    (tmp_path/'cache').mkdir()
    data=dict(layout=None,items=[dict(box=[5,4,55,20],candidates=[dict(text='Coal',confidence=.99,engine='rapid',direction='right')])])
    (tmp_path/'cache'/f'v{v3.VERSION}_{digest}.json').write_text(json.dumps(data))
    def forbidden(*args,**kwargs):raise AssertionError('OCR should not run on a cache hit')
    monkeypatch.setattr(v3.v1,'ocr',forbidden);monkeypatch.setattr(v3,'_easy',forbidden)
    entries=v3.parse(log,dict(v3.DEFAULT,easy=False))
    assert entries[0].swatch_box == [5,4,55,20]
    page=Image.new('RGB',(150,100),'white');page.paste(Image.fromarray(rgb),(10,50))
    moved=io.Log(page,{'lithology_legend':[10,50,140,90]},'another-name')
    out=v3.parse(moved,dict(v3.DEFAULT,easy=False))
    assert out[0].swatch_box == [15,54,65,70]
    assert out[0].label == 'Coal'
    assert np.array_equal(entries[0].swatch,out[0].swatch)
