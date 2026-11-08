"""Input-level checks for naming, cell isolation, and the S20 contract."""
import hashlib
import json
import numpy as np
from PIL import Image
from harness import io,legend_v4 as v4


def candidate(text,engine='rapid',confidence=.98,direction='right'):
    return dict(text=text,engine=engine,confidence=confidence,direction=direction)


def test_preserve_geological_words_and_plurals():
    for text in ('Silty Sandstone','Dolomites-CD','Volcanics','Calcareous Shale'):
        assert v4.clean_name(text).lower()==text.lower()
    assert v4.clean_name('Anhv')=='Anhy'
    assert v4.clean_name('Blank')=='Blank'


def test_readable_page_text_blocks_previous_cell():
    cs=[candidate('Sandstone','page',.8),candidate('S',confidence=.3),candidate('Sand',direction='left')]
    assert v4.choose(cs,v4.DEFAULT)=='Sandstone'


def test_separators_come_from_agreeing_page_text():
    cs=[candidate('Sh-grey'),candidate('Sh--grey','page',.8)]
    assert v4.choose(cs,v4.DEFAULT)=='Sh--grey'
    cs[-1]['text']='Sh--brown'
    assert v4.choose(cs,v4.DEFAULT)=='Sh-grey'


def test_technical_words_survive_missing_spaces():
    for word in ('MDT','MudGain','No Sample','Casing shoe','Core'):
        assert v4.excluded(word)
    assert not v4.excluded('Dolomites-D')


def test_below_cell_keeps_descenders():
    boxes=[(0,10,90,24),(0,37,90,51)]
    right=dict(v4._regions(boxes[0],boxes,[],(80,300,3)))['right']
    assert 32<=right[3]<37


def test_cached_pixels_translate_and_filter_mineral_section(tmp_path,monkeypatch):
    monkeypatch.setattr(v4,'HOME',tmp_path)
    rgb=np.full((100,200,3),255,np.uint8);rgb[35:60,5:55]=80
    digest=hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
    (tmp_path/'cache').mkdir()
    data=dict(layout=None,items=[dict(box=[5,35,55,60],candidates=[candidate('Anhydrite')])],
              ocr=[dict(box=[[0,0],[60,0],[60,20],[0,20]],text='Mineral',confidence=1)])
    (tmp_path/'cache'/f'wide3_{digest}.json').write_text(json.dumps(data))
    page=Image.new('RGB',(240,180),'white');page.paste(Image.fromarray(rgb),(20,50))
    log=io.Log(page,{'lithology_legend':[20,50,220,150]},'arbitrary')
    entries=v4.parse(log,dict(v4.DEFAULT,easy=False))
    assert entries[0].swatch_box==[25,85,75,110]
    assert entries[0].label=='Anhydrite'
    assert not entries[0].is_lithology


def test_s20_passes_fresh_legend_without_changing_context(monkeypatch):
    from strategies import s20_legend_v4 as s20
    old=object();new=object();ctx=dict(legend=old,device='cpu')
    monkeypatch.setattr(v4,'parse',lambda log:new)
    def delegate(log,win,context):
        assert context['legend'] is new
        assert context['device']=='cpu'
        return [dict(top_y=win[0],bottom_y=win[1])]
    monkeypatch.setattr(s20,'s18',delegate)
    assert s20.run(object(),(10,20),ctx)==[dict(top_y=10,bottom_y=20)]
    assert ctx['legend'] is old


def test_uniform_wide_grid_keeps_original_geometry():
    # Text baselines can sit below a sample. An intact grid must retain boxes.
    rgb=np.full((120,260,3),255,np.uint8)
    items=[dict(box=[x,y,x+60,y+10],candidates=[]) for x in (5,135) for y in (30,50,70)]
    data=dict(items=items,layout=None,ocr=[dict(box=[[0,0],[70,0],[70,12],[0,12]],text='Rock Types',confidence=1)])
    assert v4._wide_cells(rgb,data) is data


def test_fused_wide_rows_get_separate_label_crops(monkeypatch):
    rgb=np.full((110,260,3),255,np.uint8)
    words=[dict(box=[[0,0],[70,0],[70,12],[0,12]],text='Rock Types',confidence=1)]
    items=[]
    for x in (5,135):
        for y in (30,48,66,84):
            rgb[y:y+10,x:x+60]=100
            rgb[y+2:y+8,x+65:x+95]=80
            words.append(dict(box=[[x+65,y],[x+95,y],[x+95,y+10],[x+65,y+10]],text='Shale',confidence=.99))
        for y,h in ((30,10),(48,10),(66,28)):
            items.append(dict(box=[x,y,x+60,y+h],candidates=[]))
    class Recognition:
        def text_rec(self,crops):return [('Shale',.99) for c in crops],0
    class Easy:
        def recognize(self,*a,**kw):return []
    monkeypatch.setattr(v4.v1,'_ocr',lambda:Recognition())
    monkeypatch.setattr(v4,'_easy',lambda:Easy())
    out=v4._wide_cells(rgb,dict(items=items,layout=None,ocr=words))
    assert len(out['items'])==8
    assert all(q['box'][3]-q['box'][1]==10 for q in out['items'])
    assert all(v4.choose(q['candidates'],v4.DEFAULT)=='Shale' for q in out['items'])
