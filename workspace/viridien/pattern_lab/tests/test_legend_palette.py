import sys
from pathlib import Path
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from legend_palette import colours

def ids(image):return {c['id'] for c in colours(image)}

def test_colours_match_source_pixels_and_ignore_thin_black_frames():
    for colour,key in [('white','neutral'),('grey','neutral'),('yellow','yellow'),('orange','orange'),('red','red'),('lime','green'),('cyan','blue'),('blue','blue'),('purple','purple'),('black','black')]:
        image=Image.new('RGB',(80,40),colour);ImageDraw.Draw(image).rectangle((0,0,79,39),outline='black',width=1)
        assert key in ids(image)
        if key!='black':assert 'black' not in ids(image)

def test_multicolour_swatch_can_match_each_colour_without_changing_its_label():
    image=Image.new('RGB',(80,40),'white');draw=ImageDraw.Draw(image);draw.rectangle((0,0,79,7),fill='yellow');draw.rectangle((0,8,79,15),fill='cyan')
    assert ids(image)=={'neutral','yellow','blue'}
    assert all(len(c['rgb'])==3 for c in colours(image))


def test_media_cache_key_changes_for_box_or_legend_edits():
    import copy
    from review_media import ReviewMedia
    task=dict(context_box=[0,0,100,200],legend=[dict(id=1,label_printed='Shale',box=[1,1,40,20])])
    first=ReviewMedia.signature(task);other=copy.deepcopy(task)
    other['legend'][0]['image']='inline cached image'
    assert ReviewMedia.signature(other)==first
    other['legend'][0]['label_printed']='Sand'
    assert ReviewMedia.signature(other)!=first
    other=copy.deepcopy(task);other['context_box'][3]+=1
    assert ReviewMedia.signature(other)!=first

def test_review_delivery_contains_each_module_once():
    from review_media import ReviewMedia
    html=ReviewMedia.html(None,dict(portable=True))
    assert html.count('window.ReviewDetails=')==1
    assert html.count('window.ReviewConnect=')==1
    assert html.count('window.SharedReview=')==1
    assert 'Filter legend by colour' in html


def test_png_cache_retries_a_windows_reader_lock(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from review_media import ReviewMedia
    media=ReviewMedia(SimpleNamespace(store=SimpleNamespace(data=tmp_path)),SimpleNamespace(crop=lambda doc,box:Image.new('RGB',(20,30),'yellow')))
    original=Path.replace;attempts=[]
    def locked(path,target):
        attempts.append(path)
        if len(attempts)<3:raise PermissionError('Windows reader holds the PNG')
        return original(path,target)
    monkeypatch.setattr(Path,'replace',locked)
    assert media.image(dict(id='d'),[0,0,20,30]).startswith('data:image/png;base64,')
    assert len(attempts)==3
