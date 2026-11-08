import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import detect_legend as D


def example(tmp_path):
    image=Image.new('RGB',(400,220),'white');draw=ImageDraw.Draw(image)
    for x,y,color in [(10,20,'brown'),(210,20,'green'),(10,120,'gray'),(210,120,'white')]:
        draw.rectangle([x,y,x+59,y+39],fill=color,outline='black',width=1)
        if color!='white':
            for yy in range(y+6,y+35,10):draw.line([x+5,yy,x+40,yy],fill='black',width=2)
    image.save(tmp_path/'legend.png')
    (tmp_path/'meta.json').write_text(json.dumps({'layers':{'legend':{
        'box':[100,200,500,420],'levels':[{'scale':1,'tiles':[{'file':'legend.png','x0':100,'y0':200}]}]}}}))
    return image


def test_swatches_and_label_crops(tmp_path):
    image=example(tmp_path)
    boxes=D.find_swatches(image)
    assert len(boxes)==4
    assert boxes[0]==[10,20,70,60]
    regions=D.text_regions(boxes,image.size)
    # Horizontal legends have the labels to the right, outside the symbol.
    assert all(psm==7 and rect[2]>rect[0] for rect,psm in regions)
    assert regions[0][0][0]>boxes[0][2]


def test_detect_preserves_variants_filters_accessories_and_skips_blank(tmp_path):
    example(tmp_path)
    names=iter(['Shale brown','Shale green','Mud gain'])
    result=D.detect(tmp_path,ocr=lambda image,psm:(next(names),None))
    suggestions=result['suggestions']
    assert [s['label_printed'] for s in suggestions]==['Shale brown','Shale green','Mud gain']
    assert [s['include'] for s in suggestions]==[True,True,False]
    assert suggestions[0]['box']==[110,220,170,260]
    assert not result['warnings']
    # Detector is read-only: no annotation file is created.
    assert sorted(p.name for p in tmp_path.iterdir())==['legend.png','meta.json']


def test_missing_ocr_keeps_box_candidates_for_manual_names(tmp_path,monkeypatch):
    example(tmp_path);monkeypatch.setattr(D.shutil,'which',lambda _:None)
    result=D.detect(tmp_path)
    assert len(result['suggestions'])==3
    assert all(not s['include'] and s['label_printed']=='' for s in result['suggestions'])
    assert len(result['warnings'])==1 and 'OCR unavailable' in result['warnings'][0]


def test_blank_page_no_false_swatches(tmp_path):
    assert D.find_swatches(Image.new('RGB',(500,300),'white'))==[]


def test_below_label_layout():
    boxes=[[10,20,100,80],[130,20,220,80],[10,150,100,210]]
    regions=D.text_regions(boxes,(300,300))
    assert all(psm==6 for _,psm in regions)
    assert regions[0][0][1]>80 and regions[0][0][3]<150


def test_legend_tile_stitching(tmp_path):
    im=example(tmp_path);stitched,box=D.legend_image(tmp_path)
    assert box==[100,200,500,420]
    assert np.array_equal(np.asarray(im),np.asarray(stitched))
