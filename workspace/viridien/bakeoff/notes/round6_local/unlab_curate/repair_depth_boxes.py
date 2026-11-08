"""Retain clipped depth digits; image pixels and selected pages stay fixed."""
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
lane=Path(__file__).resolve().parent;sys.path.insert(0,str(lane))
from curate import card,new_source_geometry
dataset=lane.parents[3]/'external_unlabelled_test'
selected=json.loads((lane/'proposed.json').read_text(encoding='utf-8'))
changed=[];sheets=set()
for i,r in enumerate(selected):
    if r['source']!='r6_partial_oregon':continue
    name=r['id'];path=dataset/'bbox_annotation'/f'{name}.json'
    ann=json.loads(path.read_text(encoding='utf-8'));b=ann['bounding_boxes']['depth_column']
    width=.022 if new_source_geometry()[name]['style']=='horizon' else .05
    left=max(0,min(b[0],b[2]-round(ann['width']*width)))
    if left==b[0]:continue
    b[0]=left;ann['depth_box_revision']='Minimum source channel width retains full printed digits'
    ann['visual_review']='pending'
    path.write_text(json.dumps(ann,indent=2),encoding='utf-8')
    changed.append(name);sheets.add(i//4)
for n in sorted(sheets):
    sheet=Image.new('RGB',(1024,1024),'white')
    for j,r in enumerate(selected[n*4:n*4+4]):
        name=r['id'];ann=json.loads((dataset/'bbox_annotation'/f'{name}.json').read_text(encoding='utf-8'))
        with Image.open(dataset/'images'/f'{name}.tif') as im:
            c=card(im,ann['bounding_boxes'],r,n*4+j);sheet.paste(c,(j%2*512,j//2*512))
    sheet.save(dataset/'review_sheets'/f'{n:03d}.jpg',quality=94)
print('Repaired depth boxes',len(changed),'sheets',sorted(sheets),flush=True)
