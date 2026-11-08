import csv, json, sys
from pathlib import Path
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
root = Path(sys.argv[1]); data = Path(sys.argv[2]); out = root/'external_untrusted/round4/geoservices_grid/_exemplars'
names = sorted({r['log'] for r in csv.DictReader((root/'analysis/label_shares_per_log.csv').open()) if r['family']=='Geoservices grid'})
for i, name in enumerate(names, 1):
    ann = data/'bbox_annotation'/f'{name}.json'
    j = json.loads(ann.read_text()); boxes = j['bounding_boxes']; p = data/'images'/j['image_name']
    if not p.exists():
        print('missing',p,flush=True); continue
    with Image.open(p) as im:
        for kind in ('header','legend','slice'):
            if kind == 'header': box=(0,0,im.width,min(2500,im.height))
            elif kind == 'legend': box=tuple(boxes['lithology_legend'])
            else:
                x1,y1,x2,y2=boxes['interpreted_lithology_column']; y=min(y1+1000,max(y1,y2-3000)); margin=max(600, (x2-x1)*3); box=(max(0,x1-margin),y,min(im.width,x2+margin),min(im.height,y+3000))
            if not box or len(box)!=4 or box[2]<=box[0] or box[3]<=box[1]: print('invalid',name,kind,box,flush=True); continue
            im.crop(box).convert('RGB').save(out/f'{i:02d}_{name[:14]}_{kind}.png',optimize=True)
    print(i,name,flush=True)
