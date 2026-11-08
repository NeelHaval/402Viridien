import csv, json, sys
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
root, data = map(Path, sys.argv[1:3])
out = root/'external_untrusted/round4/bw_hatch/_exemplars'
out.mkdir(parents=True, exist_ok=True)
names = sorted({r['log'] for r in csv.DictReader((root/'analysis/label_shares_per_log.csv').open()) if r['family'] in ('D-582 1-bit grid','MC761 mineral/lithology')})
for i, name in enumerate(names, 1):
    j=json.loads((data/'bbox_annotation'/f'{name}.json').read_text()); boxes=j['bounding_boxes']
    with Image.open(data/'images'/j['image_name']) as im:
        print(i,name,im.size,im.mode,im.info.get('dpi'),flush=True)
        for kind in ('header','legend','slice'):
            if kind=='header': box=(0,0,im.width,min(2500,im.height))
            elif kind=='legend': box=tuple(boxes['lithology_legend'])
            else:
                x1,y1,x2,y2=boxes['interpreted_lithology_column']; y=min(y1+1000,max(y1,y2-3000)); margin=max(600,(x2-x1)*3); box=(max(0,x1-margin),y,min(im.width,x2+margin),min(im.height,y+3000))
            if box[2]<=box[0] or box[3]<=box[1]: print('invalid',kind,box,flush=True); continue
            im.crop(box).save(out/f'{i:02d}_{kind}.png',optimize=True)
