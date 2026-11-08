"""Read image data only. Run with python -I and explicit root paths."""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '3'
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(1)


def features(im):
    im = im.convert('RGB')
    im.thumbnail((400, 700))
    a = np.asarray(im)
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    ink = (g < 190).astype(np.uint8)
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(12, len(g)//8), 1), np.uint8))
    hor = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(12, g.shape[1]//8)), np.uint8))
    bins = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(bins)[:8, :8]
    bits = dct > np.median(dct.flatten()[1:])
    return dict(ink=float(ink.mean()), colour=float((hsv[:,:,1] > 50).mean()),
                saturation=float(hsv[:,:,1].mean()/255), binary=float(((g<20)|(g>235)).mean()),
                vertical=float(vert.mean()), horizontal=float(hor.mean()),
                hatch=float(np.mean(cv2.Canny(g,80,160)>0)),
                sharpness=float(cv2.Laplacian(g,cv2.CV_32F).var()),
                phash=f'{sum(int(v)<<i for i,v in enumerate(bits.flat)):016x}')


def sheets(rows, thumbs, out, prefix):
    out.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), 16):
        page = Image.new('RGB', (1024, 1024), 'white')
        d = ImageDraw.Draw(page)
        for k,r in enumerate(rows[start:start+16]):
            x,y=(k%4)*256,(k//4)*256
            im=Image.open(thumbs/r['thumbnail']).convert('RGB')
            im.thumbnail((246,216))
            page.paste(im,(x+(256-im.width)//2,y+30))
            d.text((x+4,y+2),f"{r['index']} {r.get('source','')[:23]}",fill='black')
            d.text((x+4,y+15),r['id'][-37:],fill='black')
        page.save(out/f'{prefix}_{start//16:03d}.jpg',quality=85)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('challenge',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    thumbs=a.out/'thumbs';thumbs.mkdir(exist_ok=True)
    roots=[a.root/'external_unlabelled',a.root/'external_unlabelled/round5']
    roots+=sorted((a.root/'external_unlabelled/round6').glob('*'))
    rows=[]
    for base in roots:
        manifest=base/'manifest.csv'
        if not manifest.exists():continue
        with manifest.open(encoding='utf-8-sig',newline='') as f:
            source_rows=list(csv.DictReader(f))
        images={p.stem:p for p in (base/'images').glob('*') if p.suffix.lower() in ('.tif','.tiff','.png','.jpg','.jpeg')}
        for r in source_rows:
            name=r.get('id',r.get('log',''))
            path=images.get(name)
            if path is None:continue
            record=dict(r,index=len(rows),image_path=str(path),dataset=str(base),thumbnail=f'{len(rows):04d}.jpg')
            try:
                with Image.open(path) as im:
                    record.update(width=im.width,height=im.height,mode=im.mode)
                    im.thumbnail((700,1000))
                    im=im.convert('RGB')
                    record.update(features(im))
                    im.save(thumbs/record['thumbnail'],quality=88)
            except Exception as exc:
                record['error']=str(exc)
            rows.append(record)
        print('source',str(base),len(source_rows),flush=True)
    (a.out/'inventory.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    groups=defaultdict(list)
    for r in rows:
        if 'error' not in r:groups[r.get('source','unknown')].append(r)
    for source,group in groups.items():sheets(group,thumbs,a.out/'sheets',source)
    excluded={v['log'] for v in json.loads((a.root/'bakeoff/eval_set.json').read_text(encoding='utf-8'))}
    with (a.root/'analysis/label_shares_per_log.csv').open(encoding='utf-8-sig',newline='') as f:
        families={r['log']:r['family'] for r in csv.DictReader(f)}
    exemplar=[]
    for name,family in families.items():
        if name in excluded:continue
        ann=json.loads((a.challenge/'bbox_annotation'/f'{name}.json').read_text(encoding='utf-8'))
        images=list((a.challenge/'images').glob(name+'.*'))
        if not images:continue
        boxes=ann['bounding_boxes']
        with Image.open(images[0]) as im:
            x0,y0,x1,y1=boxes['interpreted_lithology_column']
            crop=im.crop((x0,y0,x1,min(y1,y0+2000))).convert('RGB')
            feat=features(crop)
            tile=Image.new('RGB',(950,650),'white');d=ImageDraw.Draw(tile)
            d.text((8,4),f'{family} | {name[:100]}',fill='black')
            for k,key in enumerate(('interpreted_lithology_column','depth_column','lithology_legend')):
                b=list(boxes[key])
                if key!='lithology_legend':b[3]=min(b[3],b[1]+2000)
                c=im.crop(b).convert('RGB');c.thumbnail((300,600))
                tile.paste(c,(k*315,35));d.text((k*315,20),key,fill='black')
            target=a.out/'exemplars';target.mkdir(exist_ok=True)
            tile.save(target/f'{len(exemplar):02d}.jpg',quality=90)
            exemplar.append(dict(log=name,family=family,features=feat,boxes=boxes,size=im.size,thumbnail=f'exemplars/{len(exemplar):02d}.jpg'))
    (a.out/'exemplars.json').write_text(json.dumps(exemplar,indent=2),encoding='utf-8')
    print(json.dumps(dict(pages=len(rows),sources=dict(Counter(r.get('source') for r in rows)),non_eval_exemplars=len(exemplar)),indent=2),flush=True)


if __name__=='__main__':main()
