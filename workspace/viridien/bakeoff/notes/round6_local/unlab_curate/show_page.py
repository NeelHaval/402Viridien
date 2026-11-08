import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw
Image.MAX_IMAGE_PIXELS=None
p=argparse.ArgumentParser();p.add_argument('lane',type=Path);p.add_argument('indices',nargs='+',type=int)
p.add_argument('--rotation',type=int,default=0);p.add_argument('--crop-height',type=int);a=p.parse_args()
rows=json.loads((a.lane/'inventory_all.json').read_text(encoding='utf-8'))
for index in a.indices:
    r=rows[index]
    with Image.open(r['image_path']) as im:
        im=im.rotate(a.rotation,expand=True).convert('RGB')
        if a.crop_height:im=im.crop((0,0,im.width,min(im.height,a.crop_height)))
        im.thumbnail((1024,1024))
        d=ImageDraw.Draw(im)
        for frac in (.1,.2,.3,.4,.5,.6,.7,.8,.9):
            x=round(im.width*frac);d.line((x,0,x,20),fill='red');d.text((x+2,2),str(frac),fill='red')
        im.save(a.lane/'details'/f'page_{index:04d}_{a.rotation}.jpg',quality=96)
    print('made',index)
