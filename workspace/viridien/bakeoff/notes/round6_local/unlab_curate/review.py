"""Make bounded, labelled sheets from explicit inventory indices."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS=None
p=argparse.ArgumentParser();p.add_argument('lane',type=Path);p.add_argument('indices',nargs='+',type=int)
p.add_argument('--rotation',type=int,default=0);p.add_argument('--top',type=float,default=0)
p.add_argument('--bottom',type=float,default=1);p.add_argument('--name',default='detail')
a=p.parse_args();inventory=a.lane/'inventory_all.json'
if not inventory.exists():inventory=a.lane/'inventory.json'
rows=json.loads(inventory.read_text(encoding='utf-8'))
for start in range(0,len(a.indices),4):
    sheet=Image.new('RGB',(1024,1024),'white');d=ImageDraw.Draw(sheet)
    for i,idx in enumerate(a.indices[start:start+4]):
        r=rows[idx]
        with Image.open(r['image_path']) as im:
            if a.rotation:im=im.rotate(a.rotation,expand=True)
            im=im.crop((0,round(im.height*a.top),im.width,round(im.height*a.bottom))).convert('RGB')
            im.thumbnail((504,475))
            x,y=i%2*512,i//2*512
            sheet.paste(im,(x+(512-im.width)//2,y+30))
            d.text((x+5,y+2),f"{idx}: {r['id'][-58:]}",fill='black')
            d.text((x+5,y+15),f"{r['width']} x {r['height']}  {r.get('well','')[:40]}",fill='black')
    dest=a.lane/'details';dest.mkdir(exist_ok=True)
    sheet.save(dest/f'{a.name}_{start//4:02d}.jpg',quality=93)
    print(dest/f'{a.name}_{start//4:02d}.jpg')
