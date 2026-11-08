"""Native depth-scale review crops, bounded to 1024 pixels."""
import json
from pathlib import Path
from PIL import Image,ImageDraw
Image.MAX_IMAGE_PIXELS=None
lane=Path(__file__).resolve().parent;dataset=lane.parents[3]/'external_unlabelled_test'
items=json.loads((dataset/'eval_set.json').read_text(encoding='utf-8'))
group=[(i,r) for i,r in enumerate(items) if r['family']=='D-582 1-bit grid']
for start in range(0,len(group),4):
    sheet=Image.new('RGB',(1024,1024),'white');draw=ImageDraw.Draw(sheet)
    for j,(i,r) in enumerate(group[start:start+4]):
        name=r['log'];ann=json.loads((dataset/'bbox_annotation'/f'{name}.json').read_text(encoding='utf-8'))
        b=ann['bounding_boxes']['depth_column']
        with Image.open(dataset/'images'/f'{name}.tif') as im:
            crop=im.crop((b[0],b[1]+300,b[2],min(b[3],b[1]+1300)))
            crop.thumbnail((480,470));sheet.paste(crop,(j%2*512+10,j//2*512+32))
            draw.text((j%2*512+4,j//2*512+4),f'{i} {name}',fill='black')
    sheet.save(lane/'details'/f'depth_zoom_{start//4:02d}.jpg',quality=95)
print('Depth zoom sheets',len(group)//4,flush=True)
