import argparse
import json
from pathlib import Path
import sys
from PIL import Image
lane=Path(__file__).resolve().parent;sys.path.insert(0,str(lane))
from curate import source_profile,make_page,card
p=argparse.ArgumentParser();p.add_argument('indices',nargs='+',type=int);a=p.parse_args()
rows=json.loads((lane/'inventory_all.json').read_text(encoding='utf-8'))
for start in range(0,len(a.indices),4):
    sheet=Image.new('RGB',(1024,1024),'white')
    for i,index in enumerate(a.indices[start:start+4]):
        r=dict(rows[index]);r['profile']=source_profile(r,rows);r['family']=r['profile']['family']
        im,boxes,mapping=make_page(r,rows)
        c=card(im,boxes,r,index);sheet.paste(c,(i%2*512,i//2*512))
        print(index,boxes,flush=True)
    sheet.save(lane/'details'/f'probe_{start//4:02d}.jpg',quality=95)
