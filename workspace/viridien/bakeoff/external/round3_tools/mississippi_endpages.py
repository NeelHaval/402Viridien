from PIL import Image,ImageDraw
from pathlib import Path
import math,json
root=Path('external_untrusted/round3/mississippi'); dat=json.loads((root/'progress.json').read_text()); paths=[]
for log in dat['logs']:
 p=root/(log['id']+'-'+str(log['page_count']).zfill(3)+'.png')
 if p.exists(): paths.append(p)
w=350; h=340; cols=4
sheet=Image.new('RGB',(cols*w,math.ceil(len(paths)/cols)*h),'white'); d=ImageDraw.Draw(sheet)
for i,p in enumerate(paths):
 im=Image.open(p).convert('RGB'); im.thumbnail((w-6,h-25)); x=i%cols*w; y=i//cols*h
 sheet.paste(im,(x,y)); d.text((x+2,y+h-20),p.stem,fill='black')
sheet.save('/tmp/ms_endpages.png')
