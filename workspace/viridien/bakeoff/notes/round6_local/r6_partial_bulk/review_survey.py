from pathlib import Path
import json,fitz,math
from PIL import Image,ImageDraw
N=Path('viridien_work/bakeoff/notes/round6_local/r6_partial_bulk');R=Path('viridien_work/external_untrusted/round6/r6_partial_bulk')
docs=[]
extra=json.loads((N/'survey_candidates.json').read_text(encoding='utf-8')) if (N/'survey_candidates.json').exists() else []
if (N/'gdr_candidates.json').exists():extra+=json.loads((N/'gdr_candidates.json').read_text(encoding='utf-8'))
docs += [x for x in extra if x.get('path')]
items=[]
for i,x in enumerate(docs):
 try:
  d=fitz.open(x['path'])
  pages=[0] if x.get('family_match')=='bw_hatch' else list(range(min(len(d),10)))
  for pn in pages:
   p=d[pn];z=min(190/p.rect.width,700/p.rect.height);pix=p.get_pixmap(matrix=fitz.Matrix(z,z),alpha=False);im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);items.append((i,pn,im))
 except Exception:pass
for batch in range(math.ceil(len(items)/10)):
 sheet=Image.new('RGB',(1000,1024),'white');dr=ImageDraw.Draw(sheet)
 for k,(i,pn,im) in enumerate(items[batch*10:(batch+1)*10]):
  im.thumbnail((190,465));x=k%5*200;y=k//5*512;sheet.paste(im,(x,y+35));dr.text((x+2,y+3),str(i)+' p'+str(pn+1)+' '+Path(docs[i]['path']).stem[:12],fill='black')
 sheet.save(N/f'survey_review_{batch:02}.jpg')
(N/'survey_review_index.json').write_text(json.dumps(docs,indent=2),encoding='utf-8');print('REVIEW',len(docs),len(items),math.ceil(len(items)/10))
