import sys,json
from pathlib import Path
import fitz
from PIL import Image,ImageDraw
p=Path(sys.argv[1]);out=Path(sys.argv[2]);results=[];thumbs=[]
for f in p.glob('*.pdf'):
 try:
  d=fitz.open(f);pages=[]
  for i in range(len(d)):
   pg=d[i];t=pg.get_text();
   if any(k in t.lower() for k in ['master log','mud log','lithology','geoservices','sperry','weatherford','baker']):pages.append({'page':i+1,'text':t[:900],'size':list(pg.rect)})
  results.append({'file':f.name,'pages':len(d),'hits':pages[:12]})
  sel=[0]+[x['page']-1 for x in pages[:2]]
  for i in dict.fromkeys(sel):
   pg=d[i];pix=pg.get_pixmap(matrix=fitz.Matrix(min(300/pg.rect.width,600/pg.rect.height),min(300/pg.rect.width,600/pg.rect.height)),alpha=False);im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);thumbs.append((f.stem[:32]+' p'+str(i+1),im))
 except Exception as e:results.append({'file':f.name,'error':str(e)})
(out/'inspection.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
thumbs=thumbs[:16];sheet=Image.new('RGB',(1000,650*((len(thumbs)+3)//4)), 'white');dr=ImageDraw.Draw(sheet)
for j,(label,im) in enumerate(thumbs):
 im.thumbnail((245,610));x=(j%4)*250;y=(j//4)*650;dr.text((x+4,y+3),label,fill='black');sheet.paste(im,(x,y+30))
sheet.save(out/'candidate_sheet.png');print(json.dumps([{'file':r['file'],'pages':r.get('pages'),'hits':len(r.get('hits',[]))} for r in results]))
