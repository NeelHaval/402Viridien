import json,pathlib,re,math
import pymupdf as fitz
from PIL import Image,ImageDraw
root=pathlib.Path('/home/user/lith/viridien_work/external_untrusted/round3/nwt_orogo')
data=json.loads((root/'downloads.json').read_text()); docs=[]; thumbs=[]
for item in data['documents']:
 try:
  d=fitz.open(root/item['path']); info=[]; legend=[]; logpages=[]; markers=[]
  for i,p in enumerate(d):
   t=p.get_text(); low=t.lower(); drawings=len(p.get_drawings()); images=len(p.get_images());
   hits=[x for x in ['rock types','strip.log','striplog','wellsight','lithology','lithologic','mud log'] if x in low]
   if 'rock types' in low: legend.append(i+1)
   if hits or (p.rect.height>1500 and (drawings>10 or images)) or (item['path'] in ('01.pdf','03.pdf','04.pdf')) or (item['path']=='02.pdf' and i>=64): logpages.append(i+1)
   if 'wellsight' in low or 'strip.log' in low or 'striplog' in low: markers.append(i+1)
   info.append({'page':i+1,'size_points':[round(p.rect.width,1),round(p.rect.height,1)],'drawings':drawings,'images':images,'text_chars':len(t),'hits':hits})
  print(item['path'], 'pages',len(d),'legend',legend,'markers',markers,'log candidates',logpages, 'info',str(info)[:3000],flush=True)
  for n in logpages:
   p=d[n-1]; scale=1200/p.rect.height
   pix=p.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False); name=f"{item['path'][:-4]}_p{n:03}.png"; pix.save(root/'_thumbs'/name); thumbs.append((name,item['path']+' p'+str(n)))
  docs.append({**item,'pages':len(d),'producer':d.metadata.get('producer'),'creator':d.metadata.get('creator'),'page_details':info,'legend_text_pages':legend,'log_page_candidates':logpages,'wellsight_text_pages':markers})
 except Exception as e: docs.append({**item,'error':str(e)}); print('error',item['path'],repr(e),flush=True)
(root/'inspection.json').write_text(json.dumps(docs,indent=2))
if thumbs:
 width=1000; cw=200; ch=290; rows=math.ceil(len(thumbs)/5); sheet=Image.new('RGB',(width,rows*ch),'white'); draw=ImageDraw.Draw(sheet)
 for j,(name,label) in enumerate(thumbs):
  im=Image.open(root/'_thumbs'/name).convert('RGB'); im.thumbnail((190,260)); x=(j%5)*cw+(cw-im.width)//2; y=(j//5)*ch+22; sheet.paste(im,(x,y)); draw.text(((j%5)*cw+5,(j//5)*ch+4),label,fill='#777777')
 sheet.save(root/'contact_sheet.png')
