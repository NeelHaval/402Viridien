import zipfile,pathlib,fitz,json
from PIL import Image,ImageDraw
root=pathlib.Path('/home/user/lith/viridien_work/external_untrusted/round3')
z=root/'coso_forge/WestFlankMudLogs.zip'
with zipfile.ZipFile(z) as f:
 for i in f.infolist():
  p=pathlib.PurePosixPath(i.filename)
  if len(p.parts)!=1 or p.suffix.lower()!='.pdf' or i.file_size>250*1024**2: continue
  dest=root/'coso_forge'/p.name
  with f.open(i) as a,dest.open('wb') as b:
   while chunk:=a.read(1048576): b.write(chunk)
for slug in ('utah_forge','dhi_wister'):
 d=root/slug; thumbs=d/'_thumbs';thumbs.mkdir(exist_ok=True); rows=[]; metadata=[]
 for pdf in sorted(d.glob('*.pdf')):
  doc=fitz.open(pdf); pages=[]
  for n,page in enumerate(doc):
   rect=page.rect; scale=1200/rect.height
   pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
   fname=f'{pdf.stem}_p{n+1}.png'; pix.save(str(thumbs/fname))
   im=Image.open(thumbs/fname).convert('RGB'); im.thumbnail((250,600));rows.append((pdf.name+' p'+str(n+1),im.copy()))
   pages.append({'page':n+1,'size_points':[round(rect.width,1),round(rect.height,1)],'drawings':len(page.get_drawings()),'text_chars':len(page.get_text()),'images':len(page.get_images())})
  metadata.append({'file':pdf.name,'bytes':pdf.stat().st_size,'pages':pages,'metadata':doc.metadata})
 if rows:
  w=1200;cellw=300;cellh=640; sheet=Image.new('RGB',(w,((len(rows)+3)//4)*cellh),'white');draw=ImageDraw.Draw(sheet)
  for k,(label,im) in enumerate(rows):
   x=(k%4)*cellw;y=(k//4)*cellh;draw.text((x+4,y+4),label[:44],fill='#777777');sheet.paste(im,(x+4,y+25))
  sheet.save(d/'contact_sheet.png')
 (d/'technical_metadata.json').write_text(json.dumps(metadata,indent=2))
 print(slug,len(metadata),sum(len(x['pages']) for x in metadata),flush=True)
