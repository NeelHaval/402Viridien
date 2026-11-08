import json,fitz,os
from PIL import Image,ImageDraw
root='external_untrusted/round3/oregon_dogami';d=json.load(open(root+'/triage.json'))
for batch in range(4):
 sheet=Image.new('RGB',(2000,5*460),'white');draw=ImageDraw.Draw(sheet)
 for j,doc in enumerate(d['documents'][batch*10:batch*10+10]):
  pdf=fitz.open(doc['path']);page=pdf[0];r=page.rect;clip=fitz.Rect(0,0,r.width,min(r.height,r.width*1.0)); pix=page.get_pixmap(matrix=fitz.Matrix(480/r.width,480/r.width),clip=clip,alpha=False); im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);im.thumbnail((480,420)); x=(j%4)*500;y=(j//4)*460;sheet.paste(im,(x,y+25));draw.text((x,y+4),os.path.basename(doc['path'])[:45],fill='grey')
 sheet.crop((0,0,2000,1380)).save(root+'/_thumbs/review%d.png'%batch)
