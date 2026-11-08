import fitz,sys
from pathlib import Path
from PIL import Image,ImageDraw
p=Path(sys.argv[1]);out=Path(sys.argv[2]);cells=[('Non-eval Gulf exemplar',Image.open(out/'exemplar.png'))]
for f in p.glob('*.pdf'):
 d=fitz.open(f)
 ids= [0,1,5] if 'ee44' in f.name else [len(d)-2]
 for i in ids:
  pg=d[i];scale=min(400/pg.rect.width,570/pg.rect.height);pix=pg.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False);cells.append((f.name[:23]+' p'+str(i+1),Image.frombytes('RGB',[pix.width,pix.height],pix.samples)))
sheet=Image.new('RGB',(1600,620*((len(cells)+3)//4)),'white');draw=ImageDraw.Draw(sheet)
for j,(t,im) in enumerate(cells):
 im.thumbnail((390,580));x=j%4*400;y=j//4*620;draw.text((x+4,y+4),t,fill='black');sheet.paste(im,(x,y+30))
sheet.save(out/'contact_sheet.png');sheet.thumbnail((1024,1024));sheet.save(out/'contact_sheet_small.png')
