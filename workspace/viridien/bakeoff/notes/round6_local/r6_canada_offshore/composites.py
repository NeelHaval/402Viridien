import fitz,sys
from pathlib import Path
from PIL import Image,ImageDraw
p=Path(sys.argv[1]);o=Path(sys.argv[2]);sheet=Image.new('RGB',(1000,700),'white');dr=ImageDraw.Draw(sheet)
for j,f in enumerate(p.glob('*composite.pdf')):
 d=fitz.open(f);print(f.name,len(d),[(round(pg.rect.width),round(pg.rect.height)) for pg in list(d)[:3]]);print(d[0].get_text()[:400])
 pg=d[min(3,len(d)-1)];sc=min(490/pg.rect.width,650/pg.rect.height);pix=pg.get_pixmap(matrix=fitz.Matrix(sc,sc));im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);sheet.paste(im,(j*500,35));dr.text((j*500,5),f.name,fill='black')
sheet.save(o/'composites.png')
