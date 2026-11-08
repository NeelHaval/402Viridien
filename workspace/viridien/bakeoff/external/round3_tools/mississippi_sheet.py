from PIL import Image,ImageDraw,ImageOps
from pathlib import Path
import math
root=Path('external_untrusted/round3/mississippi')
files=sorted(p for p in root.glob('*.png') if p.name != 'contact_sheet.png')
w=200; h=220; cols=8
sheet=Image.new('RGB',(cols*w,math.ceil(len(files)/cols)*h),'white'); draw=ImageDraw.Draw(sheet)
for i,p in enumerate(files):
 im=Image.open(p).convert('RGB'); im.thumbnail((w-8,h-28)); x=(i%cols)*w+(w-im.width)//2; y=(i//cols)*h
 sheet.paste(im,(x,y)); draw.text(((i%cols)*w+3,y+h-25),p.stem,fill='#777777')
sheet.save(root/'contact_sheet.png',optimize=True)
