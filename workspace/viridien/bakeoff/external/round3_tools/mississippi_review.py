from PIL import Image,ImageDraw
from pathlib import Path
import math
root=Path('external_untrusted/round3/mississippi')
files=sorted(root.glob('*-001.png')); w=380; h=375; cols=4
sheet=Image.new('RGB',(cols*w,math.ceil(len(files)/cols)*h),'white'); d=ImageDraw.Draw(sheet)
for i,p in enumerate(files):
 im=Image.open(p).convert('RGB'); im.thumbnail((w-6,h-25)); x=i%cols*w; y=i//cols*h
 sheet.paste(im,(x,y)); d.text((x+2,y+h-20),p.stem,fill='black')
sheet.save('/tmp/ms_review.png')
