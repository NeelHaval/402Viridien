from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys
root=Path(sys.argv[1]); base=root/'external_untrusted/round4/geoservices_grid'; im=Image.new('RGB',(2400,2300),'white'); d=ImageDraw.Draw(im)
try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23)
except OSError: font=ImageFont.load_default()
def place(path,rect,label):
 x,y,w,h=rect; d.text((x,y),label,fill='#777777',font=font); pic=Image.open(path).convert('RGB'); pic.thumbnail((w,h-35)); im.paste(pic,(x,y+35))
place(base/'_exemplars/04_60811407300100_header.png',(25,20,950,1150),'Exemplar: BP / Geoservices header + legend')
place(base/'_exemplars/04_60811407300100_slice.png',(25,1190,950,1050),'Exemplar: 3000 px lithology slice')
place(base/'openei_wister/page1.png',(1020,20,1320,1110),'Wister 12-27: page 1, header + legend')
place(base/'openei_wister/page4.png',(1020,1160,1320,1110),'Wister 12-27: page 4, patterned beds; score 2/3')
im.save(base/'_sheets/openei_wister.png',optimize=True)
