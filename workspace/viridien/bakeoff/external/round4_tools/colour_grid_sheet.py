import sys,pathlib
from PIL import Image,ImageDraw
base=pathlib.Path(sys.argv[1]); pages=base/'nwt_orogo/_pages'
def sheet(name,files,columns=4):
 w,h=360,500; rows=(len(files)+columns-1)//columns; out=Image.new('RGB',(w*columns,h*rows),'white'); d=ImageDraw.Draw(out)
 for i,(label,p) in enumerate(files):
  im=Image.open(p).convert('RGB'); im.thumbnail((w-12,h-28)); x=(i%columns)*w; y=(i//columns)*h; d.text((x+6,y+4),label,fill='#777777');out.paste(im,(x+6,y+22))
 out.save(base/'_sheets'/name)
ex=base/'_exemplars/03_slice_mid.png'
files=[('MC765 exemplar lithology',ex)]
for stem in ('fall_stone_1','fall_stone_2','fall_stone_3'):
 for p in sorted(pages.glob(stem+'_*.png')):files.append((p.stem,p))
for n in [65,66,67,68,75,85,99,100,101,102,110,118]:
 p=pages/f'mirror_lake_report_{n:03}.png'; files.append((p.stem,p))
sheet('nwt_orogo.png',files)
sheet('occam.png',[('MC765 exemplar lithology',ex),('Bilyara 1 Display p1',base/'occam/page_001.png')],2)
