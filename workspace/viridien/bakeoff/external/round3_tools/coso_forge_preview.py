from pathlib import Path
from PIL import Image,ImageDraw
p=Path('/home/user/lith/viridien_work/external_untrusted/round3/coso_forge/_thumbs')
files=sorted(p.glob('*_p1.png'))
out=Image.new('RGB',(1500,((len(files)+3)//4)*520),'white');d=ImageDraw.Draw(out)
for i,f in enumerate(files):
 im=Image.open(f);im.thumbnail((370,480));x=i%4*375;y=i//4*520;d.text((x,y),f.name[:47],fill='black');out.paste(im,(x,y+22))
out.save('/home/user/lith/viridien_work/external_untrusted/round3/coso_forge/preview.png')
