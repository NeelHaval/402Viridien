from pathlib import Path
import fitz
from PIL import Image,ImageDraw
r=Path('C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work');o=r/'bakeoff/notes/round6_local/r6_europe_africa';s=r/'external_untrusted/round6/r6_europe_africa/sodir'
tiles=[]
for fn,inds in [('4408_16_2_2_COMPLETION_REPORT.pdf',[85,86,87,88,89,90,91,92,93,94]),('5093_7131_4_1_COMPLETION_GEOLOGICAL_REPORT.PDF',[104,105]),('4910_24_9_7_COMPLETION_LOG.pdf',[0,1])]:
 d=fitz.open(s/fn)
 for i in inds:
  p=d[i];rect=p.rect;clip=fitz.Rect(0,0,rect.width,min(rect.height,rect.width*1.5));z=300/clip.width;pix=p.get_pixmap(matrix=fitz.Matrix(z,z),clip=clip);im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);tiles.append((fn[:8]+f' p{i+1}',im))
can=Image.new('RGB',(1200,520*((len(tiles)+3)//4)),'white');dr=ImageDraw.Draw(can)
for j,(lab,im) in enumerate(tiles):x=j%4*300;y=j//4*520;can.paste(im,(x,y+25));dr.text((x+4,y+4),lab,fill='black')
can.save(o/'detail_sheet.png');can.thumbnail((1024,1024));can.save(o/'detail_sheet_preview.png')
