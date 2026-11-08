from pathlib import Path
import fitz
from PIL import Image,ImageDraw
r=Path('C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work');s=r/'external_untrusted/round6/r6_europe_africa/sodir';o=r/'bakeoff/notes/round6_local/r6_europe_africa';tiles=[]
for fn,inds in [('5093_7131_4_1_COMPLETION_GEOLOGICAL_REPORT.PDF',range(106,116)),('4423_34_7_32_COMPLETION_REPORT.PDF',[83]),('4939_6608_11_4_COMPLETION_REPORT.PDF',range(86,90))]:
 d=fitz.open(s/fn)
 for i in inds:
  p=d[i];z=min(280/p.rect.width,370/p.rect.height);px=p.get_pixmap(matrix=fitz.Matrix(z,z));im=Image.frombytes('RGB',[px.width,px.height],px.samples);tiles.append((fn[:8]+f' p{i+1}',im));print(fn[:8],i+1,p.rect,len(p.get_text()),flush=True)
can=Image.new('RGB',(1120,410*((len(tiles)+3)//4)),'white');dr=ImageDraw.Draw(can)
for j,(lab,im) in enumerate(tiles):x=j%4*280;y=j//4*410;can.paste(im,(x,y+25));dr.text((x+3,y+3),lab,fill='black')
can.save(o/'review_sheet.png');can.thumbnail((1024,1024));can.save(o/'review_sheet_preview.png')
