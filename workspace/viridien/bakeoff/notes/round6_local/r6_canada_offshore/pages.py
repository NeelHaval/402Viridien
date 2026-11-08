import fitz,sys
from pathlib import Path
p=Path(sys.argv[1])
for f in p.glob('*.pdf'):
 d=fitz.open(f);large=[(i+1,round(pg.rect.width),round(pg.rect.height),pg.get_text()[:90].replace('\n',' ')) for i,pg in enumerate(d) if pg.rect.height>1200 or pg.rect.width>1200]
 print(f.name[:55],len(d),'large',large[:8])
 for i in [1,len(d)-1]:
  t=d[i].get_text()
  if i==1:print('TOC',t[:1800].replace('\n',' '))
