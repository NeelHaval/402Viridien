import fitz,sys
from pathlib import Path
p=Path(sys.argv[1]);o=Path(sys.argv[2]);f=next(p.glob('*ee44*.pdf'));d=fitz.open(f)
for i in [1,4]:
 pg=d[i];scale=1024/max(pg.rect.width,pg.rect.height);pg.get_pixmap(matrix=fitz.Matrix(scale,scale)).save(o/('mizzen_p'+str(i+1)+'.png'))
print('page5',d[4].get_text()[:1200]);print('page55',d[-1].get_text()[:150])
