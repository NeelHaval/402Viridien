"""Extract one explicit data page. No downloaded code is executed."""
import argparse
import json
from pathlib import Path
import fitz
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('pdf',type=Path);p.add_argument('page',type=int);p.add_argument('out',type=Path)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
with fitz.open(a.pdf) as doc:
    page=doc[a.page-1]
    print('pages',len(doc),'page',a.page,'size',list(page.rect))
    print(page.get_text()[:2000])
    pix=page.get_pixmap(matrix=fitz.Matrix(200/72,200/72),alpha=False)
    pix.save(str(a.out/f'page_{a.page:03d}.png'))
    im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);im.thumbnail((1024,1024))
    im.save(a.out/f'page_{a.page:03d}_small.jpg',quality=94)
    (a.out/f'page_{a.page:03d}.json').write_text(json.dumps(dict(source_pdf=str(a.pdf),source_page=a.page,
        width=pix.width,height=pix.height,dpi=200)),encoding='utf-8')
