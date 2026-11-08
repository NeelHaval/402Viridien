import pathlib,sys,pymupdf
from PIL import Image,ImageDraw
root=pathlib.Path(sys.argv[1]); base=root/'external_untrusted/round4/colour_grid'; nwt=base/'nwt_orogo'; thumbs=nwt/'_pages';thumbs.mkdir(exist_ok=True)
for file in sorted(nwt.glob('*.pdf')):
 d=pymupdf.open(file); print(file.name,len(d),'metadata',d.metadata.get('producer'),flush=True)
 for i,p in enumerate(d):
  pix=p.get_pixmap(matrix=pymupdf.Matrix(min(.6,900/max(p.rect.width,p.rect.height)),min(.6,900/max(p.rect.width,p.rect.height))),alpha=False)
  pix.save(thumbs/f'{file.stem}_{i+1:03}.png')
for file in (base/'occam').glob('*.pdf'):
 d=pymupdf.open(file)
 for i,p in enumerate(d):
  sc=min(.6,1200/max(p.rect.width,p.rect.height)); p.get_pixmap(matrix=pymupdf.Matrix(sc,sc),alpha=False).save(file.parent/f'page_{i+1:03}.png')
