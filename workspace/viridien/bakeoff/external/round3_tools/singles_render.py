import os,glob,pymupdf
from PIL import Image,ImageDraw
for slug in ['singles','stratagraph','precision_wl']:
 root='external_untrusted/round3/'+slug; os.makedirs(root+'/_thumbs',exist_ok=True); tiles=[]
 for path in glob.glob(root+'/*.pdf'):
  doc=pymupdf.open(path); pages=range(len(doc)) if len(doc)<30 else range(55,min(65,len(doc)))
  for n in pages:
   page=doc[n]; pix=page.get_pixmap(matrix=pymupdf.Matrix(1200/page.rect.height,1200/page.rect.height),alpha=False); name=os.path.splitext(os.path.basename(path))[0]+'_p%03d.png'%(n+1); pix.save(root+'/_thumbs/'+name)
   im=Image.open(root+'/_thumbs/'+name).convert('RGB'); im.thumbnail((220,330)); tiles.append((name,im.copy()))
 if tiles:
  cols=5; rows=(len(tiles)+cols-1)//cols; sheet=Image.new('RGB',(cols*240,rows*365),'white'); draw=ImageDraw.Draw(sheet)
  for i,(name,im) in enumerate(tiles):
   x=(i%cols)*240; y=(i//cols)*365; sheet.paste(im,(x,y+23)); draw.text((x+2,y+3),name,fill='#777777')
  sheet.save(root+'/contact_sheet.png')
