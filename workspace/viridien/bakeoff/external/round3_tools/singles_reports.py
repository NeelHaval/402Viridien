import pymupdf
from PIL import Image,ImageDraw
for filename,indices in [('NYSERDA_00449_Report.pdf',range(0,90)),('0885.pdf',range(59,71))]:
 d=pymupdf.open('external_untrusted/round3/singles/'+filename); cols=10 if len(indices)>20 else 6; w,h=150,205; sheet=Image.new('RGB',(cols*w,((len(indices)+cols-1)//cols)*h),'white'); draw=ImageDraw.Draw(sheet)
 for k,i in enumerate(indices):
  p=d[i]; pix=p.get_pixmap(matrix=pymupdf.Matrix(175/p.rect.height,175/p.rect.height),alpha=False); im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples); im.thumbnail((145,180)); x=k%cols*w;y=k//cols*h;sheet.paste(im,(x,y+20));draw.text((x,y),str(i+1),fill='grey')
 sheet.save('external_untrusted/round3/singles/'+filename+'.overview.png')
