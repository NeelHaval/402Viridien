import csv,json,os,sys
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
root=sys.argv[1]; data=sys.argv[2]
out=os.path.join(root,'external_untrusted/round4/colour_grid/_exemplars')
with open(os.path.join(root,'analysis/label_shares_per_log.csv')) as f:
 names=sorted({r['log'] for r in csv.DictReader(f) if r['family'] in ('MC765 colour grid','One-off layout')})
for i,n in enumerate(names):
 with open(os.path.join(data,'bbox_annotation',n+'.json')) as f: a=json.load(f)
 im=Image.open(os.path.join(data,'images',a['image_name'])); b=a['bounding_boxes']; w,h=im.size
 def save(box,kind):
  if box and box[2]>box[0] and box[3]>box[1]:
   p=im.crop(tuple(map(int,box))); p.save(os.path.join(out,f'{i+1:02d}_{kind}.png'))
 save((0,0,w,min(2500,h)),'header')
 save(b.get('lithology_legend'),'legend')
 c=b.get('interpreted_lithology_column') or b.get('depth_column'); y=max(0,min(h-3000, int(c[1])+3000)) if c else min(2500,h-3000)
 save((0,y,w,min(h,y+3000)),'slice')
 print(i+1,n,im.size,'legend',b.get('lithology_legend'),'slice_y',y,flush=True)
