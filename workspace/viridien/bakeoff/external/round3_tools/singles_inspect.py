import sys,os,glob,json,re,pymupdf
for p in sum([glob.glob('external_untrusted/round3/'+s+'/*.pdf') for s in ('singles','stratagraph','precision_wl')],[]):
 try:
  d=pymupdf.open(p); print('\nFILE',p,'PAGES',len(d),'META',d.metadata,flush=True)
  for i,pg in enumerate(d):
   t=pg.get_text(); hits=[x for x in ['strip.log','striplog','rock types','lithology legend','lithologic log','lithology log','geostrip','lithology','legend','wellsight'] if x in t.lower()]
   if hits or len(d)<=12: print(i+1,round(pg.rect.width),round(pg.rect.height),'text',len(t),'images',len(pg.get_images()),'hits',','.join(hits),'sample',re.sub(r'\s+',' ',t[:220]),flush=True)
 except Exception as e: print('ERROR',str(e),flush=True)
