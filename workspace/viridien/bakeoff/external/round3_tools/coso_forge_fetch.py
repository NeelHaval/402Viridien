import requests,time,pathlib,sys
root=pathlib.Path('/home/user/lith/viridien_work/external_untrusted/round3')
items=[('coso_forge','https://gdr.openei.org/files/807/WestFlankMudLogs.zip','WestFlankMudLogs.zip'),('coso_forge','https://gdr.openei.org/files/758/52B-7%20Lithology%20Log.pdf','52B-7.pdf'),('utah_forge','https://gdr.openei.org/submissions/1292','submission.html'),('dhi_wister','https://gdr.openei.org/files/904/wister%2012-27%20composite%20log.pdf','wister.pdf')]
s=requests.Session();s.headers['User-Agent']='Mozilla/5.0 (research crawler)'
for slug,url,name in items:
 for i in range(4):
  try:
   time.sleep(4*(2**i)); r=s.get(url,stream=True,timeout=90)
   print(url,r.status_code,r.headers.get('Content-Length'),r.url,flush=True)
   if r.status_code in (429,503): continue
   r.raise_for_status(); size=int(r.headers.get('Content-Length',0))
   if size>250*1024**2: print('SKIP oversized',flush=True);break
   p=root/slug/name; total=0
   with p.open('wb') as f:
    for c in r.iter_content(1024*1024):
     total+=len(c)
     if total>250*1024**2: raise ValueError('oversized')
     f.write(c)
   print('saved',p,total,flush=True);break
  except Exception as e:
   print(type(e).__name__,str(e),flush=True)
   if 'p' in locals() and p.exists() and p.stat().st_size==0: p.unlink()
