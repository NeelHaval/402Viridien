import requests,time,pathlib
url='https://gdr.openei.org/files/1292/16A%2878%29-32%20Mud%20Log%20Final%20130-10987%27.pdf'
p=pathlib.Path('/home/user/lith/viridien_work/external_untrusted/round3/utah_forge/mud_log.pdf')
for i in range(4):
 time.sleep(4*2**i)
 try:
  with requests.get(url,headers={'User-Agent':'Mozilla/5.0 (research crawler)'},stream=True,timeout=90) as r:
   print(r.status_code,r.headers.get('Content-Length'),flush=True)
   if r.status_code in (429,503): continue
   r.raise_for_status()
   if int(r.headers.get('Content-Length',0))>250*1024**2: print('oversized');break
   n=0
   with p.open('wb') as f:
    for c in r.iter_content(1048576):
     n+=len(c)
     if n>250*1024**2: raise ValueError('oversized')
     f.write(c)
   print('saved',n,flush=True);break
 except Exception as e:
  p.unlink(missing_ok=True);print(str(e),flush=True)
