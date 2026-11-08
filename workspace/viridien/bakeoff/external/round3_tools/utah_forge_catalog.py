import requests,time,pathlib
root=pathlib.Path('/home/user/lith/viridien_work/external_untrusted/round3')
items=[('coso_forge','https://data.openei.org/submissions/9340','catalog9340.html'),('coso_forge','https://data.openei.org/submissions/7000','catalog7000.html'),('utah_forge','https://data.amerigeoss.org/dataset/utah-forge-well-16a78-32-logs-9c9f1','catalog.html'),('coso_forge','https://data.amerigeoss.org/dataset/west-flank-coso-forge-52b-7-image-and-mud-log-a9623','catalog.html')]
s=requests.Session();s.headers['User-Agent']='Mozilla/5.0 (research crawler)'
for slug,url,name in items:
 for i in range(4):
  time.sleep(4*2**i)
  try:
   r=s.get(url,timeout=40);print(url,r.status_code,len(r.content),r.url,flush=True)
   if r.status_code in (429,503):continue
   if r.ok:(root/slug/name).write_bytes(r.content)
   break
  except Exception as e:print(str(e),flush=True)
