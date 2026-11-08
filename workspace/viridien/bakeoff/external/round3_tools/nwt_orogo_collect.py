import urllib.request, urllib.parse, urllib.error, time, json, re, os, pathlib
from html.parser import HTMLParser
root=pathlib.Path('/home/user/lith/viridien_work/external_untrusted/round3/nwt_orogo')
base='https://www.orogo.gov.nt.ca'
last=0
class Links(HTMLParser):
 def __init__(self): super().__init__(); self.links=[]
 def handle_starttag(self,tag,attrs):
  if tag=='a':
   d=dict(attrs)
   if d.get('href'): self.links.append(d['href'])
def get(url, limit=250*1024*1024):
 global last
 for i in range(4):
  time.sleep(max(0,1.1-(time.monotonic()-last))); last=time.monotonic()
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (research crawler)'}),timeout=90) as r:
    length=int(r.headers.get('Content-Length','0'))
    if length>limit: raise ValueError('size exceeds limit: '+str(length))
    data=r.read(limit+1)
    if len(data)>limit: raise ValueError('size exceeds limit')
    return data
  except urllib.error.HTTPError as e:
   if e.code in (429,503) and i<3: time.sleep(2**(i+1)); continue
   raise
 raise ValueError('failed after four attempts')
leads=json.load(open('/home/user/lith/viridien_work/bakeoff/notes/external/round3/merged.json'))
examples=next(x['example_urls'][:2] for x in leads if x['name'].startswith('Northwest Territories OROGO'))
urls=list(examples); seen=set(); errors=[]
for term in ['Mirror Lake','Cameron','Colville','Fort Liard','Stewart']:
 try:
  html=get(base+'/en/well-data?items_per_page=100&aggregated_field='+urllib.parse.quote(term)).decode('utf-8','replace'); p=Links(); p.feed(html)
  wells=list(dict.fromkeys(x for x in p.links if x.startswith('/en/well-data/') and '?' not in x))
  print(term, 'wells',len(wells),flush=True)
  for well in wells[:8]:
   if well in seen: continue
   seen.add(well)
   try:
    h=get(base+well).decode('utf-8','replace'); q=Links(); q.feed(h)
    matches=[urllib.parse.urljoin(base,x) for x in q.links if '/system/files/well-documents/' in x and re.search(r'strip.?log|mud.?log|litholog',urllib.parse.unquote(x),re.I) and re.search(r'\.pdf(?:\?|$)',x,re.I)]
    for x in matches:
     if x not in urls: urls.append(x); print('found',urllib.parse.unquote(x),flush=True)
   except Exception as e: errors.append(well+': '+str(e))
   if len(urls)>=50: break
  if len(urls)>=50: break
 except Exception as e: errors.append(term+': '+str(e))
print('discovered',len(urls),'wells',len(seen),flush=True)
(root/'discovery.json').write_text(json.dumps({'urls':urls,'wells_checked':len(seen),'errors':errors},indent=2))
total=0; docs=[]
for i,url in enumerate(urls[:40],1):
 try:
  data=get(url,min(250*1024*1024,400*1024*1024-total)); total+=len(data)
  if not data.startswith(b'%PDF'): raise ValueError('response is not PDF')
  name=f'{i:02d}.pdf'; (root/name).write_bytes(data); docs.append({'url':url,'path':name,'bytes':len(data)})
  print('downloaded',name,len(data),flush=True)
 except Exception as e: errors.append(url+': '+str(e)); print('blocked',url,str(e),flush=True)
(root/'downloads.json').write_text(json.dumps({'documents':docs,'blocked':errors,'discovered':len(urls),'wells_checked':len(seen)},indent=2))
