import time, urllib.request, urllib.error, urllib.parse, re, json, pathlib, xml.etree.ElementTree as ET, io
from PIL import Image, ImageOps, ImageDraw
root=pathlib.Path('external_untrusted/round3/mississippi'); root.mkdir(parents=True,exist_ok=True); (root/'_thumbs').mkdir(exist_ok=True)
base='https://geology.deq.ms.gov/energy/'
last=0; blocked=[]; total=sum(p.stat().st_size for p in root.glob('*.png')); limit=400_000_000

def get(url, maxsize=250_000_000):
 global last
 for n in range(4):
  time.sleep(max(0,1.05-(time.monotonic()-last))); last=time.monotonic()
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (research crawler)'}),timeout=35) as r:
    sz=int(r.headers.get('Content-Length','0'))
    if sz>maxsize: raise ValueError('file exceeds size limit')
    data=r.read(maxsize+1)
    if len(data)>maxsize: raise ValueError('file exceeds size limit')
    return data
  except Exception as e:
   if isinstance(e,urllib.error.HTTPError) and e.code in (429,503): time.sleep(2**n)
   elif isinstance(e, urllib.error.HTTPError) and e.code in (400,403,404): break
   elif isinstance(e,ValueError): break
 blocked.append(url+': '+str(e)); return None

counties=['Hancock','Harrison','Jackson']; apis={}
for c in counties:
 url=base+'search.aspx?filter=004000%3Bcounty='+c+'%3Bapi='
 b=get(url); apis[c]=list(dict.fromkeys(re.findall(r'wellfolder\.aspx\?API=(\d+)',b.decode('utf-8','replace'),re.I))) if b else []
 print(c,len(apis[c]),flush=True)
# Preserve known successful Hancock first, then interleave search results.
queue=[('Hancock','2304500036')]
for i in range(max(map(len,apis.values()),default=0)):
 for c in counties:
  if i<len(apis[c]): queue.append((c,apis[c][i]))
seen=set(); logs=[]; docs=[]; folders=0
for county,api in queue:
 if len(logs)>=40: break
 if api[:10] in seen: continue
 seen.add(api[:10]); folders+=1
 url=base+'wellfolder.aspx?api='+api
 b=get(url)
 if not b: continue
 html=b.decode('utf-8','replace')
 xmls=[]
 for row in re.findall(r'<tr\b[^>]*>.*?</tr>',html,re.I|re.S):
  if re.search(r'Mud Log (?:1|2|5) inch',row,re.I):
   xmls.extend(re.findall(r'logviewer\.aspx\?x=([^"&]+)',row,re.I))
 for xmlurl in xmls:
  if len(logs)>=40: break
  xmlurl=urllib.parse.unquote(xmlurl); b=get(xmlurl)
  if not b: continue
  try: x=ET.fromstring(b); pages=[a.attrib['href'] for a in x.findall('.//LogImagePage')]
  except Exception as e: blocked.append(xmlurl+': XML parse error '+str(e)); continue
  if not pages: blocked.append(xmlurl+': no page images'); continue
  lid=pathlib.PurePosixPath(urllib.parse.urlparse(xmlurl).path).stem
  logs.append({'id':lid,'county':county,'api':api,'descriptor':xmlurl,'page_count':len(pages)})
  print('LOG',len(logs),county,lid,len(pages),flush=True)
  for i,href in enumerate(pages,1):
   imageurl=urllib.parse.urljoin(xmlurl,href); path=root/pathlib.PurePosixPath(href).name
   if path.exists(): data=path.read_bytes()
   else:
    if total>=limit: blocked.append('400 MB total budget reached'); break
    data=get(imageurl,min(250_000_000,limit-total))
    if not data: continue
    try: Image.open(io.BytesIO(data)).verify()
    except Exception as e: blocked.append(imageurl+': invalid image '+str(e)); continue
    path.write_bytes(data); total+=len(data)
   im=Image.open(io.BytesIO(data)); w,h=im.size
   thumb=ImageOps.contain(im.convert('RGB'),(1200,1200),Image.Resampling.LANCZOS)
   if thumb.height!=1200: thumb=thumb.resize((round(w*1200/h),1200),Image.Resampling.LANCZOS)
   thumb.save(root/'_thumbs'/path.name,optimize=True)
   docs.append({'url':imageurl,'path':str(path),'bytes':len(data),'pages':1,'page_size_pixels':[w,h],'raster_or_vector':'raster','producer':im.info.get('Software'), 'creator':None,'is_log':None,'has_lith_column':None,'has_legend':None,'legend_page':None,'layout_guess':'other','born_digital_wellsight':False,'notes':f'{county}; API {api}; viewer log {lid}, page {i} of {len(pages)}; descriptor {xmlurl}. Visual review pending.'})
  (root/'progress.json').write_text(json.dumps({'logs':logs,'documents':docs,'blocked':blocked,'search_counts':{c:len(apis[c]) for c in counties},'folders_examined':folders},indent=2))
(root/'progress.json').write_text(json.dumps({'logs':logs,'documents':docs,'blocked':blocked,'search_counts':{c:len(apis[c]) for c in counties},'folders_examined':folders},indent=2))
print('DONE',len(logs),len(docs),total,folders,flush=True)
