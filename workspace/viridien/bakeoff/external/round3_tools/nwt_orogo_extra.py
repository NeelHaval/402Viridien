import json,re,time,urllib.request,urllib.error,urllib.parse,pathlib
root=pathlib.Path('/home/user/lith/viridien_work/external_untrusted/round3/nwt_orogo'); data=json.loads((root/'downloads.json').read_text()); h=pathlib.Path('/tmp/nwt_record.html').read_text(); urls=[]
for href in re.findall(r'href="([^"]+)',h):
 if '/system/files/well-documents/' in href and (re.search(r'_SL_(?:HORIZONTAL|MD|TVD)\.pdf$',href,re.I) or 'mude%20log.PDF' in href): urls.append(urllib.parse.urljoin('https://www.orogo.gov.nt.ca',href))
for url in urls:
 for i in range(4):
  time.sleep(1.1)
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (research crawler)'}),timeout=100) as r:
    remaining=400*1024*1024-sum(x['bytes'] for x in data['documents']); lim=min(250*1024*1024,remaining); size=int(r.headers.get('Content-Length','0'))
    if size>lim: raise ValueError('file exceeds size budget')
    b=r.read(lim+1)
    if len(b)>lim: raise ValueError('file exceeds size budget')
    if not b.startswith(b'%PDF'): raise ValueError('not a PDF')
    name=f"{len(data['documents'])+1:02}.pdf"; (root/name).write_bytes(b); data['documents'].append({'url':url,'path':name,'bytes':len(b)}); print(name,len(b),flush=True); break
  except urllib.error.HTTPError as e:
   if e.code in (429,503) and i<3: time.sleep(2**(i+1)); continue
   data['blocked'].append(url+': '+str(e)); break
  except Exception as e: data['blocked'].append(url+': '+str(e)); print(str(e),flush=True); break
(root/'downloads.json').write_text(json.dumps(data,indent=2))
