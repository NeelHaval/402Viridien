import urllib.request,urllib.error,time,sys,pathlib,json,re,html.parser,urllib.parse
base='https://www.orogo.gov.nt.ca'; out=pathlib.Path(sys.argv[1]); ua={'User-Agent':'Mozilla/5.0 (research crawler)'}
class L(html.parser.HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,t,a):
  d=dict(a)
  if t=='a' and d.get('href'):self.links.append(d['href'])
u=base+'/en/well-data/fall-stone-f-01'
with urllib.request.urlopen(urllib.request.Request(u,headers=ua),timeout=30) as r:p=L();p.feed(r.read().decode())
urls=[urllib.parse.urljoin(base,x) for x in p.links if 'Morning%20Report' in x and 'Strip%20Log.pdf' in x][:3]
rows=[]
for i,u in enumerate(urls):
 time.sleep(1.2)
 try:
  with urllib.request.urlopen(urllib.request.Request(u,headers=ua),timeout=45) as r:
   if int(r.headers.get('Content-Length','0'))>150000000:raise ValueError('oversize')
   b=r.read(150000001)
  if len(b)>150000000 or not b.startswith(b'%PDF'):raise ValueError('oversize or not PDF')
  dest=out/f'fall_stone_{i+1}.pdf';dest.write_bytes(b);rows.append({'url':u,'path':str(dest),'bytes':len(b)});print(u,len(b),flush=True)
 except urllib.error.HTTPError as e:
  print(u,e,flush=True)
  if e.code in (429,503):break
 except Exception as e:print(u,e,flush=True)
(out/'downloads.json').write_text(json.dumps(rows,indent=2))
