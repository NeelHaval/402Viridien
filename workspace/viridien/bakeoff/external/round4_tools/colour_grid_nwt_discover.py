import urllib.request,urllib.parse,urllib.error,time,re,json,sys,html.parser
base='https://www.orogo.gov.nt.ca'; ua={'User-Agent':'Mozilla/5.0 (research crawler)'}; last=0
class P(html.parser.HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,t,a):
  d=dict(a)
  if t=='a' and 'href' in d:self.links.append(d['href'])
def get(u):
 global last
 time.sleep(max(0,1.2-(time.monotonic()-last)));last=time.monotonic()
 with urllib.request.urlopen(urllib.request.Request(u,headers=ua),timeout=25) as r:p=P();p.feed(r.read().decode('utf8','replace'));return p.links
leads=[]; seen=set()
for term in ['Canol','Mackenzie','Norman Wells']:
 try:
  links=get(base+'/en/well-data?items_per_page=100&aggregated_field='+urllib.parse.quote(term)); wells=list(dict.fromkeys(x for x in links if x.startswith('/en/well-data/') and '?' not in x))
  print(term,len(wells),flush=True)
  for w in wells[:5]:
   if w in seen:continue
   seen.add(w)
   try:
    links=get(base+w)
    for x in links:
     if '/system/files/well-documents/' in x and re.search(r'(mud.?log|strip.?log|_SL_)',urllib.parse.unquote(x),re.I) and re.search(r'\.pdf$',x,re.I):
      u=urllib.parse.urljoin(base,x);leads.append({'well':base+w,'url':u}); print(u,flush=True)
   except urllib.error.HTTPError as e:
    print('error',e,flush=True)
    if e.code in (429,503):raise
 except Exception as e:
  print('STOP',e,flush=True)
  if isinstance(e,urllib.error.HTTPError) and e.code in (429,503):break
open(sys.argv[1],'w').write(json.dumps({'wells_examined':len(seen),'leads':leads},indent=2))
