import time,urllib.parse,urllib.request,html.parser,sys
class P(html.parser.HTMLParser):
 def __init__(self): super().__init__(); self.on=False; self.link=''; self.title=''
 def handle_starttag(self,t,a):
  d=dict(a)
  if t=='a' and 'result__a' in d.get('class',''): self.on=True; self.link=d.get('href',''); self.title=''
 def handle_data(self,d):
  if self.on:self.title+=d
 def handle_endtag(self,t):
  if t=='a' and self.on:
   print(self.title.strip(),self.link,flush=True); self.on=False
for q in sys.argv[1:]:
 print('\nQUERY',q,flush=True)
 req=urllib.request.Request('https://html.duckduckgo.com/html/?'+urllib.parse.urlencode({'q':q}),headers={'User-Agent':'Mozilla/5.0 (research crawler)'})
 try:
  with urllib.request.urlopen(req,timeout=30) as r: P().feed(r.read().decode('utf-8','replace'))
 except Exception as e:print('ERROR',e,flush=True)
 time.sleep(1.2)
