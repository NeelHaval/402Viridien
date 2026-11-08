import time,urllib.request,urllib.parse,html.parser,base64,sys
class P(html.parser.HTMLParser):
 def __init__(self):super().__init__();self.h=False;self.a=False;self.title='';self.url=''
 def handle_starttag(self,t,a):
  d=dict(a)
  if t=='h2':self.h=True
  if self.h and t=='a':self.a=True;self.url=d.get('href','');self.title=''
 def handle_data(self,d):
  if self.a:self.title+=d
 def handle_endtag(self,t):
  if t=='a' and self.a:
   self.a=False; u=urllib.parse.parse_qs(urllib.parse.urlparse(self.url).query).get('u',[''])[0]
   if u.startswith('a1'):
    try:self.url=base64.urlsafe_b64decode(u[2:]+'===').decode()
    except:pass
   print(self.title.strip(),self.url,flush=True)
  if t=='h2':self.h=False
for q in sys.argv[1:]:
 print('\nQUERY',q,flush=True)
 req=urllib.request.Request('https://www.bing.com/search?'+urllib.parse.urlencode({'q':q,'setlang':'en','cc':'us','count':15}),headers={'User-Agent':'Mozilla/5.0 (research crawler)'})
 try:
  with urllib.request.urlopen(req,timeout=30) as r:P().feed(r.read().decode('utf8','replace'))
 except Exception as e:print('ERROR',e,flush=True)
 time.sleep(1.2)
