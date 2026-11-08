import requests,time
from urllib.parse import urljoin
from html.parser import HTMLParser
class BeautifulSoup(HTMLParser):
 def __init__(self, text, parser):
  super().__init__(); self.links=[]; self.parts=[]; self.feed(text)
 def handle_starttag(self, tag, attrs):
  if tag=='a': self.links.append(dict(attrs))
 def handle_data(self, data): self.parts.append(data)
 def select(self, q): return [Link(x) for x in self.links if 'href' in x]
 def get_text(self, *args, **kwargs): return ' '.join(self.parts)
class Link(dict):
 def get_text(self, *args, **kwargs): return ''
urls=['https://www.oregon.gov/robots.txt','https://www.oregon.gov/dogami/mlrr/Pages/oilgas-logs.aspx','https://petrora.com/robots.txt','https://petrora.com/terms','https://petrora.com/logs']
s=requests.Session(); s.headers['User-Agent']='Mozilla/5.0 (research crawler)'
last={}
for url in urls:
 host=requests.utils.urlparse(url).netloc
 wait=1-(time.monotonic()-last.get(host,0))
 if wait>0: time.sleep(wait)
 try:
  last[host]=time.monotonic(); r=s.get(url,timeout=45); print('\nURL',url,'STATUS',r.status_code,'FINAL',r.url,'LEN',len(r.content))
  if 'oilgas-logs' in url:
   soup=BeautifulSoup(r.text,'html.parser'); links=[(a.get_text(' ',strip=True),urljoin(r.url,a.get('href',''))) for a in soup.select('a[href]') if 'mud' in a.get('href','').lower()]; print('MUD LINKS',len(links)); print('\n'.join(str(x) for x in links))
  else: print(BeautifulSoup(r.text,'html.parser').get_text(' ',strip=True)[:14000])
 except Exception as e: print(type(e).__name__,str(e))
