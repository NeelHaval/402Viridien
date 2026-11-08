import json,sys
from pathlib import Path
from bs4 import BeautifulSoup
p=Path(sys.argv[1])
for f in [p/'d0f35f9152334fde89b05ffeaee38275.json',p/'0e37affa1b4042829a7eceebeeb66358.json']:
 def walk(x):
  if isinstance(x,str) and '<' in x:
   soup=BeautifulSoup(x,'html.parser')
   for row in soup.find_all('tr'):
    t=row.get_text(' ',strip=True)
    if any(k in t.lower() for k in ['mud','master','geolog','history','composite']):print(f.stem,t[:400],[(a.get('href')) for a in row.find_all('a')])
  elif isinstance(x,dict):
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(json.loads(f.read_text(encoding='utf-8')))
