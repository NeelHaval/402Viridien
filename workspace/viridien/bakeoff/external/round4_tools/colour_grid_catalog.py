import sys,urllib.request,re,json,time,html
for n in sys.argv[1:]:
 u='https://gdr.openei.org/submissions/'+n
 try:
  with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0 (research crawler)'}),timeout=30) as r: s=r.read().decode()
  print('\n',u,flush=True)
  for j in re.findall(r'<script type="application/ld\+json">(.*?)</script>',s,re.S):
   try:
    d=json.loads(j)
    for x in d.get('distribution',[]):
     if re.search('mud|lithol',str(x),re.I):print(x.get('contentUrl'),str(x.get('about'))[:300],flush=True)
   except Exception as e: print(e)
  for x in re.findall(r'<div class="resource-name" title="([^"]+)',s):
   if re.search('mud|lithol',x,re.I):print('resource:',html.unescape(x),flush=True)
 except Exception as e:print(e,flush=True)
 time.sleep(1.2)
