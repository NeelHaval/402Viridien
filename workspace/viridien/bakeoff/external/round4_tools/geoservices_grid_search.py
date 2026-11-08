import urllib.request,urllib.parse,re,html,time,sys
queries=['"Geoservices" "mud log" filetype:pdf well','"Geoservices" "mud log" site:gov.au','"geoservices" "mud log" site:nlog.nl','"geoservices" "mud log" site:nzpam.govt.nz','"geoservices" "master log" well filetype:pdf','"mud log" "geoservices" offshore well completion report','"mud log" offshore filetype:pdf lithology legend','"geoservices" "mudlog" filetype:pdf','site:search.geoscience.gov.au "mud log" geoservices']
for q in queries:
 try:
  u='https://search.brave.com/search?'+urllib.parse.urlencode({'q':q,'source':'web'})
  data=urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0 (research crawler)'}),timeout=25).read().decode('utf-8','replace')
  print('\nQUERY',q)
  for part in re.findall(r'<a\b[^>]*href="https?://[^>]+>.*?</a>',data,re.S):
   m=re.search(r'href="([^"]+)',part); title=html.unescape(re.sub('<[^>]*>',' ',part)); title=' '.join(title.split()); url=html.unescape(m.group(1)) if m else ''
   if title and not any(x in url for x in ('brave.com','google.com','bing.com','mojeek.com')): print(title[:150],url)
 except Exception as e: print('ERROR',e)
 time.sleep(1.1)
