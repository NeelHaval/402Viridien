import sys,time,urllib.parse,urllib.request,xml.etree.ElementTree as ET
for q in sys.argv[1:]:
    u='https://www.bing.com/search?'+urllib.parse.urlencode({'format':'rss','q':q,'setlang':'en-US','cc':'us'})
    try:
        req=urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0 (research crawler)'})
        tree=ET.fromstring(urllib.request.urlopen(req,timeout=20).read())
        print('\nQUERY',q)
        for item in tree.findall('.//item')[:8]: print(' | '.join(item.findtext(k,'') for k in ('title','link','description')))
    except Exception as e: print(q,repr(e))
    time.sleep(1.1)
