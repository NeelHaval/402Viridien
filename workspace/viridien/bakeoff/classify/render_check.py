"""Attempt the requested headless Edge check and validate the portable HTML."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import os
import subprocess
import time
from html.parser import HTMLParser
from classify.data import HERE, OUT, dump


class Audit(HTMLParser):
    def __init__(self):super().__init__();self.images=[];self.external=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='img':self.images.append(attrs.get('src',''))
        for name in ['src','href']:
            if attrs.get(name,'').startswith(('http:','https:','file:')):self.external.append(attrs[name])


def main():
    path=HERE/'report.html'
    audit=Audit();audit.feed(path.read_text(encoding='utf-8'))
    assert path.stat().st_size<15_000_000
    assert audit.images and all(x.startswith('data:image/webp;base64,') for x in audit.images)
    assert not audit.external
    target=OUT/('report_edge_'+time.strftime('%Y%m%d_%H%M%S')+'.png')
    profile=OUT/'edge-profile-final';cache=OUT/'edge-cache';tmp=OUT/'edge-tmp'
    for folder in [profile,cache,tmp]:folder.mkdir(exist_ok=True)
    cmd=['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe','--headless=new','--disable-gpu',
         '--no-first-run','--no-default-browser-check','--disable-extensions','--disable-crash-reporter',
         '--disable-breakpad','--user-data-dir='+str(profile),'--disk-cache-dir='+str(cache),
         '--window-size=390,844','--screenshot='+str(target),path.as_uri()]
    started=time.perf_counter();env=os.environ.copy();env.update(TEMP=str(tmp),TMP=str(tmp))
    result=dict(command=cmd,status='failed',html_bytes=path.stat().st_size,embedded_webp_images=len(audit.images),
                external_dependencies=0,viewport=[390,844],screenshot=str(target))
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=45,env=env)
        result.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        if target.exists() and target.stat().st_size>0:result['status']='passed'
    except (subprocess.TimeoutExpired,OSError) as exc:result['error']=str(exc)
    result['seconds']=time.perf_counter()-started
    dump(OUT/'edge_check.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
