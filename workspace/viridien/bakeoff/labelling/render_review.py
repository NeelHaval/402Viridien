"""Render the self-contained human review once with installed headless Edge."""
from pathlib import Path
import subprocess
import time
from llm_label import BASE, OUT, dump, read, redact

edge=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
shot=BASE/'review-check.png'
cmd=[str(edge),'--headless=new','--disable-gpu','--no-first-run',
     '--no-default-browser-check','--disable-extensions',
     '--user-data-dir='+str(OUT/'_edge_profile'),
     '--disk-cache-dir='+str(OUT/'_edge_cache'),
     '--window-size=1440,1100','--screenshot='+str(shot),
     (BASE/'review.html').resolve().as_uri()]
result={'command':cmd,'status':'failed'}
started=time.perf_counter()
try:
    proc=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace',
                        timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
    result.update(returncode=proc.returncode,stdout=redact(proc.stdout),stderr=redact(proc.stderr))
    if shot.exists() and shot.stat().st_size:
        result.update(status='rendered',screenshot=str(shot),bytes=shot.stat().st_size)
except Exception as exc:
    result['error']=redact(str(exc))
result['seconds']=time.perf_counter()-started
dump(OUT/'render_check.json',result)
summary=read(OUT/'summary.json')
summary['edge_render_status']=result['status']
summary['wall_seconds']=time.time()-read(OUT/'timing.json')['started_epoch']
dump(OUT/'summary.json',summary)
print({k:v for k,v in result.items() if k not in ['command','stdout','stderr']})
