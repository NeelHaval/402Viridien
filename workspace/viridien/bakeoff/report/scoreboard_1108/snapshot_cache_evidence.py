"""Snapshot only declared-case preprocessing caches, without changing inference."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;B=HERE.parents[1];sys.path.insert(0,str(B))
from harness import io
from PIL import Image
import numpy as np

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def stamp():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main(phase):
 manifest=read(HERE/'manifest_1108.json');root=B/'out'/'scoreboard_1108'/'cache_evidence';store=root/'store';store.mkdir(parents=True,exist_ok=True)
 first=read(root/'first_observations.json') if (root/'first_observations.json').exists() else {};entries=[];missing=[];errors=[];consumed={}
 rawpath=HERE/'raw_results.json'
 if rawpath.exists():
  for r in read(rawpath)['records']:consumed.setdefault(r['log'],[]).append(r['strategy'])
 def capture(path,kind,logs):
  path=Path(path)
  if not path.exists():missing.append(dict(path=str(path),kind=kind,logs=logs));return
  try:
   digest=sha(path);dest=store/(digest[:24]+path.suffix)
   if not dest.exists():shutil.copyfile(path,dest)
   if sha(dest)!=digest:raise RuntimeError('Source changed during copy')
   key=str(path);now=stamp()
   first.setdefault(key,dict(first_observed_utc=now,phase=phase,sha256=digest))
   entries.append(dict(path=key,kind=kind,logs=logs,sha256=digest,snapshot=str(dest),observed_utc=now,first_observation=first[key],already_scored={log:consumed.get(log,[]) for log in logs}))
  except Exception as exc:errors.append(dict(path=str(path),error=f'{type(exc).__name__}: {exc}'))
 cases={c['log']:c for c in manifest['cases']}
 # Exact declared basenames only. Never enumerate other dataset caches.
 for name,c in cases.items():
  lp=B/'out'/'_cache'/'legend'/(name+'.json');capture(lp,'shared_legend',[name])
  if lp.exists():
   try:
    for e in read(lp).get('entries',[]):
     filename=e.get('swatch_file','');sp=lp.parent/filename
     if sp.parent.resolve()!=lp.parent.resolve() or not filename.startswith(name+'.'):raise ValueError('Unsafe swatch reference')
     capture(sp,'shared_legend_swatch',[name])
   except Exception as exc:errors.append(dict(path=str(lp),error=str(exc)))
  for engine in ('rapidocr','easyocr'):capture(B/'out'/'_cache'/'depth'/(name+'.'+engine+'.json'),'shared_depth',[name])
 # S19 names its raw OCR evidence by exact native legend pixels. Derive only
 # from declared assets; do not enumerate/open unrelated raw OCR cache files.
 for name,c in cases.items():
  log=None
  try:
   log=io.Log(Image.open(c['image']),read(Path(c['annotation']))['bounding_boxes'],name)
   rgb=io.read_rgb(log,'lithology_legend');digest=hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest();filename='v1_'+digest+'.json'
   capture(B/'out'/'scoreboard_1108'/'s19_runtime'/'legend_cache'/'cache'/filename,'s19_run_owned_raw_ocr',[name])
   capture(B/'ml'/'legend_names'/'cache'/filename,'s19_copy_source_raw_ocr',[name])
  except Exception as exc:errors.append(dict(log=name,error=f'{type(exc).__name__}: {exc}'))
  finally:
   if log is not None:log.close()
 # Declared installed weight locations only; no dataset or sealed-cache scan.
 for folder in (Path.home()/'.EasyOCR'/'model',B/'out'/'_cache'/'easyocr'/'model'):
  for name in ('craft_mlt_25k.pth','english_g2.pth'):capture(folder/name,'easyocr_weight',[])
 rapid=Path(sys.executable).parent/'Lib'/'site-packages'/'rapidocr_onnxruntime'/'models'
 for name in ('ch_PP-OCRv4_det_infer.onnx','ch_PP-OCRv4_rec_infer.onnx','ch_ppocr_mobile_v2.0_cls_infer.onnx'):capture(rapid/name,'rapidocr_weight',[])
 result=dict(phase=phase,observed_utc=stamp(),manifest_sha256=sha(HERE/'manifest_1108.json'),approved_logs=len(cases),entries=entries,missing=missing,errors=errors,
  limitation='First-observed and end-state evidence only. No exact per-case cache read interception was installed. Historical consumed cache bytes and initial OCR misses were not pinned before the run; do not claim a fully frozen end-to-end comparison.')
 if phase=='final' and (root/'initial.json').exists():
  initial=read(root/'initial.json');before={e['path']:e['sha256'] for e in initial['entries']};after={e['path']:e['sha256'] for e in entries}
  result['comparison']=dict(changed=[p for p in before.keys()&after.keys() if before[p]!=after[p]],new=[p for p in after if p not in before],not_observed_at_end=[p for p in before if p not in after])
 (root/'first_observations.json').write_text(json.dumps(first,indent=2),encoding='utf-8');(root/(phase+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(f'CACHE {phase}: {len(entries)} captured; {len(missing)} absent; {len(errors)} capture errors; {len(cases)} declared logs only')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['initial','final'],required=True);main(p.parse_args().phase)
