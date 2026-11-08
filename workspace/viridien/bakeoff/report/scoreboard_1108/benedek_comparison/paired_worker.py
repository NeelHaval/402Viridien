"""Inference only. The packet contains no truth rows."""
import hashlib,json,os,random,sys,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'source'))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import numpy as np
from PIL import Image
from harness import io
from strategies import s24_benedek
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2),encoding='utf-8');t.replace(p)
m=json.loads((P/'manifest.json').read_text(encoding='utf-8'))
for name,value in m['file_hashes'].items():
    if sha(P/name)!=value:raise ValueError('Input/source hash mismatch '+name)
random.seed(1108);np.random.seed(1108)
result=P/'predictions.json';done=json.loads(result.read_text(encoding='utf-8')) if result.exists() else {'records':[],'seed':1108}
finished={r['case_id'] for r in done['records']}
for case in m['cases']:
    if case['case_id'] in finished:continue
    row=dict(case);start=time.monotonic();log=None;rows=[]
    try:
        boxes=json.loads((P/case['annotation']).read_text(encoding='utf-8'))['bounding_boxes']
        log=io.Log(Image.open(P/case['image']),boxes,case['log'])
        rows=s24_benedek.run(log,tuple(case['window']),{})
    except Exception:
        row['error']=traceback.format_exc(limit=5)
    finally:
        if log:log.close()
    pred=P/(case['case_id']+'.csv');io.save_rows(pred,rows)
    row.update(prediction_csv=str(pred),prediction_sha256=sha(pred),seconds=time.monotonic()-start)
    done['records'].append(row);write(result,done)
    print('CASE_SAVED',case['case_id'],'error='+str(bool(row.get('error'))),flush=True)
