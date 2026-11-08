import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='3'
from pathlib import Path
import json
from harness import io, legend_v4 as v4
ROOT=Path(__file__).resolve().parents[3]
excluded={r['log'] for r in json.loads((ROOT/'bakeoff/eval_set.json').read_text())}
for p in sorted((ROOT/'emil/labels').glob('*.json')):
    if p.stem in excluded: continue
    truth=json.loads(p.read_text()).get('legend',[])
    if not truth: continue
    log=io.open_log(p.stem)
    try:
        rgb,ev=v4.evidence(log)
        entries=v4.parse(log,dict(v4.DEFAULT,easy_weight=1.15))
        print(p.stem, '\nTRUTH',truth,'\nPRED',[(e.label,e.is_lithology) for e in entries],flush=True)
        for item in ev['items']: print(item,flush=True)
    finally:log.close()
