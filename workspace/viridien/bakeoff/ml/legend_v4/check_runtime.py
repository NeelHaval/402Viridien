"""Run S20 once on an allowed dev window. Check structure, not final truth."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='3'
import hashlib
import json
import time
import torch
from harness import io,legend_v4
from harness.run import Resources
from strategies import s20_legend_v4
from ml.legend_v4.experiment import HERE,EXT,allowed

def main():
    torch.set_num_threads(3)
    items,_=allowed();item=items[0];io.DATASET=EXT
    log=io.open_log(item['log']);win=(item['y0'],item['y1'])
    started=time.perf_counter()
    try:
        with Resources() as resource:
            rows=s20_legend_v4.run(log,win,dict(device='cpu',cache_dir=HERE/'runtime_cache'))
            labels={e.label for e in legend_v4.parse(log) if e.is_lithology}|{'Unknown'}
        last=win[0]
        for r in rows:
            assert last<=r['top_y']<r['bottom_y']<=win[1]
            assert r['lithology'] in labels
            assert isinstance(r['cluster_id'],int)
            assert 0<=r['confidence']<=1
            last=r['bottom_y']
        assert rows
        result=dict(log=log.name,window=win,intervals=len(rows),valid=True,
                    seconds=time.perf_counter()-started,peak_rss_gb=resource.peak/1024**3,
                    weights='inherited S18; historical Forsyth exposure; structure check only',truth_read=False)
        (HERE/'runtime_check.json').write_text(json.dumps(result,indent=2))
        print(result)
    finally:log.close()

if __name__=='__main__':main()
