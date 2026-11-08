"""Measure inference on seven 6,000-row synthetic columns only."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '3'
import json
import time
from PIL import Image
from harness import io
from harness.run import Resources
from ml.synth_v2.generator import HERE, FAMILIES, make
from ml.synth_v2.experiment import entries_for, dump
from ml.synth_v2.runtime import load
from strategies.s21_synth_v2 import run


def main():
    sources = json.loads((HERE/'assets.json').read_text())
    records = []
    with Resources() as resources:
        load()
        for i,family in enumerate(FAMILIES):
            source = next(s for s in sources if s['family']==family and s['split']=='held')
            rgb,_ = make(source, 2190000+i, height=6000)
            log = io.Log(Image.fromarray(rgb),
                         {'interpreted_lithology_column':[0,0,rgb.shape[1],len(rgb)]},'synthetic_smoke')
            entries = entries_for(source)
            started = time.perf_counter()
            rows = run(log,(0,len(rgb)),{'legend':entries})
            elapsed = time.perf_counter()-started
            assert rows and elapsed < 60
            records.append(dict(family=family,seconds=elapsed,rows=len(rows),height=len(rgb)))
            log.close()
    dump('smoke.json',dict(records=records,peak_rss_gb=resources.peak/1024**3,
                           note='Synthetic speed check only. No scores or selection.'))


if __name__ == '__main__':
    main()
