"""Run the unchanged harness with outputs/caches confined to this lane.
Usage: python bakeoff/analysis/transfer/run_external.py [strategy ...]
"""
import os
from pathlib import Path
import sys
import shutil
import json
import importlib

HERE = Path(__file__).resolve().parent
BAKE = HERE.parents[1]
REPO = BAKE.parent
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '3'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['LITHOLOGY_DATASET'] = str(REPO / 'external_tests')
os.environ['MPLCONFIGDIR'] = str(HERE / 'mplconfig')
sys.path.insert(0, str(BAKE))
sys.path.insert(0, str(HERE/'frozen'))
import torch
import cv2
torch_set_threads = torch.set_num_threads
torch.set_num_threads = lambda n: torch_set_threads(min(3, n))
torch.set_num_threads(3)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)
from harness import io, run, legend
RUNTIME = HERE / 'runtime'
io.ROOT = RUNTIME
# Strategy imports still come from the actual repository, without modifying it.
def strategy(name):
    matches = sorted((BAKE / 'strategies').glob(name + '*.py'))
    if len(matches) != 1:
        raise ValueError(matches)
    return importlib.import_module('strategies.' + matches[0].stem)
run._strategy = strategy
# Reuse existing image-derived preprocessing for dev pages only.
items = json.loads((REPO / 'external_tests/eval_set_dev.json').read_text())
for item in items:
    name = item['log']
    for kind in ('legend', 'depth'):
        for src in (BAKE / 'out/_cache' / kind).glob(name + '.*'):
            dst = RUNTIME / 'out/_cache' / kind / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copyfile(src, dst)
# Prevent S12's OCR helper from writing outside this lane.
from emil import paths

def owned_ocr_reader():
    import easyocr
    cache = HERE / 'runtime/easyocr'
    model = cache / 'model'
    model.mkdir(parents=True, exist_ok=True)
    for src in (BAKE / 'emil/cache/easyocr/model').glob('*'):
        if src.is_file() and not (model/src.name).exists():
            shutil.copyfile(src, model/src.name)
    return easyocr.Reader(['en'], gpu=False, verbose=False,
        model_storage_directory=str(model), user_network_directory=str(cache/'user_network'), download_enabled=False)
paths.ocr_reader = owned_ocr_reader
if '--synthetic' in sys.argv:
    io.DATASET = Path(os.environ.get('LITHOLOGY_VIRIDIEN_DATASET',REPO.parent/'dataset')).expanduser()
    vir_items = json.loads((BAKE/'eval_set.json').read_text())
    for item in vir_items:
        for kind in ('legend','depth'):
            for src in (BAKE/'out/_cache'/kind).glob(item['log']+'.*'):
                dst=RUNTIME/'out/_cache'/kind/src.name
                dst.parent.mkdir(parents=True,exist_ok=True)
                if not dst.exists():shutil.copyfile(src,dst)
    run.main(['--strategy','s07','--set','synthetic','--items',str(BAKE/'eval_set.json'),'--gt','silver/B'])
    sys.exit(0)
retry_errors = '--retry-errors' in sys.argv
arguments = [a for a in sys.argv[1:] if a != '--retry-errors']
for prefix in arguments or ['b01', 's01', 's02', 's03', 's04', 's05', 's07', 's11', 's12']:
    print('START', prefix, flush=True)
    strategy_id = strategy(prefix).__name__.split('.')[-1]
    metrics_path = RUNTIME/'out'/strategy_id/'ext_dev/metrics.json'
    previous = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    previous_logs = previous['logs'] if previous else []
    completed = {(r['log'],tuple(r.get('window',()))) for r in previous_logs if not (retry_errors and 'error' in r)}
    # Preprocessing errors have a window in this harness, so they resume consistently.
    pending = [i for i in items if (i['log'],(i['y0'],i['y1'])) not in completed]
    if not pending:
        print('COMPLETE', prefix, flush=True)
        continue
    item_path = HERE/(prefix+'_pending.json')
    item_path.write_text(json.dumps(pending,indent=2))
    if previous:
        (metrics_path.parent/'metrics_before_resume.json').write_text(json.dumps(previous,indent=2))
    run.main(['--strategy', prefix, '--items', str(item_path),
        '--gt-dir', str(REPO/'external_tests/truth'), '--out-tag', 'ext_dev'])
    current = json.loads(metrics_path.read_text())
    by_window = {(r['log'],tuple(r.get('window',()))):r for r in previous_logs+current['logs']}
    current['logs'] = [by_window[(i['log'],(i['y0'],i['y1']))] for i in items]
    current['mean'] = run.aggregate(current['logs'])
    metrics_path.write_text(json.dumps(current,indent=2,allow_nan=False))
