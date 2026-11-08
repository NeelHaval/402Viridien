"""Local-only API control on verified, pre-rendered boat crops."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent;OUT=HERE/'data/jev-e2e-20261108';sys.path.insert(0,str(HERE.parent/'bakeoff'))
from strategies import jev_sourcecards as api


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def extract(path,expected):
    if sha(path)!=expected:raise ValueError('Remote export archive hash mismatch')
    root=OUT/'boat-results';root.mkdir(exist_ok=True)
    with tarfile.open(path) as archive:
        for item in archive.getmembers():
            if item.name.startswith('/') or '..' in Path(item.name).parts or not (item.isfile() or item.isdir()):raise ValueError('Unsafe export archive')
        archive.extractall(root,filter='data')
    return root


def run(case_id):
    from PIL import Image
    import numpy as np
    root=OUT/'boat-results/exports'/case_id;data=read(root/'export.json');receipt=read(root/'receipt.json')
    if sha(root/'export.json')!=receipt['export_sha256']:raise ValueError('Export receipt mismatch')
    for p,h in receipt['artifacts'].items():
        if sha(root/p)!=h:raise ValueError('Artifact changed '+p)
    entries=[];queries=[];cards={}
    for e in data['entries']:
        with Image.open(root/e['path']) as im:a=np.asarray(im.convert('RGB'))
        entries.append(dict(id=e['id'],label=e['label'],pixels=a));cards[api.image_hash(a)]=(root/e['card']).read_bytes()
    for q in data['queries']:
        with Image.open(root/q['path']) as im:a=np.asarray(im.convert('RGB'))
        if api.image_hash(a)!=q['pixel_sha256']:raise ValueError('Query pixel hash changed')
        queries.append(dict(q,pixels=a));cards[q['pixel_sha256']]=(root/q['card']).read_bytes()
    # A semantic cache survives PNG compressor / platform differences. The book
    # keeps IDs and raw source pixels; no cross-document semantic label merging.
    ledger_path=OUT/'semantic-decisions.json';ledger=read(ledger_path) if ledger_path.exists() else {}
    def book_key(book):return api.digest([dict(id=e['id'],label=e['label'],source_sha256=e['source_sha256']) for e in book])
    def query_key(book,pixels):return api.digest(dict(version=api.VERSION,model=api.MODEL,legend=book,pixel_sha256=pixels,ranking='frozen top4 hand plus top2 Lab'))
    for record in read(OUT/'results.json')['records']:
        path=record.get('diagnostics')
        if not path or not Path(path).exists():continue
        diag=read(path);book=book_key(diag['legend'])
        for t in diag['trace']:
            c=t.get('jev')
            if c:ledger.setdefault(query_key(book,t['pixel_sha256']),dict(cell=c,source=record['case_id'],diagnostic_sha256=sha(path)))
    book=book_key([dict(id=e['id'],label=e['label'],source_sha256=api.image_hash(e['pixels'])) for e in entries])
    reused={};missing=[]
    for q in queries:
        cached=ledger.get(query_key(book,q['pixel_sha256']))
        if cached and cached['cell']['legend_id'] in q['candidates']:
            reused[q['item_id']]=dict(cached['cell'],item_id=q['item_id'])
        else:missing.append(q)
    # Image cards were rendered on boat. The local process only encodes API traffic.
    api.card=lambda a:cards[api.image_hash(a)]
    start=time.perf_counter();cells,details=api.classify(entries,missing,OUT/'cache/jev-api')
    cells.update(reused)
    details['semantic_cached_queries']=len(reused)
    for q in missing:
        c=cells.get(q['item_id'])
        if c:ledger[query_key(book,q['pixel_sha256'])]=dict(cell=c,source=case_id)
    api.write(ledger_path,ledger)
    by_pixel={q['pixel_sha256']:cells.get(q['item_id']) for q in queries}
    preprocessing=read(root/'preprocess.json') if (root/'preprocess.json').exists() else {}
    calibration=preprocessing.get('calibration');reference=read(HERE.parent/'bakeoff/report/scoreboard_1108/raw_results.json')['records']
    matching=[r for r in reference if r['case_id']==case_id]
    for r in matching if not preprocessing else []:
        if r.get('image_sha256')!=data['image_sha256'] or r['window']!=data['window']:raise ValueError('Baseline calibration input mismatch')
        value=r.get('initial_shared_depth_calibration') or r.get('depth_calibration')
        if value is not None:calibration=value;break
    decision_path=OUT/'boat-decisions'/f'{case_id}.json'
    elapsed=time.perf_counter()-start;history=[]
    if decision_path.exists():
        previous=read(decision_path)
        if previous['export_sha256']==sha(root/'export.json'):
            history=previous.get('run_history',[dict(seconds=previous['seconds'],api=previous['api'])])
    history.append(dict(seconds=elapsed,api=details))
    result=dict(case_id=case_id,export_sha256=sha(root/'export.json'),by_pixel=by_pixel,calibration=calibration,calibration_source=('Boat source-only OCR or verified exact-pixel depth cache' if preprocessing else 'Matched scoreboard shared OCR calibration; same image and window') if calibration else 'not available',preprocessing=preprocessing,api=details,seconds=sum(h['seconds'] for h in history),run_history=history,source_receipt_sha256=sha(root/'receipt.json'))
    api.write(decision_path,result)
    print(json.dumps(dict(case=case_id,queries=len(queries),returned=len(cells),fresh=details['fresh_requests'],cached=details['cached_requests'],semantic_cached_queries=len(reused),failures=len(details['failures']),seconds=result['seconds'],depth_available=bool(calibration))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case_id');p.add_argument('--archive',type=Path);p.add_argument('--archive-sha');a=p.parse_args()
    if a.archive:extract(a.archive,a.archive_sha)
    run(a.case_id)
