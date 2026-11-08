"""Build an immutable inference-only packet from the first three QA/gold cases."""
import hashlib,json,shutil,tarfile
from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parents[2]
O=B/'out/benedek_compare_first3';O.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(src,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
sources=list((B/'harness').glob('*.py'))+[B/'emil/paths.py',B/'strategies/__init__.py',B/'strategies/s24_benedek.py',B/'strategies/s24_benedek_core.py']
for p in sources:copy(p,O/'source'/p.relative_to(B))
copy(H/'paired_worker.py',O/'paired_worker.py')
m=json.loads((H.parent/'manifest_1108.json').read_text(encoding='utf-8'))
cases=[]
for c in [r for r in m['cases'] if r['suite'] in ('human_qa','human_gold')][:3]:
    r={k:c[k] for k in ('case_id','log','window','image_sha256','annotation_sha256')}
    for kind in ('image','annotation'):
        src=Path(c[kind]);target=O/'inputs'/(c['case_id']+src.suffix)
        if sha(src)!=c[kind+'_sha256']:raise ValueError('Input hash mismatch')
        copy(src,target);r[kind]=target.relative_to(O).as_posix()
    cases.append(r)
files={p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file() and p.name not in ('manifest.json','packet.tar.gz')}
data=dict(cases=cases,file_hashes=files,seed=1108,truth_supplied_to_worker=False,compute_timeout_seconds=300,queue_timeout_seconds=60,outer_timeout_seconds=380)
(O/'manifest.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
with tarfile.open(O/'packet.tar.gz','w:gz') as t:
    for p in O.rglob('*'):
        if p.is_file() and p.name!='packet.tar.gz':t.add(p,arcname=p.relative_to(O).as_posix())
print('PACKET_OK cases='+str(len(cases))+' source_inputs='+str(len(files))+' archive_sha256='+sha(O/'packet.tar.gz'))
