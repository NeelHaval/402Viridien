"""Stage only the next thirteen missing evaluation cases, with no truth."""
import hashlib,json,shutil,tarfile
from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parents[2];O=B/'out/benedek_compare_remaining';O.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cp(p,t):t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
for p in (B/'out/benedek_compare_first3/source').rglob('*.py'):cp(p,O/'source'/p.relative_to(B/'out/benedek_compare_first3/source'))
cp(H/'paired_worker.py',O/'paired_worker.py')
m=json.loads((H.parent/'manifest_1108.json').read_text(encoding='utf-8'))
done={r['case_id'] for r in json.loads((B/'out/benedek_compare_first3/predictions_local.json').read_text(encoding='utf-8'))['records']}
missing=[r for r in m['cases'] if r['case_id'] not in done];chosen=[r for r in missing if r['suite']=='viridien_eval'];cases=[]
for c in chosen:
 r={k:c[k] for k in ('case_id','log','window','image_sha256','annotation_sha256')}
 for kind in ('image','annotation'):
  src=Path(c[kind]);dest=O/'inputs'/(c['case_id']+src.suffix)
  if sha(src)!=c[kind+'_sha256']:raise ValueError('Input changed')
  cp(src,dest);r[kind]=dest.relative_to(O).as_posix()
 cases.append(r)
files={p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file() and p.name not in ('manifest.json','packet.tar.gz')}
data=dict(cases=cases,file_hashes=files,seed=1108,truth_supplied_to_worker=False,original_case_count=68,already_complete=sorted(done),not_staged=[r['case_id'] for r in missing if r not in chosen])
(O/'manifest.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
with tarfile.open(O/'packet.tar.gz','w:gz',compresslevel=1) as t:
 for p in O.rglob('*'):
  if p.is_file() and p.name!='packet.tar.gz':t.add(p,arcname=p.relative_to(O).as_posix())
launch=(H/'paired_launch.sh').read_text(encoding='utf-8').replace('benedek-first3-1010','benedek-remaining-1015')
(H/'paired_remaining_launch.sh').write_bytes(launch.encode('utf-8'))
print('REMAINING_PACKET_OK staged='+str(len(cases))+' not_staged='+str(len(data['not_staged']))+' sha256='+sha(O/'packet.tar.gz'))
