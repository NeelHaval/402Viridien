"""Read live service state and prepare a same-reviewer continuation file."""
import json,sqlite3,sys,urllib.request,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import Store
ROOT=Path(__file__).resolve().parents[1];data=ROOT/'data'
with sqlite3.connect(data/'shared-reviews.sqlite3') as db:
    db.row_factory=sqlite3.Row
    session=dict(db.execute('SELECT s.*,COUNT(o.id) saves FROM sessions s LEFT JOIN operations o ON o.session=s.id GROUP BY s.id ORDER BY saves DESC LIMIT 1').fetchone())
def get(path):
    req=urllib.request.Request('http://127.0.0.1:8776'+path,headers={'Authorization':'Bearer '+session['token']})
    return json.load(urllib.request.urlopen(req,timeout=30))
state=get('/api/review/state?tasks=');assert state['background']['error'] is None
assert state['summary']['complete']>=5 and state['summary']['accepted']>=25
store=Store();audit=json.loads((data/'quality-audit/accepted-audit.json').read_text(encoding='utf-8'))
for row in audit:
    doc=store.get(row['document_id']);segment=next(s for s in doc['segments'] if s['id']==row['id'])
    assert segment['approved'] and segment['box']==row['box'] and segment['legend_ids']==row['labels'],row['id']
manifest=json.loads((data/'normalized/current.json').read_text(encoding='utf-8'))
assert not {r['pixel_sha256'] for r in manifest['records'] if r['usable_for_generation']} & {r['pixel_sha256'] for r in manifest['records'] if r['usable_for_test']}
archive=data/'exports'/('normalized-'+manifest['snapshot']+'.zip')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert all(r['kind']=='real' and r['split']=='test' for r in map(json.loads,z.read('test.jsonl').decode().splitlines()))
    assert all(r['split']=='train' and r['usable_for_generation'] for r in json.loads(z.read('generator-sources.json')))
req=urllib.request.Request('http://127.0.0.1:8776/api/review/file',data=b'{"sets":5}',headers={'Authorization':'Bearer '+session['token'],'Content-Type':'application/json'})
body=urllib.request.urlopen(req,timeout=120).read();assert b'"portable": true' in body
assert body.count(b'window.ReviewDetails=')==1 and body.count(b'window.ReviewConnect=')==1 and b'Filter legend by colour' in body
file=data/'exports'/('strata-continue-colour-'+manifest['snapshot']+'.html');file.write_bytes(body)
lan=json.load(urllib.request.urlopen('http://192.168.0.92:8776/api/health',timeout=10));assert lan['ok']
result=dict(result='PASS',preserved_audited_examples=len(audit),progress=state['summary'],normalized=manifest['stats'],snapshot=manifest['snapshot'],continuation_file=str(file),lan_health=lan['ok'])
(data/'continuation-live-checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
