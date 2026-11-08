"""Prepare a review handoff from saved v4 outputs; do not change annotations."""
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

from model import Store
from review_queue import ReviewQueue
from verification import BedGuesses, build_manifest


def main():
    root=Path(__file__).resolve().parent;store=Store();queue=ReviewQueue(store)
    model=BedGuesses(root.parent/'bakeoff/analysis/categories')
    counts=queue.counts();groups=[]
    for key,group in queue.groups.items():
        if counts[key]>=5:continue
        groups.append(dict(key=key,tasks=[queue.task_payload(t['id']) for t in group['candidates'] if queue.available(t,set())]))
    manifest=build_manifest(dict(sets=groups),model,['family-43','family-51'])
    manifest.update(created_at=time.time(),ready_sets=[g['key'] for g in groups if g['key'] not in manifest['held_sets']],held_reason='Sets 44 and 52 need a source legend check. Their rejected examples are kept.')
    target=root/'static/verification-predictions.txt';temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':')),encoding='utf-8');temp.replace(target)
    sid='017fbd61d7c0cd8f5885a23e'
    with queue.db() as db:
        session=db.execute('SELECT name,token FROM sessions WHERE id=?',(sid,)).fetchone()
        # Keep the two reported problem sets out of this fast verification round.
        for key in manifest['held_sets']:
            db.executemany('INSERT OR IGNORE INTO skips VALUES(?,?)',[(sid,t['id']) for t in queue.groups[key]['candidates']])
    req=Request('http://127.0.0.1:8776/api/review/bundle?sets=10&exclude=family-43,family-51',headers={'Authorization':'Bearer '+session['token']})
    with urlopen(req,timeout=90) as response:pack=json.load(response)
    assert len(pack['sets'])==len(manifest['ready_sets'])==10
    first=pack['sets'][0]['tasks'][0]['id']
    url='http://192.168.0.92:8776/review?round=2#review='+session['token']+'&task='+first
    out=store.data/'verification-round-2';out.mkdir(exist_ok=True)
    (out/'teammate-link.txt').write_text(url+'\n\nPowerShell:\nStart-Process '+repr(url)+'\n',encoding='utf-8')
    report=dict(model_version=model.version,sets=[dict(title=g['title'],key=g['key'],tasks=len(g['tasks']),mapped=sum(bool(manifest['tasks'].get(t['id'],{}).get('prediction',{}).get('choices')) for t in g['tasks'])) for g in pack['sets']],held_sets=[44,52],guesses=len(manifest['tasks']),mapped_guesses=sum(bool(p['prediction']['choices']) for p in manifest['tasks'].values()),first_task=first)
    (out/'prepared.json').write_text(json.dumps(report,indent=1),encoding='utf-8')
    print(json.dumps(report));print('Private continuation command saved to',out/'teammate-link.txt')


if __name__=='__main__':main()
