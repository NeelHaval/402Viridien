"""Exercise a running real server; journals marked test events for safe cleanup.

python -X utf8 -B labeltool/validate_api.py
python -X utf8 -B labeltool/validate_api.py --cleanup  # stop server first
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import uuid
from io import BytesIO
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from labeltool.common import HOME, HUMAN, GOLD, load_rows, write_json, read_json


def cleanup():
    from labeltool.store import Store
    record=HOME/'cache'/'api-test-events.json'
    if not record.exists():print('No API test events to remove');return
    data=json.loads(record.read_text(encoding='utf-8'))
    path=HUMAN/'labels.jsonl'
    if path.exists():
        events=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
        test_ids=set(data['event_ids'])
        keep=[e for e in events if e.get('event_id') not in test_ids and e.get('target') not in test_ids]
        if keep:path.write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in keep),encoding='utf-8')
        else:path.unlink()
        store=Store()
        for name in data['logs']:
            rows=store.rebuild(name)
            if not rows and name not in data['original_csvs']:(HUMAN/(name+'.csv')).unlink(missing_ok=True)
    record.unlink()
    print('Removed only marked API test decisions; preserved any other human labels.')


def run(base):
    import requests
    from PIL import Image
    session=requests.Session()
    def get(path):
        response=session.get(base+path,timeout=45);response.raise_for_status();return response
    def post(path,data):
        response=session.post(base+path,json=data,timeout=45);response.raise_for_status();return response.json()
    start=time.perf_counter();page=get('/')
    assert 'Native rows' in page.text
    bootstrap=get('/api/bootstrap').json()
    assert len(bootstrap['progress']['logs'])==51
    first=get('/api/items?limit=4').json()['items']
    assert len({r['log'] for r in first})==4
    excluded=','.join(r['id'] for r in first)
    following=get('/api/items?limit=4&exclude='+excluded).json()['items']
    assert len({r['log'] for r in first+following})==8
    assert not any(r['truth_only'] for r in first)
    crop=get(first[0]['image_url'])
    with Image.open(BytesIO(crop.content)) as image:
        assert image.width==first[0]['width']
        assert image.height==first[0]['context_bottom']-first[0]['context_top']
    first_ms=(time.perf_counter()-start)*1000
    latency=[]
    for _ in range(8):
        t=time.perf_counter();get('/api/items?limit=4');latency.append((time.perf_counter()-t)*1000)
    for mode in ('uncertain','spot','disputed','walk'):
        name=first[0]['log']
        from urllib.parse import urlencode
        rows=get('/api/items?'+urlencode(dict(mode=mode,log=name,limit=8))).json()['items']
        assert all(r['log']!=GOLD for r in rows)
        if mode=='spot':assert all(r['tier']=='high' for r in rows)
        if mode=='disputed':assert all(r['disputed'] for r in rows)
        if mode=='walk':assert [r['top_y'] for r in rows]==sorted(r['top_y'] for r in rows)
    item=first[0]
    event_id='labeltool-e2e-'+uuid.uuid4().hex
    registry=HOME/'cache'/'api-test-events.json'
    record=read_json(registry,dict(event_ids=[],logs=[],original_csvs=[p.stem for p in HUMAN.glob('*.csv')]))
    record['event_ids'].append(event_id);record['logs'].append(item['log'])
    write_json(registry,record)
    payload=dict(item_id=item['id'],top_y=item['top_y']+1,bottom_y=item['bottom_y'],label='Unknown',
                 mode='uncertain',ms_taken=1234,suggestion_shown=item['suggestion'],request_id=event_id)
    result=post('/api/decision',payload)
    assert result['event']['boundary_edited'] and result['event']['ms_taken']==1234
    assert any(r['top_y']==payload['top_y'] and r['lithology']=='Unknown' for r in load_rows(HUMAN/(item['log']+'.csv')))
    duplicate=post('/api/decision',payload)
    assert duplicate['event']['event_id']==event_id
    stats=get('/api/stats').json()
    assert stats['table'] and any(r['n_rows']>=payload['bottom_y']-payload['top_y'] for r in stats['table'])
    assert stats['gold']
    invalid=session.post(base+'/api/decision',json=dict(payload,request_id=event_id+'invalid',bottom_y=payload['top_y']),timeout=30)
    assert invalid.status_code==400
    undone=post('/api/undo',{})
    assert undone['event']['event_id']==event_id
    assert undone['item']['id']==item['id']
    # Check a random confident example and Wilson CI without leaving it labelled.
    confident=get('/api/items?mode=spot&limit=1').json()['items']
    if confident:
        candidate=confident[0];spot_id=event_id+'-spot'
        record['event_ids'].append(spot_id);record['logs'].append(candidate['log'])
        write_json(HOME/'cache'/'api-test-events.json',record)
        answer=post('/api/decision',dict(item_id=candidate['id'],label=candidate['suggestion'],mode='spot',request_id=spot_id))
        assert answer['progress']['spot']['n']>=1
        assert answer['progress']['spot']['ci95'][0]<1
        assert post('/api/undo',{})['event']['event_id']==spot_id
    result=dict(status='passed',logs=51,first_page_item_crop_ms=round(first_ms,1),
                next_api_median_ms=round(statistics.median(latency),1),next_api_max_ms=round(max(latency),1),
                checks=['4-log round robin','all 4 modes','gold excluded','native crop dimensions','edited save',
                        'CSV rebuild','duplicate request','human-only stats','gold stats','invalid boundaries','undo','spot Wilson CI'])
    write_json(HOME/'validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='http://127.0.0.1:8765')
    parser.add_argument('--cleanup',action='store_true')
    args=parser.parse_args()
    cleanup() if args.cleanup else run(args.base)
